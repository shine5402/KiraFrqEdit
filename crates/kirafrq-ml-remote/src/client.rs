//! The ML server's client: one keep-alive agent, shared by the run's worker
//! threads. A short connect timeout and a generous response timeout, because
//! requests queue on the server's sessions.

use std::fmt;
use std::time::Duration;

use kirafrq_ml_provider::grid::{TABLE_HOP_SAMPLES, table_frame_count};

use crate::{
    ErrorBody, Estimate, EstimateParams, Info, MAX_BODY_BYTES, MlEstimator, PROTOCOL_VERSION, audio,
};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// `/v1/info` answers at once; anything slower is not a working server.
const INFO_TIMEOUT: Duration = Duration::from_secs(10);
/// An estimate may wait behind other requests for the server's session.
const ESTIMATE_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Why talking to the ML server failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientError {
    /// No connection, or the connection broke: the server is down, the
    /// address is wrong, or a firewall is in the way.
    Unreachable { address: String, reason: String },
    /// Something answered, but not an ML server speaking this protocol.
    Protocol { address: String, reason: String },
    /// The server requires a token and none was given.
    TokenRequired,
    /// The server rejected the token.
    TokenRejected,
    /// The server cannot run the selected ML estimator.
    Unavailable(MlEstimator),
    /// The server answered an estimate with an error.
    Server { status: u16, message: String },
    /// The audio could not be prepared for sending.
    Audio(String),
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClientError::Unreachable { address, reason } => {
                write!(f, "cannot reach the ML server at {address}: {reason}")
            }
            ClientError::Protocol { address, reason } => write!(
                f,
                "{address} is not a compatible ML server ({reason}); run the same KiraFrqGen \
                 release on both machines"
            ),
            ClientError::TokenRequired => {
                write!(f, "the ML server requires a token; set the server's token")
            }
            ClientError::TokenRejected => write!(f, "the ML server rejected the token"),
            ClientError::Unavailable(MlEstimator::Rmvpe) => write!(
                f,
                "the ML server cannot run RMVPE: its rmvpe.onnx was not found on the server"
            ),
            ClientError::Unavailable(estimator) => {
                write!(f, "the ML server cannot run {}", estimator.name())
            }
            ClientError::Server { status, message } => {
                write!(f, "the ML server answered {status}: {message}")
            }
            ClientError::Audio(message) => write!(f, "cannot prepare the audio: {message}"),
        }
    }
}

impl std::error::Error for ClientError {}

/// A client for one ML server. Cheap to share across threads: the agent pools
/// keep-alive connections.
#[derive(Clone)]
pub struct Client {
    agent: ureq::Agent,
    /// `host:port`, IPv6 hosts in brackets.
    address: String,
    token: Option<String>,
}

impl Client {
    /// A client for the server at `host` (an IPv6 address without brackets)
    /// and `port`; `token` is sent when set. Nothing is sent until the first
    /// request.
    pub fn new(host: &str, port: u16, token: Option<String>) -> Self {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .max_idle_connections_per_host(64)
            .max_idle_connections(64)
            .build()
            .new_agent();
        let address = if host.contains(':') {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        };
        Self {
            agent,
            address,
            token: token.filter(|token| !token.is_empty()),
        }
    }

    /// The server this client talks to.
    pub fn address(&self) -> &str {
        &self.address
    }

    /// `GET /v1/info`, with the token when one is set so `authorized` reflects
    /// it.
    pub fn info(&self) -> Result<Info, ClientError> {
        let mut request = self
            .agent
            .get(self.url("/v1/info"))
            .config()
            .timeout_global(Some(INFO_TIMEOUT))
            .build();
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        let mut response = request.call().map_err(|error| self.unreachable(error))?;
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .read_to_string()
            .map_err(|error| self.unreachable(error))?;
        if status != 200 {
            return Err(self.protocol(format!("/v1/info answered {status}")));
        }
        let info: Info = serde_json::from_str(&body)
            .map_err(|error| self.protocol(format!("unreadable /v1/info: {error}")))?;
        if info.protocol != PROTOCOL_VERSION {
            return Err(self.protocol(format!(
                "it speaks protocol {}, this build speaks {PROTOCOL_VERSION}",
                info.protocol
            )));
        }
        Ok(info)
    }

    /// The configuration check before a run: the server answers this
    /// protocol, accepts the token and can run `model`. A differing
    /// KiraFrqGen version is left to [`Info::version_warning`].
    pub fn check(&self, estimator: MlEstimator) -> Result<Info, ClientError> {
        let info = self.check_access()?;
        if !info.available(estimator) {
            return Err(ClientError::Unavailable(estimator));
        }
        Ok(info)
    }

    /// [`Self::check`] without an estimator: the server answers and accepts
    /// the token (Test connection).
    pub fn check_access(&self) -> Result<Info, ClientError> {
        let info = self.info()?;
        if !info.authorized {
            return Err(if self.token.is_none() {
                ClientError::TokenRequired
            } else {
                ClientError::TokenRejected
            });
        }
        Ok(info)
    }

    /// Estimate f0 on the table grid for mono 44.1 kHz `samples`: send them
    /// exactly and read back one value per table frame. A failed request is
    /// retried once; there is never a local fallback. A sub-hop wav needs no
    /// model and is answered here, as a local estimate would.
    pub fn estimate(
        &self,
        samples: &[f64],
        params: EstimateParams,
    ) -> Result<Vec<f64>, ClientError> {
        let frames = table_frame_count(samples.len());
        if samples.len() < TABLE_HOP_SAMPLES {
            return Ok(vec![0.0; frames]);
        }
        let (content_type, body) = audio::encode(samples).map_err(ClientError::Audio)?;
        let attempt = || self.post_estimate(content_type, &body, &params, frames);
        attempt().or_else(|_| attempt())
    }

    fn post_estimate(
        &self,
        content_type: &str,
        body: &[u8],
        params: &EstimateParams,
        frames: usize,
    ) -> Result<Vec<f64>, ClientError> {
        let url = format!("{}?{}", self.url("/v1/estimate"), params.to_query());
        let mut request = self
            .agent
            .post(url)
            .config()
            .timeout_global(Some(ESTIMATE_TIMEOUT))
            .build()
            .header("Content-Type", content_type);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        let mut response = request
            .send(body)
            .map_err(|error| self.unreachable(error))?;
        let status = response.status().as_u16();
        let text = response
            .body_mut()
            .with_config()
            .limit(MAX_BODY_BYTES as u64)
            .read_to_string()
            .map_err(|error| self.unreachable(error))?;
        if status != 200 {
            let message = serde_json::from_str::<ErrorBody>(&text)
                .map(|body| body.error)
                .unwrap_or(text);
            return Err(ClientError::Server { status, message });
        }
        let estimate: Estimate = serde_json::from_str(&text)
            .map_err(|error| self.protocol(format!("unreadable estimate: {error}")))?;
        if estimate.f0_hz.len() != frames {
            return Err(self.protocol(format!(
                "the estimate has {} frames, the wav needs {frames}",
                estimate.f0_hz.len()
            )));
        }
        Ok(estimate.f0_hz)
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.address)
    }

    fn unreachable(&self, error: impl fmt::Display) -> ClientError {
        ClientError::Unreachable {
            address: self.address.clone(),
            reason: error.to_string(),
        }
    }

    fn protocol(&self, reason: String) -> ClientError {
        ClientError::Protocol {
            address: self.address.clone(),
            reason,
        }
    }
}
