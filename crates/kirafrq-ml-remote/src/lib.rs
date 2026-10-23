//! The ML server (#85): a KiraFrqGen CLI in serve mode runs the ML estimators
//! for other KiraFrqGen instances on the local network.
//!
//! Only ML f0 estimation crosses the network: the client sends the wav's
//! decoded mono 44.1 kHz samples, exactly (lossless FLAC, or raw f64 when no
//! FLAC bit depth holds them), plus the estimate parameters, and gets back the
//! f0 on the 5.805 ms table grid. Nothing is resampled or rounded on the
//! wire, so the server's model input is bit-identical to a local run's. The server holds no voicebank state; decode, the
//! DSP estimators, the energy voicing gate and every writer stay on the
//! client (`docs/adr/0002-ml-server-client-split.md`).
//!
//! The crate splits by the `server` feature, like the provider splits by
//! `ort`:
//!
//! - **Client** (always compiled): the wire protocol ([`Info`],
//!   [`EstimateParams`]), the exact audio encoding and the HTTP
//!   [`Client`]. It needs only the provider's model-free layer, so no ONNX
//!   Runtime.
//! - **Server** (`server` on): [`server::Server`], which binds the port, builds
//!   and warms one session per available ML estimator and answers
//!   requests.
//!
//! Protocol version 1, plain HTTP/1.1 under `/v1`:
//!
//! - `GET /v1/info`, no token needed: [`Info`] as JSON.
//! - `POST /v1/estimate?<EstimateParams>` with the audio body: [`Estimate`]
//!   as JSON.
//! - Errors are [`ErrorBody`] JSON with a meaningful status. The optional
//!   token travels as `Authorization: Bearer <token>`; there is no TLS.
//!
//! This crate depends on the provider, never on `kirafrqgen-core`.

mod audio;
mod client;
#[cfg(feature = "server")]
pub mod server;

pub use client::{Client, ClientError};

/// The wire protocol's version; a server and client agree on it or the run
/// stops before any file.
pub const PROTOCOL_VERSION: u32 = 1;

/// The server's default port: unassigned in the IANA registry and below the
/// 49152+ range operating systems hand out to outgoing connections.
pub const DEFAULT_PORT: u16 = 39470;

/// The KiraFrqGen version this crate was built with; a differing server
/// version under the same protocol is only a warning.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The largest request body the server reads: over 12 minutes of raw f64
/// audio, and far more as FLAC.
pub const MAX_BODY_BYTES: usize = 256 << 20;

/// The rate the audio is sent at: the pipeline's own 44.1 kHz timeline.
pub const SAMPLE_RATE: u32 = 44_100;

/// An ML estimator the server can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    Rmvpe,
    SwiftF0,
}

impl Model {
    /// The wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Model::Rmvpe => "rmvpe",
            Model::SwiftF0 => "swiftf0",
        }
    }

    /// The user-facing name.
    pub fn name(self) -> &'static str {
        match self {
            Model::Rmvpe => "RMVPE",
            Model::SwiftF0 => "SwiftF0",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "rmvpe" => Some(Model::Rmvpe),
            "swiftf0" => Some(Model::SwiftF0),
            _ => None,
        }
    }
}

/// `GET /v1/info`: what the server is and what it can run.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Info {
    /// [`PROTOCOL_VERSION`] on the server.
    pub protocol: u32,
    /// The server's KiraFrqGen version.
    pub version: String,
    /// The server's ML runner, as the user knows it (`Core ML`, `CPU`, ...).
    pub runner: String,
    /// What the runner runs on, e.g. the DirectML adapter.
    pub device: String,
    /// Which ML estimators the server can run.
    pub estimators: Estimators,
    /// The server was started with a token.
    pub token_required: bool,
    /// The request carried the right token, or none is required.
    pub authorized: bool,
}

impl Info {
    /// Whether the server can run `model`.
    pub fn available(&self, model: Model) -> bool {
        match model {
            Model::Rmvpe => self.estimators.rmvpe,
            Model::SwiftF0 => self.estimators.swiftf0,
        }
    }

    /// The warning for a server whose KiraFrqGen version differs from this
    /// one's under the same protocol; `None` when they match.
    pub fn version_warning(&self) -> Option<String> {
        (self.version != VERSION).then(|| {
            format!(
                "the ML server runs KiraFrqGen {}, this is {VERSION}; the protocol matches, so \
                 the run goes ahead",
                self.version
            )
        })
    }

    /// One line for the front ends: version, runner and estimators.
    pub fn summary(&self) -> String {
        let estimators: Vec<&str> = [Model::Rmvpe, Model::SwiftF0]
            .into_iter()
            .filter(|&model| self.available(model))
            .map(Model::name)
            .collect();
        let estimators = if estimators.is_empty() {
            "none".to_string()
        } else {
            estimators.join(", ")
        };
        format!(
            "KiraFrqGen {} on {} ({}); ML estimators: {estimators}",
            self.version, self.runner, self.device
        )
    }
}

/// Per ML estimator availability on the server: RMVPE needs its model file
/// there; SwiftF0 is bundled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Estimators {
    pub rmvpe: bool,
    pub swiftf0: bool,
}

/// The `POST /v1/estimate` parameters, carried in the query string.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimateParams {
    pub model: Model,
    pub floor_hz: f64,
    pub ceiling_hz: f64,
    /// The #53 confidence override; `None` is the model's own default.
    pub confidence_threshold: Option<f64>,
}

impl EstimateParams {
    /// The query string, without the leading `?`.
    pub fn to_query(&self) -> String {
        let mut query = format!(
            "estimator={}&floor_hz={}&ceiling_hz={}",
            self.model.as_str(),
            self.floor_hz,
            self.ceiling_hz,
        );
        if let Some(threshold) = self.confidence_threshold {
            query.push_str(&format!("&threshold={threshold}"));
        }
        query
    }

    /// Parse [`Self::to_query`]'s output; the message names the bad field.
    pub fn from_query(query: &str) -> Result<Self, String> {
        let mut model = None;
        let mut floor_hz = None;
        let mut ceiling_hz = None;
        let mut confidence_threshold = None;
        for pair in query.split('&').filter(|pair| !pair.is_empty()) {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            let number = || {
                value
                    .parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| format!("`{key}` is not a number: `{value}`"))
            };
            match key {
                "estimator" => {
                    model = Some(
                        Model::parse(value)
                            .ok_or_else(|| format!("unknown estimator `{value}`"))?,
                    );
                }
                "floor_hz" => floor_hz = Some(number()?),
                "ceiling_hz" => ceiling_hz = Some(number()?),
                "threshold" => confidence_threshold = Some(number()?),
                _ => return Err(format!("unknown parameter `{key}`")),
            }
        }
        let missing = |name: &str| format!("missing parameter `{name}`");
        Ok(Self {
            model: model.ok_or_else(|| missing("estimator"))?,
            floor_hz: floor_hz.ok_or_else(|| missing("floor_hz"))?,
            ceiling_hz: ceiling_hz.ok_or_else(|| missing("ceiling_hz"))?,
            confidence_threshold,
        })
    }
}

/// `POST /v1/estimate`'s answer: the f0 in Hz on the table grid, one value
/// per table frame, `0.0` unvoiced.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Estimate {
    pub f0_hz: Vec<f64>,
}

/// Every error response's body.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ErrorBody {
    pub error: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_params_survive_the_query_string() {
        let params = EstimateParams {
            model: Model::SwiftF0,
            floor_hz: 71.0,
            ceiling_hz: 800.5,
            confidence_threshold: Some(0.25),
        };
        assert_eq!(EstimateParams::from_query(&params.to_query()), Ok(params));
        let params = EstimateParams {
            confidence_threshold: None,
            ..params
        };
        assert_eq!(EstimateParams::from_query(&params.to_query()), Ok(params));

        assert!(EstimateParams::from_query("estimator=crepe").is_err());
        assert!(EstimateParams::from_query("estimator=rmvpe&floor_hz=71").is_err());
        assert!(EstimateParams::from_query("estimator=rmvpe&floor_hz=NaN&ceiling_hz=1").is_err());
    }
}
