//! The ML server (`server` feature): binds the port on IPv4 and IPv6, loads
//! and warms one session per available ML estimator, then answers
//! `/v1/info` and `/v1/estimate` on a few worker threads until stopped.
//! Estimates run on a session one at a time, as in a local run; concurrent
//! requests queue.

use std::fmt;
use std::io::{self, Read};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use kirafrq_ml_provider::rmvpe::Rmvpe;
use kirafrq_ml_provider::swiftf0::SwiftF0;
use kirafrq_ml_provider::{ModelError, ModelSource, SessionConfig, UvPolicy};

use crate::{
    DEFAULT_PORT, ErrorBody, Estimate, EstimateParams, Estimators, Info, MAX_BODY_BYTES,
    MlEstimator, PROTOCOL_VERSION, VERSION, audio,
};

/// The ports tried after [`DEFAULT_PORT`] before an OS-assigned one.
const FALLBACK_PORTS: std::ops::RangeInclusive<u16> = DEFAULT_PORT + 1..=DEFAULT_PORT + 9;

/// Request threads: enough that `/v1/info` answers while estimates queue.
const WORKERS: usize = 4;

/// Port-0 attempts at finding a port free on both families: the OS picks the
/// IPv4 port, which another program may hold on IPv6.
const FREE_PORT_ATTEMPTS: usize = 8;

/// How often idle worker threads look at the stop flag.
const POLL: Duration = Duration::from_millis(200);

/// Which port to listen on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Port {
    /// [`DEFAULT_PORT`], else the next free one in 39471–39479, else one the
    /// OS assigns, with a warning naming the port actually bound.
    Default,
    /// Exactly this port; `0` asks the OS for any free one. A port in use is
    /// an error.
    Explicit(u16),
}

/// Which interfaces to listen on, on IPv4 and, where the system has it,
/// IPv6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interfaces {
    /// Every interface (`0.0.0.0` and `::`): what the CLI serves on.
    All,
    /// This machine only (`127.0.0.1` and `::1`).
    Loopback,
}

impl Interfaces {
    fn ips(self) -> (Ipv4Addr, Ipv6Addr) {
        match self {
            Interfaces::All => (Ipv4Addr::UNSPECIFIED, Ipv6Addr::UNSPECIFIED),
            Interfaces::Loopback => (Ipv4Addr::LOCALHOST, Ipv6Addr::LOCALHOST),
        }
    }
}

/// Everything the server needs to start.
pub struct ServerOptions {
    pub interfaces: Interfaces,
    pub port: Port,
    /// The optional shared token; `None` or empty lets anyone in.
    pub token: Option<String>,
    /// The ML runner and device the sessions run on.
    pub session_config: SessionConfig,
    /// The runner and device as the user knows them, for `/v1/info`.
    pub runner_label: String,
    pub device_label: String,
    /// ONNX Runtime intra-op threads, as `--jobs` sets them locally; `0` is
    /// its default.
    pub threads: usize,
    /// RMVPE's resolved model file, or why it is missing (then RMVPE is
    /// unavailable, which the startup report says).
    pub rmvpe: Result<PathBuf, ModelError>,
    /// SwiftF0's model: the on-disk override or the bundled one.
    pub swiftf0: ModelSource,
}

/// Why the server could not start.
#[derive(Debug)]
pub enum ServerError {
    /// The explicitly requested port is taken.
    PortInUse(u16),
    /// Binding failed for another reason.
    Bind(io::Error),
    /// An ML estimator's sessions could not be built or warmed.
    Model {
        estimator: MlEstimator,
        message: String,
    },
}

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServerError::PortInUse(port) => write!(
                f,
                "port {port} is already in use; choose another with --port, or omit it to use \
                 the default"
            ),
            ServerError::Bind(error) => write!(f, "cannot listen: {error}"),
            ServerError::Model { estimator, message } => {
                write!(f, "cannot load {}: {message}", estimator.name())
            }
        }
    }
}

impl std::error::Error for ServerError {}

/// A bound server with its models loaded and warmed, not yet answering.
pub struct Server {
    /// One per listener: IPv4, then IPv6 where the system has it.
    http: Vec<tiny_http::Server>,
    addresses: Vec<SocketAddr>,
    port_warning: Option<String>,
    rmvpe_missing: Option<ModelError>,
    handlers: Arc<Handlers>,
}

impl Server {
    /// Bind the port, then build and warm every available ML estimator.
    /// Connections queue until [`Self::spawn`].
    pub fn start(options: ServerOptions) -> Result<Self, ServerError> {
        let (listeners, port_warning) = bind(options.interfaces, options.port)?;
        let addresses = listeners
            .iter()
            .map(TcpListener::local_addr)
            .collect::<io::Result<_>>()
            .map_err(ServerError::Bind)?;

        let (rmvpe, rmvpe_missing) = match options.rmvpe {
            Ok(path) => {
                let estimator = Rmvpe::new(path, UvPolicy::rmvpe(None), options.threads)
                    .with_session_config(options.session_config);
                let estimator = estimator
                    .with_own_session()
                    .map_err(model_error(MlEstimator::Rmvpe))?;
                estimator.warm().map_err(model_error(MlEstimator::Rmvpe))?;
                (Some(estimator), None)
            }
            Err(missing) => (None, Some(missing)),
        };
        let swiftf0 = SwiftF0::new(options.swiftf0, UvPolicy::swiftf0(None), options.threads)
            .with_session_config(options.session_config);
        let swiftf0 = swiftf0
            .with_own_session()
            .map_err(model_error(MlEstimator::SwiftF0))?;
        swiftf0.warm().map_err(model_error(MlEstimator::SwiftF0))?;

        let http = listeners
            .into_iter()
            .map(|listener| tiny_http::Server::from_listener(listener, None))
            .collect::<Result<_, _>>()
            .map_err(|error| ServerError::Bind(io::Error::other(error)))?;
        Ok(Self {
            http,
            addresses,
            port_warning,
            rmvpe_missing,
            handlers: Arc::new(Handlers {
                token: options.token.filter(|token| !token.is_empty()),
                runner_label: options.runner_label,
                device_label: options.device_label,
                rmvpe,
                swiftf0,
            }),
        })
    }

    /// The bound addresses, one port on every family: IPv4, then IPv6 where
    /// the system has it.
    pub fn addresses(&self) -> &[SocketAddr] {
        &self.addresses
    }

    /// The warning for a default port that was taken; `None` when the
    /// default was free or the port was explicit.
    pub fn port_warning(&self) -> Option<&str> {
        self.port_warning.as_deref()
    }

    /// Why RMVPE is unavailable on this server, when it is.
    pub fn rmvpe_missing(&self) -> Option<&ModelError> {
        self.rmvpe_missing.as_ref()
    }

    /// What `/v1/info` reports (for a request without a token).
    pub fn info(&self) -> Info {
        self.handlers.info(false)
    }

    /// Start answering on background threads, [`WORKERS`] per listener;
    /// dropping the handle stops them.
    pub fn spawn(self) -> RunningServer {
        let stop = Arc::new(AtomicBool::new(false));
        let threads = self
            .http
            .into_iter()
            .map(Arc::new)
            .flat_map(|http| std::iter::repeat_n(http, WORKERS))
            .map(|http| {
                let stop = Arc::clone(&stop);
                let handlers = Arc::clone(&self.handlers);
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        match http.recv_timeout(POLL) {
                            Ok(Some(request)) => handlers.handle(request),
                            Ok(None) => {}
                            Err(_) => std::thread::sleep(POLL),
                        }
                    }
                })
            })
            .collect();
        RunningServer {
            addresses: self.addresses,
            stop,
            threads,
        }
    }
}

/// A server answering requests; dropping it stops the workers after their
/// current request.
pub struct RunningServer {
    addresses: Vec<SocketAddr>,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl RunningServer {
    /// As [`Server::addresses`].
    pub fn addresses(&self) -> &[SocketAddr] {
        &self.addresses
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

fn model_error(estimator: MlEstimator) -> impl Fn(kirafrq_ml_provider::Error) -> ServerError {
    move |error| ServerError::Model {
        estimator,
        message: error.to_string(),
    }
}

/// Bind `interfaces` per the port policy, returning the warning for a taken
/// default.
fn bind(
    interfaces: Interfaces,
    port: Port,
) -> Result<(Vec<TcpListener>, Option<String>), ServerError> {
    match port {
        Port::Explicit(port) => match listen(interfaces, port) {
            Ok(listeners) => Ok((listeners, None)),
            Err(error) if error.kind() == io::ErrorKind::AddrInUse => {
                Err(ServerError::PortInUse(port))
            }
            Err(error) => Err(ServerError::Bind(error)),
        },
        Port::Default => {
            for candidate in std::iter::once(DEFAULT_PORT)
                .chain(FALLBACK_PORTS)
                .chain([0])
            {
                match listen(interfaces, candidate) {
                    Ok(listeners) if candidate == DEFAULT_PORT => return Ok((listeners, None)),
                    Ok(listeners) => {
                        let bound = listeners[0].local_addr().map_err(ServerError::Bind)?.port();
                        let warning = format!(
                            "the default port {DEFAULT_PORT} is in use; listening on port \
                             {bound} instead"
                        );
                        return Ok((listeners, Some(warning)));
                    }
                    Err(error) if error.kind() == io::ErrorKind::AddrInUse => continue,
                    Err(error) => return Err(ServerError::Bind(error)),
                }
            }
            Err(ServerError::Bind(io::ErrorKind::AddrInUse.into()))
        }
    }
}

/// Listen on `port` on IPv4 and, where the system has it, IPv6; `0` asks the
/// OS for a port free on both. A port held on either family is in use.
fn listen(interfaces: Interfaces, port: u16) -> io::Result<Vec<TcpListener>> {
    let (ipv4, ipv6) = interfaces.ips();
    for _ in 0..FREE_PORT_ATTEMPTS {
        let v4 = TcpListener::bind((ipv4, port))?;
        match listen_ipv6_only(ipv6, v4.local_addr()?.port()) {
            Ok(v6) => return Ok(vec![v4, v6]),
            Err(error) if error.kind() == io::ErrorKind::AddrInUse && port == 0 => continue,
            Err(error) if error.kind() == io::ErrorKind::AddrInUse => return Err(error),
            // No IPv6 on this system: IPv4 alone.
            Err(_) => return Ok(vec![v4]),
        }
    }
    Err(io::ErrorKind::AddrInUse.into())
}

/// An IPv6-only listener beside the IPv4 one, not one dual-stack socket: on
/// macOS a dual-stack socket binds even over a port another program holds on
/// IPv4, which would hide that the port is taken.
fn listen_ipv6_only(ip: Ipv6Addr, port: u16) -> io::Result<TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};

    let socket = Socket::new(Domain::IPV6, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_only_v6(true)?;
    // As `TcpListener::bind` does on Unix, so a restart can rebind at once.
    #[cfg(unix)]
    socket.set_reuse_address(true)?;
    socket.bind(&SocketAddr::new(IpAddr::V6(ip), port).into())?;
    socket.listen(128)?;
    Ok(socket.into())
}

/// The request handlers and the ML estimators behind them.
struct Handlers {
    token: Option<String>,
    runner_label: String,
    device_label: String,
    rmvpe: Option<Rmvpe>,
    swiftf0: SwiftF0,
}

/// A response: status and JSON body.
type Reply = (u16, String);

impl Handlers {
    fn info(&self, authorized: bool) -> Info {
        Info {
            protocol: PROTOCOL_VERSION,
            version: VERSION.to_string(),
            runner: self.runner_label.clone(),
            device: self.device_label.clone(),
            estimators: Estimators {
                rmvpe: self.rmvpe.is_some(),
                swiftf0: true,
            },
            token_required: self.token.is_some(),
            authorized: self.token.is_none() || authorized,
        }
    }

    fn handle(&self, mut request: tiny_http::Request) {
        let url = request.url().to_string();
        let (path, query) = url.split_once('?').unwrap_or((&url, ""));
        let (status, body) = match (request.method(), path) {
            (tiny_http::Method::Get, "/v1/info") => {
                let authorized = self.authorized(&request);
                json(200, &self.info(authorized))
            }
            (tiny_http::Method::Post, "/v1/estimate") => self
                .estimate(&mut request, query)
                .unwrap_or_else(|reply| reply),
            (_, "/v1/info" | "/v1/estimate") => error(405, "method not allowed"),
            _ => error(404, &format!("no such endpoint: {path}")),
        };
        let response = tiny_http::Response::from_string(body)
            .with_status_code(status)
            .with_header(
                tiny_http::Header::from_bytes("Content-Type", "application/json")
                    .expect("a valid header"),
            );
        // The client may have gone away; nothing to do about it here.
        let _ = request.respond(response);
    }

    /// The bearer token matches; with no token configured, everyone is.
    fn authorized(&self, request: &tiny_http::Request) -> bool {
        let Some(token) = &self.token else {
            return true;
        };
        request.headers().iter().any(|header| {
            header.field.equiv("Authorization")
                && header.value.as_str().strip_prefix("Bearer ") == Some(token.as_str())
        })
    }

    fn estimate(&self, request: &mut tiny_http::Request, query: &str) -> Result<Reply, Reply> {
        if !self.authorized(request) {
            return Err(error(401, "this ML server requires a valid token"));
        }
        let params = EstimateParams::from_query(query).map_err(|message| error(400, &message))?;
        if request
            .body_length()
            .is_some_and(|length| length > MAX_BODY_BYTES)
        {
            return Err(too_large());
        }
        let mut body = Vec::new();
        request
            .as_reader()
            .take(MAX_BODY_BYTES as u64 + 1)
            .read_to_end(&mut body)
            .map_err(|failure| error(400, &format!("cannot read the body: {failure}")))?;
        if body.len() > MAX_BODY_BYTES {
            return Err(too_large());
        }
        let content_type = request
            .headers()
            .iter()
            .find(|header| header.field.equiv("Content-Type"))
            .map(|header| header.value.as_str().to_string())
            .unwrap_or_default();
        let samples =
            audio::decode(&content_type, &body).map_err(|message| error(400, &message))?;

        let track = match params.estimator {
            MlEstimator::Rmvpe => {
                let rmvpe = self.rmvpe.as_ref().ok_or_else(|| {
                    error(
                        503,
                        "RMVPE is not available: rmvpe.onnx was not found on the server",
                    )
                })?;
                let policy = policy(UvPolicy::rmvpe(params.confidence_threshold), &params);
                rmvpe.clone().with_policy(policy).estimate(&samples, None)
            }
            MlEstimator::SwiftF0 => {
                let policy = policy(UvPolicy::swiftf0(params.confidence_threshold), &params);
                self.swiftf0
                    .clone()
                    .with_policy(policy)
                    .estimate(&samples, None)
            }
        }
        .map_err(|failure| error(500, &format!("inference failed: {failure}")))?;
        Ok(json(200, &Estimate { f0_hz: track.f0_hz }))
    }
}

/// The model's default policy with the request's range.
fn policy(defaults: UvPolicy, params: &EstimateParams) -> UvPolicy {
    UvPolicy {
        floor_hz: params.floor_hz,
        ceiling_hz: params.ceiling_hz,
        ..defaults
    }
}

fn json(status: u16, value: &impl serde::Serialize) -> Reply {
    (
        status,
        serde_json::to_string(value).expect("protocol types serialize"),
    )
}

fn error(status: u16, message: &str) -> Reply {
    json(
        status,
        &ErrorBody {
            error: message.to_string(),
        },
    )
}

fn too_large() -> Reply {
    error(
        413,
        &format!("the body is larger than {} MiB", MAX_BODY_BYTES >> 20),
    )
}
