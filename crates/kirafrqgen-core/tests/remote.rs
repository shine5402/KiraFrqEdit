//! The remote ML runner end to end (#85): a real ML server started
//! in-process on an OS-assigned localhost port with SwiftF0 (bundled, so CI
//! needs no RMVPE model), the generate entry point pointed at it, and its
//! tables checked against a local SwiftF0 run. A scripted stand-in server
//! covers the protocol and mid-run failure paths.

#![cfg(feature = "ml")]

use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use hound::{SampleFormat, WavSpec, WavWriter};
use kirafrq_formats::frq;
use kirafrq_ml_provider::model::ModelError;
use kirafrq_ml_provider::swiftf0::resolve_source;
use kirafrq_ml_remote::server::{Interfaces, Port, RunningServer, Server, ServerOptions};
use kirafrq_ml_remote::{Estimators, Info, PROTOCOL_VERSION, VERSION};
use kirafrqgen_core::{
    CancelToken, Estimator, F0Config, GenerateOptions, GeneratorError, MlConfig, MlRunner,
    MlServer, RunSummary, Target, build_estimator, generate,
};

// --- scratch and fixtures ---------------------------------------------------

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "kirafrqgen-core-remote-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_wav(dir: &Path, name: &str, samples: &[f64]) {
    let path = dir.join(name);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let spec = WavSpec {
        channels: 1,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(&path, spec).unwrap();
    for &sample in samples {
        writer
            .write_sample((sample * f64::from(i16::MAX)) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
}

/// A harmonic stack whose pitch follows `hz(t)`, between silences.
fn phrase(seconds: f64, hz: impl Fn(f64) -> f64) -> Vec<f64> {
    const RATE: f64 = 44_100.0;
    let silence = vec![0.0; 6_000];
    let mut phase = 0.0;
    let voiced = (0..(RATE * seconds) as usize).map(|index| {
        phase += std::f64::consts::TAU * hz(index as f64 / RATE) / RATE;
        (1..=8)
            .map(|harmonic| (phase * harmonic as f64).sin() * 0.25 / harmonic as f64)
            .sum::<f64>()
    });
    silence
        .iter()
        .copied()
        .chain(voiced)
        .chain(silence.iter().copied())
        .collect()
}

/// The same small voicebank in `dir`: steady, gliding and vibrato phrases at
/// several lengths, one in a subfolder, plus a sub-hop wav.
fn voicebank(dir: &Path) {
    write_wav(dir, "a.wav", &phrase(1.2, |_| 220.0));
    write_wav(dir, "glide.wav", &phrase(2.0, |t| 150.0 + 125.0 * t));
    write_wav(
        dir,
        "sub/vibrato.wav",
        &phrase(1.0, |t| {
            330.0 * (1.0 + 0.03 * (t * 5.5 * std::f64::consts::TAU).sin())
        }),
    );
    write_wav(dir, "sub/low.wav", &phrase(0.7, |_| 98.0));
    write_wav(dir, "tiny.wav", &[0.1; 200]);
}

const JOBS: usize = 3;

fn options(root: &Path, ml: MlConfig) -> GenerateOptions {
    GenerateOptions {
        root: root.to_path_buf(),
        targets: [Target::Frq].into_iter().collect(),
        overwrite: false,
        f0: F0Config {
            estimator: Estimator::SwiftF0,
            ml,
            ..F0Config::default()
        },
        // Several workers, so requests reach the server side by side.
        jobs: JOBS,
        sharing: None,
        delete_llsm: false,
    }
}

fn run(opts: &GenerateOptions) -> Result<RunSummary, GeneratorError> {
    let estimator = build_estimator(&opts.f0, opts.jobs)?;
    let cancel: CancelToken = Arc::new(AtomicBool::new(false));
    generate(opts, estimator.as_ref(), &NoProgress, &cancel)
}

struct NoProgress;
impl kirafrqgen_core::Progress for NoProgress {}

// --- servers ----------------------------------------------------------------

fn no_rmvpe() -> Result<PathBuf, ModelError> {
    Err(ModelError {
        file_name: "rmvpe.onnx".to_string(),
        searched: Vec::new(),
    })
}

/// A real ML server with SwiftF0 only, on a free localhost port.
fn server(token: Option<&str>) -> RunningServer {
    Server::start(ServerOptions {
        interfaces: Interfaces::Loopback,
        port: Port::Explicit(0),
        token: token.map(str::to_string),
        session_config: Default::default(),
        runner_label: "CPU".to_string(),
        device_label: "test".to_string(),
        // The local runs' intra-op thread count (`jobs`): ONNX Runtime's
        // reductions differ in the last bits across thread counts.
        threads: JOBS,
        rmvpe: no_rmvpe(),
        swiftf0: resolve_source(),
    })
    .unwrap()
    .spawn()
}

/// The server's IPv4 loopback address.
fn ipv4(server: &RunningServer) -> SocketAddr {
    *server
        .addresses()
        .iter()
        .find(|address| address.is_ipv4())
        .expect("an IPv4 listener")
}

fn remote(address: SocketAddr, token: Option<&str>) -> MlConfig {
    MlConfig {
        runner: MlRunner::Remote,
        server: Some(MlServer {
            host: address.ip().to_string(),
            port: address.port(),
        }),
        server_token: token.map(str::to_string),
        ..MlConfig::default()
    }
}

/// A stand-in server: `/v1/info` answers `info`, and `/v1/estimate` answers
/// per `script` (status per attempt; past its end, 200 with all-unvoiced f0
/// for [`short_wav`]). Counts the estimate requests.
struct Scripted {
    address: SocketAddr,
    estimates: Arc<AtomicUsize>,
    _stop: StopOnDrop,
}

struct StopOnDrop(Arc<AtomicBool>);
impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn scripted(info: Info, script: Vec<u16>) -> Scripted {
    let http = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let address = http.server_addr().to_ip().unwrap();
    let estimates = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (count, done) = (Arc::clone(&estimates), Arc::clone(&stop));
    let script = Mutex::new(script.into_iter());
    std::thread::spawn(move || {
        while !done.load(Ordering::Relaxed) {
            let Ok(Some(request)) = http.recv_timeout(std::time::Duration::from_millis(50)) else {
                continue;
            };
            let url = request.url().to_string();
            let (status, body) = if url.starts_with("/v1/info") {
                (200, serde_json::to_string(&info).unwrap())
            } else {
                count.fetch_add(1, Ordering::SeqCst);
                match script.lock().unwrap().next() {
                    Some(status) => (status, r#"{"error":"scripted failure"}"#.to_string()),
                    None => (
                        200,
                        serde_json::json!({ "f0_hz": vec![0.0; short_wav().len() / 256 + 1] })
                            .to_string(),
                    ),
                }
            };
            let _ =
                request.respond(tiny_http::Response::from_string(body).with_status_code(status));
        }
    });
    Scripted {
        address,
        estimates,
        _stop: StopOnDrop(stop),
    }
}

/// The one-wav fixture of the stand-in server runs.
fn short_wav() -> Vec<f64> {
    phrase(0.3, |_| 220.0)
}

fn info() -> Info {
    Info {
        protocol: PROTOCOL_VERSION,
        version: VERSION.to_string(),
        runner: "CPU".to_string(),
        device: "scripted".to_string(),
        estimators: Estimators {
            rmvpe: false,
            swiftf0: true,
        },
        token_required: false,
        authorized: true,
    }
}

fn config_error(opts: &GenerateOptions) -> String {
    match build_estimator(&opts.f0, opts.jobs) {
        Err(GeneratorError::Config(message)) => message,
        Err(other) => panic!("expected a configuration error, got {other}"),
        Ok(_) => panic!("expected a configuration error, got an estimator"),
    }
}

// --- the main seam ----------------------------------------------------------

#[test]
fn remote_swiftf0_tables_equal_a_local_run() {
    // The audio crosses the wire exactly, so on the same runner and thread
    // count the remote tables are the local ones, bit for bit.
    let local_dir = Scratch::new("local");
    let remote_dir = Scratch::new("remote");
    voicebank(&local_dir.0);
    voicebank(&remote_dir.0);

    let local = run(&options(&local_dir.0, MlConfig::default())).unwrap();
    let server = server(None);
    let remote = run(&options(&remote_dir.0, remote(ipv4(&server), None))).unwrap();
    assert!(local.failed.is_empty(), "{:?}", local.failed);
    assert!(remote.failed.is_empty(), "{:?}", remote.failed);
    assert_eq!(remote.written, local.written);

    for name in ["a", "glide", "sub/vibrato", "sub/low", "tiny"] {
        let table = |dir: &Path| frq::read(&dir.join(format!("{name}_wav.frq"))).unwrap();
        let local = table(&local_dir.0);
        assert!(
            local.f0_hz.len() < 10 || local.f0_hz.iter().any(|&f0| f0 > 0.0),
            "{name}: the fixture must be voiced"
        );
        assert_eq!(table(&remote_dir.0), local, "{name}");
    }
}

#[test]
fn the_server_answers_over_ipv4_and_ipv6_on_one_port() {
    let server = server(None);
    let ipv4 = ipv4(&server);
    let ipv6 = *server
        .addresses()
        .iter()
        .find(|address| address.is_ipv6())
        .expect("an IPv6 listener");
    assert_eq!(ipv6.port(), ipv4.port());
    assert!(ipv6.ip().is_loopback(), "{ipv6}");

    let dir = Scratch::new("ipv6");
    write_wav(&dir.0, "a.wav", &short_wav());
    let summary = run(&options(&dir.0, remote(ipv6, None))).unwrap();
    assert_eq!(summary.written, 1, "{:?}", summary.failed);
}

#[test]
fn dsp_estimators_run_locally_with_the_remote_runner() {
    // No server at all: a WORLD run never needs one.
    let dir = Scratch::new("dsp");
    write_wav(&dir.0, "a.wav", &short_wav());
    let mut opts = options(&dir.0, remote("127.0.0.1:9".parse().unwrap(), None));
    opts.f0.estimator = Estimator::Dio;
    let summary = run(&opts).unwrap();
    assert_eq!(summary.written, 1, "{:?}", summary.failed);
}

// --- configuration errors before any file -----------------------------------

#[test]
fn an_unreachable_server_is_a_configuration_error() {
    // Bind and drop a listener: nothing listens on that port now.
    let free = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let dir = Scratch::new("unreachable");
    let message = config_error(&options(&dir.0, remote(free, None)));
    assert!(message.contains("cannot reach the ML server"), "{message}");
    assert!(message.contains(&free.port().to_string()), "{message}");
}

#[test]
fn a_protocol_mismatch_is_a_configuration_error() {
    let server = scripted(
        Info {
            protocol: PROTOCOL_VERSION + 1,
            ..info()
        },
        Vec::new(),
    );
    let dir = Scratch::new("protocol");
    let message = config_error(&options(&dir.0, remote(server.address, None)));
    assert!(message.contains("not a compatible ML server"), "{message}");
    assert!(message.contains("protocol"), "{message}");
}

#[test]
fn a_differing_version_under_the_same_protocol_still_runs() {
    let server = scripted(
        Info {
            version: "0.0.1".to_string(),
            ..info()
        },
        Vec::new(),
    );
    let dir = Scratch::new("version");
    write_wav(&dir.0, "a.wav", &short_wav());
    let ml = remote(server.address, None);
    let summary = run(&options(&dir.0, ml.clone())).unwrap();
    assert_eq!(summary.written, 1, "{:?}", summary.failed);
    let warning = kirafrqgen_core::ml_server_info(&ml)
        .unwrap()
        .version_warning
        .expect("a version warning");
    assert!(warning.contains("0.0.1"), "{warning}");
}

#[test]
fn an_estimator_the_server_cannot_run_is_a_configuration_error() {
    let server = server(None);
    let dir = Scratch::new("unavailable");
    let mut opts = options(&dir.0, remote(ipv4(&server), None));
    opts.f0.estimator = Estimator::Rmvpe;
    let message = config_error(&opts);
    assert!(message.contains("cannot run RMVPE"), "{message}");
}

#[test]
fn a_missing_or_wrong_token_is_a_configuration_error() {
    let server = server(Some("open sesame"));
    let dir = Scratch::new("token");
    let message = config_error(&options(&dir.0, remote(ipv4(&server), None)));
    assert!(message.contains("requires a token"), "{message}");
    let message = config_error(&options(&dir.0, remote(ipv4(&server), Some("nope"))));
    assert!(message.contains("rejected the token"), "{message}");

    write_wav(&dir.0, "a.wav", &short_wav());
    let opts = options(&dir.0, remote(ipv4(&server), Some("open sesame")));
    assert_eq!(run(&opts).unwrap().written, 1);
}

// --- mid-run failures -------------------------------------------------------

#[test]
fn a_failed_request_is_retried_once() {
    let server = scripted(info(), vec![500]);
    let dir = Scratch::new("retry");
    write_wav(&dir.0, "a.wav", &short_wav());
    let summary = run(&options(&dir.0, remote(server.address, None))).unwrap();
    assert_eq!(summary.written, 1, "{:?}", summary.failed);
    assert_eq!(server.estimates.load(Ordering::SeqCst), 2);
}

#[test]
fn a_request_failing_twice_is_a_per_file_failure() {
    let server = scripted(info(), vec![500, 500]);
    let dir = Scratch::new("fail");
    write_wav(&dir.0, "a.wav", &short_wav());
    let summary = run(&options(&dir.0, remote(server.address, None))).unwrap();
    assert_eq!(summary.written, 0);
    assert_eq!(summary.failed.len(), 1);
    assert!(
        summary.failed[0].1.contains("scripted failure"),
        "{:?}",
        summary.failed
    );
    assert_eq!(
        server.estimates.load(Ordering::SeqCst),
        2,
        "one retry, no more"
    );
    assert!(!dir.0.join("a_wav.frq").exists(), "never a local fallback");
}
