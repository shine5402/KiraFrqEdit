//! `--serve` (#85): run this machine as an ML server until Ctrl-C.
//!
//! The startup report goes to stdout, one fact per line, then one line per
//! answered request; warnings (a taken default port, RMVPE's missing model)
//! go to stderr like a run's.

use std::io::Write;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use kirafrq_ml_provider::{Runner, SessionConfig};
use kirafrq_ml_remote::server::{Interfaces, Port, Server, ServerOptions};
use kirafrqgen_core::{ML_DOWNLOAD_HINT, MlRunner};

use crate::args::Cli;

/// Serve until Ctrl-C; the exit code is `0` after a clean stop, `1` when the
/// server cannot start.
pub fn run(cli: &Cli) -> u8 {
    let stop = Arc::new(AtomicBool::new(false));
    let handler_stop = Arc::clone(&stop);
    if let Err(error) = ctrlc::set_handler(move || handler_stop.store(true, Ordering::SeqCst)) {
        eprintln!("warning: cannot install the Ctrl-C handler: {error}");
    }

    let runner = cli.ml_runner.map_or_else(
        || kirafrqgen_core::default_ml_runner(kirafrqgen_core::ml_runner_usable),
        MlRunner::from,
    );
    let directml_device = cli.ml_device.filter(|_| runner.is_directml());
    let session_config = SessionConfig {
        runner: match runner {
            MlRunner::Cpu => Runner::Cpu,
            MlRunner::DirectMl => Runner::DirectMl,
            MlRunner::CoreMl => Runner::CoreMl,
            // `validate_ml_flags` rejects a remote runner for a server.
            MlRunner::Remote => unreachable!("an ML server cannot use the remote runner"),
        },
        directml_device,
    };
    let runner_line = format!(
        "{} ({})",
        runner_label(runner),
        device_label(runner, directml_device)
    );

    let server = match Server::start(ServerOptions {
        interfaces: Interfaces::All,
        port: cli.port.map_or(Port::Default, Port::Explicit),
        token: cli.token.clone(),
        session_config,
        runner_label: runner_label(runner).to_string(),
        device_label: device_label(runner, directml_device),
        threads: cli.jobs,
        rmvpe: kirafrq_ml_provider::resolve_model(),
        swiftf0: kirafrq_ml_provider::swiftf0::resolve_source(),
        // Not `println!`: it panics once stdout is gone, killing the worker.
        log: Some(|line| {
            let _ = writeln!(std::io::stdout(), "{line}");
        }),
    }) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("error: {error}");
            return 1;
        }
    };
    if let Some(warning) = server.port_warning() {
        eprintln!("warning: {warning}");
    }
    if let Some(missing) = server.rmvpe_missing() {
        eprintln!("warning: RMVPE is unavailable on this server: {missing}; {ML_DOWNLOAD_HINT}");
    }

    let info = server.info();
    let addresses = server.addresses();
    let port = addresses[0].port();
    let lan: Vec<String> = lan_addresses()
        .into_iter()
        .map(|ip| SocketAddr::new(ip, port).to_string())
        .collect();

    println!("KiraFrqGen {} ML server", info.version);
    println!(
        "Listening on {}",
        addresses
            .iter()
            .map(SocketAddr::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    if lan.is_empty() {
        println!("Connect to: this machine's IP address, port {port}");
    } else {
        println!("Connect to: {}", lan.join(", "));
    }
    println!("ML runner: {runner_line}");
    println!("ML estimators: {}", info.estimator_names().join(", "));
    println!(
        "Token: {}",
        if info.token_required {
            "required"
        } else {
            "not required"
        }
    );
    println!("Press Ctrl-C to stop.");

    let running = server.spawn();
    while !stop.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(200));
    }
    drop(running);
    println!("Stopped.");
    0
}

fn runner_label(runner: MlRunner) -> &'static str {
    match runner {
        MlRunner::Cpu => "CPU",
        MlRunner::DirectMl => "DirectML",
        MlRunner::CoreMl => "Core ML",
        MlRunner::Remote => "Remote",
    }
}

/// What the runner runs on, as `/v1/info` and the startup report name it.
fn device_label(runner: MlRunner, directml_device: Option<u32>) -> String {
    match runner {
        MlRunner::Cpu => "processor".to_string(),
        MlRunner::CoreMl => "the Mac's GPU".to_string(),
        MlRunner::DirectMl => directml_device
            .and_then(|index| {
                kirafrqgen_core::ml_devices()
                    .ok()?
                    .into_iter()
                    .find(|device| device.index == index)
            })
            .map_or_else(
                || "automatic: the high-performance GPU".to_string(),
                |device| device.name,
            ),
        MlRunner::Remote => "another server".to_string(),
    }
}

/// This machine's addresses a client can type: IPv4, then IPv6, without
/// loopback and link-local ones (an IPv6 link-local address needs a zone).
fn lan_addresses() -> Vec<IpAddr> {
    let mut addresses: Vec<IpAddr> = if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .map(|interface| interface.ip())
        .filter(|ip| match ip {
            IpAddr::V4(ip) => !ip.is_loopback() && !ip.is_link_local(),
            IpAddr::V6(ip) => !ip.is_loopback() && !ip.is_unicast_link_local(),
        })
        .collect();
    addresses.sort();
    addresses.dedup();
    addresses
}
