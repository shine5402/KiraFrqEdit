//! The remote ML runner (#85): an [`F0Estimator`] that sends ML estimation
//! to an ML server. The pipeline sees only the seam, so decode, the energy
//! voicing gate and every writer run here exactly as for a local estimate.

use kirafrq_ml_remote::{Client, ClientError, EstimateParams, Model};

use crate::{
    Estimator, F0Config, F0Estimator, F0Track, FrameObserver, GeneratorError, MlConfig,
    MlServerInfo, ProgressStage,
};

/// The client for `config`'s server; a missing server is a configuration
/// error the front ends normally catch first.
fn client(config: &MlConfig) -> Result<Client, String> {
    let server = config
        .server
        .as_ref()
        .ok_or("the remote ML runner needs an ML server address")?;
    Ok(Client::new(
        &server.host,
        server.port,
        config.server_token.clone(),
    ))
}

pub(crate) fn server_info(config: &MlConfig) -> Result<MlServerInfo, String> {
    let info = client(config)?
        .check_access()
        .map_err(|error| error.to_string())?;
    Ok(MlServerInfo {
        summary: info.summary(),
        rmvpe: info.available(Model::Rmvpe),
        swiftf0: info.available(Model::SwiftF0),
        version_warning: info.version_warning(),
    })
}

/// Build the remote estimator after checking the server: it answers this
/// protocol, accepts the token and can run the selected ML estimator. Any of
/// those failing is a configuration error before any file runs.
pub(crate) fn build(config: &F0Config) -> Result<Box<dyn F0Estimator>, GeneratorError> {
    let model = match config.estimator {
        Estimator::Rmvpe => Model::Rmvpe,
        Estimator::SwiftF0 => Model::SwiftF0,
        Estimator::Dio | Estimator::Harvest => {
            return Err(GeneratorError::Config(format!(
                "{:?} is not an ML estimator",
                config.estimator
            )));
        }
    };
    let client = client(&config.ml).map_err(GeneratorError::Config)?;
    client
        .check(model)
        .map_err(|error| GeneratorError::Config(error.to_string()))?;
    Ok(Box::new(RemoteEstimator {
        client,
        params: EstimateParams {
            model,
            floor_hz: config.floor_hz,
            ceiling_hz: config.ceiling_hz,
            confidence_threshold: config.ml.confidence_threshold,
        },
    }))
}

struct RemoteEstimator {
    client: Client,
    params: EstimateParams,
}

impl F0Estimator for RemoteEstimator {
    fn estimate(
        &self,
        samples: &[f64],
        _sample_rate: u32,
        frame_period_ms: f64,
        observer: Option<&dyn FrameObserver>,
    ) -> Result<F0Track, GeneratorError> {
        // The existing coarse tick: before the request, then done.
        let report = |done| {
            if let Some(observer) = observer {
                observer.report(ProgressStage::Estimate, done, 1);
            }
        };
        report(0);
        let f0_hz = self
            .client
            .estimate(samples, self.params)
            .map_err(|error: ClientError| GeneratorError::Estimation(error.to_string()))?;
        report(1);
        let period_s = frame_period_ms / 1000.0;
        Ok(F0Track {
            frame_period_ms,
            temporal_positions: (0..f0_hz.len())
                .map(|frame| frame as f64 * period_s)
                .collect(),
            f0_hz,
        })
    }
}
