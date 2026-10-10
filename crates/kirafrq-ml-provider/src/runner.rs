//! The ML runner: which ONNX Runtime execution provider runs RMVPE and
//! SwiftF0, and the DirectML adapter list for the device picker.
//!
//! The neutral half — [`Runner`], [`SessionConfig`], [`DirectMlDevice`] — is
//! always compiled so the front ends can carry the choice in every build. The
//! DirectML parts are live only in Windows `ort` builds: `ort-sys` ships a
//! DirectML-enabled ONNX Runtime for Windows (and only Windows), and the
//! Cargo.toml target entry enables `ort/directml` there.
//!
//! DirectML's `device_id` is the adapter index in `IDXGIFactory::EnumAdapters`
//! order (directml's own docs); 0 is the default adapter, which on dual-GPU
//! laptops is often the integrated one. ONNX Runtime exposes no adapter
//! names, so [`directml_devices`] asks DXGI.

use std::fmt;

/// Which ONNX Runtime execution provider runs the ML models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Runner {
    /// No execution provider is registered: ONNX Runtime's CPU path, the
    /// standing default.
    #[default]
    Cpu,
    /// DirectML on a DirectX 12 GPU (Windows only). ONNX Runtime keeps its
    /// implicit CPU execution provider alongside it, so operations DirectML
    /// does not take — and a dispatch that cannot register at all — run on
    /// the CPU instead of failing the run.
    DirectMl,
}

/// The ML sessions' runtime settings: the runner and its device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SessionConfig {
    pub runner: Runner,
    /// The DirectML adapter index, `IDXGIFactory` order; `None` lets ONNX
    /// Runtime pick by performance preference (the high-performance GPU).
    /// Ignored unless the DirectML runner is selected.
    pub directml_device: Option<u32>,
}

/// Whether runs can use the DirectML runner at all in this build.
pub const DIRECTML_SUPPORTED: bool = cfg!(all(windows, feature = "ort"));

/// One DirectX 12 adapter, in the order DirectML's `device_id` follows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectMlDevice {
    /// The DXGI adapter index: this is the DirectML `device_id`.
    pub index: u32,
    /// The DXGI adapter description, e.g. `NVIDIA GeForce RTX 3070`.
    pub name: String,
}

/// Why the DirectML adapter list could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    /// This platform or build has no DirectML support.
    Unsupported,
    /// DXGI enumeration failed.
    Enumeration(String),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::Unsupported => {
                write!(f, "DirectML is only available in Windows ML builds")
            }
            DeviceError::Enumeration(message) => {
                write!(f, "cannot enumerate DirectML devices: {message}")
            }
        }
    }
}

impl std::error::Error for DeviceError {}

/// The hardware adapters DirectML can run on, in `device_id` order. Software
/// adapters (WARP) are skipped: they are never the intended target, and CPU
/// is the honest fallback for a host without a GPU.
///
/// # Errors
///
/// [`DeviceError::Unsupported`] outside Windows `ort` builds;
/// [`DeviceError::Enumeration`] when DXGI enumeration itself fails.
pub fn directml_devices() -> Result<Vec<DirectMlDevice>, DeviceError> {
    #[cfg(all(windows, feature = "ort"))]
    {
        windows_dxgi::devices()
    }
    #[cfg(not(all(windows, feature = "ort")))]
    {
        Err(DeviceError::Unsupported)
    }
}

#[cfg(all(windows, feature = "ort"))]
mod windows_dxgi {
    use super::{DeviceError, DirectMlDevice};
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1,
    };

    pub(super) fn devices() -> Result<Vec<DirectMlDevice>, DeviceError> {
        // SAFETY: DXGI factory creation and adapter enumeration are plain COM
        // calls; every returned interface is an owned handle released on drop.
        unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1()
                .map_err(|error| DeviceError::Enumeration(error.to_string()))?;
            let mut devices = Vec::new();
            let mut index = 0_u32;
            loop {
                // Out-of-range ends the enumeration with DXGI_ERROR_NOT_FOUND.
                let Ok(adapter) = factory.EnumAdapters1(index) else {
                    break;
                };
                let desc = adapter
                    .GetDesc1()
                    .map_err(|error| DeviceError::Enumeration(error.to_string()))?;
                let software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
                if !software {
                    let len = desc
                        .Description
                        .iter()
                        .position(|&unit| unit == 0)
                        .unwrap_or(desc.Description.len());
                    devices.push(DirectMlDevice {
                        index,
                        name: String::from_utf16_lossy(&desc.Description[..len])
                            .trim()
                            .to_owned(),
                    });
                }
                index += 1;
            }
            Ok(devices)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_session_is_the_cpu_with_automatic_device() {
        assert_eq!(
            SessionConfig::default(),
            SessionConfig {
                runner: Runner::Cpu,
                directml_device: None,
            }
        );
    }

    #[test]
    fn enumeration_is_rejected_where_directml_does_not_exist() {
        // On Windows `ort` builds this asks the host's DXGI, which may list
        // adapters or none; only the unsupported case is deterministic.
        #[cfg(not(all(windows, feature = "ort")))]
        assert!(matches!(directml_devices(), Err(DeviceError::Unsupported)));
    }
}
