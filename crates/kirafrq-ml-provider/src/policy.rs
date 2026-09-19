//! The #53 voiced/unvoiced policy on the ML estimator's native grid: a
//! confidence gate AND a finite-in-range gate, else exactly `0.0` — never
//! clamped, no smoothing, no gap-fill.

/// RMVPE's de-facto confidence threshold (RVC `maxx <= 0.03`; pitch-core
/// `peak_p >= 0.03`), the default when the config carries no override.
pub const RMVPE_DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.03;

/// The native-grid voicing decision: `f0` is voiced iff it is finite, inside
/// the inclusive `[floor_hz, ceiling_hz]` range, and its peak salience is at
/// or above the confidence threshold. Order of the two gates is immaterial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvPolicy {
    pub confidence_threshold: f64,
    pub floor_hz: f64,
    pub ceiling_hz: f64,
}

impl UvPolicy {
    /// The #53 defaults for a model with contract threshold `threshold`:
    /// RMVPE's 0.03, the map's 71-800 Hz range (#46).
    pub fn rmvpe(threshold: Option<f64>) -> Self {
        Self {
            confidence_threshold: threshold.unwrap_or(RMVPE_DEFAULT_CONFIDENCE_THRESHOLD),
            floor_hz: 71.0,
            ceiling_hz: 800.0,
        }
    }

    /// The emitted value for one native frame: `f0` when voiced, else exactly
    /// `0.0`. Out-of-range pitch is dropped, never clamped.
    pub fn apply(&self, f0: f64, confidence: f64) -> f64 {
        if confidence >= self.confidence_threshold
            && f0.is_finite()
            && f0 >= self.floor_hz
            && f0 <= self.ceiling_hz
        {
            f0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> UvPolicy {
        UvPolicy::rmvpe(None)
    }

    #[test]
    fn the_rmvpe_default_threshold_is_0_03_and_the_override_wins() {
        assert_eq!(policy().confidence_threshold, 0.03);
        assert_eq!(UvPolicy::rmvpe(Some(0.9)).confidence_threshold, 0.9);
    }

    #[test]
    fn the_boundary_spelling_joins_at_the_threshold() {
        let policy = policy();
        assert_eq!(policy.apply(220.0, 0.03 - 1e-12), 0.0);
        assert_eq!(policy.apply(220.0, 0.03), 220.0);
        assert_eq!(policy.apply(220.0, 0.03 + 1e-12), 220.0);
    }

    #[test]
    fn the_range_gate_is_inclusive_and_never_clamps() {
        let policy = policy();
        assert_eq!(policy.apply(71.0, 1.0), 71.0);
        assert_eq!(policy.apply(800.0, 1.0), 800.0);
        assert_eq!(policy.apply(70.999, 1.0), 0.0);
        assert_eq!(policy.apply(800.001, 1.0), 0.0);
        assert_eq!(policy.apply(f64::NAN, 1.0), 0.0);
        assert_eq!(policy.apply(f64::INFINITY, 1.0), 0.0);
        assert_eq!(policy.apply(-220.0, 1.0), 0.0);
    }

    #[test]
    fn a_confidence_below_threshold_unvoices_in_range_pitch() {
        let policy = policy();
        assert_eq!(policy.apply(440.0, 0.029), 0.0);
    }
}
