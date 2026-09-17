//! The `.llsm` prompt policy (#12): the only interactive question.
//!
//! An mrq write invalidates moresampler's `.llsm` cache (#4/#10; frq is not
//! moresampler's f0 source, so frq-only runs never touch caches), so a run
//! asks once upfront — but only when an mrq write is actually planned, so an
//! all-existing fill-missing run never pauses. An explicit llsm flag or `-y`
//! answers the question without asking, and outside a terminal the default
//! (delete) applies.

use std::collections::BTreeSet;
use std::io::{BufRead, IsTerminal, Write};

use kira_frqgen::{RunPlan, Target};

use crate::would_write;

/// Whether the run would write an mrq f0 entry for any wav.
pub fn mrq_write_planned(plan: &RunPlan, targets: &BTreeSet<Target>, overwrite: bool) -> bool {
    targets.contains(&Target::Mrq)
        && plan
            .files
            .iter()
            .any(|file| would_write(file, Target::Mrq, overwrite))
}

/// Resolve `.llsm` deletion (#12): `--dry-run` writes nothing and never asks;
/// an explicit flag wins; `-y` takes the default (delete); and otherwise `ask`
/// decides — only when an mrq write is planned. The caller passes the
/// interactive question in a terminal and the default (delete) outside one.
pub fn resolve_llsm(
    explicit: Option<bool>,
    yes: bool,
    dry_run: bool,
    mrq_write_planned: bool,
    ask: impl FnOnce() -> bool,
) -> bool {
    match explicit {
        Some(delete) => delete,
        None if dry_run || yes || !mrq_write_planned => true,
        None => ask(),
    }
}

/// Whether a prompt has both a terminal to read from and one to write to;
/// with either redirected the documented non-interactive default applies.
pub fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// Ask the `.llsm` question on stderr and read the answer from stdin. Empty
/// input and EOF take the default (yes); an unrecognized answer re-asks.
pub fn ask_llsm() -> bool {
    loop {
        eprint!("kira-frqgen: delete cached .llsm files after mrq writes? [Y/n] ");
        let _ = std::io::stderr().flush();
        let mut answer = String::new();
        match std::io::stdin().lock().read_line(&mut answer) {
            Ok(0) | Err(_) => return true,
            Ok(_) => match answer.trim().to_ascii_lowercase().as_str() {
                "" | "y" | "yes" => return true,
                "n" | "no" => return false,
                _ => eprintln!("kira-frqgen: please answer y or n"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::path::PathBuf;

    use kira_frqgen::{FilePlan, RunPlan};

    fn plan(existing: &[Target]) -> RunPlan {
        RunPlan {
            files: vec![FilePlan {
                wav: PathBuf::from("bank/A2.wav"),
                existing: existing.iter().copied().collect(),
                warnings: Vec::new(),
            }],
        }
    }

    fn targets(selected: &[Target]) -> BTreeSet<Target> {
        selected.iter().copied().collect()
    }

    #[test]
    fn only_mrq_writes_invalidate_llsm() {
        let all = targets(&[Target::Frq, Target::Pmk, Target::Mrq]);
        assert!(mrq_write_planned(&plan(&[]), &all, false));
        assert!(mrq_write_planned(&plan(&[Target::Frq]), &all, false));
        assert!(
            !mrq_write_planned(&plan(&[]), &targets(&[Target::Frq]), false),
            "frq is not moresampler's f0 source"
        );
        assert!(
            !mrq_write_planned(&plan(&[]), &targets(&[Target::Pmk]), false),
            "pmk alone never invalidates llsm"
        );
    }

    #[test]
    fn existing_mrq_entries_only_plan_writes_with_overwrite() {
        let all = targets(&[Target::Frq, Target::Mrq]);
        assert!(!mrq_write_planned(&plan(&[Target::Mrq]), &all, false));
        assert!(mrq_write_planned(&plan(&[Target::Mrq]), &all, true));
    }

    #[test]
    fn an_explicit_flag_wins_without_asking() {
        let asked = Cell::new(false);
        let delete = resolve_llsm(Some(false), false, false, true, || {
            asked.set(true);
            true
        });
        assert!(!delete);
        assert!(!asked.get());
    }

    #[test]
    fn yes_takes_the_default_without_asking() {
        let asked = Cell::new(false);
        let delete = resolve_llsm(None, true, false, true, || {
            asked.set(true);
            false
        });
        assert!(delete);
        assert!(!asked.get());
    }

    #[test]
    fn no_planned_mrq_write_never_asks() {
        let asked = Cell::new(false);
        let delete = resolve_llsm(None, false, false, false, || {
            asked.set(true);
            false
        });
        assert!(delete);
        assert!(!asked.get());
    }

    #[test]
    fn a_dry_run_never_asks() {
        let asked = Cell::new(false);
        let delete = resolve_llsm(None, false, true, true, || {
            asked.set(true);
            false
        });
        assert!(delete);
        assert!(!asked.get());
    }

    #[test]
    fn a_planned_write_asks_and_obeys_the_answer() {
        let asked = Cell::new(false);
        assert!(resolve_llsm(None, false, false, true, || {
            asked.set(true);
            true
        }));
        assert!(asked.replace(false));
        assert!(!resolve_llsm(None, false, false, true, || {
            asked.set(true);
            false
        }));
        assert!(asked.get());
    }
}
