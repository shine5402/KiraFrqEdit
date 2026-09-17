//! The voicebank tree the GUI shows for a planned run (#13/#23): built from
//! `kirafrqgen_core::plan`'s per-wav sidecar answers, so the "missing frq, pmk"
//! labels and the missing-only selection follow the production rules instead
//! of a GUI-local guess.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use kirafrqgen_core::{FilePlan, Target};

/// The formats the sidebar has checked.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Targets {
    pub frq: bool,
    pub pmk: bool,
    pub mrq: bool,
}

impl Targets {
    /// At least one format is checked.
    pub fn any(self) -> bool {
        self.frq || self.pmk || self.mrq
    }

    /// The checked formats as `plan` wants them.
    pub fn set(self) -> BTreeSet<Target> {
        self.checked().collect()
    }

    pub fn checked(self) -> impl Iterator<Item = Target> {
        let mut formats = Vec::new();
        if self.frq {
            formats.push(Target::Frq);
        }
        if self.pmk {
            formats.push(Target::Pmk);
        }
        if self.mrq {
            formats.push(Target::Mrq);
        }
        formats.into_iter()
    }
}

/// One wav and the targets its folder already holds for it (`plan`'s answer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavEntry {
    pub path: PathBuf,
    pub existing: BTreeSet<Target>,
}

/// One folder of the tree; `files` are its direct wavs, `dirs` its subfolders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirNode {
    pub name: String,
    pub path: PathBuf,
    pub dirs: Vec<DirNode>,
    pub files: Vec<WavEntry>,
    /// Wavs at or below this folder.
    pub total: usize,
}

/// The planned voicebank: root-level wavs plus the folder tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tree {
    pub root: PathBuf,
    pub root_files: Vec<WavEntry>,
    pub dirs: Vec<DirNode>,
    /// Per-wav plan notes (corrupt `desc.mrq` and friends), in plan order.
    pub warnings: Vec<(PathBuf, String)>,
}

impl DirNode {
    /// Every wav at or below this folder, subfolders first, in display order.
    pub(crate) fn entries(&self) -> Box<dyn Iterator<Item = &WavEntry> + '_> {
        Box::new(
            self.dirs
                .iter()
                .flat_map(|dir| dir.entries())
                .chain(self.files.iter()),
        )
    }
}

impl Tree {
    /// Group a run plan's wavs into the display tree. Paths that do not sit
    /// under `root` (a caller error) land at the root level.
    pub fn from_plan(root: &Path, files: &[FilePlan]) -> Self {
        let mut tree = Tree {
            root: root.to_path_buf(),
            root_files: Vec::new(),
            dirs: Vec::new(),
            warnings: Vec::new(),
        };
        for file in files {
            if !file.warnings.is_empty() {
                tree.warnings
                    .push((file.wav.clone(), file.warnings.join("; ")));
            }
            let entry = WavEntry {
                path: file.wav.clone(),
                existing: file.existing.clone(),
            };
            let relative = file.wav.strip_prefix(root).unwrap_or(Path::new(""));
            let components: Vec<&std::ffi::OsStr> = relative
                .components()
                .filter_map(|component| match component {
                    std::path::Component::Normal(name) => Some(name),
                    _ => None,
                })
                .collect();
            let Some((_, dirs)) = components.split_last() else {
                tree.root_files.push(entry);
                continue;
            };
            if dirs.is_empty() {
                tree.root_files.push(entry);
            } else {
                insert_entry(&mut tree.dirs, root, dirs, entry);
            }
        }
        for dir in &mut tree.dirs {
            set_totals(dir);
        }
        tree
    }
    /// Every wav in display order: root files first, then folders (subfolders
    /// ahead of the folder's own files), matching the rendered tree.
    pub fn all_wavs(&self) -> Vec<PathBuf> {
        let mut wavs: Vec<PathBuf> = self
            .root_files
            .iter()
            .map(|entry| entry.path.clone())
            .collect();
        for dir in &self.dirs {
            wavs.extend(dir.entries().map(|entry| entry.path.clone()));
        }
        wavs
    }

    /// The number of wavs in the tree, without materializing the list.
    pub fn total(&self) -> usize {
        self.root_files.len() + self.dirs.iter().map(|dir| dir.total).sum::<usize>()
    }

    /// The wavs missing at least one checked format, for "Select missing".
    pub fn select_missing(&self, targets: Targets) -> BTreeSet<PathBuf> {
        self.root_files
            .iter()
            .chain(self.dirs.iter().flat_map(|dir| dir.entries()))
            .filter(|entry| missing_any(entry, targets))
            .map(|entry| entry.path.clone())
            .collect()
    }

    /// Every wav's selected-descendant count, keyed by folder path, in one
    /// pass over the tree.
    pub fn selected_counts(&self, selected: &BTreeSet<PathBuf>) -> HashMap<PathBuf, usize> {
        let mut counts = HashMap::new();
        let mut root_count = self
            .root_files
            .iter()
            .filter(|entry| selected.contains(&entry.path))
            .count();
        for dir in &self.dirs {
            root_count += count_selected(dir, selected, &mut counts);
        }
        counts.insert(self.root.clone(), root_count);
        counts
    }
}

/// Add `entry` to the subtree named by `components` (dirs only; the file name
/// is already split off), creating folders as needed.
fn insert_entry(
    dirs: &mut Vec<DirNode>,
    parent: &Path,
    components: &[&std::ffi::OsStr],
    entry: WavEntry,
) {
    let name = components[0];
    let path = parent.join(name);
    let index = match dirs.iter().position(|dir| dir.path == path) {
        Some(index) => index,
        None => {
            dirs.push(DirNode {
                name: name.to_string_lossy().into_owned(),
                path: path.clone(),
                dirs: Vec::new(),
                files: Vec::new(),
                total: 0,
            });
            dirs.len() - 1
        }
    };
    if components.len() == 1 {
        dirs[index].files.push(entry);
    } else {
        let child_path = dirs[index].path.clone();
        insert_entry(&mut dirs[index].dirs, &child_path, &components[1..], entry);
    }
}

/// Fill in each node's wav count, deepest first.
fn set_totals(dir: &mut DirNode) -> usize {
    let mut total = dir.files.len();
    for child in &mut dir.dirs {
        total += set_totals(child);
    }
    dir.total = total;
    total
}

fn count_selected(
    dir: &DirNode,
    selected: &BTreeSet<PathBuf>,
    counts: &mut HashMap<PathBuf, usize>,
) -> usize {
    let mut count = dir
        .files
        .iter()
        .filter(|entry| selected.contains(&entry.path))
        .count();
    for child in &dir.dirs {
        count += count_selected(child, selected, counts);
    }
    counts.insert(dir.path.clone(), count);
    count
}

/// The display name of a format.
pub fn target_name(target: Target) -> &'static str {
    match target {
        Target::Frq => "frq",
        Target::Pmk => "pmk",
        Target::Mrq => "mrq",
    }
}

/// Whether `entry` lacks any checked format.
pub fn missing_any(entry: &WavEntry, targets: Targets) -> bool {
    targets
        .checked()
        .any(|target| !entry.existing.contains(&target))
}

/// The dim per-file label: the formats the entry lacks; `complete` when it has
/// them all, `—` when nothing is checked.
pub fn missing_label(entry: &WavEntry, targets: Targets) -> String {
    let missing: Vec<&str> = targets
        .checked()
        .filter(|target| !entry.existing.contains(target))
        .map(target_name)
        .collect();
    if !missing.is_empty() {
        format!("missing {}", missing.join(", "))
    } else if targets.any() {
        "complete".to_owned()
    } else {
        "—".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(root: &str, wavs: &[&str], existing: &[Target]) -> Vec<FilePlan> {
        wavs.iter()
            .map(|name| FilePlan {
                wav: Path::new(root).join(name),
                existing: existing.iter().copied().collect(),
                warnings: Vec::new(),
            })
            .collect()
    }

    fn targets(frq: bool, pmk: bool, mrq: bool) -> Targets {
        Targets { frq, pmk, mrq }
    }

    fn file(path: &str, existing: &[Target]) -> WavEntry {
        WavEntry {
            path: PathBuf::from(path),
            existing: existing.iter().copied().collect(),
        }
    }

    #[test]
    fn plan_paths_become_a_nested_tree_in_plan_order() {
        let files = plan(
            "/bank",
            &[
                "A2.wav",
                "A2/A2.wav",
                "A2/B/A3.wav",
                "A2/B/C/A4.wav",
                "D/A5.wav",
            ],
            &[],
        );

        let tree = Tree::from_plan(Path::new("/bank"), &files);

        assert_eq!(tree.root_files, [file("/bank/A2.wav", &[])]);
        assert_eq!(
            tree.dirs
                .iter()
                .map(|dir| dir.name.as_str())
                .collect::<Vec<_>>(),
            ["A2", "D"]
        );
        let a2 = &tree.dirs[0];
        assert_eq!(
            a2.files,
            [file("/bank/A2/A2.wav", &[])],
            "direct children only"
        );
        assert_eq!(
            a2.dirs
                .iter()
                .map(|dir| dir.name.as_str())
                .collect::<Vec<_>>(),
            ["B"]
        );
        assert_eq!(a2.dirs[0].files, [file("/bank/A2/B/A3.wav", &[])]);
        assert_eq!(a2.dirs[0].dirs[0].files, [file("/bank/A2/B/C/A4.wav", &[])]);
        assert_eq!(a2.total, 3, "A2/A2.wav, A2/B/A3.wav and A2/B/C/A4.wav");
        assert_eq!(tree.dirs[1].total, 1);
        assert_eq!(tree.total(), 5);
    }

    #[test]
    fn display_order_is_root_files_then_dirs() {
        let files = plan(
            "/bank",
            &["Root.wav", "A/A.wav", "A/B/B.wav", "A/C.wav"],
            &[],
        );
        let tree = Tree::from_plan(Path::new("/bank"), &files);

        assert_eq!(
            tree.all_wavs(),
            [
                Path::new("/bank").join("Root.wav"),
                Path::new("/bank").join("A").join("B").join("B.wav"),
                Path::new("/bank").join("A").join("A.wav"),
                Path::new("/bank").join("A").join("C.wav"),
            ],
            "root files, then folders; inside a folder subfolders before files"
        );
    }

    #[test]
    fn missing_uses_only_the_checked_formats() {
        let entry = file("/bank/A2.wav", &[Target::Frq]);

        assert_eq!(
            missing_label(&entry, targets(true, true, false)),
            "missing pmk"
        );
        assert_eq!(
            missing_label(&entry, targets(true, false, false)),
            "complete"
        );
        assert_eq!(
            missing_label(&entry, targets(true, true, true)),
            "missing pmk, mrq"
        );
        assert_eq!(missing_label(&entry, targets(false, false, false)), "—");
        assert!(missing_any(&entry, targets(false, true, false)));
        assert!(!missing_any(&entry, targets(true, false, false)));
    }

    #[test]
    fn select_missing_picks_files_lacking_a_checked_format() {
        let files = vec![
            FilePlan {
                wav: "/bank/A2.wav".into(),
                existing: [Target::Frq, Target::Pmk].into(),
                warnings: Vec::new(),
            },
            FilePlan {
                wav: "/bank/A3.wav".into(),
                existing: [Target::Frq].into(),
                warnings: Vec::new(),
            },
            FilePlan {
                wav: "/bank/B/A4.wav".into(),
                existing: [].into(),
                warnings: Vec::new(),
            },
        ];
        let tree = Tree::from_plan(Path::new("/bank"), &files);

        assert_eq!(
            tree.select_missing(targets(true, true, false)),
            BTreeSet::from([
                PathBuf::from("/bank/A3.wav"),
                PathBuf::from("/bank/B/A4.wav")
            ])
        );
        assert_eq!(
            tree.select_missing(targets(true, false, false)),
            BTreeSet::from([PathBuf::from("/bank/B/A4.wav")]),
            "A2 and A3 already have frq"
        );
        assert!(
            tree.select_missing(targets(false, false, false)).is_empty(),
            "nothing checked selects nothing"
        );
    }

    #[test]
    fn selected_counts_roll_up_into_folders() {
        let files = plan("/bank", &["A2.wav", "A/A.wav", "A/B/B.wav", "C/C.wav"], &[]);
        let tree = Tree::from_plan(Path::new("/bank"), &files);
        let selected = BTreeSet::from([
            PathBuf::from("/bank/A/A.wav"),
            PathBuf::from("/bank/A/B/B.wav"),
        ]);

        let counts = tree.selected_counts(&selected);

        assert_eq!(counts[Path::new("/bank")], 2);
        assert_eq!(counts[Path::new("/bank/A")], 2);
        assert_eq!(counts[Path::new("/bank/A/B")], 1);
        assert_eq!(counts[Path::new("/bank/C")], 0);
    }

    #[test]
    fn plan_warnings_are_kept_for_the_ui() {
        let files = vec![FilePlan {
            wav: "/bank/A/A2.wav".into(),
            existing: BTreeSet::new(),
            warnings: vec!["desc.mrq is corrupt".into()],
        }];
        let tree = Tree::from_plan(Path::new("/bank"), &files);

        assert_eq!(
            tree.warnings,
            [(
                PathBuf::from("/bank/A/A2.wav"),
                "desc.mrq is corrupt".into()
            )]
        );
    }
}
