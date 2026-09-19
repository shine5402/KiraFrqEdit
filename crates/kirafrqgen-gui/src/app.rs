//! The KiraFrqGen window (#23): shell B from #13 — options sidebar, drop area
//! that becomes a tri-state wav tree, and an in-place run view fed by the
//! pipeline's progress events. Settings live for the session only (#13).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use eframe::egui::{
    self, Align, Align2, Color32, FontId, Key, Layout, RichText, Sense, Shape, Stroke, StrokeKind,
    Vec2,
    widgets::text_edit::{TextEditOutput, TextEditState},
};
use egui_phosphor::regular as icons;
use kirafrqgen_core::{
    CancelToken, Estimator, F0Config, FileReport, GenerateOptions, Progress, RunSummary, Sharing,
    Target, build_estimator, generate_wavs, ml_available, plan,
};

use crate::run::{Row, RunState, Status};
use crate::text_ops::{self, CharRange};
use crate::tree::{DirNode, Targets, Tree, WavEntry, missing_label, target_name};

/// Lock the run state, recovering a poisoned mutex: a panicked worker must not
/// take the window down with it.
fn lock(state: &Mutex<RunState>) -> MutexGuard<'_, RunState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The path field's selection as a character range.
fn range_of(state: &TextEditState) -> CharRange {
    match state.cursor.char_range() {
        Some(range) => {
            let range = range.as_sorted_char_range();
            CharRange::new(range.start.0, range.end.0)
        }
        None => CharRange::caret(0),
    }
}

/// Store a character range back into the field's text-edit state.
fn set_range(state: &mut TextEditState, range: CharRange) {
    state
        .cursor
        .set_char_range(Some(egui::text::CCursorRange::two(
            egui::text::CCursor::new(range.start),
            egui::text::CCursor::new(range.end),
        )));
}

/// The system clipboard behind the path field's context menu (#32): the handle
/// is opened lazily, and `text` snapshots a read so Paste's enabled state can be
/// decided without touching the clipboard every frame.
#[derive(Default)]
struct PathClipboard {
    handle: Option<arboard::Clipboard>,
    text: Option<String>,
}

impl PathClipboard {
    /// Re-read the system clipboard for the menu's Paste item. A clipboard that
    /// cannot be opened (headless, or held by another process) yields `None`,
    /// which disables Paste rather than risking a destructive edit.
    fn read(&mut self) -> Option<String> {
        if self.handle.is_none() {
            self.handle = arboard::Clipboard::new().ok();
        }
        self.handle
            .as_mut()
            .and_then(|clipboard| clipboard.get_text().ok())
    }

    /// Read once when the menu opens, to drive its Paste item.
    fn snapshot(&mut self) {
        self.text = self.read();
    }

    fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }
}

/// The path field's context-menu choices (#32).
#[derive(Clone, Copy)]
enum PathMenu {
    Cut,
    Copy,
    Paste,
    SelectAll,
}

/// Draw the path field's context menu, returning the item the user picked.
fn path_menu(ui: &mut egui::Ui, menu: &text_ops::MenuState) -> Option<PathMenu> {
    let item = |ui: &mut egui::Ui, enabled: bool, label: &str, choice: PathMenu| {
        if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
            ui.close();
            Some(choice)
        } else {
            None
        }
    };
    item(ui, menu.cut, "Cut", PathMenu::Cut)
        .or_else(|| item(ui, menu.copy, "Copy", PathMenu::Copy))
        .or_else(|| item(ui, menu.paste, "Paste", PathMenu::Paste))
        .or_else(|| {
            ui.separator();
            item(ui, menu.select_all, "Select All", PathMenu::SelectAll)
        })
}

/// A run in flight: shared state the worker writes and the window reads, plus
/// the token that stops it. The worker thread is detached; a run is only ever
/// replaced once its state says it finished.
struct RunSession {
    state: Arc<Mutex<RunState>>,
    stop: CancelToken,
}

impl RunSession {
    /// The Cancel button: flag the UI and ask the pipeline to stop after the
    /// file it is on.
    fn cancel(&self) {
        self.stop.store(true, Ordering::SeqCst);
        lock(&self.state).request_cancel();
    }
}

impl Drop for RunSession {
    fn drop(&mut self) {
        // A detached worker may be mid-file: ask it to stop after the current
        // one rather than blocking the UI thread on a join.
        self.stop.store(true, Ordering::SeqCst);
    }
}

/// The [`Progress`] bridge into the shared run state; every event asks egui
/// for a repaint so the rows move while the worker is busy.
struct GuiProgress {
    state: Arc<Mutex<RunState>>,
    ctx: egui::Context,
}

impl Progress for GuiProgress {
    fn file_started(&self, wav: &Path) {
        lock(&self.state).started(wav);
        self.ctx.request_repaint();
    }

    fn wants_file_progress(&self) -> bool {
        // The run view draws the determinate pie (#33/#34).
        true
    }

    fn file_progress(&self, wav: &Path, done: u64, total: u64) {
        lock(&self.state).progressed(wav, done, total);
        self.ctx.request_repaint();
    }

    fn file_finished(&self, report: &FileReport) {
        lock(&self.state).finished_file(report);
        self.ctx.request_repaint();
    }

    fn finished(&self, summary: &RunSummary) {
        lock(&self.state).finished(summary);
        self.ctx.request_repaint();
    }
}

pub struct KiraFrqGenApp {
    // Options (session-only).
    estimator: Estimator,
    /// Whether the ML tier can run; resolves the capability-aware default
    /// once at startup (#49) and greys the RMVPE radio when false.
    ml_available: bool,
    world_quirks: bool,
    targets: Targets,
    delete_llsm: bool,
    japanese_codepage: bool,
    cores_auto: bool,
    cores: u32,
    max_cores: u32,

    // The voicebank.
    path: String,
    tree: Option<Tree>,
    plan_error: Option<String>,
    expanded: HashSet<PathBuf>,
    selected: BTreeSet<PathBuf>,

    /// The system clipboard, opened lazily by the path field's context menu
    /// (#32); an unavailable clipboard disables its Paste item.
    clipboard: PathClipboard,

    /// The Help > Credits window: whether it is open, whether it shows the
    /// full license texts (the default), and the rendered text it displays.
    credits_open: bool,
    credits_full: bool,
    credits_text: String,

    run: Option<RunSession>,
}

impl KiraFrqGenApp {
    pub fn new() -> Self {
        let max_cores = std::thread::available_parallelism()
            .map(|cores| cores.get() as u32)
            .unwrap_or(4);
        let ml_available = ml_available(&F0Config::default());
        Self {
            estimator: kirafrqgen_core::default_estimator(ml_available),
            ml_available,
            world_quirks: true,
            targets: Targets {
                frq: true,
                pmk: false,
                mrq: false,
            },
            // Deleting the cache is what makes moresampler notice a changed
            // `desc.mrq` (#4), so the correct behavior is the default.
            delete_llsm: true,
            japanese_codepage: false,
            cores_auto: true,
            cores: max_cores,
            max_cores,
            path: String::new(),
            tree: None,
            plan_error: None,
            expanded: HashSet::new(),
            selected: BTreeSet::new(),
            clipboard: PathClipboard::default(),
            credits_open: false,
            credits_full: true,
            credits_text: String::new(),
            run: None,
        }
    }

    fn run_state(&self) -> Option<MutexGuard<'_, RunState>> {
        self.run.as_ref().map(|session| lock(&session.state))
    }

    fn is_running(&self) -> bool {
        self.run_state().is_some_and(|state| state.is_running())
    }

    fn run_ended(&self) -> bool {
        self.run_state().is_some_and(|state| !state.is_running())
    }

    /// `Sharing::native` is a 932 no-op on platforms without an 8-bit code
    /// page, so checking the box cannot force a conversion there.
    fn sharing(&self) -> Option<Sharing> {
        (self.japanese_codepage && self.targets.mrq).then(Sharing::native)
    }

    /// Plan with every format so the tree can label each one, regardless of
    /// what the sidebar has checked.
    fn plan_options(&self, root: PathBuf) -> GenerateOptions {
        GenerateOptions {
            root,
            targets: [Target::Frq, Target::Pmk, Target::Mrq]
                .into_iter()
                .collect(),
            overwrite: false,
            f0: F0Config::default(),
            jobs: 0,
            sharing: self.sharing(),
            delete_llsm: false,
        }
    }

    /// Run options: the checked formats only. `overwrite` is on because the
    /// GUI has no overwrite control — the *selection* is the overwrite choice
    /// (#13), with "Select missing" as the safe default, so every selected
    /// wav is (re)generated.
    fn run_options(&self, root: PathBuf) -> GenerateOptions {
        GenerateOptions {
            root,
            targets: self.targets.set(),
            overwrite: true,
            f0: F0Config {
                estimator: self.estimator,
                world_quirks: self.world_quirks,
                ..F0Config::default()
            },
            jobs: if self.cores_auto {
                0
            } else {
                self.cores as usize
            },
            sharing: self.sharing(),
            delete_llsm: self.delete_llsm,
        }
    }

    fn reset(&mut self) {
        self.run = None;
        self.path.clear();
        self.tree = None;
        self.plan_error = None;
        self.expanded.clear();
        self.selected.clear();
    }

    fn set_path(&mut self, path: PathBuf) {
        self.path = path.display().to_string();
        self.rescan();
    }

    /// Re-plan the folder: the scan plus the sidecar rules decide the labels
    /// and the missing-only default selection.
    fn rescan(&mut self) {
        self.run = None;
        self.tree = None;
        self.plan_error = None;
        self.selected.clear();
        self.expanded.clear();
        let root = PathBuf::from(self.path.trim());
        if root.as_os_str().is_empty() {
            return;
        }
        match plan(&self.plan_options(root.clone())) {
            Ok(run_plan) => {
                let tree = Tree::from_plan(&root, &run_plan.files);
                for dir in &tree.dirs {
                    collect_dir_paths(dir, &mut self.expanded);
                }
                self.selected = tree.select_missing(self.targets);
                self.tree = Some(tree);
            }
            Err(error) => self.plan_error = Some(error.to_string()),
        }
    }

    fn browse(&mut self) {
        if let Some(picked) = rfd::FileDialog::new()
            .set_title("Voicebank folder")
            .pick_folder()
        {
            self.set_path(picked);
        }
    }

    /// Re-render the credits for the current full/compact choice, so the window
    /// pays the markdown render once instead of every frame.
    fn refresh_credits(&mut self) {
        self.credits_text = kirafrq_credits::render(
            kirafrq_credits::for_display(self.credits_full),
            kirafrq_credits::Style::Plain,
        );
    }

    /// The Help > Credits window.
    fn credits_window(&mut self, ctx: &egui::Context) {
        if !self.credits_open {
            return;
        }
        let mut open = self.credits_open;
        egui::Window::new("Credits")
            .open(&mut open)
            .default_size([640.0, 480.0])
            .resizable(true)
            .show(ctx, |ui| {
                if ui
                    .checkbox(&mut self.credits_full, "Show full license texts")
                    .changed()
                {
                    self.refresh_credits();
                }
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.add(
                            egui::Label::new(RichText::new(self.credits_text.as_str()).monospace())
                                .selectable(true)
                                .wrap(),
                        );
                    });
            });
        self.credits_open = open;
    }

    /// Hand the selected wavs to the real pipeline on a worker thread.
    fn start_run(&mut self, ctx: &egui::Context) {
        let Some(tree) = &self.tree else {
            return;
        };
        let wavs: Vec<PathBuf> = tree
            .all_wavs()
            .into_iter()
            .filter(|wav| self.selected.contains(wav))
            .collect();
        if wavs.is_empty() || !self.targets.any() {
            return;
        }
        let opts = self.run_options(tree.root.clone());
        let state = Arc::new(Mutex::new(RunState::new(wavs.clone())));
        let stop: CancelToken = Arc::new(AtomicBool::new(false));
        let progress = GuiProgress {
            state: Arc::clone(&state),
            ctx: ctx.clone(),
        };
        let worker_state = Arc::clone(&state);
        let worker_stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let outcome = match build_estimator(&opts.f0, opts.jobs) {
                Ok(estimator) => {
                    generate_wavs(&opts, &wavs, estimator.as_ref(), &progress, &worker_stop)
                }
                Err(error) => Err(error),
            };
            let mut state = lock(&worker_state);
            match outcome {
                Ok(summary) => state.finished(&summary),
                Err(error) => state.aborted(error.to_string()),
            }
        });
        self.run = Some(RunSession { state, stop });
    }

    fn tree_total(&self) -> usize {
        self.tree.as_ref().map(Tree::total).unwrap_or(0)
    }

    // ---------------------------------------------------------------- chrome

    fn sidebar(&mut self, ui: &mut egui::Ui) {
        let running = self.is_running();
        let mut open_credits = false;
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("KiraFrqGen").size(22.0).strong());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.menu_button(icons::LIST, |ui| {
                    ui.menu_button("Help", |ui| {
                        if ui.button("Credits").clicked() {
                            ui.close();
                            open_credits = true;
                        }
                    });
                })
                .response
                .on_hover_text("Help");
            });
        });
        if open_credits {
            self.credits_open = true;
            self.refresh_credits();
        }
        ui.weak("Bulk-generate frq tables for your UTAU voicebank.");
        ui.add_space(10.0);
        ui.separator();

        ui.add_enabled_ui(!running, |ui| {
            ui.label(RichText::new("ESTIMATOR").size(13.0).strong());
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.estimator, Estimator::Harvest, "Harvest");
                ui.radio_value(&mut self.estimator, Estimator::Dio, "DIO");
                // #48: the compat build has no ML option at all.
                if cfg!(feature = "ml") {
                    let rmvpe = ui.add_enabled(
                        self.ml_available,
                        egui::RadioButton::new(self.estimator == Estimator::Rmvpe, "RMVPE"),
                    );
                    if rmvpe.clicked() {
                        self.estimator = Estimator::Rmvpe;
                    }
                    if !self.ml_available {
                        rmvpe.on_hover_text(
                            "RMVPE needs a model file. Put `rmvpe.onnx` next to the executable \
                             or in the directory named by KIRAFRQ_ML_DIR, then restart.",
                        );
                    }
                    // #69: SwiftF0 ships a bundled model, so it is always
                    // selectable in a modern build.
                    ui.radio_value(&mut self.estimator, Estimator::SwiftF0, "SwiftF0");
                }
            });
            ui.label(RichText::new(self.estimator.description()).weak());

            ui.add_space(8.0);
            let quirks = ui.add_enabled(
                self.estimator.is_world(),
                egui::Checkbox::new(&mut self.world_quirks, "Use WORLD quirks"),
            );
            quirks.on_hover_text(
                "Enable tuning and migrations for WORLD estimators that can reduce false \
                 positive on noise or breaths. Disable this to gain the original WORLD \
                 behavior.",
            );

            ui.add_space(8.0);
            ui.label(RichText::new("FORMATS").size(13.0).strong());
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.targets.frq, "frq");
                ui.checkbox(&mut self.targets.pmk, "pmk");
                ui.checkbox(&mut self.targets.mrq, "mrq");
            });

            ui.add_space(8.0);
            ui.label(RichText::new("BEHAVIOR").size(13.0).strong());
            ui.checkbox(&mut self.delete_llsm, "Delete .llsm caches")
                .on_hover_text(
                    "moresampler reads its .llsm cache instead of a changed desc.mrq (#4); \
                 deleted only for wavs whose mrq entry is written",
                );
            let codepage_hint = if self.targets.mrq {
                "Also keys each mrq entry by the filename's Japanese-code-page spelling, so the \
                 generated tables keep working on Japanese-locale machines"
            } else {
                "Applies to mrq entries only; select mrq to use this"
            };
            if ui
                .add_enabled(
                    self.targets.mrq,
                    egui::Checkbox::new(&mut self.japanese_codepage, "Ensure Japanese codepage"),
                )
                .on_hover_text(codepage_hint)
                .changed()
                && self.tree.is_some()
            {
                // This option changes which mrq entries count as existing, so
                // the tree's labels and default selection need a re-plan.
                self.rescan();
            }

            ui.add_space(8.0);
            ui.label(RichText::new("CORES").size(13.0).strong());
            ui.horizontal(|ui| {
                if ui.checkbox(&mut self.cores_auto, "All cores").changed() && self.cores_auto {
                    self.cores = self.max_cores;
                }
                ui.add_enabled(
                    !self.cores_auto,
                    egui::DragValue::new(&mut self.cores).range(1..=self.max_cores),
                );
            });
        });

        ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
            ui.add_space(10.0);
            let enabled = !self.is_running()
                && !self.selected.is_empty()
                && self.targets.any()
                && self.tree.is_some();
            if running {
                let cancelling = self
                    .run_state()
                    .is_some_and(|state| state.cancel_requested());
                ui.scope(|ui| {
                    let red = if ui.visuals().dark_mode {
                        Color32::from_rgb(127, 29, 29)
                    } else {
                        Color32::from_rgb(254, 202, 202)
                    };
                    let widgets = &mut ui.visuals_mut().widgets;
                    widgets.hovered.weak_bg_fill = red;
                    widgets.hovered.bg_fill = red;
                    widgets.hovered.bg_stroke = Stroke::new(1.0, red);
                    let label = if cancelling {
                        "Cancelling…".to_owned()
                    } else {
                        format!("{} Cancel", icons::X)
                    };
                    if ui
                        .add_enabled(
                            !cancelling,
                            egui::Button::new(label)
                                .min_size(Vec2::new(ui.available_width(), 34.0)),
                        )
                        .on_hover_text("Stop after the current file")
                        .clicked()
                        && let Some(session) = &self.run
                    {
                        session.cancel();
                    }
                });
            } else if self.run_ended() {
                if ui
                    .add_sized(
                        [ui.available_width(), 34.0],
                        egui::Button::new(format!("{} Finish", icons::CHECK)),
                    )
                    .on_hover_text("Close the folder and start over")
                    .clicked()
                {
                    self.reset();
                }
            } else {
                let label = format!("{} Generate", icons::PLAY);
                if ui
                    .add_enabled(
                        enabled,
                        egui::Button::new(RichText::new(label).strong())
                            .min_size(Vec2::new(ui.available_width(), 34.0)),
                    )
                    .clicked()
                {
                    let ctx = ui.ctx().clone();
                    self.start_run(&ctx);
                }
            }
            ui.add_space(4.0);
            if self.tree.is_some() {
                ui.weak(format!(
                    "{} of {} wavs selected",
                    self.selected.len(),
                    self.tree_total()
                ));
            } else {
                ui.weak("no folder selected");
            }
            ui.separator();
        });
    }

    fn path_bar(&mut self, ui: &mut egui::Ui) {
        let running = self.is_running();
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} Voicebank", icons::FOLDER)).strong());
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .button(format!("{} Rescan", icons::ARROWS_CLOCKWISE))
                        .clicked()
                    {
                        self.rescan();
                    }
                    if ui
                        .button(format!("{} Browse", icons::FOLDER_OPEN))
                        .clicked()
                    {
                        self.browse();
                    }

                    let field_id = ui.make_persistent_id("voicebank_path");
                    // Snapshot the selection before the field runs: a right
                    // click collapses it onto the click position, and the menu
                    // must not lose it (#32).
                    let previous = egui::TextEdit::load_state(ui.ctx(), field_id)
                        .and_then(|state| state.cursor.char_range());
                    let TextEditOutput {
                        response,
                        mut state,
                        ..
                    } = egui::TextEdit::singleline(&mut self.path)
                        .id(field_id)
                        .hint_text("drop a folder, browse, or type a path…")
                        .desired_width(ui.available_width())
                        .show(ui);

                    if !running {
                        if response.secondary_clicked() {
                            if let Some(range) = previous {
                                state.cursor.set_char_range(Some(range));
                            }
                            state.clone().store(ui.ctx(), field_id);
                            // Restore focus so the selection survives while the
                            // menu is open.
                            response.request_focus();
                            self.clipboard.snapshot();
                        }

                        let selection = range_of(&state);
                        let menu =
                            text_ops::menu_state(&self.path, selection, self.clipboard.text());
                        let mut action = None;
                        response.context_menu(|ui| action = path_menu(ui, &menu));

                        if let Some(action) = action {
                            let range = range_of(&state);
                            match action {
                                PathMenu::Cut => {
                                    if let Some((taken, caret)) =
                                        text_ops::cut(&mut self.path, range)
                                    {
                                        ui.ctx().copy_text(taken);
                                        set_range(&mut state, caret);
                                    }
                                }
                                PathMenu::Copy => {
                                    let taken = text_ops::selected(&self.path, range);
                                    if !taken.is_empty() {
                                        ui.ctx().copy_text(taken.to_owned());
                                    }
                                }
                                PathMenu::Paste => {
                                    // Read again at action time so the paste
                                    // uses the current clipboard, not the
                                    // snapshot taken when the menu opened.
                                    if let Some(clip) = self.clipboard.read()
                                        && let Some(caret) =
                                            text_ops::paste(&mut self.path, range, &clip)
                                    {
                                        set_range(&mut state, caret);
                                    }
                                }
                                PathMenu::SelectAll => {
                                    set_range(&mut state, text_ops::select_all(&self.path));
                                }
                            }
                            state.clone().store(ui.ctx(), field_id);
                            // Focus returns to the field so typing acts on the
                            // resulting selection/caret.
                            response.request_focus();
                        }
                    }

                    if response.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                        self.rescan();
                    }
                });
            });
        });
        if let Some(error) = &self.plan_error {
            ui.colored_label(ui.visuals().error_fg_color, format!("{} {error}", icons::X));
        }
    }

    fn selection_bar(&mut self, ui: &mut egui::Ui) {
        let Some(tree) = &self.tree else {
            return;
        };
        let running = self.is_running();
        let targets = self.targets;
        let selected = &mut self.selected;
        ui.add_enabled_ui(!running, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button(format!("{} Select missing", icons::MAGIC_WAND))
                    .clicked()
                {
                    *selected = tree.select_missing(targets);
                }
                if ui
                    .button(format!("{} Select all", icons::SELECTION_ALL))
                    .clicked()
                {
                    *selected = tree.all_wavs().into_iter().collect();
                }
                ui.weak(format!("{} of {} selected", selected.len(), tree.total()));
                if !tree.warnings.is_empty() {
                    let warnings = tree
                        .warnings
                        .iter()
                        .map(|(wav, warning)| format!("{}: {warning}", wav.display()))
                        .collect::<Vec<_>>()
                        .join("\n");
                    ui.colored_label(ui.visuals().warn_fg_color, icons::WARNING)
                        .on_hover_text(warnings);
                }
            });
        });
    }

    fn drop_card(&mut self, ui: &mut egui::Ui) {
        let width = ui.available_width();
        let height = ui.available_height().max(160.0) - 4.0;
        let (rect, response) = ui.allocate_exact_size(Vec2::new(width, height), Sense::click());
        let hover = response.hovered();
        let visuals = ui.visuals();
        let fill = if hover {
            visuals.widgets.hovered.bg_fill
        } else {
            visuals.faint_bg_color
        };
        let stroke = if hover {
            visuals.widgets.hovered.bg_stroke
        } else {
            visuals.widgets.noninteractive.bg_stroke
        };
        ui.painter()
            .rect_filled(rect, visuals.widgets.noninteractive.corner_radius, fill);
        dashed_border(ui.painter(), rect.shrink(1.0), stroke, 9.0, 7.0);
        ui.painter().text(
            rect.center() - Vec2::new(0.0, 44.0),
            Align2::CENTER_CENTER,
            icons::DOWNLOAD_SIMPLE,
            FontId::proportional(40.0),
            visuals.strong_text_color(),
        );
        ui.painter().text(
            rect.center() - Vec2::new(0.0, 2.0),
            Align2::CENTER_CENTER,
            "Drop a voicebank folder or .wav files",
            FontId::proportional(18.0),
            visuals.strong_text_color(),
        );
        ui.painter().text(
            rect.center() + Vec2::new(0.0, 24.0),
            Align2::CENTER_CENTER,
            "or click to browse",
            FontId::proportional(13.0),
            visuals.weak_text_color(),
        );
        if response.clicked() {
            self.browse();
        }
    }

    fn tree_card(&mut self, ui: &mut egui::Ui) {
        let Some(tree) = &self.tree else {
            return;
        };
        let counts = tree.selected_counts(&self.selected);
        let mut ctx = TreeCtx {
            selected: &mut self.selected,
            expanded: &mut self.expanded,
            targets: self.targets,
            counts: &counts,
        };
        egui::Frame::group(ui.style()).show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("tree")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for entry in &tree.root_files {
                        tree_file(ui, entry, &mut ctx, 0);
                    }
                    for dir in &tree.dirs {
                        tree_dir(ui, dir, &mut ctx, 0);
                    }
                });
        });
    }

    fn run_view(&mut self, ui: &mut egui::Ui) {
        let Some(session) = &self.run else {
            return;
        };
        let root = self
            .tree
            .as_ref()
            .map(|tree| tree.root.clone())
            .unwrap_or_default();
        let state = lock(&session.state);
        let mut back = false;
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.add(egui::ProgressBar::new(state.fraction()).text(format!(
                "{}/{}",
                state.resolved(),
                state.total()
            )));
            ui.add_space(4.0);
            let ended = !state.is_running();
            let reserve = if ended { 44.0 } else { 0.0 };
            let time = ui.input(|input| input.time);
            egui::ScrollArea::vertical()
                .id_salt("run_log")
                .max_height((ui.available_height() - reserve).max(60.0))
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for row in state.rows() {
                        run_row(ui, row, &root, time);
                    }
                });
            if ended {
                ui.separator();
                ui.horizontal(|ui| {
                    if let Some(summary) = state.summary() {
                        let head = if summary.cancelled {
                            "Cancelled"
                        } else {
                            "Finished"
                        };
                        let mut parts = vec![
                            format!("{} written", summary.written),
                            format!("{} failed", state.failed_rows()),
                        ];
                        if !summary.warnings.is_empty() {
                            parts.push(format!("{} warnings", summary.warnings.len()));
                        }
                        ui.label(format!("{head} — {}", parts.join(" · ")));
                    } else if let Some(error) = state.error() {
                        ui.colored_label(
                            ui.visuals().error_fg_color,
                            format!("Run failed: {error}"),
                        );
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui
                            .button(format!("{} Back to files", icons::ARROW_LEFT))
                            .clicked()
                        {
                            back = true;
                        }
                    });
                });
            }
        });
        drop(state);
        if back {
            self.run = None;
        }
    }

    fn overlay(&self, ctx: &egui::Context) {
        let window = ctx.content_rect();
        let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("drop_overlay"));
        let painter = ctx.layer_painter(layer);
        painter.rect_filled(window, 0.0, Color32::from_black_alpha(140));
        let r = window.shrink(18.0);
        dashed_border(&painter, r, Stroke::new(2.5, Color32::WHITE), 14.0, 10.0);
        painter.text(
            r.center() - Vec2::new(0.0, 50.0),
            Align2::CENTER_CENTER,
            icons::DOWNLOAD_SIMPLE,
            FontId::proportional(44.0),
            Color32::WHITE,
        );
        painter.text(
            r.center(),
            Align2::CENTER_CENTER,
            "Drop to use as voicebank folder",
            FontId::proportional(22.0),
            Color32::WHITE,
        );
        painter.text(
            r.center() + Vec2::new(0.0, 26.0),
            Align2::CENTER_CENTER,
            "folders and .wav files both work",
            FontId::proportional(14.0),
            Color32::from_gray(195),
        );
    }
}

impl eframe::App for KiraFrqGenApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let running = self.is_running();

        if !running {
            let dropped = ui.input(|input| {
                input
                    .raw
                    .dropped_files
                    .first()
                    .map(|file| file.path().to_path_buf())
            });
            if let Some(path) = dropped {
                self.set_path(path);
            }
        }
        let hovering = !running && ui.input(|input| !input.raw.hovered_files.is_empty());

        egui::Panel::left("sidebar")
            .default_size(260.0)
            .resizable(false)
            .show(ui, |ui| {
                self.sidebar(ui);
            });

        let central_frame =
            egui::Frame::central_panel(ui.style()).inner_margin(egui::Margin::same(12));
        egui::CentralPanel::default()
            .frame(central_frame)
            .show(ui, |ui| {
                self.path_bar(ui);
                ui.add_space(8.0);

                if self.tree.is_some() {
                    self.selection_bar(ui);
                    ui.add_space(4.0);
                    if self.run.is_some() {
                        self.run_view(ui);
                    } else {
                        self.tree_card(ui);
                    }
                } else {
                    self.drop_card(ui);
                }
            });

        if hovering {
            self.overlay(ui.ctx());
        }
        self.credits_window(ui.ctx());
        if self.is_running() {
            ui.ctx().request_repaint();
        }
    }
}

// --------------------------------------------------------------------- tree

/// What every tree row needs besides its node: the shared selection, the
/// expand state, the checked formats and the per-folder selected counts.
struct TreeCtx<'a> {
    selected: &'a mut BTreeSet<PathBuf>,
    expanded: &'a mut HashSet<PathBuf>,
    targets: Targets,
    counts: &'a HashMap<PathBuf, usize>,
}

fn collect_dir_paths(dir: &DirNode, expanded: &mut HashSet<PathBuf>) {
    expanded.insert(dir.path.clone());
    for child in &dir.dirs {
        collect_dir_paths(child, expanded);
    }
}

fn tree_dir(ui: &mut egui::Ui, dir: &DirNode, ctx: &mut TreeCtx, depth: usize) {
    let is_open = ctx.expanded.contains(&dir.path);
    ui.horizontal(|ui| {
        ui.add_space(depth as f32 * 18.0);
        let caret = if is_open {
            icons::CARET_DOWN
        } else {
            icons::CARET_RIGHT
        };
        if ui.add(egui::Button::new(caret).frame(false)).clicked() {
            toggle(ctx.expanded, &dir.path);
        }
        let count = ctx.counts.get(&dir.path).copied().unwrap_or(0);
        let state = if count == 0 {
            Tri::Off
        } else if count == dir.total {
            Tri::On
        } else {
            Tri::Mixed
        };
        if tri_checkbox(ui, state).clicked() {
            let want = state != Tri::On;
            for path in dir.entries().map(|entry| &entry.path) {
                if want {
                    ctx.selected.insert(path.clone());
                } else {
                    ctx.selected.remove(path);
                }
            }
        }
        let folder_icon = if is_open {
            icons::FOLDER_OPEN
        } else {
            icons::FOLDER
        };
        if ui
            .add(
                egui::Button::new(format!("{folder_icon} {}", dir.name))
                    .frame(false)
                    .selected(is_open),
            )
            .clicked()
        {
            toggle(ctx.expanded, &dir.path);
        }
        ui.weak(format!("{count}/{}", dir.total));
    });
    if is_open {
        for child in &dir.dirs {
            tree_dir(ui, child, ctx, depth + 1);
        }
        for entry in &dir.files {
            tree_file(ui, entry, ctx, depth + 1);
        }
    }
}

fn tree_file(ui: &mut egui::Ui, entry: &WavEntry, ctx: &mut TreeCtx, depth: usize) {
    ui.horizontal(|ui| {
        ui.add_space(depth as f32 * 18.0 + 26.0);
        let mut checked = ctx.selected.contains(&entry.path);
        if ui.checkbox(&mut checked, "").changed() {
            if checked {
                ctx.selected.insert(entry.path.clone());
            } else {
                ctx.selected.remove(&entry.path);
            }
        }
        let name = entry
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        ui.label(format!("{} {name}", icons::FILE_AUDIO));
        ui.weak(missing_label(entry, ctx.targets));
    });
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tri {
    Off,
    Mixed,
    On,
}

fn tri_checkbox(ui: &mut egui::Ui, state: Tri) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(18.0), Sense::click());
    let visuals = ui.style().interact(&response);
    let fill = match state {
        Tri::On | Tri::Mixed => ui.visuals().selection.bg_fill,
        Tri::Off => visuals.weak_bg_fill,
    };
    let corner = visuals.corner_radius;
    ui.painter().rect_filled(rect, corner, fill);
    ui.painter()
        .rect_stroke(rect, corner, visuals.bg_stroke, StrokeKind::Inside);
    let glyph = match state {
        Tri::On => Some(icons::CHECK),
        Tri::Mixed => Some(icons::MINUS),
        Tri::Off => None,
    };
    if let Some(glyph) = glyph {
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            glyph,
            FontId::proportional(12.0),
            ui.visuals().strong_text_color(),
        );
    }
    response
}

fn toggle(set: &mut HashSet<PathBuf>, path: &Path) {
    if !set.remove(path) {
        set.insert(path.to_path_buf());
    }
}

// --------------------------------------------------------------------- rows

fn run_row(ui: &mut egui::Ui, row: &Row, root: &Path, time: f64) {
    let relative = row.wav.strip_prefix(root).unwrap_or(&row.wav);
    let shown = if relative.as_os_str().is_empty() {
        // A single-wav root: the wav *is* the root, so there is nothing to strip.
        row.wav.display().to_string()
    } else {
        relative.display().to_string()
    };
    let (color, text) = match &row.status {
        Status::Pending => (ui.visuals().weak_text_color(), "pending".to_owned()),
        Status::Running { .. } => (ui.visuals().text_color(), "working…".to_owned()),
        Status::Written(targets) => (
            ui.visuals().text_color(),
            format!("wrote {}", target_names(targets)),
        ),
        Status::Failed(reasons) => (
            ui.visuals().error_fg_color,
            format!("failed — {}", reasons.join("; ")),
        ),
        Status::Skipped { empty, no_entry } => (
            ui.visuals().weak_text_color(),
            skipped_text(*empty, no_entry),
        ),
        Status::Cancelled => (ui.visuals().weak_text_color(), "cancelled".to_owned()),
    };
    let response = ui
        .horizontal(|ui| {
            // One fixed icon slot for every state, so rows never shift.
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(16.0), Sense::hover());
            match &row.status {
                Status::Pending => {
                    progress_spinner(ui, rect, time);
                }
                Status::Running { permille } => match permille {
                    // No frame event yet: spin like a pending row. Once the
                    // fraction lands the wedge takes over (#33/#34).
                    None => progress_spinner(ui, rect, time),
                    Some(permille) => progress_pie(
                        ui,
                        rect,
                        PIE_TOP,
                        *permille as f32 / 1000.0 * std::f32::consts::TAU,
                    ),
                },
                Status::Written(_) => {
                    status_glyph(ui, rect, icons::CHECK, ui.visuals().text_color())
                }
                Status::Failed(_) => status_glyph(ui, rect, icons::X, ui.visuals().error_fg_color),
                Status::Skipped { .. } => status_glyph(
                    ui,
                    rect,
                    icons::CIRCLE_NOTCH,
                    ui.visuals().weak_text_color(),
                ),
                Status::Cancelled => {
                    status_glyph(ui, rect, icons::MINUS, ui.visuals().weak_text_color())
                }
            }
            ui.monospace(shown);
            ui.colored_label(color, text);
            if !row.warnings.is_empty() {
                ui.colored_label(ui.visuals().warn_fg_color, icons::WARNING);
            }
        })
        .response;

    let mut hover = Vec::new();
    if let Status::Failed(reasons) = &row.status {
        hover.extend(reasons.iter().cloned());
    }
    hover.extend(
        row.warnings
            .iter()
            .map(|warning| format!("warning: {warning}")),
    );
    if !hover.is_empty() {
        response.on_hover_text(hover.join("\n"));
    }
}

fn target_names(targets: &BTreeSet<Target>) -> String {
    targets
        .iter()
        .copied()
        .map(target_name)
        .collect::<Vec<_>>()
        .join(", ")
}

fn skipped_text(empty: bool, no_entry: &BTreeSet<Target>) -> String {
    if empty {
        "skipped — empty wav".to_owned()
    } else if !no_entry.is_empty() {
        format!("skipped — too short for a {} entry", target_names(no_entry))
    } else {
        "skipped".to_owned()
    }
}

fn status_glyph(ui: &egui::Ui, rect: egui::Rect, glyph: &str, color: Color32) {
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        glyph,
        FontId::new(14.0, egui::FontFamily::Name("phosphor".into())),
        color,
    );
}

fn dashed_border(painter: &egui::Painter, rect: egui::Rect, stroke: Stroke, dash: f32, gap: f32) {
    let path = [
        rect.left_top(),
        rect.right_top(),
        rect.right_bottom(),
        rect.left_bottom(),
        rect.left_top(),
    ];
    for shape in Shape::dashed_line(&path, stroke, dash, gap) {
        painter.add(shape);
    }
}

/// The indeterminate spinner: faint filled disc plus the accent arc rotating
/// with the clock. No ring anywhere (#33).
fn progress_spinner(ui: &egui::Ui, rect: egui::Rect, time: f64) {
    let (center, radius) = progress_disc(ui, rect);
    let arc = ui.visuals().selection.bg_fill;
    let painter = ui.painter();
    let start = (time * 2.4).rem_euclid(std::f64::consts::TAU) as f32;
    let sweep = 1.7_f32;
    let segments = 10;
    let points: Vec<egui::Pos2> = (0..=segments)
        .map(|index| {
            let angle = start + sweep * (index as f32 / segments as f32);
            center + Vec2::new(angle.cos(), angle.sin()) * (radius - 1.5)
        })
        .collect();
    painter.add(Shape::line(points, Stroke::new(1.8, arc)));
}

// ------------------------------------------------------------ pie (#33/#34)
//
// Strokeless determinate indicator: the disc above plus a filled accent wedge
// whose sweep is the wav's composed frame fraction. Rows with no fraction
// keep the spinner above, so no wedge is ever drawn without progress.

/// 12 o'clock on screen (y down): angle `-PI/2`. Larger angles run clockwise,
/// so a wedge from `PIE_TOP` with a positive sweep grows clockwise from the top.
const PIE_TOP: f32 = -std::f32::consts::FRAC_PI_2;

/// The faint filled disc behind the wedge.
fn progress_disc(ui: &egui::Ui, rect: egui::Rect) -> (egui::Pos2, f32) {
    let center = rect.center();
    let radius = rect.width() * 0.5;
    let track = ui.visuals().weak_text_color().gamma_multiply(0.15);
    ui.painter().circle_filled(center, radius, track);
    (center, radius)
}

/// Triangle fan for a wedge sweeping `sweep` radians clockwise from `start` on
/// screen (y down): the centre followed by one rim point per segment step.
fn wedge_points(
    center: egui::Pos2,
    radius: f32,
    start: f32,
    sweep: f32,
    segments: u32,
) -> Vec<egui::Pos2> {
    let mut points = vec![center];
    if sweep <= 0.0 {
        return points;
    }
    let segments = segments.max(1);
    for index in 0..=segments {
        let angle = start + sweep * (index as f32 / segments as f32);
        points.push(center + Vec2::new(angle.cos(), angle.sin()) * radius);
    }
    points
}

/// Strokeless pie: faint filled disc plus a filled accent wedge sweeping
/// clockwise from 12 o'clock. A zero sweep leaves the disc alone.
fn progress_pie(ui: &egui::Ui, rect: egui::Rect, start: f32, sweep: f32) {
    let (center, radius) = progress_disc(ui, rect);
    if sweep <= 0.0 {
        return;
    }
    let wedge = ui.visuals().selection.bg_fill;
    // A filled closed path, not a mesh and not a convex polygon: the
    // tessellator triangulates concave sweeps correctly *and* feathers the
    // edge, while a raw mesh reaches the screen aliased.
    let points = wedge_points(center, radius, start, sweep, 24);
    ui.painter().add(Shape::Path(egui::epaint::PathShape {
        points,
        closed: true,
        fill: wedge,
        stroke: Default::default(),
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wedge_starts_at_twelve_oclock() {
        let center = egui::Pos2::new(100.0, 100.0);
        let points = wedge_points(center, 7.0, PIE_TOP, 1.0, 8);
        let first_rim = points[1];
        assert!((first_rim.x - center.x).abs() < 1e-4);
        assert!((first_rim.y - (center.y - 7.0)).abs() < 1e-4);
    }

    #[test]
    fn a_wedge_grows_clockwise_from_the_top() {
        let center = egui::Pos2::new(100.0, 100.0);
        let points = wedge_points(center, 7.0, PIE_TOP, 1.0, 8);
        // The middle of a small clockwise sweep from the top sits north-east.
        let mid = points[points.len() / 2];
        assert!(mid.x > center.x);
        assert!(mid.y < center.y);
    }

    #[test]
    fn a_zero_sweep_is_just_the_centre() {
        let center = egui::Pos2::new(100.0, 100.0);
        assert_eq!(wedge_points(center, 7.0, PIE_TOP, 0.0, 8), vec![center]);
    }

    #[test]
    fn a_wedge_fan_holds_the_centre_plus_one_rim_point_per_segment_step() {
        let center = egui::Pos2::new(100.0, 100.0);
        let points = wedge_points(center, 7.0, PIE_TOP, 1.0, 8);
        assert_eq!(points.len(), 8 + 2);
        assert_eq!(points[0], center);
    }

    #[test]
    fn a_large_sweep_reaches_past_half_the_circle() {
        let center = egui::Pos2::new(100.0, 100.0);
        // 270° clockwise from the top ends due west of the centre.
        let sweep = 3.0 * std::f32::consts::FRAC_PI_2;
        let points = wedge_points(center, 7.0, PIE_TOP, sweep, 24);
        let last_rim = points[points.len() - 1];
        assert!(last_rim.x < center.x);
        assert!((last_rim.y - center.y).abs() < 1e-4);
    }

    #[test]
    fn the_default_estimator_follows_the_ml_capability() {
        assert_eq!(kirafrqgen_core::default_estimator(true), Estimator::Rmvpe);
        assert_eq!(kirafrqgen_core::default_estimator(false), Estimator::Dio);
    }

    #[test]
    fn a_selection_survives_a_round_trip_through_the_field_state() {
        let mut state = TextEditState::default();
        set_range(&mut state, CharRange::new(2, 5));
        assert_eq!(range_of(&state), CharRange::new(2, 5));
    }

    #[test]
    fn a_caret_survives_a_round_trip_through_the_field_state() {
        let mut state = TextEditState::default();
        set_range(&mut state, CharRange::caret(3));
        assert_eq!(range_of(&state), CharRange::caret(3));
    }

    #[test]
    fn a_field_with_no_cursor_reads_as_a_caret_at_the_start() {
        assert_eq!(range_of(&TextEditState::default()), CharRange::caret(0));
    }
}
