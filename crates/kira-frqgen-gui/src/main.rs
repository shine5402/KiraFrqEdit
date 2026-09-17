//! `KiraFrqGen` binary scaffold. The GUI shell is #13's prototype-driven
//! ticket; this binary only proves the crate wiring builds and opens a window.

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "KiraFrqGen",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(KiraFrqGenApp))),
    )
}

struct KiraFrqGenApp;

impl eframe::App for KiraFrqGenApp {
    fn ui(&mut self, ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {
        eframe::egui::CentralPanel::default().show(ui, |ui| {
            ui.label("KiraFrqGen scaffold — the GUI shell lands with #13.");
        });
    }
}
