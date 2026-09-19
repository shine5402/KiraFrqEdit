//! Cost measurements for the Credits window. Run with:
//!   cargo test -p kirafrqgen-gui --test credits_cost -- --ignored --nocapture
//! Add `--release` for the numbers that matter in a shipped build.

use std::time::Instant;

use eframe::egui;

/// The GUI adapter, compiled into the test binary via `#[path]`.
#[path = "../src/credits.rs"]
mod credits;

/// A realistic window, so the scroll area has a real viewport.
fn raw_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(640.0, 480.0),
        )),
        ..Default::default()
    }
}

/// One cold frame (which lays the document out) and then steady-state frames.
#[test]
#[ignore = "measurement, not an assertion"]
fn report_frame_cost() {
    let ctx = egui::Context::default();
    let mut view = credits::CreditsView::new(kirafrq_credits::for_display(true));

    let start = Instant::now();
    let mut output = ctx.run_ui(raw_input(), |ui| view.ui(ui));
    println!("cold frame (parse + layout): {:?}", start.elapsed());
    output.textures_delta.clear();

    for _ in 0..5 {
        let start = Instant::now();
        let mut output = ctx.run_ui(raw_input(), |ui| view.ui(ui));
        let layout = start.elapsed();
        let start = Instant::now();
        let _ = ctx.tessellate(output.shapes.clone(), output.pixels_per_point);
        println!(
            "warm frame: layout {layout:?}  tessellate {:?}",
            start.elapsed()
        );
        output.textures_delta.clear();
    }
}
