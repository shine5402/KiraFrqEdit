//! Render the real `CREDITS.md` through egui and check that the styling
//! reaches the galley: headings grow, inline code gets a background, and links
//! paint in the hyperlink colour.

use eframe::egui;
use kirafrq_credits::{Line, Span, document};

/// The GUI adapter, compiled into the test binary via `#[path]`.
#[path = "../src/credits.rs"]
mod credits;

/// Lay `lines` out in a fresh context and return every text galley drawn.
fn galleys(lines: &[Line]) -> Vec<std::sync::Arc<egui::Galley>> {
    let ctx = egui::Context::default();
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
        credits::show(ui, lines);
    });
    output.textures_delta.clear();
    let mut galleys = Vec::new();
    for clipped in &output.shapes {
        collect_galleys(&clipped.shape, &mut galleys);
    }
    galleys
}

fn collect_galleys(shape: &egui::Shape, out: &mut Vec<std::sync::Arc<egui::Galley>>) {
    match shape {
        egui::Shape::Text(text) => out.push(text.galley.clone()),
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_galleys(shape, out);
            }
        }
        _ => {}
    }
}

/// Every section format across the drawn galleys.
fn formats(lines: &[Line]) -> Vec<egui::TextFormat> {
    galleys(lines)
        .iter()
        .flat_map(|galley| galley.job.sections.iter().map(|s| s.format.clone()))
        .collect()
}

/// Lay `lines` out and return the hyperlink colour the ui used.
fn hyperlink_color(lines: &[Line]) -> egui::Color32 {
    let ctx = egui::Context::default();
    let mut color = None;
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
        color = Some(ui.visuals().hyperlink_color);
        credits::show(ui, lines);
    });
    output.textures_delta.clear();
    color.expect("the ui ran")
}

/// Every text shape's fallback colour: egui's hyperlink widget leaves the
/// galley colour as a placeholder and passes the link colour here.
fn fallback_colors(lines: &[Line]) -> Vec<egui::Color32> {
    let ctx = egui::Context::default();
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
        credits::show(ui, lines);
    });
    output.textures_delta.clear();
    let mut colors = Vec::new();
    for clipped in &output.shapes {
        collect_fallbacks(&clipped.shape, &mut colors);
    }
    colors
}

fn collect_fallbacks(shape: &egui::Shape, out: &mut Vec<egui::Color32>) {
    match shape {
        egui::Shape::Text(text) => out.push(text.fallback_color),
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_fallbacks(shape, out);
            }
        }
        _ => {}
    }
}

/// The concatenated text of every drawn galley.
fn drawn_text(lines: &[Line]) -> String {
    galleys(lines)
        .iter()
        .map(|galley| galley.text().to_owned())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_real_document_lays_out_without_panicking() {
    let lines = document(kirafrq_credits::for_display(true));
    assert!(lines.len() > 1000, "the full document is large");
    assert!(!galleys(&lines).is_empty());
}

#[test]
fn a_heading_uses_a_larger_font_than_body_text() {
    let heading = formats(&document("# Title\n"));
    let body = formats(&document("Title\n"));
    let heading_size = heading[0].font_id.size;
    let body_size = body[0].font_id.size;
    assert!(
        heading_size > body_size,
        "heading {heading_size} vs body {body_size}"
    );
}

#[test]
fn a_deeper_heading_uses_a_smaller_font() {
    let h1 = formats(&document("# Title\n"))[0].font_id.size;
    let h2 = formats(&document("## Title\n"))[0].font_id.size;
    assert!(h1 > h2, "h1 {h1} vs h2 {h2}");
}

#[test]
fn an_inline_code_span_paints_a_background() {
    let formats = formats(&document("run `kirafrqgen-cli` now\n"));
    let code = formats
        .iter()
        .find(|format| format.font_id.family == egui::FontFamily::Monospace)
        .expect("the code span is monospace");
    assert_ne!(
        code.background,
        egui::Color32::TRANSPARENT,
        "inline code should paint a background"
    );
}

#[test]
fn a_link_paints_in_the_hyperlink_colour() {
    let lines = document("see [WORLD](https://example.test/w)\n");
    let hyperlink = hyperlink_color(&lines);
    assert!(
        fallback_colors(&lines).contains(&hyperlink),
        "the link should paint in the hyperlink colour"
    );
}

#[test]
fn a_code_block_is_drawn_as_one_monospace_panel() {
    let lines = vec![
        Line::Code {
            text: "line one".into(),
        },
        Line::Code {
            text: "line two".into(),
        },
    ];
    let formats = formats(&lines);
    assert!(
        formats
            .iter()
            .all(|format| format.font_id.family == egui::FontFamily::Monospace),
        "the block is monospace"
    );
    assert_eq!(
        drawn_text(&lines),
        "line one\nline two",
        "the run joins into one panel"
    );
}

#[test]
fn a_link_span_keeps_its_url_for_the_widget() {
    let Line::Text { spans } = &document("see [WORLD](https://example.test/w)\n")[0] else {
        panic!("expected a text line");
    };
    assert_eq!(spans[1].link.as_deref(), Some("https://example.test/w"));
}

#[test]
fn a_plain_span_has_no_styling_flags() {
    assert_eq!(
        spans("x"),
        vec![Span {
            text: "x".into(),
            code: false,
            bold: false,
            italic: false,
            link: None,
        }]
    );
}

fn spans(text: &str) -> Vec<Span> {
    vec![Span {
        text: text.into(),
        code: false,
        bold: false,
        italic: false,
        link: None,
    }]
}
