//! Render the real `CREDITS.md` through egui and check that the styling
//! reaches the one cached galley: headings grow, inline code gets a background,
//! links paint in the hyperlink colour, and code blocks get a panel.

use std::sync::Arc;

use eframe::egui;
use kirafrq_credits::{Line, Span, document};

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

/// What one frame of the credits window produced.
struct Frame {
    output: egui::FullOutput,
    hyperlink_color: egui::Color32,
    code_bg: egui::Color32,
}

fn frame(markdown: &str) -> Frame {
    let ctx = egui::Context::default();
    let mut view = credits::CreditsView::new(markdown);
    let mut hyperlink_color = None;
    let mut code_bg = None;
    let mut output = ctx.run_ui(raw_input(), |ui| {
        hyperlink_color = Some(ui.visuals().hyperlink_color);
        code_bg = Some(ui.visuals().code_bg_color);
        view.ui(ui);
    });
    output.textures_delta.clear();
    Frame {
        output,
        hyperlink_color: hyperlink_color.expect("the ui ran"),
        code_bg: code_bg.expect("the ui ran"),
    }
}

/// Every text shape drawn this frame. The credits are one galley, so this is
/// normally (and deliberately) a single shape.
fn text_shapes(frame: &Frame) -> Vec<egui::Shape> {
    let mut out = Vec::new();
    for clipped in &frame.output.shapes {
        collect_text(&clipped.shape, &mut out);
    }
    out
}

fn collect_text(shape: &egui::Shape, out: &mut Vec<egui::Shape>) {
    match shape {
        egui::Shape::Text(_) => out.push(shape.clone()),
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_text(shape, out);
            }
        }
        _ => {}
    }
}

fn galleys(frame: &Frame) -> Vec<Arc<egui::Galley>> {
    text_shapes(frame)
        .iter()
        .filter_map(|shape| match shape {
            egui::Shape::Text(text) => Some(text.galley.clone()),
            _ => None,
        })
        .collect()
}

/// The single galley the credits window draws.
fn galley(frame: &Frame) -> Arc<egui::Galley> {
    let galleys = galleys(frame);
    assert_eq!(galleys.len(), 1, "the credits are one galley");
    galleys.into_iter().next().unwrap()
}

/// `(text, format)` for every section of the galley.
fn sections(frame: &Frame) -> Vec<(String, egui::TextFormat)> {
    let galley = galley(frame);
    let text = &galley.job.text;
    galley
        .job
        .sections
        .iter()
        .map(|section| {
            let range = section.byte_range.start.0..section.byte_range.end.0;
            (text[range].to_owned(), section.format.clone())
        })
        .collect()
}

fn section_containing(frame: &Frame, needle: &str) -> egui::TextFormat {
    sections(frame)
        .into_iter()
        .find(|(text, _)| text.contains(needle))
        .unwrap_or_else(|| panic!("no section contains {needle:?}"))
        .1
}

#[test]
fn the_real_document_lays_out_as_one_galley() {
    let frame = frame(kirafrq_credits::for_display(true));
    let galley = galley(&frame);
    assert!(
        galley.rows.len() > 1000,
        "the full document is large: {} rows",
        galley.rows.len()
    );
}

#[test]
fn a_heading_uses_a_larger_font_than_body_text() {
    let heading = section_containing(&frame("# Title\n"), "Title");
    let body = section_containing(&frame("Title\n"), "Title");
    assert!(
        heading.font_id.size > body.font_id.size,
        "heading {} vs body {}",
        heading.font_id.size,
        body.font_id.size
    );
}

#[test]
fn a_deeper_heading_uses_a_smaller_font() {
    let h1 = section_containing(&frame("# Title\n"), "Title")
        .font_id
        .size;
    let h2 = section_containing(&frame("## Title\n"), "Title")
        .font_id
        .size;
    assert!(h1 > h2, "h1 {h1} vs h2 {h2}");
}

#[test]
fn an_inline_code_span_paints_a_background() {
    let code = section_containing(&frame("run `kirafrqgen-cli` now\n"), "kirafrqgen-cli");
    assert_eq!(code.font_id.family, egui::FontFamily::Monospace);
    assert_ne!(code.background, egui::Color32::TRANSPARENT);
}

#[test]
fn a_link_paints_in_the_hyperlink_colour() {
    let frame = frame("see [WORLD](https://example.test/w)\n");
    let link = section_containing(&frame, "WORLD");
    assert_eq!(link.color, frame.hyperlink_color);
    assert!(link.underline.width > 0.0, "the link is underlined");
}

#[test]
fn a_fenced_block_is_monospace_and_paints_a_panel() {
    let frame = frame("````text\nline one\nline two\n````\n");
    let code = section_containing(&frame, "line one");
    assert_eq!(code.font_id.family, egui::FontFamily::Monospace);

    let painted = frame.output.shapes.iter().any(
        |clipped| matches!(&clipped.shape, egui::Shape::Rect(rect) if rect.fill == frame.code_bg),
    );
    assert!(painted, "a code panel in the code colour is painted");
}

#[test]
fn a_soft_wrapped_paragraph_is_joined_into_one_line() {
    let frame = frame("KiraFrqGen stands on other people's work.\nOur heartfelt thanks.\n");
    let text = galley(&frame).job.text.clone();
    assert!(
        text.contains("work. Our heartfelt thanks."),
        "the source's hard wrap is joined with a space: {text:?}"
    );
}

#[test]
fn a_link_span_keeps_its_url_for_hit_testing() {
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
