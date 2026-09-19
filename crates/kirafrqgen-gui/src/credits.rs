//! Draw the parsed credits document with egui styling.
//!
//! [`kirafrq_credits::document`] turns `CREDITS.md` into styled lines; this
//! module is the egui adapter that gives those lines their look: headings in a
//! larger strong font, inline code on a tinted background, links in the
//! hyperlink colour and actually clickable, and fenced blocks as one monospace
//! panel.

use eframe::egui::{self, RichText};
use kirafrq_credits::{Line, Span};

/// Draw every line, top to bottom, into `ui`.
pub fn show(ui: &mut egui::Ui, lines: &[Line]) {
    let mut index = 0;
    while index < lines.len() {
        match &lines[index] {
            Line::Blank => {
                ui.add_space(6.0);
                index += 1;
            }
            Line::Code { .. } => {
                let (block, next) = collect_code_block(lines, index);
                code_block(ui, &block);
                index = next;
            }
            line => {
                show_line(ui, line);
                index += 1;
            }
        }
    }
}

/// Join the run of [`Line::Code`] starting at `start` into one block, returning
/// it and the index just past the run.
fn collect_code_block(lines: &[Line], start: usize) -> (String, usize) {
    let mut block = String::new();
    let mut index = start;
    while let Some(Line::Code { text }) = lines.get(index) {
        if !block.is_empty() {
            block.push('\n');
        }
        block.push_str(text);
        index += 1;
    }
    (block, index)
}

fn show_line(ui: &mut egui::Ui, line: &Line) {
    match line {
        Line::Blank | Line::Code { .. } => {}
        Line::Heading { level, spans } => {
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                show_spans(ui, spans, Some(heading_size(*level)));
            });
        }
        Line::Bullet { indent, spans } => {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                if *indent > 0 {
                    ui.add_space(*indent as f32 * 6.0);
                }
                ui.label("- ");
                show_spans(ui, spans, None);
            });
        }
        Line::Text { spans } => {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                show_spans(ui, spans, None);
            });
        }
    }
}

/// Draw the inline spans of one line. Zero item spacing keeps the words of a
/// line glued together; the wrapping happens inside the labels.
fn show_spans(ui: &mut egui::Ui, spans: &[Span], size: Option<f32>) {
    for span in spans {
        let mut text = RichText::new(&span.text);
        if let Some(size) = size {
            text = text.size(size).strong();
        }
        if span.bold {
            text = text.strong();
        }
        if span.italic {
            text = text.italics();
        }
        if span.code {
            text = text.code();
        }
        if let Some(url) = &span.link {
            ui.hyperlink_to(text, url);
        } else {
            ui.label(text);
        }
    }
}

fn heading_size(level: usize) -> f32 {
    match level {
        1 => 20.0,
        2 => 16.0,
        _ => 14.0,
    }
}

/// One fenced block, drawn as a single monospace panel so it reads as code
/// rather than as one tinted line per row.
fn code_block(ui: &mut egui::Ui, block: &str) {
    egui::Frame::default()
        .fill(ui.visuals().code_bg_color)
        .inner_margin(6.0)
        .show(ui, |ui| {
            ui.add(egui::Label::new(RichText::new(block).monospace()).wrap());
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_code_run_joins_into_one_block() {
        let lines = vec![
            Line::Text { spans: vec![] },
            Line::Code { text: "a".into() },
            Line::Code { text: "b".into() },
            Line::Blank,
        ];
        let (block, next) = collect_code_block(&lines, 1);
        assert_eq!(block, "a\nb");
        assert_eq!(next, 3);
    }

    #[test]
    fn a_single_code_line_still_collects_cleanly() {
        let lines = vec![Line::Code {
            text: "only".into(),
        }];
        assert_eq!(collect_code_block(&lines, 0), ("only".to_owned(), 1));
    }

    #[test]
    fn headings_shrink_with_depth() {
        assert!(heading_size(1) > heading_size(2));
        assert!(heading_size(2) > heading_size(3));
        assert_eq!(heading_size(3), heading_size(6));
    }
}
