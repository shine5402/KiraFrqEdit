//! Draw the parsed credits document with egui styling.
//!
//! The full document is thousands of lines, and egui is immediate mode: a tree
//! of thousands of labels would be laid out again on every single frame, which
//! makes scrolling burn CPU. Instead the whole document becomes one [`Galley`]
//! that is laid out once and cached; epaint culls the rows outside the clip
//! rect when it tessellates, so a frame only pays for what is on screen.
//!
//! Links are not widgets in a single galley, so they are hit-tested against the
//! galley: hovering a link shows the pointing hand and clicking opens the URL.

use std::ops::Range;
use std::sync::Arc;

use eframe::egui::{
    self, Color32, FontId, Galley, Sense, Stroke, TextFormat, TextStyle,
    text::{LayoutJob, TextWrapping},
};
use kirafrq_credits::{Line, Span};

/// Vertical padding above and below a fenced code block's panel.
const CODE_PAD: f32 = 4.0;

/// A link's character range in the laid-out text, with where it points.
struct Link {
    chars: Range<usize>,
    url: String,
}

/// The credits document, laid out lazily and cached across frames.
pub struct CreditsView {
    lines: Vec<Line>,
    cache: Option<Cache>,
}

struct Cache {
    /// The wrap width the galley was laid out for.
    width: f32,
    /// The width seen on the previous frame: a resize is allowed to settle for
    /// a frame before the document is re-wrapped, so dragging the window edge
    /// does not re-lay-out everything on every frame.
    settled_width: f32,
    pixels_per_point: f32,
    dark_mode: bool,
    galley: Arc<Galley>,
    links: Vec<Link>,
    /// The galley rows each fenced code block covers, for painting its panel.
    code_rows: Vec<Range<usize>>,
    code_bg: Color32,
}

impl CreditsView {
    /// Parse `markdown` for later drawing.
    pub fn new(markdown: &str) -> Self {
        Self {
            lines: kirafrq_credits::document(markdown),
            cache: None,
        }
    }

    /// Draw the document in a vertical scroll area.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| self.show(ui));
    }

    fn show(&mut self, ui: &mut egui::Ui) {
        self.ensure_cache(ui);
        let Some(cache) = &self.cache else {
            return;
        };
        let galley = Arc::clone(&cache.galley);

        let (rect, response) = ui.allocate_exact_size(galley.size(), Sense::click_and_drag());

        if ui.is_rect_visible(rect) {
            for rows in &cache.code_rows {
                if rows.is_empty() {
                    continue;
                }
                let top = rect.top() + galley.rows[rows.start].rect().top() - CODE_PAD;
                let bottom = rect.top() + galley.rows[rows.end - 1].rect().bottom() + CODE_PAD;
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(rect.left(), top),
                        egui::pos2(rect.right(), bottom),
                    ),
                    4.0,
                    cache.code_bg,
                );
            }
            egui::text_selection::LabelSelectionState::label_text_selection(
                ui,
                &response,
                rect.left_top(),
                Arc::clone(&galley),
                ui.visuals().text_color(),
                Stroke::NONE,
            );
        }

        if let Some(pointer) = response.hover_pos() {
            let cursor = galley.cursor_from_pos(pointer - rect.left_top());
            let index = cursor.index.0;
            if let Some(link) = cache.links.iter().find(|link| link.chars.contains(&index)) {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                if response.clicked() {
                    ui.ctx().open_url(egui::OpenUrl::new_tab(&link.url));
                }
            }
        }
    }

    /// Lay the document out if the cache is missing, stale, or the window is
    /// no longer the width it was measured at.
    fn ensure_cache(&mut self, ui: &mut egui::Ui) {
        let width = ui.available_width();
        let pixels_per_point = ui.ctx().pixels_per_point();
        let dark_mode = ui.visuals().dark_mode;
        let same_theme = |cache: &Cache| {
            cache.pixels_per_point == pixels_per_point && cache.dark_mode == dark_mode
        };
        let current = |cache: &Cache| (cache.width - width).abs() < 0.5;

        if self
            .cache
            .as_ref()
            .is_some_and(|c| same_theme(c) && current(c))
        {
            return;
        }

        // Re-wrap only once a resize settles, so dragging the window edge does
        // not re-lay-out the whole document on every frame of the drag.
        if let Some(cache) = self.cache.as_mut()
            && same_theme(cache)
            && (cache.settled_width - width).abs() >= 0.5
        {
            cache.settled_width = width;
            ui.ctx().request_repaint();
            return;
        }

        self.rebuild(ui, width, pixels_per_point, dark_mode);
    }

    fn rebuild(&mut self, ui: &egui::Ui, width: f32, pixels_per_point: f32, dark_mode: bool) {
        let (job, links, code_blocks) = document_job(ui.style(), &self.lines, width);
        let galley = ui.ctx().fonts_mut(|fonts| fonts.layout_job(job));
        let code_rows = code_row_ranges(&galley, &code_blocks)
            .into_iter()
            .filter(|rows| !rows.is_empty())
            .collect();
        self.cache = Some(Cache {
            width,
            settled_width: width,
            pixels_per_point,
            dark_mode,
            galley,
            links,
            code_rows,
            code_bg: ui.visuals().code_bg_color,
        });
    }
}

/// Build one [`LayoutJob`] for the whole document, recording where its links
/// and fenced code blocks sit. Both are character ranges into the job text.
fn document_job(
    style: &egui::Style,
    lines: &[Line],
    width: f32,
) -> (LayoutJob, Vec<Link>, Vec<Range<usize>>) {
    let body = TextStyle::Body.resolve(style);
    let mono = TextStyle::Monospace.resolve(style);
    let text_color = style.visuals.text_color();
    let strong_color = style.visuals.strong_text_color();
    let link_color = style.visuals.hyperlink_color;
    let code_bg = style.visuals.code_bg_color;

    let mut builder = JobBuilder {
        job: LayoutJob {
            wrap: TextWrapping {
                max_width: width,
                ..Default::default()
            },
            ..Default::default()
        },
        chars: 0,
        links: Vec::new(),
        code_blocks: Vec::new(),
        code_start: None,
        mono: mono.clone(),
        text: text_color,
        strong: strong_color,
        link: link_color,
        code_bg,
    };

    for line in lines {
        if !matches!(line, Line::Code { .. })
            && let Some(start) = builder.code_start.take()
        {
            builder.code_blocks.push(start..builder.chars);
        }
        match line {
            Line::Blank => {}
            Line::Heading { level, spans } => {
                let base = TextFormat {
                    font_id: FontId::new(heading_size(*level), body.family.clone()),
                    color: strong_color,
                    ..Default::default()
                };
                for span in spans {
                    builder.push_span(span, &base);
                }
            }
            Line::Bullet { indent, spans } => {
                let base = builder.plain(body.clone());
                builder.push("- ", *indent as f32 * 6.0, base.clone());
                for span in spans {
                    builder.push_span(span, &base);
                }
            }
            Line::Text { spans } => {
                let base = builder.plain(body.clone());
                for span in spans {
                    builder.push_span(span, &base);
                }
            }
            Line::Code { text } => {
                builder.code_start.get_or_insert(builder.chars);
                builder.push(text, 0.0, builder.plain(mono.clone()));
            }
        }
        builder.push("\n", 0.0, builder.plain(body.clone()));
    }
    if let Some(start) = builder.code_start.take() {
        builder.code_blocks.push(start..builder.chars);
    }
    (builder.job, builder.links, builder.code_blocks)
}

/// One [`LayoutJob`] under construction, tracking what sits where.
struct JobBuilder {
    job: LayoutJob,
    /// Characters appended so far.
    chars: usize,
    links: Vec<Link>,
    code_blocks: Vec<Range<usize>>,
    code_start: Option<usize>,
    mono: FontId,
    text: Color32,
    strong: Color32,
    link: Color32,
    code_bg: Color32,
}

impl JobBuilder {
    fn plain(&self, font: FontId) -> TextFormat {
        TextFormat {
            font_id: font,
            color: self.text,
            ..Default::default()
        }
    }

    fn push(&mut self, text: &str, leading_space: f32, format: TextFormat) {
        self.job.append(text, leading_space, format);
        self.chars += text.chars().count();
    }

    /// Push one inline span, applying its code/emphasis/link styling and
    /// remembering a link's range.
    fn push_span(&mut self, span: &Span, base: &TextFormat) {
        let mut format = base.clone();
        if span.code {
            format.font_id = self.mono.clone();
            format.background = self.code_bg;
        }
        if span.bold {
            format.color = self.strong;
        }
        if span.italic {
            format.italics = true;
        }
        let start = self.chars;
        if let Some(url) = &span.link {
            format.color = self.link;
            format.underline = Stroke::new(1.0, self.link);
            self.push(&span.text, 0.0, format);
            self.links.push(Link {
                chars: start..self.chars,
                url: url.clone(),
            });
        } else {
            self.push(&span.text, 0.0, format);
        }
    }
}

/// Map each code block's character range to the galley rows it covers.
fn code_row_ranges(galley: &Galley, code_blocks: &[Range<usize>]) -> Vec<Range<usize>> {
    if code_blocks.is_empty() {
        return Vec::new();
    }
    let mut row_starts = Vec::with_capacity(galley.rows.len());
    let mut chars = 0usize;
    for row in &galley.rows {
        row_starts.push(chars);
        chars += row.char_count_including_newline().0;
    }
    code_blocks
        .iter()
        .map(|block| {
            let first = row_starts
                .partition_point(|start| *start <= block.start)
                .saturating_sub(1);
            let last = row_starts
                .partition_point(|start| *start < block.end)
                .saturating_sub(1);
            first..last + 1
        })
        .collect()
}

fn heading_size(level: usize) -> f32 {
    match level {
        1 => 20.0,
        2 => 16.0,
        _ => 14.0,
    }
}
