//! Parse `CREDITS.md` into lines of styled spans.
//!
//! One parser feeds both surfaces: [`crate::render`] writes the spans as
//! terminal text, and the GUI lays them out with egui, so the two cannot
//! drift. The document is ours, so this handles exactly what it contains: ATX
//! headings, `- ` bullets, `[text](url)` links, inline code, emphasis, and
//! fenced code blocks (kept verbatim). Anything else passes through.
//!
//! Like GitHub, a single newline inside a paragraph is a soft break: the
//! source's hard-wrapped lines are joined into one logical line and each
//! surface reflows it to its own width. A blank line ends the paragraph.

/// A run of text with the inline styling that applies to it. Nesting is
/// flattened, so a link wrapping inline code carries both flags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    /// The literal text, with all markup removed.
    pub text: String,
    /// Rendered in a monospace font on a tinted background.
    pub code: bool,
    /// Rendered with the stronger text colour.
    pub bold: bool,
    /// Rendered slanted.
    pub italic: bool,
    /// When set, the span is a link to this URL.
    pub link: Option<String>,
}

/// One line of the document, with its inline markup already parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Line {
    /// A blank separator line.
    Blank,
    /// An ATX heading (`#` through `######`).
    Heading { level: usize, spans: Vec<Span> },
    /// A `- ` / `* ` / `+ ` bullet, with its leading indentation in columns.
    Bullet { indent: usize, spans: Vec<Span> },
    /// A paragraph, its soft-wrapped source lines already joined.
    Text { spans: Vec<Span> },
    /// A line inside a fenced code block, kept verbatim.
    Code { text: String },
}

/// Parse `markdown` into styled lines. The fence markers themselves are
/// consumed; their contents become [`Line::Code`].
pub fn document(markdown: &str) -> Vec<Line> {
    let mut lines = Vec::new();
    let mut in_fence = false;
    let mut paragraph: Vec<&str> = Vec::new();
    for line in markdown.lines() {
        if is_fence(line) {
            flush_paragraph(&mut paragraph, &mut lines);
            in_fence = !in_fence;
        } else if in_fence {
            lines.push(Line::Code {
                text: line.to_owned(),
            });
        } else if let Some((level, text)) = heading(line) {
            flush_paragraph(&mut paragraph, &mut lines);
            lines.push(Line::Heading {
                level,
                spans: inline(text),
            });
        } else if let Some((indent, text)) = bullet(line) {
            flush_paragraph(&mut paragraph, &mut lines);
            lines.push(Line::Bullet {
                indent,
                spans: inline(text),
            });
        } else if line.trim().is_empty() {
            flush_paragraph(&mut paragraph, &mut lines);
            lines.push(Line::Blank);
        } else {
            paragraph.push(line.trim());
        }
    }
    flush_paragraph(&mut paragraph, &mut lines);
    lines
}

/// Join the soft-wrapped source lines of a paragraph into one [`Line::Text`],
/// exactly as GitHub reflows a paragraph.
fn flush_paragraph(paragraph: &mut Vec<&str>, lines: &mut Vec<Line>) {
    if paragraph.is_empty() {
        return;
    }
    let joined = paragraph.join(" ");
    paragraph.clear();
    lines.push(Line::Text {
        spans: inline(&joined),
    });
}

/// The text of an ATX heading line, with its level.
fn heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let level = trimmed.bytes().take_while(|byte| *byte == b'#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let text = trimmed[level..].strip_prefix(' ')?;
    Some((level, text.trim_end()))
}

/// The indentation (in bytes, which for our space-indented document is also
/// columns) and text of a `- ` / `* ` / `+ ` bullet line.
fn bullet(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let indent = line.len() - trimmed.len();
    let text = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))?;
    Some((indent, text))
}

fn is_fence(line: &str) -> bool {
    line.trim_start().starts_with("```")
}

/// The inline spans of `text` (links, code, emphasis), leaving anything else
/// alone.
fn inline(text: &str) -> Vec<Span> {
    let mut spans = Vec::new();
    push_inline(text, &Acc::default(), &mut spans);
    spans
}

/// The styling accumulated from enclosing inline markup.
#[derive(Clone, Default)]
struct Acc {
    code: bool,
    bold: bool,
    italic: bool,
    link: Option<String>,
}

fn push_inline(text: &str, acc: &Acc, spans: &mut Vec<Span>) {
    let mut rest = text;
    let mut plain = String::new();
    while !rest.is_empty() {
        if let Some((label, url, used)) = link(rest) {
            flush(&mut plain, acc, spans);
            let mut nested = acc.clone();
            nested.link = Some(url.to_owned());
            push_inline(label, &nested, spans);
            rest = &rest[used..];
        } else if let Some((code, used)) = delimited(rest, "`") {
            flush(&mut plain, acc, spans);
            let mut nested = acc.clone();
            nested.code = true;
            spans.push(nested.span(code));
            rest = &rest[used..];
        } else if let Some((inner, used)) = delimited(rest, "**") {
            flush(&mut plain, acc, spans);
            let mut nested = acc.clone();
            nested.bold = true;
            push_inline(inner, &nested, spans);
            rest = &rest[used..];
        } else if let Some((inner, used)) = delimited(rest, "*") {
            flush(&mut plain, acc, spans);
            let mut nested = acc.clone();
            nested.italic = true;
            push_inline(inner, &nested, spans);
            rest = &rest[used..];
        } else {
            let ch = rest.chars().next().expect("rest is not empty");
            plain.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
    flush(&mut plain, acc, spans);
}

fn flush(plain: &mut String, acc: &Acc, spans: &mut Vec<Span>) {
    if !plain.is_empty() {
        spans.push(acc.span(plain));
        plain.clear();
    }
}

impl Acc {
    fn span(&self, text: &str) -> Span {
        Span {
            text: text.to_owned(),
            code: self.code,
            bold: self.bold,
            italic: self.italic,
            link: self.link.clone(),
        }
    }
}

/// Parse a `[label](url)` at the start of `text`; returns the label, url and
/// the bytes consumed.
fn link(text: &str) -> Option<(&str, &str, usize)> {
    let rest = text.strip_prefix('[')?;
    let label_end = rest.find("](")?;
    let after = &rest[label_end + 2..];
    let url_end = after.find(')')?;
    let used = 1 + label_end + 2 + url_end + 1;
    Some((&rest[..label_end], &after[..url_end], used))
}

/// The content between a leading `open` and the next `open`, plus the bytes
/// consumed including both delimiters.
fn delimited<'a>(text: &'a str, open: &str) -> Option<(&'a str, usize)> {
    let rest = text.strip_prefix(open)?;
    let end = rest.find(open)?;
    Some((&rest[..end], open.len() * 2 + end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(spans: &[Span]) -> String {
        spans.iter().map(|span| span.text.as_str()).collect()
    }

    #[test]
    fn a_heading_carries_its_level_and_strips_the_markers() {
        let lines = document("## Project credits\n");
        let Line::Heading { level, spans } = &lines[0] else {
            panic!("expected a heading: {lines:?}");
        };
        assert_eq!(*level, 2);
        assert_eq!(text(spans), "Project credits");
        assert!(!spans[0].bold, "headings style whole-line, not per span");
    }

    #[test]
    fn a_link_becomes_one_span_that_keeps_its_url() {
        let lines = document("See [WORLD](https://example.test/w).\n");
        let Line::Text { spans } = &lines[0] else {
            panic!("expected a text line: {lines:?}");
        };
        assert_eq!(text(spans), "See WORLD.");
        assert_eq!(spans[1].text, "WORLD");
        assert_eq!(spans[1].link.as_deref(), Some("https://example.test/w"));
    }

    #[test]
    fn a_bullet_keeps_its_indent_and_parses_inline_code() {
        let lines = document("  - `dep` 1.0.0 - MIT\n");
        let Line::Bullet { indent, spans } = &lines[0] else {
            panic!("expected a bullet: {lines:?}");
        };
        assert_eq!(*indent, 2);
        assert_eq!(text(spans), "dep 1.0.0 - MIT");
        assert!(spans[0].code);
    }

    #[test]
    fn a_fenced_block_is_kept_verbatim_without_its_fences() {
        let lines = document("````text\n# not a heading\n````\n");
        assert_eq!(
            lines,
            vec![Line::Code {
                text: "# not a heading".into()
            }]
        );
    }

    #[test]
    fn emphasis_nests_and_flattens() {
        let lines = document("a **b** *c*\n");
        let Line::Text { spans } = &lines[0] else {
            panic!("expected a text line: {lines:?}");
        };
        assert!(spans[1].bold);
        assert!(spans[3].italic);
        assert_eq!(text(spans), "a b c");
    }
}
