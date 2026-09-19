//! Write the parsed [`document`] as terminal text.
//!
//! [`Style::Plain`] drops the markup; [`Style::Ansi`] adds bold headings,
//! coloured code and links, and a dimmed URL after each link.

use crate::parse::{Line, Span, document};

/// Whether the rendered text carries ANSI styling.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Style {
    /// Plain text, for pipes and files.
    Plain,
    /// Bold headings and coloured code and links, for a terminal.
    Ansi,
}

/// Render `markdown` for a terminal (or, with [`Style::Plain`], for a pipe).
pub fn render(markdown: &str, style: Style) -> String {
    let mut out = String::with_capacity(markdown.len());
    for line in document(markdown) {
        match line {
            Line::Blank => {}
            Line::Heading { level, spans } => {
                push_styled(&mut out, style, heading_code(level), &spans);
            }
            Line::Bullet { indent, spans } => {
                for _ in 0..indent {
                    out.push(' ');
                }
                out.push_str("- ");
                push_styled(&mut out, style, "", &spans);
            }
            Line::Text { spans } => push_styled(&mut out, style, "", &spans),
            Line::Code { text } => out.push_str(&text),
        }
        out.push('\n');
    }
    out
}

fn heading_code(level: usize) -> &'static str {
    if level == 1 { "1;4" } else { "1" }
}

fn push_styled(out: &mut String, style: Style, code: &str, spans: &[Span]) {
    let text = spans_to_string(spans, style);
    out.push_str(&style.styled(code, &text));
}

fn spans_to_string(spans: &[Span], style: Style) -> String {
    let mut out = String::new();
    for span in spans {
        if span.link.is_some() {
            out.push_str(&style.link(&span.text));
        } else if span.code {
            out.push_str(&style.code(&span.text));
        } else if span.bold {
            out.push_str(&style.bold(&span.text));
        } else if span.italic {
            out.push_str(&style.italic(&span.text));
        } else {
            out.push_str(&span.text);
        }
        if let Some(url) = &span.link {
            out.push_str(&style.link_url(url));
        }
    }
    out
}

impl Style {
    fn code(self, text: &str) -> String {
        self.styled("36", text)
    }

    fn bold(self, text: &str) -> String {
        self.styled("1", text)
    }

    fn italic(self, text: &str) -> String {
        self.styled("3", text)
    }

    fn link(self, text: &str) -> String {
        self.styled("34", text)
    }

    fn link_url(self, url: &str) -> String {
        match self {
            Style::Plain => format!(" ({url})"),
            Style::Ansi => format!(" \x1b[2m({url})\x1b[0m"),
        }
    }

    fn styled(self, code: &str, text: &str) -> String {
        match self {
            Style::Plain => text.to_owned(),
            Style::Ansi => format!("\x1b[{code}m{text}\x1b[0m"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_rendering_drops_the_markup() {
        let text = render(
            "# Title\n\nSee [WORLD](https://example.test/w) and `code`.\n\n- one\n",
            Style::Plain,
        );
        assert_eq!(
            text,
            "Title\n\nSee WORLD (https://example.test/w) and code.\n\n- one\n"
        );
    }

    #[test]
    fn headings_are_styled_only_in_ansi() {
        assert_eq!(render("# T\n", Style::Plain), "T\n");
        assert_eq!(render("# T\n", Style::Ansi), "\x1b[1;4mT\x1b[0m\n");
        assert_eq!(render("### T\n", Style::Ansi), "\x1b[1mT\x1b[0m\n");
    }

    #[test]
    fn fenced_code_is_kept_verbatim_and_the_fences_dropped() {
        let text = render("````text\n# not a heading\n````\n", Style::Plain);
        assert_eq!(text, "# not a heading\n");
    }

    #[test]
    fn emphasis_is_stripped() {
        assert_eq!(render("a **b** *c*\n", Style::Plain), "a b c\n");
    }
}
