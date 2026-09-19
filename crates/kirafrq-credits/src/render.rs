//! A small Markdown-to-terminal renderer for `CREDITS.md`.
//!
//! The document is ours, so this handles exactly what it contains: ATX
//! headings, `- ` bullets, `[text](url)` links, inline code, emphasis, and
//! fenced code blocks (printed verbatim). Anything else passes through.

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
    let mut in_fence = false;
    for line in markdown.lines() {
        if is_fence(line) {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            out.push_str(line);
        } else if let Some((level, text)) = heading(line) {
            out.push_str(&style.heading(level, &inline(text, style)));
        } else if let Some((indent, text)) = bullet(line) {
            out.push_str(indent);
            out.push_str("- ");
            out.push_str(&inline(text, style));
        } else {
            out.push_str(&inline(line, style));
        }
        out.push('\n');
    }
    out
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

/// The indentation and text of a `- ` / `* ` / `+ ` bullet line.
fn bullet(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim_start();
    let indent = &line[..line.len() - trimmed.len()];
    let text = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))?;
    Some((indent, text))
}

fn is_fence(line: &str) -> bool {
    line.trim_start().starts_with("```")
}

/// Apply the inline spans of `text` (links, code, emphasis), leaving anything
/// else alone.
fn inline(text: &str, style: Style) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        if let Some((label, url, used)) = link(rest) {
            out.push_str(&style.link(&inline(label, style)));
            out.push_str(&style.link_url(url));
            rest = &rest[used..];
        } else if let Some((code, used)) = delimited(rest, "`") {
            out.push_str(&style.code(code));
            rest = &rest[used..];
        } else if let Some((inner, used)) = delimited(rest, "**") {
            out.push_str(&style.bold(&inline(inner, style)));
            rest = &rest[used..];
        } else if let Some((inner, used)) = delimited(rest, "*") {
            out.push_str(&style.italic(&inline(inner, style)));
            rest = &rest[used..];
        } else {
            let ch = rest.chars().next().expect("rest is not empty");
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
    out
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

impl Style {
    fn heading(self, level: usize, text: &str) -> String {
        self.styled(if level == 1 { "1;4" } else { "1" }, text)
    }

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
