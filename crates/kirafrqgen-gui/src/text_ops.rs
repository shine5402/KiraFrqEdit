//! Single-line text editing behind the voicebank path field's context menu
//! (#32): Cut / Copy / Paste / Select All semantics kept free of egui so they
//! can be unit-tested. `app.rs` adapts egui's cursor state onto these.

/// A half-open character range (`start..end`) into a single-line field.
///
/// Indices count characters, not bytes, so they line up with egui's cursors and
/// stay valid across Unicode text (voicebank paths are full of CJK).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CharRange {
    pub start: usize,
    pub end: usize,
}

impl CharRange {
    /// The range between two character indices, in either order.
    pub fn new(a: usize, b: usize) -> Self {
        Self {
            start: a.min(b),
            end: b.max(a),
        }
    }

    /// A collapsed range: a caret with nothing selected.
    pub fn caret(index: usize) -> Self {
        Self {
            start: index,
            end: index,
        }
    }

    pub fn is_empty(self) -> bool {
        self.start == self.end
    }

    /// Clamp to `[0, chars]`, keeping the ends ordered.
    fn clamp(self, chars: usize) -> Self {
        Self::new(self.start.min(chars), self.end.min(chars))
    }
}

/// Which menu items are available for a field's current text, selection and
/// clipboard.
pub struct MenuState {
    pub cut: bool,
    pub copy: bool,
    pub paste: bool,
    pub select_all: bool,
}

/// Decide the path field's menu items from its text, its selection and the
/// clipboard snapshot: Cut and Copy need a selection, Paste needs clipboard
/// text, and Select All needs a non-empty field.
pub fn menu_state(text: &str, selection: CharRange, clipboard: Option<&str>) -> MenuState {
    let has_selection = !selected(text, selection).is_empty();
    MenuState {
        cut: has_selection,
        copy: has_selection,
        paste: clipboard.is_some_and(|clip| !clip.is_empty()),
        select_all: !text.is_empty(),
    }
}

/// The characters `range` covers.
pub fn selected(text: &str, range: CharRange) -> &str {
    let range = range.clamp(char_count(text));
    &text[byte_of(text, range.start)..byte_of(text, range.end)]
}

/// Delete the selection and return the removed text (for the clipboard) plus
/// the caret left behind, or `None` when there is no selection.
pub fn cut(text: &mut String, range: CharRange) -> Option<(String, CharRange)> {
    let range = range.clamp(char_count(text));
    if range.is_empty() {
        return None;
    }
    let (start, end) = (byte_of(text, range.start), byte_of(text, range.end));
    let taken = text[start..end].to_owned();
    text.replace_range(start..end, "");
    Some((taken, CharRange::caret(range.start)))
}

/// Paste `clip` over the selection, returning the caret after the insertion.
///
/// Line breaks become spaces so the field stays a single line, and an empty
/// clipboard is a no-op (`None`), never a destructive edit.
pub fn paste(text: &mut String, range: CharRange, clip: &str) -> Option<CharRange> {
    if clip.is_empty() {
        return None;
    }
    let single_line = clip.replace(['\r', '\n'], " ");
    Some(replace(text, range, &single_line))
}

/// The range covering all of `text` (a caret at zero for an empty field).
pub fn select_all(text: &str) -> CharRange {
    CharRange::new(0, char_count(text))
}

fn replace(text: &mut String, range: CharRange, insert: &str) -> CharRange {
    let range = range.clamp(char_count(text));
    let (start, end) = (byte_of(text, range.start), byte_of(text, range.end));
    text.replace_range(start..end, insert);
    CharRange::caret(range.start + insert.chars().count())
}

fn char_count(text: &str) -> usize {
    text.chars().count()
}

/// The byte offset of a character index; past the end clamps to the length.
fn byte_of(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map_or(text.len(), |(byte, _)| byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_order_and_clamp() {
        assert_eq!(CharRange::new(5, 2), CharRange { start: 2, end: 5 });
        assert_eq!(
            CharRange::new(9, 4).clamp(6),
            CharRange { start: 4, end: 6 }
        );
        assert!(CharRange::caret(3).is_empty());
    }

    #[test]
    fn selected_slices_by_characters_not_bytes() {
        // "音" is 3 bytes, so a byte-based slice would panic or mis-slice.
        let text = "声/あ.wav";
        assert_eq!(selected(text, CharRange::new(0, 0)), "");
        assert_eq!(selected(text, CharRange::new(0, 1)), "声");
        assert_eq!(selected(text, CharRange::new(2, 3)), "あ");
        assert_eq!(selected(text, CharRange::new(0, 99)), text);
    }

    #[test]
    fn cut_removes_only_the_selection() {
        let mut text = "声/aiueo.wav".to_owned();
        let (taken, caret) = cut(&mut text, CharRange::new(2, 7)).unwrap();
        assert_eq!(taken, "aiueo");
        assert_eq!(text, "声/.wav");
        assert_eq!(caret, CharRange::caret(2));
    }

    #[test]
    fn cut_without_a_selection_is_a_no_op() {
        let mut text = "abc".to_owned();
        assert_eq!(cut(&mut text, CharRange::caret(1)), None);
        assert_eq!(text, "abc");
    }

    #[test]
    fn paste_replaces_the_selection_and_lands_the_caret_after_it() {
        let mut text = "C:/banks/old".to_owned();
        let caret = paste(&mut text, CharRange::new(9, 12), "new").unwrap();
        assert_eq!(text, "C:/banks/new");
        assert_eq!(caret, CharRange::caret(12));
    }

    #[test]
    fn paste_flattens_line_breaks_and_counts_unicode() {
        let mut text = "ab".to_owned();
        let caret = paste(&mut text, CharRange::caret(1), "声\n").unwrap();
        assert_eq!(text, "a声 b");
        assert_eq!(caret, CharRange::caret(3));
    }

    #[test]
    fn paste_of_an_empty_clipboard_changes_nothing() {
        let mut text = "abc".to_owned();
        assert_eq!(paste(&mut text, CharRange::new(0, 2), ""), None);
        assert_eq!(text, "abc");
    }

    #[test]
    fn replace_with_empty_insert_deletes_the_range() {
        let mut text = "aXb".to_owned();
        let caret = replace(&mut text, CharRange::new(1, 2), "");
        assert_eq!(text, "ab");
        assert_eq!(caret, CharRange::caret(1));
    }

    #[test]
    fn select_all_covers_unicode_and_is_empty_for_an_empty_field() {
        assert_eq!(select_all("声/あ.wav"), CharRange::new(0, 7));
        assert!(select_all("").is_empty());
    }

    #[test]
    fn out_of_range_inputs_do_not_panic() {
        let mut text = "ab".to_owned();
        assert_eq!(selected(&text, CharRange::new(1, 99)), "b");
        let (taken, caret) = cut(&mut text, CharRange::new(1, 99)).unwrap();
        assert_eq!(taken, "b");
        assert_eq!(text, "a");
        assert_eq!(caret, CharRange::caret(1));
    }

    #[test]
    fn cut_and_copy_need_a_selection_but_paste_needs_clipboard_text() {
        let none = menu_state("abc", CharRange::caret(1), Some("clip"));
        assert!(!none.cut && !none.copy);
        assert!(none.paste && none.select_all);

        let some = menu_state("abc", CharRange::new(0, 2), Some("clip"));
        assert!(some.cut && some.copy && some.paste && some.select_all);
    }

    #[test]
    fn an_unavailable_or_empty_clipboard_disables_paste() {
        for clipboard in [None, Some("")] {
            let state = menu_state("abc", CharRange::caret(0), clipboard);
            assert!(!state.paste);
        }
    }

    #[test]
    fn an_empty_field_disables_select_all_and_a_unicode_selection_enables_cut() {
        assert!(!menu_state("", CharRange::caret(0), None).select_all);
        let state = menu_state("声/あ.wav", CharRange::new(0, 2), None);
        assert!(state.cut && state.copy);
    }
}
