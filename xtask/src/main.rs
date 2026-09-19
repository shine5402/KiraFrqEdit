//! `cargo xtask credits`: regenerate the dependency notices in `CREDITS.md`.
//!
//! Runs `cargo about` twice, once for the compact dependency index and once for
//! the full license texts, and splices the two rendered blocks between the
//! markers in `CREDITS.md`. The result is committed, so normal builds need
//! neither cargo-about nor the network. `cargo xtask credits --check` verifies
//! the committed file is current, for CI.

use std::path::Path;

use kirafrq_credits::{FULL_BEGIN, FULL_END, SUMMARY_BEGIN, SUMMARY_END};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let check = match args.as_slice() {
        [command] if command == "credits" => false,
        [command, flag] if command == "credits" && flag == "--check" => true,
        _ => return Err("usage: cargo xtask credits [--check]".to_string()),
    };

    let root = repo_root()?;
    let templates = root.join("xtask").join("templates");
    let summary = generate(&root, &templates.join("summary.hbs"))?;
    let full = generate(&root, &templates.join("full.hbs"))?;

    let credits_path = root.join("CREDITS.md");
    let current = std::fs::read_to_string(&credits_path)
        .map_err(|error| format!("read {}: {error}", credits_path.display()))?;
    let updated = replace_block(&current, SUMMARY_BEGIN, SUMMARY_END, &summary)?;
    let updated = replace_block(&updated, FULL_BEGIN, FULL_END, &full)?;

    if updated == current {
        println!("CREDITS.md is up to date");
        return Ok(());
    }
    if check {
        return Err("CREDITS.md is stale; run `cargo xtask credits`".to_string());
    }
    std::fs::write(&credits_path, updated)
        .map_err(|error| format!("write {}: {error}", credits_path.display()))?;
    println!("wrote {}", credits_path.display());
    Ok(())
}

/// The workspace root: the parent of this crate's manifest directory.
fn repo_root() -> Result<std::path::PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "xtask has no parent directory".to_string())
}

/// Render `template` through cargo-about for the whole workspace. cargo-about
/// refuses to write to a redirected stdout on Windows, so it always writes a
/// temp file that is read back here.
fn generate(root: &Path, template: &Path) -> Result<String, String> {
    let stem = template
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("credits");
    let out = std::env::temp_dir().join(format!("krq-credits-{stem}.md"));
    let status = std::process::Command::new("cargo")
        .current_dir(root)
        .arg("about")
        .arg("generate")
        .arg("--workspace")
        .arg("-o")
        .arg(&out)
        .arg(template)
        .status()
        .map_err(|error| format!("run cargo-about: {error}; is it installed?"))?;
    if !status.success() {
        return Err(format!(
            "cargo-about failed ({status}) for {}",
            template.display()
        ));
    }
    let text = std::fs::read_to_string(&out)
        .map_err(|error| format!("read {}: {error}", out.display()))?;
    let _ = std::fs::remove_file(&out);
    Ok(text)
}

/// Replace whatever sits between the `begin` and `end` marker lines with
/// `body`, keeping the markers themselves.
fn replace_block(doc: &str, begin: &str, end: &str, body: &str) -> Result<String, String> {
    #[derive(PartialEq)]
    enum State {
        Outside,
        Inside,
    }

    let mut out = String::new();
    let mut state = State::Outside;
    let mut found_begin = false;
    let mut found_end = false;
    for line in doc.lines() {
        match state {
            State::Outside if line.trim() == begin => {
                found_begin = true;
                out.push_str(line);
                out.push('\n');
                out.push('\n');
                out.push_str(body.trim_end());
                out.push('\n');
                state = State::Inside;
            }
            State::Inside if line.trim() == end => {
                found_end = true;
                out.push('\n');
                out.push_str(line);
                out.push('\n');
                state = State::Outside;
            }
            State::Inside => {}
            State::Outside => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    if !found_begin || !found_end {
        return Err(format!("CREDITS.md is missing the {begin} / {end} markers"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "head\n<!-- b -->\nold\n<!-- e -->\ntail\n";

    #[test]
    fn a_block_is_replaced_between_its_markers() {
        let out = replace_block(DOC, "<!-- b -->", "<!-- e -->", "new\n").unwrap();
        assert_eq!(out, "head\n<!-- b -->\n\nnew\n\n<!-- e -->\ntail\n");
    }

    #[test]
    fn a_missing_marker_is_an_error() {
        assert!(replace_block("no markers\n", "<!-- b -->", "<!-- e -->", "x").is_err());
    }

    #[test]
    fn replacement_is_idempotent() {
        let once = replace_block(DOC, "<!-- b -->", "<!-- e -->", "new\n").unwrap();
        let twice = replace_block(&once, "<!-- b -->", "<!-- e -->", "new\n").unwrap();
        assert_eq!(once, twice);
    }
}
