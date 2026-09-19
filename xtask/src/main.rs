//! `cargo xtask credits`: regenerate the dependency notices in `CREDITS.md`.
//!
//! Runs `cargo about` twice, once for the compact dependency index and once for
//! the full license texts, and splices the two rendered blocks between the
//! markers in `CREDITS.md`. The result is committed, so normal builds need
//! neither cargo-about nor the network. `cargo xtask credits --check` reports
//! whether the committed file is current instead of writing it.

use std::path::{Path, PathBuf};
use std::process::Command;

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
    let updated = kirafrq_credits::assemble(&current, &summary, &full)?;

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
fn repo_root() -> Result<PathBuf, String> {
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
    let status = Command::new("cargo")
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
