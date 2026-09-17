//! KiraFrqGen: the egui front end over `kirafrqgen-core` (#23, shell B from #13).

mod app;
mod run;
mod tree;

use app::KiraFrqGenApp;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 680.0])
            .with_min_inner_size([760.0, 500.0]),
        ..Default::default()
    };
    eframe::run_native(
        "KiraFrqGen",
        native_options,
        Box::new(|cc| {
            let mut fonts = eframe::egui::FontDefinitions::default();
            egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
            add_cjk_fallback(&mut fonts);
            cc.egui_ctx.set_fonts(fonts);
            Ok(Box::new(KiraFrqGenApp::new()))
        }),
    )
}

/// egui's bundled fonts have no CJK coverage, and UTAU voicebanks are full of
/// Japanese/Chinese filenames. Fall back to whatever CJK font the OS ships,
/// so filenames render instead of showing tofu (#13).
fn add_cjk_fallback(fonts: &mut eframe::egui::FontDefinitions) {
    #[cfg(target_os = "windows")]
    const CANDIDATES: &[&str] = &[
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\msjh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ];
    #[cfg(target_os = "macos")]
    const CANDIDATES: &[&str] = &[
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ];
    #[cfg(all(unix, not(target_os = "macos")))]
    const CANDIDATES: &[&str] = &[
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    ];

    for (index, path) in CANDIDATES.iter().enumerate() {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let name = format!("cjk-{index}");
        let data = eframe::egui::FontData::from_owned(bytes);
        fonts.font_data.insert(name.clone(), data.into());
        for family in [
            eframe::egui::FontFamily::Proportional,
            eframe::egui::FontFamily::Monospace,
        ] {
            fonts.families.entry(family).or_default().push(name.clone());
        }
    }
}
