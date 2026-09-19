fn main() {
    let world_src = "../../third_party/World/src";
    let sources = [
        "dio.cpp",
        "harvest.cpp",
        "stonemask.cpp",
        "d4c.cpp",
        "common.cpp",
        "fft.cpp",
        "matlabfunctions.cpp",
    ];

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .include(world_src)
        .include("shim")
        .warnings(false)
        .flag_if_supported("/EHsc");
    for source in sources {
        build.file(format!("{world_src}/{source}"));
    }
    build.file("shim/world_shim.cpp");

    prefer_clang_cl(&mut build);
    build.compile("world");

    println!("cargo:rerun-if-changed={world_src}");
    println!("cargo:rerun-if-changed=shim/world_shim.cpp");
    println!("cargo:rerun-if-changed=shim/world_shim.h");
    println!("cargo:rerun-if-changed=shim/progress_hook.h");
    // The toolchain selection above reads these.
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CXX");
}

/// Toolchain policy (map Notes / #15): on Windows MSVC targets, prefer
/// clang-cl when it is installed; otherwise fall back to cc's default (MSVC
/// `cl`). An explicit `CC`/`CXX` in the environment overrides either — `cc`
/// already honors those, so auto-detection steps aside whenever they are set.
/// Non-Windows targets keep cc's default (the WORLD build stays
/// toolchain-agnostic).
fn prefer_clang_cl(build: &mut cc::Build) {
    if std::env::var_os("CC").is_some() || std::env::var_os("CXX").is_some() {
        println!(
            "cargo:warning=kirafrq-world-binding: CC/CXX set in the environment; cc honors them"
        );
        return;
    }
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os != "windows" || target_env != "msvc" {
        return;
    }
    match find_clang_cl() {
        Some(clang_cl) => {
            println!(
                "cargo:warning=kirafrq-world-binding: using clang-cl at {}",
                clang_cl.display()
            );
            build.compiler(clang_cl);
        }
        None => {
            println!(
                "cargo:warning=kirafrq-world-binding: clang-cl not found; falling back to MSVC cl"
            );
        }
    }
}

/// Look for `clang-cl.exe` on `PATH`, then in LLVM's default install location
/// (sessions spawned before a PATH update may not see an installed LLVM).
fn find_clang_cl() -> Option<std::path::PathBuf> {
    std::env::var_os("PATH")
        .and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join("clang-cl.exe"))
                .find(|candidate| candidate.is_file())
        })
        .or_else(|| {
            let default = std::path::PathBuf::from(r"C:\Program Files\LLVM\bin\clang-cl.exe");
            default.is_file().then_some(default)
        })
}
