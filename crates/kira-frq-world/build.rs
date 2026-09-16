fn main() {
    let world_src = "../../third_party/World/src";
    let sources = [
        "dio.cpp",
        "harvest.cpp",
        "stonemask.cpp",
        "common.cpp",
        "fft.cpp",
        "matlabfunctions.cpp",
    ];

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .include(world_src)
        .warnings(false)
        .flag_if_supported("/EHsc");
    for source in sources {
        build.file(format!("{world_src}/{source}"));
    }
    build.file("shim/world_shim.cpp");
    build.compile("world");

    println!("cargo:rerun-if-changed={world_src}");
    println!("cargo:rerun-if-changed=shim/world_shim.cpp");
    println!("cargo:rerun-if-changed=shim/world_shim.h");
}
