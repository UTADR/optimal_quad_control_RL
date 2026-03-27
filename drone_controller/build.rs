use std::path::Path;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("generated.rs");

    let real = Path::new("src/generated.rs");
    let stub = Path::new("src/generated_stub.rs");
    let source = if real.exists() { real } else { stub };

    std::fs::copy(source, &dest).expect("failed to copy generated.rs to OUT_DIR");

    println!("cargo:rerun-if-changed=src/generated.rs");
    println!("cargo:rerun-if-changed=src/generated_stub.rs");
}
