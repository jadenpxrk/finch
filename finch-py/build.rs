fn main() {
    // When building a Python extension module on macOS, allow unresolved Python
    // C-API symbols. The Python interpreter provides them at load time.
    //
    // This keeps `cargo build -p finch-py` working without requiring the caller
    // to provide an explicit libpython link configuration (maturin handles that).
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
