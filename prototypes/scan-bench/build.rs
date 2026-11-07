// Monero's own curve code, the code wallet2's generate_key_derivation runs on a
// phone (its faster assembly options are x86-64 only).
fn main() {
    println!("cargo:rerun-if-changed=vendor");
    cc::Build::new()
        .files(["vendor/crypto-ops.c", "vendor/crypto-ops-data.c"])
        .include("vendor")
        .opt_level(3)
        .warnings(false)
        .compile("cryptoops");
}
