# monokulo-8ar

Finds the Monero blocks that may pay a wallet, fast, on the GPU or the CPU.

Proving that a block doesn't pay a wallet is nearly all of a scan's cost:
8·a·R for each transaction key (a is the view key, R the key), then a view
tag for each output. monero-oxide's `Scanner` works out 8aR once per output
and decompresses every output key. This crate does it once per transaction
key, with one field inversion shared across each 64 keys, and only
decompresses an output key past a matching view tag. On a GPU (Vulkan,
through `ash`) it computes 8aR for tens of thousands of keys per dispatch.

It decides nothing about money. A block it flags goes whole to the
unmodified `Scanner`, which alone finds the outputs, their amounts and their
subaddresses. Its one duty is never to miss, and anything it doesn't
understand (a newer fork, missing transactions) is flagged. Its reading of
`Extra` and of bad keys is held to monero-oxide's by tests.

A GPU is only trusted after it passes test vectors, once per process, and
again with each wallet's own view key; a GPU that fails is retired until the
process restarts, and the CPU does the work. Which engine runs is a matter of
speed and power, never of the answer.

- `crates/monokulo-8ar`: the crate. `Prefilter` holds a wallet's view key and
  the spend keys it watches; `Prefilter::flag` takes a batch of
  `ScannableBlock`s and says which may pay it. `Mode` (Auto, the default; GPU;
  CPU), `warm_up`, `check_wallet`, `rest` and `set_cache_dir` manage the one
  GPU a process has. The compiled pipeline is cached as `scan-gpu.cache` in
  the folder given to `set_cache_dir`.
- `prototypes/`: the experiments it came from, kept for the record. Each is a
  workspace of its own and isn't built with this one. `scan-vulkan` asked
  whether a phone's GPU could find 8aR faster than its cores; `scan-bench`
  compared ways of scanning on one fixed set of blocks and wrote
  `fixtures/`. `scan-bench` fetched blocks through a Fob crate (`fob-core`)
  that no longer exists, so it doesn't build as it is;
  `monero-wallet-patched` is the patched monero-wallet it builds against
  (a `[patch.crates-io]` in its `Cargo.toml`).

## Who uses it

[Fob](https://github.com/OceanEilonwy/monokulo-fob), a Monero wallet for
phones and the desktop, where it was written: `fob-scanner` wraps it with
Fob's thread fan-out over flagged blocks and its choice of where Auto hands
over to the GPU (512 keys, about nine blocks, measured on a Mali-G715 for
energy). Fob's `docs/decisions.md` gives the reasons (decisions 331, 332,
350, 351).

[monokulo-lws](https://github.com/OceanEilonwy/monokulo-lws), a light wallet
server, scans the chain for every registered view key with it.

Each caller passes its own threshold for Auto as `gpu_from` to
`Prefilter::flag`: below it the CPU's cores do the work, from it the GPU.

## Testing

```sh
cargo test --workspace
```

Tests that need a GPU are ignored by default. To run them as well:

```sh
cargo test --workspace -- --include-ignored
```

`examples/scan_speed.rs` times the engines on a fixture of real mainnet
blocks, 2,000 from height 3,797,000, written by `prototypes/scan-bench`:

```sh
cargo run --release -p monokulo-8ar --example scan_speed -- \
    prototypes/fixtures/mainnet-3797000-2000.bin
```

Each run prints the XOR of all derivations, `e0c4c620` for this fixture,
whichever engine did the work. `--cpu N` uses N cores instead of the GPU,
and `--tip 1,10,100` compares the energy of small catch-ups on Android (see
the example's header). `examples/gpu_check.rs` holds a GPU to the CPU's
answers at length.

## The shader

`shaders/gen_shader.py` writes `shaders/scan.comp` (GLSL), and `glslc` from
the Vulkan SDK or shaderc compiles it to `shaders/scan.spv`, which the crate
embeds. Both outputs are checked in, so building needs no shader tools. After
changing `gen_shader.py`:

```sh
crates/monokulo-8ar/shaders/build.sh
```

which runs `python3 gen_shader.py` and
`glslc --target-env=vulkan1.0 -O -o scan.spv scan.comp`. Then run the tests
with a GPU.

## Licence

GPL-3.0-only, as Fob is. See `LICENSE`.
