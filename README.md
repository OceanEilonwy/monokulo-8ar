# monokulo-8ar

Finds the Monero blocks that may pay a wallet, fast, on the GPU.

Proving that a block doesn't pay a wallet is nearly all of a scan's cost:
8·a·R for each transaction key (a is the view key, R the key), then a view
tag for each output. monero-oxide's `Scanner` works out 8aR once per output
and decompresses every output key. This crate has a GPU (Vulkan, through
`ash`) work out 8aR once per transaction key, tens of thousands of keys per
dispatch, compressed on the GPU, and only decompresses an output key past a
matching view tag.

It decides nothing about money. A block it flags goes whole to the
unmodified `Scanner`, which alone finds the outputs, their amounts and their
subaddresses. Its one duty is never to miss, and anything it doesn't
understand (a newer fork, missing transactions) is flagged. Its reading of
`Extra` and of bad keys is held to monero-oxide's by tests.

The prefilter runs on the GPU only. There is no CPU version of its curve
arithmetic: without a GPU, the caller scans every block with the `Scanner`,
which is slower but needs nothing from this crate. Any Vulkan GPU will do,
integrated ones included; Apple's GPUs are meant to come later through
MoltenVK. A GPU is only trusted after it passes test vectors, once per
process, and again with each wallet's own view key. A GPU that fails is
retired until the process restarts. Which way a batch goes is a matter of
speed and power, never of the answer.

- `crates/monokulo-8ar`: the crate. `Prefilter` holds a wallet's view key and
  the spend keys it watches. `Prefilter::flag` takes a batch of
  `ScannableBlock`s and says which may pay it, or `None` if it didn't filter
  them: the GPU isn't ready or was retired, or the batch holds fewer than the
  caller's `min_keys` transaction keys. Then the caller scans every block.
  `warm_up`, `check_wallet`, `rest` and `set_cache_dir` manage the one GPU a
  process has, and `engine()` says what's doing the work (the GPU and its
  name, or the CPU and why), for a setting to show. The compiled pipeline is
  cached as `scan-gpu.cache` in the folder given to `set_cache_dir`.
- `prototypes/fixtures`: 2,000 real mainnet blocks' transaction keys and view
  tags, from height 3,797,000, for `examples/scan_speed.rs`. The prototypes
  that wrote them are in the git history.

## Who uses it

[Fob](https://github.com/OceanEilonwy/monokulo-fob), a Monero wallet for
phones and the desktop, where it was written: `fob-scanner` wraps it with
Fob's thread fan-out over the blocks the `Scanner` reads, and passes 512 as
`min_keys` (about nine blocks: a GPU batch has a fixed cost however small,
46 mJ on a Mali-G715 and 3.7 ms on an RX 6800 XT). Fob's `docs/decisions.md` gives the reasons
(decisions 331, 332, 350, 351, 359 and 360).

[monokulo-lws](https://github.com/OceanEilonwy/monokulo-lws), a light wallet
server, scans the chain for every registered view key with it.

## Testing

```sh
cargo test --workspace
```

Without a GPU, the tests hold the prefilter to monero-oxide's `Scanner` with
curve25519-dalek doing the GPU's part. Tests that need a GPU are ignored by
default. To run them as well:

```sh
cargo test --workspace -- --include-ignored
```

`examples/scan_speed.rs` times the GPU on the fixture:

```sh
cargo run --release -p monokulo-8ar --example scan_speed -- \
    prototypes/fixtures/mainnet-3797000-2000.bin
```

Each run prints the XOR of all derivations, which must be `e0c4c620` for this
fixture. `--tip 1,10,100` measures the energy of small catch-ups on Android
(see the example's header). `examples/gpu_check.rs` holds a GPU to
curve25519-dalek's answers at length.

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
