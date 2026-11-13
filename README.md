# monokulo-8ar

Finds the Monero blocks that may pay a wallet, fast, on the GPU.

Proving that a block doesn't pay a wallet is nearly all of a scan's cost:
8·a·R for each transaction key (a is the view key, R the key), then a view
tag for each output. monero-oxide's `Scanner` works out 8aR once per output
and decompresses every output key. This crate has a GPU (Vulkan, through
`ash`, or Metal on Apple's platforms, through `objc2-metal`) work out 8aR once
per transaction key, tens of thousands of keys per
dispatch, compressed on the GPU, and only decompresses an output key past a
matching view tag.

It decides nothing about money. A block it flags goes to the
unmodified `Scanner`, which alone finds the outputs, their amounts and their
subaddresses. Its one duty is never to miss, and anything it doesn't
understand (a newer fork, missing transactions) is flagged. Its reading of
`Extra` and of bad keys is held to monero-oxide's by tests.

The prefilter runs on the GPU only. There is no CPU version of its curve
arithmetic: without a GPU, the caller scans every block with the `Scanner`,
which is slower but needs nothing from this crate. Any Vulkan GPU will do,
integrated ones included, and so will any Apple GPU through Metal, with the
`metal` feature (below). A GPU is only trusted after it passes test vectors, once per
process, and again with each wallet's own view key, and each point it hands
back is checked: one off the curve fails the batch. A GPU that fails is
retired until the process restarts. Which way a batch goes is a matter of
speed and power, never of the answer.

- `crates/monokulo-8ar`: the crate. `Prefilter` holds a wallet's view key and
  the spend keys it watches. `Prefilter::flag` takes a batch of
  `ScannableBlock`s and says which may pay it, or `None` if it didn't filter
  them: the GPU isn't ready or was retired, or the batch holds fewer than the
  caller's `min_keys` transaction keys. Then the caller scans every block.
  `Prefilter::flag_txs` says the same per transaction: for each block,
  `Txs::Clear`, `Txs::These` (0 the miner transaction, 1 on
  `block.transactions[tx - 1]`), or `Txs::All` for a block it can't vouch
  for (a newer fork, transactions missing, keys it can't read), so a caller
  can hand the `Scanner` only the transactions that may pay.
  `warm_up`, `check_wallet`, `rest` and `set_cache_dir` manage the one GPU a
  process has, and `engine()` says what's doing the work (the GPU and its
  name, or the CPU and why), for a setting to show. The compiled pipeline is
  cached as `scan-gpu.cache` in the folder given to `set_cache_dir`.
- `Keyring`: many accounts in one GPU pass, for a server. `Keyring::add(pair,
  view) -> AccountId`, `watch(account, pair, subaddress)`, and
  `flag_many(blocks, threads, min_keys) -> Option<Vec<Hit>>`, each `Hit` an
  account, a block and a transaction in it (0 the miner transaction, 1 on the
  block's others, `Hit::WHOLE_BLOCK` a block the account must have read
  whole). `None` means what it does for `flag`. `Prefilter` is a keyring of
  one. The GPU works through a grid of accounts by transaction keys that's
  never written out: each key is uploaded once, each account's digits are 64
  bytes, and a batch of 32,768 items is a slab of the grid, so GPU memory
  doesn't grow with accounts. Once the GPU has done a slab, the cores do each
  item's view tags and spend check, against that item's account's spend keys:
  one after the other, never side by side, as a scan runs on the GPU or the
  CPU and never both.
- `Arena`: one caller's share of the GPU. The process has one `gpu::Device`
  (instance, logical device, compute queues, pipelines and their cache, the
  self-test), and each arena its own buffers, command buffer and fence, about
  43 MB once used. Arenas share only a queue's lock, for the moment of a
  submit, and take queues of their own round-robin where the GPU has several.
  `Keyring::new` and `Prefilter` use the shared arena, one a process as a
  phone wants; `Keyring::with_arena(Arena::new())` gives a keyring its own,
  so a server's lanes don't wait on each other's batches.
- `metal` (the `metal` feature, built on Apple targets only): the same
  `Device` and `Arena` in Metal, for iOS and macOS, which have no Vulkan.
  `backend` is whichever a build uses: Metal on Apple's platforms with the
  feature, Vulkan everywhere else. `Prefilter` and `Keyring` don't change, and
  neither does the trust model: the same vectors, the same check of every point,
  and a GPU that fails is retired. One thing is Apple's own: iOS refuses the GPU
  to an app in the background, failing its command buffers with
  `MTLCommandBufferErrorNotPermitted`. That's `GpuErrorKind::Background`: the
  batch goes to the CPU, as for any failure, but the GPU isn't retired for it,
  so an app back in the foreground has it again. A warm-up that runs into it
  ends as if it hadn't started, and the next batch starts it again. Metal keeps
  its compiled shaders in a cache of its own, so `set_cache_dir` isn't used.
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
fixture. `--keys N` runs it through a keyring of N copies of the view key and
checks every copy's digest. On an RX 6800 XT (RADV, Ryzen 9 5950X, 15
finishing threads), warm:

| keys | blocks/s | derivations/s | account-blocks/s |
|---|---|---|---|
| 1 | 39,300 | 2.24 M | 39,300 |
| 10 | 3,960 | 2.26 M | 39,600 |
| 100 | 386 | 2.20 M | 38,600 |
| 1,000 | 38 | 2.20 M | 38,500 |

The GPU does about 2.65 million derivations a second whatever the account
count; the cores' view tags, after each slab rather than beside the next,
bring that to about 2.2 million. So a day of mainnet (720 blocks) for 1,000
accounts is about 19 s. `--tip 1,10,100` measures the energy of small catch-ups on Android
(see the example's header). `examples/gpu_check.rs` holds a GPU to
curve25519-dalek's answers at length.

### Metal

On a Mac, the same with `--features monokulo-8ar/metal`:

```sh
cargo test --workspace --features monokulo-8ar/metal -- --include-ignored
cargo run --release --features monokulo-8ar/metal --example gpu_check
cargo run --release --features monokulo-8ar/metal --example scan_speed -- \
    prototypes/fixtures/mainnet-3797000-2000.bin
```

CI does all three on a GitHub macOS runner, and builds the crate for
`aarch64-apple-ios`. The runner's GPU is the virtual machine's "Apple
Paravirtual device", which passes the vectors and gives the fixture's
digest, but is no measure of an Apple GPU's speed (warm, default tuning):

| GPU | keys | blocks/s | derivations/s |
|---|---|---|---|
| Apple Paravirtual device (CI, macOS 15) | 1 | 750 | 43,000 |
| Apple Paravirtual device (CI, macOS 15) | 10 | 74 | 42,000 |

Its first start, compiling the shader, took 14 s, and later ones about 8 s.
An iPad's numbers need `gpu_check` and `scan_speed` run on the device, which
CI can't do: they're for the owner to add here, beside the RX 6800 XT's.

## The shader

`shaders/gen_shader.py` writes `shaders/scan.comp` (GLSL), and `glslc` from
the Vulkan SDK or shaderc compiles it to `shaders/scan.spv`, which the Vulkan
backend embeds. SPIRV-Cross writes Metal's copy, `shaders/scan.metal`, from
SPIR-V compiled again without `-O`: the optimised SPIR-V turned into 6 MB of
MSL with every name stripped, and Metal's compiler optimises it anyway. All
three are checked in, so building needs no shader tools. After changing
`gen_shader.py`:

```sh
crates/monokulo-8ar/shaders/build.sh
```

with `glslc` and `spirv-cross` from the Vulkan SDK on the path. CI runs it with
the SDK version named in `.github/workflows/ci.yml` and fails if any of the
three differs from what's checked in, and a test holds `metal.rs`'s buffer
numbers to the ones SPIRV-Cross gave `scan.metal`. Then run the tests with a
GPU, on Vulkan and on Metal.

## Licence

GPL-3.0-only, as Fob is. See `LICENSE`.
