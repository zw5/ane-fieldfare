# ANE Fieldfare

ANE Fieldfare runs Gemma 4 26B-A4B (a 30-layer, 128-expert, top-8 MoE) on
Apple Silicon. Experts stream from disk with bounded memory. The model runs on
Metal, with Apple Neural Engine (ANE) placements as guests.

Metal is the host rail. It owns attention, routing, layer tails, logits and
decode, and holds a complete reference implementation of every operation. ANE
programs are **guests**: a device supervisor places them on specific operations
and layers, monitors them, and evicts them when they fault.

> **Status: experimental research software.** The ANE rails use private,
> undocumented Apple frameworks. They may break with any macOS update and are not
> suitable for production or App Store use. ANE FP16 arithmetic differs
> numerically from the Metal reference; the hybrid path is not bit-equal to Metal.

## Requirements

- Apple Silicon Mac with a Neural Engine, 16 GiB RAM or more. Development and
  measurement used an M5 with 16 GiB.
- macOS 26 or later. Measured on macOS 27 build `26A5406e`.
- Xcode with Swift 6.2 or later. XCTest needs a full Xcode, not just the Command
  Line Tools; the test script selects one automatically.
- An installed `gemma4.gturbo` checkpoint (about 13 GB). Produce it with the
  installer from [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare).
  This repository reads that format but does not include an installer. The
  default location is `checkpoints/gturbo/gemma4.gturbo`, which git ignores.

## Build and test

```bash
Scripts/build.sh   # release CLI and logits oracle, into .build/
Scripts/test.sh    # every Swift Testing and XCTest suite
```

Most tests use synthetic fixtures. Some exercise the real ANE and need
ANE hardware. Tests that need the installed checkpoint run when it is present
and are skipped otherwise. Set `ANE_FIELDFARE_TEST_CHECKPOINT=/path/to/gemma4.gturbo`
to use a checkpoint outside the default location.

## Run

```bash
.build/swift-hotpath/release/ANEFieldfareHotpathCLI \
  --model checkpoints/gturbo/gemma4.gturbo \
  --chat-prompt "What is the capital of France? Answer briefly." \
  --backend hybrid-ane --max-new 64 --temperature 0
```

`--help` lists every option. The ones that control placement:

| Option | Meaning |
| --- | --- |
| `--backend metal` | Everything on Metal. This is the default. |
| `--backend hybrid-ane` | Routed top-8 experts on the ANE during prefill, all 30 layers. Requires exact top-8 routing and no expert cap. |
| `--fault-policy fail-closed` | A guest fault aborts the request. This is the default. |
| `--fault-policy failover-metal` | A guest fault evicts that guest. The layer, and every later one, re-executes on Metal. |

## Architecture

```
                 ┌────────────────────────── RealForwardRunner ──────────────────────────┐
  tokens ──────▶ │ embed → [ attention → router → shared FFN → routed experts → tail ]×30 │ ──▶ logits
                 └──────────────────────────────┬─────────────────────────────────────────┘
                                                │ "does the ANE guest run layer L?"
                                     ┌──────────▼──────────┐
                                     │  DeviceSupervisor   │  ANE layer count (typed, immutable)
                                     │  admit · fault ·    │  RuntimeFaultPolicy
                                     │  evict · report     │  GuestFault ledger → profile JSON
                                     └──────────┬──────────┘
                                                │ RoutedPrefillGuest
                                      ┌─────────▼──────────┐
                                      │ ANE routed-expert  │
                                      │ carrier (2 banks)  │
                                      └────────────────────┘
               Metal host rail: always present, the reference, and the failover target
```

`Sources/ANEFieldfareHotpath/Runtime/Supervisor/DeviceSupervisor.swift` holds the
abstraction. The supervisor owns the ANE guest and answers, per layer, whether the
guest runs routed prefill; the forward loop never holds the guest directly. Placement
is resolved once, from the CLI backend. An evicted guest is never admitted again; to
reconstruct one, build a new runner. Guest construction is fail-closed: a requested
placement whose program or shape is unavailable aborts startup instead of silently
running on Metal. `Sources/ANEFieldfareANECarrier` is the small Objective-C layer over
the private ANE runtime that compiles the expert program and maps its two banks.

When a guest is evicted, failover happens at layer granularity. Both routed
kernels overwrite each `(token, rank)` row rather than accumulating into it, so
the host clears the routed partials and recomputes the layer.

With `ANE_FIELDFARE_HOTPATH_PROFILE=1`, the CLI writes a profile JSON to stderr. It includes a `supervisor`
section with the placement, the fault policy, evictions and every recorded
fault. It also includes a claim boundary that states whether the output mixes
ANE and Metal arithmetic.

`ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS=n` places routed prefill on ANE for layers
`0..<n` only, with either backend. Invalid values fail closed at startup.

## Verification

The supervisor is held to these oracles:

- **Regression parity.** Every refactor step is checked against stored logits:
  `metal` and `hybrid-ane` must reproduce them exactly, 17 × 262,144 FP16 values
  each, on a 3,015-token prompt plus 16 fixed continuation tokens. The hybrid run
  uses 1,024-token prefill chunks and executes 3,163 ANE evaluations across 30
  layers, from 1 compile and 1 load.
- **Failover parity.** These tests (`DeviceSupervisorFailoverTests`) run on the
  installed checkpoint. Injected guests write NaN over the host's buffers and
  then fault:
  - Routed-prefill failover reproduces pure-Metal logits bit for bit.
  - A real ANE guest evicted at layer 7 reproduces a plan that placed exactly
    layers 0–6 on ANE.
  - `fail-closed` aborts the request.
- **Placement.** Resolution, invalid values, and eviction rules are covered by
  unit tests (`DeviceSupervisorPolicyTests`).

### Prefill optimizations

The work was guided by interval traces of disk reads, GPU command buffers and ANE
evaluations on the 3,015-token prompt. These are single screening runs of prefill
wall time on an M5, not the protocol below:

| Change | `hybrid-ane` prefill | `metal` prefill | Numerics |
| --- | ---: | ---: | --- |
| Before | 22.2 s | 24.0 s | — |
| ANE carrier on its own command queue. The ANE no longer idles while the shared FFN occupies the host queue. | 19.5 s | — | Bit-exact |
| Batched shared FFN, 3 tensor-op GEMMs replacing 1,024 per-token GEMVs. Shared FFN goes from 7.1 s to 0.83 s. | 16.9 s | 18.7 s | Same FP32-reference error as the GEMV (relative L2 ≈ 3–4e-4; `PrefillSharedExpertBatchedTests`); against the previous logits, mean KL 0.015 (metal) and 0.003 (hybrid); greedy 17/17 |
| Concurrent expert-read lookahead of 4. SSD throughput: 4.7 GB/s at queue depth 1, 6.7 GB/s at depth 2. | 16.2 s | — | Bit-exact |
| Prefill chunks up to 4,096 tokens, chosen per prompt (`--prefill-chunk-tokens auto`, now the default). Each layer's experts are read once per prompt: 32.1 GB → 11.7 GB of reads. | 11.7 s | 15.3 s | Bit-exact |
| ANE program with 128 activation lanes per expert instead of 64. Each request streams four experts' weights (about 0.95 ms), so more rows per request halve the evaluations: 3,352 → 1,980. | 8.8 s | — | Bit-exact |

Short prompts were dominated by integrity checking rather than compute: the CLI
hashed all 13 GB of expert files on every launch. It now trusts the installer's
verified receipt when it matches the install, and `--verify-weights` restores the
full SHA-256 check. Prefill on the 28-token prompt went from 6.3 s to 1.6 s, and on
the 430-token prompt from 6.7 s to 2.2 s.

Measured and rejected: running the two halves of a chunk one stage apart, so the
GPU's attention overlaps the ANE's experts, took 13.4 s against 9.2 s. Each ANE
request costs the GPU a weight expansion (the ANE cannot dequantize INT4 itself),
so overlapping saturates the GPU, and the halves re-read and re-expand the same
experts. A third request bank made no measurable difference.

Remaining prefill time on the 3,015-token prompt (8.8 s): pre-router GPU work
(attention and projections) 4.2 s, then the routed experts 4.4 s, of which the ANE
evaluates for 3.5 s.

### Decode

Turbo Fieldfare's community protocol (`docs/COMMUNITY_BENCHMARKS.md` upstream:
temperature 0.2, top-k 64, top-p 0.95, 4,096-token context, up to 1,024 tokens,
fixed seeds), `--backend metal`, one run per case on an M5 with 16 GB. Every
case stopped at the end of the model turn.

| Case (prompt / generated tokens) | 16 slots | 48 slots | 48 slots + early read |
| --- | ---: | ---: | ---: |
| short (61 / 533) | — | 25.2 tok/s | 28.5 tok/s |
| medium (430 / 745) | — | 23.0 tok/s | 24.2 tok/s |
| long (3,015 / 613) | — | 18.9 tok/s | 20.2 tok/s |

What changed, in order:

- The expert cache defaults to 48 slots per layer instead of 16 (about 4.9 GB
  resident). A greedy 256-token code prompt went from 22.2 to 25.7 tok/s.
- Each expert read is four concurrent `pread`s, so one 3.4 MB read no longer
  runs at queue depth 1.
- Early expert read, adapted from Turbo Fieldfare PRs #187/#188: after a
  layer's own reads, the two likeliest experts of the next layer that its cache
  does not hold are read into staging buffers, scored by a fitted next-layer
  guess. A staged read the next layer misses is swapped into its slot. Here a
  layer with reads still in flight issues none, and candidates must rank in the
  guess's top 8; about 63% of early reads are used. `--early-expert-read off`
  disables it. Generated text and cache contents do not depend on it.

### Measured performance

The table below predates all of the optimizations above. It compares unmodified Turbo Fieldfare `4c6db1e`, this repo's Metal backend, and the
ANE hybrid. Setup: M5 with 16 GiB, Swift 6.4, temperature 0.2, top-k 64,
top-p 0.95, a 4,096-token context, and a 1,024-token output cap. Each value is
the mean of two measured runs after warmups, with backend order reversed between
passes. These results also predate the supervisor refactor.

| Prompt tokens | Turbo Fieldfare | Metal | ANE hybrid |
| ---: | ---: | ---: | ---: |
| 61 | 36.62 s | 37.24 s | 37.45 s |
| 430 | 56.76 s | 54.73 s | 55.77 s |
| 3,015 | 102.96 s | 71.96 s | 71.28 s |

The ANE advantage is confined to long-prompt prefill. On the 3,015-token prompt,
first visible text arrived after 24.39 s, against 26.49 s for Metal. Decode rates
were similar for all three: 13.0–13.7 tok/s.

## Repository layout

| Path | Contents |
| --- | --- |
| `Sources/ANEFieldfareHotpath` | Swift runtime: model I/O, Metal kernels, forward runner, supervisor, tokenizer. |
| `Sources/ANEFieldfareHotpathCLI(Core)` | Command-line runner. |
| `Sources/ANEFieldfareANECarrier` | Objective-C ANE expert program and its two request banks. |
| `Sources/ANEFieldfareImportedFormat` | `.gturbo` compact checkpoint format. |
| `Sources/ANEFieldfareQualityGate` | Fixed-input logits oracle used for regression parity. |
| `Tests/` | Swift Testing and XCTest suites, plus their CPU references and fixture writers. |
| `Scripts/` | Build, test and logits-comparison tooling. |

## Claim discipline

Code, profiles and test names keep these statuses distinct:

- **present**: the API exists.
- **accepted**: a call succeeds, with no oracle check.
- **validated**: the result passes an independent numerical or lifecycle
  oracle.
- **rejected**: the tested construction fails cleanly.

The ANE routed-expert path expands ordinary runtime-input weights through
Metal. It does not use prepared mutable weights or resident expert constants.

## License and credits

Apache License 2.0; see [LICENSE](LICENSE). Some parts are adapted from
[Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare) by drumih (Apache 2.0):
the Swift runtime, the compact format, the CLI and the upstream regression
tests. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[UPSTREAM.json](UPSTREAM.json) for the source mapping.

Gemma is a model family from Google; its weights are distributed under
Google's Gemma terms and are not included in this repository.
