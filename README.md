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
Scripts/build.sh   # release CLI plus native probes, into .build/
Scripts/test.sh    # Swift suites plus native gates
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
                                                │ "who runs op X on layer L?"
                                     ┌──────────▼──────────┐
                                     │  DeviceSupervisor   │  ExecutionPlacementPlan (typed, immutable)
                                     │  admit · fault ·    │  RuntimeFaultPolicy
                                     │  evict · report     │  GuestFault ledger → profile JSON
                                     └───┬─────────┬───────┘
                       RoutedPrefillGuest│         │SharedFFNGuest / RoutedDecodeGuest
                         ┌───────────────▼──┐   ┌──▼───────────────────┐
                         │ ANE routed-expert│   │ ANE shared FFN,      │
                         │ carrier (2 banks)│   │ AOT decode experts   │
                         └──────────────────┘   └──────────────────────┘
               Metal host rail: always present, the reference, and the failover target
```

`Sources/ANEFieldfareHotpath/Runtime/Supervisor/` contains the abstraction:

- **`ExecutionPlacement.swift`**: `ExecutionPlacementPlan` maps each placeable
  operation (`routed-experts-prefill`, `shared-ffn-prefill`,
  `shared-ffn-decode`, `routed-experts-decode`) to a rail for each layer. Placement
  is resolved once, from the CLI backend and the experiment variables below.
  The forward loop never reads process environment.
- **`GuestRails.swift`**: the guest contracts. A guest replaces only the
  routed-expert or shared-FFN arithmetic, and writes into buffers the host owns.
- **`DeviceSupervisor.swift`**: owns every guest. It answers admission queries
  and applies the fault policy. An evicted guest is never admitted again; to
  reconstruct one, build a new runner. Guest construction is fail-closed: an
  explicit placement whose program or shape is unavailable aborts startup. It
  never silently falls back to Metal.

When a guest is evicted, failover happens at layer granularity. Both routed
kernels overwrite each `(token, rank)` row rather than accumulating into it, so
the host clears the routed partials and recomputes the layer. Shared-FFN
guests are checked before the host consumes their output, and a failed output
is replaced by Metal arithmetic. Under `fail-closed`, the original lock-free
overlap is unchanged.

With `ANE_FIELDFARE_HOTPATH_PROFILE=1`, the CLI writes a profile JSON to stderr. It includes a `supervisor`
section with the placement, the fault policy, evictions and every recorded
fault. It also includes a claim boundary that states whether the output mixes
ANE and Metal arithmetic.

### Performance controls

| Variable | Effect |
| --- | --- |
| `ANE_FIELDFARE_PREFILL_SHARED_GEMM=0` | Restore the per-token shared-FFN GEMV order of the original numerical reference. By default the shared FFN runs as batched INT4 tensor-op GEMMs. |
| `ANE_FIELDFARE_ANE_FETCH_LOOKAHEAD=n` | Expert groups read concurrently during ANE prefill (default 4; 1 = sequential). |
| `ANE_FIELDFARE_TIMELINE=<path>` | Write a JSON interval trace of disk reads, GPU command buffers and ANE evaluations. |

### Experiment placements

These variables select the experimental rails. Invalid values fail closed at
startup.

| Variable | Effect |
| --- | --- |
| `ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS=n` | Routed prefill on ANE for layers `0..<n`. Works with either backend. |
| `ANE_FIELDFARE_ANE_SHARED_LAYERS=n` + `ANE_FIELDFARE_ANE_SHARED_MODE=prefill\|decode\|both` | Shared FFN on ANE with immutable constants. |
| `ANE_FIELDFARE_AOT_EXPERT_ROOT=<dir>` | Decode experts through compiler-authored ANE procedures. |

## Verification

The supervisor is held to these oracles:

- **Regression parity.** With `ANE_FIELDFARE_PREFILL_SHARED_GEMM=0`, `metal` and `hybrid-ane`
  reproduce their stored pre-refactor logits exactly: 17 × 262,144 FP16 values
  each, on a 3,015-token prompt plus 16 fixed continuation tokens. The hybrid run
  executed 4,407 ANE evaluations across 30 layers, from 1 compile and 1 load.
- **Failover parity.** These tests (`DeviceSupervisorFailoverTests`) run on the
  installed checkpoint. Injected guests write NaN over the host's buffers and
  then fault:
  - Routed-prefill and shared-FFN failover reproduces pure-Metal logits bit
    for bit.
  - A real ANE guest evicted at layer 7 reproduces a plan that placed exactly
    layers 0–6 on ANE.
  - `fail-closed` aborts the request.
- **Placement.** Resolution, prefix semantics, invalid values, and eviction
  rules are covered by unit tests (`ExecutionPlacementTests`).

Not yet covered: failover of the AOT decode rail, which needs compiled AOT
artifacts.

### Prefill optimizations

The work was guided by `ANE_FIELDFARE_TIMELINE` traces of the 3,015-token prompt. These are
single screening runs of prefill wall time on an M5, not the protocol below:

| Change | `hybrid-ane` prefill | `metal` prefill | Numerics |
| --- | ---: | ---: | --- |
| Before | 22.2 s | 24.0 s | — |
| ANE carrier on its own command queue. The ANE no longer idles while the shared FFN occupies the host queue. | 19.5 s | — | Bit-exact |
| Batched shared FFN, 3 tensor-op GEMMs replacing 1,024 per-token GEMVs. Shared FFN goes from 7.1 s to 0.83 s. | 16.9 s | 18.7 s | Same FP32-reference error as the GEMV (relative L2 ≈ 3–4e-4); end-to-end mean KL 0.015 (metal) and 0.003 (hybrid); greedy 17/17 |
| Concurrent expert-read lookahead of 4. SSD throughput: 4.7 GB/s at queue depth 1, 6.7 GB/s at depth 2. | 16.2 s | — | Bit-exact |

The traces show where prefill time still goes on the hybrid path, in order of
expected payoff:
- **Disk.** About 32 GB of expert reads per prompt. Each 1,024-token chunk
  re-reads most experts; a layer-major chunk order would read each expert once
  per layer.
- **Pre-router GPU work** (attention, QKV), 4.2 s, all on the critical path.
- **Weight expansion**, about 35% redundant. An expert split across several
  64-row jobs is expanded once per job.
- **ANE row utilization**, 64%.

### Measured performance

The table below predates the optimizations above. It compares unmodified Turbo Fieldfare `4c6db1e`, this repo's Metal backend, and the
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
| `Sources/ANEFieldfareRuntime` | C/Objective-C ANE program, carrier and native reference runtime. |
| `Sources/ANEFieldfareAOTExpertRuntime` | SwiftPM shim that exposes the native ANE ABI to Swift. |
| `Sources/ANEFieldfareImportedFormat` | `.gturbo` compact checkpoint format. |
| `Sources/ANEFieldfare*Gate`, `*Probe` | Standalone API and numerical gates from the ANE investigation. `Scripts/build.sh` builds them. |
| `Tests/` | Swift Testing and XCTest suites, plus native C tests. |
| `Scripts/` | Build, test and comparison tooling. |

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
