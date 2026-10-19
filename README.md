# ANE Fieldfare

ANE Fieldfare runs Gemma 4 26B-A4B (a 30-layer, 128-expert, top-8 MoE) on
Apple Silicon. Routed experts are bound to Metal straight out of the OS page
cache, so the expert cache is whatever memory is free. The model runs on
Metal, with Apple Neural Engine (ANE) placements as guests, and can decode
speculatively with a DFlash drafter.

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

## Benchmark loop

`ANEFieldfareBench` loads the model once and runs a fixed suite. Each case runs
the CLI's own greedy generation loop with a fresh KV cache, plain and with
`--drafter`. It then times a decode step and verifies of 1, 2, 4 and 8 rows at
a fixed context, and writes raw samples plus per-case counters (GPU time by
phase, storage wait, expert hits and misses, speculation steps) to JSON. The
quick suite (3 prompts, 192 tokens, 2 reps) takes about 90 s with the drafter.

```bash
Scripts/bench/snapshot.sh base        # freeze the current build
# ...change something...
Scripts/bench/snapshot.sh try
Scripts/bench/ab.sh base try 2 --drafter checkpoints/dflash/gemma-4-26B-A4B-it-DFlash
```

`ab.sh` runs the two snapshots in ABBA-ordered rounds, each in its own
process, so drift lands on both sides. It then calls `Scripts/bench/compare.py`,
which prints a summary and writes a self-contained HTML report under
`.build/bench-runs/`. The report covers:

- each case's throughput and each step's latency as a B/A ratio with a
  bootstrap 95% interval;
- a geometric-mean headline;
- the per-token decode breakdown;
- speculation statistics;
- token agreement with A's plain greedy output.

A change is reported only if its interval excludes 1 *and* it clears the A/A
noise floor, the largest spread between two of A's own rounds. This follows
vLLM PR #61867, which judged accuracy against a 1-ulp-perturbation floor.

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
| `--drafter <dir>` | Speculative greedy decoding with a DFlash drafter, e.g. `z-lab/gemma-4-26B-A4B-it-DFlash`. Requires `--temperature 0` and a text prompt. See [Speculative decoding](#speculative-decoding). |

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

### Expert residency

Every routed expert has one fixed, zero-copy `MTLBuffer` over its page-aligned
slice of a read-only mapping of its layer file
(`Infrastructure/Streaming/ExpertStore.swift`). Nothing is copied and nothing
is evicted by the runtime: the kernel's unified buffer cache is the expert
cache, it grows to whatever memory is free, and the OS reclaims it under
pressure. A forward pass binds every routed expert at once. It asks `mincore`
which are absent (8 sampled pages per expert: a full 205-page check costs
54 µs, and decode asks 240 times a token). Kernels for resident experts run
while the absent ones are read with concurrent `pread`s. Correctness never
depends on residency, because the GPU faults in any page that is still absent.

Early reads are `F_RDADVISE` advisories: after a layer's own reads, the
likeliest absent experts of the next layer, scored by a fitted next-layer
guess, are queued to storage and nothing waits for them. Decode advises 2 per
layer. A verify block advises each row's guessed top 8. Long prefill chunks
advise nothing, because there the advice competes with the layer's own reads.

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

Single greedy runs on an M5 with 16 GB, warm, alternating old and new
binaries. The old runtime held a 48-slot-per-layer app cache (4.9 GB); the
new one has no app cache. Generated text is byte-identical.

| Case (prompt / generated tokens) | 48-slot cache | page cache |
| --- | ---: | ---: |
| code (35 / 256) | 26.6 tok/s | 30.6 tok/s |
| essay (31 / 512), two runs | 26.7, 27.1 tok/s | 31.3, 30.2 tok/s |
| long, after a 4,033-token prefill (24), `metal`, two runs | 12.1, 13.7 tok/s | 14.7, 15.5 tok/s |
| long, after a 4,033-token prefill (24), `hybrid-ane`, two runs | 15.4, 15.6 tok/s | 16.8, 18.1 tok/s |

Prefill time is unchanged on the 4,033-token prompt: 20.3 vs 20.2 s on Metal,
11.2 vs 11.4 s hybrid. On the code case, misses fall from 21 to 3.3 per token. The
page cache held 55–64 experts per layer, and the advisory early read is used
59–62% of the time. `--early-expert-read off` disables it.

#### GPU-routed decode

The host no longer sits between a layer's router and its experts. While the
GPU runs layer L, the host encodes layer L+1's attention and shared FFN. Once
layer L's routes are known, it binds the resident experts among the fitted
next-layer guess's best 24 and commits layer L+1's routed experts unrouted. On
the GPU, a gather kernel reads the router's top 8 and points each routed slot
at its expert through a per-layer address table. A slot whose expert is not
bound reads a zeroed expert, and the host patches that layer. It restores the
saved layer input, recomputes only the unbound slots' gate/up/GELU, and reruns
the down projection and tail. The result is bitwise the host-routed one. Layer
0 always takes the host path. Further changes:

- One dispatch for the shared FFN's gate, up and GELU. The intermediate is
  rounded to FP16 as before, so the result stays bitwise.
- The router and the next-layer guess share one GEMV dispatch.
- Top-8 select is a SIMD-wide argmax instead of one serial thread (12 → 1.8 µs).

Decode step, ABBA A/B on M5 16 GB: 29.3 → 23.9 ms for GPU routing (33 → ~42
tok/s), then −5% for the three changes above. Output is byte-identical
throughout. Per token, the GPU now reads about 2.2 GB, whose floor at this
machine's measured ~132 GB/s is 16.4 ms. The largest remaining costs are the
routed experts (7.4 ms), QKV (4.0 ms), the head (3.9 ms) and the shared FFN
(3.2 ms). `ANE_FIELDFARE_KERNEL_PROFILE=1` prints that per-kernel breakdown.

#### Hot-vocabulary head

Plain greedy decoding reads the 262k-row head once per 4 tokens instead of
once per token, with output bitwise equal to the full head's
(`Runtime/Generation/HotHeadDecoder.swift`).

- **Guess.** Each step scores only the hot vocabulary: 56k token IDs shipped
  in `HotVocabulary/vocab-262144.bin`, plus every prompt token and every token a
  check found outside the list. The kernel reads those rows where they sit in
  the head, so the list costs 4 bytes per token and no copy (0.37 ms → 0.75
  ms GPU for 32k → 56k rows, against 3.3 ms for the full head). The step feeds
  its guess to the next step.
- **Check.** The fourth step skips the guess and runs the full head over all
  four saved normalized rows in one pass, 2 weight rows per SIMD group (3.4 ms
  for 4 rows). Each row's logits repeat the single-row head operation for
  operation, so each token is the one plain decoding would have produced
  (tested bitwise, ties included). Guesses up to the first wrong one are
  kept; the wrong one is replaced by the checked token and the KV rows after
  it are rewound.
- **Misses.** On the quick suite, 0–3 guesses in 192 tokens were wrong; on
  the chat and essay prompts, 5–7. A miss costs the steps after it in its block.
  The list was ranked from ShareGPT English answers and Python sources
  (`Scripts/hotvocab/build.py`). With 28k tokens the essay and chat prompts
  missed about 5% of the time, so the list grew to 56k.

The CLI uses it for greedy text decoding with chunked prefill;
`ANE_FIELDFARE_HOT_HEAD=0` turns it off in the bench. A list for another
vocabulary only changes speed: every token is checked against the full head.

### Speculative decoding

`--drafter` runs greedy DFlash speculative decoding through the ordinary
generation loop (`Runtime/DFlash/SpeculativeDecoder.swift`). On a streamed MoE
target the binding cost is verification, not drafting. Each verified row pulls
its own experts: an 8-row block touches about 29 experts per layer instead of
8, and rejected rows pay in full. The scheduler (`SpeculationPolicy`) therefore
prices rows in measured wall time.

- **Draft length.** It fits a per-draft acceptance rate from recent outcomes
  (DISCO-style) and drafts the block length (2–16) that maximizes expected tokens
  per second. Or it drafts nothing: a plain decode step still feeds the drafter's
  context. While plain decoding wins, it retries a 4-row block with exponential
  back-off.
- **Verify length.** After drafting, it truncates DSpark-style. The head returns
  each draft's softmax probability under the target's softcap. A per-decile
  calibration learned from verify outcomes turns that into P(accept), and the
  scheduler verifies the prefix whose expected tokens, net of verify time at the
  current throughput, is largest. A block of unlikely drafts is not verified at
  all.

Measured step cost at a fixed context, M5 16 GB (median of 34). The last
column is `Scripts/bench/ab.sh`, ABBA-ordered against the build before the
verify work below:

| Rows | First | Second | Now | Change that did it |
| ---: | ---: | ---: | ---: | --- |
| plain decode | 28 ms | 28 ms | 26 ms | |
| 1 | 49.5 ms | 32.8 ms | 29.3 ms | pipelined layer tails; GEMV specialized by row count |
| 2 | | 44.0 ms | 36.7 ms | |
| 4 | 66.8 ms | 44.8 ms | 43.3 ms | as above; 2 weight rows per SIMD group at 3–4 rows |
| 8 | 89.3 ms | 77.2 ms | 56.3 ms | SIMD-group matrix GEMV; verify routed kernels |

What made verification affordable:

- **Pipelining.** Verify no longer waits on each layer's tail; the next layer's
  attention orders it, as in decode. This saves 30 host round trips per verify.
- **One multi-row INT4 GEMV for every small-batch projection.** Verify
  projections, the shared FFN, the drafter and both greedy heads use it.
  Pipelines are specialized by row count and width. Through 4 rows it is an ALU
  kernel whose row r is bitwise the single-row GEMV's result (tested), at the
  weight-bandwidth floor; at 3–4 rows each SIMD group computes 2 weight rows. At
  1 row, QKV went from 6.9 to 3.8 ms, O from 3.7 to 2.0 ms and the shared FFN
  from 5.6 to 3.0 ms. Past 4 rows that kernel turns ALU-bound (8 rows cost 2×, 16
  rows 4×), so 5–16 rows run on the SIMD-group matrix units instead: each weight
  is dequantized once, as q·s + b in FP32, into an 8×8 fragment that one multiply
  applies to 8 rows. It matches the reference to FP32 rounding, not bitwise
  (tested). The 262k-row head at 8 rows went from 7.0 to 3.3 ms (125 GB/s).
- **Greedy head.** The multi-row head became that GEMV into FP32 logits plus a
  per-row argmax, which also yields the draft's probability. The tiled kernel it
  replaced cost ~3 ms per row.
- **Verify routed kernels.** One SIMD group per expert weight row applies it to
  every row routed to that expert, so a block reads each expert of the union
  once. The batched kernels they replace gave each (weight row, row) pair its
  own thread. 8-row routed time fell from ~37 to ~28 ms.
- **Decode's kernels per row.** Verify routes with decode's router and its
  next-layer guess in one dispatch, with a SIMD-group top-8 select per row
  (1 thread per row selected serially before; −3.4 ms a verify). Q/K/V
  normalization and RoPE run as decode's fused epilogue with one grid row per
  token (5 dispatches before; −2 ms at 8 rows).
- **Host critical path.** The shared FFN is queued behind attention, before the
  host waits, so it runs while routes are planned. The host spins on the
  router's event instead of sleeping. Residency checks probe 2 pages per expert
  instead of 8 (−2.7 ms a verify), and next-layer read advice is issued from a
  background queue (−3.8 ms).
- **Early reads for verify rows.** Each verify row's guessed next-layer experts
  are advised ahead, which halves verify misses (93 → 45 per 8-row step).
- **INT4 drafter.** The drafter's projections are quantized at load to the
  target's INT4 format: ~860 MB → ~360 MB resident, with acceptance about
  unchanged (τ 4.9 vs 5.1 at block 8).
- **Pruned draft head.** The drafter scores only a 16k-token hot vocabulary
  (the plain decoder's list, cut down) plus the prompt's and every accepted
  token, through the SIMD-group matrix GEMV reading listed head rows in place.
  A draft outside it is just a rejected draft, so nothing is checked. With 8k
  to 56k listed tokens, acceptance did not change beyond noise.
- **Draft projections on the matrix units.** The draft step's BF16
  activations used to force the ALU kernel, which is ALU-bound at 8 rows; the
  matrix kernel now takes BF16 in and out. With the pruned head, an 8-row draft
  went from ~8.6 to ~4.6 ms.
- **Context append.** Appends of up to 16 rows (after a verify or plain
  steps) read INT4 copies of `fc` and the K/V projections instead of BF16
  (~150 → ~43 MB), and the append is committed without a wait: the next
  propose, on the same queue, runs after it. Host time per step fell from
  2.6 ms to ~0.
- **Verify tweaks.** The shared FFN's down projection (K = 2112) now runs on
  the matrix kernel, which takes a final 64-wide group as a half step; a
  verify no longer zero-fills routed partials it overwrites; and experts the
  previous layer's early read found resident are not probed again (0.63 →
  0.19 ms of host time per verify). None of these moved the step timings
  beyond noise.

Measured and rejected:

- Decode's router per verify row, one dispatch per row: −1 ms at 1 row, +5 ms
  at 8 rows. (One dispatch for all rows, above, is what replaced it.)
- An FP32 input pre-pass for the multi-row GEMV: slower.
- 2–4 weight rows per SIMD group past 4 rows: up to 2× slower.
- The first SIMD-group matrix layouts: one 8-byte weight load per lane per
  group was 2× slower than the ALU kernel, and 16 live fragments per lane
  spilled registers (5× slower). The kernel now reads 64 contiguous bytes per
  weight row per step and builds each fragment just before its multiply.
- Advising every next-layer expert during long prefill: no gain on Metal, 16%
  slower on `hybrid-ane`.
- An ANE drafter, from the numbers: the draft is 8–15 ms of a 50–95 ms step and
  cannot overlap the verify it depends on, and the ANE would stream ~860 MB of
  FP16 weights at ~70 GB/s, slower than the GPU's INT4 drafter.
- Faster 8-row matrix GEMV. At 8 rows it runs at ~105 GB/s and at 16 rows half
  that: it is bound by FP32 multiply throughput (~3 TFLOPS), not by memory.
  Half-precision fragments gave the same time, and so did rearranging the x
  loads so that none is wasted (kept, since it is no slower). A tensor-unit
  version (`matmul2d`, weights dequantized to FP16 in threadgroup memory, or
  fed as `uint4b_format` with the affine terms applied per group) ran at a
  flat ~75 GB/s for 4–16 rows: slower at 8 rows, faster only at 16.
- Binding more of the guess in GPU-routed decode: the top 48 gave the same
  speed as the top 24; all 128 cut decode to ~30 tok/s.

A verify of 8 rows is ~50 ms, and the GPU is busy for all but ~5–6 ms of it
(command buffer timestamps): routed experts ~26.7 ms, QKV 4.5, the shared FFN
4.6, the head 4.2, O 2.6, the router 1.8, attention 1.4. Most of the idle,
~100 µs a layer at any row count, is the round trip from the router's event
to the host and back to a committed tile, which outlasts the shared FFN queued
to cover it; the rest waits on expert reads. Closing it would take decode's
GPU-side routing, which needs a bound superset of the union's experts.

End to end, current build against the build before the hot-vocabulary head
and the drafter changes above: `Scripts/bench/ab.sh` quick suite (192 tokens,
two ABBA rounds of 2 reps), M5 16 GB with few apps running; the A/A noise floor
was 2.8%.

| Prompt | plain before | plain now | `--drafter` before | `--drafter` now |
| --- | ---: | ---: | ---: | ---: |
| code | 46.3 tok/s | 49.2 tok/s (+6%) | 73.8 tok/s | 89.4 tok/s (+21%) |
| math word problem | 45.2 tok/s | 48.0 tok/s (+6%) | 62.6 tok/s | 66.4 tok/s (+6%) |
| explanation | 45.5 tok/s | 48.1 tok/s (+6%) | 58.2 tok/s | 65.0 tok/s (+12%) |

Plain output was byte-identical between the builds. Speculative output first
differed from plain greedy at token 90 on math, in both builds.

The verify work before that, same suite but with desktop apps running (A/A
noise floor 3%); "before" is the build preceding it:

| Prompt | plain | `--drafter` before | `--drafter` now |
| --- | ---: | ---: | ---: |
| code | 39.5 tok/s | 55.1 tok/s | 71.8 tok/s (+30%) |
| math word problem | 38.9 tok/s | 48.1 tok/s | 57.5 tok/s (+20%) |
| explanation | 38.0 tok/s | 41.9 tok/s | 51.6 tok/s (+23%) |

Plain decode output was byte-identical to the earlier build. Speculative output
first differed from plain greedy at token 90 on math in one run, before and
after, as below.

Earlier end-to-end runs: greedy, 384 tokens, three alternating runs per cell:

| Prompt | plain | `--drafter` | Speedup |
| --- | ---: | ---: | ---: |
| code | 32.9 tok/s | 56.2 tok/s | 1.71× |
| math word problem | 32.9 tok/s | 53.1 tok/s | 1.61× |
| explanation | 32.9 tok/s | 42.7 tok/s | 1.30× |
| travel chat | 33.4 tok/s | 37.0 tok/s | 1.11× |
| essay | 33.0 tok/s | 35.7 tok/s | 1.08× |

Plain decode output was identical across all runs, and to the build before
these changes.

Claim boundary: speculative output equals plain greedy decoding only up to the
numerical difference between a batched verify chunk and single-row decode. The
first differing token was at 356 (code), 90 (math), 256 (essay), and in one of
three explanation runs at 331; each was a near-tie between two plausible
continuations. Because the scheduler's choices depend on measured time, which
block shapes run, and so how near-ties resolve, can differ from run to run.

### Measured performance

The table below predates all of the optimizations above, including the page
cache and speculative decoding. It compares unmodified Turbo Fieldfare `4c6db1e`, this repo's Metal backend, and the
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
| `Sources/ANEFieldfareHotpath` | Swift runtime: model I/O, page-cache expert store, Metal kernels, forward runner, supervisor, DFlash speculative decoding, tokenizer. |
| `Sources/ANEFieldfareHotpathCLI(Core)` | Command-line runner. |
| `Sources/ANEFieldfareANECarrier` | Objective-C ANE expert program and its two request banks. |
| `Sources/ANEFieldfareImportedFormat` | `.gturbo` compact checkpoint format. |
| `Sources/ANEFieldfareQualityGate` | Fixed-input logits oracle used for regression parity. |
| `Tests/` | Swift Testing and XCTest suites, plus their CPU references and fixture writers. |
| `Sources/ANEFieldfareBench` | Benchmark harness for the A/B loop (`Scripts/bench/`). |
| `Scripts/` | Build, test, benchmark and logits-comparison tooling. |

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
