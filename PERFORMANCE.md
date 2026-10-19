# Performance log

How the runtime got to its current speed, change by change, with what was
measured and what was tried and rejected. All measurements are on an M5 with
16 GB unless stated. Terms are defined in the [README glossary](README.md#glossary).

"Bit-exact" or "byte-identical" means the change did not alter the output.

## Prefill (processing the prompt)

Guided by timelines of disk reads, GPU work and ANE calls on a 3,015-token
prompt. Each number is a single run of prefill time, not the A/B protocol.

| Change | `hybrid-ane` | `metal` | Effect on output |
| --- | ---: | ---: | --- |
| Starting point | 22.2 s | 24.0 s | — |
| The ANE wrapper got its own GPU command queue, so the ANE no longer sat idle while the shared FFN held the main queue. | 19.5 s | — | Bit-exact |
| The shared FFN over a whole chunk became 3 matrix multiplies instead of 1,024 per-token matrix × vector calls (7.1 s → 0.83 s). | 16.9 s | 18.7 s | Same error against an FP32 reference as before (relative L2 ≈ 3–4e-4; `PrefillSharedExpertBatchedTests`). Against the previous build's logits: mean KL 0.015 (`metal`), 0.003 (`hybrid-ane`); greedy tokens 17/17 equal. |
| Read 4 experts from the SSD concurrently instead of 1 (SSD: 4.7 GB/s at 1 outstanding read, 6.7 GB/s at 2). | 16.2 s | — | Bit-exact |
| Chunks of up to 4,096 prompt tokens, sized per prompt (`--prefill-chunk-tokens auto`, now the default). Each chunk re-reads every expert it uses, so fewer chunks read less: 32.1 GB → 11.7 GB. | 11.7 s | 15.3 s | Bit-exact |
| The ANE program handles 128 tokens per expert per call instead of 64. Each call has to stream 4 experts' weights (about 0.95 ms), so twice the tokens per call halves the calls: 3,352 → 1,980. | 8.8 s | — | Bit-exact |

Short prompts were dominated by checking file hashes, not by compute: the CLI
hashed all 13 GB of expert files at every launch. Without that, prefill took
1.6 s instead of 6.3 s on a 28-token prompt and 2.2 s instead of 6.7 s on a
430-token prompt. Loading now only checks file sizes against the manifest.

Remaining prefill time on the 3,015-token prompt (8.8 s): 4.2 s of GPU work
before the router (attention and projections), then 4.4 s of routed experts, of
which the ANE computes for 3.5 s.

Tried and rejected:

- Splitting each chunk in two and running the halves one stage apart, so the
  GPU's attention overlaps the ANE's experts: 13.4 s against 9.2 s. Every ANE
  call needs the GPU to expand the INT4 weights to FP16 first, so overlapping
  overloads the GPU, and both halves re-read and re-expand the same experts.
- A third ANE buffer set (bank): no measurable difference.
- Early reads of every next-layer expert during long prefill: no gain on
  `metal`, 16% slower on `hybrid-ane`, because the early reads compete with the
  layer's own reads.

## Decode: the OS file cache as the expert cache

The runtime used to keep its own cache of 48 experts per layer (4.9 GB). It now
has none and relies on the OS file cache. Single greedy runs, warm, alternating
old and new builds; generated text is byte-identical.

| Case (prompt / generated tokens) | Own 48-slot cache | OS file cache |
| --- | ---: | ---: |
| code (35 / 256) | 26.6 tok/s | 30.6 tok/s |
| essay (31 / 512), two runs | 26.7, 27.1 tok/s | 31.3, 30.2 tok/s |
| after a 4,033-token prompt (24 generated), `metal`, two runs | 12.1, 13.7 tok/s | 14.7, 15.5 tok/s |
| after a 4,033-token prompt (24 generated), `hybrid-ane`, two runs | 15.4, 15.6 tok/s | 16.8, 18.1 tok/s |

Prefill time on the 4,033-token prompt did not change: 20.3 vs 20.2 s on
`metal`, 11.2 vs 11.4 s on `hybrid-ane`. On the code case, misses fell from 21
to 3.3 per token. The OS cache held 55–64 experts per layer, and the early
read was used 59–62% of the time.

Details of the residency check: it samples 8 pages per expert, because checking
all 205 pages costs 54 µs and decode checks 240 times per token. Decode issues
early reads for the 2 likeliest missing experts of the next layer.

## Decode: routing on the GPU

Before, the CPU waited for each layer's router result before it could tell the
GPU which experts to run. Now, while the GPU runs layer L, the CPU already
encodes layer L+1's attention and shared FFN. Once layer L's routes are known,
the CPU binds whichever of the next-layer guess's top 24 experts are in memory
and submits layer L+1's expert work without knowing its routes. On the GPU, a
small kernel reads the router's top 8 and points each expert slot at its
weights through a per-layer address table.

If a chosen expert was not bound, that slot reads a zeroed expert and the CPU
repairs the layer afterwards: it restores the saved layer input, recomputes
only the missing experts' part, and reruns the rest of the layer. The result is
bit-identical to CPU routing. Layer 0 always uses CPU routing.

Further changes in the same round:

- One GPU dispatch for the shared FFN's gate, up and GELU. The intermediate is
  rounded to FP16 as before, so the output is unchanged.
- The router and the next-layer guess share one dispatch.
- Top-8 selection uses all 32 threads of a SIMD group instead of one thread
  (12 → 1.8 µs).

Decode step, A/B in ABBA order: 29.3 → 23.9 ms from GPU routing (33 → ~42
tok/s), then −5% from the three changes above. Output byte-identical
throughout. Each token now reads about 2.2 GB, which at this machine's measured
~132 GB/s takes at least 16.4 ms. The largest remaining costs per token: routed
experts 7.4 ms, QKV projection 4.0 ms, head 3.9 ms, shared FFN 3.2 ms.

Tried and rejected: binding more of the guess. The top 48 gave the same speed
as the top 24; all 128 cut decode to ~30 tok/s.

## Decode: hot-vocabulary head

Plain greedy decoding reads the full 262,144-row head once per 4 tokens instead
of every token, with the same output (`Runtime/Generation/HotHeadDecoder.swift`).

- **Guess steps.** The step scores only the hot vocabulary: 56k token IDs from
  `HotVocabulary/vocab-262144.bin`, plus every prompt token and every token a
  check found outside the list. The kernel reads those rows in place in the
  head, so the list costs 4 bytes per token and no copy. GPU time is 0.75 ms
  for 56k rows (0.37 ms for 32k) against 3.3 ms for the full head.
- **Check step.** Every fourth step runs the full head over the four saved
  positions in one pass (3.4 ms for 4). Each position's scores are computed in
  exactly the same order as the one-token head, so the result is what plain
  decoding would have produced (tested bitwise, ties included). Guesses before
  the first wrong one are kept; the wrong one is replaced and the cached
  attention keys and values after it are discarded.
- **Wrong guesses.** On the quick suite, 0–3 in 192 tokens; on the chat and
  essay prompts, 5–7. A wrong guess wastes the remaining steps of its block of
  4. The list was ranked from ShareGPT English answers and Python source code
  (`Scripts/hotvocab/build.py`). With 28k tokens the essay and chat prompts
  missed about 5% of the time, so the list was grown to 56k.

The CLI uses it for greedy text decoding with chunked prefill. A list for a
different vocabulary would only change speed, since every token is checked
against the full head.

## Speculative decoding

### How the scheduler chooses block sizes

- **How many tokens to draft.** It estimates the chance that each drafted
  token is accepted from recent outcomes (after DISCO) and drafts the block
  length, 2 to 16, that gives the most expected tokens per second. Or it drafts
  nothing and takes a plain decode step, which still updates the drafter's
  context. While plain decoding is winning, it retries a 4-token block now and
  then, waiting longer between retries each time.
- **How many to verify.** After drafting, it may verify only a prefix (after
  DSpark). The head reports each drafted token's probability under the main
  model. A calibration table, learned from past verify results in ten
  probability bands, turns that into a chance of acceptance, and the scheduler
  verifies the prefix with the most expected tokens net of its verify time. If
  every draft looks unlikely, it verifies none.
- Since the [rewrite](#speculative-decoding-rebuilt), the next block is drafted
  on the GPU right behind the verify, so its length is chosen when that verify
  is queued, before its outcome is known. The prefix to verify is still chosen
  after drafting, from the drafts' probabilities.

### Verify step cost

Median of 34 at a fixed context. "Rows" is the number of tokens verified at
once. For the rewrite's numbers, see [below](#results).

| Rows | First version | Second | Third | What changed |
| ---: | ---: | ---: | ---: | --- |
| plain decode | 28 ms | 28 ms | 26 ms | |
| 1 | 49.5 ms | 32.8 ms | 29.3 ms | layer tails pipelined; matrix × vector kernel specialized by row count |
| 2 | | 44.0 ms | 36.7 ms | |
| 4 | 66.8 ms | 44.8 ms | 43.3 ms | as above; each SIMD group computes 2 weight rows at 3–4 rows |
| 8 | 89.3 ms | 77.2 ms | 56.3 ms | matrix-unit kernel; verify-specific expert kernels |

What made verifying cheaper:

- **Pipelining.** Verify no longer waits for each layer's tail; ordering on the
  GPU queue takes care of it, as in decode. That removes 30 CPU–GPU round trips
  per verify.
- **One INT4 matrix × vector kernel for every small batch.** Used by verify's
  projections, the shared FFN, the drafter and both heads, and specialized by
  row count and width.
  - Up to 4 rows it runs on ordinary GPU arithmetic, and row r is bit-identical
    to the one-row result (tested). It runs at the memory-bandwidth limit. At 3–4
    rows each SIMD group computes 2 weight rows. At 1 row this took QKV from 6.9
    to 3.8 ms, O from 3.7 to 2.0 ms and the shared FFN from 5.6 to 3.0 ms.
  - Past 4 rows that kernel becomes arithmetic-bound (8 rows cost 2×, 16 rows
    4×), so 5–16 rows use the GPU's matrix units instead. Each weight is
    unpacked once (as q·s + b in FP32) into an 8×8 tile that one multiply
    applies to 8 rows. It matches the reference within FP32 rounding, not
    bitwise (tested). The head at 8 rows went from 7.0 to 3.3 ms (125 GB/s).
- **Greedy head.** The multi-row head is now that kernel into FP32 scores plus
  an argmax per row, which also gives each draft's probability. The kernel it
  replaced cost ~3 ms per row.
- **Verify expert kernels.** One SIMD group per expert weight row applies it to
  every verify row routed to that expert, so a block reads each expert it uses
  only once. The kernels they replaced gave each (weight row, token) pair its
  own thread. 8-row expert time fell from ~37 to ~28 ms.
- **Decode's kernels, one grid row per token.** Verify routes with decode's
  router and next-layer guess in one dispatch, with a SIMD-group top-8 per
  token (before: one thread per token, serially; −3.4 ms per verify). Q/K/V
  normalization and rotary embedding run as decode's fused kernel with one grid
  row per token (5 dispatches before; −2 ms at 8 rows).
- **Less CPU waiting.** The shared FFN is queued behind attention before the
  CPU waits, so it runs while routes are planned. The CPU spins on the router's
  completion event instead of sleeping. Residency checks sample 2 pages per
  expert instead of 8 (−2.7 ms per verify), and next-layer early reads are
  issued from a background queue (−3.8 ms).
- **Early reads for verify rows.** Each verify row's guessed next-layer
  experts are requested ahead, which halves verify misses (93 → 45 per 8-row
  step).
- **INT4 drafter.** The drafter's projections are quantized at load to the
  main model's INT4 format: ~860 MB → ~360 MB in memory, with acceptance about
  the same (τ 4.9 vs 5.1 at block 8).
- **Smaller draft head.** The drafter scores only a 16k-token hot vocabulary
  (the plain decoder's list, cut down) plus the prompt's tokens and every
  accepted token, reading those head rows in place with the matrix-unit kernel.
  A draft outside the list just becomes a rejected draft, so nothing needs
  checking. Between 8k and 56k listed tokens, acceptance did not change beyond
  noise.
- **Draft projections on the matrix units.** The drafter's BF16 activations
  used to force the ordinary-arithmetic kernel, which is slow at 8 rows. The
  matrix-unit kernel now takes BF16 in and out. With the smaller head, an 8-row
  draft went from ~8.6 to ~4.6 ms.
- **Drafter context update.** Updates of up to 16 rows (after a verify or plain
  steps) read INT4 copies of `fc` and the K/V projections instead of BF16 (~150
  → ~43 MB). They are submitted without waiting; the next draft, on the same
  queue, runs after them. CPU time per step fell from 2.6 ms to ~0.
- **Small verify changes.** The shared FFN's down projection (K = 2112) runs on
  the matrix-unit kernel, which handles a final 64-wide group as a half step; a
  verify no longer zero-fills expert outputs it overwrites; experts the previous
  layer's early read found in memory are not checked again (0.63 → 0.19 ms of
  CPU time per verify). None of these changed step timings beyond noise.

Tried and rejected:

- Decode's router run once per verify row: −1 ms at 1 row, +5 ms at 8 rows.
  (Running it once for all rows, above, replaced it.)
- An FP32 conversion pass over the inputs before the multi-row kernel: slower.
- 2–4 weight rows per SIMD group past 4 rows: up to 2× slower.
- The first matrix-unit layouts: one 8-byte weight load per thread per group
  was 2× slower than the ordinary kernel, and 16 live tiles per thread ran out
  of registers (5× slower). The kernel now reads 64 contiguous bytes per weight
  row per step and builds each tile just before its multiply.
- Running the drafter on the ANE. The draft is 8–15 ms of a 50–95 ms step and
  cannot overlap the verify that depends on it, and the ANE would have to
  stream ~860 MB of FP16 weights at ~70 GB/s, slower than the GPU's INT4
  drafter.
- A faster 8-row matrix-unit kernel. At 8 rows it runs at ~105 GB/s and at 16
  rows half that: it is limited by FP32 multiply throughput (~3 TFLOPS), not by
  memory. FP16 tiles gave the same time, and so did rearranging the input loads
  so none is wasted (kept, since it is no slower). A tensor-unit version
  (`matmul2d`, with weights unpacked to FP16 in threadgroup memory, or passed as
  `uint4b_format` with the scale and offset applied per group) ran at a flat
  ~75 GB/s for 4–16 rows: slower at 8 rows, faster only at 16.

### Where verify time went before the rewrite

An 8-row verify took ~50 ms, and the GPU was busy for all but ~5–6 ms of it
(from command buffer timestamps): routed experts ~26.7 ms, QKV 4.5, shared FFN
4.6, head 4.2, O 2.6, router 1.8, attention 1.4. Most of the idle time, ~100 µs
per layer at any row count, was the router's result travelling to the CPU and
the next expert work coming back, which took longer than the shared FFN queued
to cover it. The rest was waiting on expert reads. The rewrite below closes it.

## Speculative decoding, rebuilt

A ground-up rewrite of the drafter, the verify pass and the loop between them,
around what this machine is: unified memory, one GPU queue, and experts that
stream through the OS file cache. Code: `Runtime/DFlash/` and
`RealForwardRunner.verifyBlock`.

### What the old pipeline spent its time on

From the GPU trace of the quick suite's code prompt, before the rewrite
(untraced timings in brackets):

- **Verify, 89% of decode time.** ~44 ms per step (41.7 ms). In an 8-row
  verify the GPU idled 5.8 ms, almost all of it in one gap per layer: from the
  end of the shared FFN to the first routed-expert tile, 130–190 µs while the
  routes went to the CPU, were grouped into tiles, probed for residency and
  bound, and the tiles came back.
- **Draft, 10%.** 5.1 ms per step (4.4 ms), of which ~3.4 ms GPU work. The rest
  was host-side: encoding ~80 encoders through string-keyed weight lookups
  before committing, then a blocking wait.
- **Context features through the CPU.** The target's captured layers went GPU
  buffer → `memmove` → Swift array → staging copy → GPU, and plain steps
  re-laid them out on the CPU.
- **Memory.** Loading the drafter read 860 MB of BF16 into a buffer, quantized
  it on the CPU and kept 153 MB of BF16 copies for long context updates: peak
  footprint 2.43 GB against 0.78 GB without the drafter.

Measured and ruled out before designing:

- **Encoder count.** 100 dependent tiny dispatches cost ~2.7 µs each on the
  GPU in one encoder or in 100 encoders, and ~1 µs each to encode either way.
  The tracer adds ~4 µs per encoder, which is why many small encoders looked
  like idle time.
- **Pipelining the encoding.** Encoding a verify layer's tail and the next
  layer's attention and shared FFN while waiting on the router (kept, since it
  costs nothing) did not move the per-layer gap: that encoding was already
  overlapped. The gap was the round trip itself.

### GPU-routed verify

A verify layer's routed experts are now queued before the host knows its
routes, as decode's are:

- **Bound experts.** The previous layer's router already scores each row's
  guess at this layer. The union of every row's best 48 guesses, filtered to
  experts in memory, is bound for this layer while the GPU runs it. Layer 0,
  with no previous layer, binds every resident expert during its own
  attention.
- **Coverage**, measured on teacher-forced 8-row blocks (code / math):

  | Per-row guesses bound | Routes covered | Layers fully covered | Experts bound |
  | ---: | ---: | ---: | ---: |
  | 8 | 93.5% / 92.3% | 7% / 5% | 27 |
  | 16 | 99.0% / 98.4% | 68% / 60% | 45 |
  | 24 | 99.7% / 99.4% | 87% / 80% | 60 |
  | 32 | 99.9% / 99.7% | 93% / 88% | 73 |

- **Kernels.** `verify_bound_plan` (one thread per bound expert) finds each
  expert's rows in the block once, lists the routed experts by row class (up
  to 2, 4, 8 or 16 rows) and writes indirect grids; per class, the same
  gate/up and down row math as the host-planned tiles runs over exactly the
  routed experts.
- **Patches.** Once the routes are back, the host runs only the routes the
  bound set missed (an unguessed expert, or one not in memory, read first)
  with the existing tile kernels, then commits the layer's tail. When nothing
  was missed, the host's only work between router and tail is reading 64
  routes and committing a pre-encoded command buffer, while the GPU computes
  the experts.

What it took:

- The first version was 2× slower than the host tiles: every one of ~3M
  threads per layer re-scanned the block's 64 routes, so the time grew with
  the number of bound experts, not the routed ones. The planning kernel and
  indirect dispatch fixed it.
- Kernels specialized to the block's row count were 8% slower than the tiles
  at 8 rows (register arrays sized for 8 rows cost occupancy, while most
  experts take 1–3). Row classes brought 8-row expert time from 28.7 to
  26.5 ms.
- How many guesses to bind: 16 left 7.3 ms of idle per 8-row verify (patches
  late), 24–128 all ~2 ms; 48 is the default (`ANE_FIELDFARE_VERIFY_BIND_TOP`).
- Binding is host work (~2 µs per expert for residency probes and
  `useResources`) done while the GPU runs the layer's attention and shared
  FFN, so it stays off the critical path.

GPU idle per verify fell from 5.1–5.5 ms to 0.5–1.1 ms at every row count. The
GPU-routed result is bitwise the host-routed one: same greedy tokens and same
final hidden rows at 1, 3, 8 and 16 rows (`GPURoutedVerifyTests`).
`ANE_FIELDFARE_HOST_ROUTED_VERIFY=1` plans every layer on the host.

### The drafter on the GPU

- **Weights in the file cache.** A 256 MB streaming read measured 133.8 GB/s
  from an anonymous shared buffer, 133.8 GB/s from a private buffer and
  133.9 GB/s from a no-copy buffer over a memory-mapped file in the page cache
  (warm clocks; the first anonymous run read 109 GB/s while the GPU clock
  ramped). File-backed weights cost no bandwidth, so the drafter's INT4 weights
  are installed once, next to the checkpoint, as `fieldfare-dflash-int4.bin`
  (242 MB, built in ~1.5 s, rebuilt if the checkpoint changes), and the GPU
  reads them in place. Load: 1.5 s → 1.0 s. Peak memory footprint with the
  drafter, on the quick suite's short prompts: 2.43 GB → 0.93 GB (0.78 GB
  without it). The footprint does not count the 242 MB of weights, which are
  file-backed page cache, and during prefill the feature ring below grows to
  the chunk size (35 MB at 1,024 tokens, 138 MB at 4,096). The pages are clean, so
  under pressure the OS can drop them like any cached file.
- **Fused layout.** Q/K/V and gate/up are concatenated by rows at install, so
  each is one dispatch, and a context update projects K|V as one slice of the
  Q/K/V matrix. The residual add is fused into the next norm, bitwise
  (`DFlashKernelTests`). The BF16 copies are gone: long context updates run the
  INT4 matrix kernel 16 rows at a time.
- **Sliding-window K/V ring.** The drafter's four sliding layers keep only
  their 2,048-position window (plus two blocks) as a ring; only the full layer
  grows with context.
- **Features never touch the CPU.** Prefill chunks, verify blocks and decode
  steps copy the six captured target layers into a feature ring on the GPU, in
  the command buffer of each layer's tail, at slot `position % capacity`. The
  drafter reads the rows it needs from there on the same queue. The ring holds
  256 positions, grown to the chunk size during prefill.

### One host wait per step

The drafter's context length lives on the GPU. Behind each verify's head, in
one command buffer, the drafter:

1. adds the block's rows to its context from the feature ring,
2. accepts on the GPU (`dflash_accept`): compares the drafts with the verified
   tokens, advances the context past the accepted ones and makes the target's
   next token the next anchor,
3. drafts the next block from there.

The verify reads its token IDs straight from the drafter's block buffer. The
host waits once, then reads the verified tokens, the acceptance (checked
against its own) and the next drafts and their confidences together. The
draft's host time is gone, and the draft runs back to back with the verify on
the GPU. The chained draft is bitwise the host-driven one
(`DFlashChainedDraftTests`).

### Results

`Scripts/bench/ab.sh dflash-v1 dflash-v2-final 2`, quick suite (192 tokens,
two ABBA rounds of 2 repetitions), desktop apps running, A/A noise floor 2.3%.
The machine receipt (`results/dflash-v2-m5-macos27.json`) is kept locally; `results/` is not published.

| Prompt | `--drafter` before | `--drafter` after | Change (95% CI) |
| --- | ---: | ---: | --- |
| code | 91.4 tok/s | 98.7 tok/s | +8.0% [+5.7%, +10.8%] |
| explanation | 65.8 tok/s | 72.7 tok/s | +10.4% [+7.3%, +13.5%] |
| math word problem | 68.6 tok/s | 78.3 tok/s | +14.1% [+9.2%, +18.4%] |

| Verify rows | Before | After |
| ---: | ---: | ---: |
| 1 | 24.6 ms | 21.8 ms (−11.3%) |
| 2 | 29.4 ms | 25.8 ms (−12.2%) |
| 4 | 36.3 ms | 33.3 ms (−8.2%) |
| 8 | 50.3 ms | 48.1 ms (−4.4%) |

Plain decoding did not change beyond noise, and every output matched the
earlier build's (math's speculative output first differs from plain greedy at
token 90 in both). Over the decode phase of the code and math cases (GPU
trace, union of encoder intervals from the end of prefill to the end of the
case) the GPU is now busy 93–95% of the
time, up from 81–83%: speculative decoding is GPU-bound. What remains per step
is ~0.4 ms between the draft finishing and the next verify starting (host
wake-up, reading the results, encoding layer 0, GPU start-up) and patches
that wait on SSD reads. Further gains have to come from the GPU work itself
(the 8-row expert and projection kernels run at ~105 GB/s of ~134) or from
higher acceptance.

## End-to-end results

### Before the rewrite

Against the build before the hot-vocabulary head and the drafter changes
above. `Scripts/bench/ab.sh` quick suite (192 tokens, two ABBA rounds of 2
repetitions), few other apps running; the A/A noise floor was 2.8%.

| Prompt | Plain before | Plain now | `--drafter` before | `--drafter` now |
| --- | ---: | ---: | ---: | ---: |
| code | 46.3 tok/s | 49.2 tok/s (+6%) | 73.8 tok/s | 89.4 tok/s (+21%) |
| math word problem | 45.2 tok/s | 48.0 tok/s (+6%) | 62.6 tok/s | 66.4 tok/s (+6%) |
| explanation | 45.5 tok/s | 48.1 tok/s (+6%) | 58.2 tok/s | 65.0 tok/s (+12%) |

Plain output was byte-identical between the builds. Speculative output first
differed from plain greedy at token 90 on math, in both builds.

### The verify work

Same suite, but with desktop apps running (A/A noise floor 3%); "before" is the
build just before the verify changes:

| Prompt | Plain | `--drafter` before | `--drafter` now |
| --- | ---: | ---: | ---: |
| code | 39.5 tok/s | 55.1 tok/s | 71.8 tok/s (+30%) |
| math word problem | 38.9 tok/s | 48.1 tok/s | 57.5 tok/s (+20%) |
| explanation | 38.0 tok/s | 41.9 tok/s | 51.6 tok/s (+23%) |

Plain output was byte-identical to the earlier build. Speculative output first
differed from plain greedy at token 90 on math in one run, before and after.

### First speculative decoding runs

Greedy, 384 tokens, three alternating runs per cell:

| Prompt | Plain | `--drafter` | Speedup |
| --- | ---: | ---: | ---: |
| code | 32.9 tok/s | 56.2 tok/s | 1.71× |
| math word problem | 32.9 tok/s | 53.1 tok/s | 1.61× |
| explanation | 32.9 tok/s | 42.7 tok/s | 1.30× |
| travel chat | 33.4 tok/s | 37.0 tok/s | 1.11× |
| essay | 33.0 tok/s | 35.7 tok/s | 1.08× |

Plain output was identical across all runs and to the build before.

**What this does and does not show:** speculative output equals plain greedy
output only up to the rounding difference between verifying several tokens at
once and decoding one at a time. The first differing token was at 356 (code),
90 (math), 256 (essay), and at 331 in one of three explanation runs; each was a
near-tie between two plausible continuations. Because the scheduler picks block
sizes from measured time, which near-ties flip can vary from run to run.

### Before all of the above

This comparison predates every change in this log, including the OS file cache,
speculative decoding and the supervisor. It compares unmodified Turbo
Fieldfare `4c6db1e`, this repo's `metal` backend and `hybrid-ane`. Setup: Swift
6.4, temperature 0.2, top-k 64, top-p 0.95, 4,096-token context, 1,024-token
output cap. Each value, in seconds, is the mean of two runs after
warm-up, with backend order reversed between passes.

| Prompt tokens | Turbo Fieldfare | `metal` | `hybrid-ane` |
| ---: | ---: | ---: | ---: |
| 61 | 36.62 s | 37.24 s | 37.45 s |
| 430 | 56.76 s | 54.73 s | 55.77 s |
| 3,015 | 102.96 s | 71.96 s | 71.28 s |

The ANE helped only with long prompts. On the 3,015-token prompt, the first
text appeared after 24.39 s with `hybrid-ane` against 26.49 s with `metal`.
Generation speed was 13.0–13.7 tok/s for all three.
