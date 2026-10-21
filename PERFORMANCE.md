# Performance log

Changes to the runtime in roughly chronological order, each with its measured
effect and the alternatives that were rejected. Hardware is an M5 with 16 GB
unless stated. "Bit-exact" means the change did not alter output tokens or the
compared tensors.

## Prefill

Optimized from timelines of SSD reads, GPU command buffers and ANE calls on a
3,015-token prompt. Single runs of prefill wall time.

| Change | `hybrid-ane` | `metal` | Output |
| --- | ---: | ---: | --- |
| Baseline | 22.2 s | 24.0 s | — |
| Separate `MTLCommandQueue` for the ANE path's weight expansion, so the ANE no longer stalls behind the shared FFN on the main queue | 19.5 s | — | Bit-exact |
| Shared FFN over a chunk as 3 GEMMs instead of 1,024 per-token GEMVs (7.1 s → 0.83 s) | 16.9 s | 18.7 s | Relative L2 vs FP32 reference unchanged (≈ 3–4e-4, `PrefillSharedExpertBatchedTests`); logits vs previous build: mean KL 0.015 (`metal`), 0.003 (`hybrid-ane`); greedy tokens 17/17 equal |
| 4 concurrent expert reads instead of 1 (SSD: 4.7 GB/s at queue depth 1, 6.7 GB/s at 2) | 16.2 s | — | Bit-exact |
| Chunk size chosen per prompt, up to 4,096 tokens (`--prefill-chunk-tokens auto`, default). Every chunk re-reads each expert it routes to, so fewer chunks read less: 32.1 → 11.7 GB | 11.7 s | 15.3 s | Bit-exact |
| ANE program at 128 tokens per expert per call instead of 64. Each call streams 4 experts' expanded weights (~0.95 ms), so doubling tokens per call halves calls: 3,352 → 1,980 | 8.8 s | — | Bit-exact |

Short prompts were dominated by integrity checks, not compute: the CLI
SHA-256'd all 13 GB of expert files on every launch. Removing that took prefill
from 6.3 s to 1.6 s on a 28-token prompt and from 6.7 s to 2.2 s on a
430-token prompt. Load now validates file sizes against the manifest;
`ANEFieldfareInstall --verify-install` re-hashes on demand.

Remaining time on the 3,015-token prompt (8.8 s): 4.2 s of GPU work before the
router (attention and projections), 4.4 s of routed experts, 3.5 s of which is
ANE compute.

Rejected:

- Splitting each chunk in two and pipelining the halves one stage apart, so GPU
  attention overlaps ANE experts: 13.4 s vs 9.2 s. Every ANE call needs the GPU
  to expand INT4 weights to FP16 first, so the overlap saturates the GPU, and
  both halves re-read and re-expand the same experts.
- A third ANE input/output buffer pair: no measurable change.
- Prefetching every next-layer expert during long prefill: no gain on `metal`,
  16% slower on `hybrid-ane`; the prefetches compete with the layer's own reads.

## Decode: page cache as the expert cache

The runtime previously kept an application-level LFU cache of 48 experts per
layer (4.9 GB). It now keeps none: each expert is a no-copy `MTLBuffer` over
the `mmap`ed layer file, and residency is the OS page cache's job. Greedy,
warm, alternating builds; output byte-identical.

| Case (prompt / generated tokens) | 48-slot cache | Page cache |
| --- | ---: | ---: |
| code (35 / 256) | 26.6 tok/s | 30.6 tok/s |
| essay (31 / 512), two runs | 26.7, 27.1 tok/s | 31.3, 30.2 tok/s |
| after a 4,033-token prompt (24 generated), `metal`, two runs | 12.1, 13.7 tok/s | 14.7, 15.5 tok/s |
| after a 4,033-token prompt (24 generated), `hybrid-ane`, two runs | 15.4, 15.6 tok/s | 16.8, 18.1 tok/s |

Prefill on the 4,033-token prompt was unchanged (20.3 vs 20.2 s `metal`, 11.2
vs 11.4 s `hybrid-ane`). On the code case, expert misses fell from 21 to 3.3
per token; the page cache held 55–64 experts per layer, and 59–62% of
next-layer prefetches were used.

Residency is tested with `mincore` on 8 sampled pages per expert: a full scan
of 205 pages costs 54 µs, and decode checks 240 experts per token. Decode
prefetches (`F_RDADVISE`) the 2 highest-scoring absent experts of the next
layer, scored by a fitted next-layer router guess.

## Decode: GPU-side expert routing

Previously the CPU waited for each layer's router output before encoding that
layer's expert work. Now, while the GPU executes layer L, the CPU encodes layer
L+1's attention and shared FFN; once layer L's routes are read back, it binds
the resident experts among the next-layer guess's top 24 (`useResources`) and
submits layer L+1's expert work before its routes exist. On the GPU, a small
kernel reads the router's top-8 and points each expert slot at its weights
through a per-layer address table.

If a routed expert was not bound, its slot reads a zeroed expert and the CPU
runs a fix-up pass: it restores the saved layer input, recomputes the missing
experts' contribution, and re-runs the rest of the layer. The result is
bit-identical to CPU routing. Layer 0 is always CPU-routed.

Also in this round:

- Shared FFN gate, up and GELU in one dispatch; the intermediate is still
  rounded to FP16, so output is unchanged.
- Router and next-layer guess in one dispatch.
- Top-8 selection across the 32 lanes of a SIMD group instead of one thread
  (12 → 1.8 µs).

Decode step, A-B-B-A: 29.3 → 23.9 ms from GPU routing (33 → ~42 tok/s), then
−5% from the three changes above; output byte-identical throughout. A decode
step reads ~2.2 GB, a 16.4 ms floor at the measured ~132 GB/s. Largest
remaining per-token costs: routed experts 7.4 ms, QKV 4.0 ms, LM head 3.9 ms,
shared FFN 3.2 ms.

Rejected: binding more of the guess. Top 48 matched top 24; binding all 128
dropped decode to ~30 tok/s.

## Decode: partial LM head

Greedy decode reads the full 262,144-row LM head once every 4 tokens instead
of every token, with identical output (`Runtime/Generation/HotHeadDecoder.swift`).

- **Speculative steps** score only a frequent-token subset: 56k IDs from
  `HotVocabulary/vocab-262144.bin`, plus all prompt tokens and every token a
  check found outside the subset. The kernel gathers those rows in place from
  the head (4 bytes per listed token, no copy): 0.75 ms GPU for 56k rows
  (0.37 ms for 32k) vs 3.3 ms for the full head.
- **Every fourth step** runs the full head over the 4 buffered positions in one
  pass (3.4 ms). Each position's reduction order matches the single-token head,
  so the argmax is exactly plain decoding's (tested bitwise, ties included).
  Tokens before the first mismatch are kept; the mismatch is replaced and the
  KV cache after it is rolled back.
- **Mispredictions**: 0–3 per 192 tokens on the quick suite, 5–7 on chat and
  essay prompts; each wastes the rest of its 4-token block. The subset is
  ranked by frequency over ShareGPT English answers and Python source
  (`Scripts/hotvocab/build.py`). At 28k entries, essay and chat mispredicted
  ~5% of the time, so it was grown to 56k.

Used for greedy text decode with chunked prefill. A subset for a different
vocabulary only affects speed, since every token is checked against the full
head.

## Speculative decoding

### Scheduler

- **Draft length.** Per-draft acceptance is modelled as i.i.d. with a rate
  fitted from decayed recent outcomes (after DISCO); the scheduler drafts the
  block length in 2–16 that maximizes expected tokens per second, or takes a
  plain decode step (which still extends the drafter's context). While plain
  decoding wins, it retries a 4-token block with exponential backoff.
- **Verify length.** After drafting, it may verify only a prefix (after
  DSpark). Each draft's probability under the target head is calibrated
  against observed acceptance in ten probability bins, and the prefix with the
  highest expected tokens net of its verify cost is verified; possibly none.
- Since the [rewrite](#speculative-decoding-rebuilt), the next block is drafted
  on the GPU immediately behind the verify, so its length is chosen when the
  verify is enqueued, before its outcome is known. The verify prefix is still
  chosen after drafting.
- Cost priors (until online measurements replace them) were refitted on
  2026-10-20 to the current build: plain step 21 ms (was 30), verify 19 +
  3.6 ms per row (was 26 + 5.5), draft 4.2 + 0.15 ms per row (was 4 + 1.2).
  `ab.sh`, 3 rounds: math with the drafter +9.2% [+2.6%, +15.9%], code and
  explanation unchanged.

### Verify cost, first three versions

Median of 34 at fixed context. Rows = tokens verified in one pass. For the
rewrite's numbers see [Results](#results).

| Rows | v1 | v2 | v3 | Change |
| ---: | ---: | ---: | ---: | --- |
| plain decode | 28 ms | 28 ms | 26 ms | |
| 1 | 49.5 ms | 32.8 ms | 29.3 ms | pipelined layer epilogues; GEMV specialized by row count |
| 2 | | 44.0 ms | 36.7 ms | |
| 4 | 66.8 ms | 44.8 ms | 43.3 ms | as above; 2 weight rows per SIMD group at 3–4 rows |
| 8 | 89.3 ms | 77.2 ms | 56.3 ms | MMA kernel; verify-specific expert kernels |

What reduced verify cost:

- **Pipelining.** Verify no longer waits on each layer's epilogue (residual,
  norms, feature copy); queue ordering handles the dependency, as in decode.
  Removes 30 CPU–GPU round trips per verify.
- **One INT4 GEMV for all small batches**, specialized by row count and width,
  shared by verify projections, the shared FFN, the drafter and both heads.
  - ≤4 rows: SIMD ALU path, row r bit-identical to the single-row result
    (tested), bandwidth-bound; 2 weight rows per SIMD group at 3–4 rows. At 1
    row: QKV 6.9 → 3.8 ms, O 3.7 → 2.0 ms, shared FFN 5.6 → 3.0 ms.
  - 5–16 rows: the ALU path turns compute-bound (8 rows cost 2×, 16 rows 4×),
    so these use `simdgroup_matrix` MMA. Each weight is dequantized once
    (q·s + b in FP32) into an 8×8 tile applied to 8 rows per multiply. Matches
    the reference within FP32 rounding, not bitwise (tested). 8-row head:
    7.0 → 3.3 ms (125 GB/s).
- **Greedy head** as that kernel into FP32 logits plus a per-row argmax, which
  also yields each draft's probability. The kernel it replaced cost ~3 ms per
  row.
- **Verify expert kernels.** One SIMD group per expert weight row applies it to
  every verify row routed to that expert, so a block reads each used expert
  once. The previous kernels used one thread per (weight row, token). 8-row
  expert time: ~37 → ~28 ms.
- **Decode kernels with one grid row per token.** Verify runs decode's router
  and next-layer guess in one dispatch with a SIMD-group top-8 per token
  (previously one serial thread per token; −3.4 ms per verify). Q/K/V norms
  and RoPE run as decode's fused kernel with one grid row per token (5
  dispatches before; −2 ms at 8 rows).
- **Less CPU stall.** The shared FFN is enqueued behind attention before the
  CPU blocks, so it executes while routes are planned. The CPU spins on the
  router's `MTLSharedEvent` instead of sleeping. Residency checks sample 2
  pages per expert instead of 8 (−2.7 ms per verify), and next-layer prefetch
  is issued from a background queue (−3.8 ms).
- **Prefetch for verify rows.** Each row's guessed next-layer experts are
  prefetched, halving verify misses (93 → 45 per 8-row step).
- **INT4 drafter.** Drafter projections are quantized at load to the target's
  INT4 format: ~860 → ~360 MB resident, acceptance about unchanged (τ 4.9 vs
  5.1 at block 8).
- **Partial drafter head.** The drafter scores a 16k frequent-token subset plus
  prompt and accepted tokens, gathering those head rows in place with the MMA
  kernel. A draft outside the subset is simply rejected by verify, so no check
  is needed. Acceptance did not change beyond noise between 8k and 56k entries.
- **Drafter projections on MMA.** BF16 drafter activations used to force the
  ALU kernel, which is slow at 8 rows; the MMA kernel now takes BF16 in and
  out. With the partial head, an 8-row draft went from ~8.6 to ~4.6 ms.
- **Drafter context update.** Updates of up to 16 rows read INT4 copies of `fc`
  and the K/V projections instead of BF16 (~150 → ~43 MB) and are submitted
  without waiting; the next draft on the same queue orders after them. CPU
  time per step: 2.6 ms → ~0.
- **Minor.** The shared FFN down projection (K = 2112) runs on the MMA kernel,
  which handles a trailing 64-wide group as a half step; verify no longer
  zero-fills expert outputs it overwrites; experts found resident by the
  previous layer's prefetch are not re-probed (0.63 → 0.19 ms CPU per verify).
  None changed step time beyond noise.

Rejected:

- Decode's router once per verify row: −1 ms at 1 row, +5 ms at 8 rows
  (superseded by the batched router above).
- An FP32 conversion pass over inputs before the multi-row kernel: slower.
- 2–4 weight rows per SIMD group above 4 rows: up to 2× slower.
- Early MMA layouts: one 8-byte weight load per thread per group was 2× slower
  than the ALU kernel; 16 live tiles per thread spilled registers (5× slower).
  The kernel now loads 64 contiguous bytes per weight row per step and builds
  each tile just before its multiply.
- Drafter on the ANE. The draft is 8–15 ms of a 50–95 ms step and can't
  overlap the verify that depends on it, and the ANE would stream ~860 MB of
  FP16 weights at ~70 GB/s, slower than the GPU's INT4 drafter.
- A faster 8-row MMA kernel. At 8 rows it reaches ~105 GB/s and half that at
  16: it is bound by FP32 FMA throughput (~3 TFLOPS), not memory. FP16 tiles
  gave the same time, as did reordering input loads to avoid waste (kept; no
  slower). A tensor-op version (`matmul2d`, weights dequantized to FP16 in
  threadgroup memory, or passed as `uint4b_format` with per-group scale and
  offset) ran at a flat ~75 GB/s for 4–16 rows: slower at 8, faster only at 16.

### Verify profile before the rewrite

An 8-row verify took ~50 ms with the GPU idle for ~5–6 ms of it (command
buffer timestamps): routed experts ~26.7 ms, QKV 4.5, shared FFN 4.6, head 4.2,
O 2.6, router 1.8, attention 1.4. Most idle time, ~100 µs per layer at any row
count, was the router-output round trip to the CPU and back, which exceeded
the shared FFN enqueued to hide it; the rest was waiting on expert reads.

## Speculative decoding, rebuilt

A rewrite of the drafter, the verify pass and the loop between them for this
platform's constraints: unified memory, a single GPU queue, and experts
streamed through the page cache. Code: `Runtime/DFlash/` and
`RealForwardRunner.verifyBlock`.

### Profile of the previous pipeline

GPU trace of the quick suite's code prompt (untraced timings in brackets):

- **Verify, 89% of decode time.** ~44 ms per step (41.7 ms). An 8-row verify
  idled 5.8 ms, almost entirely one gap per layer between the shared FFN and
  the first routed-expert tile (130–190 µs): route readback, tiling, residency
  probes, binding, and resubmission.
- **Draft, 10%.** 5.1 ms per step (4.4 ms), ~3.4 ms of it GPU. The rest was
  CPU: encoding ~80 encoders through string-keyed weight lookups, then a
  blocking wait.
- **Target hidden states via the CPU.** Captured layers went GPU buffer →
  `memmove` → Swift array → staging copy → GPU, and plain steps re-laid them
  out on the CPU.
- **Memory.** Drafter load read 860 MB of BF16, quantized it on the CPU, and
  kept 153 MB of BF16 copies for long context updates: peak footprint 2.43 GB
  vs 0.78 GB without the drafter.

Ruled out before designing:

- **Encoder count.** 100 dependent tiny dispatches cost ~2.7 µs each on the GPU
  whether in one encoder or 100, and ~1 µs each to encode either way. The
  tracer adds ~4 µs per encoder, which made many small encoders look like idle
  time.
- **Encoding ahead.** Encoding a layer's epilogue and the next layer's
  attention and shared FFN while waiting on the router (kept; free) did not
  move the per-layer gap, since that encoding was already overlapped. The gap
  was the round trip itself.

### GPU-routed verify

Verify enqueues a layer's routed experts before the CPU knows its routes, as
decode does:

- **Pre-bound set.** The previous layer's router already scores each row's
  guess for this layer. The union of each row's top-k guesses, filtered to
  resident experts, is bound for this layer while the GPU runs it. Layer 0,
  with no previous layer, binds every resident expert during its attention.
- **Coverage** on teacher-forced 8-row blocks (code / math):

  | Guesses per row | Routes covered | Layers fully covered | Experts bound |
  | ---: | ---: | ---: | ---: |
  | 8 | 93.5% / 92.3% | 7% / 5% | 27 |
  | 16 | 99.0% / 98.4% | 68% / 60% | 45 |
  | 24 | 99.7% / 99.4% | 87% / 80% | 60 |
  | 32 | 99.9% / 99.7% | 93% / 88% | 73 |

- **Kernels.** `verify_bound_plan` (one thread per bound expert) finds each
  expert's rows in the block once, buckets the routed experts by row count (up
  to 2, 4, 8 or 16) and writes indirect dispatch arguments; per bucket, the
  same gate/up and down kernels as the CPU-planned tiles run over exactly the
  routed experts.
- **Fix-up.** Once routes are read back, the CPU runs only the routes outside
  the bound set (an unguessed expert, or one that was not resident and is read
  first) with the existing tile kernels, then commits the layer epilogue. With
  full coverage, the CPU's work between router and epilogue is reading 64
  routes and committing a pre-encoded command buffer.

Development notes:

- The first version was 2× slower than CPU-planned tiles: each of ~3M threads
  per layer rescanned the block's 64 routes, so cost scaled with bound experts
  rather than routed ones. The planning kernel plus indirect dispatch fixed it.
- Kernels specialized to the block's row count were 8% slower than the tiles
  at 8 rows (register arrays sized for 8 rows cut occupancy, while most experts
  receive 1–3 rows). Row-count buckets brought 8-row expert time from 28.7 to
  26.5 ms.
- Guesses bound per row (`ANE_FIELDFARE_VERIFY_BIND_TOP`). By GPU idle alone,
  16 left 7.3 ms per 8-row verify (late fix-ups) and 24–128 all ~2 ms. End to
  end, binding has a CPU cost per expert (`mincore` probes and `useResources`)
  and the 8-row union grows quickly. Quick suite with the drafter, three runs
  of 2 repetitions, 2026-10-20 (tok/s):

  | Guesses per row | code | math | explanation |
  | ---: | ---: | ---: | ---: |
  | 96 | 67.7 | 54.6 | 46.4 |
  | 64 | 73.7 | 63.7 | 56.6 |
  | 48 (previous default) | 89.9 | 71.1 | 66.0 |
  | 32 | 95.7 | 74.7 | 69.4 |
  | 24 | 95.9 | 74.9 | 70.5 |
  | 20 (default) | 95.9 | 75.6 | 71.3 |
  | 16 | 94.0 | 75.1 | 70.1 |

  Prefetching 3 or 4 experts per layer instead of 2 made no difference beyond
  noise at 20 or 24.

GPU idle per verify fell from 5.1–5.5 ms to 0.5–1.1 ms at every row count. The
GPU-routed result is bit-identical to the CPU-routed one: same greedy tokens
and same final hidden states at 1, 3, 8 and 16 rows (`GPURoutedVerifyTests`).
`ANE_FIELDFARE_HOST_ROUTED_VERIFY=1` routes every layer on the CPU.

### Drafter on the GPU

- **Weights in the page cache.** A 256 MB streaming read ran at 133.8 GB/s from
  an anonymous shared buffer, 133.8 GB/s from a private buffer and 133.9 GB/s
  from a no-copy buffer over a page-cache-resident `mmap` (warm clocks; the
  first anonymous run read 109 GB/s while the GPU clock ramped). File-backed
  weights cost no bandwidth, so the drafter's INT4 weights are written once
  next to the checkpoint as `fieldfare-dflash-int4.bin` (242 MB, ~1.5 s,
  rebuilt if the checkpoint changes) and read in place. Load: 1.5 → 1.0 s.
  Peak footprint with the drafter on short prompts: 2.43 → 0.93 GB (0.78 GB
  without). The footprint excludes the 242 MB of file-backed weights; during
  prefill the hidden-state ring (below) grows to the chunk size (35 MB at
  1,024 tokens, 138 MB at 4,096). Those pages are clean and evictable.
- **Fused layout.** Q/K/V and gate/up are row-concatenated at install, so each
  is one dispatch, and a context update projects K|V as one slice of the Q/K/V
  matrix. The residual add is fused into the following RMSNorm, bit-exact
  (`DFlashKernelTests`). The BF16 copies are gone; long context updates run
  the INT4 MMA kernel 16 rows at a time.
- **Sliding-window KV ring.** The drafter's four sliding-window layers keep
  only their 2,048-position window (plus two blocks) as a ring buffer; only the
  full-attention layer grows with context.
- **Hidden states stay on the GPU.** Prefill chunks, verify blocks and decode
  steps copy the six captured target layers into a GPU ring buffer at slot
  `position % capacity`, in each layer's epilogue command buffer. The drafter
  reads from it on the same queue. Capacity is 256 positions, grown to the
  chunk size during prefill.

### One CPU wait per step

The drafter's context length lives in GPU memory. Behind each verify's LM
head, in one command buffer, the drafter:

1. appends the block's rows to its context from the hidden-state ring,
2. runs acceptance on the GPU (`dflash_accept`): compares drafts with verified
   tokens, advances its context past the accepted prefix, and anchors the next
   block on the target's next token,
3. drafts the next block.

Verify reads its token IDs directly from the drafter's block buffer. The CPU
waits once per step, then reads the verified tokens, the acceptance (checked
against its own), and the next drafts with their probabilities. Draft CPU time
is eliminated and the draft runs back to back with the verify. The chained
draft is bit-identical to the CPU-driven one (`DFlashChainedDraftTests`).

### Results

`Scripts/bench/ab.sh dflash-v1 dflash-v2-final 2`: quick suite, 192 tokens,
two A-B-B-A rounds of 2 repetitions, desktop apps running, noise floor 2.3%.

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

Plain decode was unchanged within noise and all outputs matched the previous
build (math's speculative output first differs from plain greedy at token 90
in both). GPU utilization over the decode phase of the code and math cases
(union of encoder intervals from end of prefill to end of case) rose from
81–83% to 93–95%: speculative decoding is now GPU-bound. Remaining per-step
overhead is ~0.4 ms between draft completion and the next verify (CPU wake-up,
result readback, layer-0 encode, GPU ramp) plus fix-ups stalled on SSD reads.
Further gains need faster GPU work (the 8-row expert and projection kernels
run at ~105 of ~134 GB/s) or higher acceptance.

## Speculative sampling with shared noise

Speculative decoding was greedy-only. All token producers now sample with one
GPU kernel (`sample_rows_f32`): the token at position j is the argmax of
softcap(logit) / T + G(seed, j, token) over the top-k / top-p survivors, where
G is a standard Gumbel variate from a counter-based hash. This is an exact
sample from the truncated tempered softmax (Gumbel-max trick) and depends only
on the logits and (seed, j). Consequences:

- The target's sample at a position is identical whether produced by plain
  decode, a verify row or prefill's last row. Teacher-forcing a plain sampled
  decode through 1-, 3- and 8-row verifies reproduces every token
  (`sampledVerifyReproducesSampledDecode`; `verify-check` with
  `ANE_FIELDFARE_VERIFY_SAMPLE`).
- The drafter samples with the same noise (keyed from the GPU-side context
  length), so acceptance remains an exact token match and the GPU acceptance,
  chained loop and scheduler are unchanged. A draft is accepted when both
  models' perturbed maxima coincide, which shared noise makes likely when the
  distributions are close.
- Output equals the target's sampled text for the seed with or without the
  drafter, up to near-ties between batched and single-row arithmetic.

Top-k (k ≤ 256) is exact in two passes per row: a 2,048-bin histogram of the
softcapped logits locates the bin containing the k-th largest, the entries at
or above it are gathered into threadgroup memory and sorted, then top-p and the
sample run on the survivors.

Measured beforehand: sampling was not a bottleneck. The GPU sampler took
~0.1–0.9 ms per token, and sampled decode ran within ~10–15% of greedy, the
difference being the full 262k-row head (greedy uses the partial head). An
earlier 18.6 tok/s sampled figure was a contended measurement.

Quick suite, 192 tokens, top-k 64, top-p 0.95, seed 5, single runs, desktop
apps running (plain → `--drafter`, tok/s):

| Temperature | code | math | explanation |
| --- | --- | --- | --- |
| 0 (greedy) | 48 → 89 (1.87×) | 45 → 69 (1.54×) | 46 → 66 (1.44×) |
| 0.6 | 43 → 82 (1.90×) | 40 → 70 (1.75×) | 40 → 58 (1.44×) |
| 0.8 | 42 → 73 (1.74×) | 41 → 78 (1.89×) | 41 → 62 (1.52×) |
| 1.0 | 42 → 78 (1.84×) | 41 → 71 (1.73×) | 41 → 64 (1.58×) |

Mean accepted drafts per step: 1.3–2.5 sampled vs 1.8–3.3 greedy. Eleven of
twelve sampled speculative outputs equalled plain sampled output; explanation
at 0.8 diverged at token 32, a flipped near-tie caused by a race (a plain
decode step reading KV rows a verify had written).

Sampled text now differs from the old CPU sampler's for the same seed
(different noise source). Repetition penalty still uses the CPU sampler,
without the drafter.

## End-to-end results

### Partial head and drafter changes

Against the build before the partial LM head and the drafter changes above.
`Scripts/bench/ab.sh`, quick suite, 192 tokens, two A-B-B-A rounds of 2
repetitions, few other apps running, noise floor 2.8%.

| Prompt | Plain before | Plain after | `--drafter` before | `--drafter` after |
| --- | ---: | ---: | ---: | ---: |
| code | 46.3 tok/s | 49.2 tok/s (+6%) | 73.8 tok/s | 89.4 tok/s (+21%) |
| math word problem | 45.2 tok/s | 48.0 tok/s (+6%) | 62.6 tok/s | 66.4 tok/s (+6%) |
| explanation | 45.5 tok/s | 48.1 tok/s (+6%) | 58.2 tok/s | 65.0 tok/s (+12%) |

Plain output byte-identical between builds. Speculative output first differed
from plain greedy at token 90 on math in both.

### Verify optimizations

Same suite with desktop apps running (noise floor 3%); "before" is the build
just prior to the verify changes:

| Prompt | Plain | `--drafter` before | `--drafter` after |
| --- | ---: | ---: | ---: |
| code | 39.5 tok/s | 55.1 tok/s | 71.8 tok/s (+30%) |
| math word problem | 38.9 tok/s | 48.1 tok/s | 57.5 tok/s (+20%) |
| explanation | 38.0 tok/s | 41.9 tok/s | 51.6 tok/s (+23%) |

Plain output byte-identical to the earlier build. Speculative output first
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

Plain output identical across runs and to the previous build.

Speculative output equals plain greedy output only up to rounding differences
between batched verification and single-token decode. First divergences: token
356 (code), 90 (math), 256 (essay), and 331 in one of three explanation runs;
each was a near-tie between two plausible continuations. Because block sizes
are chosen from measured timings, which near-ties flip varies run to run.

### Baseline

Predates every change in this log, including the page-cache expert path,
speculative decoding and the ANE supervisor. Unmodified Turbo Fieldfare
`4c6db1e` vs this repo's `metal` and `hybrid-ane` backends. Swift 6.4,
temperature 0.2, top-k 64, top-p 0.95, 4,096-token context, 1,024-token output
cap. Mean wall time of two runs after warm-up, backend order reversed between
passes.

| Prompt tokens | Turbo Fieldfare | `metal` | `hybrid-ane` |
| ---: | ---: | ---: | ---: |
| 61 | 36.62 s | 37.24 s | 37.45 s |
| 430 | 56.76 s | 54.73 s | 55.77 s |
| 3,015 | 102.96 s | 71.96 s | 71.28 s |

The ANE only helped on long prompts: on the 3,015-token prompt, time to first
output was 24.39 s with `hybrid-ane` vs 26.49 s with `metal`. Decode was
13.0–13.7 tok/s for all three.
