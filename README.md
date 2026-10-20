# ANE Fieldfare

**Gemma 4 26B-A4B at up to 99 tokens/s on a 16 GB Mac.** Prefill on the Apple
Neural Engine, speculative decoding on the GPU, and 13 GB of experts served
straight from the OS file cache.

| Same M5, 16 GB, same prompts | ANE Fieldfare | [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare) | llama.cpp |
| --- | ---: | ---: | ---: |
| Generation, greedy | **65–87 tok/s** | 24–25 tok/s | 5–9 tok/s |
| Generation, sampled (temperature 0.2) | **37–53 tok/s** | 21–25 tok/s | |
| First token, 3,015-token prompt | **10 s** | 48 s | ~110 s (est.) |

Measured on 2026-10-20 against Turbo Fieldfare's latest commit: greedy on this
repo's benchmark prompts, sampled on Turbo's own benchmark; first token
includes model load. Up to 99 tok/s is this repo's benchmark on code
([Performance](#performance)). llama.cpp runs Google's Q4_0 GGUF, which is
over Metal's 12.7 GB limit on a 16 GB Mac, so even its best layer split pages
(`llama-bench`; first token estimated from its 24–32 tok/s prefill).

## Architecture

```
 PROMPT (prefill)                              GENERATION (decode)

 GPU  attention, router, shared FFN            GPU  DFlash drafter drafts a block
  │                                             │
 GPU  unpacks the next 4 INT4 experts ──┐      GPU  the model verifies the whole block
 ANE  computes the previous 4      ◄────┘           in one pass, routing experts on the GPU
  │                                             │
  └───── 13 GB of experts, memory-mapped and zero-copy: the OS file cache is the expert cache
```

- **The file cache is the expert cache.** Experts are zero-copy Metal buffers
  over the mapped model file, so a 13 GB model runs in under 1 GB of process
  memory on short prompts.
- **The Neural Engine does prefill.** It can't unpack 4-bit weights, so the GPU
  unpacks the next 4 experts while the ANE computes the current ones: 1.7×
  faster than the GPU alone on long prompts.
- **Speculation never leaves the GPU.** The drafter, the verify's expert
  routing and the acceptance all run on the GPU, so the CPU waits once per
  step and the GPU stays 93–95% busy.

The ANE path uses private Apple frameworks: a macOS update can break it
(`--backend metal` runs everything on the GPU), and it can't ship in App Store
apps.

## Requirements

- An Apple Silicon Mac with 16 GB of memory or more. Development and the
  measurements below used an M5 with 16 GB.
- macOS 26 or later. Measured on macOS 27 build `26A5406e`.
- Xcode with Swift 6.2 or later.
- The model in `.gturbo` format (about 13 GB). This repository reads that
  format but cannot create it. Use the installer from
  [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare) and put the
  result at `checkpoints/gturbo/gemma4.gturbo` (git ignores `checkpoints/`).
- `z-lab/gemma-4-26B-A4B-it-DFlash` (dflash drafter) at
  `checkpoints/dflash/gemma-4-26B-A4B-it-DFlash` (or beside the model), where
  the CLI and the server find it. Without it the model still runs,
  1.5–1.9× slower. The first run quantizes it to this runtime's 4-bit format into
  `fieldfare-dflash-int4.bin` (242 MB) in the same directory.

## Quick start

```bash
Scripts/build.sh     # builds the CLI and the server into .build/

.build/swift-hotpath/release/ANEFieldfareHotpathCLI \
  --model checkpoints/gturbo/gemma4.gturbo \
  --chat-prompt "What is the capital of France? Answer briefly." \
  --temperature 0 --max-new 64
```

Speculative decoding is enabled whenever the drafter is installed, greedy or
sampled; image prompts and a repetition penalty decode without the drafter. The
prompt runs on the Neural Engine (`hybrid-ane`) unless you pass `--backend metal`.
`--help` lists every option, including `--image` for image input.

### Options that choose where work runs

| Option | What it does |
| --- | --- |
| `--backend hybrid-ane` | During prompt processing, the 8 chosen experts of every layer run on the ANE. Default. Needs the default `--expert-top-k 8`, no `--prefill-expert-cap` and prefill on; without them the CLI uses `metal`. If the ANE cannot be set up, startup fails rather than falling back. |
| `--backend metal` | Everything on the GPU. |
| `--fault-policy fail-closed` | If the ANE fails mid-request, stop with an error. Default. |
| `--fault-policy failover-metal` | If the ANE fails mid-request, stop using it and finish the request on the GPU, starting again from the layer that failed. |
| `--drafter <dir\|off>` | The DFlash draft model for speculative decoding, if not in the default place; `off` decodes one token at a time. See [How it works](#speculative-decoding). |
| `--early-expert-read off\|router\|fitted` | Whether generation asks the SSD for next layer's likely experts ahead of time. Default `fitted`. |
| `--prefill-chunk-tokens <n\|auto>` | How many prompt tokens are processed per pass. Default `auto`: the smallest size that covers the prompt. |

### Environment variables

| Variable | Effect |
| --- | --- |
| `ANE_FIELDFARE_HOTPATH_PROFILE=1` | Write a JSON timing profile to stderr, including which work ran on the ANE and any ANE failures. |
| `ANE_FIELDFARE_GPU_TRACE=<path>` | Record every GPU encoder's execution and the host's waits on one clock, and write a Chrome trace to `<path>` (open it at ui.perfetto.dev). `Scripts/profile/summarize.py <path>` prints GPU time per kernel and GPU idle time by what the host was doing. Costs no measurable speed. |
| `ANE_FIELDFARE_ANE_ROUTED_PREFILL_LAYERS=n` | Use the ANE for layers `0..<n` only. |
| `ANE_FIELDFARE_HOT_HEAD=0` | `ANEFieldfareBench` only: turn off the [hot-vocabulary shortcut](#hot-vocabulary-shortcut). |
| `ANE_FIELDFARE_HOST_ROUTED_VERIFY=1` | Plan every verify layer's experts on the CPU instead of routing them on the GPU. Slower; meant for checking the GPU routing. |
| `ANE_FIELDFARE_VERIFY_BIND_TOP=n` | Guessed experts per verify row bound for GPU routing. Default 20. |
| `ANE_FIELDFARE_TEST_CHECKPOINT=<path>` | Model location for tests, if not the default. |
| `ANE_FIELDFARE_API_KEY=<key>` | Server only: require `Authorization: Bearer <key>`, like `--api-key`, without putting the key on the command line. |

## OpenAI-compatible server

`ANEFieldfareServer` serves the model over the OpenAI HTTP API, so chat apps
that speak it (Open WebUI, LibreChat, Jan, Msty, Chatbox, SillyTavern,
continue.dev, the `openai` SDKs) can use it. It keeps one model in memory and
generates through the same loop as the CLI: for the same prompt and settings
it returns the CLI's text.

```bash
Scripts/build.sh     # also builds the server

.build/swift-hotpath/release/ANEFieldfareServer \
  --model checkpoints/gturbo/gemma4.gturbo
# [server] listening on http://127.0.0.1:8080/v1 (model id gemma-4-26b-a4b-it)

curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-4-26b-a4b-it", "temperature": 0, "stream": true,
  "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

Point a client at base URL `http://127.0.0.1:8080/v1` with any API key (or the
one given to `--api-key`). In Open WebUI: *Admin Settings → Connections →
OpenAI API → +*, URL `http://127.0.0.1:8080/v1` (from Open WebUI in Docker,
`http://host.docker.internal:8080/v1` with `--host 0.0.0.0`). Clients that
expect OpenRouter's base URL can use `http://127.0.0.1:8080/api/v1`.

| Server option | What it does |
| --- | --- |
| `--model <dir>` | The `.gturbo` model. Required. |
| `--drafter <dir\|off>` | DFlash drafter, if not in the default place: greedy and sampled requests decode speculatively (not with a repetition penalty). `off` turns it off. |
| `--host <address>`, `--port <n>` | Where to listen. Default `127.0.0.1:8080`; `--host 0.0.0.0` for other machines. |
| `--max-context <n>` | Prompt plus completion limit in tokens. Default 32768. |
| `--model-id <id>` | The id `/v1/models` advertises. Default `gemma-4-26b-a4b-it`. Requests may name any model. |
| `--api-key <key>` | Require `Authorization: Bearer <key>` on `/v1` (not on `/health`). Also `ANE_FIELDFARE_API_KEY`. |
| `--max-queue <n>` | Requests that may wait while one generates. Default 8; beyond that, 503. |
| `--temperature`, `--top-k`, `--top-p`, `--max-tokens` | Defaults for requests that leave the field out. Default 0.2, 64, 0.95 (the CLI's), and the rest of the context. |
| `--backend`, `--fault-policy`, `--early-expert-read` | As for the CLI. |

**Endpoints.** `GET /v1/models`, `GET /v1/models/{id}`,
`POST /v1/chat/completions`, `POST /v1/completions` (raw prompt: `<bos>` plus
the text, like `--prompt`; or token IDs), `GET /health`; every `/v1` route also
under `/api/v1`. CORS is open (`Access-Control-Allow-Origin: *`, preflights
answered), so browser apps can call it.

**Request fields.** `messages` (`system`, `developer`, `user`, `assistant`;
string content or an array of `text` parts), `max_tokens` /
`max_completion_tokens` (clamped to the context), `temperature`, `top_p`,
`top_k` (0 or -1: off), `repetition_penalty`, `stop` (string or up to 16
strings, matched on the decoded text, also across token boundaries while
streaming), `seed`, `stream`, `stream_options.include_usage`, `n` (1 only).
Non-empty `tools`, a `response_format` other than `text`, and non-zero
`frequency_penalty` or `presence_penalty` are refused with 400; other fields,
`model` and `user` among them, are ignored. Responses carry `usage`, and a
llama.cpp-style `timings` object with tokens per second and the decode path.
Streams send a first chunk with `role`, content deltas, a chunk with
`finish_reason` (`stop` or `length`), the usage chunk if asked, and
`data: [DONE]`.

**Decoding.** As in the CLI: temperature 0 is greedy (the hot-vocabulary head
without a drafter), anything above it samples on the GPU, and with a drafter
loaded both decode speculatively, with the same text for a given `seed` as
without it. A request with a repetition penalty samples on the host, without
the drafter.

**Concurrency.** One generation runs at a time; others wait in order, up to
`--max-queue`, and further requests get `503` with `Retry-After`. A client that
disconnects cancels its generation (between tokens, or between prefill chunks)
or leaves the line. The server keeps the last request's KV cache: when the
next request starts with the same tokens and decodes the same way (a chat
re-sending its history), only the new part of the prompt is prefilled.

**Errors** use OpenAI's shape, `{"error": {"message", "type", "param",
"code"}}`: 400 for invalid requests, 401 for a missing or wrong key, 404, 405,
413 (`context_length_exceeded`) when the prompt does not fit, 503
(`server_busy`), 500.

**Limitations.** Text only: image parts are refused with a 400 (the CLI's
image path needs the vision pack, and is not wired to the API). No tool
calling, logprobs, `n` > 1, `echo` or JSON mode. `system` messages must come
before the conversation. A client that half-closes its socket after sending
a request is taken as gone. The listener speaks plain HTTP/1.1; put a TLS proxy
in front of it before exposing it beyond the machine.

`Scripts/server/smoke.sh` checks the server against the CLI on the real model:
greedy chat (whole and streamed), seeded sampling and a raw completion must
return the CLI's text. Run it under the benchmark lock:
`Scripts/with_benchmark_lock.sh Scripts/server/smoke.sh`
(`SMOKE_DRAFTER=<dir>` adds a drafter run).

## How it works

### One forward pass

```
tokens → embed → 30 × [ attention → router → shared FFN + 8 routed experts → layer tail ] → head → next token
```

- **Attention** mixes information between tokens.
- The **router** scores all 128 experts for each token and picks the top 8.
- The **shared FFN** is one feed-forward block every token goes through.
- The 8 **routed experts** are small feed-forward blocks; only the chosen ones
  run.
- The **layer tail** is the normalization and residual add that finish a layer.
- The **head** turns the final hidden state into a score for each of the
  262,144 vocabulary tokens.

Generation has two phases. **Prefill** processes the whole prompt, many tokens
at once. **Decode** then produces the answer one token at a time.

### Where the expert weights live

Each expert's weights sit at a fixed spot in a per-layer file, which the
runtime maps into memory read-only. Every expert gets one Metal buffer pointing
straight at that mapped region, so nothing is copied and the runtime has no
cache of its own: the macOS file cache *is* the expert cache. It grows into
whatever memory is free and shrinks when other apps need memory.

Before running a layer, the runtime asks the OS (`mincore`) which of the
chosen experts are already in memory. Kernels for those start at once, while
the missing ones are read from the SSD in parallel. The answer never depends on
this: if a page is still missing, the GPU simply waits for it.

To hide SSD latency, after finishing a layer's reads the runtime asks the OS to
start reading the experts the *next* layer will probably need (an "early
read"). It predicts them with the **next-layer guess**: a small matrix per
layer, fitted offline on routing traces, that predicts the next layer's router
choices from the current hidden state (`Sources/ANEFieldfareHotpath/NextLayerGuess/`).

Code: `Sources/ANEFieldfareHotpath/Infrastructure/Streaming/MappedExpertLayer.swift`.

### GPU and Neural Engine

The GPU (Metal) implementation is complete: it can run every operation, and it
is the reference the ANE path is compared against. The ANE is used only for one
job, routed experts during prefill, and the code treats it as optional.

- The **supervisor** (`Runtime/Supervisor/DeviceSupervisor.swift`) decides at
  startup which layers use the ANE and answers "does the ANE run layer L?" for
  each layer. If the ANE was requested but cannot be set up, startup fails
  rather than quietly falling back to the GPU.
- If an ANE call fails during a request, the supervisor records the failure.
  With `fail-closed` the request stops. With `failover-metal` the ANE is
  dropped for the rest of that run (it is never re-enabled) and the GPU redoes
  the failed layer and all later ones.
- The **ANE carrier** (`Sources/ANEFieldfareANECarrier`) is a small
  Objective-C wrapper over Apple's private ANE runtime. It compiles one ANE
  program that computes 4 experts at a time, for up to 128 tokens each. It has
  two input/output buffer sets ("banks") so the GPU can fill one while the ANE
  works on the other.
- The ANE cannot unpack the model's 4-bit weights, so for every ANE call the GPU
  first expands the 4 experts' weights to FP16 and passes them in as ordinary
  inputs. That expansion is the main limit on ANE speed.

### Speculative decoding

Speculative decoding (`Runtime/DFlash/`) uses a small DFlash draft model to propose a
block of up to 16 next tokens. The main model then **verifies** the block in
one pass: it computes its own choice at each position and keeps the drafted
tokens up to the first one it disagrees with, plus its own token there.

On this model the expensive part is verifying, not drafting: each extra token
in a block may pick different experts, so an 8-token block reads about 29
experts per layer instead of 8. The scheduler (`SpeculationPolicy`) therefore
decides each step, from measured timings and recent acceptance rates, how many
tokens to draft and how many of those to verify, and drafts nothing when plain
decoding would be faster.

The pipeline is built so the GPU never waits for the CPU:

- **Verify routes experts on the GPU.** Each layer's router also guesses the
  next layer's experts for every row. Those guesses (every resident expert, for
  layer 0) are bound while the GPU runs the layer, and a planning kernel picks
  out the ones the rows actually chose. The CPU only patches the rare route
  that wasn't guessed or whose expert is still on disk. It plans the same
  routes as the CPU would; the tests check this end to end, by comparing
  speculative output with plain decoding.
- **The drafter's inputs never pass through the CPU.** The main model copies
  the six hidden-state layers the drafter reads into a ring buffer on the GPU,
  as part of each layer's work.
- **One wait per step.** Behind each verify, on the GPU, the drafter adds the
  verified tokens to its context, compares its drafts with the main model's
  choices, moves to the main model's next token and drafts the next block. The
  next verify reads that block straight from the drafter's buffer.
- **Weights from the file cache.** The drafter's 4-bit weights are installed
  once into a file and read in place, like the experts.

- **Sampling keeps the speedup.** With a temperature above 0, every token is
  drawn the same way everywhere: the largest of (logit ÷ T + noise), over the
  tokens top-k and top-p allow, where the noise depends only on the seed, the
  token's position and the token (the Gumbel-max trick, an exact sample from
  the tempered distribution). The drafter draws its guesses with the same
  noise, so a draft is kept exactly when it equals the main model's own draw.
  The output is therefore the main model's sampled text for that seed, the
  same text plain decoding draws, and on the quick suite sampling at
  temperature 0.6–1.0 ran 1.4–1.9× faster with the drafter, as greedy does.

Speculative output is the same as plain output for the same settings except
where two tokens score almost equally: verifying several tokens at once does
the arithmetic in a different order, which can flip a near-tie. In greedy
testing the first such difference came after 90 to 356 tokens; with sampling
it is a near-tie in the noisy race, as rare (one of twelve 192-token sampled
runs differed, from token 32).

### Hot-vocabulary shortcut

The head is the single largest weight read in decode (262,144 rows). Plain
greedy decoding (`Runtime/Generation/HotHeadDecoder.swift`) scores only a list
of 56k common tokens for three steps, then on the fourth step runs the full head
over all four saved positions at once. If any of the three guesses was wrong,
it is replaced with the correct token and the steps after it are discarded and
redone. The output is therefore exactly what full-head decoding produces; the
list only affects speed. The list ships in `HotVocabulary/vocab-262144.bin` and
was built by `Scripts/hotvocab/build.py`.

## Performance

M5, 16 GB, macOS 27. Greedy decoding, 192 generated tokens, from the benchmark
quick suite:

| Prompt | `--drafter off` | Default (drafter) |
| --- | ---: | ---: |
| code | 51.2 tok/s | 98.7 tok/s |
| math word problem | 49.1 tok/s | 78.3 tok/s |
| explanation | 49.8 tok/s | 72.7 tok/s |

Prefill of a 3,015-token prompt: 8.8 s with `hybrid-ane`, 15.3 s with `metal`
(single runs).

How these numbers were reached, step by step, including what was tried and
rejected: [PERFORMANCE.md](PERFORMANCE.md).

## Testing

```bash
Scripts/test.sh    # lint, build, then every test suite
```

Most tests use small synthetic model files the tests generate themselves (the
code calls these *fixtures*). The tokenizer and server tests download Gemma's
tokenizer from the Hugging Face Hub on first run. Some tests call the real
ANE and need ANE hardware. Tests that need the real model run when it is at the
default path or `ANE_FIELDFARE_TEST_CHECKPOINT`, and are skipped otherwise.

Important checks:

- **Failover gives the GPU answer** (`DeviceSupervisorFailoverTests`, needs the
  model). A fake ANE that writes garbage (NaN) into the output and then reports
  failure must leave the final logits bit-identical to a GPU-only run. A
  fault injected around the real ANE at layer 7 must match a run that used the
  ANE for layers 0–6 only. With `fail-closed` the request must stop.
- **Shortcuts are exact where they claim to be.** The hot-vocabulary head and
  the multi-token matrix × vector kernel (up to 4 tokens) are tested to give
  bit-identical results to the one-token path. The matrix-unit kernel used for
  5–16 tokens is tested against a CPU reference at 7 and 12 tokens, within
  0.5% relative error.

## Contributing

- **Concurrency by design.** The code builds under Swift 6 strict
  concurrency without silencing it. Shared mutable state has one owner (a
  `Mutex` or an actor), and values that cross isolation are `Sendable` because
  their parts are. Types holding `MTLBuffer`, `MTLCommandBuffer` or raw
  pointers stay inside the task that owns them; only plans and indices cross.
  Blocking file I/O leaves the cooperative pool through `BlockingIO`.
- **Lint.** `Scripts/lint_escape_hatches.py` (run by `Scripts/test.sh`)
  rejects `@unchecked Sendable`, `nonisolated(unsafe)`, `@preconcurrency`,
  `assumeIsolated`, `DispatchQueue` and `Task.detached` unless a `// SAFETY:`
  comment states the invariant the compiler cannot check, and rejects `try!`,
  `as!` and force unwraps in `Sources/`.
- **Receipts.** A performance or correctness claim comes with a machine
  receipt: host, OS build, exact command, exit status and the measured result.

## Benchmarking a change

`ANEFieldfareBench` loads the model once and runs a fixed set of prompts, with
and without the drafter, plus timings of single decode and verify steps. To
compare two builds:

```bash
Scripts/bench/snapshot.sh base        # save a copy of the current build as "base"
# ...change something...
Scripts/bench/snapshot.sh try
Scripts/bench/ab.sh base try 2 --drafter checkpoints/dflash/gemma-4-26B-A4B-it-DFlash
```

`ab.sh` alternates the two builds in A-B-B-A order, each in a fresh process,
so thermal and background drift affects both equally. It writes an HTML report
to `.build/bench-runs/` with each result as a ratio (try / base) and a 95%
confidence interval. A difference is reported only if it is larger than the
**noise floor**: the largest difference seen between two runs of the *same*
build. The quick suite takes about 90 s.

## Repository layout

| Path | Contents |
| --- | --- |
| `Sources/ANEFieldfareHotpath` | The runtime: model loading, expert store, Metal kernels, forward pass, supervisor, speculative decoding, tokenizer. |
| `Sources/ANEFieldfareHotpathCLI`, `…CLICore` | The command-line program. |
| `Sources/ANEFieldfareServer`, `…ServerCore` | The OpenAI-compatible HTTP server: HTTP/1.1 on Network.framework, the OpenAI request and response shapes, and the serialized engine. |
| `Sources/ANEFieldfareANECarrier` | The Objective-C ANE wrapper. |
| `Sources/ANEFieldfareImportedFormat` | Reader for the `.gturbo` format. |
| `Sources/ANEFieldfareBench` | The benchmark program used by `Scripts/bench/`. |
| `Sources/ANEFieldfareDFlashCapture` | Records hidden states and routes for drafter work (`Scripts/dflash/`). |
| `Tests/` | Test suites, CPU reference implementations and fixture generators. |
| `Scripts/` | Build, test, lint and benchmark scripts. |
| `PERFORMANCE.md` | The optimization history: every change, its measurements, and what was tried and rejected. |

## License and credits

Apache License 2.0; see [LICENSE](LICENSE). Parts are adapted from
[Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare) by drumih (Apache
2.0): the Swift runtime, the `.gturbo` format, the CLI and the upstream
regression tests. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[UPSTREAM.json](UPSTREAM.json) for what came from where.

Gemma is a model family from Google. Its weights are distributed under Google's
Gemma terms and are not included in this repository.
