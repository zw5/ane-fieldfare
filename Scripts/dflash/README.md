# DFlash acceptance gate

Offline check of the z-lab DFlash drafter against this runtime's INT4 target,
recorded in `results/dflash-acceptance-gate-m5-macos27.json`. Run from an
empty output directory:

```bash
python3 Scripts/dflash/gen.py 256 <dir> [slots]      # greedy target continuations with token IDs
B=.build/swift-hotpath/release/ANEFieldfareDFlashCapture CK=checkpoints/gturbo/gemma4.gturbo
$B embedding $CK <dir>/embed                        # tied INT4 embedding for the drafter
for n in code math explain chat; do $B capture $CK <dir>/$n.gen.json <dir>/$n; done
(cd <dir> && python3 .../dflash_ref.py . 53.0660 16,8,4)   # acceptance lengths
(cd <dir> && python3 .../union.py && python3 .../model.py) # route union, cache/time model
```

The drafter must be at `checkpoints/dflash/gemma-4-26B-A4B-it-DFlash`.

## Runtime checks

`ANEFieldfareDFlashCapture` also drives the runtime implementation
(`Sources/ANEFieldfareHotpath/Runtime/DFlash`). Experts stream through the OS
page cache; there is no expert cache to size. The drafter's INT4 weights are
installed on first use as `fieldfare-dflash-int4.bin` next to its checkpoint.

```bash
$B verify-check $CK <dir>/code.gen.json [rows]                # teacher-forced verify == greedy decode (rows default 16)
$B draft-check $CK $DRAFTER <dir>/code.gen.json 200,150,100   # Metal drafts at descending anchors, compare with dflash_ref.py
$B generate $CK $DRAFTER <dir>/code.gen.json 16               # speculative greedy (blocks of at most 16) vs recorded plain greedy
```

`ANE_FIELDFARE_HOST_ROUTED_VERIFY=1` runs `verify-check` with every layer's
experts planned on the CPU, for comparison with the GPU-routed default.
