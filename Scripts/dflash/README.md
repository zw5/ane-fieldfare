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
(`Sources/ANEFieldfareHotpath/Runtime/DFlash`). `SLOTS` sets the expert cache.

```bash
$B verify-check $CK <dir>/code.gen.json                       # teacher-forced verify == greedy decode
$B draft-check $CK $DRAFTER <dir>/code.gen.json 200,150,100   # Metal drafts, compare with dflash_ref.py
SLOTS=48 $B generate $CK $DRAFTER <dir>/code.gen.json 8       # speculative greedy vs recorded plain greedy
```
