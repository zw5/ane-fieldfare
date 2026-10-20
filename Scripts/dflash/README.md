# DFlash acceptance gate

Offline check of the z-lab DFlash drafter against this runtime's INT4 target.
The drafter must be at `checkpoints/dflash/gemma-4-26B-A4B-it-DFlash`. From
the repository root, with an empty output directory `$OUT`:

```bash
B=.build/swift-hotpath/release/ANEFieldfareDFlashCapture CK=checkpoints/gturbo/gemma4.gturbo
python3 Scripts/dflash/gen.py 256 $OUT              # greedy target continuations with token IDs
$B embedding $CK $OUT/embed                         # tied INT4 embedding for the drafter
for n in code math explain chat; do $B capture $CK $OUT/$n.gen.json $OUT/$n; done
python3 Scripts/dflash/dflash_ref.py $OUT 53.0660 16,8,4   # acceptance lengths into $OUT/acceptance.json
```

`53.0660` is √2816, the embedding scale; the last argument lists the block
sizes to simulate.

## Runtime checks

`ANEFieldfareDFlashCapture` also drives the runtime implementation
(`Sources/ANEFieldfareHotpath/Runtime/DFlash`):

```bash
$B verify-check $CK $OUT/code.gen.json [rows]                 # teacher-forced verify == greedy decode (rows default 16)
$B draft-check $CK $DRAFTER $OUT/code.gen.json 200,150,100    # drafts at descending anchors, compare with dflash_ref.py
$B generate $CK $DRAFTER $OUT/code.gen.json 16                # speculative greedy (blocks of at most 16) vs recorded plain greedy
```

`ANE_FIELDFARE_HOST_ROUTED_VERIFY=1` runs `verify-check` with every layer's
experts planned on the CPU, for comparison with the GPU-routed default.
