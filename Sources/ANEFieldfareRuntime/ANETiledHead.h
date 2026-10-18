#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ANEProgram.h"
#include "TensorIndex.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANEFieldfareANETiledHead ANEFieldfareANETiledHead;

typedef struct {
    double preparationMilliseconds;
    double enqueueMilliseconds;
    double evaluationMilliseconds;
    double readbackMilliseconds;
} ANEFieldfareANEHeadTileTiming;

typedef struct {
    ANEFieldfareANEProgramInfo program;
    uint32_t inputWidth;
    uint32_t tileRows;
    uint32_t groupSize;
    uint32_t groupsPerRow;
    uint32_t requestBanks;
    uint64_t inputBytesPerBank;
    uint64_t outputBytesPerBank;
} ANEFieldfareANETiledHeadInfo;

/*
 * Compiles one reusable private-ANE program for a groupwise affine-quantized
 * projection tile. The carrier contains the activation, unpacked 4-bit codes,
 * and the original BF16 group scale/bias values. Dequantization and matmul run
 * in the ANE graph; the host never materializes a full FP16 tied-head matrix.
 *
 * Decode-shaped position counts of one are unreliable on the tested runtime.
 * Instead of spending 63 padded positions on zeros, the graph uses 64 batch
 * lanes with independent row tiles and replicates the hidden vector across the
 * lanes. Thus every padded-lane dot product produces a distinct vocabulary
 * logit. `tileRows` is the total across all 64 lanes and must be divisible by
 * 64.
 */
ANEFieldfareANETiledHead *ANEFieldfareANETiledHeadCreate(
    uint32_t inputWidth,
    uint32_t tileRows,
    uint32_t groupSize,
    uint32_t requestBanks,
    const char **errorMessage);

/*
 * Evaluates rows [firstRow, firstRow + rowCount) of `weights`. `rowCount` may
 * be smaller than the compiled tile size for the final tile. `output` must hold
 * at least rowCount floats. Only affine 4-bit matrices are currently accepted.
 */
bool ANEFieldfareANETiledHeadEvaluateTile(
    ANEFieldfareANETiledHead *head,
    uint32_t bank,
    const float *input,
    const ANEFieldfareTensor *weights,
    uint32_t firstRow,
    uint32_t rowCount,
    float *output,
    ANEFieldfareANEHeadTileTiming *timing,
    const char **errorMessage);

ANEFieldfareANETiledHeadInfo ANEFieldfareANETiledHeadGetInfo(
    ANEFieldfareANETiledHead *head);
void ANEFieldfareANETiledHeadDestroy(ANEFieldfareANETiledHead *head);

#ifdef __cplusplus
}
#endif
