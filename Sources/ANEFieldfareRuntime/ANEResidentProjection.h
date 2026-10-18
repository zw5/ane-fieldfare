#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "TensorIndex.h"
#include "Attention.h"
#include "MetalAttention.h"

// Attention input/output projections executed on the ANE as class-1 resident
// constants: the FP16 weights are baked into the loaded program through the
// BLOBFILE route and no weight bytes cross the input boundary at evaluation
// time. Only the 2816-wide activation is carried per token.
//
// This is the production wiring of the C1.2 construction retained in
// results/resident-projection-layer0-m5-macos27.json, which measured 1.126 ms
// warmed against 4.781 ms for the CPU affine oracle on one layer and validated
// the outputs against ANEFieldfareAffineMatVec. C1.2 covered one layer, one
// position, and a synthetic input; this component covers the layers it
// actually promotes, at the production geometry, on real checkpoint tensors.
//
// Why this and not the earlier rejected ANE paths: the tied head (126 ms), the
// compact expert (76 ms), and the FP16-carrier expert all shipped weights
// across the input boundary on every evaluation. Resident constants are the
// one measured construction where the ANE fetches weights at 63.9 GB/s from
// inside the loaded program instead.
//
// Residency is phase-scoped and bounded. The production construction measures
// about 72.8 MB per promoted layer, so a 30-layer promotion is about 2.18 GB.
// Long-prefill production admits that set temporarily and destroys it before
// recurrent decode; callers with a tighter envelope may supply a smaller byte
// budget. Layers beyond the budget fall back exactly to the Metal dense rail.

typedef struct ANEFieldfareANEResidentProjections ANEFieldfareANEResidentProjections;

typedef struct {
    uint32_t requestedLayers;
    uint32_t promotedLayers;
    uint32_t programCount;
    uint32_t compileCount;
    uint32_t loadCount;
    uint64_t constantBytes;
    uint64_t budgetBytes;
    double compileMilliseconds;
    double loadMilliseconds;
    uint64_t inputEvaluations;
    uint64_t inputPositions;
    double inputEvaluationMilliseconds;
    uint64_t outputEvaluations;
    uint64_t outputPositions;
    double outputEvaluationMilliseconds;
    uint64_t fallbacks;
    bool sharedEventRequested;
    bool sharedEventEnabled;
    uint64_t sharedEventEvaluations;
    uint64_t sharedEventFallbacks;
    uint64_t blockingANEHostWaits;
    uint64_t zeroTimeoutANECompletionDrains;
    uint64_t metalCompletionHostWaits;
    double sharedEventEvaluationMilliseconds;
    uint64_t devicePipelineEvaluations;
    uint64_t devicePipelinePositions;
    uint64_t devicePipelineBlockingHostWaits;
    uint64_t devicePipelineIntermediateHostCopies;
    double devicePipelineMilliseconds;
} ANEFieldfareANEResidentProjectionInfo;

// Builds one compiler-authored resident program with named QKV and output
// procedures for each promoted layer in [0, layerCount), stopping when the next
// layer's constants would exceed budgetBytes. The implementation retains the
// older two-program construction as a host/OS fallback if multi-procedure
// compilation or metadata mapping rejects. Returns NULL only on hard allocation/index failure; a budget too
// small to promote any layer yields a live object that promotes nothing and
// reports every call as a fallback.
//
// The index must outlive the returned object.
ANEFieldfareANEResidentProjections *ANEFieldfareANEResidentProjectionsCreate(
    const ANEFieldfareTensorIndex *index,
    uint32_t layerCount,
    uint64_t budgetBytes,
    const char **error);

// Both calls match the decoder's existing attention dispatch seams. They return
// false when the referenced layer was not promoted, when the supplied tensors do
// not match the baked constants, or on evaluation failure, so the caller must
// treat false as "run the Metal rail" rather than as an error. Genuine
// evaluation errors are recorded in LastError.
//
// Tensors are matched by their byte offset within the weights file, not by
// absolute pointer: the decoder maps model_weights.bin independently of the
// caller, so the same tensor has a different address in each mapping.
//
// Not thread safe. The decoder invokes attention on one thread, ahead of the
// routed-layer fork, which is the only supported caller.
bool ANEFieldfareANEResidentProjectionsCallAttentionInput(
    ANEFieldfareANEResidentProjections *projections,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *query,
    const ANEFieldfareTensor *key,
    const ANEFieldfareTensor *value,
    bool valueEqualsKey,
    const float *input,
    float *queryOutput,
    float *keyOutput,
    float *valueOutput);

// Fills the program's complete 64-position carrier in one evaluation. Inputs
// and outputs are row-major by position. count must be in [1, 64].
bool ANEFieldfareANEResidentProjectionsCallAttentionInputPositions(
    ANEFieldfareANEResidentProjections *projections,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *query,
    const ANEFieldfareTensor *key,
    const ANEFieldfareTensor *value,
    bool valueEqualsKey,
    uint32_t count,
    const float *inputs,
    float *queryOutputs,
    float *keyOutputs,
    float *valueOutputs);

bool ANEFieldfareANEResidentProjectionsCallAffine(
    ANEFieldfareANEResidentProjections *projections,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    const float *input,
    float *output);

bool ANEFieldfareANEResidentProjectionsCallAffinePositions(
    ANEFieldfareANEResidentProjections *projections,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    uint32_t count,
    const float *inputs,
    float *outputs);

// Executes the complete promoted attention span with one final host wait:
// ANE QKV -> shared-event Metal attention/KV -> shared-event ANE output
// projection. Projected Q/K/V and combined heads never return through CPU.
// Returns false without side effects when the referenced layer is unpromoted.
bool ANEFieldfareANEResidentProjectionsCallDeviceAttentionPositions(
    ANEFieldfareANEResidentProjections *projections,
    ANEFieldfareMetalAttention *metalAttention,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t firstPosition,
    uint32_t count,
    const float *normalizedInputs,
    float *outputs);

ANEFieldfareANEResidentProjectionInfo ANEFieldfareANEResidentProjectionsGetInfo(
    const ANEFieldfareANEResidentProjections *projections);
const char *ANEFieldfareANEResidentProjectionsLastError(
    const ANEFieldfareANEResidentProjections *projections);
const char *ANEFieldfareANEResidentProjectionsEventFallbackReason(
    const ANEFieldfareANEResidentProjections *projections);
void ANEFieldfareANEResidentProjectionsDestroy(
    ANEFieldfareANEResidentProjections *projections);
