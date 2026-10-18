#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <CoreFoundation/CoreFoundation.h>

#include "RoutedLayer.h"
#include "Attention.h"

typedef struct ANEFieldfareMetalExpertDispatcher ANEFieldfareMetalExpertDispatcher;

typedef struct {
    uint64_t dispatchCount;
    uint64_t denseDispatchCount;
    double totalWallMilliseconds;
    double totalGPUMilliseconds;
    double totalCompactStagingMilliseconds;
    double totalExactCacheFillMilliseconds;
    double totalDenseWallMilliseconds;
    double totalDenseGPUMilliseconds;
    uint64_t attentionInputDispatchCount;
    double attentionInputWallMilliseconds;
    double attentionInputGPUMilliseconds;
    uint64_t attentionOutputDispatchCount;
    double attentionOutputWallMilliseconds;
    double attentionOutputGPUMilliseconds;
    uint64_t sharedFFNDispatchCount;
    double sharedFFNWallMilliseconds;
    double sharedFFNGPUMilliseconds;
    uint64_t headDispatchCount;
    double headWallMilliseconds;
    double headGPUMilliseconds;
    uint64_t compactLayerBytes;
    uint64_t compactModelBytes;
    uint64_t mappedExpertBytesPerDispatch;
    uint64_t compactStagingBytes;
    uint64_t exactCacheHits;
    uint64_t exactCacheMisses;
    uint64_t exactCacheCapacityBytes;
    uint32_t exactCacheSlotsPerLayer;
    uint32_t mappedLayerCount;
    uint32_t libraryCompileCount;
    uint32_t pipelineLoadCount;
    bool splitQueueEnabled;
} ANEFieldfareMetalExpertDispatchInfo;

// Maps one packed_experts/layer_XX.bin directly into unified memory and binds
// selected compact INT4 expert slices to Metal. No FP16 weight carrier exists.
ANEFieldfareMetalExpertDispatcher *ANEFieldfareMetalExpertDispatcherCreate(
    const char *packedLayerPath,
    const char *metalSourcePath,
    const char **error);

// Creates one model-scoped executor: all 30 compact expert layers are mapped
// read-only, while the Metal library, pipelines, queue, argument encoder, and
// activation buffers are created only once and reused for every layer/token.
ANEFieldfareMetalExpertDispatcher *ANEFieldfareMetalExpertDispatcherCreateModel(
    const char *modelDirectory,
    const char *metalSourcePath,
    const char **error);

// Executes all eight selected experts and their weighted reduction in one
// command buffer. Input/output activation conversion is FP16; weights remain
// in the checkpoint's compact affine INT4 representation throughout.
bool ANEFieldfareMetalExpertDispatcherCallReduced(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float weights[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]);

bool ANEFieldfareMetalExpertDispatcherCallReducedAtLayer(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    uint32_t layer,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float weights[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]);

// Turbo-style grouped prefill: sorts the position-major token/expert pairs,
// binds at most the configured cache-slot count at once, and executes all
// positions selecting those experts before advancing to the next tile.
bool ANEFieldfareMetalExpertDispatcherCallReducedPositionsAtLayer(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    uint32_t layer,
    uint32_t positionCount,
    const uint32_t *experts,
    const float *weights,
    const float *inputs,
    float *outputs);

// Executes one compact affine-INT4 tensor from model_weights.bin. The model
// dispatcher owns the file mapping; tensor pointers must refer into that same
// mapping. This is the common resident-dense rail for projections and head.
bool ANEFieldfareMetalExpertDispatcherCallAffine(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    const float *input,
    float *output);

// Greedy-only tied-head path. Reduces the compact INT4 projection on the GPU
// and returns one token/raw logit without a vocabulary-sized host carrier.
bool ANEFieldfareMetalExpertDispatcherCallGreedyHead(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    const float *input,
    uint32_t *token,
    float *winningRawLogit);
bool ANEFieldfareMetalExpertDispatcherCallGreedyHeadPositions(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    uint32_t positionCount,
    const float *inputs,
    uint32_t *tokens,
    float *winningRawLogits);

// Internal device-resident composition seam: append one compact affine to an
// existing Metal command buffer, consuming an existing FP32 MTLBuffer. The
// returned output buffer is retained until CompleteExternalAffine releases it.
bool ANEFieldfareMetalExpertDispatcherEncodeExternalAffine(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    CFTypeRef commandBuffer,
    CFTypeRef inputBuffer,
    CFTypeRef *outputBuffer);
bool ANEFieldfareMetalExpertDispatcherCompleteExternalAffine(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    CFTypeRef commandBuffer,
    CFTypeRef outputBuffer,
    uint64_t startedTicks,
    float *output);

bool ANEFieldfareMetalExpertDispatcherCallAttentionInput(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *query,
    const ANEFieldfareTensor *key,
    const ANEFieldfareTensor *value,
    bool valueEqualsKey,
    const float *input,
    float *queryOutput,
    float *keyOutput,
    float *valueOutput);

// Decode-only fused attention rail. Q/K/V compact projections, per-head
// normalization/RoPE, KV mutation, causal attention, and compact o_proj share
// one Metal command buffer and one completion wait.
bool ANEFieldfareMetalExpertDispatcherCallProjectedAttention(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t position,
    const float normalizedHidden[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]);

// Batched prefill variants. Inputs and outputs are position-major contiguous
// rows; count is bounded to the runtime's 64-position accelerator tile.
bool ANEFieldfareMetalExpertDispatcherCallAffinePositions(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    uint32_t count,
    const float *inputs,
    float *outputs);

bool ANEFieldfareMetalExpertDispatcherCallAttentionInputPositions(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
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

// Executes gate/up, GELU product, and down for the shared FFN in one Metal
// command buffer, avoiding three host round trips.
bool ANEFieldfareMetalExpertDispatcherCallSharedFFN(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *gate,
    const ANEFieldfareTensor *up,
    const ANEFieldfareTensor *down,
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]);

bool ANEFieldfareMetalExpertDispatcherCallSharedFFNPositions(
    ANEFieldfareMetalExpertDispatcher *dispatcher,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *gate,
    const ANEFieldfareTensor *up,
    const ANEFieldfareTensor *down,
    uint32_t count,
    const float *inputs,
    float *outputs);

ANEFieldfareMetalExpertDispatchInfo ANEFieldfareMetalExpertDispatcherGetInfo(
    const ANEFieldfareMetalExpertDispatcher *dispatcher);
const char *ANEFieldfareMetalExpertDispatcherLastError(
    const ANEFieldfareMetalExpertDispatcher *dispatcher);
void ANEFieldfareMetalExpertDispatcherDestroy(
    ANEFieldfareMetalExpertDispatcher *dispatcher);
