#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "TensorIndex.h"

typedef enum {
    ANEFieldfareAttentionSliding = 0,
    ANEFieldfareAttentionFull = 1,
} ANEFieldfareAttentionKind;

typedef struct {
    uint32_t layer;
    ANEFieldfareAttentionKind kind;
    uint32_t hiddenSize;
    uint32_t queryHeads;
    uint32_t keyValueHeads;
    uint32_t headDimension;
    uint32_t rotaryAngles;
    uint32_t slidingWindow;
    float ropeTheta;
    float rmsEpsilon;
    float attentionScale;
    // Zero means disabled. The installed Gemma 4 text checkpoint has no
    // attention-logit softcap; this is deliberately distinct from its final
    // logit softcap and from Gemma 4 audio attention.
    float attentionSoftcap;
    bool valueEqualsKeyProjection;
    bool sharesKeyValueAcrossLayers;
} ANEFieldfareAttentionDescriptor;

typedef struct {
    ANEFieldfareAttentionDescriptor descriptor;
    ANEFieldfareTensor queryProjection;
    ANEFieldfareTensor keyProjection;
    ANEFieldfareTensor valueProjection;
    ANEFieldfareTensor outputProjection;
    ANEFieldfareTensor queryNorm;
    ANEFieldfareTensor keyNorm;
    // Decoded once when the layer is loaded. Gemma 4's largest attention head
    // is 512-wide and uses at most 128 active RoPE frequencies.
    float queryNormValues[512];
    float keyNormValues[512];
    double ropeInverseFrequencies[128];
} ANEFieldfareAttentionWeights;

typedef bool (*ANEFieldfareAttentionAffineDispatch)(
    void *context,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    const float *input,
    float *output);
typedef bool (*ANEFieldfareAttentionInputDispatch)(
    void *context,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *query,
    const ANEFieldfareTensor *key,
    const ANEFieldfareTensor *value,
    bool valueEqualsKey,
    const float *input,
    float *queryOutput,
    float *keyOutput,
    float *valueOutput);
typedef bool (*ANEFieldfareAttentionAffinePositionDispatch)(
    void *context,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *matrix,
    uint32_t positionCount,
    const float *inputs,
    float *outputs);
typedef bool (*ANEFieldfareAttentionInputPositionDispatch)(
    void *context,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *query,
    const ANEFieldfareTensor *key,
    const ANEFieldfareTensor *value,
    bool valueEqualsKey,
    uint32_t positionCount,
    const float *inputs,
    float *queryOutputs,
    float *keyOutputs,
    float *valueOutputs);

// A cache owns float K/V storage for exactly one model layer. Sliding layers
// use a tagged ring; full-attention layers use absolute slots.
typedef struct {
    ANEFieldfareAttentionDescriptor descriptor;
    uint64_t capacity;
    uint64_t *positions;
    float *keys;
    float *values;
    // One-token decode workspace retained with the cache to keep allocation out
    // of the 30-layer recurrent path. This cache is intentionally single-call.
    float *scratchQuery;
    float *scratchKey;
    float *scratchValue;
    float *scratchCombined;
    float *scratchScores;
} ANEFieldfareKVCache;

bool ANEFieldfareAttentionDescriptorForLayer(uint32_t layer,
                                             ANEFieldfareAttentionDescriptor *descriptor);
bool ANEFieldfareAttentionWeightsLoad(const ANEFieldfareTensorIndex *index,
                                      uint32_t layer,
                                      ANEFieldfareAttentionWeights *weights);

bool ANEFieldfareKVCacheInitialize(ANEFieldfareKVCache *cache,
                                   ANEFieldfareAttentionDescriptor descriptor,
                                   uint64_t capacity);
void ANEFieldfareKVCacheDestroy(ANEFieldfareKVCache *cache);
void ANEFieldfareKVCacheClear(ANEFieldfareKVCache *cache);
bool ANEFieldfareKVCacheStore(ANEFieldfareKVCache *cache,
                              uint64_t position,
                              const float *key,
                              const float *value);
bool ANEFieldfareKVCacheLoad(const ANEFieldfareKVCache *cache,
                             uint64_t position,
                             float *key,
                             float *value);

// Applies Gemma 4's split-half RoPE in place. rotaryAngles is the number of
// non-zero frequency entries in each half: 128 for sliding heads and 64 for
// full heads. The remaining full-attention coordinates are unchanged.
bool ANEFieldfareAttentionApplyRoPE(float *head,
                                    uint32_t headDimension,
                                    uint32_t rotaryAngles,
                                    float theta,
                                    uint64_t position);

// Correctness-first, one-token causal CPU control. `normalizedHidden` is the
// already input-normalized 2816-wide layer input. The call projects Q/K/V,
// applies per-head norms and RoPE, commits K/V at `position`, attends over the
// available causal cache (last 1024 positions for sliding layers), and applies
// the checkpoint output projection.
bool ANEFieldfareAttentionDecode(const ANEFieldfareAttentionWeights *weights,
                                 ANEFieldfareKVCache *cache,
                                 uint64_t position,
                                 const float *normalizedHidden,
                                 float *output);

bool ANEFieldfareAttentionDecodeDispatched(
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t position,
    const float *normalizedHidden,
    const uint8_t *tensorIndexBase,
    ANEFieldfareAttentionInputDispatch inputDispatch,
    ANEFieldfareAttentionAffineDispatch affineDispatch,
    void *affineContext,
    float *output);

// Projects up to 64 consecutive rows in one backend evaluation, preserves
// causal KV mutation in row order, then projects all combined rows in one
// backend evaluation. Buffers are row-major by position.
bool ANEFieldfareAttentionPrefillDispatched(
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t firstPosition,
    uint32_t positionCount,
    const float *normalizedHiddenRows,
    const uint8_t *tensorIndexBase,
    ANEFieldfareAttentionInputPositionDispatch inputDispatch,
    ANEFieldfareAttentionAffinePositionDispatch affineDispatch,
    void *affineContext,
    float *outputRows);
