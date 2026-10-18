#pragma once

#include "Attention.h"
#include "RoutedLayer.h"

typedef struct ANEFieldfareDecoder ANEFieldfareDecoder;

typedef struct {
    uint64_t tokenCount;
    uint64_t layerCount;
    double inputNormalizationMilliseconds;
    double attentionMilliseconds;
    double postAttentionMilliseconds;
    double routedFeedForwardMilliseconds;
} ANEFieldfareDecoderTiming;

typedef bool (*ANEFieldfareDecoderExpertDispatch)(
    void *context,
    uint32_t layer,
    uint32_t expert,
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]);
typedef bool (*ANEFieldfareDecoderExpertBatchDispatch)(
    void *context,
    uint32_t layer,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize]);
typedef bool (*ANEFieldfareDecoderExpertReducedDispatch)(
    void *context,
    uint32_t layer,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float weights[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float reduced[ANEFieldfareGemma4HiddenSize]);
typedef bool (*ANEFieldfareDecoderExpertReducedPositionDispatch)(
    void *context,
    uint32_t layer,
    uint32_t positionCount,
    const uint32_t *experts,
    const float *weights,
    const float *inputs,
    float *reduced);
// Executes one expert for several prompt positions. inputs and outputs contain
// positionCount contiguous hidden-size rows. A prefill implementation can use
// this seam to read/prepare an expert once for every position that selected it.
typedef bool (*ANEFieldfareDecoderExpertPositionDispatch)(
    void *context,
    uint32_t layer,
    uint32_t expert,
    uint32_t positionCount,
    const float *inputs,
    float *outputs);

// Executes a complete attention block for a position tile, including QKV
// projections, causal KV mutation, and output projection. This is the seam for
// a device-resident ANE/Metal pipeline that must avoid returning projected
// tensors through the host between engines.
typedef bool (*ANEFieldfareAttentionDevicePositionDispatch)(
    void *context,
    uint32_t layer,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    const uint8_t *tensorIndexBase,
    uint64_t firstPosition,
    uint32_t positionCount,
    const float *normalizedHiddenRows,
    float *outputRows);

bool ANEFieldfareDecoderOpen(const char *modelDirectory,
                             uint64_t maximumContext,
                             ANEFieldfareDecoder **decoder);
void ANEFieldfareDecoderClose(ANEFieldfareDecoder *decoder);
void ANEFieldfareDecoderResetTiming(ANEFieldfareDecoder *decoder);
ANEFieldfareDecoderTiming ANEFieldfareDecoderGetTiming(const ANEFieldfareDecoder *decoder);
bool ANEFieldfareDecoderToken(ANEFieldfareDecoder *decoder,
                              uint64_t position,
                              float hidden[ANEFieldfareGemma4HiddenSize],
                              ANEFieldfareDecoderExpertDispatch expertDispatch,
                              void *expertContext);
bool ANEFieldfareDecoderTokenBatched(ANEFieldfareDecoder *decoder,
                                     uint64_t position,
                                     float hidden[ANEFieldfareGemma4HiddenSize],
                                     ANEFieldfareDecoderExpertBatchDispatch expertDispatch,
                                     void *expertContext);
bool ANEFieldfareDecoderTokenReducedBatched(
    ANEFieldfareDecoder *decoder,
    uint64_t position,
    float hidden[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareDecoderExpertReducedDispatch expertDispatch,
    void *expertContext);

bool ANEFieldfareDecoderTokenHybrid(
    ANEFieldfareDecoder *decoder,
    uint64_t position,
    float hidden[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareDecoderExpertReducedDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext);

bool ANEFieldfareDecoderTokenFused(
    ANEFieldfareDecoder *decoder,
    uint64_t position,
    float hidden[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareAttentionInputDispatch attentionInputDispatch,
    ANEFieldfareAttentionAffineDispatch attentionDispatch,
    void *attentionContext,
    ANEFieldfareDecoderExpertReducedDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext);

// Optional complete-device attention seam for one-token decode. The dispatcher
// owns QKV, causal KV mutation/attention, and output projection for each layer;
// passing NULL retains the independently validated dispatched CPU middle.
bool ANEFieldfareDecoderTokenFusedDevice(
    ANEFieldfareDecoder *decoder,
    uint64_t position,
    float hidden[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareAttentionInputDispatch attentionInputDispatch,
    ANEFieldfareAttentionAffineDispatch attentionDispatch,
    void *attentionContext,
    ANEFieldfareAttentionDevicePositionDispatch deviceAttentionDispatch,
    void *deviceAttentionContext,
    ANEFieldfareDecoderExpertReducedDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext);

// Runs up to 64 causally consecutive prompt positions together. The control
// entry point keeps attention in position order and joins routed experts. The
// fused entry point may batch QKV/output projections while preserving causal
// attention and KV mutation in position order. hiddenRows is row-major and is
// replaced in place with the corresponding decoder outputs.
bool ANEFieldfareDecoderPrefillBatched(
    ANEFieldfareDecoder *decoder,
    uint64_t firstPosition,
    uint32_t count,
    float *hiddenRows,
    ANEFieldfareDecoderExpertPositionDispatch expertDispatch,
    void *expertContext);

bool ANEFieldfareDecoderPrefillBatchedFused(
    ANEFieldfareDecoder *decoder,
    uint64_t firstPosition,
    uint32_t count,
    float *hiddenRows,
    ANEFieldfareAttentionInputPositionDispatch attentionInputDispatch,
    ANEFieldfareAttentionAffinePositionDispatch attentionDispatch,
    void *attentionContext,
    ANEFieldfareAttentionDevicePositionDispatch deviceAttentionDispatch,
    void *deviceAttentionContext,
    ANEFieldfareDecoderExpertPositionDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareDecoderExpertReducedDispatch reducedDispatch,
    void *reducedContext,
    ANEFieldfareDecoderExpertReducedPositionDispatch reducedPositionDispatch,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext,
    ANEFieldfareSharedFFNPositionDispatch sharedPositionDispatch);
