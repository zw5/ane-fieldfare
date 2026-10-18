#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>

#include "Attention.h"

typedef struct ANEFieldfareMetalAttention ANEFieldfareMetalAttention;
typedef bool (*ANEFieldfareExternalAffineEncode)(
    void *context, const uint8_t *tensorIndexBase, const ANEFieldfareTensor *matrix,
    CFTypeRef commandBuffer, CFTypeRef inputBuffer, CFTypeRef *outputBuffer);
typedef bool (*ANEFieldfareExternalAffineComplete)(
    void *context, CFTypeRef commandBuffer, CFTypeRef outputBuffer,
    uint64_t startedTicks, float *output);

typedef struct {
    uint64_t dispatchCount;
    uint64_t positionCount;
    uint64_t hostWaitCount;
    double wallMilliseconds;
    double gpuMilliseconds;
    uint64_t directEventDispatchCount;
    uint64_t directEventPositionCount;
    uint64_t directEventHostWaitCount;
    double directEventWallMilliseconds;
} ANEFieldfareMetalAttentionInfo;

ANEFieldfareMetalAttention *ANEFieldfareMetalAttentionCreate(
    const char *metalSourcePath, const char **errorMessage);

// Consumes row-major FP32 projected Q/K/V, performs the exact per-head norms,
// RoPE, causal KV mutation, softmax attention, and returns row-major FP32
// combined heads. The cache's page-aligned K/V storage is wrapped directly by
// Metal; no cache copy is performed.
bool ANEFieldfareMetalAttentionCall(
    ANEFieldfareMetalAttention *attention,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t firstPosition,
    uint32_t positionCount,
    const float *query,
    const float *key,
    const float *value,
    float *combined);

// Decode composition: attention and the compact output projection share one
// Metal command buffer and one completion wait. Q/K/V still enter as FP32;
// the combined-head carrier remains device-resident between the two kernels.
bool ANEFieldfareMetalAttentionCallProjected(
    ANEFieldfareMetalAttention *attention,
    void *affineContext,
    ANEFieldfareExternalAffineEncode encodeAffine,
    ANEFieldfareExternalAffineComplete completeAffine,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    const uint8_t *tensorIndexBase,
    uint64_t firstPosition,
    uint32_t positionCount,
    const float *query,
    const float *key,
    const float *value,
    float *projected);

// Commits an entirely device-side middle stage. The command waits for the ANE
// QKV signal, reads the channel-major FP16 ANE output IOSurface directly,
// performs attention/KV, writes channel-major FP16 combined heads directly to
// the ANE output-projection input IOSurface, then signals Metal completion.
// The returned commandBuffer is retained and must be passed to CompleteDirect.
bool ANEFieldfareMetalAttentionEncodeDirect(
    ANEFieldfareMetalAttention *attention,
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t firstPosition,
    uint32_t positionCount,
    IOSurfaceRef qkvOutput,
    uint32_t queryChannels,
    uint32_t keyChannels,
    bool valueStacked,
    IOSurfaceRef outputProjectionInput,
    CFTypeRef metalSharedEvent,
    uint64_t waitValue,
    uint64_t signalValue,
    CFTypeRef *commandBuffer,
    const char **errorMessage);
bool ANEFieldfareMetalAttentionCompleteDirect(
    ANEFieldfareMetalAttention *attention,
    CFTypeRef commandBuffer,
    uint64_t startedTicks,
    const char **errorMessage);

ANEFieldfareMetalAttentionInfo ANEFieldfareMetalAttentionGetInfo(
    const ANEFieldfareMetalAttention *attention);
const char *ANEFieldfareMetalAttentionLastError(
    const ANEFieldfareMetalAttention *attention);
void ANEFieldfareMetalAttentionDestroy(ANEFieldfareMetalAttention *attention);
