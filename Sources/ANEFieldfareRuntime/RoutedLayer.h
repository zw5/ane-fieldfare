#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "TensorIndex.h"

enum {
    ANEFieldfareGemma4HiddenSize = 2816,
    ANEFieldfareGemma4SharedIntermediateSize = 2112,
    ANEFieldfareGemma4ExpertIntermediateSize = 704,
    ANEFieldfareGemma4ExpertCount = 128,
    ANEFieldfareGemma4TopK = 8,
};

typedef struct {
    uint32_t expert;
    float probability;
    float weight;
} ANEFieldfareRoute;

typedef struct ANEFieldfareRoutedLayer ANEFieldfareRoutedLayer;

// A dispatch implementation receives one normalized 2816-wide expert input and
// must write one 2816-wide expert output. The runtime applies the routed weight
// and reduction. This is the seam used by the hot ANE expert program.
typedef bool (*ANEFieldfareExpertDispatch)(void *context,
                                           uint32_t expert,
                                           const float *input,
                                           float *output);
typedef bool (*ANEFieldfareExpertBatchDispatch)(
    void *context,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float *input,
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize]);
// A compact companion backend can reduce the eight routed experts itself,
// avoiding eight FP16 output carriers and a redundant host reduction.
typedef bool (*ANEFieldfareExpertReducedDispatch)(
    void *context,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float weights[ANEFieldfareGemma4TopK],
    const float *input,
    float reduced[ANEFieldfareGemma4HiddenSize]);
// Prefill companion rail. The backend receives every token's top-k route for
// one layer at once and returns one already-reduced routed row per position.
// experts/weights are position-major [positionCount][TopK].
typedef bool (*ANEFieldfareExpertReducedPositionDispatch)(
    void *context,
    uint32_t positionCount,
    const uint32_t *experts,
    const float *weights,
    const float *inputs,
    float *reduced);
typedef bool (*ANEFieldfareSharedFFNDispatch)(
    void *context,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *gate,
    const ANEFieldfareTensor *up,
    const ANEFieldfareTensor *down,
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]);
typedef bool (*ANEFieldfareSharedFFNPositionDispatch)(
    void *context,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *gate,
    const ANEFieldfareTensor *up,
    const ANEFieldfareTensor *down,
    uint32_t positionCount,
    const float *inputs,
    float *outputs);

// Opens model_weights.bin and the selected packed_experts/layer_XX.bin. Both
// files remain read-only and memory mapped for the lifetime of the layer.
bool ANEFieldfareRoutedLayerOpen(const char *modelDirectory,
                                 uint32_t layerIndex,
                                 ANEFieldfareRoutedLayer **layer);

// Opens a production routed layer against a caller-owned tensor index. Expert
// files are not mapped because the supplied dispatch callback owns expert
// execution. The index must outlive the layer.
bool ANEFieldfareRoutedLayerOpenForDispatch(const ANEFieldfareTensorIndex *index,
                                            uint32_t layerIndex,
                                            ANEFieldfareRoutedLayer **layer);
void ANEFieldfareRoutedLayerClose(ANEFieldfareRoutedLayer *layer);

// Gemma 4 RMSNorm. A NULL scale selects the router's scale-free variant.
// Otherwise scale must contain width entries.
bool ANEFieldfareRMSNorm(const float *input,
                         const float *scale,
                         uint32_t width,
                         float epsilon,
                         float *output);

// Computes the exact softmax -> top-k -> renormalize -> per-expert-scale route.
bool ANEFieldfareRoutedLayerRoute(const ANEFieldfareRoutedLayer *layer,
                                  const float input[ANEFieldfareGemma4HiddenSize],
                                  ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]);

// Independent compact-checkpoint CPU expert oracle. Production can replace it
// with the ANE dispatch callback while retaining identical routing/reduction.
bool ANEFieldfareRoutedLayerCPUExpert(const ANEFieldfareRoutedLayer *layer,
                                      uint32_t expert,
                                      const float input[ANEFieldfareGemma4HiddenSize],
                                      float output[ANEFieldfareGemma4HiddenSize]);

// Executes the complete Gemma 4 feed-forward half of a decoder layer:
// shared 2816->2112->2816 branch, routed top-8 branch, all branch/post norms,
// residual addition, and layer scalar. `input` is the post-attention residual.
// Pass dispatch=NULL to use the compact CPU oracle.
bool ANEFieldfareRoutedLayerForward(ANEFieldfareRoutedLayer *layer,
                                    const float input[ANEFieldfareGemma4HiddenSize],
                                    ANEFieldfareExpertDispatch dispatch,
                                    void *dispatchContext,
                                    float output[ANEFieldfareGemma4HiddenSize],
                                    ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]);
// The batched callback is invoked once on a joined helper thread while the
// caller computes the independent shared FFN. It is never invoked concurrently
// with another callback from the same ForwardBatched call, and this function
// does not return until the callback completes. Callback contexts must not
// require invocation on the caller's thread.
bool ANEFieldfareRoutedLayerForwardBatched(ANEFieldfareRoutedLayer *layer,
                                          const float input[ANEFieldfareGemma4HiddenSize],
                                          ANEFieldfareExpertBatchDispatch dispatch,
                                          void *dispatchContext,
                                          float output[ANEFieldfareGemma4HiddenSize],
                                          ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]);
bool ANEFieldfareRoutedLayerForwardBatchedHybrid(
    ANEFieldfareRoutedLayer *layer,
    const float input[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareExpertBatchDispatch dispatch,
    void *dispatchContext,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext,
    float output[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]);

// Same layer contract as ForwardBatched, but the companion backend consumes
// route weights and returns the already-reduced routed branch. The routed
// branch remains overlapped with the independent shared FFN.
bool ANEFieldfareRoutedLayerForwardReducedBatched(
    ANEFieldfareRoutedLayer *layer,
    const float input[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareExpertReducedDispatch dispatch,
    void *dispatchContext,
    float output[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]);

bool ANEFieldfareRoutedLayerForwardHybrid(
    ANEFieldfareRoutedLayer *layer,
    const float input[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareExpertReducedDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext,
    float output[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]);

// Prefill form of the hybrid layer. Shared gate/up/down is submitted once for
// all positions while compact routed experts retain their per-position top-8
// contract. Inputs and outputs are position-major contiguous rows.
bool ANEFieldfareRoutedLayerForwardPositionsHybrid(
    ANEFieldfareRoutedLayer *layer,
    uint32_t positionCount,
    const float *inputs,
    ANEFieldfareExpertReducedDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNPositionDispatch sharedDispatch,
    void *sharedContext,
    float *outputs);

bool ANEFieldfareRoutedLayerForwardPositionsGroupedHybrid(
    ANEFieldfareRoutedLayer *layer,
    uint32_t positionCount,
    const float *inputs,
    ANEFieldfareExpertReducedPositionDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNPositionDispatch sharedDispatch,
    void *sharedContext,
    float *outputs);
