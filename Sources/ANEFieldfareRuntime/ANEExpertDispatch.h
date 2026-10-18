#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ANEProgram.h"
#include "RoutedLayer.h"

typedef struct ANEFieldfareANEExpertDispatcher ANEFieldfareANEExpertDispatcher;

typedef struct {
    ANEFieldfareANEProgramInfo program;
    uint64_t dispatchCount;
    double totalPreparationMilliseconds;
    double totalEvaluationMilliseconds;
    uint64_t explicitReadBytes;
    double explicitReadMilliseconds;
    bool explicitReadEnabled;
    uint64_t layerPrefetchRequests;
    uint64_t layerPrefetchExpertHints;
    uint64_t layerPrefetchHintBytes;
    uint64_t layerPrefetchFailures;
    double layerPrefetchWaitMilliseconds;
    uint32_t lastBank;
} ANEFieldfareANEExpertDispatchInfo;

// Creates one compact Gemma 4 expert program and eight explicitly mapped request
// banks. packedLayerPath must name the selected read-only layer_XX.bin.
ANEFieldfareANEExpertDispatcher *ANEFieldfareANEExpertDispatcherCreate(
    const char *packedLayerPath,
    const char **errorMessage);

// Two device-filled request banks. Weights remain ordinary runtime inputs;
// Metal expands the validated compact record into the declared FP16 layout.
// Layout is [batch,2816,2176]: 64 activation lanes, then gate/up/down weights.
// Activation lanes MUST be multiplied by 64. The program reverses this scale
// before GELU and after the down projection; output values are unscaled FP32.
// Each batch item owns a separate expert. Supported batch sizes are 1 and 4.
// Keep the input bank unchanged until EvaluateDeviceCarrier completes.
ANEFieldfareANEExpertDispatcher *ANEFieldfareANEExpertDispatcherCreateDeviceCarrier(
    uint32_t batchExperts, const char **errorMessage);
IOSurfaceRef ANEFieldfareANEExpertDispatcherInput(ANEFieldfareANEExpertDispatcher *, uint32_t bank);
IOSurfaceRef ANEFieldfareANEExpertDispatcherOutput(ANEFieldfareANEExpertDispatcher *, uint32_t bank);
bool ANEFieldfareANEExpertDispatcherSetOutputEvent(ANEFieldfareANEExpertDispatcher *,
    uint32_t bank, CFTypeRef event, uint64_t value, const char **errorMessage);
bool ANEFieldfareANEExpertDispatcherEvaluateDeviceCarrier(ANEFieldfareANEExpertDispatcher *,
    uint32_t bank, const char **errorMessage);

// Atomically replaces only the read-only compact expert mapping. The compiled
// and loaded program, mapped IOSurfaces, and all eight request banks are preserved.
// A failed replacement leaves the previous packed layer active.
bool ANEFieldfareANEExpertDispatcherSetPackedLayerPath(
    ANEFieldfareANEExpertDispatcher *dispatcher,
    const char *packedLayerPath,
    const char **errorMessage);

// Issues non-semantic L+1 F_RDADVISE storage hints through a separate descriptor.
// WaitForLayerPrefetch releases that descriptor; it is not an I/O-completion
// oracle. No expert mapping or active-layer selection is changed by hints.
bool ANEFieldfareANEExpertDispatcherPrefetchLayerExperts(
    ANEFieldfareANEExpertDispatcher *dispatcher,
    const char *packedLayerPath,
    const uint32_t *experts,
    uint32_t expertCount);
void ANEFieldfareANEExpertDispatcherWaitForLayerPrefetch(
    ANEFieldfareANEExpertDispatcher *dispatcher);

// Signature-compatible with ANEFieldfareExpertDispatch. The normalized input
// is quantized to the program's FP16 carrier; the selected compact expert is
// expanded into the next alternating bank, evaluated, and returned as FP32.
bool ANEFieldfareANEExpertDispatcherCall(
    void *dispatcher,
    uint32_t expert,
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]);

bool ANEFieldfareANEExpertDispatcherCallBatch(
    void *dispatcher,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize]);

// Evaluate one expert for 1...64 position-major input rows in one request.
bool ANEFieldfareANEExpertDispatcherCallPositions(
    void *dispatcher, uint32_t expert, uint32_t count,
    const float *inputs, float *outputs);

ANEFieldfareANEExpertDispatchInfo ANEFieldfareANEExpertDispatcherGetInfo(
    const ANEFieldfareANEExpertDispatcher *dispatcher);
const char *ANEFieldfareANEExpertDispatcherLastError(
    const ANEFieldfareANEExpertDispatcher *dispatcher);
void ANEFieldfareANEExpertDispatcherDestroy(ANEFieldfareANEExpertDispatcher *dispatcher);
