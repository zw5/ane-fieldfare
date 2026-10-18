#pragma once

#include <IOSurface/IOSurface.h>
#include <stdbool.h>
#include <stdint.h>

// One compiled ANE program for Gemma 4 routed experts, with two request banks
// whose IOSurfaces the host fills on the GPU. Each bank carries 4 experts as
// [4, 2816, 2176] FP16: 64 activation lanes followed
// by the expanded gate/up/down weights, which are ordinary runtime inputs.
// Activation lanes must be scaled by 64; the program undoes the scale, and the
// output is [4, 2816, 64] unscaled FP32. Uses private AppleNeuralEngine
// classes; construction fails cleanly when they are unavailable.
typedef struct ANEExpertCarrier ANEExpertCarrier;

typedef struct {
    double compileMilliseconds;
    double loadMilliseconds;
    uint64_t evaluations;
    double evaluationMilliseconds;
} ANEExpertCarrierInfo;

// Errors are returned as strdup'd strings the caller frees.
ANEExpertCarrier *ANEExpertCarrierCreate(const char **error);
IOSurfaceRef ANEExpertCarrierInput(ANEExpertCarrier *, uint32_t bank);
IOSurfaceRef ANEExpertCarrierOutput(ANEExpertCarrier *, uint32_t bank);
// Signal `metalSharedEvent` with `value` when the bank's next evaluation
// completes, so Metal consumers can wait on it.
bool ANEExpertCarrierSetOutputEvent(ANEExpertCarrier *, uint32_t bank,
                                    CFTypeRef metalSharedEvent, uint64_t value,
                                    const char **error);
// Evaluate one bank and wait for completion (5 s timeout). Keep the bank's
// input unchanged until this returns. Calls must be serialized.
bool ANEExpertCarrierEvaluate(ANEExpertCarrier *, uint32_t bank, const char **error);
ANEExpertCarrierInfo ANEExpertCarrierGetInfo(const ANEExpertCarrier *);
void ANEExpertCarrierDestroy(ANEExpertCarrier *);
