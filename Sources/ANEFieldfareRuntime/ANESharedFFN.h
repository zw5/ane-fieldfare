#pragma once
#include "ANEProgram.h"

// Constant-weight shared FFN. Activations use channel-major [D, positions] FP16
// carriers. This is an immutable program, not prepared mutable weights.
typedef struct ANEFieldfareANESharedFFN ANEFieldfareANESharedFFN;
typedef struct {
    uint64_t constantBytes, evaluations;
    uint32_t compileCount, loadCount;
    double compileMilliseconds, loadMilliseconds, evaluationMilliseconds;
} ANEFieldfareANESharedFFNInfo;

ANEFieldfareANESharedFFN *ANEFieldfareANESharedFFNCreate(
    const char *weightsPath, uint32_t layer, uint32_t hidden,
    uint32_t intermediate, uint32_t bits, uint32_t positions, const char **error);
IOSurfaceRef ANEFieldfareANESharedFFNInput(ANEFieldfareANESharedFFN *ffn);
IOSurfaceRef ANEFieldfareANESharedFFNOutput(ANEFieldfareANESharedFFN *ffn);
bool ANEFieldfareANESharedFFNBegin(ANEFieldfareANESharedFFN *ffn, const char **error);
bool ANEFieldfareANESharedFFNSetOutputEvent(ANEFieldfareANESharedFFN *ffn,
    CFTypeRef event, uint64_t value, const char **error);
bool ANEFieldfareANESharedFFNWait(ANEFieldfareANESharedFFN *ffn,
                                uint64_t timeoutNanoseconds, const char **error);
ANEFieldfareANESharedFFNInfo ANEFieldfareANESharedFFNGetInfo(ANEFieldfareANESharedFFN *ffn);
void ANEFieldfareANESharedFFNDestroy(ANEFieldfareANESharedFFN *ffn);
