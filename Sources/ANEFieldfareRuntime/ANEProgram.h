#pragma once

#include <IOSurface/IOSurface.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct ANEFieldfareANEProgram ANEFieldfareANEProgram;
typedef struct ANEFieldfareANESharedSurfaceRequestSet
    ANEFieldfareANESharedSurfaceRequestSet;

typedef struct {
    double compileMilliseconds;
    double loadMilliseconds;
    int8_t reportedQueueDepth;
    uint32_t requestBankCount;
} ANEFieldfareANEProgramInfo;

typedef struct {
    double enqueueMilliseconds;
    double completionMilliseconds;
    uint32_t completionOrdinal;
    uint32_t inFlightAfterEnqueue;
} ANEFieldfareANEEvaluationTiming;

bool ANEFieldfareANEProgramEvaluateBegin(ANEFieldfareANEProgram *program,
                                         uint32_t bank,
                                         ANEFieldfareANEEvaluationTiming *timing,
                                         const char **errorMessage);
bool ANEFieldfareANEProgramEvaluateWait(ANEFieldfareANEProgram *program,
                                        uint32_t bank,
                                        uint64_t timeoutNanoseconds,
                                        ANEFieldfareANEEvaluationTiming *timing,
                                        const char **errorMessage);
uint32_t ANEFieldfareANEProgramCurrentAsyncRequestsInFlight(
    ANEFieldfareANEProgram *program);
uint32_t ANEFieldfareANEProgramMaximumAsyncRequestsInFlight(
    ANEFieldfareANEProgram *program);
uint32_t ANEFieldfareANEProgramPerformanceStatsMask(ANEFieldfareANEProgram *program);
bool ANEFieldfareANEProgramSetPerformanceStatsMask(ANEFieldfareANEProgram *program,
                                                   uint32_t mask,
                                                   const char **errorMessage);
CFTypeRef ANEFieldfareANEProgramCopyRequestPerformanceStats(
    ANEFieldfareANEProgram *program, uint32_t bank);
CFTypeRef ANEFieldfareANEProgramCopyRequestPerformanceStatsArray(
    ANEFieldfareANEProgram *program, uint32_t bank);
CFTypeRef ANEFieldfareANEProgramCopyConfiguredPerformanceStats(
    ANEFieldfareANEProgram *program, uint32_t bank);
// Bidirectional primitive: attaches an ABI-validated _ANESharedEvents object
// containing signal events, wait events, or both to an already mapped request.
// Mapping owns the IOSurfaces; the request property owns the event descriptor
// consumed at evaluation submission. transactionHandle remains runtime-owned
// and is deliberately not synthesized here.
bool ANEFieldfareANEProgramSetRequestSharedEvents(
    ANEFieldfareANEProgram *program, uint32_t bank, CFTypeRef sharedEvents,
    const char **errorMessage);
// Builds the validated output-symbol signal descriptor over a public Metal
// shared event and attaches it to the request. All private selector encodings
// are checked before invocation. No transaction handle is created or changed.
bool ANEFieldfareANEProgramSetOutputSignalEvent(
    ANEFieldfareANEProgram *program, uint32_t bank, CFTypeRef metalSharedEvent,
    uint32_t outputSymbolIndex, uint64_t signalValue, const char **errorMessage);
// Attaches the runtime-derived wait half of the validated shared-event pair.
// The request may be submitted before Metal signals waitValue; ANE remains in
// flight without consuming its input until the event advances.
bool ANEFieldfareANEProgramSetInputWaitEvent(
    ANEFieldfareANEProgram *program, uint32_t bank, CFTypeRef metalSharedEvent,
    uint64_t waitValue, const char **errorMessage);
CFTypeRef ANEFieldfareANEProgramCopyRequestDescription(
    ANEFieldfareANEProgram *program, uint32_t bank);

// Validated request-selection primitive. It constructs and maps multiple
// compiler-named requests over one existing input/output IOSurface pair. The
// loaded program and surfaces are not duplicated. A successful set owns its
// source bank until full unmap: the original request, mapping mutation, and a
// second catalog on that carrier pair reject deterministically. Selection is
// legal only while the owned carrier pair is idle. The set must be destroyed
// before its owning program; program destruction is deferred if needed.
ANEFieldfareANESharedSurfaceRequestSet *
ANEFieldfareANEProgramCreateSharedSurfaceRequestSet(
    ANEFieldfareANEProgram *program, uint32_t sourceBank,
    const char *const *procedureNames, uint32_t procedureCount,
    uint32_t *mappedBeforeFailure,
    const char **errorMessage);
uint32_t ANEFieldfareANESharedSurfaceRequestSetMappedCount(
    ANEFieldfareANESharedSurfaceRequestSet *set);
CFTypeRef ANEFieldfareANESharedSurfaceRequestSetCopyRequestDescription(
    ANEFieldfareANESharedSurfaceRequestSet *set, uint32_t requestIndex);
bool ANEFieldfareANESharedSurfaceRequestSetEvaluate(
    ANEFieldfareANESharedSurfaceRequestSet *set, uint32_t requestIndex,
    uint64_t timeoutNanoseconds, ANEFieldfareANEEvaluationTiming *timing,
    const char **errorMessage);
bool ANEFieldfareANESharedSurfaceRequestSetEvaluateBegin(
    ANEFieldfareANESharedSurfaceRequestSet *set, uint32_t requestIndex,
    ANEFieldfareANEEvaluationTiming *timing, const char **errorMessage);
bool ANEFieldfareANESharedSurfaceRequestSetEvaluateWait(
    ANEFieldfareANESharedSurfaceRequestSet *set, uint64_t timeoutNanoseconds,
    ANEFieldfareANEEvaluationTiming *timing, const char **errorMessage);
bool ANEFieldfareANESharedSurfaceRequestSetUnmapAll(
    ANEFieldfareANESharedSurfaceRequestSet *set, uint32_t *unmappedCount,
    const char **errorMessage);
void ANEFieldfareANESharedSurfaceRequestSetDestroy(
    ANEFieldfareANESharedSurfaceRequestSet *set);

ANEFieldfareANEProgram *ANEFieldfareANEProgramCreate(const char *milUTF8,
                                                      uint64_t inputBytes,
                                                      uint64_t outputBytes,
                                                      uint32_t requestBanks,
                                                      const char **errorMessage);

// Class-1 transport: weights are delivered as a compile-time BLOBFILE constant
// resolved at "@model_path/weights/<blobName>" and baked into the loaded
// program. No weight bytes cross the input boundary at evaluation time.
//
// blobBytes must be a complete ANE weight blob as produced by
// ANEFieldfareANEWeightBlobCreate. The MIL must reference it with
// offset = uint64(64).
ANEFieldfareANEProgram *ANEFieldfareANEProgramCreateWithConstantWeights(
    const char *milUTF8,
    const char *blobName,
    const void *blobBytes,
    uint64_t blobByteCount,
    uint64_t inputBytes,
    uint64_t outputBytes,
    uint32_t requestBanks,
    const char **errorMessage);

ANEFieldfareANEProgram *ANEFieldfareANEProgramCreateWithTwoProceduresAndConstantWeights(
    const char *milUTF8,
    const char *blobName0,
    const void *blobBytes0,
    uint64_t blobByteCount0,
    const char *blobName1,
    const void *blobBytes1,
    uint64_t blobByteCount1,
    const char *procedureName0,
    const char *procedureName1,
    const char **errorMessage);

// Creates one compiled/loaded handle with one request bank per compiler-named
// procedure. Procedure IDs and symbol indices are resolved only from the
// compiler-emitted model attributes; callers never synthesize them.
ANEFieldfareANEProgram *ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
    const char *milUTF8,
    const char *blobName,
    const void *blobBytes,
    uint64_t blobByteCount,
    const char *const *procedureNames,
    uint32_t procedureCount,
    const char **errorMessage);

// Allocates a single-chunk ANE weight blob: a 64-byte file header, a 64-byte
// chunk header, then elementCount FP16 values left uninitialized. The chunk
// header sits at offset 64, which is the offset the MIL must declare.
//
// Returns the blob allocation, which the caller frees. On success *payload
// points at the FP16 region for the caller to fill in place, so a large
// constant never requires a second full-size staging buffer.
void *ANEFieldfareANEWeightBlobCreate(uint64_t elementCount,
                                      uint64_t *blobByteCount,
                                      void **payload);

// As above, but sizes the payload in bytes rather than FP16 elements, so a
// compressed constant (uint8 codes, packed uint4 codes) can be delivered
// through the same BLOBFILE route.
void *ANEFieldfareANEWeightBlobCreateWithBytes(uint64_t payloadBytes,
                                               uint64_t *blobByteCount,
                                               void **payload);
ANEFieldfareANEProgramInfo ANEFieldfareANEProgramGetInfo(ANEFieldfareANEProgram *program);
IOSurfaceRef ANEFieldfareANEProgramInput(ANEFieldfareANEProgram *program, uint32_t bank);
IOSurfaceRef ANEFieldfareANEProgramOutput(ANEFieldfareANEProgram *program, uint32_t bank);
bool ANEFieldfareANEProgramEvaluate(ANEFieldfareANEProgram *program,
                                    uint32_t bank,
                                    uint64_t timeoutNanoseconds,
                                    ANEFieldfareANEEvaluationTiming *timing,
                                    const char **errorMessage);
void ANEFieldfareANEProgramDestroy(ANEFieldfareANEProgram *program);
