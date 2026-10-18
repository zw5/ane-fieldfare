#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "RoutedLayer.h"

enum {
    ANEFieldfareAOTExpertProviderABIVersion = 1,
    ANEFieldfareAOTExpertRequiredRequestBanks = 16,
    ANEFieldfareAOTExpertProcedureNameCapacity = 192,
};

enum {
    ANEFieldfareAOTCapabilityCompilerNamedProcedures = 1u << 0,
    ANEFieldfareAOTCapabilityCompletionGenerations = 1u << 1,
};

typedef struct {
    uint32_t hiddenSize;
    uint32_t intermediateSize;
    uint32_t expertCount;
    uint32_t topK;
    uint32_t layerCount;
    uint32_t requestBanks;
    uint32_t capabilityFlags;
    uint32_t procedureCacheCapacity;
} ANEFieldfareAOTExpertShape;

// Artifact implementations own compiler/runtime details. The product seam
// supplies only shape plus layer/expert identity; the provider must resolve a
// compiler-authored procedure name and must reject missing or mismatched
// artifacts. `wait` must complete exactly the submitted bank generation.
typedef struct {
    uint32_t abiVersion;
    void *context;
    bool (*loadNamedProcedure)(
        void *context, const char *artifactRoot,
        const ANEFieldfareAOTExpertShape *shape, uint32_t layer, uint32_t expert,
        void **procedure, char procedureName[ANEFieldfareAOTExpertProcedureNameCapacity],
        const char **errorMessage);
    void (*unloadProcedure)(void *context, void *procedure);
    bool (*submit)(void *context, void *procedure, uint32_t bank, uint64_t generation,
                   const float *input, float *output, const char **errorMessage);
    bool (*wait)(void *context, uint32_t bank, uint64_t generation,
                 uint64_t timeoutNanoseconds, const char **errorMessage);
    void (*destroyContext)(void *context);
} ANEFieldfareAOTExpertProvider;

typedef struct ANEFieldfareAOTExpertRuntime ANEFieldfareAOTExpertRuntime;

typedef struct {
    ANEFieldfareAOTExpertShape shape;
    uint64_t procedureLoadCount;
    uint64_t procedureCacheHitCount;
    uint64_t procedureUnloadCount;
    uint64_t batchCount;
    uint64_t submissionCount;
    uint64_t completionCount;
    uint32_t lastBankBase;
    uint64_t lastCompletedGeneration;
    bool failedClosed;
} ANEFieldfareAOTExpertRuntimeInfo;

ANEFieldfareAOTExpertRuntime *ANEFieldfareAOTExpertRuntimeCreate(
    const char *artifactRoot, ANEFieldfareAOTExpertShape shape,
    ANEFieldfareAOTExpertProvider provider, const char **errorMessage);

// Product integration entry point. The artifact provider resolves only
// compiler-authored procedures from shape-keyed layer artifacts. Missing or
// incompatible artifacts make opt-in construction/lookup fail closed.
ANEFieldfareAOTExpertRuntime *ANEFieldfareAOTExpertRuntimeCreateFromArtifactRoot(
    const char *artifactRoot, ANEFieldfareAOTExpertShape shape,
    const char **errorMessage);

bool ANEFieldfareAOTExpertRuntimeCallTop8(
    ANEFieldfareAOTExpertRuntime *runtime, uint32_t layer,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize],
    uint64_t timeoutNanoseconds);

// Swift-friendly equivalent with flat row-major output storage [8, hidden].
bool ANEFieldfareAOTExpertRuntimeCallTop8Flat(
    ANEFieldfareAOTExpertRuntime *runtime, uint32_t layer,
    const uint32_t *experts, const float *input, float *outputs,
    uint64_t timeoutNanoseconds);

ANEFieldfareAOTExpertRuntimeInfo ANEFieldfareAOTExpertRuntimeGetInfo(
    const ANEFieldfareAOTExpertRuntime *runtime);
const char *ANEFieldfareAOTExpertRuntimeLastError(
    const ANEFieldfareAOTExpertRuntime *runtime);
void ANEFieldfareAOTExpertRuntimeDestroy(ANEFieldfareAOTExpertRuntime *runtime);
