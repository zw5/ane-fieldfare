#include "AOTExpertRuntime.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MaximumProcedureCacheCapacity = 64 };

typedef struct {
    bool valid;
    uint32_t layer;
    uint32_t expert;
    void *procedure;
    char name[ANEFieldfareAOTExpertProcedureNameCapacity];
    uint64_t lastUse;
} ProcedureEntry;

typedef struct {
    bool inFlight;
    uint64_t generation;
} BankState;

struct ANEFieldfareAOTExpertRuntime {
    char *artifactRoot;
    ANEFieldfareAOTExpertProvider provider;
    ANEFieldfareAOTExpertRuntimeInfo info;
    ProcedureEntry cache[MaximumProcedureCacheCapacity];
    BankState banks[ANEFieldfareAOTExpertRequiredRequestBanks];
    uint64_t useOrdinal;
    uint32_t nextBankBase;
    char error[512];
    pthread_mutex_t lock;
};

__attribute__((weak)) bool ANEFieldfareAOTExpertProviderCreate(
    const char *artifactRoot, const ANEFieldfareAOTExpertShape *shape,
    ANEFieldfareAOTExpertProvider *provider, const char **errorMessage) {
    (void)artifactRoot; (void)shape; (void)provider;
    if (errorMessage) *errorMessage = "compiler-authored AOT expert provider is not linked";
    return false;
}

static void setError(ANEFieldfareAOTExpertRuntime *runtime, const char *message) {
    if (!runtime) return;
    snprintf(runtime->error, sizeof(runtime->error), "%s",
             message ? message : "unknown AOT expert error");
}

static bool validShape(ANEFieldfareAOTExpertShape shape) {
    uint32_t required = ANEFieldfareAOTCapabilityCompilerNamedProcedures |
        ANEFieldfareAOTCapabilityCompletionGenerations;
    return shape.hiddenSize == ANEFieldfareGemma4HiddenSize &&
        shape.intermediateSize == ANEFieldfareGemma4ExpertIntermediateSize &&
        shape.expertCount == ANEFieldfareGemma4ExpertCount &&
        shape.topK == ANEFieldfareGemma4TopK && shape.layerCount > 0 &&
        shape.requestBanks == ANEFieldfareAOTExpertRequiredRequestBanks &&
        (shape.capabilityFlags & required) == required &&
        shape.procedureCacheCapacity >= shape.topK &&
        shape.procedureCacheCapacity <= MaximumProcedureCacheCapacity;
}

static bool validProvider(ANEFieldfareAOTExpertProvider provider) {
    return provider.abiVersion == ANEFieldfareAOTExpertProviderABIVersion &&
        provider.loadNamedProcedure && provider.unloadProcedure &&
        provider.submit && provider.wait;
}

ANEFieldfareAOTExpertRuntime *ANEFieldfareAOTExpertRuntimeCreate(
    const char *artifactRoot, ANEFieldfareAOTExpertShape shape,
    ANEFieldfareAOTExpertProvider provider, const char **errorMessage) {
    if (errorMessage) *errorMessage = NULL;
    if (!artifactRoot || !artifactRoot[0] || !validShape(shape) || !validProvider(provider)) {
        if (errorMessage) *errorMessage = "invalid AOT expert runtime construction";
        return NULL;
    }
    ANEFieldfareAOTExpertRuntime *runtime = calloc(1, sizeof(*runtime));
    if (!runtime || !(runtime->artifactRoot = strdup(artifactRoot)) ||
        pthread_mutex_init(&runtime->lock, NULL) != 0) {
        if (runtime) { free(runtime->artifactRoot); free(runtime); }
        if (errorMessage) *errorMessage = "AOT expert runtime allocation failed";
        return NULL;
    }
    runtime->provider = provider;
    runtime->info.shape = shape;
    runtime->nextBankBase = 0;
    return runtime;
}

ANEFieldfareAOTExpertRuntime *ANEFieldfareAOTExpertRuntimeCreateFromArtifactRoot(
    const char *artifactRoot, ANEFieldfareAOTExpertShape shape,
    const char **errorMessage) {
    ANEFieldfareAOTExpertProvider provider = {0};
    if (!ANEFieldfareAOTExpertProviderCreate(
            artifactRoot, &shape, &provider, errorMessage)) return NULL;
    ANEFieldfareAOTExpertRuntime *runtime = ANEFieldfareAOTExpertRuntimeCreate(
        artifactRoot, shape, provider, errorMessage);
    if (!runtime && provider.destroyContext) provider.destroyContext(provider.context);
    return runtime;
}

static ProcedureEntry *procedureFor(ANEFieldfareAOTExpertRuntime *runtime,
                                    uint32_t layer, uint32_t expert) {
    uint32_t capacity = runtime->info.shape.procedureCacheCapacity;
    ProcedureEntry *victim = NULL;
    for (uint32_t index = 0; index < capacity; ++index) {
        ProcedureEntry *entry = &runtime->cache[index];
        if (entry->valid && entry->layer == layer && entry->expert == expert) {
            entry->lastUse = ++runtime->useOrdinal;
            ++runtime->info.procedureCacheHitCount;
            return entry;
        }
        if (!victim || !entry->valid ||
            (victim->valid && entry->lastUse < victim->lastUse)) victim = entry;
    }
    if (!victim) { setError(runtime, "AOT expert procedure cache unavailable"); return NULL; }
    void *procedure = NULL;
    char name[ANEFieldfareAOTExpertProcedureNameCapacity] = {0};
    const char *error = NULL;
    if (!runtime->provider.loadNamedProcedure(
            runtime->provider.context, runtime->artifactRoot, &runtime->info.shape,
            layer, expert, &procedure, name, &error) || !procedure || !name[0]) {
        setError(runtime, error ? error : "compiler-authored expert procedure lookup failed");
        return NULL;
    }
    if (victim->valid) {
        runtime->provider.unloadProcedure(runtime->provider.context, victim->procedure);
        ++runtime->info.procedureUnloadCount;
    }
    *victim = (ProcedureEntry){
        .valid = true, .layer = layer, .expert = expert,
        .procedure = procedure, .lastUse = ++runtime->useOrdinal,
    };
    snprintf(victim->name, sizeof(victim->name), "%s", name);
    ++runtime->info.procedureLoadCount;
    return victim;
}

bool ANEFieldfareAOTExpertRuntimeCallTop8(
    ANEFieldfareAOTExpertRuntime *runtime, uint32_t layer,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize],
    uint64_t timeoutNanoseconds) {
    if (!runtime || !experts || !input || !outputs || !timeoutNanoseconds) return false;
    pthread_mutex_lock(&runtime->lock);
    if (runtime->info.failedClosed || layer >= runtime->info.shape.layerCount) {
        setError(runtime, runtime->info.failedClosed
            ? "AOT expert runtime is failed closed" : "AOT expert layer is out of range");
        pthread_mutex_unlock(&runtime->lock);
        return false;
    }
    for (uint32_t left = 0; left < ANEFieldfareGemma4TopK; ++left) {
        if (experts[left] >= runtime->info.shape.expertCount) {
            setError(runtime, "AOT expert identity is out of range");
            pthread_mutex_unlock(&runtime->lock);
            return false;
        }
        for (uint32_t right = 0; right < left; ++right) if (experts[left] == experts[right]) {
            setError(runtime, "AOT top-8 expert identities must be unique");
            pthread_mutex_unlock(&runtime->lock);
            return false;
        }
    }
    uint32_t base = runtime->nextBankBase;
    ProcedureEntry *procedures[ANEFieldfareGemma4TopK] = {0};
    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
        if (runtime->banks[base + rank].inFlight ||
            !(procedures[rank] = procedureFor(runtime, layer, experts[rank]))) {
            setError(runtime, runtime->banks[base + rank].inFlight
                ? "AOT request bank overwrite rejected" : runtime->error);
            pthread_mutex_unlock(&runtime->lock);
            return false;
        }
    }
    uint32_t submitted = 0;
    for (; submitted < ANEFieldfareGemma4TopK; ++submitted) {
        uint32_t bank = base + submitted;
        uint64_t generation = ++runtime->banks[bank].generation;
        if (!generation) generation = ++runtime->banks[bank].generation;
        runtime->banks[bank].inFlight = true;
        const char *error = NULL;
        if (!runtime->provider.submit(
                runtime->provider.context, procedures[submitted]->procedure,
                bank, generation, input, outputs[submitted], &error)) {
            runtime->banks[bank].inFlight = false;
            setError(runtime, error ? error : "AOT expert submission failed");
            runtime->info.failedClosed = true;
            break;
        }
        ++runtime->info.submissionCount;
    }
    bool valid = submitted == ANEFieldfareGemma4TopK;
    for (uint32_t rank = 0; rank < submitted; ++rank) {
        uint32_t bank = base + rank;
        uint64_t generation = runtime->banks[bank].generation;
        const char *error = NULL;
        bool completed = runtime->provider.wait(
            runtime->provider.context, bank, generation, timeoutNanoseconds, &error);
        if (!completed) {
            setError(runtime, error ? error : "AOT expert completion failed");
            runtime->info.failedClosed = true;
            valid = false;
        } else {
            runtime->banks[bank].inFlight = false;
            runtime->info.lastCompletedGeneration = generation;
            ++runtime->info.completionCount;
        }
    }
    if (valid) {
        ++runtime->info.batchCount;
        runtime->info.lastBankBase = base;
        runtime->nextBankBase = base ? 0 : ANEFieldfareGemma4TopK;
    }
    pthread_mutex_unlock(&runtime->lock);
    return valid;
}

bool ANEFieldfareAOTExpertRuntimeCallTop8Flat(
    ANEFieldfareAOTExpertRuntime *runtime, uint32_t layer,
    const uint32_t *experts, const float *input, float *outputs,
    uint64_t timeoutNanoseconds) {
    return ANEFieldfareAOTExpertRuntimeCallTop8(
        runtime, layer, experts, input,
        (float (*)[ANEFieldfareGemma4HiddenSize])outputs, timeoutNanoseconds);
}

ANEFieldfareAOTExpertRuntimeInfo ANEFieldfareAOTExpertRuntimeGetInfo(
    const ANEFieldfareAOTExpertRuntime *runtime) {
    return runtime ? runtime->info : (ANEFieldfareAOTExpertRuntimeInfo){0};
}

const char *ANEFieldfareAOTExpertRuntimeLastError(
    const ANEFieldfareAOTExpertRuntime *runtime) {
    return runtime && runtime->error[0] ? runtime->error : NULL;
}

void ANEFieldfareAOTExpertRuntimeDestroy(ANEFieldfareAOTExpertRuntime *runtime) {
    if (!runtime) return;
    pthread_mutex_lock(&runtime->lock);
    for (uint32_t index = 0; index < runtime->info.shape.procedureCacheCapacity; ++index) {
        if (!runtime->cache[index].valid) continue;
        runtime->provider.unloadProcedure(runtime->provider.context,
                                          runtime->cache[index].procedure);
        ++runtime->info.procedureUnloadCount;
    }
    if (runtime->provider.destroyContext)
        runtime->provider.destroyContext(runtime->provider.context);
    pthread_mutex_unlock(&runtime->lock);
    pthread_mutex_destroy(&runtime->lock);
    free(runtime->artifactRoot);
    free(runtime);
}
