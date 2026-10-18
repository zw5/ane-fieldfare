#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ANEFieldfareRuntime/AOTExpertRuntime.h"

typedef struct {
    uint32_t layer;
    uint32_t expert;
} FakeProcedure;

typedef struct {
    bool inFlight[ANEFieldfareAOTExpertRequiredRequestBanks];
    uint64_t generation[ANEFieldfareAOTExpertRequiredRequestBanks];
    uint64_t loads;
    uint64_t submits;
    uint64_t waits;
} FakeContext;

static uint64_t unloads, destroyedContexts;

static bool loadProcedure(
    void *rawContext, const char *root, const ANEFieldfareAOTExpertShape *shape,
    uint32_t layer, uint32_t expert, void **result,
    char name[ANEFieldfareAOTExpertProcedureNameCapacity], const char **errorMessage) {
    FakeContext *context = rawContext;
    if (!context || strcmp(root, "fixture://compiler-authored") ||
        shape->requestBanks != 16 || layer >= shape->layerCount ||
        expert >= shape->expertCount) {
        if (errorMessage) *errorMessage = "fake named-procedure lookup rejected";
        return false;
    }
    FakeProcedure *procedure = malloc(sizeof(*procedure));
    if (!procedure) return false;
    *procedure = (FakeProcedure){layer, expert};
    snprintf(name, ANEFieldfareAOTExpertProcedureNameCapacity,
             "expert_l%02u_e%03u", layer, expert);
    *result = procedure;
    ++context->loads;
    return true;
}

static void unloadProcedure(void *context, void *procedure) {
    (void)context;
    ++unloads;
    free(procedure);
}

static bool submit(void *rawContext, void *rawProcedure, uint32_t bank,
                   uint64_t generation, const float *input, float *output,
                   const char **errorMessage) {
    FakeContext *context = rawContext;
    FakeProcedure *procedure = rawProcedure;
    if (!context || !procedure || bank >= 16 || context->inFlight[bank] ||
        generation <= context->generation[bank]) {
        if (errorMessage) *errorMessage = "fake bank overwrite/generation rejection";
        return false;
    }
    context->inFlight[bank] = true;
    context->generation[bank] = generation;
    ++context->submits;
    float add = (float)(procedure->layer * 1000 + procedure->expert);
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
        output[index] = input[index] + add;
    return true;
}

static bool waitForGeneration(void *rawContext, uint32_t bank, uint64_t generation,
                              uint64_t timeout, const char **errorMessage) {
    FakeContext *context = rawContext;
    if (!context || bank >= 16 || !timeout || !context->inFlight[bank] ||
        context->generation[bank] != generation) {
        if (errorMessage) *errorMessage = "fake completion generation mismatch";
        return false;
    }
    context->inFlight[bank] = false;
    ++context->waits;
    return true;
}

static void destroyContext(void *rawContext) {
    ++destroyedContexts;
    free(rawContext);
}

int main(void) {
    FakeContext *context = calloc(1, sizeof(*context));
    ANEFieldfareAOTExpertShape shape = {
        .hiddenSize = ANEFieldfareGemma4HiddenSize,
        .intermediateSize = ANEFieldfareGemma4ExpertIntermediateSize,
        .expertCount = ANEFieldfareGemma4ExpertCount,
        .topK = ANEFieldfareGemma4TopK,
        .layerCount = 30,
        .requestBanks = ANEFieldfareAOTExpertRequiredRequestBanks,
        .capabilityFlags = ANEFieldfareAOTCapabilityCompilerNamedProcedures |
            ANEFieldfareAOTCapabilityCompletionGenerations,
        .procedureCacheCapacity = 16,
    };
    ANEFieldfareAOTExpertProvider provider = {
        .abiVersion = ANEFieldfareAOTExpertProviderABIVersion,
        .context = context,
        .loadNamedProcedure = loadProcedure,
        .unloadProcedure = unloadProcedure,
        .submit = submit,
        .wait = waitForGeneration,
        .destroyContext = destroyContext,
    };
    const char *error = NULL;
    ANEFieldfareAOTExpertRuntime *runtime = ANEFieldfareAOTExpertRuntimeCreate(
        "fixture://compiler-authored", shape, provider, &error);
    float *input = calloc(ANEFieldfareGemma4HiddenSize, sizeof(*input));
    float (*outputs)[ANEFieldfareGemma4HiddenSize] = calloc(
        ANEFieldfareGemma4TopK, sizeof(*outputs));
    uint32_t experts[ANEFieldfareGemma4TopK] = {3, 7, 11, 19, 23, 31, 47, 61};
    for (uint32_t index = 0; input && index < ANEFieldfareGemma4HiddenSize; ++index)
        input[index] = (float)index / 256.0f;
    bool valid = runtime && input && outputs;
    uint32_t observedBases[3] = {0};
    uint64_t observedGenerations[3] = {0};
    for (uint32_t run = 0; valid && run < 3; ++run) {
        valid = ANEFieldfareAOTExpertRuntimeCallTop8(
            runtime, 4, experts, input, outputs, 1000000000ull);
        ANEFieldfareAOTExpertRuntimeInfo info =
            ANEFieldfareAOTExpertRuntimeGetInfo(runtime);
        observedBases[run] = info.lastBankBase;
        observedGenerations[run] = info.lastCompletedGeneration;
        for (uint32_t rank = 0; valid && rank < ANEFieldfareGemma4TopK; ++rank)
            for (uint32_t index = 0; valid && index < ANEFieldfareGemma4HiddenSize; ++index)
                valid = outputs[rank][index] ==
                    input[index] + (4000.0f + (float)experts[rank]);
    }
    ANEFieldfareAOTExpertRuntimeInfo info =
        ANEFieldfareAOTExpertRuntimeGetInfo(runtime);
    valid = valid && observedBases[0] == 0 && observedBases[1] == 8 &&
        observedBases[2] == 0 && observedGenerations[0] == 1 &&
        observedGenerations[1] == 1 && observedGenerations[2] == 2 &&
        info.procedureLoadCount == 8 && info.procedureCacheHitCount == 16 &&
        info.batchCount == 3 && info.submissionCount == 24 &&
        info.completionCount == 24 && !info.failedClosed &&
        context->submits == 24 && context->waits == 24;

    uint32_t duplicate[ANEFieldfareGemma4TopK] = {3, 3, 11, 19, 23, 31, 47, 61};
    uint64_t submissionsBeforeNegative = context ? context->submits : 0;
    valid = valid && !ANEFieldfareAOTExpertRuntimeCallTop8(
        runtime, 4, duplicate, input, outputs, 1000000000ull) &&
        context->submits == submissionsBeforeNegative;

    ANEFieldfareAOTExpertShape invalidShape = shape;
    invalidShape.requestBanks = 8;
    FakeContext unusedContext = {0};
    ANEFieldfareAOTExpertProvider unusedProvider = provider;
    unusedProvider.context = &unusedContext;
    valid = valid && !ANEFieldfareAOTExpertRuntimeCreate(
        "fixture://compiler-authored", invalidShape, unusedProvider, &error) &&
        !ANEFieldfareAOTExpertRuntimeCreateFromArtifactRoot(
            "fixture://missing-provider", shape, &error);

    ANEFieldfareAOTExpertRuntimeDestroy(runtime);
    valid = valid && unloads == 8 && destroyedContexts == 1;
    printf("{\"schema\":\"ane-fieldfare.aot-expert-runtime.v1\","
           "\"status\":\"%s\",\"request_banks\":16,\"top_k\":8,"
           "\"bank_bases\":[%u,%u,%u],\"completion_generations\":[%llu,%llu,%llu],"
           "\"procedure_loads\":%llu,\"procedure_cache_hits\":%llu,"
           "\"submissions\":%llu,\"completions\":%llu,"
           "\"duplicate_top8_rejected_before_submit\":true,"
           "\"eight_bank_shape_rejected\":true,\"missing_provider_rejected\":true,"
           "\"unloads\":%llu,\"context_destroys\":%llu,"
           "\"claim_boundary\":\"lifecycle and dispatch seam validated with a fake provider; no AOT artifact or ANE execution is claimed\"}\n",
           valid ? "validated" : "rejected", observedBases[0], observedBases[1],
           observedBases[2], (unsigned long long)observedGenerations[0],
           (unsigned long long)observedGenerations[1],
           (unsigned long long)observedGenerations[2],
           (unsigned long long)info.procedureLoadCount,
           (unsigned long long)info.procedureCacheHitCount,
           (unsigned long long)info.submissionCount,
           (unsigned long long)info.completionCount,
           (unsigned long long)unloads, (unsigned long long)destroyedContexts);
    free(input); free(outputs);
    return valid ? 0 : 1;
}
