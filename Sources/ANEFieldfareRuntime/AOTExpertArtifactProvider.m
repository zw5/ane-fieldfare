#import "AOTExpertRuntime.h"
#import "ANEProgram.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

enum { ArtifactPositions = 64, MaximumLayers = 64 };

typedef struct ProviderContext ProviderContext;
typedef struct {
    ProviderContext *owner;
    uint32_t layer;
    uint32_t expert;
    uint32_t physicalBank;
} ProcedureToken;

typedef struct {
    ANEFieldfareANEProgram *program;
    uint32_t references;
    bool available[ANEFieldfareGemma4ExpertCount];
    uint32_t physicalBank[ANEFieldfareGemma4ExpertCount];
} LayerProgram;

typedef struct {
    bool inFlight;
    uint64_t generation;
    uint32_t layer;
    uint32_t physicalBank;
    float *output;
} LogicalRequest;

struct ProviderContext {
    char *root;
    ANEFieldfareAOTExpertShape shape;
    LayerProgram layers[MaximumLayers];
    LogicalRequest requests[ANEFieldfareAOTExpertRequiredRequestBanks];
    char error[512];
};

static void providerError(ProviderContext *context, const char **result,
                          NSString *message) {
    snprintf(context->error, sizeof(context->error), "%s", message.UTF8String);
    if (result) *result = context->error;
}

static NSString *layerArtifactPath(ProviderContext *context, uint32_t layer) {
    NSString *root = [NSString stringWithUTF8String:context->root];
    return [root stringByAppendingPathComponent:
        [NSString stringWithFormat:@"layer-%02u.mlmodelc", layer]];
}

static bool loadLayer(ProviderContext *context, uint32_t layer,
                      const char **errorMessage) {
    @autoreleasepool {
        if (layer >= context->shape.layerCount || layer >= MaximumLayers) {
            providerError(context, errorMessage, @"AOT layer identity is out of range");
            return false;
        }
        LayerProgram *record = &context->layers[layer];
        if (record->program) return true;
        NSString *path = layerArtifactPath(context, layer);
        NSData *milData = [NSData dataWithContentsOfFile:
            [path stringByAppendingPathComponent:@"model.mil"]];
        NSData *weights = [NSData dataWithContentsOfFile:
            [path stringByAppendingPathComponent:@"weights/weight.bin"]];
        NSString *mil = milData ? [[NSString alloc] initWithData:milData
            encoding:NSUTF8StringEncoding] : nil;
        if (!mil || !weights) {
            providerError(context, errorMessage,
                [NSString stringWithFormat:@"missing compiler artifact %@", path]);
            return false;
        }
        NSMutableArray<NSString *> *names = [NSMutableArray array];
        for (uint32_t expert = 0; expert < context->shape.expertCount; ++expert) {
            NSString *name = [NSString stringWithFormat:@"expert_%u", expert];
            if ([mil containsString:[NSString stringWithFormat:@"func %@<", name]]) {
                record->available[expert] = true;
                record->physicalBank[expert] = (uint32_t)names.count;
                [names addObject:name];
            }
        }
        // The current private request mapping has been validated for at most
        // eight named procedures behind one handle. Larger compiler families
        // must extend ANEProgram with their authoritative mapping first.
        if (!names.count || names.count > ANEFieldfareGemma4TopK) {
            providerError(context, errorMessage,
                @"artifact named-procedure count is outside validated 1...8 bound");
            memset(record->available, 0, sizeof(record->available));
            return false;
        }
        const char *procedureNames[ANEFieldfareGemma4TopK] = {0};
        for (NSUInteger index = 0; index < names.count; ++index)
            procedureNames[index] = names[index].UTF8String;
        const char *programError = NULL;
        record->program = ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
            mil.UTF8String, "weight.bin", weights.bytes, weights.length,
            procedureNames, (uint32_t)names.count, &programError);
        if (!record->program) {
            providerError(context, errorMessage,
                [NSString stringWithFormat:@"ANE load rejected: %s",
                    programError ? programError : "unknown"]);
            free((void *)programError);
            memset(record->available, 0, sizeof(record->available));
            return false;
        }
        free((void *)programError);
        return true;
    }
}

static bool loadNamedProcedure(
    void *raw, const char *artifactRoot, const ANEFieldfareAOTExpertShape *shape,
    uint32_t layer, uint32_t expert, void **result,
    char procedureName[ANEFieldfareAOTExpertProcedureNameCapacity],
    const char **errorMessage) {
    ProviderContext *context = raw;
    if (!context || !artifactRoot || strcmp(artifactRoot, context->root) ||
        memcmp(shape, &context->shape, sizeof(*shape)) ||
        expert >= context->shape.expertCount || !loadLayer(context, layer, errorMessage))
        return false;
    LayerProgram *record = &context->layers[layer];
    if (!record->available[expert]) {
        providerError(context, errorMessage,
            [NSString stringWithFormat:@"expert_%u is absent from layer-%02u artifact",
             expert, layer]);
        return false;
    }
    ProcedureToken *token = calloc(1, sizeof(*token));
    if (!token) return false;
    *token = (ProcedureToken){context, layer, expert, record->physicalBank[expert]};
    ++record->references;
    snprintf(procedureName, ANEFieldfareAOTExpertProcedureNameCapacity,
             "expert_%u", expert);
    *result = token;
    return true;
}

static void unloadProcedure(void *raw, void *procedure) {
    ProviderContext *context = raw;
    ProcedureToken *token = procedure;
    if (context && token && token->owner == context && token->layer < MaximumLayers &&
        context->layers[token->layer].references)
        --context->layers[token->layer].references;
    free(token);
}

static bool submit(void *raw, void *procedure, uint32_t bank, uint64_t generation,
                   const float *input, float *output, const char **errorMessage) {
    ProviderContext *context = raw;
    ProcedureToken *token = procedure;
    if (!context || !token || token->owner != context || bank >= 16 ||
        context->requests[bank].inFlight) return false;
    LayerProgram *layer = &context->layers[token->layer];
    IOSurfaceRef surface = ANEFieldfareANEProgramInput(layer->program, token->physicalBank);
    if (!surface || IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) {
        providerError(context, errorMessage, @"ANE input surface lock failed");
        return false;
    }
    __fp16 *base = IOSurfaceGetBaseAddress(surface);
    for (uint32_t column = 0; column < context->shape.hiddenSize; ++column)
        for (uint32_t position = 0; position < ArtifactPositions; ++position)
            base[(uint64_t)column * ArtifactPositions + position] = (__fp16)input[column];
    IOSurfaceUnlock(surface, 0, NULL);
    ANEFieldfareANEEvaluationTiming timing = {0}; const char *error = NULL;
    if (!ANEFieldfareANEProgramEvaluateBegin(layer->program, token->physicalBank,
                                              &timing, &error)) {
        providerError(context, errorMessage,
            [NSString stringWithFormat:@"ANE begin rejected: %s", error ? error : "unknown"]);
        free((void *)error);
        return false;
    }
    free((void *)error);
    context->requests[bank] = (LogicalRequest){true, generation, token->layer,
                                               token->physicalBank, output};
    return true;
}

static bool waitForGeneration(void *raw, uint32_t bank, uint64_t generation,
                              uint64_t timeout, const char **errorMessage) {
    ProviderContext *context = raw;
    if (!context || bank >= 16 || !context->requests[bank].inFlight ||
        context->requests[bank].generation != generation) return false;
    LogicalRequest request = context->requests[bank];
    LayerProgram *layer = &context->layers[request.layer];
    ANEFieldfareANEEvaluationTiming timing = {0}; const char *error = NULL;
    if (!ANEFieldfareANEProgramEvaluateWait(layer->program, request.physicalBank,
                                            timeout, &timing, &error)) {
        providerError(context, errorMessage,
            [NSString stringWithFormat:@"ANE wait rejected: %s", error ? error : "unknown"]);
        free((void *)error);
        return false;
    }
    free((void *)error);
    IOSurfaceRef surface = ANEFieldfareANEProgramOutput(layer->program,
                                                        request.physicalBank);
    if (!surface || IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
        providerError(context, errorMessage, @"ANE output surface lock failed");
        return false;
    }
    const float *base = IOSurfaceGetBaseAddress(surface);
    for (uint32_t row = 0; row < context->shape.hiddenSize; ++row)
        request.output[row] = base[(uint64_t)row * ArtifactPositions];
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    context->requests[bank].inFlight = false;
    return true;
}

static void destroyContext(void *raw) {
    ProviderContext *context = raw;
    if (!context) return;
    for (uint32_t layer = 0; layer < context->shape.layerCount && layer < MaximumLayers; ++layer)
        ANEFieldfareANEProgramDestroy(context->layers[layer].program);
    free(context->root); free(context);
}

bool ANEFieldfareAOTExpertProviderCreate(
    const char *artifactRoot, const ANEFieldfareAOTExpertShape *shape,
    ANEFieldfareAOTExpertProvider *provider, const char **errorMessage) {
    if (!artifactRoot || !artifactRoot[0] || !shape || !provider) return false;
    ProviderContext *context = calloc(1, sizeof(*context));
    if (!context || !(context->root = strdup(artifactRoot))) {
        free(context); return false;
    }
    context->shape = *shape;
    *provider = (ANEFieldfareAOTExpertProvider){
        .abiVersion = ANEFieldfareAOTExpertProviderABIVersion,
        .context = context,
        .loadNamedProcedure = loadNamedProcedure,
        .unloadProcedure = unloadProcedure,
        .submit = submit,
        .wait = waitForGeneration,
        .destroyContext = destroyContext,
    };
    if (errorMessage) *errorMessage = NULL;
    return true;
}
