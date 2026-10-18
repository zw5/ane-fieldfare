#include "Decoder.h"

#include <mach/mach_time.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LayerCount = 30, Hidden = 2816 };

struct ANEFieldfareDecoder {
    char *modelDirectory;
    ANEFieldfareTensorIndex index;
    ANEFieldfareAttentionWeights attention[LayerCount];
    ANEFieldfareKVCache caches[LayerCount];
    ANEFieldfareRoutedLayer *routed[LayerCount];
    float inputNorm[LayerCount][Hidden];
    float postAttentionNorm[LayerCount][Hidden];
    ANEFieldfareDecoderTiming timing;
    FILE *routeTrace;
};

// The surviving analysis tools consume: position layer expert0 ... expert7.
// Own the optional trace per decoder, not in process-global state.
static bool routeTraceRecord(FILE *trace, uint64_t position, uint32_t layer,
                              const ANEFieldfareRoute *routes) {
    if (!trace) return true;
    if (fprintf(trace, "%llu %u", (unsigned long long)position, layer) < 0) return false;
    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
        if (fprintf(trace, " %u", routes[rank].expert) < 0) return false;
    return fputc('\n', trace) != EOF && fflush(trace) == 0;
}

static mach_timebase_info_data_t decoderTimebase;

static double elapsedMilliseconds(uint64_t started) {
    if (!decoderTimebase.denom) mach_timebase_info(&decoderTimebase);
    uint64_t elapsed = mach_absolute_time() - started;
    return (double)elapsed * decoderTimebase.numer / decoderTimebase.denom / 1e6;
}

static bool findNorm(const ANEFieldfareTensorIndex *index,
                     uint32_t layer,
                     const char *suffix,
                     float output[Hidden]) {
    char name[192];
    int length = snprintf(name, sizeof(name), "language_model.model.layers.%u.%s", layer, suffix);
    ANEFieldfareTensor tensor = {0};
    if (length <= 0 || (size_t)length >= sizeof(name) ||
        !ANEFieldfareTensorIndexFind(index, name, &tensor) ||
        tensor.weightBytes != Hidden * 2 || tensor.scaleBytes || tensor.biasBytes) return false;
    for (uint32_t column = 0; column < Hidden; ++column)
        output[column] = ANEFieldfareBF16(tensor.weights + (uint64_t)column * 2);
    return true;
}

bool ANEFieldfareDecoderOpen(const char *modelDirectory,
                             uint64_t maximumContext,
                             ANEFieldfareDecoder **result) {
    if (!modelDirectory || !result || !maximumContext) return false;
    *result = NULL;
    ANEFieldfareDecoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) return false;
    decoder->index.fileDescriptor = -1;
    decoder->modelDirectory = strdup(modelDirectory);
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/model_weights.bin", modelDirectory);
    bool valid = decoder->modelDirectory && length > 0 && (size_t)length < sizeof(path) &&
        ANEFieldfareTensorIndexOpen(path, &decoder->index);
    for (uint32_t layer = 0; valid && layer < LayerCount; ++layer) {
        valid = ANEFieldfareAttentionWeightsLoad(&decoder->index, layer, &decoder->attention[layer]) &&
            findNorm(&decoder->index, layer, "input_layernorm.weight", decoder->inputNorm[layer]) &&
            findNorm(&decoder->index, layer, "post_attention_layernorm.weight", decoder->postAttentionNorm[layer]);
        uint64_t capacity = decoder->attention[layer].descriptor.kind == ANEFieldfareAttentionSliding
            ? decoder->attention[layer].descriptor.slidingWindow : maximumContext;
        valid = valid && ANEFieldfareKVCacheInitialize(
            &decoder->caches[layer], decoder->attention[layer].descriptor, capacity);
        valid = valid && ANEFieldfareRoutedLayerOpenForDispatch(
            &decoder->index, layer, &decoder->routed[layer]);
    }
    const char *tracePath = getenv("ANE_FIELDFARE_ROUTE_TRACE");
    if (valid && tracePath && *tracePath) {
        decoder->routeTrace = fopen(tracePath, "w");
        valid = decoder->routeTrace != NULL;
    }
    if (!valid) {
        ANEFieldfareDecoderClose(decoder);
        return false;
    }
    *result = decoder;
    return true;
}

void ANEFieldfareDecoderClose(ANEFieldfareDecoder *decoder) {
    if (!decoder) return;
    if (decoder->routeTrace) fclose(decoder->routeTrace);
    for (uint32_t layer = 0; layer < LayerCount; ++layer)
        ANEFieldfareKVCacheDestroy(&decoder->caches[layer]);
    for (uint32_t layer = 0; layer < LayerCount; ++layer)
        ANEFieldfareRoutedLayerClose(decoder->routed[layer]);
    ANEFieldfareTensorIndexClose(&decoder->index);
    free(decoder->modelDirectory);
    free(decoder);
}

void ANEFieldfareDecoderResetTiming(ANEFieldfareDecoder *decoder) {
    if (decoder) decoder->timing = (ANEFieldfareDecoderTiming){0};
}

ANEFieldfareDecoderTiming ANEFieldfareDecoderGetTiming(const ANEFieldfareDecoder *decoder) {
    return decoder ? decoder->timing : (ANEFieldfareDecoderTiming){0};
}

typedef struct {
    ANEFieldfareDecoderExpertDispatch dispatch;
    void *context;
    uint32_t layer;
} DispatchBridge;

static bool bridgeDispatch(void *raw,
                           uint32_t expert,
                           const float *input,
                           float *output) {
    DispatchBridge *bridge = raw;
    return bridge->dispatch(bridge->context, bridge->layer, expert, input, output);
}

typedef struct {
    ANEFieldfareDecoderExpertBatchDispatch dispatch;
    void *context;
    uint32_t layer;
} BatchDispatchBridge;

static bool bridgeBatchDispatch(
    void *raw,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float *input,
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize]) {
    BatchDispatchBridge *bridge = raw;
    return bridge->dispatch(bridge->context, bridge->layer, experts, input, outputs);
}

typedef struct {
    ANEFieldfareDecoderExpertReducedDispatch dispatch;
    void *context;
    uint32_t layer;
} ReducedDispatchBridge;

static bool bridgeReducedDispatch(
    void *raw,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float weights[ANEFieldfareGemma4TopK],
    const float *input,
    float *reduced) {
    ReducedDispatchBridge *bridge = raw;
    return bridge->dispatch(
        bridge->context, bridge->layer, experts, weights, input, reduced);
}

typedef struct {
    ANEFieldfareDecoderExpertReducedPositionDispatch dispatch;
    void *context;
    uint32_t layer;
} ReducedPositionDispatchBridge;

static bool bridgeReducedPositionDispatch(
    void *raw, uint32_t positionCount, const uint32_t *experts,
    const float *weights, const float *inputs, float *reduced) {
    ReducedPositionDispatchBridge *bridge = raw;
    return bridge->dispatch(bridge->context, bridge->layer, positionCount,
                            experts, weights, inputs, reduced);
}

bool ANEFieldfareDecoderToken(ANEFieldfareDecoder *decoder,
                              uint64_t position,
                              float hidden[Hidden],
                              ANEFieldfareDecoderExpertDispatch expertDispatch,
                              void *expertContext) {
    if (!decoder || !hidden) return false;
    float normalized[Hidden], attentionOutput[Hidden], residual[Hidden];
    for (uint32_t layerIndex = 0; layerIndex < LayerCount; ++layerIndex) {
        memcpy(residual, hidden, sizeof(residual));
        uint64_t phaseStart = mach_absolute_time();
        if (!ANEFieldfareRMSNorm(hidden, decoder->inputNorm[layerIndex], Hidden, 1e-6f, normalized))
            return false;
        decoder->timing.inputNormalizationMilliseconds += elapsedMilliseconds(phaseStart);
        phaseStart = mach_absolute_time();
        if (!ANEFieldfareAttentionDecode(&decoder->attention[layerIndex],
                                         &decoder->caches[layerIndex], position,
                                         normalized, attentionOutput)) return false;
        decoder->timing.attentionMilliseconds += elapsedMilliseconds(phaseStart);
        phaseStart = mach_absolute_time();
        if (!ANEFieldfareRMSNorm(attentionOutput, decoder->postAttentionNorm[layerIndex],
                                 Hidden, 1e-6f, normalized)) return false;
        for (uint32_t column = 0; column < Hidden; ++column)
            hidden[column] = residual[column] + normalized[column];
        decoder->timing.postAttentionMilliseconds += elapsedMilliseconds(phaseStart);

        ANEFieldfareRoutedLayer *routed = decoder->routed[layerIndex];
        ANEFieldfareRoutedLayer *cpuRouted = NULL;
        if (!expertDispatch) {
            if (!ANEFieldfareRoutedLayerOpen(decoder->modelDirectory, layerIndex, &cpuRouted)) return false;
            routed = cpuRouted;
        }
        DispatchBridge bridge = {expertDispatch, expertContext, layerIndex};
        phaseStart = mach_absolute_time();
        bool valid = ANEFieldfareRoutedLayerForward(
            routed, hidden, expertDispatch ? bridgeDispatch : NULL,
            expertDispatch ? &bridge : NULL, residual, NULL);
        decoder->timing.routedFeedForwardMilliseconds += elapsedMilliseconds(phaseStart);
        ++decoder->timing.layerCount;
        ANEFieldfareRoutedLayerClose(cpuRouted);
        if (!valid) return false;
        memcpy(hidden, residual, sizeof(residual));
    }
    ++decoder->timing.tokenCount;
    return true;
}

bool ANEFieldfareDecoderTokenBatched(ANEFieldfareDecoder *decoder,
                                     uint64_t position,
                                     float hidden[Hidden],
                                     ANEFieldfareDecoderExpertBatchDispatch expertDispatch,
                                     void *expertContext) {
    if (!decoder || !hidden || !expertDispatch) return false;
    float normalized[Hidden], attentionOutput[Hidden], residual[Hidden];
    for (uint32_t layerIndex = 0; layerIndex < LayerCount; ++layerIndex) {
        memcpy(residual, hidden, sizeof(residual));
        uint64_t phaseStart = mach_absolute_time();
        if (!ANEFieldfareRMSNorm(hidden, decoder->inputNorm[layerIndex], Hidden, 1e-6f, normalized))
            return false;
        decoder->timing.inputNormalizationMilliseconds += elapsedMilliseconds(phaseStart);
        phaseStart = mach_absolute_time();
        if (!ANEFieldfareAttentionDecode(&decoder->attention[layerIndex],
                                         &decoder->caches[layerIndex], position,
                                         normalized, attentionOutput)) return false;
        decoder->timing.attentionMilliseconds += elapsedMilliseconds(phaseStart);
        phaseStart = mach_absolute_time();
        if (!ANEFieldfareRMSNorm(attentionOutput, decoder->postAttentionNorm[layerIndex],
                                 Hidden, 1e-6f, normalized)) return false;
        for (uint32_t column = 0; column < Hidden; ++column)
            hidden[column] = residual[column] + normalized[column];
        decoder->timing.postAttentionMilliseconds += elapsedMilliseconds(phaseStart);
        BatchDispatchBridge bridge = {expertDispatch, expertContext, layerIndex};
        phaseStart = mach_absolute_time();
        bool valid = ANEFieldfareRoutedLayerForwardBatched(
            decoder->routed[layerIndex], hidden, bridgeBatchDispatch, &bridge, residual, NULL);
        decoder->timing.routedFeedForwardMilliseconds += elapsedMilliseconds(phaseStart);
        ++decoder->timing.layerCount;
        if (!valid) return false;
        memcpy(hidden, residual, sizeof(residual));
    }
    ++decoder->timing.tokenCount;
    return true;
}

bool ANEFieldfareDecoderTokenFusedDevice(
    ANEFieldfareDecoder *decoder,
    uint64_t position,
    float hidden[Hidden],
    ANEFieldfareAttentionInputDispatch attentionInputDispatch,
    ANEFieldfareAttentionAffineDispatch attentionDispatch,
    void *attentionContext,
    ANEFieldfareAttentionDevicePositionDispatch deviceAttentionDispatch,
    void *deviceAttentionContext,
    ANEFieldfareDecoderExpertReducedDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext) {
    if (!decoder || !hidden || !expertDispatch) return false;
    float normalized[Hidden], attentionOutput[Hidden], residual[Hidden];
    for (uint32_t layerIndex = 0; layerIndex < LayerCount; ++layerIndex) {
        memcpy(residual, hidden, sizeof(residual));
        uint64_t phaseStart = mach_absolute_time();
        if (!ANEFieldfareRMSNorm(hidden, decoder->inputNorm[layerIndex], Hidden, 1e-6f, normalized))
            return false;
        decoder->timing.inputNormalizationMilliseconds += elapsedMilliseconds(phaseStart);
        phaseStart = mach_absolute_time();
        bool attentionValid = deviceAttentionDispatch
            ? deviceAttentionDispatch(
                  deviceAttentionContext, layerIndex, &decoder->attention[layerIndex],
                  &decoder->caches[layerIndex], decoder->index.bytes, position, 1,
                  normalized, attentionOutput)
            : ANEFieldfareAttentionDecodeDispatched(
                  &decoder->attention[layerIndex], &decoder->caches[layerIndex], position,
                  normalized, decoder->index.bytes, attentionInputDispatch,
                  attentionDispatch, attentionContext, attentionOutput);
        if (!attentionValid) return false;
        decoder->timing.attentionMilliseconds += elapsedMilliseconds(phaseStart);
        phaseStart = mach_absolute_time();
        if (!ANEFieldfareRMSNorm(attentionOutput, decoder->postAttentionNorm[layerIndex],
                                 Hidden, 1e-6f, normalized)) return false;
        for (uint32_t column = 0; column < Hidden; ++column)
            hidden[column] = residual[column] + normalized[column];
        decoder->timing.postAttentionMilliseconds += elapsedMilliseconds(phaseStart);

        ReducedDispatchBridge bridge = {expertDispatch, expertContext, layerIndex};
        phaseStart = mach_absolute_time();
        FILE *trace = decoder->routeTrace;
        ANEFieldfareRoute traced[ANEFieldfareGemma4TopK];
        bool valid = ANEFieldfareRoutedLayerForwardHybrid(
            decoder->routed[layerIndex], hidden, bridgeReducedDispatch, &bridge,
            sharedDispatch, sharedContext,
            residual, trace ? traced : NULL);
        if (valid) valid = routeTraceRecord(trace, position, layerIndex, traced);
        decoder->timing.routedFeedForwardMilliseconds += elapsedMilliseconds(phaseStart);
        ++decoder->timing.layerCount;
        if (!valid) return false;
        memcpy(hidden, residual, sizeof(residual));
    }
    ++decoder->timing.tokenCount;
    return true;
}

bool ANEFieldfareDecoderTokenFused(
    ANEFieldfareDecoder *decoder,
    uint64_t position,
    float hidden[Hidden],
    ANEFieldfareAttentionInputDispatch attentionInputDispatch,
    ANEFieldfareAttentionAffineDispatch attentionDispatch,
    void *attentionContext,
    ANEFieldfareDecoderExpertReducedDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext) {
    return ANEFieldfareDecoderTokenFusedDevice(
        decoder, position, hidden, attentionInputDispatch, attentionDispatch,
        attentionContext, NULL, NULL, expertDispatch, expertContext,
        sharedDispatch, sharedContext);
}

bool ANEFieldfareDecoderTokenHybrid(
    ANEFieldfareDecoder *decoder,
    uint64_t position,
    float hidden[Hidden],
    ANEFieldfareDecoderExpertReducedDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext) {
    return ANEFieldfareDecoderTokenFused(
        decoder, position, hidden, NULL, NULL, NULL, expertDispatch, expertContext,
        sharedDispatch, sharedContext);
}

bool ANEFieldfareDecoderTokenReducedBatched(
    ANEFieldfareDecoder *decoder,
    uint64_t position,
    float hidden[Hidden],
    ANEFieldfareDecoderExpertReducedDispatch expertDispatch,
    void *expertContext) {
    return ANEFieldfareDecoderTokenHybrid(
        decoder, position, hidden, expertDispatch, expertContext, NULL, NULL);
}

typedef struct PrefillLayerWork PrefillLayerWork;

typedef struct {
    PrefillLayerWork *layer;
    uint32_t row;
} PrefillRowWork;

struct PrefillLayerWork {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    ANEFieldfareDecoderExpertPositionDispatch dispatch;
    void *context;
    uint32_t layerIndex;
    uint32_t count;
    uint32_t arrived;
    bool complete;
    bool valid;
    ANEFieldfareRoute *routes;
    uint32_t *experts;
    float *expertInputs;
    float **expertOutputs;
    uint8_t *coverage;
    float *gatheredInputs;
    float *gatheredOutputs;
};

static bool prefillJoinedDispatch(
    void *raw,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float *input,
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize]) {
    PrefillRowWork *rowWork = raw;
    PrefillLayerWork *work = rowWork->layer;
    const uint32_t row = rowWork->row;
    pthread_mutex_lock(&work->mutex);
    memcpy(work->experts + (uint64_t)row * ANEFieldfareGemma4TopK,
           experts, ANEFieldfareGemma4TopK * sizeof(*experts));
    memcpy(work->expertInputs + (uint64_t)row * Hidden, input, Hidden * sizeof(*input));
    work->expertOutputs[row] = &outputs[0][0];
    // Missing scatter work must be observable even if a future edit weakens the
    // explicit coverage check below.
    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
        for (uint32_t column = 0; column < Hidden; ++column)
            outputs[rank][column] = NAN;
    ++work->arrived;
    if (work->arrived == work->count) {
        for (uint32_t expert = 0; work->valid && expert < ANEFieldfareGemma4ExpertCount; ++expert) {
            uint32_t selectedRows[64];
            uint32_t selectedRanks[64];
            uint32_t selectedCount = 0;
            for (uint32_t candidateRow = 0; candidateRow < work->count; ++candidateRow) {
                const uint32_t *rowExperts = work->experts +
                    (uint64_t)candidateRow * ANEFieldfareGemma4TopK;
                for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
                    if (rowExperts[rank] != expert) continue;
                    // Top-k routing must not select one expert twice for a row.
                    if (selectedCount >= work->count) { work->valid = false; break; }
                    selectedRows[selectedCount] = candidateRow;
                    selectedRanks[selectedCount] = rank;
                    memcpy(work->gatheredInputs + (uint64_t)selectedCount * Hidden,
                           work->expertInputs + (uint64_t)candidateRow * Hidden,
                           Hidden * sizeof(float));
                    ++selectedCount;
                }
            }
            if (!work->valid || !selectedCount) continue;
            work->valid = work->dispatch(
                work->context, work->layerIndex, expert, selectedCount,
                work->gatheredInputs, work->gatheredOutputs);
            for (uint32_t lane = 0; work->valid && lane < selectedCount; ++lane) {
                const uint32_t selectedRow = selectedRows[lane];
                const uint32_t selectedRank = selectedRanks[lane];
                const uint64_t coverageIndex =
                    (uint64_t)selectedRow * ANEFieldfareGemma4TopK + selectedRank;
                if (work->coverage[coverageIndex] != 0) { work->valid = false; break; }
                memcpy(work->expertOutputs[selectedRow] + (uint64_t)selectedRank * Hidden,
                       work->gatheredOutputs + (uint64_t)lane * Hidden,
                       Hidden * sizeof(float));
                work->coverage[coverageIndex] = 1;
            }
        }
        for (uint32_t index = 0;
             work->valid && index < work->count * ANEFieldfareGemma4TopK; ++index)
            work->valid = work->coverage[index] == 1;
        work->complete = true;
        pthread_cond_broadcast(&work->condition);
    } else {
        while (!work->complete) pthread_cond_wait(&work->condition, &work->mutex);
    }
    bool valid = work->valid;
    pthread_mutex_unlock(&work->mutex);
    return valid;
}

typedef struct {
    ANEFieldfareRoutedLayer *routed;
    const float *input;
    float *output;
    ANEFieldfareRoute *routes;
    PrefillRowWork dispatch;
    ANEFieldfareSharedFFNDispatch sharedDispatch;
    void *sharedContext;
    bool valid;
} PrefillForwardWork;

static void *prefillForward(void *raw) {
    PrefillForwardWork *work = raw;
    work->valid = ANEFieldfareRoutedLayerForwardBatchedHybrid(
        work->routed, work->input, prefillJoinedDispatch, &work->dispatch,
        work->sharedDispatch, work->sharedContext,
        work->output, work->routes);
    return NULL;
}

bool ANEFieldfareDecoderPrefillBatchedFused(
    ANEFieldfareDecoder *decoder,
    uint64_t firstPosition,
    uint32_t count,
    float *hiddenRows,
    ANEFieldfareAttentionInputPositionDispatch attentionInputDispatch,
    ANEFieldfareAttentionAffinePositionDispatch attentionDispatch,
    void *attentionContext,
    ANEFieldfareAttentionDevicePositionDispatch deviceAttentionDispatch,
    void *deviceAttentionContext,
    ANEFieldfareDecoderExpertPositionDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareDecoderExpertReducedDispatch reducedDispatch,
    void *reducedContext,
    ANEFieldfareDecoderExpertReducedPositionDispatch reducedPositionDispatch,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext,
    ANEFieldfareSharedFFNPositionDispatch sharedPositionDispatch) {
    if (!decoder || !hiddenRows || (!expertDispatch && !reducedDispatch && !reducedPositionDispatch) ||
        !count || count > 64 ||
        firstPosition > UINT64_MAX - (count - 1)) return false;

    const size_t rowFloats = (size_t)count * Hidden;
    float *normalized = malloc(rowFloats * sizeof(float));
    float *attentionOutputs = malloc(rowFloats * sizeof(float));
    float *attentionResiduals = malloc(rowFloats * sizeof(float));
    float *layerOutputs = malloc(rowFloats * sizeof(float));
    PrefillForwardWork *forward = calloc(count, sizeof(*forward));
    pthread_t *threads = calloc(count, sizeof(*threads));
    bool valid = normalized && attentionOutputs && attentionResiduals && layerOutputs && forward && threads;

    for (uint32_t layerIndex = 0; valid && layerIndex < LayerCount; ++layerIndex) {
        uint64_t phaseStart = mach_absolute_time();
        for (uint32_t row = 0; valid && row < count; ++row) {
            float *hidden = hiddenRows + (uint64_t)row * Hidden;
            float *normalizedRow = normalized + (uint64_t)row * Hidden;
            memcpy(attentionResiduals + (uint64_t)row * Hidden, hidden, Hidden * sizeof(float));
            valid = ANEFieldfareRMSNorm(hidden, decoder->inputNorm[layerIndex],
                                        Hidden, 1e-6f, normalizedRow);
        }
        decoder->timing.inputNormalizationMilliseconds += elapsedMilliseconds(phaseStart);
        phaseStart = mach_absolute_time();
        if (deviceAttentionDispatch) {
            valid = deviceAttentionDispatch(
                deviceAttentionContext, layerIndex, &decoder->attention[layerIndex],
                &decoder->caches[layerIndex], decoder->index.bytes,
                firstPosition, count, normalized,
                attentionOutputs);
        } else if (attentionInputDispatch || attentionDispatch) {
            valid = ANEFieldfareAttentionPrefillDispatched(
                &decoder->attention[layerIndex], &decoder->caches[layerIndex],
                firstPosition, count, normalized, decoder->index.bytes,
                attentionInputDispatch, attentionDispatch, attentionContext,
                attentionOutputs);
        } else {
            // KV mutation remains causal in the non-batched numerical control.
            for (uint32_t row = 0; valid && row < count; ++row)
                valid = ANEFieldfareAttentionDecode(
                    &decoder->attention[layerIndex], &decoder->caches[layerIndex],
                    firstPosition + row, normalized + (uint64_t)row * Hidden,
                    attentionOutputs + (uint64_t)row * Hidden);
        }
        decoder->timing.attentionMilliseconds += elapsedMilliseconds(phaseStart);
        phaseStart = mach_absolute_time();
        for (uint32_t row = 0; valid && row < count; ++row) {
            float *normalizedRow = normalized + (uint64_t)row * Hidden;
            valid = ANEFieldfareRMSNorm(
                attentionOutputs + (uint64_t)row * Hidden,
                decoder->postAttentionNorm[layerIndex], Hidden, 1e-6f, normalizedRow);
            for (uint32_t column = 0; valid && column < Hidden; ++column)
                hiddenRows[(uint64_t)row * Hidden + column] =
                    attentionResiduals[(uint64_t)row * Hidden + column] + normalizedRow[column];
        }
        decoder->timing.postAttentionMilliseconds += elapsedMilliseconds(phaseStart);
        if (!valid) break;

        if (!expertDispatch) {
            phaseStart = mach_absolute_time();
            if (sharedPositionDispatch && reducedPositionDispatch) {
                ReducedPositionDispatchBridge bridge = {
                    reducedPositionDispatch, reducedContext, layerIndex};
                valid = ANEFieldfareRoutedLayerForwardPositionsGroupedHybrid(
                    decoder->routed[layerIndex], count, hiddenRows,
                    bridgeReducedPositionDispatch, &bridge,
                    sharedPositionDispatch, sharedContext, layerOutputs);
            } else if (sharedPositionDispatch) {
                ReducedDispatchBridge bridge = {
                    reducedDispatch, reducedContext, layerIndex};
                valid = ANEFieldfareRoutedLayerForwardPositionsHybrid(
                    decoder->routed[layerIndex], count, hiddenRows,
                    bridgeReducedDispatch, &bridge,
                    sharedPositionDispatch, sharedContext, layerOutputs);
            } else {
                for (uint32_t row = 0; valid && row < count; ++row) {
                    ReducedDispatchBridge bridge = {
                        reducedDispatch, reducedContext, layerIndex};
                    valid = ANEFieldfareRoutedLayerForwardHybrid(
                        decoder->routed[layerIndex],
                        hiddenRows + (uint64_t)row * Hidden,
                        bridgeReducedDispatch, &bridge,
                        sharedDispatch, sharedContext,
                        layerOutputs + (uint64_t)row * Hidden, NULL);
                }
            }
            decoder->timing.routedFeedForwardMilliseconds += elapsedMilliseconds(phaseStart);
            if (valid) {
                memcpy(hiddenRows, layerOutputs, rowFloats * sizeof(float));
                decoder->timing.layerCount += count;
            }
            continue;
        }

        ANEFieldfareRoute *routes = calloc(
            (size_t)count * ANEFieldfareGemma4TopK, sizeof(*routes));
        uint32_t *experts = calloc(
            (size_t)count * ANEFieldfareGemma4TopK, sizeof(*experts));
        float *expertInputs = malloc(rowFloats * sizeof(float));
        float **expertOutputs = calloc(count, sizeof(*expertOutputs));
        uint8_t *coverage = calloc((size_t)count * ANEFieldfareGemma4TopK, 1);
        float *gatheredInputs = malloc(rowFloats * sizeof(float));
        float *gatheredOutputs = malloc(rowFloats * sizeof(float));
        PrefillLayerWork joined = {
            .dispatch = expertDispatch, .context = expertContext,
            .layerIndex = layerIndex, .count = count, .valid = true,
            .routes = routes, .experts = experts, .expertInputs = expertInputs,
            .expertOutputs = expertOutputs, .coverage = coverage,
            .gatheredInputs = gatheredInputs, .gatheredOutputs = gatheredOutputs,
        };
        valid = routes && experts && expertInputs && expertOutputs && coverage &&
            gatheredInputs && gatheredOutputs;
        // A failed route would otherwise prevent that row from reaching the
        // rendezvous. Preflight it before creating any threads so failure stays
        // clean rather than stranding the other rows at the barrier.
        for (uint32_t row = 0; valid && row < count; ++row)
            valid = ANEFieldfareRoutedLayerRoute(
                decoder->routed[layerIndex], hiddenRows + (uint64_t)row * Hidden,
                routes + (uint64_t)row * ANEFieldfareGemma4TopK);
        bool mutexInitialized = valid && pthread_mutex_init(&joined.mutex, NULL) == 0;
        valid = valid && mutexInitialized;
        bool conditionInitialized = valid && pthread_cond_init(&joined.condition, NULL) == 0;
        valid = valid && conditionInitialized;
        uint32_t created = 0;
        phaseStart = mach_absolute_time();
        for (uint32_t row = 0; valid && row < count; ++row) {
            forward[row] = (PrefillForwardWork){
                .routed = decoder->routed[layerIndex],
                .input = hiddenRows + (uint64_t)row * Hidden,
                .output = layerOutputs + (uint64_t)row * Hidden,
                .routes = routes + (uint64_t)row * ANEFieldfareGemma4TopK,
                .dispatch = {.layer = &joined, .row = row},
                .sharedDispatch = sharedDispatch,
                .sharedContext = sharedContext,
            };
            valid = pthread_create(&threads[row], NULL, prefillForward, &forward[row]) == 0;
            if (valid) ++created;
        }
        if (!valid && created) {
            pthread_mutex_lock(&joined.mutex);
            joined.valid = false;
            joined.complete = true;
            pthread_cond_broadcast(&joined.condition);
            pthread_mutex_unlock(&joined.mutex);
        }
        for (uint32_t row = 0; row < created; ++row) {
            bool joinedThread = pthread_join(threads[row], NULL) == 0;
            valid = valid && joinedThread && forward[row].valid;
        }
        valid = valid && joined.valid;
        decoder->timing.routedFeedForwardMilliseconds += elapsedMilliseconds(phaseStart);
        if (valid) {
            memcpy(hiddenRows, layerOutputs, rowFloats * sizeof(float));
            FILE *trace = decoder->routeTrace;
            for (uint32_t row = 0; valid && row < count; ++row)
                valid = routeTraceRecord(trace, firstPosition + row, layerIndex,
                                 routes + (uint64_t)row * ANEFieldfareGemma4TopK);
            decoder->timing.layerCount += count;
        }
        if (conditionInitialized) pthread_cond_destroy(&joined.condition);
        if (mutexInitialized) pthread_mutex_destroy(&joined.mutex);
        free(routes); free(experts); free(expertInputs); free(expertOutputs);
        free(coverage); free(gatheredInputs); free(gatheredOutputs);
    }
    if (valid) decoder->timing.tokenCount += count;
    free(normalized); free(attentionOutputs); free(attentionResiduals); free(layerOutputs);
    free(forward); free(threads);
    return valid;
}

bool ANEFieldfareDecoderPrefillBatched(
    ANEFieldfareDecoder *decoder,
    uint64_t firstPosition,
    uint32_t count,
    float *hiddenRows,
    ANEFieldfareDecoderExpertPositionDispatch expertDispatch,
    void *expertContext) {
    return ANEFieldfareDecoderPrefillBatchedFused(
        decoder, firstPosition, count, hiddenRows, NULL, NULL, NULL, NULL, NULL,
        expertDispatch, expertContext, NULL, NULL, NULL, NULL, NULL, NULL);
}
