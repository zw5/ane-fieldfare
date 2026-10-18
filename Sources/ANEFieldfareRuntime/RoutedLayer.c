#include "RoutedLayer.h"

#include <fcntl.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    QuantGroupSize = 64,
    ExpertStride = 3358720,
    GateWeightOffset = 0,
    GateScaleOffset = 991232,
    GateBiasOffset = 1053184,
    UpWeightOffset = 1115136,
    UpScaleOffset = 2106368,
    UpBiasOffset = 2168320,
    DownWeightOffset = 2230272,
    DownScaleOffset = 3221504,
    DownBiasOffset = 3283456,
};

struct ANEFieldfareRoutedLayer {
    ANEFieldfareTensorIndex index;
    const ANEFieldfareTensorIndex *tensorIndex;
    bool ownsIndex;
    uint32_t layerIndex;
    int expertDescriptor;
    const uint8_t *expertBytes;
    uint64_t expertFileBytes;

    ANEFieldfareTensor routerProjection;
    ANEFieldfareTensor sharedGate;
    ANEFieldfareTensor sharedUp;
    ANEFieldfareTensor sharedDown;

    float routerScale[ANEFieldfareGemma4HiddenSize];
    float perExpertScale[ANEFieldfareGemma4ExpertCount];
    float preFeedforwardScale[ANEFieldfareGemma4HiddenSize];
    float preFeedforwardScale2[ANEFieldfareGemma4HiddenSize];
    float postFeedforwardScale[ANEFieldfareGemma4HiddenSize];
    float postFeedforwardScale1[ANEFieldfareGemma4HiddenSize];
    float postFeedforwardScale2[ANEFieldfareGemma4HiddenSize];
    float layerScalar;
};

static bool findLayerTensor(const ANEFieldfareTensorIndex *index,
                            uint32_t layerIndex,
                            const char *suffix,
                            ANEFieldfareTensor *tensor);
static bool rawBF16(const ANEFieldfareTensor *tensor, uint32_t count, float *output);

static bool loadLayerTensors(ANEFieldfareRoutedLayer *layer,
                             const ANEFieldfareTensorIndex *index,
                             uint32_t layerIndex) {
    ANEFieldfareTensor routerScale = {0}, perExpertScale = {0};
    ANEFieldfareTensor pre = {0}, pre2 = {0}, post = {0}, post1 = {0}, post2 = {0}, scalar = {0};
    bool valid =
        findLayerTensor(index, layerIndex, "router.proj.weight", &layer->routerProjection) &&
        findLayerTensor(index, layerIndex, "router.scale", &routerScale) &&
        findLayerTensor(index, layerIndex, "router.per_expert_scale", &perExpertScale) &&
        findLayerTensor(index, layerIndex, "mlp.gate_proj.weight", &layer->sharedGate) &&
        findLayerTensor(index, layerIndex, "mlp.up_proj.weight", &layer->sharedUp) &&
        findLayerTensor(index, layerIndex, "mlp.down_proj.weight", &layer->sharedDown) &&
        findLayerTensor(index, layerIndex, "pre_feedforward_layernorm.weight", &pre) &&
        findLayerTensor(index, layerIndex, "pre_feedforward_layernorm_2.weight", &pre2) &&
        findLayerTensor(index, layerIndex, "post_feedforward_layernorm.weight", &post) &&
        findLayerTensor(index, layerIndex, "post_feedforward_layernorm_1.weight", &post1) &&
        findLayerTensor(index, layerIndex, "post_feedforward_layernorm_2.weight", &post2) &&
        findLayerTensor(index, layerIndex, "layer_scalar", &scalar);
    return valid &&
        layer->routerProjection.rows == ANEFieldfareGemma4ExpertCount &&
        layer->routerProjection.columns == ANEFieldfareGemma4HiddenSize &&
        layer->sharedGate.rows == ANEFieldfareGemma4SharedIntermediateSize &&
        layer->sharedGate.columns == ANEFieldfareGemma4HiddenSize &&
        layer->sharedUp.rows == ANEFieldfareGemma4SharedIntermediateSize &&
        layer->sharedUp.columns == ANEFieldfareGemma4HiddenSize &&
        layer->sharedDown.rows == ANEFieldfareGemma4HiddenSize &&
        layer->sharedDown.columns == ANEFieldfareGemma4SharedIntermediateSize &&
        rawBF16(&routerScale, ANEFieldfareGemma4HiddenSize, layer->routerScale) &&
        rawBF16(&perExpertScale, ANEFieldfareGemma4ExpertCount, layer->perExpertScale) &&
        rawBF16(&pre, ANEFieldfareGemma4HiddenSize, layer->preFeedforwardScale) &&
        rawBF16(&pre2, ANEFieldfareGemma4HiddenSize, layer->preFeedforwardScale2) &&
        rawBF16(&post, ANEFieldfareGemma4HiddenSize, layer->postFeedforwardScale) &&
        rawBF16(&post1, ANEFieldfareGemma4HiddenSize, layer->postFeedforwardScale1) &&
        rawBF16(&post2, ANEFieldfareGemma4HiddenSize, layer->postFeedforwardScale2) &&
        rawBF16(&scalar, 1, &layer->layerScalar);
}

typedef struct {
    uint32_t expert;
    float probability;
} RankedExpert;

static int compareRankedExperts(const void *leftRaw, const void *rightRaw) {
    const RankedExpert *left = leftRaw;
    const RankedExpert *right = rightRaw;
    if (left->probability > right->probability) return -1;
    if (left->probability < right->probability) return 1;
    return left->expert < right->expert ? -1 : left->expert > right->expert;
}

static bool findLayerTensor(const ANEFieldfareTensorIndex *index,
                            uint32_t layerIndex,
                            const char *suffix,
                            ANEFieldfareTensor *tensor) {
    char name[192];
    int length = snprintf(name, sizeof(name), "language_model.model.layers.%u.%s", layerIndex, suffix);
    return length > 0 && (size_t)length < sizeof(name) && ANEFieldfareTensorIndexFind(index, name, tensor);
}

static bool rawBF16(const ANEFieldfareTensor *tensor, uint32_t count, float *output) {
    if (!tensor || !output || tensor->weightBytes != (uint64_t)count * 2 ||
        tensor->scaleBytes || tensor->biasBytes) return false;
    for (uint32_t index = 0; index < count; ++index)
        output[index] = ANEFieldfareBF16(tensor->weights + (uint64_t)index * 2);
    return true;
}

static bool rmsInverse(const float *input,
                       uint32_t width,
                       float epsilon,
                       float *inverse) {
    if (!input || !inverse || !width || !(epsilon >= 0.0f)) return false;
    double sum = 0.0;
    for (uint32_t index = 0; index < width; ++index) {
        if (!isfinite(input[index])) return false;
        sum += (double)input[index] * input[index];
    }
    *inverse = powf((float)(sum / width) + epsilon, -0.5f);
    return isfinite(*inverse);
}

bool ANEFieldfareRMSNorm(const float *input,
                         const float *scale,
                         uint32_t width,
                         float epsilon,
                         float *output) {
    if (!output) return false;
    float inverse = 0.0f;
    if (!rmsInverse(input, width, epsilon, &inverse)) return false;
    for (uint32_t index = 0; index < width; ++index)
        output[index] = input[index] * inverse * (scale ? scale[index] : 1.0f);
    return true;
}

static float geluTanh(float value) {
    const float cubic = value * value * value;
    return 0.5f * value * (1.0f + tanhf(0.7978845608028654f * (value + 0.044715f * cubic)));
}

static ANEFieldfareTensor packedTensor(const uint8_t *base,
                                       uint32_t rows,
                                       uint32_t columns,
                                       uint32_t weightOffset,
                                       uint32_t scaleOffset,
                                       uint32_t biasOffset) {
    uint64_t elements = (uint64_t)rows * columns;
    uint64_t groups = elements / QuantGroupSize;
    return (ANEFieldfareTensor){
        .weights = base + weightOffset,
        .weightBytes = elements / 2,
        .scales = base + scaleOffset,
        .scaleBytes = groups * 2,
        .biases = base + biasOffset,
        .biasBytes = groups * 2,
        .rows = rows,
        .columns = columns,
    };
}

bool ANEFieldfareRoutedLayerOpen(const char *modelDirectory,
                                 uint32_t layerIndex,
                                 ANEFieldfareRoutedLayer **result) {
    if (!modelDirectory || !result || layerIndex >= 30) return false;
    *result = NULL;
    ANEFieldfareRoutedLayer *layer = calloc(1, sizeof(*layer));
    if (!layer) return false;
    layer->index.fileDescriptor = -1;
    layer->expertDescriptor = -1;
    layer->layerIndex = layerIndex;

    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/model_weights.bin", modelDirectory);
    bool valid = length > 0 && (size_t)length < sizeof(path) &&
        ANEFieldfareTensorIndexOpen(path, &layer->index);

    layer->tensorIndex = &layer->index;
    layer->ownsIndex = true;
    valid = valid && loadLayerTensors(layer, layer->tensorIndex, layerIndex);

    length = snprintf(path, sizeof(path), "%s/packed_experts/layer_%02u.bin", modelDirectory, layerIndex);
    if (valid && length > 0 && (size_t)length < sizeof(path)) {
        layer->expertDescriptor = open(path, O_RDONLY | O_CLOEXEC);
        struct stat status = {0};
        valid = layer->expertDescriptor >= 0 && fstat(layer->expertDescriptor, &status) == 0 &&
            status.st_size == (off_t)((uint64_t)ExpertStride * ANEFieldfareGemma4ExpertCount);
        if (valid) {
            layer->expertFileBytes = (uint64_t)status.st_size;
            layer->expertBytes = mmap(NULL, (size_t)layer->expertFileBytes, PROT_READ, MAP_PRIVATE,
                                      layer->expertDescriptor, 0);
            valid = layer->expertBytes != MAP_FAILED;
            if (!valid) layer->expertBytes = NULL;
        }
    } else {
        valid = false;
    }

    if (!valid) {
        ANEFieldfareRoutedLayerClose(layer);
        return false;
    }
    *result = layer;
    return true;
}

bool ANEFieldfareRoutedLayerOpenForDispatch(const ANEFieldfareTensorIndex *index,
                                            uint32_t layerIndex,
                                            ANEFieldfareRoutedLayer **result) {
    if (!index || !result || layerIndex >= 30) return false;
    *result = NULL;
    ANEFieldfareRoutedLayer *layer = calloc(1, sizeof(*layer));
    if (!layer) return false;
    layer->index.fileDescriptor = -1;
    layer->expertDescriptor = -1;
    layer->layerIndex = layerIndex;
    layer->tensorIndex = index;
    if (!loadLayerTensors(layer, index, layerIndex)) {
        ANEFieldfareRoutedLayerClose(layer);
        return false;
    }
    *result = layer;
    return true;
}

void ANEFieldfareRoutedLayerClose(ANEFieldfareRoutedLayer *layer) {
    if (!layer) return;
    if (layer->expertBytes && layer->expertFileBytes)
        munmap((void *)layer->expertBytes, (size_t)layer->expertFileBytes);
    if (layer->expertDescriptor >= 0) close(layer->expertDescriptor);
    if (layer->ownsIndex) ANEFieldfareTensorIndexClose(&layer->index);
    free(layer);
}

static bool routePrepared(const ANEFieldfareRoutedLayer *layer,
                          const float routerInput[ANEFieldfareGemma4HiddenSize],
                          ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]) {
    if (!layer || !routerInput || !routes) return false;
    float logits[ANEFieldfareGemma4ExpertCount];
    if (!ANEFieldfareAffineMatVec(&layer->routerProjection, 8, QuantGroupSize, routerInput, logits)) return false;

    float maximum = -INFINITY;
    for (uint32_t expert = 0; expert < ANEFieldfareGemma4ExpertCount; ++expert)
        maximum = fmaxf(maximum, logits[expert]);
    double denominator = 0.0;
    float exponentials[ANEFieldfareGemma4ExpertCount];
    for (uint32_t expert = 0; expert < ANEFieldfareGemma4ExpertCount; ++expert) {
        exponentials[expert] = expf(logits[expert] - maximum);
        denominator += exponentials[expert];
    }
    if (!(denominator > 0.0) || !isfinite(denominator)) return false;
    RankedExpert ranking[ANEFieldfareGemma4TopK];
    uint32_t rankedCount = 0;
    for (uint32_t expert = 0; expert < ANEFieldfareGemma4ExpertCount; ++expert) {
        RankedExpert candidate = {
            expert, (float)(exponentials[expert] / denominator),
        };
        uint32_t position = rankedCount < ANEFieldfareGemma4TopK
            ? rankedCount++ : ANEFieldfareGemma4TopK;
        while (position > 0 &&
               compareRankedExperts(&candidate, &ranking[position - 1]) < 0) {
            if (position < ANEFieldfareGemma4TopK)
                ranking[position] = ranking[position - 1];
            --position;
        }
        if (position < ANEFieldfareGemma4TopK) ranking[position] = candidate;
    }

    float selectedSum = 0.0f;
    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
        selectedSum += ranking[rank].probability;
    if (!(selectedSum > 0.0f) || !isfinite(selectedSum)) return false;
    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
        uint32_t expert = ranking[rank].expert;
        routes[rank] = (ANEFieldfareRoute){
            .expert = expert,
            .probability = ranking[rank].probability,
            .weight = ranking[rank].probability / selectedSum * layer->perExpertScale[expert],
        };
    }
    return true;
}

bool ANEFieldfareRoutedLayerRoute(const ANEFieldfareRoutedLayer *layer,
                                  const float input[ANEFieldfareGemma4HiddenSize],
                                  ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]) {
    if (!layer || !input || !routes) return false;
    float normalized[ANEFieldfareGemma4HiddenSize];
    float routerInput[ANEFieldfareGemma4HiddenSize];
    if (!ANEFieldfareRMSNorm(input, NULL, ANEFieldfareGemma4HiddenSize, 1e-6f, normalized)) return false;
    const float scalarRootSize = 1.0f / sqrtf((float)ANEFieldfareGemma4HiddenSize);
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
        routerInput[index] = normalized[index] * layer->routerScale[index] * scalarRootSize;
    return routePrepared(layer, routerInput, routes);
}

bool ANEFieldfareRoutedLayerCPUExpert(const ANEFieldfareRoutedLayer *layer,
                                      uint32_t expert,
                                      const float input[ANEFieldfareGemma4HiddenSize],
                                      float output[ANEFieldfareGemma4HiddenSize]) {
    if (!layer || expert >= ANEFieldfareGemma4ExpertCount || !input || !output) return false;
    const uint8_t *base = layer->expertBytes + (uint64_t)expert * ExpertStride;
    ANEFieldfareTensor gate = packedTensor(base, ANEFieldfareGemma4ExpertIntermediateSize,
                                           ANEFieldfareGemma4HiddenSize,
                                           GateWeightOffset, GateScaleOffset, GateBiasOffset);
    ANEFieldfareTensor up = packedTensor(base, ANEFieldfareGemma4ExpertIntermediateSize,
                                         ANEFieldfareGemma4HiddenSize,
                                         UpWeightOffset, UpScaleOffset, UpBiasOffset);
    ANEFieldfareTensor down = packedTensor(base, ANEFieldfareGemma4HiddenSize,
                                           ANEFieldfareGemma4ExpertIntermediateSize,
                                           DownWeightOffset, DownScaleOffset, DownBiasOffset);
    float gateOutput[ANEFieldfareGemma4ExpertIntermediateSize];
    float upOutput[ANEFieldfareGemma4ExpertIntermediateSize];
    if (!ANEFieldfareAffineMatVec(&gate, 4, QuantGroupSize, input, gateOutput) ||
        !ANEFieldfareAffineMatVec(&up, 4, QuantGroupSize, input, upOutput)) return false;
    for (uint32_t index = 0; index < ANEFieldfareGemma4ExpertIntermediateSize; ++index)
        gateOutput[index] = geluTanh(gateOutput[index]) * upOutput[index];
    return ANEFieldfareAffineMatVec(&down, 4, QuantGroupSize, gateOutput, output);
}

bool ANEFieldfareRoutedLayerForward(ANEFieldfareRoutedLayer *layer,
                                    const float input[ANEFieldfareGemma4HiddenSize],
                                    ANEFieldfareExpertDispatch dispatch,
                                    void *dispatchContext,
                                    float output[ANEFieldfareGemma4HiddenSize],
                                    ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]) {
    if (!layer || !input || !output) return false;
    ANEFieldfareRoute localRoutes[ANEFieldfareGemma4TopK];
    ANEFieldfareRoute *selected = routes ? routes : localRoutes;
    float sharedInput[ANEFieldfareGemma4HiddenSize];
    float sharedGate[ANEFieldfareGemma4SharedIntermediateSize];
    float sharedUp[ANEFieldfareGemma4SharedIntermediateSize];
    float sharedOutput[ANEFieldfareGemma4HiddenSize];
    float expertInput[ANEFieldfareGemma4HiddenSize];
    float expertOutput[ANEFieldfareGemma4HiddenSize];
    float expertReduced[ANEFieldfareGemma4HiddenSize] = {0};
    float combined[ANEFieldfareGemma4HiddenSize];

    bool valid =
        ANEFieldfareRMSNorm(input, layer->preFeedforwardScale,
                            ANEFieldfareGemma4HiddenSize, 1e-6f, sharedInput) &&
        ANEFieldfareAffineMatVec(&layer->sharedGate, 4, QuantGroupSize, sharedInput, sharedGate) &&
        ANEFieldfareAffineMatVec(&layer->sharedUp, 4, QuantGroupSize, sharedInput, sharedUp);
    if (!valid) return false;
    for (uint32_t index = 0; index < ANEFieldfareGemma4SharedIntermediateSize; ++index)
        sharedGate[index] = geluTanh(sharedGate[index]) * sharedUp[index];
    valid = ANEFieldfareAffineMatVec(&layer->sharedDown, 4, QuantGroupSize, sharedGate, sharedOutput) &&
        ANEFieldfareRMSNorm(sharedOutput, layer->postFeedforwardScale1,
                            ANEFieldfareGemma4HiddenSize, 1e-6f, sharedOutput) &&
        ANEFieldfareRoutedLayerRoute(layer, input, selected) &&
        ANEFieldfareRMSNorm(input, layer->preFeedforwardScale2,
                            ANEFieldfareGemma4HiddenSize, 1e-6f, expertInput);
    if (!valid) return false;

    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
        valid = dispatch ? dispatch(dispatchContext, selected[rank].expert, expertInput, expertOutput)
                         : ANEFieldfareRoutedLayerCPUExpert(layer, selected[rank].expert,
                                                            expertInput, expertOutput);
        if (!valid) return false;
        for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
            expertReduced[index] += selected[rank].weight * expertOutput[index];
    }
    if (!ANEFieldfareRMSNorm(expertReduced, layer->postFeedforwardScale2,
                             ANEFieldfareGemma4HiddenSize, 1e-6f, expertReduced)) return false;
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
        combined[index] = sharedOutput[index] + expertReduced[index];
    if (!ANEFieldfareRMSNorm(combined, layer->postFeedforwardScale,
                             ANEFieldfareGemma4HiddenSize, 1e-6f, combined)) return false;
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
        output[index] = (input[index] + combined[index]) * layer->layerScalar;
    return true;
}

typedef struct {
    ANEFieldfareRoutedLayer *layer;
    ANEFieldfareExpertBatchDispatch dispatch;
    void *context;
    const float *residualInput;
    ANEFieldfareRoute *routes;
    uint32_t *experts;
    float *expertInput;
    float (*outputs)[ANEFieldfareGemma4HiddenSize];
    float normalizationInverse;
    bool valid;
} BatchedDispatchWork;

static void *runBatchedDispatch(void *rawWork) {
    BatchedDispatchWork *work = rawWork;
    float routerInput[ANEFieldfareGemma4HiddenSize];
    const float scalarRootSize = 1.0f / sqrtf((float)ANEFieldfareGemma4HiddenSize);
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index) {
        float normalized = work->residualInput[index] * work->normalizationInverse;
        routerInput[index] = normalized * work->layer->routerScale[index] * scalarRootSize;
        work->expertInput[index] = normalized * work->layer->preFeedforwardScale2[index];
    }
    work->valid = routePrepared(work->layer, routerInput, work->routes);
    for (uint32_t rank = 0; work->valid && rank < ANEFieldfareGemma4TopK; ++rank)
        work->experts[rank] = work->routes[rank].expert;
    if (work->valid)
        work->valid = work->dispatch(
            work->context, work->experts, work->expertInput, work->outputs);
    return NULL;
}

bool ANEFieldfareRoutedLayerForwardBatchedHybrid(
    ANEFieldfareRoutedLayer *layer,
    const float input[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareExpertBatchDispatch dispatch,
    void *dispatchContext,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext,
    float output[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]) {
    if (!layer || !input || !output || !dispatch) return false;
    ANEFieldfareRoute localRoutes[ANEFieldfareGemma4TopK];
    ANEFieldfareRoute *selected = routes ? routes : localRoutes;
    float sharedInput[ANEFieldfareGemma4HiddenSize];
    float sharedGate[ANEFieldfareGemma4SharedIntermediateSize];
    float sharedUp[ANEFieldfareGemma4SharedIntermediateSize];
    float sharedOutput[ANEFieldfareGemma4HiddenSize];
    float expertInput[ANEFieldfareGemma4HiddenSize];
    float expertOutputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize];
    float expertReduced[ANEFieldfareGemma4HiddenSize] = {0};
    float combined[ANEFieldfareGemma4HiddenSize];
    uint32_t expertIDs[ANEFieldfareGemma4TopK];

    float sharedInverse = 0.0f;
    if (!rmsInverse(input, ANEFieldfareGemma4HiddenSize, 1e-6f, &sharedInverse)) return false;

    // The router, routed normalization, and expert batch are independent of the
    // shared branch. Run that complete branch on a helper while the caller
    // computes the shared FFN. The join occurs before the original weighted
    // reduction, so all floating-point accumulation orders remain unchanged.
    BatchedDispatchWork work = {
        .layer = layer,
        .dispatch = dispatch,
        .context = dispatchContext,
        .residualInput = input,
        .routes = selected,
        .experts = expertIDs,
        .expertInput = expertInput,
        .outputs = expertOutputs,
        .normalizationInverse = sharedInverse,
    };
    pthread_t worker = {0};
    bool threaded = pthread_create(&worker, NULL, runBatchedDispatch, &work) == 0;
    if (!threaded) runBatchedDispatch(&work);

    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
        sharedInput[index] = input[index] * sharedInverse * layer->preFeedforwardScale[index];
    bool sharedValid = sharedDispatch
        ? sharedDispatch(sharedContext, layer->tensorIndex->bytes,
                         &layer->sharedGate, &layer->sharedUp, &layer->sharedDown,
                         sharedInput, sharedOutput)
        : ANEFieldfareAffineMatVec(
              &layer->sharedGate, 4, QuantGroupSize, sharedInput, sharedGate) &&
          ANEFieldfareAffineMatVec(
              &layer->sharedUp, 4, QuantGroupSize, sharedInput, sharedUp);
    if (sharedValid && !sharedDispatch) {
        for (uint32_t index = 0; index < ANEFieldfareGemma4SharedIntermediateSize; ++index)
            sharedGate[index] = geluTanh(sharedGate[index]) * sharedUp[index];
        sharedValid = ANEFieldfareAffineMatVec(
            &layer->sharedDown, 4, QuantGroupSize, sharedGate, sharedOutput);
    }
    if (sharedValid)
        sharedValid = ANEFieldfareRMSNorm(sharedOutput, layer->postFeedforwardScale1,
                                          ANEFieldfareGemma4HiddenSize, 1e-6f, sharedOutput);
    bool joined = !threaded || pthread_join(worker, NULL) == 0;
    if (!sharedValid || !joined || !work.valid) return false;

    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
        for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
            expertReduced[index] += selected[rank].weight * expertOutputs[rank][index];
    if (!ANEFieldfareRMSNorm(expertReduced, layer->postFeedforwardScale2,
                             ANEFieldfareGemma4HiddenSize, 1e-6f, expertReduced)) return false;
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
        combined[index] = sharedOutput[index] + expertReduced[index];
    if (!ANEFieldfareRMSNorm(combined, layer->postFeedforwardScale,
                             ANEFieldfareGemma4HiddenSize, 1e-6f, combined)) return false;
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
        output[index] = (input[index] + combined[index]) * layer->layerScalar;
    return true;
}

bool ANEFieldfareRoutedLayerForwardBatched(ANEFieldfareRoutedLayer *layer,
                                          const float input[ANEFieldfareGemma4HiddenSize],
                                          ANEFieldfareExpertBatchDispatch dispatch,
                                          void *dispatchContext,
                                          float output[ANEFieldfareGemma4HiddenSize],
                                          ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]) {
    return ANEFieldfareRoutedLayerForwardBatchedHybrid(
        layer, input, dispatch, dispatchContext, NULL, NULL, output, routes);
}

typedef struct {
    ANEFieldfareRoutedLayer *layer;
    ANEFieldfareExpertReducedDispatch dispatch;
    void *context;
    const float *residualInput;
    ANEFieldfareRoute *routes;
    float *expertInput;
    float *expertReduced;
    float normalizationInverse;
    bool valid;
} ReducedDispatchWork;

static void *runReducedDispatch(void *rawWork) {
    ReducedDispatchWork *work = rawWork;
    float routerInput[ANEFieldfareGemma4HiddenSize];
    uint32_t experts[ANEFieldfareGemma4TopK];
    float weights[ANEFieldfareGemma4TopK];
    const float scalarRootSize = 1.0f / sqrtf((float)ANEFieldfareGemma4HiddenSize);
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index) {
        float normalized = work->residualInput[index] * work->normalizationInverse;
        routerInput[index] = normalized * work->layer->routerScale[index] * scalarRootSize;
        work->expertInput[index] = normalized * work->layer->preFeedforwardScale2[index];
    }
    work->valid = routePrepared(work->layer, routerInput, work->routes);
    for (uint32_t rank = 0; work->valid && rank < ANEFieldfareGemma4TopK; ++rank) {
        experts[rank] = work->routes[rank].expert;
        weights[rank] = work->routes[rank].weight;
    }
    if (work->valid)
        work->valid = work->dispatch(
            work->context, experts, weights, work->expertInput, work->expertReduced);
    return NULL;
}

bool ANEFieldfareRoutedLayerForwardHybrid(
    ANEFieldfareRoutedLayer *layer,
    const float input[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareExpertReducedDispatch dispatch,
    void *dispatchContext,
    ANEFieldfareSharedFFNDispatch sharedDispatch,
    void *sharedContext,
    float output[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]) {
    if (!layer || !input || !output || !dispatch) return false;
    ANEFieldfareRoute localRoutes[ANEFieldfareGemma4TopK];
    ANEFieldfareRoute *selected = routes ? routes : localRoutes;
    float sharedInput[ANEFieldfareGemma4HiddenSize];
    float sharedGate[ANEFieldfareGemma4SharedIntermediateSize];
    float sharedUp[ANEFieldfareGemma4SharedIntermediateSize];
    float sharedOutput[ANEFieldfareGemma4HiddenSize];
    float expertInput[ANEFieldfareGemma4HiddenSize];
    float expertReduced[ANEFieldfareGemma4HiddenSize];
    float combined[ANEFieldfareGemma4HiddenSize];
    float inverse = 0.0f;
    if (!rmsInverse(input, ANEFieldfareGemma4HiddenSize, 1e-6f, &inverse)) return false;

    ReducedDispatchWork work = {
        .layer = layer,
        .dispatch = dispatch,
        .context = dispatchContext,
        .residualInput = input,
        .routes = selected,
        .expertInput = expertInput,
        .expertReduced = expertReduced,
        .normalizationInverse = inverse,
    };
    pthread_t worker = {0};
    dispatch_group_t dispatchGroup = NULL;
    bool useWorkerPool = getenv("ANE_FIELDFARE_PERSISTENT_FFN_WORKERS") != NULL;
    bool threaded = false;
    if (useWorkerPool) {
        dispatchGroup = dispatch_group_create();
        ReducedDispatchWork *workPointer = &work;
        dispatch_group_async(dispatchGroup,
            dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
                runReducedDispatch(workPointer);
            });
        threaded = true;
    } else {
        threaded = pthread_create(&worker, NULL, runReducedDispatch, &work) == 0;
        if (!threaded) runReducedDispatch(&work);
    }

    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
        sharedInput[index] = input[index] * inverse * layer->preFeedforwardScale[index];
    bool sharedValid = false;
    if (sharedDispatch) {
        sharedValid = sharedDispatch(sharedContext, layer->tensorIndex->bytes,
                                     &layer->sharedGate, &layer->sharedUp, &layer->sharedDown,
                                     sharedInput, sharedOutput);
    } else {
        sharedValid =
            ANEFieldfareAffineMatVec(&layer->sharedGate, 4, QuantGroupSize, sharedInput, sharedGate) &&
            ANEFieldfareAffineMatVec(&layer->sharedUp, 4, QuantGroupSize, sharedInput, sharedUp);
    }
    if (sharedValid && !sharedDispatch) {
        for (uint32_t index = 0; index < ANEFieldfareGemma4SharedIntermediateSize; ++index)
            sharedGate[index] = geluTanh(sharedGate[index]) * sharedUp[index];
        sharedValid = ANEFieldfareAffineMatVec(
            &layer->sharedDown, 4, QuantGroupSize, sharedGate, sharedOutput);
    }
    if (sharedValid)
        sharedValid = ANEFieldfareRMSNorm(sharedOutput, layer->postFeedforwardScale1,
                                          ANEFieldfareGemma4HiddenSize, 1e-6f, sharedOutput);
    bool joined = true;
    if (dispatchGroup) dispatch_group_wait(dispatchGroup, DISPATCH_TIME_FOREVER);
    else if (threaded) joined = pthread_join(worker, NULL) == 0;
    if (!sharedValid || !joined || !work.valid ||
        !ANEFieldfareRMSNorm(expertReduced, layer->postFeedforwardScale2,
                             ANEFieldfareGemma4HiddenSize, 1e-6f, expertReduced)) return false;
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
        combined[index] = sharedOutput[index] + expertReduced[index];
    if (!ANEFieldfareRMSNorm(combined, layer->postFeedforwardScale,
                             ANEFieldfareGemma4HiddenSize, 1e-6f, combined)) return false;
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
        output[index] = (input[index] + combined[index]) * layer->layerScalar;
    return true;
}

bool ANEFieldfareRoutedLayerForwardReducedBatched(
    ANEFieldfareRoutedLayer *layer,
    const float input[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareExpertReducedDispatch dispatch,
    void *dispatchContext,
    float output[ANEFieldfareGemma4HiddenSize],
    ANEFieldfareRoute routes[ANEFieldfareGemma4TopK]) {
    return ANEFieldfareRoutedLayerForwardHybrid(
        layer, input, dispatch, dispatchContext, NULL, NULL, output, routes);
}

typedef struct {
    ANEFieldfareRoutedLayer *layer;
    ANEFieldfareSharedFFNPositionDispatch dispatch;
    void *context;
    uint32_t count;
    const float *inputs;
    float *outputs;
    bool valid;
} SharedPositionWork;

static void *runSharedPositions(void *raw) {
    SharedPositionWork *work = raw;
    work->valid = work->dispatch(
        work->context, work->layer->tensorIndex->bytes,
        &work->layer->sharedGate, &work->layer->sharedUp, &work->layer->sharedDown,
        work->count, work->inputs, work->outputs);
    return NULL;
}

bool ANEFieldfareRoutedLayerForwardPositionsHybrid(
    ANEFieldfareRoutedLayer *layer,
    uint32_t positionCount,
    const float *inputs,
    ANEFieldfareExpertReducedDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNPositionDispatch sharedDispatch,
    void *sharedContext,
    float *outputs) {
    if (!layer || !positionCount || positionCount > 64 || !inputs || !expertDispatch ||
        !sharedDispatch || !outputs) return false;
    const size_t rows = (size_t)positionCount * ANEFieldfareGemma4HiddenSize;
    float *sharedInputs = malloc(rows * sizeof(float));
    float *sharedOutputs = malloc(rows * sizeof(float));
    float *expertInputs = malloc(rows * sizeof(float));
    float *expertReduced = malloc(rows * sizeof(float));
    ANEFieldfareRoute *routes = calloc(
        (size_t)positionCount * ANEFieldfareGemma4TopK, sizeof(*routes));
    bool valid = sharedInputs && sharedOutputs && expertInputs && expertReduced && routes;
    const float scalarRootSize = 1.0f / sqrtf((float)ANEFieldfareGemma4HiddenSize);
    for (uint32_t row = 0; valid && row < positionCount; ++row) {
        const float *input = inputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float *sharedInput = sharedInputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float *expertInput = expertInputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float routerInput[ANEFieldfareGemma4HiddenSize];
        float inverse = 0.0f;
        valid = rmsInverse(input, ANEFieldfareGemma4HiddenSize, 1e-6f, &inverse);
        for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column) {
            float normalized = input[column] * inverse;
            sharedInput[column] = normalized * layer->preFeedforwardScale[column];
            expertInput[column] = normalized * layer->preFeedforwardScale2[column];
            routerInput[column] = normalized * layer->routerScale[column] * scalarRootSize;
        }
        valid = valid && routePrepared(
            layer, routerInput, routes + (uint64_t)row * ANEFieldfareGemma4TopK);
    }

    SharedPositionWork shared = {
        .layer = layer, .dispatch = sharedDispatch, .context = sharedContext,
        .count = positionCount, .inputs = sharedInputs, .outputs = sharedOutputs,
    };
    pthread_t worker = {0};
    bool threaded = valid && pthread_create(&worker, NULL, runSharedPositions, &shared) == 0;
    if (valid && !threaded) runSharedPositions(&shared);

    for (uint32_t row = 0; valid && row < positionCount; ++row) {
        uint32_t experts[ANEFieldfareGemma4TopK];
        float weights[ANEFieldfareGemma4TopK];
        ANEFieldfareRoute *selected = routes + (uint64_t)row * ANEFieldfareGemma4TopK;
        for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
            experts[rank] = selected[rank].expert;
            weights[rank] = selected[rank].weight;
        }
        valid = expertDispatch(
            expertContext, experts, weights,
            expertInputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize,
            expertReduced + (uint64_t)row * ANEFieldfareGemma4HiddenSize);
    }
    bool joined = !threaded || pthread_join(worker, NULL) == 0;
    valid = valid && joined && shared.valid;
    for (uint32_t row = 0; valid && row < positionCount; ++row) {
        const float *input = inputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float *sharedOutput = sharedOutputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float *expertOutput = expertReduced + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float *output = outputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        valid = ANEFieldfareRMSNorm(sharedOutput, layer->postFeedforwardScale1,
                                    ANEFieldfareGemma4HiddenSize, 1e-6f, sharedOutput) &&
            ANEFieldfareRMSNorm(expertOutput, layer->postFeedforwardScale2,
                                ANEFieldfareGemma4HiddenSize, 1e-6f, expertOutput);
        for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column)
            output[column] = sharedOutput[column] + expertOutput[column];
        valid = valid && ANEFieldfareRMSNorm(output, layer->postFeedforwardScale,
                                             ANEFieldfareGemma4HiddenSize, 1e-6f, output);
        for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column)
            output[column] = (input[column] + output[column]) * layer->layerScalar;
    }
    free(sharedInputs); free(sharedOutputs); free(expertInputs); free(expertReduced); free(routes);
    return valid;
}

bool ANEFieldfareRoutedLayerForwardPositionsGroupedHybrid(
    ANEFieldfareRoutedLayer *layer,
    uint32_t positionCount,
    const float *inputs,
    ANEFieldfareExpertReducedPositionDispatch expertDispatch,
    void *expertContext,
    ANEFieldfareSharedFFNPositionDispatch sharedDispatch,
    void *sharedContext,
    float *outputs) {
    if (!layer || !positionCount || positionCount > 64 || !inputs || !expertDispatch ||
        !sharedDispatch || !outputs) return false;
    const size_t rows = (size_t)positionCount * ANEFieldfareGemma4HiddenSize;
    const size_t routeCount = (size_t)positionCount * ANEFieldfareGemma4TopK;
    float *sharedInputs = malloc(rows * sizeof(float));
    float *sharedOutputs = malloc(rows * sizeof(float));
    float *expertInputs = malloc(rows * sizeof(float));
    float *expertReduced = malloc(rows * sizeof(float));
    ANEFieldfareRoute *routes = calloc(routeCount, sizeof(*routes));
    uint32_t *experts = malloc(routeCount * sizeof(*experts));
    float *weights = malloc(routeCount * sizeof(*weights));
    bool valid = sharedInputs && sharedOutputs && expertInputs && expertReduced && routes &&
        experts && weights;
    const float scalarRootSize = 1.0f / sqrtf((float)ANEFieldfareGemma4HiddenSize);
    for (uint32_t row = 0; valid && row < positionCount; ++row) {
        const float *input = inputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float *sharedInput = sharedInputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float *expertInput = expertInputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float routerInput[ANEFieldfareGemma4HiddenSize];
        float inverse = 0.0f;
        valid = rmsInverse(input, ANEFieldfareGemma4HiddenSize, 1e-6f, &inverse);
        for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column) {
            float normalized = input[column] * inverse;
            sharedInput[column] = normalized * layer->preFeedforwardScale[column];
            expertInput[column] = normalized * layer->preFeedforwardScale2[column];
            routerInput[column] = normalized * layer->routerScale[column] * scalarRootSize;
        }
        ANEFieldfareRoute *selected = routes + (uint64_t)row * ANEFieldfareGemma4TopK;
        valid = valid && routePrepared(layer, routerInput, selected);
        for (uint32_t rank = 0; valid && rank < ANEFieldfareGemma4TopK; ++rank) {
            size_t index = (size_t)row * ANEFieldfareGemma4TopK + rank;
            experts[index] = selected[rank].expert;
            weights[index] = selected[rank].weight;
        }
    }

    SharedPositionWork shared = {
        .layer = layer, .dispatch = sharedDispatch, .context = sharedContext,
        .count = positionCount, .inputs = sharedInputs, .outputs = sharedOutputs,
    };
    pthread_t worker = {0};
    bool threaded = valid && pthread_create(&worker, NULL, runSharedPositions, &shared) == 0;
    if (valid && !threaded) runSharedPositions(&shared);
    if (valid) valid = expertDispatch(
        expertContext, positionCount, experts, weights, expertInputs, expertReduced);
    bool joined = !threaded || pthread_join(worker, NULL) == 0;
    valid = valid && joined && shared.valid;
    for (uint32_t row = 0; valid && row < positionCount; ++row) {
        const float *input = inputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float *sharedOutput = sharedOutputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float *expertOutput = expertReduced + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        float *output = outputs + (uint64_t)row * ANEFieldfareGemma4HiddenSize;
        valid = ANEFieldfareRMSNorm(sharedOutput, layer->postFeedforwardScale1,
                                    ANEFieldfareGemma4HiddenSize, 1e-6f, sharedOutput) &&
            ANEFieldfareRMSNorm(expertOutput, layer->postFeedforwardScale2,
                                ANEFieldfareGemma4HiddenSize, 1e-6f, expertOutput);
        for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column)
            output[column] = sharedOutput[column] + expertOutput[column];
        valid = valid && ANEFieldfareRMSNorm(output, layer->postFeedforwardScale,
                                             ANEFieldfareGemma4HiddenSize, 1e-6f, output);
        for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column)
            output[column] = (input[column] + output[column]) * layer->layerScalar;
    }
    free(sharedInputs); free(sharedOutputs); free(expertInputs); free(expertReduced);
    free(routes); free(experts); free(weights);
    return valid;
}
