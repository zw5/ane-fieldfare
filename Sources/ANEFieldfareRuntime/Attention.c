#include "Attention.h"

#include <dispatch/dispatch.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    GemmaHidden = 2816,
    GemmaQueryHeads = 16,
    GemmaGroupSize = 64,
    GemmaLayers = 30,
    GemmaSlidingWindow = 1024,
};

static bool rawBF16(const ANEFieldfareTensor *tensor, uint32_t count, float *output) {
    if (!tensor || !output || tensor->weightBytes != (uint64_t)count * 2 ||
        tensor->scaleBytes || tensor->biasBytes) return false;
    for (uint32_t index = 0; index < count; ++index)
        output[index] = ANEFieldfareBF16(tensor->weights + (uint64_t)index * 2);
    return true;
}

bool ANEFieldfareAttentionDescriptorForLayer(uint32_t layer,
                                             ANEFieldfareAttentionDescriptor *descriptor) {
    if (!descriptor || layer >= GemmaLayers) return false;
    bool full = layer % 6 == 5;
    *descriptor = (ANEFieldfareAttentionDescriptor){
        .layer = layer,
        .kind = full ? ANEFieldfareAttentionFull : ANEFieldfareAttentionSliding,
        .hiddenSize = GemmaHidden,
        .queryHeads = GemmaQueryHeads,
        .keyValueHeads = full ? 2 : 8,
        .headDimension = full ? 512 : 256,
        .rotaryAngles = full ? 64 : 128,
        .slidingWindow = full ? 0 : GemmaSlidingWindow,
        .ropeTheta = full ? 1000000.0f : 10000.0f,
        .rmsEpsilon = 1e-6f,
        .attentionScale = 1.0f,
        .attentionSoftcap = 0.0f,
        .valueEqualsKeyProjection = full,
        .sharesKeyValueAcrossLayers = false,
    };
    return true;
}

static bool findLayerTensor(const ANEFieldfareTensorIndex *index,
                            uint32_t layer,
                            const char *suffix,
                            ANEFieldfareTensor *tensor) {
    char name[192];
    int length = snprintf(name, sizeof(name),
                          "language_model.model.layers.%u.self_attn.%s", layer, suffix);
    return length > 0 && (size_t)length < sizeof(name) &&
           ANEFieldfareTensorIndexFind(index, name, tensor);
}

static bool affineShape(const ANEFieldfareTensor *tensor, uint32_t rows, uint32_t columns) {
    uint64_t elements = (uint64_t)rows * columns;
    uint64_t groups = elements / GemmaGroupSize;
    return tensor->rows == rows && tensor->columns == columns &&
           tensor->weightBytes == (elements + 1) / 2 &&
           tensor->scaleBytes == groups * 2 && tensor->biasBytes == groups * 2;
}

typedef struct {
    const ANEFieldfareTensor *matrix;
    const float *input;
    float *output;
    uint32_t groupsPerRow;
} AttentionAffineContext;

static bool initializeAttentionAffineContext(const ANEFieldfareTensor *matrix,
                                             const float *input,
                                             float *output,
                                             AttentionAffineContext *context) {
    if (!matrix || !input || !output || !context || !matrix->rows || !matrix->columns ||
        matrix->columns % GemmaGroupSize) return false;
    uint64_t elements = (uint64_t)matrix->rows * matrix->columns;
    uint64_t groups = elements / GemmaGroupSize;
    if (matrix->weightBytes != (elements + 1) / 2 ||
        matrix->scaleBytes != groups * 2 || matrix->biasBytes != groups * 2) return false;
    *context = (AttentionAffineContext){
        .matrix = matrix,
        .input = input,
        .output = output,
        .groupsPerRow = matrix->columns / GemmaGroupSize,
    };
    return true;
}

static void attentionAffineRow(void *rawContext, size_t rowValue) {
    const AttentionAffineContext *context = rawContext;
    const ANEFieldfareTensor *matrix = context->matrix;
    uint32_t row = (uint32_t)rowValue;
    uint64_t rowGroup = (uint64_t)row * context->groupsPerRow;
    const uint8_t *rowWeights = matrix->weights +
        (uint64_t)row * matrix->columns / 2;
    float sum = 0.0f;
    for (uint32_t group = 0; group < context->groupsPerRow; ++group) {
        float scale = ANEFieldfareBF16(matrix->scales + (rowGroup + group) * 2);
        float bias = ANEFieldfareBF16(matrix->biases + (rowGroup + group) * 2);
        const uint8_t *packed = rowWeights + (uint64_t)group * GemmaGroupSize / 2;
        const float *input = context->input + (uint64_t)group * GemmaGroupSize;
        for (uint32_t byte = 0; byte < GemmaGroupSize / 2; ++byte) {
            uint8_t quantized = packed[byte];
            float lowWeight = (quantized & 0x0f) * scale + bias;
            sum += input[2 * byte] * lowWeight;
            float highWeight = (quantized >> 4) * scale + bias;
            sum += input[2 * byte + 1] * highWeight;
        }
    }
    context->output[row] = sum;
}

static bool attentionAffineMatVec(const ANEFieldfareTensor *matrix,
                                  const float *input,
                                  float *output) {
    AttentionAffineContext context = {0};
    if (!initializeAttentionAffineContext(matrix, input, output, &context)) return false;
    dispatch_apply_f(matrix->rows,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &context, attentionAffineRow);
    return true;
}

typedef struct {
    AttentionAffineContext projections[3];
    uint32_t offsets[4];
    uint32_t count;
} AttentionInputProjectionContext;

static void attentionInputProjectionRow(void *rawContext, size_t rowValue) {
    const AttentionInputProjectionContext *context = rawContext;
    uint32_t row = (uint32_t)rowValue;
    for (uint32_t projection = 0; projection < context->count; ++projection) {
        if (row < context->offsets[projection + 1]) {
            attentionAffineRow((void *)&context->projections[projection],
                               row - context->offsets[projection]);
            return;
        }
    }
}

static bool projectAttentionInput(const ANEFieldfareAttentionWeights *weights,
                                  const float *input,
                                  float *query,
                                  float *key,
                                  float *value) {
    AttentionInputProjectionContext context = {0};
    const ANEFieldfareTensor *matrices[3] = {
        &weights->queryProjection, &weights->keyProjection, &weights->valueProjection,
    };
    float *outputs[3] = {query, key, value};
    context.count = weights->descriptor.valueEqualsKeyProjection ? 2 : 3;
    for (uint32_t projection = 0; projection < context.count; ++projection) {
        if (!initializeAttentionAffineContext(matrices[projection], input, outputs[projection],
                                              &context.projections[projection])) return false;
        context.offsets[projection + 1] = context.offsets[projection] + matrices[projection]->rows;
    }
    dispatch_apply_f(context.offsets[context.count],
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &context, attentionInputProjectionRow);
    if (weights->descriptor.valueEqualsKeyProjection)
        memcpy(value, key, (size_t)weights->keyProjection.rows * sizeof(float));
    return true;
}

bool ANEFieldfareAttentionWeightsLoad(const ANEFieldfareTensorIndex *index,
                                      uint32_t layer,
                                      ANEFieldfareAttentionWeights *weights) {
    if (!index || !weights) return false;
    ANEFieldfareAttentionWeights loaded = {0};
    if (!ANEFieldfareAttentionDescriptorForLayer(layer, &loaded.descriptor) ||
        !findLayerTensor(index, layer, "q_proj.weight", &loaded.queryProjection) ||
        !findLayerTensor(index, layer, "k_proj.weight", &loaded.keyProjection) ||
        !findLayerTensor(index, layer, "o_proj.weight", &loaded.outputProjection) ||
        !findLayerTensor(index, layer, "q_norm.weight", &loaded.queryNorm) ||
        !findLayerTensor(index, layer, "k_norm.weight", &loaded.keyNorm)) return false;
    if (!loaded.descriptor.valueEqualsKeyProjection &&
        !findLayerTensor(index, layer, "v_proj.weight", &loaded.valueProjection)) return false;

    uint32_t queryWidth = loaded.descriptor.queryHeads * loaded.descriptor.headDimension;
    uint32_t keyValueWidth = loaded.descriptor.keyValueHeads * loaded.descriptor.headDimension;
    if (!affineShape(&loaded.queryProjection, queryWidth, loaded.descriptor.hiddenSize) ||
        !affineShape(&loaded.keyProjection, keyValueWidth, loaded.descriptor.hiddenSize) ||
        (!loaded.descriptor.valueEqualsKeyProjection &&
         !affineShape(&loaded.valueProjection, keyValueWidth, loaded.descriptor.hiddenSize)) ||
        !affineShape(&loaded.outputProjection, loaded.descriptor.hiddenSize, queryWidth) ||
        loaded.queryNorm.rows != loaded.descriptor.headDimension || loaded.queryNorm.columns != 0 ||
        loaded.queryNorm.weightBytes != (uint64_t)loaded.descriptor.headDimension * 2 ||
        loaded.queryNorm.scaleBytes || loaded.queryNorm.biasBytes ||
        loaded.keyNorm.rows != loaded.descriptor.headDimension || loaded.keyNorm.columns != 0 ||
        loaded.keyNorm.weightBytes != (uint64_t)loaded.descriptor.headDimension * 2 ||
        loaded.keyNorm.scaleBytes || loaded.keyNorm.biasBytes) return false;
    if (!rawBF16(&loaded.queryNorm, loaded.descriptor.headDimension,
                 loaded.queryNormValues) ||
        !rawBF16(&loaded.keyNorm, loaded.descriptor.headDimension,
                 loaded.keyNormValues)) return false;
    for (uint32_t angle = 0; angle < loaded.descriptor.rotaryAngles; ++angle)
        loaded.ropeInverseFrequencies[angle] = pow(
            (double)loaded.descriptor.ropeTheta,
            -(double)(2 * angle) / loaded.descriptor.headDimension);
    *weights = loaded;
    return true;
}

static bool cacheSlot(const ANEFieldfareKVCache *cache, uint64_t position, uint64_t *slot) {
    if (!cache || !slot || !cache->capacity || position == UINT64_MAX) return false;
    if (cache->descriptor.kind == ANEFieldfareAttentionFull) {
        if (position >= cache->capacity) return false;
        *slot = position;
    } else {
        *slot = position % cache->capacity;
    }
    return true;
}

static void *pageAlignedZeroAllocation(size_t bytes) {
    size_t page = (size_t)getpagesize();
    if (!bytes || !page || bytes > SIZE_MAX - (page - 1)) return NULL;
    size_t rounded = (bytes + page - 1) / page * page;
    void *allocation = NULL;
    if (posix_memalign(&allocation, page, rounded) != 0) return NULL;
    memset(allocation, 0, rounded);
    return allocation;
}

bool ANEFieldfareKVCacheInitialize(ANEFieldfareKVCache *cache,
                                   ANEFieldfareAttentionDescriptor descriptor,
                                   uint64_t capacity) {
    if (!cache || !capacity || !descriptor.queryHeads || !descriptor.keyValueHeads ||
        descriptor.queryHeads % descriptor.keyValueHeads || !descriptor.headDimension ||
        (descriptor.kind == ANEFieldfareAttentionSliding && capacity < descriptor.slidingWindow)) return false;
    *cache = (ANEFieldfareKVCache){0};
    uint64_t width = (uint64_t)descriptor.keyValueHeads * descriptor.headDimension;
    uint64_t queryWidth = (uint64_t)descriptor.queryHeads * descriptor.headDimension;
    if (width > SIZE_MAX / sizeof(float) / capacity ||
        queryWidth > SIZE_MAX / sizeof(float)) return false;
    cache->positions = malloc((size_t)capacity * sizeof(*cache->positions));
    size_t cacheBytes = (size_t)(capacity * width) * sizeof(*cache->keys);
    cache->keys = pageAlignedZeroAllocation(cacheBytes);
    cache->values = pageAlignedZeroAllocation(cacheBytes);
    cache->scratchQuery = malloc((size_t)queryWidth * sizeof(*cache->scratchQuery));
    cache->scratchKey = malloc((size_t)width * sizeof(*cache->scratchKey));
    cache->scratchValue = malloc((size_t)width * sizeof(*cache->scratchValue));
    cache->scratchCombined = malloc((size_t)queryWidth * sizeof(*cache->scratchCombined));
    cache->scratchScores = malloc((size_t)capacity * sizeof(*cache->scratchScores));
    if (!cache->positions || !cache->keys || !cache->values ||
        !cache->scratchQuery || !cache->scratchKey || !cache->scratchValue ||
        !cache->scratchCombined || !cache->scratchScores) {
        ANEFieldfareKVCacheDestroy(cache);
        return false;
    }
    cache->descriptor = descriptor;
    cache->capacity = capacity;
    ANEFieldfareKVCacheClear(cache);
    return true;
}

void ANEFieldfareKVCacheDestroy(ANEFieldfareKVCache *cache) {
    if (!cache) return;
    free(cache->positions);
    free(cache->keys);
    free(cache->values);
    free(cache->scratchQuery);
    free(cache->scratchKey);
    free(cache->scratchValue);
    free(cache->scratchCombined);
    free(cache->scratchScores);
    *cache = (ANEFieldfareKVCache){0};
}

void ANEFieldfareKVCacheClear(ANEFieldfareKVCache *cache) {
    if (!cache || !cache->positions) return;
    for (uint64_t index = 0; index < cache->capacity; ++index) cache->positions[index] = UINT64_MAX;
}

bool ANEFieldfareKVCacheStore(ANEFieldfareKVCache *cache,
                              uint64_t position,
                              const float *key,
                              const float *value) {
    uint64_t slot = 0;
    if (!cache || !key || !value || !cacheSlot(cache, position, &slot)) return false;
    uint64_t width = (uint64_t)cache->descriptor.keyValueHeads * cache->descriptor.headDimension;
    memcpy(cache->keys + slot * width, key, (size_t)width * sizeof(float));
    memcpy(cache->values + slot * width, value, (size_t)width * sizeof(float));
    cache->positions[slot] = position;
    return true;
}

bool ANEFieldfareKVCacheLoad(const ANEFieldfareKVCache *cache,
                             uint64_t position,
                             float *key,
                             float *value) {
    uint64_t slot = 0;
    if (!cache || !cacheSlot(cache, position, &slot) || cache->positions[slot] != position) return false;
    uint64_t width = (uint64_t)cache->descriptor.keyValueHeads * cache->descriptor.headDimension;
    if (key) memcpy(key, cache->keys + slot * width, (size_t)width * sizeof(float));
    if (value) memcpy(value, cache->values + slot * width, (size_t)width * sizeof(float));
    return key || value;
}

bool ANEFieldfareAttentionApplyRoPE(float *head,
                                    uint32_t headDimension,
                                    uint32_t rotaryAngles,
                                    float theta,
                                    uint64_t position) {
    if (!head || !headDimension || (headDimension & 1) ||
        rotaryAngles > headDimension / 2 || !(theta > 0) || !isfinite(theta)) return false;
    uint32_t half = headDimension / 2;
    for (uint32_t angle = 0; angle < rotaryAngles; ++angle) {
        double inverseFrequency = pow((double)theta, -(double)(2 * angle) / headDimension);
        double phase = (double)position * inverseFrequency;
        float cosine = (float)cos(phase), sine = (float)sin(phase);
        float first = head[angle], second = head[half + angle];
        head[angle] = first * cosine - second * sine;
        head[half + angle] = second * cosine + first * sine;
    }
    return true;
}

static bool applyRoPEHeads(float *values,
                           uint32_t heads,
                           const ANEFieldfareAttentionWeights *weights,
                           uint64_t position) {
    const ANEFieldfareAttentionDescriptor *descriptor = &weights->descriptor;
    if (!values || !heads || !descriptor->headDimension ||
        (descriptor->headDimension & 1) ||
        descriptor->rotaryAngles > descriptor->headDimension / 2) return false;
    uint32_t half = descriptor->headDimension / 2;
    for (uint32_t angle = 0; angle < descriptor->rotaryAngles; ++angle) {
        double phase = (double)position * weights->ropeInverseFrequencies[angle];
        float cosine = (float)cos(phase), sine = (float)sin(phase);
        for (uint32_t head = 0; head < heads; ++head) {
            float *row = values + (uint64_t)head * descriptor->headDimension;
            float first = row[angle], second = row[half + angle];
            row[angle] = first * cosine - second * sine;
            row[half + angle] = second * cosine + first * sine;
        }
    }
    return true;
}

static bool normalizeHeads(float *values,
                           uint32_t heads,
                           uint32_t dimension,
                           float epsilon,
                           const float *scale) {
    if (!values || !heads || !dimension) return false;
    for (uint32_t head = 0; head < heads; ++head) {
        float *row = values + (uint64_t)head * dimension;
        double square = 0;
        for (uint32_t column = 0; column < dimension; ++column)
            square += (double)row[column] * row[column];
        float inverse = 1.0f / sqrtf((float)(square / dimension) + epsilon);
        if (!isfinite(inverse)) return false;
        for (uint32_t column = 0; column < dimension; ++column)
            row[column] *= inverse * (scale ? scale[column] : 1.0f);
    }
    return true;
}

static bool sameDescriptor(const ANEFieldfareAttentionDescriptor *left,
                           const ANEFieldfareAttentionDescriptor *right) {
    return left->layer == right->layer && left->kind == right->kind &&
        left->hiddenSize == right->hiddenSize && left->queryHeads == right->queryHeads &&
        left->keyValueHeads == right->keyValueHeads &&
        left->headDimension == right->headDimension &&
        left->rotaryAngles == right->rotaryAngles &&
        left->slidingWindow == right->slidingWindow && left->ropeTheta == right->ropeTheta &&
        left->rmsEpsilon == right->rmsEpsilon &&
        left->attentionScale == right->attentionScale &&
        left->attentionSoftcap == right->attentionSoftcap &&
        left->valueEqualsKeyProjection == right->valueEqualsKeyProjection &&
        left->sharesKeyValueAcrossLayers == right->sharesKeyValueAcrossLayers;
}

bool ANEFieldfareAttentionDecodeDispatched(
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t position,
    const float *normalizedHidden,
    const uint8_t *tensorIndexBase,
    ANEFieldfareAttentionInputDispatch inputDispatch,
    ANEFieldfareAttentionAffineDispatch affineDispatch,
    void *affineContext,
    float *output) {
    if (!weights || !cache || !normalizedHidden || !output ||
        !sameDescriptor(&weights->descriptor, &cache->descriptor)) return false;
    const ANEFieldfareAttentionDescriptor *descriptor = &weights->descriptor;
    uint32_t queryWidth = descriptor->queryHeads * descriptor->headDimension;
    float *query = cache->scratchQuery;
    float *key = cache->scratchKey;
    float *value = cache->scratchValue;
    float *combined = cache->scratchCombined;
    uint64_t first = descriptor->kind == ANEFieldfareAttentionSliding &&
                     position + 1 > descriptor->slidingWindow
                         ? position + 1 - descriptor->slidingWindow : 0;
    uint64_t count = position - first + 1;
    float *scores = count <= cache->capacity ? cache->scratchScores : NULL;
    if (combined) memset(combined, 0, (size_t)queryWidth * sizeof(*combined));
    bool valid = query && key && value && combined && scores;
    if (valid && inputDispatch) {
        valid = inputDispatch(affineContext, tensorIndexBase,
                              &weights->queryProjection, &weights->keyProjection,
                              &weights->valueProjection,
                              weights->descriptor.valueEqualsKeyProjection,
                              normalizedHidden, query, key, value);
    } else if (valid && affineDispatch) {
        valid = affineDispatch(affineContext, tensorIndexBase,
                               &weights->queryProjection, normalizedHidden, query) &&
            affineDispatch(affineContext, tensorIndexBase,
                           &weights->keyProjection, normalizedHidden, key);
        if (valid && weights->descriptor.valueEqualsKeyProjection)
            memcpy(value, key, (size_t)weights->keyProjection.rows * sizeof(float));
        else if (valid)
            valid = affineDispatch(affineContext, tensorIndexBase,
                                   &weights->valueProjection, normalizedHidden, value);
    } else if (valid) {
        valid = projectAttentionInput(weights, normalizedHidden, query, key, value);
    }
    valid = valid &&
        normalizeHeads(query, descriptor->queryHeads, descriptor->headDimension,
                       descriptor->rmsEpsilon, weights->queryNormValues) &&
        normalizeHeads(key, descriptor->keyValueHeads, descriptor->headDimension,
                       descriptor->rmsEpsilon, weights->keyNormValues) &&
        normalizeHeads(value, descriptor->keyValueHeads, descriptor->headDimension,
                       descriptor->rmsEpsilon, NULL);
    valid = valid &&
        applyRoPEHeads(query, descriptor->queryHeads, weights, position) &&
        applyRoPEHeads(key, descriptor->keyValueHeads, weights, position);
    valid = valid && ANEFieldfareKVCacheStore(cache, position, key, value);

    uint32_t groups = descriptor->queryHeads / descriptor->keyValueHeads;
    for (uint32_t queryHead = 0; valid && queryHead < descriptor->queryHeads; ++queryHead) {
        uint32_t keyValueHead = queryHead / groups;
        const float *queryRow = query + (uint64_t)queryHead * descriptor->headDimension;
        float maximum = -FLT_MAX;
        for (uint64_t offset = 0; offset < count; ++offset) {
            uint64_t absolute = first + offset, slot = 0;
            if (!cacheSlot(cache, absolute, &slot) || cache->positions[slot] != absolute) {
                valid = false;
                break;
            }
            const float *keyRow = cache->keys +
                (slot * descriptor->keyValueHeads + keyValueHead) * descriptor->headDimension;
            double dot = 0;
            for (uint32_t column = 0; column < descriptor->headDimension; ++column)
                dot += (double)queryRow[column] * keyRow[column];
            float logit = (float)dot * descriptor->attentionScale;
            if (descriptor->attentionSoftcap > 0)
                logit = tanhf(logit / descriptor->attentionSoftcap) * descriptor->attentionSoftcap;
            scores[offset] = logit;
            maximum = fmaxf(maximum, scores[offset]);
        }
        double denominator = 0;
        for (uint64_t offset = 0; valid && offset < count; ++offset)
            denominator += exp((double)scores[offset] - maximum);
        valid = valid && isfinite(denominator) && denominator > 0;
        float *combinedRow = combined + (uint64_t)queryHead * descriptor->headDimension;
        for (uint64_t offset = 0; valid && offset < count; ++offset) {
            uint64_t absolute = first + offset, slot = 0;
            valid = cacheSlot(cache, absolute, &slot) && cache->positions[slot] == absolute;
            if (!valid) break;
            float probability = (float)(exp((double)scores[offset] - maximum) / denominator);
            const float *valueRow = cache->values +
                (slot * descriptor->keyValueHeads + keyValueHead) * descriptor->headDimension;
            for (uint32_t column = 0; column < descriptor->headDimension; ++column)
                combinedRow[column] += probability * valueRow[column];
        }
    }
    if (valid)
        valid = affineDispatch
            ? affineDispatch(affineContext, tensorIndexBase,
                             &weights->outputProjection, combined, output)
            : attentionAffineMatVec(&weights->outputProjection, combined, output);
    return valid;
}

bool ANEFieldfareAttentionDecode(const ANEFieldfareAttentionWeights *weights,
                                 ANEFieldfareKVCache *cache,
                                 uint64_t position,
                                 const float *normalizedHidden,
                                 float *output) {
    return ANEFieldfareAttentionDecodeDispatched(
        weights, cache, position, normalizedHidden, NULL, NULL, NULL, NULL, output);
}

static bool attendProjectedRow(const ANEFieldfareAttentionWeights *weights,
                               ANEFieldfareKVCache *cache,
                               uint64_t position,
                               float *query,
                               float *key,
                               float *value,
                               float *combined) {
    const ANEFieldfareAttentionDescriptor *descriptor = &weights->descriptor;
    uint32_t queryWidth = descriptor->queryHeads * descriptor->headDimension;
    uint64_t first = descriptor->kind == ANEFieldfareAttentionSliding &&
                     position + 1 > descriptor->slidingWindow
                         ? position + 1 - descriptor->slidingWindow : 0;
    uint64_t count = position - first + 1;
    float *scores = count <= cache->capacity ? cache->scratchScores : NULL;
    if (!scores) return false;
    memset(combined, 0, (size_t)queryWidth * sizeof(*combined));
    bool valid =
        normalizeHeads(query, descriptor->queryHeads, descriptor->headDimension,
                       descriptor->rmsEpsilon, weights->queryNormValues) &&
        normalizeHeads(key, descriptor->keyValueHeads, descriptor->headDimension,
                       descriptor->rmsEpsilon, weights->keyNormValues) &&
        normalizeHeads(value, descriptor->keyValueHeads, descriptor->headDimension,
                       descriptor->rmsEpsilon, NULL) &&
        applyRoPEHeads(query, descriptor->queryHeads, weights, position) &&
        applyRoPEHeads(key, descriptor->keyValueHeads, weights, position) &&
        ANEFieldfareKVCacheStore(cache, position, key, value);
    uint32_t groups = descriptor->queryHeads / descriptor->keyValueHeads;
    for (uint32_t queryHead = 0; valid && queryHead < descriptor->queryHeads; ++queryHead) {
        uint32_t keyValueHead = queryHead / groups;
        const float *queryRow = query + (uint64_t)queryHead * descriptor->headDimension;
        float maximum = -FLT_MAX;
        for (uint64_t offset = 0; offset < count; ++offset) {
            uint64_t absolute = first + offset, slot = 0;
            if (!cacheSlot(cache, absolute, &slot) || cache->positions[slot] != absolute) {
                valid = false;
                break;
            }
            const float *keyRow = cache->keys +
                (slot * descriptor->keyValueHeads + keyValueHead) * descriptor->headDimension;
            double dot = 0;
            for (uint32_t column = 0; column < descriptor->headDimension; ++column)
                dot += (double)queryRow[column] * keyRow[column];
            float logit = (float)dot * descriptor->attentionScale;
            if (descriptor->attentionSoftcap > 0)
                logit = tanhf(logit / descriptor->attentionSoftcap) * descriptor->attentionSoftcap;
            scores[offset] = logit;
            maximum = fmaxf(maximum, logit);
        }
        double denominator = 0;
        for (uint64_t offset = 0; valid && offset < count; ++offset)
            denominator += exp((double)scores[offset] - maximum);
        valid = valid && isfinite(denominator) && denominator > 0;
        float *combinedRow = combined + (uint64_t)queryHead * descriptor->headDimension;
        for (uint64_t offset = 0; valid && offset < count; ++offset) {
            uint64_t absolute = first + offset, slot = 0;
            valid = cacheSlot(cache, absolute, &slot) && cache->positions[slot] == absolute;
            if (!valid) break;
            float probability = (float)(exp((double)scores[offset] - maximum) / denominator);
            const float *valueRow = cache->values +
                (slot * descriptor->keyValueHeads + keyValueHead) * descriptor->headDimension;
            for (uint32_t column = 0; column < descriptor->headDimension; ++column)
                combinedRow[column] += probability * valueRow[column];
        }
    }
    return valid;
}

bool ANEFieldfareAttentionPrefillDispatched(
    const ANEFieldfareAttentionWeights *weights,
    ANEFieldfareKVCache *cache,
    uint64_t firstPosition,
    uint32_t positionCount,
    const float *normalizedHiddenRows,
    const uint8_t *tensorIndexBase,
    ANEFieldfareAttentionInputPositionDispatch inputDispatch,
    ANEFieldfareAttentionAffinePositionDispatch affineDispatch,
    void *affineContext,
    float *outputRows) {
    if (!weights || !cache || !normalizedHiddenRows || !outputRows || !positionCount ||
        positionCount > 64 || firstPosition > UINT64_MAX - (positionCount - 1) ||
        !sameDescriptor(&weights->descriptor, &cache->descriptor)) return false;
    const ANEFieldfareAttentionDescriptor *descriptor = &weights->descriptor;
    uint32_t queryWidth = descriptor->queryHeads * descriptor->headDimension;
    uint32_t keyWidth = descriptor->keyValueHeads * descriptor->headDimension;
    float *query = malloc((size_t)positionCount * queryWidth * sizeof(float));
    float *key = malloc((size_t)positionCount * keyWidth * sizeof(float));
    float *value = malloc((size_t)positionCount * keyWidth * sizeof(float));
    float *combined = malloc((size_t)positionCount * queryWidth * sizeof(float));
    bool valid = query && key && value && combined;
    if (valid && inputDispatch) {
        valid = inputDispatch(affineContext, tensorIndexBase,
                              &weights->queryProjection, &weights->keyProjection,
                              &weights->valueProjection,
                              descriptor->valueEqualsKeyProjection, positionCount,
                              normalizedHiddenRows, query, key, value);
    } else {
        for (uint32_t row = 0; valid && row < positionCount; ++row)
            valid = projectAttentionInput(
                weights, normalizedHiddenRows + (uint64_t)row * descriptor->hiddenSize,
                query + (uint64_t)row * queryWidth,
                key + (uint64_t)row * keyWidth,
                value + (uint64_t)row * keyWidth);
    }
    for (uint32_t row = 0; valid && row < positionCount; ++row)
        valid = attendProjectedRow(
            weights, cache, firstPosition + row,
            query + (uint64_t)row * queryWidth,
            key + (uint64_t)row * keyWidth,
            value + (uint64_t)row * keyWidth,
            combined + (uint64_t)row * queryWidth);
    if (valid && affineDispatch) {
        valid = affineDispatch(affineContext, tensorIndexBase,
                               &weights->outputProjection, positionCount,
                               combined, outputRows);
    } else {
        for (uint32_t row = 0; valid && row < positionCount; ++row)
            valid = attentionAffineMatVec(
                &weights->outputProjection,
                combined + (uint64_t)row * queryWidth,
                outputRows + (uint64_t)row * descriptor->hiddenSize);
    }
    free(query); free(key); free(value); free(combined);
    return valid;
}
