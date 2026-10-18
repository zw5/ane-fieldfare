#import <Foundation/Foundation.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../ANEFieldfareRuntime/Attention.h"

enum { Hidden = 2816, GroupSize = 64 };

static void emit(NSDictionary *value) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static NSDictionary *shape(ANEFieldfareAttentionDescriptor descriptor) {
    return @{
        @"layer": @(descriptor.layer),
        @"kind": descriptor.kind == ANEFieldfareAttentionFull ? @"full" : @"sliding_1024",
        @"query_heads": @(descriptor.queryHeads),
        @"key_value_heads": @(descriptor.keyValueHeads),
        @"head_dimension": @(descriptor.headDimension),
        @"rotary_angles_per_half": @(descriptor.rotaryAngles),
        @"attention_scale": @(descriptor.attentionScale),
        @"attention_softcap": @(descriptor.attentionSoftcap),
        @"value_equals_key_projection": @(descriptor.valueEqualsKeyProjection),
        @"shares_key_value_across_layers": @(descriptor.sharesKeyValueAcrossLayers),
    };
}

static BOOL rawBF16Vector(ANEFieldfareTensor tensor, NSUInteger count, float *output) {
    if (!output || tensor.weightBytes != count * 2 || tensor.scaleBytes || tensor.biasBytes) return NO;
    for (NSUInteger index = 0; index < count; ++index)
        output[index] = ANEFieldfareBF16(tensor.weights + index * 2);
    return YES;
}

static BOOL normalizeOracle(float *values,
                            NSUInteger heads,
                            NSUInteger dimension,
                            float epsilon,
                            const float *scale) {
    for (NSUInteger head = 0; head < heads; ++head) {
        float *row = values + head * dimension;
        long double square = 0;
        for (NSUInteger column = 0; column < dimension; ++column)
            square += (long double)row[column] * row[column];
        float inverse = 1.0f / sqrtf((float)(square / dimension) + epsilon);
        if (!isfinite(inverse)) return NO;
        for (NSUInteger column = 0; column < dimension; ++column)
            row[column] *= inverse * (scale ? scale[column] : 1.0f);
    }
    return YES;
}

// Independent transcription of the official proportional/default RoPE path:
// construct the padded half-frequency vector, concatenate it with itself,
// then apply x*cos + rotate_half(x)*sin. This intentionally does not call the
// runtime's pairwise in-place implementation.
static BOOL ropeOracle(const float *input,
                       float *output,
                       NSUInteger dimension,
                       NSUInteger activeAngles,
                       double theta,
                       uint64_t position) {
    if (!input || !output || !dimension || dimension % 2 || activeAngles > dimension / 2) return NO;
    NSUInteger half = dimension / 2;
    for (NSUInteger coordinate = 0; coordinate < dimension; ++coordinate) {
        NSUInteger frequency = coordinate % half;
        double inverse = frequency < activeAngles
            ? pow(theta, -(double)(2 * frequency) / dimension) : 0.0;
        double phase = position * inverse;
        float rotatedHalf = coordinate < half ? -input[coordinate + half] : input[coordinate - half];
        output[coordinate] = input[coordinate] * (float)cos(phase) + rotatedHalf * (float)sin(phase);
    }
    return YES;
}

static BOOL applyRoPEOracle(float *values,
                            NSUInteger heads,
                            ANEFieldfareAttentionDescriptor descriptor,
                            uint64_t position) {
    float *temporary = malloc(descriptor.headDimension * sizeof(float));
    if (!temporary) return NO;
    BOOL valid = YES;
    for (NSUInteger head = 0; valid && head < heads; ++head) {
        float *row = values + head * descriptor.headDimension;
        valid = ropeOracle(row, temporary, descriptor.headDimension, descriptor.rotaryAngles,
                           descriptor.ropeTheta, position);
        if (valid) memcpy(row, temporary, descriptor.headDimension * sizeof(float));
    }
    free(temporary);
    return valid;
}

static BOOL independentTwoPositionOracle(const ANEFieldfareAttentionWeights *weights,
                                         const float *firstHidden,
                                         const float *secondHidden,
                                         const float *runtimeOutput,
                                         double *maximumError,
                                         double *projectionOrientationError) {
    ANEFieldfareAttentionDescriptor descriptor = weights->descriptor;
    NSUInteger queryWidth = descriptor.queryHeads * descriptor.headDimension;
    NSUInteger keyValueWidth = descriptor.keyValueHeads * descriptor.headDimension;
    float *query = calloc(queryWidth, sizeof(float));
    float *firstKey = calloc(keyValueWidth, sizeof(float));
    float *secondKey = calloc(keyValueWidth, sizeof(float));
    float *firstValue = calloc(keyValueWidth, sizeof(float));
    float *secondValue = calloc(keyValueWidth, sizeof(float));
    float *queryScale = calloc(descriptor.headDimension, sizeof(float));
    float *keyScale = calloc(descriptor.headDimension, sizeof(float));
    float *combined = calloc(queryWidth, sizeof(float));
    float *expected = calloc(Hidden, sizeof(float));
    float *projectionRow = calloc(descriptor.hiddenSize, sizeof(float));
    BOOL valid = query && firstKey && secondKey && firstValue && secondValue && queryScale &&
        keyScale && combined && expected && projectionRow &&
        ANEFieldfareAffineMatVec(&weights->queryProjection, 4, GroupSize, secondHidden, query) &&
        ANEFieldfareAffineMatVec(&weights->keyProjection, 4, GroupSize, firstHidden, firstKey) &&
        ANEFieldfareAffineMatVec(&weights->keyProjection, 4, GroupSize, secondHidden, secondKey) &&
        rawBF16Vector(weights->queryNorm, descriptor.headDimension, queryScale) &&
        rawBF16Vector(weights->keyNorm, descriptor.headDimension, keyScale);
    NSUInteger selectedRow = queryWidth - 1;
    float orientationExpected = 0;
    if (valid) {
        valid = ANEFieldfareAffineRow(&weights->queryProjection, 4, GroupSize,
                                      (uint32_t)selectedRow, projectionRow);
        for (NSUInteger column = 0; valid && column < descriptor.hiddenSize; ++column)
            orientationExpected += secondHidden[column] * projectionRow[column];
    }
    double orientationError = valid ? fabs((double)query[selectedRow] - orientationExpected) : INFINITY;
    if (projectionOrientationError) *projectionOrientationError = orientationError;
    valid = valid && orientationError == 0;
    if (valid && descriptor.valueEqualsKeyProjection) {
        memcpy(firstValue, firstKey, keyValueWidth * sizeof(float));
        memcpy(secondValue, secondKey, keyValueWidth * sizeof(float));
    } else if (valid) {
        valid = ANEFieldfareAffineMatVec(&weights->valueProjection, 4, GroupSize,
                                         firstHidden, firstValue) &&
            ANEFieldfareAffineMatVec(&weights->valueProjection, 4, GroupSize,
                                     secondHidden, secondValue);
    }
    valid = valid &&
        normalizeOracle(query, descriptor.queryHeads, descriptor.headDimension,
                        descriptor.rmsEpsilon, queryScale) &&
        normalizeOracle(firstKey, descriptor.keyValueHeads, descriptor.headDimension,
                        descriptor.rmsEpsilon, keyScale) &&
        normalizeOracle(secondKey, descriptor.keyValueHeads, descriptor.headDimension,
                        descriptor.rmsEpsilon, keyScale) &&
        normalizeOracle(firstValue, descriptor.keyValueHeads, descriptor.headDimension,
                        descriptor.rmsEpsilon, NULL) &&
        normalizeOracle(secondValue, descriptor.keyValueHeads, descriptor.headDimension,
                        descriptor.rmsEpsilon, NULL) &&
        applyRoPEOracle(query, descriptor.queryHeads, descriptor, 1) &&
        applyRoPEOracle(firstKey, descriptor.keyValueHeads, descriptor, 0) &&
        applyRoPEOracle(secondKey, descriptor.keyValueHeads, descriptor, 1);

    NSUInteger groups = descriptor.queryHeads / descriptor.keyValueHeads;
    for (NSUInteger queryHead = 0; valid && queryHead < descriptor.queryHeads; ++queryHead) {
        NSUInteger keyValueHead = queryHead / groups;
        const float *queryRow = query + queryHead * descriptor.headDimension;
        const float *firstKeyRow = firstKey + keyValueHead * descriptor.headDimension;
        const float *secondKeyRow = secondKey + keyValueHead * descriptor.headDimension;
        long double firstDot = 0, secondDot = 0;
        for (NSUInteger column = 0; column < descriptor.headDimension; ++column) {
            firstDot += (long double)queryRow[column] * firstKeyRow[column];
            secondDot += (long double)queryRow[column] * secondKeyRow[column];
        }
        double firstLogit = (double)firstDot * descriptor.attentionScale;
        double secondLogit = (double)secondDot * descriptor.attentionScale;
        if (descriptor.attentionSoftcap > 0) {
            firstLogit = tanh(firstLogit / descriptor.attentionSoftcap) * descriptor.attentionSoftcap;
            secondLogit = tanh(secondLogit / descriptor.attentionSoftcap) * descriptor.attentionSoftcap;
        }
        double maximum = fmax(firstLogit, secondLogit);
        double firstExp = exp(firstLogit - maximum), secondExp = exp(secondLogit - maximum);
        float firstProbability = (float)(firstExp / (firstExp + secondExp));
        float secondProbability = 1.0f - firstProbability;
        const float *firstValueRow = firstValue + keyValueHead * descriptor.headDimension;
        const float *secondValueRow = secondValue + keyValueHead * descriptor.headDimension;
        float *combinedRow = combined + queryHead * descriptor.headDimension;
        for (NSUInteger column = 0; column < descriptor.headDimension; ++column)
            combinedRow[column] = firstProbability * firstValueRow[column] +
                                  secondProbability * secondValueRow[column];
    }
    valid = valid && ANEFieldfareAffineMatVec(&weights->outputProjection, 4, GroupSize, combined, expected);
    double error = 0;
    for (NSUInteger column = 0; valid && column < Hidden; ++column)
        error = fmax(error, fabs((double)runtimeOutput[column] - expected[column]));
    if (maximumError) *maximumError = error;
    // The oracle deliberately uses long-double dot products and a separately
    // constructed RoPE, so exact equality is neither required nor expected.
    valid = valid && error < 2e-4;
    free(query);
    free(firstKey);
    free(secondKey);
    free(firstValue);
    free(secondValue);
    free(queryScale);
    free(keyScale);
    free(combined);
    free(expected);
    free(projectionRow);
    return valid;
}

static BOOL runLayer(const ANEFieldfareTensorIndex *index,
                     uint32_t layer,
                     const float *firstHidden,
                     const float *secondHidden,
                     NSDictionary **receipt) {
    ANEFieldfareAttentionWeights weights = {0};
    if (!ANEFieldfareAttentionWeightsLoad(index, layer, &weights)) return NO;
    uint64_t capacity = weights.descriptor.kind == ANEFieldfareAttentionSliding ? 1024 : 2;
    ANEFieldfareKVCache cache = {0};
    float *firstOutput = calloc(Hidden, sizeof(float));
    float *secondOutput = calloc(Hidden, sizeof(float));
    float *firstRepeat = calloc(Hidden, sizeof(float));
    float *secondRepeat = calloc(Hidden, sizeof(float));
    BOOL valid = firstOutput && secondOutput && firstRepeat && secondRepeat &&
        ANEFieldfareKVCacheInitialize(&cache, weights.descriptor, capacity) &&
        ANEFieldfareAttentionDecode(&weights, &cache, 0, firstHidden, firstOutput) &&
        ANEFieldfareAttentionDecode(&weights, &cache, 1, secondHidden, secondOutput);
    double square = 0, maximum = 0;
    for (NSUInteger column = 0; valid && column < Hidden; ++column) {
        valid = isfinite(secondOutput[column]);
        square += (double)secondOutput[column] * secondOutput[column];
        maximum = fmax(maximum, fabs(secondOutput[column]));
    }
    double oracleError = INFINITY, projectionOrientationError = INFINITY;
    valid = valid && independentTwoPositionOracle(&weights, firstHidden, secondHidden,
                                                  secondOutput, &oracleError,
                                                  &projectionOrientationError);
    ANEFieldfareKVCacheClear(&cache);
    valid = valid && ANEFieldfareAttentionDecode(&weights, &cache, 0, firstHidden, firstRepeat) &&
        ANEFieldfareAttentionDecode(&weights, &cache, 1, secondHidden, secondRepeat) &&
        memcmp(firstOutput, firstRepeat, Hidden * sizeof(float)) == 0 &&
        memcmp(secondOutput, secondRepeat, Hidden * sizeof(float)) == 0;
    if (valid) {
        *receipt = @{
            @"shape": shape(weights.descriptor),
            @"output_rms": @(sqrt(square / Hidden)),
            @"output_abs_max": @(maximum),
            @"causal_positions": @2,
            @"deterministic_replay_exact": @YES,
            @"independent_two_position_max_abs_error": @(oracleError),
            @"independent_two_position_oracle": @YES,
            @"projection_orientation_max_abs_error": @(projectionOrientationError),
            @"projection_orientation_row_is_output": @YES,
        };
    }
    ANEFieldfareKVCacheDestroy(&cache);
    free(firstOutput);
    free(secondOutput);
    free(firstRepeat);
    free(secondRepeat);
    return valid;
}

static BOOL ropeControl(double *maximumError) {
    float values[512], original[512], expected[512];
    for (NSUInteger index = 0; index < 512; ++index)
        values[index] = original[index] = (float)index / 257.0f - 0.5f;
    if (!ropeOracle(original, expected, 512, 64, 1000000.0, 7) ||
        !ANEFieldfareAttentionApplyRoPE(values, 512, 64, 1000000.0f, 7)) return NO;
    double error = 0;
    for (NSUInteger index = 0; index < 512; ++index)
        error = fmax(error, fabs((double)values[index] - expected[index]));
    if (maximumError) *maximumError = error;
    return error < 2e-7;
}

static BOOL slidingCacheBoundaryControl(void) {
    ANEFieldfareAttentionDescriptor descriptor = {0};
    if (!ANEFieldfareAttentionDescriptorForLayer(0, &descriptor)) return NO;
    ANEFieldfareKVCache cache = {0};
    NSUInteger width = descriptor.keyValueHeads * descriptor.headDimension;
    float *key = calloc(width, sizeof(float)), *value = calloc(width, sizeof(float));
    BOOL valid = key && value && ANEFieldfareKVCacheInitialize(&cache, descriptor, 1024);
    for (uint64_t position = 0; valid && position <= 1024; ++position) {
        key[0] = (float)position;
        value[0] = -(float)position;
        valid = ANEFieldfareKVCacheStore(&cache, position, key, value);
    }
    valid = valid && !ANEFieldfareKVCacheLoad(&cache, 0, key, value) &&
        ANEFieldfareKVCacheLoad(&cache, 1, key, value) && key[0] == 1.0f && value[0] == -1.0f &&
        ANEFieldfareKVCacheLoad(&cache, 1024, key, value) && key[0] == 1024.0f && value[0] == -1024.0f;
    ANEFieldfareKVCacheDestroy(&cache);
    free(key);
    free(value);
    return valid;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            emit(@{@"schema": @"ane-fieldfare.attention-gate.v1", @"status": @"rejected",
                   @"stage": @"arguments", @"usage": @"ane-fieldfare-attention-gate MODEL_DIR"});
            return 2;
        }
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *path = [root stringByAppendingPathComponent:@"model_weights.bin"];
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor embedding = {0};
        float *firstHidden = calloc(Hidden, sizeof(float));
        float *secondHidden = calloc(Hidden, sizeof(float));
        double ropeError = INFINITY;
        BOOL valid = firstHidden && secondHidden &&
            ANEFieldfareTensorIndexOpen(path.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.embed_tokens.weight", &embedding) &&
            embedding.rows == 262144 && embedding.columns == Hidden &&
            ANEFieldfareAffineRow(&embedding, 4, GroupSize, 2, firstHidden) &&
            ANEFieldfareAffineRow(&embedding, 4, GroupSize, 3, secondHidden) &&
            ropeControl(&ropeError) && slidingCacheBoundaryControl();
        NSDictionary *sliding = nil, *full = nil;
        valid = valid && runLayer(&index, 0, firstHidden, secondHidden, &sliding) &&
            runLayer(&index, 5, firstHidden, secondHidden, &full);
        emit(@{
            @"schema": @"ane-fieldfare.attention-gate.v1",
            @"status": valid ? @"validated" : @"rejected",
            @"stage": valid ? @"independent_numeric_control" : @"attention_contract",
            @"sliding_layer": sliding ?: @{},
            @"full_layer": full ?: @{},
            @"rope_control": @{ @"full_rotated_coordinates": @128,
                                @"full_unrotated_coordinates": @384,
                                @"official_padded_frequency_oracle_max_abs_error": @(ropeError) },
            @"sliding_cache_boundary": @{ @"window": @1024,
                                           @"position_0_evicted_after_1024": @(valid) },
            @"cache": @"float-owned, explicit layer and absolute-position API; tagged 1024-slot ring for sliding layers",
            @"claim_boundary": @"Authentic quantized checkpoint Q/K/V/O projections, per-head Q/K RMS norms, unscaled V RMS norm, proportional partial RoPE, grouped-query causal attention, and both Gemma 4 attention geometries execute deterministically on the CPU control path. A separately structured two-position numerical oracle checks both layer geometries, and an official-form padded-frequency oracle checks proportional RoPE. This validates the CPU attention construction, not ANE attention execution or a complete decoder layer.",
        });
        ANEFieldfareTensorIndexClose(&index);
        free(firstHidden);
        free(secondHidden);
        return valid ? 0 : 1;
    }
}
