#import <Foundation/Foundation.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../ANEFieldfareRuntime/Attention.h"
#include "../ANEFieldfareRuntime/MetalAttention.h"

typedef struct {
    const float *query;
    const float *key;
    const float *value;
    uint32_t count;
    uint32_t queryWidth;
    uint32_t keyWidth;
} ProjectionFixture;

static bool provideQKV(void *raw, const uint8_t *base,
                       const ANEFieldfareTensor *q, const ANEFieldfareTensor *k,
                       const ANEFieldfareTensor *v, bool same,
                       uint32_t count, const float *inputs,
                       float *qo, float *ko, float *vo) {
    (void)base; (void)q; (void)k; (void)v; (void)same; (void)inputs;
    ProjectionFixture *f = raw;
    if (!f || count != f->count) return false;
    memcpy(qo, f->query, (size_t)count * f->queryWidth * sizeof(float));
    memcpy(ko, f->key, (size_t)count * f->keyWidth * sizeof(float));
    memcpy(vo, f->value, (size_t)count * f->keyWidth * sizeof(float));
    return true;
}

static bool copyCombined(void *raw, const uint8_t *base,
                         const ANEFieldfareTensor *matrix, uint32_t count,
                         const float *inputs, float *outputs) {
    (void)base; (void)matrix;
    ProjectionFixture *f = raw;
    if (!f || count != f->count) return false;
    memcpy(outputs, inputs, (size_t)count * f->queryWidth * sizeof(float));
    return true;
}

static void fillInputs(float *query, float *key, float *value,
                       uint32_t count, uint32_t queryWidth, uint32_t keyWidth,
                       uint32_t seed) {
    for (uint64_t i = 0; i < (uint64_t)count * queryWidth; ++i)
        query[i] = 0.17f * sinf((float)(i + 1 + seed) * 0.013f) +
                   0.03f * cosf((float)(i + 7) * 0.007f);
    for (uint64_t i = 0; i < (uint64_t)count * keyWidth; ++i) {
        key[i] = 0.19f * sinf((float)(i + 3 + seed) * 0.011f);
        value[i] = 0.23f * cosf((float)(i + 5 + seed) * 0.009f);
    }
}

static NSDictionary *compare(const float *a, const float *b, uint64_t count) {
    double maximum = 0.0, mean = 0.0;
    uint64_t failures = 0;
    for (uint64_t i = 0; i < count; ++i) {
        double delta = fabs((double)a[i] - b[i]);
        maximum = fmax(maximum, delta); mean += delta;
        if (!isfinite(a[i]) || !isfinite(b[i]) || delta > 0.003) failures++;
    }
    return @{ @"values": @(count), @"max_absolute_error": @(maximum),
              @"mean_absolute_error": @(mean / count),
              @"tolerance_failures": @(failures), @"valid": @(failures == 0) };
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        const char *error = NULL;
        ANEFieldfareMetalAttention *metal = ANEFieldfareMetalAttentionCreate(argv[1], &error);
        ANEFieldfareAttentionWeights weights = {0};
        ANEFieldfareAttentionDescriptorForLayer(0, &weights.descriptor);
        uint32_t qw = weights.descriptor.queryHeads * weights.descriptor.headDimension;
        uint32_t kw = weights.descriptor.keyValueHeads * weights.descriptor.headDimension;
        for (uint32_t i = 0; i < weights.descriptor.headDimension; ++i) {
            weights.queryNormValues[i] = 0.91f + 0.0002f * i;
            weights.keyNormValues[i] = 0.87f + 0.0003f * i;
        }
        for (uint32_t i = 0; i < weights.descriptor.rotaryAngles; ++i)
            weights.ropeInverseFrequencies[i] = pow(
                weights.descriptor.ropeTheta,
                -(double)(2 * i) / weights.descriptor.headDimension);
        ANEFieldfareKVCache cpu = {0}, gpu = {0};
        bool valid = metal &&
            ANEFieldfareKVCacheInitialize(&cpu, weights.descriptor, 1024) &&
            ANEFieldfareKVCacheInitialize(&gpu, weights.descriptor, 1024);
        NSMutableArray *tiles = [NSMutableArray array];
        const uint32_t counts[2] = {64, 32};
        uint64_t first = 0;
        for (uint32_t tile = 0; valid && tile < 2; ++tile) {
            uint32_t count = counts[tile];
            float *q = malloc((size_t)count * qw * sizeof(float));
            float *k = malloc((size_t)count * kw * sizeof(float));
            float *v = malloc((size_t)count * kw * sizeof(float));
            float *dummy = calloc((size_t)count * weights.descriptor.hiddenSize, sizeof(float));
            float *cpuOut = malloc((size_t)count * qw * sizeof(float));
            float *gpuOut = malloc((size_t)count * qw * sizeof(float));
            valid = q && k && v && dummy && cpuOut && gpuOut;
            if (valid) {
                fillInputs(q, k, v, count, qw, kw, tile * 101);
                ProjectionFixture fixture = {q, k, v, count, qw, kw};
                valid = ANEFieldfareAttentionPrefillDispatched(
                    &weights, &cpu, first, count, dummy, (const uint8_t *)1,
                    provideQKV, copyCombined, &fixture, cpuOut) &&
                    ANEFieldfareMetalAttentionCall(
                        metal, &weights, &gpu, first, count, q, k, v, gpuOut);
                NSDictionary *output = valid ? compare(cpuOut, gpuOut, (uint64_t)count * qw) : @{};
                NSDictionary *keys = valid ? compare(cpu.keys, gpu.keys,
                    (first + count) * kw) : @{};
                NSDictionary *values = valid ? compare(cpu.values, gpu.values,
                    (first + count) * kw) : @{};
                valid = valid && [output[@"valid"] boolValue] &&
                    [keys[@"valid"] boolValue] && [values[@"valid"] boolValue];
                [tiles addObject:@{ @"first_position": @(first), @"positions": @(count),
                    @"combined_oracle": output, @"key_cache_oracle": keys,
                    @"value_cache_oracle": values, @"valid": @(valid) }];
            }
            free(q); free(k); free(v); free(dummy); free(cpuOut); free(gpuOut);
            first += count;
        }
        ANEFieldfareMetalAttentionInfo info = ANEFieldfareMetalAttentionGetInfo(metal);
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.metal-attention.v1",
            @"status": valid ? @"validated" : @"rejected",
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"shape": @{ @"layer": @0, @"query_heads": @16, @"kv_heads": @8,
                         @"head_dimension": @256, @"capacity": @1024 },
            @"tiles": tiles,
            @"metal": @{ @"dispatches": @(info.dispatchCount),
                @"positions": @(info.positionCount), @"host_waits": @(info.hostWaitCount),
                @"wall_ms": @(info.wallMilliseconds), @"gpu_ms": @(info.gpuMilliseconds) },
            @"error": error ? @(error) : @(ANEFieldfareMetalAttentionLastError(metal)),
            @"claim_boundary": valid
                ? @"Metal performed authentic layer-0 head normalization, RoPE, causal sliding KV mutation, softmax, and value accumulation for a 64-position tile followed by a 32-position tile. Combined heads and complete populated K/V cache regions pass the independent CPU oracle. This does not yet validate ANE event wiring or routed-layer integration."
                : @"The Metal attention construction or CPU oracle failed; no placement claim is made."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
            options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        free((void *)error);
        ANEFieldfareKVCacheDestroy(&cpu); ANEFieldfareKVCacheDestroy(&gpu);
        ANEFieldfareMetalAttentionDestroy(metal);
        return valid ? 0 : 1;
    }
}
