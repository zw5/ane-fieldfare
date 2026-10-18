#import <Foundation/Foundation.h>

#include <mach/mach_time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>

#include "../ANEFieldfareRuntime/ANEExpertDispatch.h"
#include "../ANEFieldfareRuntime/MetalExpertDispatch.h"

static double milliseconds(uint64_t ticks) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)ticks * tb.numer / tb.denom / 1e6;
}

static bool closeEnough(const float *expected, const float *actual, uint64_t count,
                        double *maximum, double *rmse) {
    double sum = 0.0; *maximum = 0.0;
    for (uint64_t i = 0; i < count; ++i) {
        if (!isfinite(expected[i]) || !isfinite(actual[i])) return false;
        double delta = fabs((double)expected[i] - actual[i]);
        *maximum = fmax(*maximum, delta); sum += delta * delta;
    }
    *rmse = sqrt(sum / count);
    return *maximum <= 0.05 && *rmse <= 0.008;
}

static void makeInput(float *input, uint32_t position) {
    for (uint32_t i = 0; i < ANEFieldfareGemma4HiddenSize; ++i)
        input[i] = (float)((int32_t)((i * 17 + position * 13) % 101) - 50) / 512.0f;
}

static bool cpuReduced(ANEFieldfareRoutedLayer *layer, const uint32_t *experts,
                       const float *weights, const float *input, float *output) {
    memset(output, 0, ANEFieldfareGemma4HiddenSize * sizeof(float));
    float *one = malloc(ANEFieldfareGemma4HiddenSize * sizeof(float));
    if (!one) return false;
    bool valid = true;
    for (uint32_t rank = 0; valid && rank < ANEFieldfareGemma4TopK; ++rank) {
        valid = ANEFieldfareRoutedLayerCPUExpert(layer, experts[rank], input, one);
        for (uint32_t i = 0; valid && i < ANEFieldfareGemma4HiddenSize; ++i)
            output[i] += weights[rank] * one[i];
    }
    free(one); return valid;
}

static int runMetal(NSString *root, NSString *mode, uint32_t positions, uint32_t repetitions) {
    unsetenv("ANE_FIELDFARE_METAL_STAGING");
    if ([mode isEqualToString:@"metal-direct-mmap"])
        setenv("ANE_FIELDFARE_EXACT_EXPERT_CACHE", "0", 1);
    else if ([mode isEqualToString:@"metal-copied-staging"]) {
        setenv("ANE_FIELDFARE_EXACT_EXPERT_CACHE", "0", 1);
        setenv("ANE_FIELDFARE_METAL_STAGING", "1", 1);
    } else
        setenv("ANE_FIELDFARE_EXACT_EXPERT_CACHE", "1", 1);
    NSString *source = @"Sources/ANEFieldfareRuntime/Metal/moe.metal";
    const char *error = NULL;
    ANEFieldfareRoutedLayer *layer = NULL;
    ANEFieldfareMetalExpertDispatcher *metal = ANEFieldfareMetalExpertDispatcherCreateModel(
        root.fileSystemRepresentation, source.fileSystemRepresentation, &error);
    if (!metal || !ANEFieldfareRoutedLayerOpen(root.fileSystemRepresentation, 0, &layer)) {
        fprintf(stderr, "metal setup: %s\n", error ?: "layer open failed"); free((void *)error); return 1;
    }
    free((void *)error);
    const uint32_t experts[8] = {0, 17, 31, 43, 57, 71, 103, 127};
    const float weights[8] = {.05f, .075f, .10f, .125f, .15f, .175f, .15f, .175f};
    uint64_t elements = (uint64_t)positions * ANEFieldfareGemma4HiddenSize;
    float *inputs = calloc(elements, sizeof(float)), *outputs = calloc(elements, sizeof(float));
    uint32_t *routes = calloc((uint64_t)positions * 8, sizeof(uint32_t));
    float *routeWeights = calloc((uint64_t)positions * 8, sizeof(float));
    float *expected = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
    bool grouped = [mode isEqualToString:@"metal-grouped-hot"];
    bool valid = inputs && outputs && routes && routeWeights && expected;
    for (uint32_t p = 0; p < positions; ++p) {
        makeInput(inputs + (uint64_t)p * ANEFieldfareGemma4HiddenSize, p);
        for (uint32_t r = 0; r < 8; ++r) { routes[p * 8 + r] = experts[r]; routeWeights[p * 8 + r] = weights[r]; }
    }
    if (valid) {
        if (grouped) valid = ANEFieldfareMetalExpertDispatcherCallReducedPositionsAtLayer(
            metal, 0, positions, routes, routeWeights, inputs, outputs);
        else for (uint32_t p = 0; valid && p < positions; ++p)
            valid = ANEFieldfareMetalExpertDispatcherCallReducedAtLayer(metal, 0, experts, weights,
                inputs + (uint64_t)p * ANEFieldfareGemma4HiddenSize,
                outputs + (uint64_t)p * ANEFieldfareGemma4HiddenSize);
    }
    double maxError = 0.0, rmse = 0.0;
    double worstMax = 0.0, worstRMSE = 0.0;
    for (uint32_t p = 0; valid && p < positions; ++p) {
        valid = cpuReduced(layer, experts, weights,
            inputs + (uint64_t)p * ANEFieldfareGemma4HiddenSize, expected) &&
            closeEnough(expected, outputs + (uint64_t)p * ANEFieldfareGemma4HiddenSize,
                        ANEFieldfareGemma4HiddenSize, &maxError, &rmse);
        worstMax = fmax(worstMax, maxError); worstRMSE = fmax(worstRMSE, rmse);
    }
    uint64_t started = mach_absolute_time();
    for (uint32_t rep = 0; valid && rep < repetitions; ++rep) {
        if (grouped) valid = ANEFieldfareMetalExpertDispatcherCallReducedPositionsAtLayer(
            metal, 0, positions, routes, routeWeights, inputs, outputs);
        else for (uint32_t p = 0; valid && p < positions; ++p)
            valid = ANEFieldfareMetalExpertDispatcherCallReducedAtLayer(metal, 0, experts, weights,
                inputs + (uint64_t)p * ANEFieldfareGemma4HiddenSize,
                outputs + (uint64_t)p * ANEFieldfareGemma4HiddenSize);
    }
    double wall = milliseconds(mach_absolute_time() - started);
    ANEFieldfareMetalExpertDispatchInfo info = ANEFieldfareMetalExpertDispatcherGetInfo(metal);
    double perWindow = wall / repetitions;
    printf("{\"mode\":\"%s\",\"status\":\"%s\",\"positions\":%u,\"repetitions\":%u,"
           "\"wall_ms_per_window\":%.6f,\"unique_experts\":8,\"wall_ms_per_unique_expert\":%.6f,"
           "\"expert_position_services_per_second\":%.3f,\"gpu_total_ms\":%.6f,"
           "\"cache_hits\":%llu,\"cache_misses\":%llu,\"cache_fill_ms\":%.6f,"
           "\"max_absolute_error\":%.9g,\"rmse\":%.9g}\n",
           mode.UTF8String, valid ? "validated" : "rejected", positions, repetitions,
           perWindow, perWindow / 8.0, (double)positions * 8.0 * 1000.0 / perWindow,
           info.totalGPUMilliseconds, info.exactCacheHits, info.exactCacheMisses,
           info.totalExactCacheFillMilliseconds, worstMax, worstRMSE);
    free(inputs); free(outputs); free(routes); free(routeWeights); free(expected);
    ANEFieldfareMetalExpertDispatcherDestroy(metal); ANEFieldfareRoutedLayerClose(layer);
    return valid ? 0 : 1;
}

static int runANE(NSString *root, uint32_t repetitions) {
    NSString *packed = [root stringByAppendingPathComponent:@"packed_experts/layer_00.bin"];
    ANEFieldfareRoutedLayer *layer = NULL; const char *error = NULL;
    ANEFieldfareANEExpertDispatcher *ane = ANEFieldfareANEExpertDispatcherCreate(
        packed.fileSystemRepresentation, &error);
    if (!ane || !ANEFieldfareRoutedLayerOpen(root.fileSystemRepresentation, 0, &layer)) {
        fprintf(stderr, "ANE setup: %s\n", error ?: "layer open failed"); free((void *)error); return 1;
    }
    free((void *)error);
    const uint32_t experts[8] = {0, 17, 31, 43, 57, 71, 103, 127};
    float input[ANEFieldfareGemma4HiddenSize]; makeInput(input, 0);
    float (*outputs)[ANEFieldfareGemma4HiddenSize] = calloc(8, sizeof(*outputs));
    float *expected = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
    bool valid = outputs && expected && ANEFieldfareANEExpertDispatcherCallBatch(ane, experts, input, outputs);
    double maxError = 0.0, rmse = 0.0, worstMax = 0.0, worstRMSE = 0.0;
    for (uint32_t rank = 0; valid && rank < 8; ++rank) {
        valid = ANEFieldfareRoutedLayerCPUExpert(layer, experts[rank], input, expected) &&
            closeEnough(expected, outputs[rank], ANEFieldfareGemma4HiddenSize, &maxError, &rmse);
        worstMax = fmax(worstMax, maxError); worstRMSE = fmax(worstRMSE, rmse);
    }
    ANEFieldfareANEExpertDispatchInfo before = ANEFieldfareANEExpertDispatcherGetInfo(ane);
    uint64_t started = mach_absolute_time();
    for (uint32_t rep = 0; valid && rep < repetitions; ++rep)
        valid = ANEFieldfareANEExpertDispatcherCallBatch(ane, experts, input, outputs);
    double wall = milliseconds(mach_absolute_time() - started);
    ANEFieldfareANEExpertDispatchInfo after = ANEFieldfareANEExpertDispatcherGetInfo(ane);
    double perBatch = wall / repetitions;
    printf("{\"mode\":\"ane-concurrent-bank8\",\"status\":\"%s\",\"repetitions\":%u,"
           "\"experts_per_batch\":8,\"request_banks\":%u,\"wall_ms_per_batch\":%.6f,"
           "\"wall_ms_per_expert\":%.6f,\"experts_per_second\":%.3f,"
           "\"summed_completion_ms_per_batch\":%.6f,\"preparation_ms_per_batch\":%.6f,"
           "\"compile_count\":%u,\"load_count\":%u,\"max_absolute_error\":%.9g,\"rmse\":%.9g}\n",
           valid ? "validated" : "rejected", repetitions, after.program.requestBankCount,
           perBatch, perBatch / 8.0, 8.0 * 1000.0 / perBatch,
           (after.totalEvaluationMilliseconds - before.totalEvaluationMilliseconds) / repetitions,
           (after.totalPreparationMilliseconds - before.totalPreparationMilliseconds) / repetitions,
           1u, 1u, worstMax, worstRMSE);
    free(outputs); free(expected); ANEFieldfareANEExpertDispatcherDestroy(ane);
    ANEFieldfareRoutedLayerClose(layer); return valid ? 0 : 1;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) { fprintf(stderr, "usage: %s <model> <mode> [positions] [repetitions]\n", argv[0]); return 2; }
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *mode = [NSString stringWithUTF8String:argv[2]];
        uint32_t positions = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 64;
        uint32_t reps = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 3;
        if ([mode isEqualToString:@"ane-bank8"]) return runANE(root, reps ?: 1);
        return runMetal(root, mode, positions, reps ?: 1);
    }
}
