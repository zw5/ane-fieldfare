#import <Foundation/Foundation.h>

#include <mach/mach.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>

#include "../ANEFieldfareRuntime/MetalExpertDispatch.h"

static double milliseconds(uint64_t ticks) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)ticks * timebase.numer / timebase.denom / 1.0e6;
}

static NSString *jsonString(NSString *value) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:@[value ?: @""] options:0 error:nil];
    NSString *array = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    return [array substringWithRange:NSMakeRange(1, array.length - 2)];
}

static bool cpuReduced(ANEFieldfareRoutedLayer *layer,
                       const uint32_t experts[ANEFieldfareGemma4TopK],
                       const float weights[ANEFieldfareGemma4TopK],
                       const float input[ANEFieldfareGemma4HiddenSize],
                       float output[ANEFieldfareGemma4HiddenSize]) {
    memset(output, 0, ANEFieldfareGemma4HiddenSize * sizeof(float));
    float one[ANEFieldfareGemma4HiddenSize];
    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
        if (!ANEFieldfareRoutedLayerCPUExpert(layer, experts[rank], input, one)) return false;
        for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
            output[index] += weights[rank] * one[index];
    }
    return true;
}

static bool compare(const float *expected, const float *actual,
                    double *maxAbsolute, double *rmse) {
    double squared = 0.0;
    *maxAbsolute = 0.0;
    for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index) {
        if (!isfinite(expected[index]) || !isfinite(actual[index])) return false;
        double delta = fabs((double)expected[index] - actual[index]);
        if (delta > *maxAbsolute) *maxAbsolute = delta;
        squared += delta * delta;
    }
    *rmse = sqrt(squared / ANEFieldfareGemma4HiddenSize);
    return *maxAbsolute <= 0.035 && *rmse <= 0.006;
}

static bool metalReduced(void *context,
                         const uint32_t experts[ANEFieldfareGemma4TopK],
                         const float weights[ANEFieldfareGemma4TopK],
                         const float *input,
                         float *output) {
    return ANEFieldfareMetalExpertDispatcherCallReducedAtLayer(
        context, 0, experts, weights, input, output);
}

static bool metalShared(void *context, const uint8_t *base,
                        const ANEFieldfareTensor *gate,
                        const ANEFieldfareTensor *up,
                        const ANEFieldfareTensor *down,
                        const float *input, float *output) {
    return ANEFieldfareMetalExpertDispatcherCallSharedFFN(
        context, base, gate, up, down, input, output);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <model-directory> [iterations]\n", argv[0]);
            return 2;
        }
        NSString *model = [NSString stringWithUTF8String:argv[1]];
        uint32_t iterations = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 12;
        if (!iterations) iterations = 1;
        NSString *sourcePath = [[[NSString stringWithUTF8String:argv[0]]
            stringByDeletingLastPathComponent] stringByDeletingLastPathComponent];
        sourcePath = [sourcePath stringByAppendingPathComponent:
            @"Sources/ANEFieldfareRuntime/Metal/moe.metal"];

        ANEFieldfareRoutedLayer *layer = NULL;
        const char *createError = NULL;
        uint64_t createStart = mach_absolute_time();
        ANEFieldfareMetalExpertDispatcher *metal = ANEFieldfareMetalExpertDispatcherCreateModel(
            model.fileSystemRepresentation, sourcePath.fileSystemRepresentation, &createError);
        double createMilliseconds = milliseconds(mach_absolute_time() - createStart);
        if (!metal || !ANEFieldfareRoutedLayerOpen(model.fileSystemRepresentation, 0, &layer)) {
            fprintf(stderr, "gate setup failed: %s\n", createError ?: "layer open failed");
            free((void *)createError);
            ANEFieldfareMetalExpertDispatcherDestroy(metal);
            return 1;
        }
        free((void *)createError);

        float input[ANEFieldfareGemma4HiddenSize];
        for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
            input[index] = (float)((int32_t)(index % 97) - 48) / 256.0f;
        const uint32_t selections[3][ANEFieldfareGemma4TopK] = {
            {0, 1, 2, 3, 4, 5, 6, 7},
            {17, 31, 43, 57, 71, 89, 103, 127},
            {0, 1, 2, 3, 4, 5, 6, 7},
        };
        const float weights[ANEFieldfareGemma4TopK] =
            {0.05f, 0.075f, 0.10f, 0.125f, 0.15f, 0.175f, 0.15f, 0.175f};
        float expected[ANEFieldfareGemma4HiddenSize];
        float actual[ANEFieldfareGemma4HiddenSize];
        float firstA[ANEFieldfareGemma4HiddenSize];
        double maxAbsolute = 0.0, rmse = 0.0, worstMax = 0.0, worstRMSE = 0.0;
        bool valid = true, abaExact = true;

        for (uint32_t sequence = 0; sequence < 3 && valid; ++sequence) {
            valid = cpuReduced(layer, selections[sequence], weights, input, expected) &&
                ANEFieldfareMetalExpertDispatcherCallReducedAtLayer(
                    metal, 0, selections[sequence], weights, input, actual) &&
                compare(expected, actual, &maxAbsolute, &rmse);
            if (maxAbsolute > worstMax) worstMax = maxAbsolute;
            if (rmse > worstRMSE) worstRMSE = rmse;
            if (sequence == 0) memcpy(firstA, actual, sizeof(firstA));
            if (sequence == 2) abaExact = memcmp(firstA, actual, sizeof(firstA)) == 0;
        }
        valid = valid && abaExact;

        uint64_t timedStart = mach_absolute_time();
        for (uint32_t iteration = 0; iteration < iterations && valid; ++iteration)
            valid = ANEFieldfareMetalExpertDispatcherCallReducedAtLayer(
                metal, 0, selections[iteration & 1], weights, input, actual);
        double timedMilliseconds = milliseconds(mach_absolute_time() - timedStart);
        ANEFieldfareMetalExpertDispatchInfo info =
            ANEFieldfareMetalExpertDispatcherGetInfo(metal);

        struct utsname system = {0};
        uname(&system);
        size_t modelLength = 0;
        sysctlbyname("hw.model", NULL, &modelLength, NULL, 0);
        char *hostModel = calloc(modelLength ?: 1, 1);
        if (modelLength) sysctlbyname("hw.model", hostModel, &modelLength, NULL, 0);
        NSString *command = [NSString stringWithFormat:@"%s %s %u", argv[0], argv[1], iterations];
        float cpuLayerOutput[ANEFieldfareGemma4HiddenSize];
        float hybridLayerOutput[ANEFieldfareGemma4HiddenSize];
        double layerMax = 0.0, layerRMSE = 0.0;
        bool layerValid = ANEFieldfareRoutedLayerForward(
                layer, input, NULL, NULL, cpuLayerOutput, NULL) &&
            ANEFieldfareRoutedLayerForwardReducedBatched(
                layer, input, metalReduced, metal, hybridLayerOutput, NULL) &&
            compare(cpuLayerOutput, hybridLayerOutput, &layerMax, &layerRMSE);
        valid = valid && layerValid;
        float fusedLayerOutput[ANEFieldfareGemma4HiddenSize];
        double fusedLayerMax = 0.0, fusedLayerRMSE = 0.0;
        bool fusedLayerValid = ANEFieldfareRoutedLayerForwardHybrid(
                layer, input, metalReduced, metal, metalShared, metal,
                fusedLayerOutput, NULL) &&
            compare(cpuLayerOutput, fusedLayerOutput, &fusedLayerMax, &fusedLayerRMSE);
        valid = valid && fusedLayerValid;

        printf("{\n");
        printf("  \"schema\": \"ane-fieldfare.metal-compact-expert.v1\",\n");
        printf("  \"status\": \"%s\",\n", valid ? "validated" : "rejected");
        printf("  \"host\": \"%s\",\n", hostModel ?: "unknown");
        printf("  \"os_build\": \"%s\",\n", system.release);
        printf("  \"ane_architecture\": \"not_used_by_this_companion_rail\",\n");
        printf("  \"exact_command\": %s,\n", jsonString(command).UTF8String);
        printf("  \"exit_status\": %d,\n", valid ? 0 : 1);
        printf("  \"compile_count\": %u,\n", info.libraryCompileCount);
        printf("  \"load_count\": %u,\n", info.pipelineLoadCount);
        printf("  \"creation_ms\": %.6f,\n", createMilliseconds);
        printf("  \"timed_dispatches\": %u,\n", iterations);
        printf("  \"mean_wall_ms\": %.6f,\n", timedMilliseconds / iterations);
        printf("  \"mean_gpu_ms\": %.6f,\n",
               info.dispatchCount ? info.totalGPUMilliseconds / info.dispatchCount : 0.0);
        printf("  \"compact_layer_bytes\": %llu,\n", info.compactLayerBytes);
        printf("  \"selected_compact_bytes\": %llu,\n", info.mappedExpertBytesPerDispatch);
        printf("  \"fp16_weight_carrier_bytes\": 0,\n");
        printf("  \"max_absolute_error\": %.9g,\n", worstMax);
        printf("  \"rmse\": %.9g,\n", worstRMSE);
        printf("  \"aba_bit_exact\": %s,\n", abaExact ? "true" : "false");
        printf("  \"full_layer_max_absolute_error\": %.9g,\n", layerMax);
        printf("  \"full_layer_rmse\": %.9g,\n", layerRMSE);
        printf("  \"full_layer_validated\": %s,\n", layerValid ? "true" : "false");
        printf("  \"fused_shared_layer_max_absolute_error\": %.9g,\n", fusedLayerMax);
        printf("  \"fused_shared_layer_rmse\": %.9g,\n", fusedLayerRMSE);
        printf("  \"fused_shared_layer_validated\": %s,\n", fusedLayerValid ? "true" : "false");
        printf("  \"claim_boundary\": \"Selected compact INT4 experts execute and reduce on Metal without an FP16 weight carrier and pass the complete routed-layer seam. Thirty-layer recurrent decode is independently covered by the generation gate.\"\n");
        printf("}\n");

        free(hostModel);
        ANEFieldfareRoutedLayerClose(layer);
        ANEFieldfareMetalExpertDispatcherDestroy(metal);
        return valid ? 0 : 1;
    }
}
