#import <Foundation/Foundation.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../ANEFieldfareRuntime/ANEExpertDispatch.h"

static bool rawBF16(const ANEFieldfareTensor *tensor, uint32_t count, float *output) {
    if (!tensor || tensor->weightBytes != (uint64_t)count * 2 || tensor->scaleBytes || tensor->biasBytes)
        return false;
    for (uint32_t index = 0; index < count; ++index)
        output[index] = ANEFieldfareBF16(tensor->weights + (uint64_t)index * 2);
    return true;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *weights = [root stringByAppendingPathComponent:@"model_weights.bin"];
        NSString *packed = [root stringByAppendingPathComponent:@"packed_experts/layer_00.bin"];
        NSString *packed1 = [root stringByAppendingPathComponent:@"packed_experts/layer_01.bin"];
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor embedding = {0}, norm = {0};
        float *residual = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
        float *scale = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
        float *input = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
        float *actual = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
        float *expected = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
        float *layerActual = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
        float *layerExpected = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
        ANEFieldfareRoutedLayer *layer = NULL;
        ANEFieldfareRoutedLayer *layer1 = NULL;
        const char *error = NULL;
        ANEFieldfareANEExpertDispatcher *dispatcher = NULL;
        bool valid = residual && scale && input && actual && expected && layerActual && layerExpected &&
            ANEFieldfareTensorIndexOpen(weights.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.embed_tokens.weight", &embedding) &&
            ANEFieldfareTensorIndexFind(&index,
                "language_model.model.layers.0.pre_feedforward_layernorm_2.weight", &norm) &&
            ANEFieldfareAffineRow(&embedding, 4, 64, 2, residual) &&
            rawBF16(&norm, ANEFieldfareGemma4HiddenSize, scale) &&
            ANEFieldfareRMSNorm(residual, scale, ANEFieldfareGemma4HiddenSize, 1e-6f, input) &&
            ANEFieldfareRoutedLayerOpen(root.fileSystemRepresentation, 0, &layer) &&
            ANEFieldfareRoutedLayerOpen(root.fileSystemRepresentation, 1, &layer1) &&
            (dispatcher = ANEFieldfareANEExpertDispatcherCreate(packed.fileSystemRepresentation, &error));
        uint32_t sequence[] = {0, 1, 0};
        double maximumError = 0.0;
        double firstChecksum = 0.0, lastChecksum = 0.0;
        for (uint32_t run = 0; valid && run < 3; ++run) {
            valid = ANEFieldfareRoutedLayerCPUExpert(layer, sequence[run], input, expected) &&
                ANEFieldfareANEExpertDispatcherCall(dispatcher, sequence[run], input, actual);
            double checksum = 0.0;
            for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column) {
                double difference = fabs((double)actual[column] - expected[column]);
                maximumError = fmax(maximumError, difference);
                valid = isfinite(actual[column]) && difference <= 0.05 + 0.03 * fabs(expected[column]);
                checksum += actual[column] * (column + 1.0);
            }
            if (run == 0) firstChecksum = checksum;
            if (run == 2) lastChecksum = checksum;
        }
        valid = valid && firstChecksum == lastChecksum;
        ANEFieldfareRoute cpuRoutes[ANEFieldfareGemma4TopK] = {0};
        ANEFieldfareRoute aneRoutes[ANEFieldfareGemma4TopK] = {0};
        valid = valid &&
            ANEFieldfareRoutedLayerForward(layer, residual, NULL, NULL,
                                           layerExpected, cpuRoutes) &&
            ANEFieldfareRoutedLayerForward(layer, residual,
                                           ANEFieldfareANEExpertDispatcherCall, dispatcher,
                                           layerActual, aneRoutes);
        double layerMaximumError = 0.0, layerRMSE = 0.0;
        for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column) {
            double difference = fabs((double)layerActual[column] - layerExpected[column]);
            layerMaximumError = fmax(layerMaximumError, difference);
            layerRMSE += difference * difference;
            valid = isfinite(layerActual[column]) && difference <= 0.05 + 0.03 * fabs(layerExpected[column]);
        }
        layerRMSE = sqrt(layerRMSE / ANEFieldfareGemma4HiddenSize);
        for (uint32_t rank = 0; valid && rank < ANEFieldfareGemma4TopK; ++rank)
            valid = cpuRoutes[rank].expert == aneRoutes[rank].expert &&
                    cpuRoutes[rank].weight == aneRoutes[rank].weight;
        ANEFieldfareANEProgramInfo programBeforeSwitch =
            ANEFieldfareANEExpertDispatcherGetInfo(dispatcher).program;
        valid = valid && ANEFieldfareANEExpertDispatcherSetPackedLayerPath(
            dispatcher, packed1.fileSystemRepresentation, &error) &&
            ANEFieldfareRoutedLayerCPUExpert(layer1, 0, input, expected) &&
            ANEFieldfareANEExpertDispatcherCall(dispatcher, 0, input, actual);
        for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column) {
            double difference = fabs((double)actual[column] - expected[column]);
            valid = difference <= 0.05 + 0.03 * fabs(expected[column]);
        }
        valid = valid && ANEFieldfareANEExpertDispatcherSetPackedLayerPath(
            dispatcher, packed.fileSystemRepresentation, &error) &&
            ANEFieldfareRoutedLayerCPUExpert(layer, 0, input, expected) &&
            ANEFieldfareANEExpertDispatcherCall(dispatcher, 0, input, actual);
        for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column) {
            double difference = fabs((double)actual[column] - expected[column]);
            valid = difference <= 0.05 + 0.03 * fabs(expected[column]);
        }
        ANEFieldfareANEExpertDispatchInfo info = ANEFieldfareANEExpertDispatcherGetInfo(dispatcher);
        valid = valid && info.program.compileMilliseconds == programBeforeSwitch.compileMilliseconds &&
                info.program.loadMilliseconds == programBeforeSwitch.loadMilliseconds;
        printf("{\"schema\":\"ane-fieldfare.ane-expert-dispatch.v1\","
               "\"status\":\"%s\",\"sequence\":[0,1,0],\"dispatches\":%llu,"
               "\"request_banks\":%u,\"compile_ms\":%.3f,\"load_ms\":%.3f,"
               "\"preparation_ms\":%.3f,\"evaluation_ms\":%.3f,"
               "\"max_absolute_error\":%.9g,\"aba_checksum_restored\":%s,"
               "\"routed_layer_max_absolute_error\":%.9g,\"routed_layer_rmse\":%.9g,"
               "\"layer_switch_sequence\":[0,1,0],\"recompiled_on_switch\":false,"
               "\"error\":\"%s\"}\n",
               valid ? "validated" : "rejected", info.dispatchCount,
               info.program.requestBankCount, info.program.compileMilliseconds,
               info.program.loadMilliseconds, info.totalPreparationMilliseconds,
               info.totalEvaluationMilliseconds, maximumError,
               firstChecksum == lastChecksum ? "true" : "false",
               layerMaximumError, layerRMSE,
               error ?: ANEFieldfareANEExpertDispatcherLastError(dispatcher) ?: "none");
        free((void *)error);
        ANEFieldfareANEExpertDispatcherDestroy(dispatcher);
        ANEFieldfareRoutedLayerClose(layer);
        ANEFieldfareRoutedLayerClose(layer1);
        ANEFieldfareTensorIndexClose(&index);
        free(residual); free(scale); free(input); free(actual); free(expected);
        free(layerActual); free(layerExpected);
        return valid ? 0 : 1;
    }
}
