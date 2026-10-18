#include "../ANEFieldfareRuntime/RoutedLayer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonicMilliseconds(void) {
    struct timespec now = {0};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0;
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: ane-fieldfare-routed-layer-gate MODEL_DIR\n");
        return 2;
    }
    char weightsPath[4096];
    int length = snprintf(weightsPath, sizeof(weightsPath), "%s/model_weights.bin", argv[1]);
    ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
    ANEFieldfareTensor embedding = {0};
    float *input = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
    float *output = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
    ANEFieldfareRoutedLayer *layer = NULL;
    bool valid = input && output && length > 0 && (size_t)length < sizeof(weightsPath) &&
        ANEFieldfareTensorIndexOpen(weightsPath, &index) &&
        ANEFieldfareTensorIndexFind(&index, "language_model.model.embed_tokens.weight", &embedding) &&
        ANEFieldfareAffineRow(&embedding, 4, 64, 2, input) &&
        ANEFieldfareRoutedLayerOpen(argv[1], 0, &layer);
    ANEFieldfareRoute routes[ANEFieldfareGemma4TopK] = {0};
    double started = monotonicMilliseconds();
    valid = valid && ANEFieldfareRoutedLayerForward(layer, input, NULL, NULL, output, routes);
    double elapsed = monotonicMilliseconds() - started;

    double checksum = 0.0, norm = 0.0;
    for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column) {
        valid = isfinite(output[column]);
        checksum += output[column] * (column + 1.0);
        norm += (double)output[column] * output[column];
    }
    const uint32_t expected[ANEFieldfareGemma4TopK] = {79, 114, 42, 55, 16, 101, 51, 75};
    double routeWeight = 0.0;
    for (uint32_t rank = 0; valid && rank < ANEFieldfareGemma4TopK; ++rank) {
        valid = routes[rank].expert == expected[rank] && isfinite(routes[rank].weight);
        routeWeight += routes[rank].weight;
    }
    printf("{\"schema\":\"ane-fieldfare.routed-layer-gate.v1\","
           "\"status\":\"%s\",\"layer\":0,\"token\":2,"
           "\"top8\":[%u,%u,%u,%u,%u,%u,%u,%u],"
           "\"routed_weight_sum\":%.9g,\"output_l2\":%.9g,"
           "\"output_checksum\":%.17g,\"cpu_ms\":%.3f,"
           "\"claim_boundary\":\"Complete authentic layer-0 feed-forward half on the CPU oracle; attention and ANE dispatch are outside this gate.\"}\n",
           valid ? "validated" : "rejected",
           routes[0].expert, routes[1].expert, routes[2].expert, routes[3].expert,
           routes[4].expert, routes[5].expert, routes[6].expert, routes[7].expert,
           routeWeight, sqrt(norm), checksum, elapsed);
    ANEFieldfareRoutedLayerClose(layer);
    ANEFieldfareTensorIndexClose(&index);
    free(input);
    free(output);
    return valid ? 0 : 1;
}
