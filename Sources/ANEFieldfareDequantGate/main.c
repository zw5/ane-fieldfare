#include "../ANEFieldfareRuntime/Dequant.h"
#include "../ANEFieldfareRuntime/TensorIndex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    enum { groups = 4096, elements = groups * 64 };
    uint8_t *weights = malloc(groups * 32), *scales = malloc(groups * 2);
    uint8_t *biases = malloc(groups * 2);
    __fp16 *storage = malloc((elements + 2) * sizeof(__fp16));
    if (!weights || !scales || !biases || !storage) return 2;
    uint32_t state = 17;
    for (size_t i = 0; i < groups * 32; ++i) {
        state = state * 1664525u + 1013904223u;
        weights[i] = (uint8_t)(state >> 16);
    }
    for (size_t i = 0; i < groups; ++i) {
        // Signed, finite BF16 values over a wide exponent range.
        uint16_t scale = (uint16_t)(((i & 1) << 15) | ((100 + i % 36) << 7) | (i % 128));
        uint16_t bias = (uint16_t)(((i & 2) << 14) | ((100 + i % 35) << 7) | ((i * 7) % 128));
        memcpy(scales + i * 2, &scale, 2);
        memcpy(biases + i * 2, &bias, 2);
    }
    storage[0] = storage[elements + 1] = (__fp16)123;
    ANEFieldfareDequantGroupsFP16(NULL, NULL, NULL, 0, NULL);
    ANEFieldfareDequantGroupsFP16(weights, scales, biases, groups, storage + 1);
    ANEFieldfareTensor matrix = {
        .weights = weights, .weightBytes = groups * 32,
        .scales = scales, .scaleBytes = groups * 2,
        .biases = biases, .biasBytes = groups * 2,
        .rows = groups, .columns = 64,
    };
    size_t mismatches = 0;
    float maxError = 0;
    for (size_t i = 0; i < elements; ++i) {
        float reference;
        if (!ANEFieldfareAffineValue(&matrix, 4, 64, (uint32_t)(i / 64),
                                     (uint32_t)(i % 64), &reference)) return 3;
        __fp16 expected = (__fp16)reference;
        float error = fabsf((float)storage[i + 1] - (float)expected);
        if (error > maxError) maxError = error;
        mismatches += storage[i + 1] != expected;
    }
    int guards = storage[0] == (__fp16)123 && storage[elements + 1] == (__fp16)123;
    printf("{\"schema\":\"ane-fieldfare.dequant-restoration.v1\","
           "\"status\":\"%s\",\"comparisons\":%d,\"mismatches\":%zu,"
           "\"maximum_absolute_error\":%.9g,\"guard_values_preserved\":%s,"
           "\"compile_count\":0,\"load_count\":0,"
           "\"claim_boundary\":\"New CPU preparation implementation; exact FP16 equality to the independent scalar affine oracle. No ANE execution or generation throughput claim.\"}\n",
           !mismatches && guards ? "validated" : "rejected", elements,
           mismatches, maxError, guards ? "true" : "false");
    free(weights); free(scales); free(biases); free(storage);
    return mismatches || !guards ? 1 : 0;
}
