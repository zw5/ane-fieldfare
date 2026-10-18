#include "../ANEFieldfareRuntime/Decoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double milliseconds(void) {
    struct timespec now = {0};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1000.0 + now.tv_nsec / 1e6;
}

int main(int argc, const char *argv[]) {
    if (argc != 2) return 2;
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/model_weights.bin", argv[1]);
    ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
    ANEFieldfareTensor embedding = {0};
    float *hidden = calloc(ANEFieldfareGemma4HiddenSize, sizeof(float));
    ANEFieldfareDecoder *decoder = NULL;
    bool valid = hidden && length > 0 && (size_t)length < sizeof(path) &&
        ANEFieldfareTensorIndexOpen(path, &index) &&
        ANEFieldfareTensorIndexFind(&index, "language_model.model.embed_tokens.weight", &embedding) &&
        ANEFieldfareAffineRow(&embedding, 4, 64, 2, hidden) &&
        ANEFieldfareDecoderOpen(argv[1], 2, &decoder);
    float embeddingScale = sqrtf((float)ANEFieldfareGemma4HiddenSize);
    for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column)
        hidden[column] *= embeddingScale;
    double started = milliseconds();
    valid = valid && ANEFieldfareDecoderToken(decoder, 0, hidden, NULL, NULL);
    double elapsed = milliseconds() - started;
    double square = 0, checksum = 0, maximum = 0;
    for (uint32_t column = 0; valid && column < ANEFieldfareGemma4HiddenSize; ++column) {
        valid = isfinite(hidden[column]);
        square += (double)hidden[column] * hidden[column];
        checksum += hidden[column] * (column + 1.0);
        maximum = fmax(maximum, fabs(hidden[column]));
    }
    printf("{\"schema\":\"ane-fieldfare.decoder-gate.v1\",\"status\":\"%s\","
           "\"token\":2,\"position\":0,\"layers\":30,\"elapsed_ms\":%.3f,"
           "\"output_rms\":%.12g,\"output_abs_max\":%.12g,\"output_checksum\":%.17g,"
           "\"claim_boundary\":\"All 30 authentic attention and routed feed-forward layer controls executed in model order with KV state; final norm, tied logits, token selection, and independent numerical oracle remain outside this gate.\"}\n",
           valid ? "accepted" : "rejected", elapsed,
           sqrt(square / ANEFieldfareGemma4HiddenSize), maximum, checksum);
    ANEFieldfareDecoderClose(decoder);
    ANEFieldfareTensorIndexClose(&index);
    free(hidden);
    return valid ? 0 : 1;
}
