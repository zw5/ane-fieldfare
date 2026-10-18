#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <libproc.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../ANEFieldfareRuntime/ANEExpertDispatch.h"
#include "../ANEFieldfareRuntime/Decode.h"
#include "../ANEFieldfareRuntime/Decoder.h"
#include "../ANEFieldfareRuntime/Tokenizer.h"

enum { PromptPositions = 21, ExpertStride = 3358720 };

typedef struct {
    const char *modelDirectory;
    ANEFieldfareANEExpertDispatcher *dispatcher;
    uint32_t activeLayer;
    bool poisonLastLane;
    char error[512];
} ExpertContext;

static bool selectLayer(ExpertContext *context, uint32_t layer) {
    if (context->activeLayer == layer) return true;
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/packed_experts/layer_%02u.bin",
                          context->modelDirectory, layer);
    const char *error = NULL;
    bool valid = length > 0 && (size_t)length < sizeof(path) &&
        ANEFieldfareANEExpertDispatcherSetPackedLayerPath(context->dispatcher, path, &error);
    if (!valid) snprintf(context->error, sizeof(context->error), "%s",
                         error ?: "packed-layer switch failed");
    free((void *)error);
    if (valid) context->activeLayer = layer;
    return valid;
}

static bool dispatchBatch(
    void *raw, uint32_t layer,
    const uint32_t experts[ANEFieldfareGemma4TopK], const float *input,
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize]) {
    ExpertContext *context = raw;
    bool valid = selectLayer(context, layer) &&
        ANEFieldfareANEExpertDispatcherCallBatch(context->dispatcher, experts, input, outputs);
    if (!valid && !context->error[0])
        snprintf(context->error, sizeof(context->error), "%s",
                 ANEFieldfareANEExpertDispatcherLastError(context->dispatcher) ?:
                 "expert batch dispatch failed");
    return valid;
}

static bool dispatchPositions(void *raw, uint32_t layer, uint32_t expert,
                              uint32_t count, const float *inputs, float *outputs) {
    ExpertContext *context = raw;
    bool valid = selectLayer(context, layer) &&
        ANEFieldfareANEExpertDispatcherCallPositions(
            context->dispatcher, expert, count, inputs, outputs);
    // Negative control: emulate a callback that claims success while dropping
    // one requested position. The decoder's NaN and coverage defenses must
    // reject this construction rather than reporting a false speedup.
    if (valid && context->poisonLastLane)
        outputs[(uint64_t)(count - 1) * ANEFieldfareGemma4HiddenSize] = NAN;
    if (!valid && !context->error[0])
        snprintf(context->error, sizeof(context->error), "%s",
                 ANEFieldfareANEExpertDispatcherLastError(context->dispatcher) ?:
                 "expert position dispatch failed");
    return valid;
}

static double milliseconds(uint64_t start, uint64_t end) {
    mach_timebase_info_data_t info = {0};
    mach_timebase_info(&info);
    return (double)(end - start) * info.numer / info.denom / 1e6;
}

static uint64_t diskBytesRead(void) {
    struct rusage_info_v4 usage = {0};
    return proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&usage) == 0
        ? usage.ri_diskio_bytesread : 0;
}

static ANEFieldfareANEExpertDispatcher *createDispatcher(NSString *root, const char **error) {
    NSString *layer = [root stringByAppendingPathComponent:@"packed_experts/layer_00.bin"];
    return ANEFieldfareANEExpertDispatcherCreate(layer.fileSystemRepresentation, error);
}

static void emit(NSDictionary *value) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(data.bytes, 1, data.length, stdout);
    fputc('\n', stdout);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: ane-fieldfare-prefill-batched-gate MODEL_DIR\n");
            return 2;
        }
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *weights = [root stringByAppendingPathComponent:@"model_weights.bin"];
        NSString *tokenizerPath = [root stringByAppendingPathComponent:@"tokenizer/tokenizer.json"];
        const char *prompt = "Explain why reducing repeated storage reads improves transformer prompt processing.";
        char *framed = ANEFieldfareGemmaUserPrompt(prompt);
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTokenizer *tokenizer = NULL;
        ANEFieldfareTokenIDs ids = {0};
        ANEFieldfareDecoder *sequentialDecoder = NULL, *batchedDecoder = NULL, *negativeDecoder = NULL;
        const char *sequentialError = NULL, *batchedError = NULL, *negativeError = NULL;
        ExpertContext sequential = {.modelDirectory = root.fileSystemRepresentation, .activeLayer = UINT32_MAX};
        ExpertContext batched = {.modelDirectory = root.fileSystemRepresentation, .activeLayer = UINT32_MAX};
        ExpertContext negative = {.modelDirectory = root.fileSystemRepresentation, .activeLayer = UINT32_MAX,
                                  .poisonLastLane = true};
        float *sequentialRows = calloc((size_t)PromptPositions * ANEFieldfareGemma4HiddenSize, sizeof(float));
        float *batchedRows = calloc((size_t)PromptPositions * ANEFieldfareGemma4HiddenSize, sizeof(float));
        float *negativeRows = calloc(2 * ANEFieldfareGemma4HiddenSize, sizeof(float));
        bool valid = framed && sequentialRows && batchedRows && negativeRows &&
            ANEFieldfareTensorIndexOpen(weights.fileSystemRepresentation, &index) &&
            (tokenizer = ANEFieldfareTokenizerOpen(tokenizerPath.fileSystemRepresentation)) &&
            ANEFieldfareTokenizerEncode(tokenizer, framed, &ids) && ids.count >= PromptPositions;
        for (uint32_t row = 0; valid && row < PromptPositions; ++row)
            valid = ANEFieldfareScaledEmbedding(
                &index, ids.values[row], sequentialRows + (uint64_t)row * ANEFieldfareGemma4HiddenSize);
        if (valid) memcpy(batchedRows, sequentialRows,
                          (size_t)PromptPositions * ANEFieldfareGemma4HiddenSize * sizeof(float));
        if (valid) memcpy(negativeRows, sequentialRows,
                          2 * ANEFieldfareGemma4HiddenSize * sizeof(float));
        valid = valid &&
            ANEFieldfareDecoderOpen(root.fileSystemRepresentation, PromptPositions + 1, &sequentialDecoder) &&
            ANEFieldfareDecoderOpen(root.fileSystemRepresentation, PromptPositions + 1, &batchedDecoder) &&
            ANEFieldfareDecoderOpen(root.fileSystemRepresentation, 3, &negativeDecoder) &&
            (sequential.dispatcher = createDispatcher(root, &sequentialError)) &&
            (batched.dispatcher = createDispatcher(root, &batchedError)) &&
            (negative.dispatcher = createDispatcher(root, &negativeError));

        uint64_t sequentialDiskStart = diskBytesRead();
        uint64_t sequentialStart = mach_absolute_time();
        for (uint32_t row = 0; valid && row < PromptPositions; ++row)
            valid = ANEFieldfareDecoderTokenBatched(
                sequentialDecoder, row,
                sequentialRows + (uint64_t)row * ANEFieldfareGemma4HiddenSize,
                dispatchBatch, &sequential);
        uint64_t sequentialEnd = mach_absolute_time();
        uint64_t sequentialPhysicalDiskBytes = diskBytesRead() - sequentialDiskStart;
        uint64_t batchedDiskStart = diskBytesRead();
        uint64_t batchedStart = mach_absolute_time();
        bool batchedAccepted = valid && ANEFieldfareDecoderPrefillBatched(
            batchedDecoder, 0, PromptPositions, batchedRows, dispatchPositions, &batched);
        uint64_t batchedEnd = mach_absolute_time();
        uint64_t batchedPhysicalDiskBytes = diskBytesRead() - batchedDiskStart;
        bool hiddenBitExact = batchedAccepted && memcmp(
            sequentialRows + (PromptPositions - 1ULL) * ANEFieldfareGemma4HiddenSize,
            batchedRows + (PromptPositions - 1ULL) * ANEFieldfareGemma4HiddenSize,
            ANEFieldfareGemma4HiddenSize * sizeof(float)) == 0;
        uint32_t sequentialToken = 0, batchedToken = 0;
        float sequentialLogit = 0.0f, batchedLogit = 0.0f;
        bool decoded = hiddenBitExact &&
            ANEFieldfareDecodeGreedy(
                &index, sequentialRows + (PromptPositions - 1ULL) * ANEFieldfareGemma4HiddenSize,
                256, &sequentialToken, &sequentialLogit) &&
            ANEFieldfareDecodeGreedy(
                &index, batchedRows + (PromptPositions - 1ULL) * ANEFieldfareGemma4HiddenSize,
                256, &batchedToken, &batchedLogit);
        char *sequentialText = decoded ? ANEFieldfareTokenizerDecode(tokenizer, &sequentialToken, 1, true) : NULL;
        char *batchedText = decoded ? ANEFieldfareTokenizerDecode(tokenizer, &batchedToken, 1, true) : NULL;
        bool textBitExact = sequentialText && batchedText && !strcmp(sequentialText, batchedText);

        bool negativeRejected = valid && !ANEFieldfareDecoderPrefillBatched(
            negativeDecoder, 0, 2, negativeRows, dispatchPositions, &negative);
        ANEFieldfareANEExpertDispatchInfo sequentialInfo =
            ANEFieldfareANEExpertDispatcherGetInfo(sequential.dispatcher);
        ANEFieldfareANEExpertDispatchInfo batchedInfo =
            ANEFieldfareANEExpertDispatcherGetInfo(batched.dispatcher);
        double sequentialMS = milliseconds(sequentialStart, sequentialEnd);
        double batchedMS = milliseconds(batchedStart, batchedEnd);
        // CallBatch advances dispatchCount once per expert (top-k times), while
        // CallPositions advances it once per union-expert call.  Both counters
        // are therefore already in expert-source visits; multiplying the
        // sequential count by top-k again would double-count it.
        uint64_t sequentialSourceByteVisits =
            sequentialInfo.dispatchCount * (uint64_t)ExpertStride;
        uint64_t batchedSourceByteVisits = batchedInfo.dispatchCount * (uint64_t)ExpertStride;
        bool validated = valid && batchedAccepted && hiddenBitExact && decoded && textBitExact &&
            sequentialToken == batchedToken && negativeRejected &&
            batchedInfo.dispatchCount < sequentialInfo.dispatchCount * ANEFieldfareGemma4TopK;
        emit(@{
            @"schema": @"ane-fieldfare.prefill-batched.v1",
            @"status": validated ? @"validated" : (batchedAccepted ? @"accepted" : @"rejected"),
            @"positions": @(PromptPositions),
            @"sequential": @{
                @"prefill_ms": @(sequentialMS), @"expert_dispatches": @(sequentialInfo.dispatchCount),
                @"expert_source_byte_visits": @(sequentialSourceByteVisits),
                @"process_physical_disk_bytes": @(sequentialPhysicalDiskBytes),
                @"next_token": @(sequentialToken),
                @"text": sequentialText ? [NSString stringWithUTF8String:sequentialText] : @""
            },
            @"batched": @{
                @"prefill_ms": @(batchedMS), @"expert_dispatches": @(batchedInfo.dispatchCount),
                @"expert_source_byte_visits": @(batchedSourceByteVisits),
                @"process_physical_disk_bytes": @(batchedPhysicalDiskBytes),
                @"next_token": @(batchedToken),
                @"text": batchedText ? [NSString stringWithUTF8String:batchedText] : @""
            },
            @"oracles": @{
                @"last_hidden_bit_exact": @(hiddenBitExact), @"generated_text_byte_identical": @(textBitExact),
                @"dropped_position_negative_control_rejected": @(negativeRejected),
                @"all_position_rank_pairs_covered": @(batchedAccepted)
            },
            @"speedup": batchedMS > 0 ? @(sequentialMS / batchedMS) : @0,
            @"source_byte_visit_reduction": batchedSourceByteVisits > 0
                ? @((double)sequentialSourceByteVisits / batchedSourceByteVisits) : @0,
            @"baseline_reference_ms": @14066,
            @"claim_boundary": validated
                ? @"Twenty-one causal prompt rows traversed all 30 authentic layers. Joined expert dispatch preserved the final hidden state and generated text bit-for-bit, every row/rank was covered exactly once, and a poisoned dropped-position control failed closed. Expert source-byte visits are structural fill traffic. process_physical_disk_bytes is the process rusage counter and remains order/cache sensitive; neither is claimed as a cold-device throughput measurement."
                : @"The batched prefill construction did not clear every numerical, coverage, and negative-control oracle; no throughput improvement is validated.",
            @"errors": @{
                @"sequential": sequential.error[0] ? [NSString stringWithUTF8String:sequential.error] : @"none",
                @"batched": batched.error[0] ? [NSString stringWithUTF8String:batched.error] : @"none",
                @"negative": negative.error[0] ? [NSString stringWithUTF8String:negative.error] : @"none"
            }
        });

        free(sequentialText); free(batchedText); free(framed);
        free(sequentialRows); free(batchedRows); free(negativeRows);
        ANEFieldfareTokenIDsFree(&ids);
        ANEFieldfareANEExpertDispatcherDestroy(sequential.dispatcher);
        ANEFieldfareANEExpertDispatcherDestroy(batched.dispatcher);
        ANEFieldfareANEExpertDispatcherDestroy(negative.dispatcher);
        ANEFieldfareDecoderClose(sequentialDecoder); ANEFieldfareDecoderClose(batchedDecoder);
        ANEFieldfareDecoderClose(negativeDecoder); ANEFieldfareTokenizerClose(tokenizer);
        ANEFieldfareTensorIndexClose(&index);
        free((void *)sequentialError); free((void *)batchedError); free((void *)negativeError);
        return validated ? 0 : 1;
    }
}
