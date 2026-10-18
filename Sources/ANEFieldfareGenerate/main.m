#import <Foundation/Foundation.h>
#import <mach/mach.h>
#import <mach/mach_time.h>

#include <libproc.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ANEFieldfareRuntime/ANEExpertDispatch.h"
#include "../ANEFieldfareRuntime/Decode.h"
#include "../ANEFieldfareRuntime/Decoder.h"
#include "../ANEFieldfareRuntime/MetalExpertDispatch.h"
#include "../ANEFieldfareRuntime/Tokenizer.h"

static uint64_t currentPhysicalFootprint(void) {
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS
        ? info.phys_footprint : UINT64_MAX;
}

typedef struct {
    const char *modelDirectory;
    ANEFieldfareANEExpertDispatcher *dispatcher;
    uint32_t activeLayer;
    char error[512];
    uint32_t prefetchMode;
    uint32_t prefetchTopN;
    uint32_t routeHistory[30][2][ANEFieldfareGemma4TopK];
    uint32_t routeHistoryCount[30];
    uint64_t routeFrequency[30][ANEFieldfareGemma4ExpertCount];
    uint32_t pendingPrediction[30][2 * ANEFieldfareGemma4TopK];
    uint32_t pendingPredictionCount[30];
    uint64_t predictionOpportunities;
    uint64_t predictionHits;
    uint64_t predictionScheduled;
    uint64_t predictionExperts;
} ExpertContext;

typedef struct {
    ANEFieldfareMetalExpertDispatcher *dispatcher;
    char error[512];
} MetalExpertContext;

static bool dispatchMetalReduced(
    void *rawContext,
    uint32_t layer,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float weights[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float reduced[ANEFieldfareGemma4HiddenSize]) {
    MetalExpertContext *context = rawContext;
    bool valid = context && ANEFieldfareMetalExpertDispatcherCallReducedAtLayer(
        context->dispatcher, layer, experts, weights, input, reduced);
    if (!valid && context && !context->error[0])
        snprintf(context->error, sizeof(context->error), "%s",
                 context->dispatcher
                     ? ANEFieldfareMetalExpertDispatcherLastError(context->dispatcher)
                     : "Metal expert dispatcher is unavailable");
    return valid;
}

static bool dispatchMetalShared(
    void *rawContext,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *gate,
    const ANEFieldfareTensor *up,
    const ANEFieldfareTensor *down,
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]) {
    MetalExpertContext *context = rawContext;
    bool valid = context && ANEFieldfareMetalExpertDispatcherCallSharedFFN(
        context->dispatcher, tensorIndexBase, gate, up, down, input, output);
    if (!valid && context && !context->error[0])
        snprintf(context->error, sizeof(context->error), "%s",
                 ANEFieldfareMetalExpertDispatcherLastError(context->dispatcher));
    return valid;
}

static bool dispatchMetalAffine(void *rawContext,
                                const uint8_t *tensorIndexBase,
                                const ANEFieldfareTensor *matrix,
                                const float *input,
                                float *output) {
    MetalExpertContext *context = rawContext;
    bool valid = context && ANEFieldfareMetalExpertDispatcherCallAffine(
        context->dispatcher, tensorIndexBase, matrix, input, output);
    if (!valid && context && !context->error[0])
        snprintf(context->error, sizeof(context->error), "%s",
                 ANEFieldfareMetalExpertDispatcherLastError(context->dispatcher));
    return valid;
}

static bool dispatchMetalAttentionInput(
    void *rawContext,
    const uint8_t *tensorIndexBase,
    const ANEFieldfareTensor *query,
    const ANEFieldfareTensor *key,
    const ANEFieldfareTensor *value,
    bool valueEqualsKey,
    const float *input,
    float *queryOutput,
    float *keyOutput,
    float *valueOutput) {
    MetalExpertContext *context = rawContext;
    bool valid = context && ANEFieldfareMetalExpertDispatcherCallAttentionInput(
        context->dispatcher, tensorIndexBase, query, key, value, valueEqualsKey,
        input, queryOutput, keyOutput, valueOutput);
    if (!valid && context && !context->error[0])
        snprintf(context->error, sizeof(context->error), "%s",
                 ANEFieldfareMetalExpertDispatcherLastError(context->dispatcher));
    return valid;
}

static bool decodeMetalGreedy(ANEFieldfareMetalExpertDispatcher *dispatcher,
                              const ANEFieldfareTensorIndex *index,
                              const ANEFieldfareTensor *embedding,
                              const float *hidden,
                              float *logits,
                              uint32_t *token,
                              float *winningLogit) {
    float normalized[ANEFieldfareGemmaHiddenSize];
    memcpy(normalized, hidden, sizeof(normalized));
    if (!ANEFieldfareFinalRMSNorm(index, 1e-6f, normalized)) return false;
    if (getenv("ANE_FIELDFARE_FUSED_GREEDY_HEAD")) {
        float raw = 0.0f;
        if (!ANEFieldfareMetalExpertDispatcherCallGreedyHead(
                dispatcher, index->bytes, embedding, normalized, token, &raw)) return false;
        *winningLogit = 30.0f * tanhf(raw / 30.0f);
        return isfinite(*winningLogit);
    }
    if (!ANEFieldfareMetalExpertDispatcherCallAffine(
            dispatcher, index->bytes, embedding, normalized, logits)) return false;
    float best = -FLT_MAX;
    uint32_t bestToken = 0;
    for (uint32_t candidate = 0; candidate < embedding->rows; ++candidate) {
        if (!isfinite(logits[candidate])) return false;
        if (logits[candidate] > best) { best = logits[candidate]; bestToken = candidate; }
    }
    *token = bestToken;
    *winningLogit = 30.0f * tanhf(best / 30.0f);
    return isfinite(*winningLogit);
}

static uint32_t buildPrediction(ExpertContext *context, uint32_t layer,
                                uint32_t predicted[2 * ANEFieldfareGemma4TopK]) {
    if (!context || layer >= 30 || !context->prefetchMode) return 0;
    uint32_t count = 0;
    if (context->prefetchMode == 1) {
        uint32_t history = MIN(context->routeHistoryCount[layer], 2u);
        for (uint32_t age = 0; age < history; ++age)
            for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
                uint32_t expert = context->routeHistory[layer][age][rank];
                bool seen = false;
                for (uint32_t index = 0; index < count; ++index) seen |= predicted[index] == expert;
                if (!seen) predicted[count++] = expert;
            }
    } else if (context->prefetchMode == 2) {
        bool observed = false;
        for (uint32_t expert = 0; expert < ANEFieldfareGemma4ExpertCount; ++expert)
            observed |= context->routeFrequency[layer][expert] != 0;
        if (!observed) return 0;
        for (uint32_t slot = 0; slot < context->prefetchTopN; ++slot) {
            uint32_t best = UINT32_MAX;
            uint64_t bestCount = 0;
            for (uint32_t expert = 0; expert < ANEFieldfareGemma4ExpertCount; ++expert) {
                bool already = false;
                for (uint32_t index = 0; index < count; ++index) already |= predicted[index] == expert;
                uint64_t frequency = context->routeFrequency[layer][expert];
                if (!already && (best == UINT32_MAX || frequency > bestCount)) {
                    best = expert; bestCount = frequency;
                }
            }
            if (best == UINT32_MAX) break;
            predicted[count++] = best;
        }
    } else {
        // Deliberately content-free predictor used only as a negative control.
        // It exercises identical async I/O without consulting routing state.
        for (uint32_t slot = 0; slot < context->prefetchTopN; ++slot)
            predicted[count++] = ANEFieldfareGemma4ExpertCount - 1 - slot;
    }
    return count;
}

static void observeRoute(ExpertContext *context, uint32_t layer,
                         const uint32_t experts[ANEFieldfareGemma4TopK]) {
    if (!context || layer >= 30) return;
    uint32_t predicted = context->pendingPredictionCount[layer];
    if (predicted) {
        context->predictionOpportunities += ANEFieldfareGemma4TopK;
        for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
            for (uint32_t index = 0; index < predicted; ++index)
                if (experts[rank] == context->pendingPrediction[layer][index]) {
                    ++context->predictionHits; break;
                }
    }
    memcpy(context->routeHistory[layer][1], context->routeHistory[layer][0],
           sizeof(context->routeHistory[layer][0]));
    memcpy(context->routeHistory[layer][0], experts,
           sizeof(context->routeHistory[layer][0]));
    ++context->routeHistoryCount[layer];
    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
        ++context->routeFrequency[layer][experts[rank]];
}

static bool selectLayer(ExpertContext *context, uint32_t layer) {
    if (!context || !context->dispatcher || layer >= 30) return false;
    if (context->activeLayer == layer) return true;
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/packed_experts/layer_%02u.bin",
                          context->modelDirectory, layer);
    const char *error = NULL;
    bool switched = length > 0 && (size_t)length < sizeof(path) &&
        ANEFieldfareANEExpertDispatcherSetPackedLayerPath(context->dispatcher, path, &error);
    if (!switched) snprintf(context->error, sizeof(context->error), "%s",
                           error ?: "packed-layer switch failed");
    free((void *)error);
    if (switched) context->activeLayer = layer;
    return switched;
}

static bool dispatchExpertBatch(
    void *rawContext,
    uint32_t layer,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize]) {
    ExpertContext *context = rawContext;
    if (context && context->prefetchMode) {
        ANEFieldfareANEExpertDispatcherWaitForLayerPrefetch(context->dispatcher);
        if (layer + 1 < 30) {
            uint32_t predicted[2 * ANEFieldfareGemma4TopK] = {0};
            uint32_t count = buildPrediction(context, layer + 1, predicted);
            context->pendingPredictionCount[layer + 1] = count;
            if (count) {
                memcpy(context->pendingPrediction[layer + 1], predicted,
                       (size_t)count * sizeof(*predicted));
                char path[4096];
                int length = snprintf(path, sizeof(path), "%s/packed_experts/layer_%02u.bin",
                                      context->modelDirectory, layer + 1);
                if (length > 0 && (size_t)length < sizeof(path) &&
                    ANEFieldfareANEExpertDispatcherPrefetchLayerExperts(
                        context->dispatcher, path, predicted, count)) {
                    ++context->predictionScheduled;
                    context->predictionExperts += count;
                }
            }
        }
    }
    bool valid = selectLayer(context, layer) &&
        ANEFieldfareANEExpertDispatcherCallBatch(context->dispatcher, experts, input, outputs);
    if (valid) observeRoute(context, layer, experts);
    if (!valid && context && !context->error[0])
        snprintf(context->error, sizeof(context->error), "%s",
                 ANEFieldfareANEExpertDispatcherLastError(context->dispatcher) ?:
                 "batched ANE expert dispatch failed");
    return valid;
}

static double elapsedMilliseconds(uint64_t start, uint64_t end) {
    mach_timebase_info_data_t info = {0};
    mach_timebase_info(&info);
    return (double)(end - start) * info.numer / info.denom / 1e6;
}

static bool isGemmaStopToken(uint32_t token) {
    // Installed generation_config.json: eos_token_id = [1, 106, 50].
    // 106 is <turn|> (the normal chat end-of-turn marker); 50 closes a tool
    // response. None should be fed back through the decoder or exposed as text.
    return token == 1 || token == 106 || token == 50;
}

static void emit(NSDictionary *receipt) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(data.bytes, 1, data.length, stdout);
    fputc('\n', stdout);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2 || argc > 4) {
            fprintf(stderr, "usage: ane-fieldfare-generate MODEL_DIR [PROMPT] [MAX_NEW_TOKENS]\n");
            return 2;
        }
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        const char *prompt = argc >= 3 ? argv[2] : "The capital of France is";
        unsigned long requested = argc == 4 ? strtoul(argv[3], NULL, 10) : 1;
        // This is an end-to-end generator, not the old 64-token gate.  Keep a
        // bounded CLI admission for allocation arithmetic, but allow the
        // production 1K-token quality rail and longer continuations.
        if (!requested || requested > 4096 || requested > UINT32_MAX) return 2;
        uint32_t maximumNewTokens = (uint32_t)requested;

        NSString *weightsPath = [root stringByAppendingPathComponent:@"model_weights.bin"];
        NSString *tokenizerPath = [root stringByAppendingPathComponent:@"tokenizer/tokenizer.json"];
        NSString *firstLayerPath = [root stringByAppendingPathComponent:@"packed_experts/layer_00.bin"];
        NSString *executable = [NSString stringWithUTF8String:argv[0]];
        NSString *projectRoot = [[[executable stringByStandardizingPath]
            stringByDeletingLastPathComponent] stringByDeletingLastPathComponent];
        const char *metalOverride = getenv("ANE_FIELDFARE_METAL_SOURCE");
        NSString *metalSourcePath = metalOverride
            ? [NSString stringWithUTF8String:metalOverride]
            : [projectRoot stringByAppendingPathComponent:@"Sources/ANEFieldfareRuntime/Metal/moe.metal"];
        const char *backendValue = getenv("ANE_FIELDFARE_EXPERT_BACKEND");
        bool useMetal = !backendValue || strcmp(backendValue, "ane") != 0;
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor embedding = {0};
        ANEFieldfareTokenizer *tokenizer = NULL;
        ANEFieldfareDecoder *decoder = NULL;
        ExpertContext experts = {.modelDirectory = root.fileSystemRepresentation,
                                  .activeLayer = UINT32_MAX};
        MetalExpertContext metalExperts = {0};
        const char *prefetchMode = getenv("ANE_FIELDFARE_LAYER_PREFETCH");
        if (prefetchMode && !strcmp(prefetchMode, "previous")) experts.prefetchMode = 1;
        else if (prefetchMode && !strcmp(prefetchMode, "frequency")) experts.prefetchMode = 2;
        else if (prefetchMode && !strcmp(prefetchMode, "false")) experts.prefetchMode = 3;
        const char *topNValue = getenv("ANE_FIELDFARE_PREFETCH_TOP_N");
        experts.prefetchTopN = topNValue ? (uint32_t)strtoul(topNValue, NULL, 10) : 8;
        if (experts.prefetchTopN != 8 && experts.prefetchTopN != 16) experts.prefetchTopN = 8;
        ANEFieldfareTokenIDs promptTokens = {0};
        uint32_t *generated = calloc(maximumNewTokens, sizeof(*generated));
        float *hidden = calloc(ANEFieldfareGemma4HiddenSize, sizeof(*hidden));
        float *headLogits = calloc(ANEFieldfareGemmaVocabularySize, sizeof(*headLogits));
        bool rawPrompt = getenv("ANE_FIELDFARE_RAW_PROMPT") != NULL;
        char *framed = NULL;
        if (rawPrompt) {
            NSString *raw = [NSString stringWithFormat:@"<bos>%s", prompt];
            framed = strdup(raw.UTF8String);
        } else {
            framed = ANEFieldfareGemmaUserPrompt(prompt);
        }
        const char *creationError = NULL;

        bool valid = generated && hidden && headLogits && framed &&
            ANEFieldfareTensorIndexOpen(weightsPath.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.embed_tokens.weight", &embedding) &&
            (tokenizer = ANEFieldfareTokenizerOpen(tokenizerPath.fileSystemRepresentation)) &&
            ANEFieldfareTokenizerEncode(tokenizer, framed, &promptTokens) && promptTokens.count &&
            ANEFieldfareDecoderOpen(root.fileSystemRepresentation,
                                    promptTokens.count + maximumNewTokens, &decoder) &&
            (useMetal
                ? (metalExperts.dispatcher = ANEFieldfareMetalExpertDispatcherCreateModel(
                       root.fileSystemRepresentation,
                       metalSourcePath.fileSystemRepresentation,
                       &creationError)) != NULL
                : (experts.dispatcher = ANEFieldfareANEExpertDispatcherCreate(
                       firstLayerPath.fileSystemRepresentation, &creationError)) != NULL);

        struct rusage_info_v4 rusageBefore = {0}, rusageAfterPrefill = {0}, rusageAfter = {0};
        proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&rusageBefore);
        uint64_t footprintBefore = currentPhysicalFootprint();
        uint64_t start = mach_absolute_time();
        uint64_t prefillStart = start;
        uint64_t prefillEnd = start;
        double headMilliseconds = 0.0;
        double recurrentMilliseconds = 0.0;
        uint32_t emitted = 0;
        uint32_t stopToken = UINT32_MAX;
        NSString *stopReason = @"max_tokens";
        float winningLogit = 0.0f;
        for (size_t position = 0; valid && position < promptTokens.count; ++position) {
            valid = ANEFieldfareScaledEmbedding(&index, promptTokens.values[position], hidden) &&
                (useMetal
                    ? ANEFieldfareDecoderTokenFused(
                          decoder, position, hidden, dispatchMetalAttentionInput,
                          dispatchMetalAffine, &metalExperts,
                          dispatchMetalReduced, &metalExperts,
                          dispatchMetalShared, &metalExperts)
                    : ANEFieldfareDecoderTokenBatched(
                          decoder, position, hidden, dispatchExpertBatch, &experts));
        }
        prefillEnd = mach_absolute_time();
        proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&rusageAfterPrefill);
        uint64_t footprintAfterPrefill = currentPhysicalFootprint();
        for (uint32_t step = 0; valid && step < maximumNewTokens; ++step) {
            uint32_t token = 0;
            uint64_t headStart = mach_absolute_time();
            valid = useMetal
                ? decodeMetalGreedy(metalExperts.dispatcher, &index, &embedding, hidden,
                                    headLogits, &token, &winningLogit)
                : ANEFieldfareDecodeGreedy(&index, hidden, 256, &token, &winningLogit);
            headMilliseconds += elapsedMilliseconds(headStart, mach_absolute_time());
            if (!valid) break;
            if (isGemmaStopToken(token)) {
                stopToken = token;
                stopReason = @"stop_token";
                break;
            }
            generated[emitted++] = token;
            if (step + 1 == maximumNewTokens) break;
            uint64_t position = promptTokens.count + step;
            uint64_t recurrentStart = mach_absolute_time();
            valid = ANEFieldfareScaledEmbedding(&index, token, hidden) &&
                (useMetal
                    ? ANEFieldfareDecoderTokenFused(
                          decoder, position, hidden, dispatchMetalAttentionInput,
                          dispatchMetalAffine, &metalExperts,
                          dispatchMetalReduced, &metalExperts,
                          dispatchMetalShared, &metalExperts)
                    : ANEFieldfareDecoderTokenBatched(
                          decoder, position, hidden, dispatchExpertBatch, &experts));
            recurrentMilliseconds += elapsedMilliseconds(recurrentStart, mach_absolute_time());
        }
        uint64_t end = mach_absolute_time();
        if (!useMetal) ANEFieldfareANEExpertDispatcherWaitForLayerPrefetch(experts.dispatcher);
        proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&rusageAfter);
        uint64_t footprintAfterDecode = currentPhysicalFootprint();
        uint64_t prefillDiskBytesRead =
            rusageAfterPrefill.ri_diskio_bytesread >= rusageBefore.ri_diskio_bytesread
                ? rusageAfterPrefill.ri_diskio_bytesread - rusageBefore.ri_diskio_bytesread : 0;
        uint64_t decodeDiskBytesRead =
            rusageAfter.ri_diskio_bytesread >= rusageAfterPrefill.ri_diskio_bytesread
                ? rusageAfter.ri_diskio_bytesread - rusageAfterPrefill.ri_diskio_bytesread : 0;
        uint64_t diskBytesRead = rusageAfter.ri_diskio_bytesread >= rusageBefore.ri_diskio_bytesread
            ? rusageAfter.ri_diskio_bytesread - rusageBefore.ri_diskio_bytesread : 0;
        char *text = valid ? ANEFieldfareTokenizerDecode(tokenizer, generated, emitted, true) : NULL;
        valid = valid && text;
        ANEFieldfareANEExpertDispatchInfo info =
            ANEFieldfareANEExpertDispatcherGetInfo(experts.dispatcher);
        ANEFieldfareMetalExpertDispatchInfo metalInfo =
            ANEFieldfareMetalExpertDispatcherGetInfo(metalExperts.dispatcher);
        ANEFieldfareDecoderTiming decoderTiming = ANEFieldfareDecoderGetTiming(decoder);
        NSMutableArray *tokenArray = [NSMutableArray arrayWithCapacity:emitted];
        for (uint32_t indexValue = 0; indexValue < emitted; ++indexValue)
            [tokenArray addObject:@(generated[indexValue])];
        NSMutableArray *promptTokenArray = [NSMutableArray arrayWithCapacity:promptTokens.count];
        for (size_t indexValue = 0; indexValue < promptTokens.count; ++indexValue)
            [promptTokenArray addObject:@(promptTokens.values[indexValue])];
        NSString *failure = valid ? @"none" : [NSString stringWithUTF8String:
            (useMetal && metalExperts.error[0]) ? metalExperts.error :
            (experts.error[0] ? experts.error :
              (creationError ?: "generation pipeline failed"))];
        emit(@{
            @"schema": @"ane-fieldfare.generation.v1",
            @"status": valid ? @"accepted" : @"rejected",
            @"prompt": [NSString stringWithUTF8String:prompt],
            @"prompt_mode": rawPrompt ? @"bos_plus_raw_text" : @"gemma_user_chat_template",
            @"prompt_tokens": @(promptTokens.count),
            @"prompt_token_ids": promptTokenArray,
            @"generated_tokens": tokenArray,
            @"text": text ? [NSString stringWithUTF8String:text] : @"",
            @"stop_reason": stopReason,
            @"stop_token": stopToken == UINT32_MAX ? NSNull.null : @(stopToken),
            @"elapsed_ms": @(elapsedMilliseconds(start, end)),
            @"timing": @{
                @"prefill_ms": @(elapsedMilliseconds(prefillStart, prefillEnd)),
                @"head_ms": @(headMilliseconds),
                @"recurrent_decoder_ms": @(recurrentMilliseconds),
                @"decode_ms": @(headMilliseconds + recurrentMilliseconds),
                @"decode_tokens_per_second": headMilliseconds + recurrentMilliseconds > 0.0
                    ? @(1000.0 * emitted / (headMilliseconds + recurrentMilliseconds)) : @0,
                @"decoder_breakdown": @{
                    @"token_count": @(decoderTiming.tokenCount),
                    @"layer_count": @(decoderTiming.layerCount),
                    @"input_norm_ms": @(decoderTiming.inputNormalizationMilliseconds),
                    @"attention_ms": @(decoderTiming.attentionMilliseconds),
                    @"post_attention_ms": @(decoderTiming.postAttentionMilliseconds),
                    @"routed_ffn_ms": @(decoderTiming.routedFeedForwardMilliseconds)
                }
            },
            @"winning_softcapped_logit": @(winningLogit),
            @"expert_backend": useMetal ? @"metal_compact_int4" : @"private_ane_fp16_carrier",
            @"metal_compact": @{
                @"mapped_layers": @(metalInfo.mappedLayerCount),
                @"mapped_model_bytes": @(metalInfo.compactModelBytes),
                @"selected_bytes_per_dispatch": @(metalInfo.mappedExpertBytesPerDispatch),
                @"compact_staging_bytes": @(metalInfo.compactStagingBytes),
                @"compact_staging_ms": @(metalInfo.totalCompactStagingMilliseconds),
                @"exact_cache_capacity_bytes": @(metalInfo.exactCacheCapacityBytes),
                @"exact_cache_slots_per_layer": @(metalInfo.exactCacheSlotsPerLayer),
                @"exact_cache_hits": @(metalInfo.exactCacheHits),
            @"exact_cache_misses": @(metalInfo.exactCacheMisses),
                @"exact_cache_fill_ms": @(metalInfo.totalExactCacheFillMilliseconds),
                @"dispatches": @(metalInfo.dispatchCount),
                @"library_compile_count": @(metalInfo.libraryCompileCount),
                @"pipeline_load_count": @(metalInfo.pipelineLoadCount),
                @"split_queue_enabled": @(metalInfo.splitQueueEnabled),
                @"expert_cache_scope": @"fixed_per_layer",
                @"wall_ms": @(metalInfo.totalWallMilliseconds),
                @"gpu_ms": @(metalInfo.totalGPUMilliseconds),
                @"submission_and_wait_ms": @(
                    metalInfo.totalWallMilliseconds >= metalInfo.totalGPUMilliseconds
                        ? metalInfo.totalWallMilliseconds - metalInfo.totalGPUMilliseconds : 0.0),
                @"dense_dispatches": @(metalInfo.denseDispatchCount),
                @"dense_wall_ms": @(metalInfo.totalDenseWallMilliseconds),
                @"dense_gpu_ms": @(metalInfo.totalDenseGPUMilliseconds),
                @"dense_submission_and_wait_ms": @(
                    metalInfo.totalDenseWallMilliseconds >= metalInfo.totalDenseGPUMilliseconds
                        ? metalInfo.totalDenseWallMilliseconds - metalInfo.totalDenseGPUMilliseconds : 0.0),
                @"timer_semantics": @"Cumulative synchronous command latencies. GPU time is Metal command GPUStartTime..GPUEndTime; wall minus GPU includes CPU submission, queueing, scheduling, and completion wait. Branches may overlap, so these sums are not additive to phase wall time.",
                @"dense_breakdown": @{
                    @"attention_input": @{
                        @"dispatches": @(metalInfo.attentionInputDispatchCount),
                        @"wall_ms": @(metalInfo.attentionInputWallMilliseconds),
                        @"gpu_ms": @(metalInfo.attentionInputGPUMilliseconds)
                    },
                    @"attention_output": @{
                        @"dispatches": @(metalInfo.attentionOutputDispatchCount),
                        @"wall_ms": @(metalInfo.attentionOutputWallMilliseconds),
                        @"gpu_ms": @(metalInfo.attentionOutputGPUMilliseconds)
                    },
                    @"shared_ffn": @{
                        @"dispatches": @(metalInfo.sharedFFNDispatchCount),
                        @"wall_ms": @(metalInfo.sharedFFNWallMilliseconds),
                        @"gpu_ms": @(metalInfo.sharedFFNGPUMilliseconds)
                    },
                    @"head": @{
                        @"dispatches": @(metalInfo.headDispatchCount),
                        @"wall_ms": @(metalInfo.headWallMilliseconds),
                        @"gpu_ms": @(metalInfo.headGPUMilliseconds)
                    }
                },
                @"fp16_weight_carrier_bytes": @0
            },
            @"private_ane": @{
                @"compile_count": experts.dispatcher ? @1 : @0,
                @"load_count": experts.dispatcher ? @1 : @0,
                @"request_banks": @(info.program.requestBankCount),
                @"expert_dispatches": @(info.dispatchCount),
                @"compile_ms": @(info.program.compileMilliseconds),
                @"load_ms": @(info.program.loadMilliseconds),
                @"preparation_ms": @(info.totalPreparationMilliseconds),
                @"evaluation_ms": @(info.totalEvaluationMilliseconds),
                @"explicit_read_bytes": @(info.explicitReadBytes),
                @"layer_prefetch": @{
                    @"mode": experts.prefetchMode == 1 ? @"previous_two_token_union" :
                        (experts.prefetchMode == 2 ? @"frequency_top_n" :
                         (experts.prefetchMode == 3 ? @"fixed_false_prediction" :
                         (getenv("ANE_FIELDFARE_PREFETCH_EXPERTS") ? @"immediate_same_layer" : @"disabled"))),
                    @"top_n": @(experts.prefetchTopN),
                    @"scheduled_layers": @(experts.predictionScheduled),
                    @"predicted_experts": @(experts.predictionExperts),
                    @"actual_route_hits": @(experts.predictionHits),
                    @"actual_route_opportunities": @(experts.predictionOpportunities),
                    @"hit_rate": experts.predictionOpportunities
                        ? @((double)experts.predictionHits / experts.predictionOpportunities) : @0,
                    @"runtime_requests": @(info.layerPrefetchRequests),
                    @"runtime_expert_hints": @(info.layerPrefetchExpertHints),
                    @"hint_bytes": @(info.layerPrefetchHintBytes),
                    @"hint_failures": @(info.layerPrefetchFailures),
                    @"boundary_wait_ms": @(info.layerPrefetchWaitMilliseconds)
                }
            },
            @"disk_io": @{
                @"bytes_read": @(diskBytesRead),
                @"prefill_bytes_read": @(prefillDiskBytesRead),
                @"decode_bytes_read": @(decodeDiskBytesRead),
                @"mb_per_second": elapsedMilliseconds(start, end) > 0
                    ? @((double)diskBytesRead / elapsedMilliseconds(start, end) / 1000.0) : @0,
                @"oracle": @"proc_pid_rusage RUSAGE_INFO_V4; process physical disk reads only; mapped-file page-cache hits are not counted"
            },
            @"memory": @{
                @"physical_footprint_before_timed_work_bytes":
                    footprintBefore == UINT64_MAX ? NSNull.null : @(footprintBefore),
                @"physical_footprint_after_prefill_bytes":
                    footprintAfterPrefill == UINT64_MAX ? NSNull.null : @(footprintAfterPrefill),
                @"physical_footprint_after_decode_bytes":
                    footprintAfterDecode == UINT64_MAX ? NSNull.null : @(footprintAfterDecode),
                @"oracle": @"TASK_VM_INFO phys_footprint snapshots; process accounting, not engine residency"
            },
            @"error": failure,
            @"claim_boundary": valid
                ? (useMetal
                    ? @"All prompt tokens traversed the authentic 30-layer checkpoint in order; attention and KV state, compact INT4 Metal router-selected experts, shared FFNs, norms, residuals, tied softcapped logits, greedy selection, and detokenization executed together."
                    : @"All prompt tokens traversed the authentic 30-layer checkpoint in order; attention and KV state, private-ANE router-selected experts, shared FFNs, norms, residuals, tied softcapped logits, greedy selection, and detokenization executed together.")
                : @"Generation did not complete; no token is claimed."
        });

        free((void *)creationError);
        ANEFieldfareMetalExpertDispatcherDestroy(metalExperts.dispatcher);
        free(text);
        free(framed);
        free(hidden);
        free(headLogits);
        free(generated);
        ANEFieldfareTokenIDsFree(&promptTokens);
        ANEFieldfareANEExpertDispatcherDestroy(experts.dispatcher);
        ANEFieldfareDecoderClose(decoder);
        ANEFieldfareTokenizerClose(tokenizer);
        ANEFieldfareTensorIndexClose(&index);
        return valid ? 0 : 1;
    }
}
