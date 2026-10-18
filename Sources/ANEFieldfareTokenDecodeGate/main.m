#import <Foundation/Foundation.h>

#include <math.h>
#include <stdlib.h>

#include "../ANEFieldfareRuntime/Decode.h"
#include "../ANEFieldfareRuntime/Tokenizer.h"

static void emit(NSDictionary *value) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2 || argc > 3) {
            emit(@{@"schema": @"ane-fieldfare.token-decode-gate.v1", @"status": @"rejected",
                   @"stage": @"arguments", @"usage": @"ane-fieldfare-token-decode-gate MODEL_DIR [PROMPT]"});
            return 2;
        }
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *tokenizerPath = [root stringByAppendingPathComponent:@"tokenizer/tokenizer.json"];
        NSString *weightsPath = [root stringByAppendingPathComponent:@"model_weights.bin"];
        const char *user = argc == 3 ? argv[2] : "The capital of France is";
        ANEFieldfareTokenizer *tokenizer = ANEFieldfareTokenizerOpen(tokenizerPath.fileSystemRepresentation);
        char *framed = ANEFieldfareGemmaUserPrompt(user);
        ANEFieldfareTokenIDs ids = {0};
        bool encoded = tokenizer && framed && ANEFieldfareTokenizerEncode(tokenizer, framed, &ids);
        char *decoded = encoded ? ANEFieldfareTokenizerDecode(tokenizer, ids.values, ids.count, false) : NULL;
        bool roundTrip = decoded && strcmp(decoded, framed) == 0;

        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        bool opened = ANEFieldfareTensorIndexOpen(weightsPath.fileSystemRepresentation, &index);
        float embedding[ANEFieldfareGemmaHiddenSize] = {0};
        bool embedded = opened && encoded && ids.count &&
            ANEFieldfareScaledEmbedding(&index, ids.values[0], embedding);
        double squareSum = 0.0;
        for (uint32_t i = 0; embedded && i < ANEFieldfareGemmaHiddenSize; ++i)
            squareSum += (double)embedding[i] * embedding[i];
        float normalized[ANEFieldfareGemmaHiddenSize] = {0};
        memcpy(normalized, embedding, sizeof(normalized));
        bool finalNorm = embedded && ANEFieldfareFinalRMSNorm(&index, 1e-6f, normalized);
        bool scanHead = getenv("ANE_FIELDFARE_SCAN_HEAD") != NULL;
        uint32_t selectedToken = 0;
        float selectedLogit = 0;
        bool headScanned = scanHead && finalNorm &&
            ANEFieldfareTiedGreedy(&index, normalized, 256, NULL, &selectedToken, &selectedLogit);
        bool valid = tokenizer && ANEFieldfareTokenizerVocabularySize(tokenizer) == ANEFieldfareGemmaVocabularySize &&
            encoded && ids.count && roundTrip && embedded && finalNorm && isfinite(squareSum) &&
            (!scanHead || headScanned);

        NSMutableArray *preview = [NSMutableArray array];
        for (size_t i = 0; i < ids.count && i < 16; ++i) [preview addObject:@(ids.values[i])];
        emit(@{
            @"schema": @"ane-fieldfare.token-decode-gate.v1",
            @"status": valid ? @"validated" : @"rejected",
            @"stage": valid ? @"tokenizer_embedding_final_norm" : @"runtime_contract",
            @"vocabulary_size": @(tokenizer ? ANEFieldfareTokenizerVocabularySize(tokenizer) : 0),
            @"prompt_token_count": @(ids.count),
            @"prompt_token_preview": preview,
            @"prompt_round_trip": @(roundTrip),
            @"scaled_embedding_l2": @(sqrt(squareSum)),
            @"final_norm_present": @(finalNorm),
            @"head_scanned": @(headScanned),
            @"selected_token": headScanned ? @(selectedToken) : NSNull.null,
            @"selected_softcapped_logit": headScanned ? @(selectedLogit) : NSNull.null,
            @"head_contract": @{
                @"tied_embedding": @YES, @"rows_per_tile": @256,
                @"embedding_shape": @[@262144, @2816],
                @"embedding_quantization": @"affine 4-bit, group size 64, BF16 scale and bias",
                @"embedding_scale": @53.0,
                @"final_norm": @"direct learned multiplicative scale, epsilon 1e-6",
                @"softcap": @30, @"selection": @"deterministic greedy, lowest ID wins ties"
            },
            @"claim_boundary": headScanned
                ? @"The installed tokenizer/chat framing, authentic scaled tied embedding row, final Gemma RMSNorm, complete tiled tied head, softcap, and deterministic greedy selection execute through callable runtime APIs. The head fixture is an embedding row rather than a completed transformer hidden state, so no generated token is claimed."
                : @"The installed tokenizer/chat framing, authentic scaled tied embedding row, and final Gemma RMSNorm execute through callable runtime APIs. Set ANE_FIELDFARE_SCAN_HEAD=1 to scan the complete tiled tied head; no transformer layer or generated token is claimed."
        });
        if (opened) ANEFieldfareTensorIndexClose(&index);
        ANEFieldfareTokenIDsFree(&ids);
        ANEFieldfareTokenizerClose(tokenizer);
        free(framed);
        free(decoded);
        return valid ? 0 : 1;
    }
}
