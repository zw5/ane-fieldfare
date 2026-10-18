#import <Foundation/Foundation.h>

#include <math.h>
#include <stdlib.h>

#include "../ANEFieldfareRuntime/TensorIndex.h"

static void emit(NSDictionary *value) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static NSString *weightsPath(NSString *root) {
    return [root stringByAppendingPathComponent:@"model_weights.bin"];
}

static BOOL find(ANEFieldfareTensorIndex *index, NSString *name, ANEFieldfareTensor *tensor) {
    return ANEFieldfareTensorIndexFind(index, name.UTF8String, tensor);
}

static BOOL rawBF16Vector(ANEFieldfareTensor tensor, NSUInteger count, float *output) {
    if (tensor.weightBytes != count * 2 || tensor.scaleBytes || tensor.biasBytes) return NO;
    for (NSUInteger index = 0; index < count; ++index)
        output[index] = ANEFieldfareBF16(tensor.weights + index * 2);
    return YES;
}

typedef struct {
    uint32_t expert;
    float probability;
} RankedExpert;

static int compareExperts(const void *leftRaw, const void *rightRaw) {
    const RankedExpert *left = leftRaw, *right = rightRaw;
    if (left->probability > right->probability) return -1;
    if (left->probability < right->probability) return 1;
    return left->expert < right->expert ? -1 : left->expert > right->expert;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2 || argc > 3) {
            emit(@{@"schema": @"ane-fieldfare.router-gate.v1", @"status": @"rejected",
                   @"stage": @"arguments", @"usage": @"ane-fieldfare-router-gate MODEL_DIR [TOKEN_ID]"});
            return 2;
        }
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        uint32_t token = argc == 3 ? (uint32_t)strtoul(argv[2], NULL, 10) : 2;
        enum { Hidden = 2816, Experts = 128, TopK = 8, GroupSize = 64 };
        if (token >= 262144) {
            emit(@{@"schema": @"ane-fieldfare.router-gate.v1", @"status": @"rejected", @"stage": @"token"});
            return 2;
        }

        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        if (!ANEFieldfareTensorIndexOpen(weightsPath(root).fileSystemRepresentation, &index)) {
            emit(@{@"schema": @"ane-fieldfare.router-gate.v1", @"status": @"rejected", @"stage": @"tensor_index"});
            return 1;
        }
        ANEFieldfareTensor embedding = {0}, projection = {0}, scale = {0}, perExpertScale = {0};
        BOOL tensorsPresent =
            find(&index, @"language_model.model.embed_tokens.weight", &embedding) &&
            find(&index, @"language_model.model.layers.0.router.proj.weight", &projection) &&
            find(&index, @"language_model.model.layers.0.router.scale", &scale) &&
            find(&index, @"language_model.model.layers.0.router.per_expert_scale", &perExpertScale);
        float *hidden = calloc(Hidden, sizeof(float));
        float *routerInput = calloc(Hidden, sizeof(float));
        float *routerScale = calloc(Hidden, sizeof(float));
        float logits[Experts] = {0}, expertScale[Experts] = {0};
        BOOL valid = tensorsPresent && hidden && routerInput && routerScale &&
            embedding.rows == 262144 && embedding.columns == Hidden &&
            projection.rows == Experts && projection.columns == Hidden &&
            ANEFieldfareAffineRow(&embedding, 4, GroupSize, token, hidden) &&
            rawBF16Vector(scale, Hidden, routerScale) &&
            rawBF16Vector(perExpertScale, Experts, expertScale);
        double meanSquare = 0;
        if (valid) {
            for (NSUInteger column = 0; column < Hidden; ++column)
                meanSquare += (double)hidden[column] * hidden[column];
            meanSquare /= Hidden;
            float normalization = 1.0f / sqrtf((float)meanSquare + 1e-6f);
            float scalarRootSize = 1.0f / sqrtf((float)Hidden);
            for (NSUInteger column = 0; column < Hidden; ++column)
                routerInput[column] = hidden[column] * normalization * routerScale[column] * scalarRootSize;
            valid = ANEFieldfareAffineMatVec(&projection, 8, GroupSize, routerInput, logits);
        }

        RankedExpert ranking[Experts] = {0};
        float maximum = -INFINITY, denominator = 0;
        if (valid) {
            for (NSUInteger expert = 0; expert < Experts; ++expert) maximum = fmaxf(maximum, logits[expert]);
            for (NSUInteger expert = 0; expert < Experts; ++expert) denominator += expf(logits[expert] - maximum);
            valid = isfinite(denominator) && denominator > 0;
            for (NSUInteger expert = 0; valid && expert < Experts; ++expert)
                ranking[expert] = (RankedExpert){(uint32_t)expert, expf(logits[expert] - maximum) / denominator};
            qsort(ranking, Experts, sizeof(ranking[0]), compareExperts);
        }

        NSMutableArray *selected = [NSMutableArray arrayWithCapacity:TopK];
        float selectedProbability = 0;
        for (NSUInteger rank = 0; valid && rank < TopK; ++rank) selectedProbability += ranking[rank].probability;
        for (NSUInteger rank = 0; valid && rank < TopK; ++rank) {
            uint32_t expert = ranking[rank].expert;
            float normalized = ranking[rank].probability / selectedProbability;
            [selected addObject:@{@"rank": @(rank), @"expert": @(expert),
                                  @"probability": @(ranking[rank].probability),
                                  @"normalized_top_k_weight": @(normalized),
                                  @"per_expert_scale": @(expertScale[expert]),
                                  @"routed_weight": @(normalized * expertScale[expert])}];
        }
        emit(@{
            @"schema": @"ane-fieldfare.router-gate.v1",
            @"status": valid ? @"validated" : @"rejected",
            @"stage": valid ? @"deterministic_top8" : @"tensor_contract",
            @"token_id": @(token),
            @"tensor_count": @(index.tensorCount),
            @"tensor_contract": @{
                @"present": @(tensorsPresent),
                @"embedding": @{@"rows": @(embedding.rows), @"columns": @(embedding.columns),
                                  @"weights": @(embedding.weightBytes), @"scales": @(embedding.scaleBytes), @"biases": @(embedding.biasBytes)},
                @"projection": @{@"rows": @(projection.rows), @"columns": @(projection.columns),
                                   @"weights": @(projection.weightBytes), @"scales": @(projection.scaleBytes), @"biases": @(projection.biasBytes)},
                @"scale_bytes": @(scale.weightBytes),
                @"per_expert_scale_bytes": @(perExpertScale.weightBytes),
            },
            @"shape": @{@"hidden": @(Hidden), @"experts": @(Experts), @"top_k": @(TopK)},
            @"router_input": @"checkpoint embedding row, router internal RMS normalization without learned norm scale",
            @"selected": selected,
            @"claim_boundary": @"Parses the resident installation tensor index and computes the exact Gemma 4 router formula from authentic quantized embedding, router projection, router scale, and per-expert scales. The input is a checkpoint embedding row used as a deterministic integration fixture, not the post-attention layer-0 residual; routed expert execution and a complete layer are not yet claimed.",
        });
        free(hidden);
        free(routerInput);
        free(routerScale);
        ANEFieldfareTensorIndexClose(&index);
        return valid ? 0 : 1;
    }
}
