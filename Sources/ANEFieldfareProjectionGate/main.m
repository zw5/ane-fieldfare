#import <Foundation/Foundation.h>

#include <math.h>

#include "../ANEFieldfareRuntime/ANEDynamicProjection.h"

static void emit(NSDictionary *value) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static BOOL rawBF16(ANEFieldfareTensor tensor, NSUInteger count, float *output) {
    if (tensor.weightBytes != count * 2 || tensor.scaleBytes || tensor.biasBytes) return NO;
    for (NSUInteger index = 0; index < count; ++index)
        output[index] = ANEFieldfareBF16(tensor.weights + index * 2);
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        enum { Hidden = 2816, Experts = 128 };
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *path = [root stringByAppendingPathComponent:@"model_weights.bin"];
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor embedding = {0}, router = {0}, scale = {0};
        float *hidden = calloc(Hidden, sizeof(float));
        float *routerInput = calloc(Hidden, sizeof(float));
        float *routerScale = calloc(Hidden, sizeof(float));
        float cpu[Experts] = {0}, ane[Experts] = {0};
        BOOL valid = ANEFieldfareTensorIndexOpen(path.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.embed_tokens.weight", &embedding) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.layers.0.router.proj.weight", &router) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.layers.0.router.scale", &scale) &&
            ANEFieldfareAffineRow(&embedding, 4, 64, 2, hidden) && rawBF16(scale, Hidden, routerScale);
        if (valid) {
            double square = 0;
            for (NSUInteger i = 0; i < Hidden; ++i) square += (double)hidden[i] * hidden[i];
            float factor = 1.0f / sqrtf((float)(square / Hidden) + 1e-6f) / sqrtf(Hidden);
            for (NSUInteger i = 0; i < Hidden; ++i) routerInput[i] = hidden[i] * routerScale[i] * factor;
            valid = ANEFieldfareAffineMatVec(&router, 8, 64, routerInput, cpu);
        }
        const char *error = NULL;
        ANEFieldfareANEDynamicProjection *projection = valid
            ? ANEFieldfareANEDynamicProjectionCreate(Hidden, Experts, 1, &error) : NULL;
        double preparationMs = 0;
        ANEFieldfareANEEvaluationTiming timing = {0};
        valid = projection && ANEFieldfareANEDynamicProjectionEvaluate(
            projection, 0, routerInput, &router, 8, 64, ane,
            &preparationMs, &timing, &error);
        NSUInteger mismatches = 0;
        float maximumError = 0;
        for (NSUInteger i = 0; valid && i < Experts; ++i) {
            float difference = fabsf(cpu[i] - ane[i]);
            maximumError = fmaxf(maximumError, difference);
            if (difference > 0.03f + 0.02f * fabsf(cpu[i])) mismatches++;
        }
        valid = valid && mismatches == 0;
        ANEFieldfareANEProgramInfo info = ANEFieldfareANEDynamicProjectionGetInfo(projection);
        emit(@{
            @"schema": @"ane-fieldfare.dynamic-projection.v1",
            @"status": valid ? @"validated" : @"rejected",
            @"stage": valid ? @"router_projection_oracle" : @"projection",
            @"shape": @{@"input": @(Hidden), @"output": @(Experts)},
            @"mismatches": @(mismatches), @"max_absolute_error": @(maximumError),
            @"preparation_ms": @(preparationMs),
            @"enqueue_ms": @(timing.enqueueMilliseconds),
            @"completion_ms": @(timing.completionMilliseconds),
            @"compile_ms": @(info.compileMilliseconds), @"load_ms": @(info.loadMilliseconds),
            @"queue_depth": @(info.reportedQueueDepth),
            @"error": error ? [NSString stringWithUTF8String:error] : @"none",
            @"claim_boundary": @"Authentic layer-0 router projection executed through the reusable private-ANE dynamic projection adapter and matched an independent affine CPU oracle. This validates the generic projection backend at 2816 to 128, not a complete router-selected layer or token generation.",
        });
        ANEFieldfareANEDynamicProjectionDestroy(projection);
        ANEFieldfareTensorIndexClose(&index);
        free(hidden); free(routerInput); free(routerScale);
        return valid ? 0 : 1;
    }
}
