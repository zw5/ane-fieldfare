#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <mach/mach_time.h>

#include "../ANEFieldfareRuntime/ANEDynamicProjection.h"

static double milliseconds(uint64_t ticks) {
    mach_timebase_info_data_t info = {0};
    mach_timebase_info(&info);
    return (double)ticks * info.numer / info.denom / 1e6;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        enum { Hidden = 2816, Experts = 128, Banks = 8 };
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *path = [root stringByAppendingPathComponent:@"model_weights.bin"];
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor embedding = {0}, router = {0};
        float *input = calloc(Hidden, sizeof(float));
        float *outputs = calloc(Banks * Experts, sizeof(float));
        const char *creationError = NULL;
        BOOL valid = input && outputs &&
            ANEFieldfareTensorIndexOpen(path.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.embed_tokens.weight", &embedding) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.layers.0.router.proj.weight", &router) &&
            ANEFieldfareAffineRow(&embedding, 4, 64, 2, input);
        ANEFieldfareANEDynamicProjection *projection = valid
            ? ANEFieldfareANEDynamicProjectionCreate(Hidden, Experts, Banks, &creationError) : NULL;
        valid = valid && projection;
        NSMutableArray *preparation = [NSMutableArray array];
        for (uint32_t bank = 0; valid && bank < Banks; ++bank) {
            double preparationMs = 0;
            ANEFieldfareANEEvaluationTiming timing = {0};
            const char *error = NULL;
            valid = ANEFieldfareANEDynamicProjectionEvaluate(
                projection, bank, input, &router, 8, 64, outputs + bank * Experts,
                &preparationMs, &timing, &error);
            [preparation addObject:@{ @"bank": @(bank), @"preparation_ms": @(preparationMs),
                                      @"evaluation_ms": @(timing.completionMilliseconds),
                                      @"error": error ? [NSString stringWithUTF8String:error] : @"none" }];
            free((void *)error);
        }
        ANEFieldfareANEProgram *program = ANEFieldfareANEDynamicProjectionProgram(projection);
        uint64_t started = mach_absolute_time();
        for (uint32_t bank = 0; valid && bank < Banks; ++bank)
            valid = ANEFieldfareANEProgramEvaluate(program, bank, 5 * NSEC_PER_SEC, NULL, NULL);
        double serialMs = milliseconds(mach_absolute_time() - started);

        __block BOOL parallelValid = valid;
        started = mach_absolute_time();
        dispatch_apply(Banks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t bank) {
            if (!ANEFieldfareANEProgramEvaluate(program, (uint32_t)bank, 5 * NSEC_PER_SEC, NULL, NULL))
                parallelValid = NO;
        });
        double parallelMs = milliseconds(mach_absolute_time() - started);
        valid = valid && parallelValid;
        ANEFieldfareANEProgramInfo info = ANEFieldfareANEDynamicProjectionGetInfo(projection);
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.queue-gate.v1",
            @"status": valid ? @"accepted" : @"rejected",
            @"shape": @{ @"input": @(Hidden), @"output": @(Experts), @"banks": @(Banks) },
            @"reported_queue_depth": @(info.reportedQueueDepth),
            @"serial_eight_ms": @(serialMs),
            @"parallel_eight_ms": @(parallelMs),
            @"concurrency_speedup": parallelMs > 0 ? @(serialMs / parallelMs) : @0,
            @"initial_runs": preparation,
            @"compile_ms": @(info.compileMilliseconds), @"load_ms": @(info.loadMilliseconds),
            @"error": creationError ? [NSString stringWithUTF8String:creationError] : @"none",
            @"claim_boundary": @"Eight independently mapped requests for one authentic 2816-to-128 router projection were submitted concurrently to one loaded private-ANE program. Acceptance validates request concurrency; speedup is specific to this graph and host."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        free((void *)creationError);
        ANEFieldfareANEDynamicProjectionDestroy(projection);
        ANEFieldfareTensorIndexClose(&index);
        free(outputs); free(input);
        return valid ? 0 : 1;
    }
}
