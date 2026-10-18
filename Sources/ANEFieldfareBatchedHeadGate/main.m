#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../ANEFieldfareRuntime/MetalExpertDispatch.h"
#include "../ANEFieldfareRuntime/TensorIndex.h"

enum { Positions = 8, Hidden = ANEFieldfareGemma4HiddenSize };

static double milliseconds(uint64_t ticks) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)ticks * timebase.numer / timebase.denom / 1e6;
}

static uint64_t footprint(void) {
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS
        ? info.phys_footprint : 0;
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
        options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:nil];
    fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *weights = [root stringByAppendingPathComponent:@"model_weights.bin"];
        NSString *metal = [[[NSString stringWithUTF8String:argv[0]] stringByStandardizingPath]
            stringByDeletingLastPathComponent];
        metal = [[metal stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"Sources/ANEFieldfareRuntime/Metal/moe.metal"];
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor embedding = {0};
        const char *error = NULL;
        ANEFieldfareMetalExpertDispatcher *dispatcher = NULL;
        float *inputs = calloc((size_t)Positions * Hidden, sizeof(float));
        bool valid = inputs && ANEFieldfareTensorIndexOpen(weights.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.embed_tokens.weight", &embedding) &&
            (dispatcher = ANEFieldfareMetalExpertDispatcherCreateModel(
                root.fileSystemRepresentation, metal.fileSystemRepresentation, &error));
        for (uint32_t p = 0; valid && p < Positions; ++p)
            for (uint32_t c = 0; c < Hidden; ++c)
                inputs[(uint64_t)p * Hidden + c] =
                    (float)((int32_t)((c * 17u + p * 31u) % 257u) - 128) / 256.0f;

        uint32_t serialTokens[Positions] = {0}, batchTokens[Positions] = {0};
        float serialValues[Positions] = {0}, batchValues[Positions] = {0};
        double serialBest = INFINITY, batchBest = INFINITY;
        if (valid) {
            for (uint32_t warm = 0; warm < 2; ++warm) {
                for (uint32_t p = 0; p < Positions; ++p)
                    valid = valid && ANEFieldfareMetalExpertDispatcherCallGreedyHead(
                        dispatcher, index.bytes, &embedding, inputs + (uint64_t)p * Hidden,
                        &serialTokens[p], &serialValues[p]);
                valid = valid && ANEFieldfareMetalExpertDispatcherCallGreedyHeadPositions(
                    dispatcher, index.bytes, &embedding, Positions, inputs,
                    batchTokens, batchValues);
            }
            for (uint32_t rep = 0; valid && rep < 5; ++rep) {
                uint64_t started = mach_absolute_time();
                for (uint32_t p = 0; valid && p < Positions; ++p)
                    valid = ANEFieldfareMetalExpertDispatcherCallGreedyHead(
                        dispatcher, index.bytes, &embedding, inputs + (uint64_t)p * Hidden,
                        &serialTokens[p], &serialValues[p]);
                serialBest = fmin(serialBest, milliseconds(mach_absolute_time() - started));
                started = mach_absolute_time();
                valid = valid && ANEFieldfareMetalExpertDispatcherCallGreedyHeadPositions(
                    dispatcher, index.bytes, &embedding, Positions, inputs,
                    batchTokens, batchValues);
                batchBest = fmin(batchBest, milliseconds(mach_absolute_time() - started));
            }
        }
        uint32_t mismatches = 0;
        double maximumError = 0.0;
        for (uint32_t p = 0; valid && p < Positions; ++p) {
            mismatches += serialTokens[p] != batchTokens[p];
            maximumError = fmax(maximumError, fabs((double)serialValues[p] - batchValues[p]));
        }
        valid = valid && !mismatches && maximumError == 0.0;
        ANEFieldfareMetalExpertDispatchInfo info =
            ANEFieldfareMetalExpertDispatcherGetInfo(dispatcher);
        emit(@{
            @"schema": @"ane-fieldfare.batched-greedy-head.v1",
            @"status": valid ? @"validated" : @"rejected",
            @"positions": @(Positions),
            @"serial_best_ms": @(serialBest),
            @"batched_best_ms": @(batchBest),
            @"speedup": @(serialBest / batchBest),
            @"token_mismatches": @(mismatches),
            @"maximum_raw_logit_error": @(maximumError),
            @"compile_count": @(info.libraryCompileCount),
            @"pipeline_load_count": @(info.pipelineLoadCount),
            @"physical_footprint_bytes": @(footprint()),
            @"error": @(error ?: ANEFieldfareMetalExpertDispatcherLastError(dispatcher) ?: "none"),
            @"claim_boundary": @"Eight independent hidden rows share one compact INT4 tied-head command and return only token IDs and raw maxima. Exact agreement is against eight serial GPU-reduced heads on the same authentic checkpoint. This validates the verification primitive, not a drafter, accepted speculative tokens, or end-to-end speedup."
        });
        free((void *)error); free(inputs);
        ANEFieldfareMetalExpertDispatcherDestroy(dispatcher);
        ANEFieldfareTensorIndexClose(&index);
        return valid ? 0 : 1;
    }
}
