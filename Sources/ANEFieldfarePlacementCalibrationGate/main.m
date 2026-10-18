#import <Foundation/Foundation.h>
#import <mach/mach_time.h>

#include <math.h>
#include <stdlib.h>

#include "../ANEFieldfareRuntime/ANEResidentProjection.h"
#include "../ANEFieldfareRuntime/Attention.h"
#include "../ANEFieldfareRuntime/MetalExpertDispatch.h"

static mach_timebase_info_data_t timebase;
static double ms(uint64_t ticks) {
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)ticks * timebase.numer / timebase.denom / 1e6;
}
static int compareDouble(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) { fprintf(stderr, "usage: %s <model-directory>\n", argv[0]); return 2; }
        NSString *model = @(argv[1]);
        NSString *weightsPath = [model stringByAppendingPathComponent:@"model_weights.bin"];
        NSString *metalPath = [@(__FILE__) stringByDeletingLastPathComponent];
        metalPath = [[metalPath stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:@"ANEFieldfareRuntime/Metal/moe.metal"];
        ANEFieldfareTensorIndex index = {0};
        ANEFieldfareAttentionWeights weights = {0};
        if (!ANEFieldfareTensorIndexOpen(weightsPath.fileSystemRepresentation, &index) ||
            !ANEFieldfareAttentionWeightsLoad(&index, 0, &weights)) {
            fprintf(stderr, "failed to open layer-0 attention weights\n"); return 1;
        }
        const char *error = NULL;
        ANEFieldfareANEResidentProjections *ane = ANEFieldfareANEResidentProjectionsCreate(
            &index, 1, 80ull * 1024 * 1024, &error);
        ANEFieldfareMetalExpertDispatcher *metal = ANEFieldfareMetalExpertDispatcherCreateModel(
            model.fileSystemRepresentation, metalPath.fileSystemRepresentation, &error);
        if (!ane || !metal) { fprintf(stderr, "backend creation failed: %s\n", error ?: "unknown"); return 1; }

        enum { Iterations = 7 };
        const uint32_t counts[] = {1, 8, 18, 64};
        const uint32_t hidden = weights.descriptor.hiddenSize;
        const uint32_t qRows = weights.queryProjection.rows, kRows = weights.keyProjection.rows;
        float *input = calloc((size_t)64 * hidden, sizeof(float));
        float *combined = calloc((size_t)64 * weights.outputProjection.columns, sizeof(float));
        float *aq = calloc((size_t)64 * qRows, sizeof(float));
        float *ak = calloc((size_t)64 * kRows, sizeof(float));
        float *av = calloc((size_t)64 * kRows, sizeof(float));
        float *ao = calloc((size_t)64 * hidden, sizeof(float));
        float *mq = calloc((size_t)64 * qRows, sizeof(float));
        float *mk = calloc((size_t)64 * kRows, sizeof(float));
        float *mv = calloc((size_t)64 * kRows, sizeof(float));
        float *mo = calloc((size_t)64 * hidden, sizeof(float));
        for (uint64_t i = 0; i < (uint64_t)64 * hidden; ++i)
            input[i] = (float)((int32_t)(i % 97) - 48) / 128.0f;
        for (uint64_t i = 0; i < (uint64_t)64 * weights.outputProjection.columns; ++i)
            combined[i] = (float)((int32_t)(i % 89) - 44) / 128.0f;

        NSMutableArray *arms = [NSMutableArray array];
        BOOL allValid = YES;
        double minimumSpeedup = INFINITY;
        for (NSUInteger c = 0; c < sizeof(counts) / sizeof(counts[0]); ++c) {
            uint32_t count = counts[c];
            // Warm both complete QKV+O paths; compile/load are already complete.
            for (uint32_t warm = 0; warm < 2; ++warm) {
                ANEFieldfareANEResidentProjectionsCallAttentionInputPositions(
                    ane, index.bytes, &weights.queryProjection, &weights.keyProjection,
                    &weights.valueProjection, weights.descriptor.valueEqualsKeyProjection,
                    count, input, aq, ak, av);
                ANEFieldfareANEResidentProjectionsCallAffinePositions(
                    ane, index.bytes, &weights.outputProjection, count, combined, ao);
                ANEFieldfareMetalExpertDispatcherCallAttentionInputPositions(
                    metal, index.bytes, &weights.queryProjection, &weights.keyProjection,
                    &weights.valueProjection, weights.descriptor.valueEqualsKeyProjection,
                    count, input, mq, mk, mv);
                ANEFieldfareMetalExpertDispatcherCallAffinePositions(
                    metal, index.bytes, &weights.outputProjection, count, combined, mo);
            }
            double aneTimes[Iterations], metalTimes[Iterations];
            BOOL valid = YES;
            for (uint32_t iteration = 0; iteration < Iterations; ++iteration) {
                for (uint32_t pass = 0; pass < 2; ++pass) {
                    bool runANE = ((iteration + pass) & 1u) == 0;
                    uint64_t start = mach_continuous_time();
                    if (runANE) {
                        valid &= ANEFieldfareANEResidentProjectionsCallAttentionInputPositions(
                            ane, index.bytes, &weights.queryProjection, &weights.keyProjection,
                            &weights.valueProjection, weights.descriptor.valueEqualsKeyProjection,
                            count, input, aq, ak, av);
                        valid &= ANEFieldfareANEResidentProjectionsCallAffinePositions(
                            ane, index.bytes, &weights.outputProjection, count, combined, ao);
                        aneTimes[iteration] = ms(mach_continuous_time() - start);
                    } else {
                        valid &= ANEFieldfareMetalExpertDispatcherCallAttentionInputPositions(
                            metal, index.bytes, &weights.queryProjection, &weights.keyProjection,
                            &weights.valueProjection, weights.descriptor.valueEqualsKeyProjection,
                            count, input, mq, mk, mv);
                        valid &= ANEFieldfareMetalExpertDispatcherCallAffinePositions(
                            metal, index.bytes, &weights.outputProjection, count, combined, mo);
                        metalTimes[iteration] = ms(mach_continuous_time() - start);
                    }
                }
            }
            qsort(aneTimes, Iterations, sizeof(double), compareDouble);
            qsort(metalTimes, Iterations, sizeof(double), compareDouble);
            double maximumError = 0;
            uint64_t comparisons = (uint64_t)count * (qRows + 2 * kRows + hidden);
            const float *left[] = {aq, ak, av, ao}, *right[] = {mq, mk, mv, mo};
            const uint64_t sizes[] = {(uint64_t)count*qRows, (uint64_t)count*kRows,
                                      (uint64_t)count*kRows, (uint64_t)count*hidden};
            for (uint32_t part = 0; part < 4; ++part)
                for (uint64_t i = 0; i < sizes[part]; ++i)
                    maximumError = fmax(maximumError, fabs((double)left[part][i] - right[part][i]));
            valid &= isfinite(maximumError) && maximumError < 0.02;
            allValid &= valid;
            double aneMedian = aneTimes[Iterations/2], metalMedian = metalTimes[Iterations/2];
            minimumSpeedup = fmin(minimumSpeedup, metalMedian / aneMedian);
            [arms addObject:@{@"positions": @(count), @"ane_median_ms": @(aneMedian),
                              @"metal_median_ms": @(metalMedian),
                              @"ane_speedup_over_metal": @(metalMedian / aneMedian),
                              @"ane_wins": @(aneMedian < metalMedian),
                              @"max_abs_cross_backend_error": @(maximumError),
                              @"comparisons": @(comparisons), @"iterations": @(Iterations)}];
        }
        ANEFieldfareANEResidentProjectionInfo aneInfo =
            ANEFieldfareANEResidentProjectionsGetInfo(ane);
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.ane-placement-calibration.v1",
            @"status": allValid ? @"validated" : @"rejected",
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"command": [NSString stringWithFormat:@".build/ane-fieldfare-placement-calibration-gate %@", model],
            @"layer": @0, @"arms": arms,
            @"timing_boundary": @"Median of seven warm, alternating-order paired calls. Each includes float-to-carrier preparation, QKV completion and readback, output-projection preparation, completion and readback. Compile/load excluded for both backends.",
            @"ane_lifecycle": @{@"compile_count": @(aneInfo.compileCount), @"load_count": @(aneInfo.loadCount),
                                 @"program_count": @(aneInfo.programCount), @"constant_bytes": @(aneInfo.constantBytes)},
            @"admission_gate": @{@"minimum_required_speedup": @1.05,
                                  @"minimum_observed_speedup": @(minimumSpeedup),
                                  @"all_widths_clear": @(allValid && minimumSpeedup > 1.05),
                                  @"decision": allValid && minimumSpeedup > 1.05
                                      ? @"passes isolated projection timing; require separate memory-displacement and end-to-end gate before changing default placement"
                                      : @"retain compact-Metal default"},
            @"placement_changed": @NO,
            @"claim_boundary": @"Layer 0 only, authentic compact checkpoint tensors and identical synthetic position-major inputs. Calibrates warm resident-ANE versus production compact-Metal QKV plus O projection at actual tile widths. It does not include attention/KV/softmax, compile/load, other layers, memory-displacement cost, or end-to-end generation."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        ANEFieldfareMetalExpertDispatcherDestroy(metal);
        ANEFieldfareANEResidentProjectionsDestroy(ane);
        ANEFieldfareTensorIndexClose(&index);
        free(input); free(combined); free(aq); free(ak); free(av); free(ao); free(mq); free(mk); free(mv); free(mo);
        return allValid ? 0 : 1;
    }
}
