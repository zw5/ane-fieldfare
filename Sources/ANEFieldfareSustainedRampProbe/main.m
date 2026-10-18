// Sustained-load ramp probe.
//
// Repeats one hot expert so storage is outside the timed question, then keeps
// host carrier preparation separate from request-to-completion ANE evaluation.
// The machine-readable result tests whether the ANE evaluation median improves
// by at least five percent between the first 100 and last 200 iterations. It
// deliberately does not extrapolate a multi-layer or 128-position prefill rate.

#import <Foundation/Foundation.h>

#include <mach/mach_time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../ANEFieldfareRuntime/ANEExpertDispatch.h"

enum {
    Hidden = ANEFieldfareGemma4HiddenSize,
    Intermediate = ANEFieldfareGemma4ExpertIntermediateSize,
};

static double milliseconds(uint64_t elapsed) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)elapsed * timebase.numer / timebase.denom / 1.0e6;
}

static int compareDouble(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static double percentile(const double *values, uint32_t count, double fraction) {
    if (!values || count == 0) return 0.0;
    double *copy = malloc((size_t)count * sizeof(double));
    if (!copy) return 0.0;
    memcpy(copy, values, (size_t)count * sizeof(double));
    qsort(copy, count, sizeof(double), compareDouble);
    uint32_t index = (uint32_t)floor(fraction * (double)(count - 1));
    double result = copy[index];
    free(copy);
    return result;
}

static NSDictionary *distribution(const double *values, uint32_t count) {
    double sum = 0.0, minimum = INFINITY, maximum = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        sum += values[i];
        if (values[i] < minimum) minimum = values[i];
        if (values[i] > maximum) maximum = values[i];
    }
    return @{
        @"minimum_ms": @(isfinite(minimum) ? minimum : 0.0),
        @"p10_ms": @(percentile(values, count, 0.10)),
        @"median_ms": @(percentile(values, count, 0.50)),
        @"mean_ms": @(count ? sum / count : 0.0),
        @"p90_ms": @(percentile(values, count, 0.90)),
        @"maximum_ms": @(maximum),
    };
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <model-directory> [iterations] [positions]\n", argv[0]);
            return 2;
        }
        uint32_t iterations = argc > 2 ? (uint32_t)atoi(argv[2]) : 1200;
        uint32_t positions = argc > 3 ? (uint32_t)atoi(argv[3]) : 64;
        const uint32_t expert = 7;
        if (iterations < 200 || positions == 0 ||
            positions > ANEFieldfareANEExpertPositionLanes) {
            fprintf(stderr, "iterations must be >= 200 and positions must be in 1..%u\n",
                    ANEFieldfareANEExpertPositionLanes);
            return 2;
        }

        NSString *layer = [NSString stringWithFormat:@"%s/packed_experts/layer_00.bin", argv[1]];
        const char *error = NULL;
        ANEFieldfareANEExpertDispatcher *dispatcher =
            ANEFieldfareANEExpertDispatcherCreate(layer.UTF8String, &error);
        if (!dispatcher) {
            fprintf(stderr, "dispatcher create failed: %s\n", error ?: "unknown");
            free((void *)error);
            return 1;
        }
        free((void *)error);

        size_t valueCount = (size_t)positions * Hidden;
        float *inputs = calloc(valueCount, sizeof(float));
        float *outputs = calloc(valueCount, sizeof(float));
        float *firstOutputs = calloc(valueCount, sizeof(float));
        double *wallSamples = calloc(iterations, sizeof(double));
        double *prepSamples = calloc(iterations, sizeof(double));
        double *evalSamples = calloc(iterations, sizeof(double));
        if (!inputs || !outputs || !firstOutputs || !wallSamples ||
            !prepSamples || !evalSamples) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }
        for (uint32_t p = 0; p < positions; ++p)
            for (uint32_t c = 0; c < Hidden; ++c)
                inputs[(size_t)p * Hidden + c] =
                    (float)((int32_t)((c + p * 31) % 97) - 48) / 64.0f;

        ANEFieldfareANEExpertDispatchInfo previous =
            ANEFieldfareANEExpertDispatcherGetInfo(dispatcher);
        bool valid = true;
        for (uint32_t i = 0; i < iterations; ++i) {
            uint64_t started = mach_absolute_time();
            valid = ANEFieldfareANEExpertDispatcherCallPositions(
                dispatcher, expert, positions, inputs, outputs);
            wallSamples[i] = milliseconds(mach_absolute_time() - started);
            ANEFieldfareANEExpertDispatchInfo now =
                ANEFieldfareANEExpertDispatcherGetInfo(dispatcher);
            prepSamples[i] = now.totalPreparationMilliseconds -
                             previous.totalPreparationMilliseconds;
            evalSamples[i] = now.totalEvaluationMilliseconds -
                             previous.totalEvaluationMilliseconds;
            previous = now;
            if (!valid) {
                fprintf(stderr, "dispatch %u failed: %s\n", i,
                        ANEFieldfareANEExpertDispatcherLastError(dispatcher) ?: "unknown");
                break;
            }
            if (i == 0) memcpy(firstOutputs, outputs, valueCount * sizeof(float));
        }
        if (!valid) return 1;

        bool outputsBitExact = memcmp(firstOutputs, outputs,
                                      valueCount * sizeof(float)) == 0;
        uint32_t firstCount = iterations < 100 ? iterations : 100;
        uint32_t tailCount = iterations < 200 ? iterations : 200;
        uint32_t tailStart = iterations - tailCount;
        double firstEvalMedian = percentile(evalSamples, firstCount, 0.50);
        double tailEvalMedian = percentile(evalSamples + tailStart, tailCount, 0.50);
        double firstPrepMedian = percentile(prepSamples, firstCount, 0.50);
        double tailPrepMedian = percentile(prepSamples + tailStart, tailCount, 0.50);
        double allPrepMedian = percentile(prepSamples, iterations, 0.50);
        double allEvalMedian = percentile(evalSamples, iterations, 0.50);
        double bestOf40 = percentile(wallSamples, iterations < 40 ? iterations : 40, 0.0);
        double tailBest = percentile(wallSamples + tailStart, tailCount, 0.0);
        double aneTailOverFirst = firstEvalMedian > 0.0 ?
            tailEvalMedian / firstEvalMedian : 0.0;
        bool materialANERamp = aneTailOverFirst < 0.95;
        uint64_t fp16WeightBytes =
            (uint64_t)Hidden * 3 * Intermediate * sizeof(__fp16);
        double effectiveGBs = allEvalMedian > 0.0 ?
            (double)fp16WeightBytes / (allEvalMedian * 1.0e6) : 0.0;

        NSMutableArray *buckets = [NSMutableArray array];
        uint32_t bucketWidth = iterations / 12;
        if (bucketWidth == 0) bucketWidth = 1;
        for (uint32_t base = 0; base < iterations; base += bucketWidth) {
            uint32_t count = bucketWidth;
            if (base + count > iterations) count = iterations - base;
            [buckets addObject:@{
                @"first_iteration": @(base),
                @"last_iteration": @(base + count - 1),
                @"wall": distribution(wallSamples + base, count),
                @"host_preparation": distribution(prepSamples + base, count),
                @"ane_request_to_completion": distribution(evalSamples + base, count),
            }];
        }

        ANEFieldfareANEExpertDispatchInfo finalInfo =
            ANEFieldfareANEExpertDispatcherGetInfo(dispatcher);
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.sustained-ramp.v1",
            @"status": outputsBitExact ? @"validated" : @"rejected",
            @"gate": @"C2.2",
            @"stage": @"sustained_hot_expert_dispatch",
            @"host": [NSHost currentHost].localizedName ?: @"unknown",
            @"os_version": [NSProcessInfo processInfo].operatingSystemVersionString,
            @"ane_architecture": @"not directly observed by this probe",
            @"exact_command": @"Scripts/with_benchmark_lock.sh .build/ane-fieldfare-transport-benchmarks sustained-ramp checkpoints/gturbo/gemma4.gturbo 1200 64 > results/sustained-ramp-m5-macos27.json",
            @"exit_status": @(outputsBitExact ? 0 : 1),
            @"compile_count": @1,
            @"load_count": @1,
            @"iterations": @(iterations),
            @"positions": @(positions),
            @"expert": @(expert),
            @"request_banks": @(finalInfo.program.requestBankCount),
            @"source_state": @"same compact expert throughout; pages hot after the first access",
            @"timing_semantics": @{
                @"host_preparation": @"fillCarrier wall time: compact source access, FP16 expansion, and carrier writes",
                @"ane_request_to_completion": @"evaluateWithQoS call start through completion-handler timestamp; excludes output-surface copy",
                @"dispatch_wall": @"complete CallPositions including preparation, ANE request, wait, and output-surface copy",
            },
            @"ramp_test": @{
                @"material_improvement_threshold": @0.05,
                @"first_100_ane_median_ms": @(firstEvalMedian),
                @"last_200_ane_median_ms": @(tailEvalMedian),
                @"last_over_first": @(aneTailOverFirst),
                @"material_ane_ramp_detected": @(materialANERamp),
                @"hypothesis_status": materialANERamp ? @"accepted_at_threshold" : @"rejected_at_threshold",
                @"first_100_host_preparation_median_ms": @(firstPrepMedian),
                @"last_200_host_preparation_median_ms": @(tailPrepMedian),
                @"best_of_first_40_dispatch_ms": @(bestOf40),
                @"best_of_last_200_dispatch_ms": @(tailBest),
            },
            @"steady_state": @{
                @"host_preparation_median_ms": @(allPrepMedian),
                @"ane_request_to_completion_median_ms": @(allEvalMedian),
                @"timed_stage_host_fraction": @(allPrepMedian / (allPrepMedian + allEvalMedian)),
                @"logical_fp16_weight_bytes": @(fp16WeightBytes),
                @"effective_logical_fp16_gb_per_second": @(effectiveGBs),
            },
            @"buckets": buckets,
            @"numerical_result": @{
                @"first_and_last_outputs_bit_exact": @(outputsBitExact),
                @"compared_float32_values": @(valueCount),
            },
            @"claim_boundary": @"For one layer-0 expert, 64 positions, one hot compact source mapping, and 1200 serial calls, separates host carrier preparation from request-to-completion ANE evaluation. Rejects an ANE sustained-load improvement of at least 5 percent on this fixture and measures the FP16 expert evaluation floor. It does not isolate on-device arithmetic from ANE carrier fetch, validate a 128-position expert program, determine a 128-token expert union, or project full-layer, full-prefill, decode, GPU-overlap, SSD, or token throughput.",
        };

        NSError *jsonError = nil;
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:&jsonError];
        if (!json || jsonError) {
            fprintf(stderr, "JSON serialization failed: %s\n",
                    jsonError.localizedDescription.UTF8String ?: "unknown");
            return 1;
        }
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);

        free(wallSamples);
        free(prepSamples);
        free(evalSamples);
        free(inputs);
        free(outputs);
        free(firstOutputs);
        ANEFieldfareANEExpertDispatcherDestroy(dispatcher);
        return outputsBitExact ? 0 : 1;
    }
}
