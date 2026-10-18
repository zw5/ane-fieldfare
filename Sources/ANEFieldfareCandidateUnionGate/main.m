#import <Foundation/Foundation.h>

#include <math.h>
#include <sys/stat.h>

static NSDictionary *readJSON(NSString *path) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    return data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
}

static void emit(NSDictionary *value) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(data.bytes, 1, data.length, stdout);
    fputc('\n', stdout);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSString *root = NSFileManager.defaultManager.currentDirectoryPath;
        NSString *model = argc > 1 ? [@(argv[1]) stringByStandardizingPath]
                                   : [root stringByAppendingPathComponent:
                                       @"checkpoints/gturbo/gemma4.gturbo"];
        NSDictionary *route = readJSON([root stringByAppendingPathComponent:
            @"results/route-overlap-m5-macos27.json"]);
        NSDictionary *hotpath = readJSON([root stringByAppendingPathComponent:
            @"results/hotpath-optimizer-m5-macos27.json"]);
        NSDictionary *ramp = readJSON([root stringByAppendingPathComponent:
            @"results/sustained-ramp-m5-macos27.json"]);
        NSDictionary *layout = readJSON([root stringByAppendingPathComponent:
            @"results/expert-layout-a4-m5-macos27.json"]);

        const NSUInteger layers = [route[@"trace"][@"layers"] unsignedIntegerValue];
        const NSUInteger topK = [route[@"trace"][@"top_k"] unsignedIntegerValue];
        const NSUInteger traceTokens = [route[@"trace"][@"tokens"] unsignedIntegerValue];
        const NSUInteger distinctUsed = [route[@"trace"][@"distinct_experts_used"] unsignedIntegerValue];
        const uint64_t stride = [layout[@"fixture"][@"expert_stride_bytes"] unsignedLongLongValue];
        const double ioGBs = [hotpath[@"representative_sustained_owned_profile"]
            [@"effective_page_cache_gigabytes_per_second"] doubleValue];
        const double evaluationMS = [ramp[@"steady_state"]
            [@"ane_request_to_completion_median_ms"] doubleValue];
        const double preparationMS = [ramp[@"steady_state"]
            [@"host_preparation_median_ms"] doubleValue];
        const NSUInteger measuredPositions = [ramp[@"positions"] unsignedIntegerValue];

        BOOL filesValid = layers == 30 && topK == 8 && traceTokens == 68 && stride > 0;
        NSMutableArray *layerFiles = [NSMutableArray array];
        for (NSUInteger layer = 0; layer < layers; ++layer) {
            NSString *path = [model stringByAppendingPathComponent:
                [NSString stringWithFormat:@"packed_experts/layer_%02lu.bin",
                                           (unsigned long)layer]];
            struct stat info = {0};
            BOOL present = stat(path.fileSystemRepresentation, &info) == 0;
            uint64_t expected = 128 * stride;
            BOOL exact = present && (uint64_t)info.st_size == expected;
            filesValid &= exact;
            [layerFiles addObject:@{@"layer": @(layer), @"bytes": present ? @((uint64_t)info.st_size) : @0,
                                    @"expected_bytes": @(expected), @"exact": @(exact)}];
        }

        NSMutableArray *windows = [NSMutableArray array];
        {
            const NSUInteger width = 1;
            const double unionSize = topK;
            const double unionBytes = layers * unionSize * (double)stride;
            const double bytesPerToken = unionBytes;
            const double callsPerToken = layers * unionSize;
            const double ioMS = bytesPerToken / (ioGBs * 1e9) * 1000.0;
            const double evalMS = callsPerToken * evaluationMS;
            const double prepMS = callsPerToken * preparationMS;
            const double overlapFloorMS = fmax(ioMS, evalMS);
            [windows addObject:@{
                @"candidate_positions": @(width), @"window_kind": @"exact scalar top-k identity",
                @"mean_expert_union_per_layer": @(unionSize),
                @"compact_union_bytes_per_window": @((uint64_t)llround(unionBytes)),
                @"compact_expert_bytes_per_accepted_token": @((uint64_t)llround(bytesPerToken)),
                @"fetch_amortization": @1, @"current_runtime_compact_io_ms_per_token": @(ioMS),
                @"measured_shape_ane_calls_per_token": @(callsPerToken),
                @"conditional_ane_evaluation_ms_per_token": @(evalMS),
                @"conditional_host_preparation_ms_per_token": @(prepMS),
                @"serial_expert_io_plus_evaluation_ms": @(ioMS + evalMS),
                @"perfect_io_compute_overlap_floor_ms": @(overlapFloorMS),
                @"perfect_overlap_expert_only_ceiling_tokens_per_second": @(1000.0 / overlapFloorMS),
                @"clears_100_tokens_per_second_expert_only": @((BOOL)(overlapFloorMS <= 10.0))
            }];
        }
        for (NSDictionary *entry in route[@"window_union"] ?: @[]) {
            NSUInteger width = [entry[@"k"] unsignedIntegerValue];
            double unionSize = [entry[@"mean_union"] doubleValue];
            double unionBytes = layers * unionSize * (double)stride;
            double bytesPerToken = unionBytes / width;
            double callsPerToken = layers * unionSize / width;
            double ioMS = bytesPerToken / (ioGBs * 1e9) * 1000.0;
            double evalMS = callsPerToken * evaluationMS;
            double prepMS = callsPerToken * preparationMS;
            double serialExpertMS = ioMS + evalMS;
            double overlapFloorMS = fmax(ioMS, evalMS);
            [windows addObject:@{
                @"candidate_positions": @(width), @"mean_expert_union_per_layer": @(unionSize),
                @"compact_union_bytes_per_window": @((uint64_t)llround(unionBytes)),
                @"compact_expert_bytes_per_accepted_token": @((uint64_t)llround(bytesPerToken)),
                @"fetch_amortization": @(width * topK / unionSize),
                @"current_runtime_compact_io_ms_per_token": @(ioMS),
                @"measured_shape_ane_calls_per_token": @(callsPerToken),
                @"conditional_ane_evaluation_ms_per_token": @(evalMS),
                @"conditional_host_preparation_ms_per_token": @(prepMS),
                @"serial_expert_io_plus_evaluation_ms": @(serialExpertMS),
                @"perfect_io_compute_overlap_floor_ms": @(overlapFloorMS),
                @"perfect_overlap_expert_only_ceiling_tokens_per_second": @(1000.0 / overlapFloorMS),
                @"clears_100_tokens_per_second_expert_only": @((BOOL)(overlapFloorMS <= 10.0))
            }];
        }

        double fullUnion = (double)distinctUsed / layers;
        double fullBytes = layers * fullUnion * (double)stride / traceTokens;
        double fullCalls = layers * fullUnion / traceTokens;
        double fullIO = fullBytes / (ioGBs * 1e9) * 1000.0;
        double fullEval = fullCalls * evaluationMS;
        double fullPrep = fullCalls * preparationMS;
        double fullOverlap = fmax(fullIO, fullEval);
        [windows addObject:@{
            @"candidate_positions": @(traceTokens),
            @"window_kind": @"entire retained accepted-token trace",
            @"mean_expert_union_per_layer": @(fullUnion),
            @"compact_union_bytes_per_window": @(distinctUsed * stride),
            @"compact_expert_bytes_per_accepted_token": @((uint64_t)llround(fullBytes)),
            @"fetch_amortization": @(traceTokens * topK / fullUnion),
            @"current_runtime_compact_io_ms_per_token": @(fullIO),
            @"measured_shape_ane_calls_per_token": @(fullCalls),
            @"conditional_ane_evaluation_ms_per_token": @(fullEval),
            @"conditional_host_preparation_ms_per_token": @(fullPrep),
            @"serial_expert_io_plus_evaluation_ms": @(fullIO + fullEval),
            @"perfect_io_compute_overlap_floor_ms": @(fullOverlap),
            @"perfect_overlap_expert_only_ceiling_tokens_per_second": @(1000.0 / fullOverlap),
            @"clears_100_tokens_per_second_expert_only": @((BOOL)(fullOverlap <= 10.0))
        }];

        double target100MaxUnionCompute = 10.0 * traceTokens / (layers * evaluationMS);
        double target100MaxUnionIO = 0.010 * ioGBs * 1e9 * traceTokens /
            (layers * (double)stride);
        double target150BudgetMS = 1000.0 / 150.0;
        double target150MaxUnionCompute = target150BudgetMS * traceTokens /
            (layers * evaluationMS);
        double target150MaxUnionIO = (target150BudgetMS / 1000.0) * ioGBs * 1e9 * traceTokens /
            (layers * (double)stride);
        BOOL validated = filesValid && ioGBs > 0 && evaluationMS > 0 && preparationMS > 0 &&
            windows.count == 7 && fullUnion > target100MaxUnionCompute &&
            fullUnion > target100MaxUnionIO && fullOverlap > 10.0;

        emit(@{
            @"schema": @"ane-fieldfare.candidate-union-law.v1",
            @"status": validated ? @"validated_negative" : @"rejected",
            @"stage": validated ? @"expert_union_100_tps_exclusion" : @"input_contract",
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"training_required": @NO, @"model_token_special_cases": @NO,
            @"authentic_shape": @{@"layers": @(layers), @"experts_per_layer": @128,
                @"top_k": @(topK), @"moe_intermediate": @704, @"hidden": @2816,
                @"compact_expert_stride_bytes": @(stride),
                @"ane_measurement_positions": @(measuredPositions)},
            @"compact_layer_files": @{@"count": @(layerFiles.count),
                @"all_exact_128_expert_stride": @(filesValid),
                @"per_layer_expected_bytes": @(128 * stride)},
            @"measured_service_rates": @{
                @"current_runtime_effective_compact_io_gb_per_second": @(ioGBs),
                @"ane_request_to_completion_ms_per_expert_at_64_positions": @(evaluationMS),
                @"host_preparation_ms_per_expert_at_64_positions": @(preparationMS),
                @"sources": @[@"results/hotpath-optimizer-m5-macos27.json",
                               @"results/sustained-ramp-m5-macos27.json",
                               @"results/expert-layout-a4-m5-macos27.json"]},
            @"law": @{
                @"compact_bytes_per_accepted_token": @"layers * U(W) * expert_stride / W",
                @"ane_evaluation_ms_per_accepted_token": @"layers * U(W) * measured_expert_evaluation_ms / W",
                @"perfect_overlap_expert_floor": @"max(compact_IO_ms, ANE_evaluation_ms)",
                @"best_case_assumptions": @[@"all candidate positions accepted",
                    @"each union expert fetched and evaluated exactly once",
                    @"perfect IO/ANE overlap", @"zero dense, router, reduction, verifier, and draft cost"]},
            @"windows": windows,
            @"full_trace_target_thresholds": @{
                @"observed_mean_union": @(fullUnion),
                @"max_union_for_100_tps_compute": @(target100MaxUnionCompute),
                @"max_union_for_100_tps_io": @(target100MaxUnionIO),
                @"max_union_for_150_tps_compute": @(target150MaxUnionCompute),
                @"max_union_for_150_tps_io": @(target150MaxUnionIO)},
            @"decision": @{
                @"routed_union_alone_reaches_100_tps": @NO,
                @"strongest_directly_measured_window": @32,
                @"directly_measured_expert_only_ceiling_tokens_per_second":
                    [windows[5] objectForKey:@"perfect_overlap_expert_only_ceiling_tokens_per_second"],
                @"full_trace_conditional_window": @(traceTokens),
                @"full_trace_conditional_expert_only_ceiling_tokens_per_second": @(1000.0 / fullOverlap),
                @"reason": @"Even treating all 68 traced tokens as accepted candidates, evaluating the observed expert union exceeds 10 ms/token before dense weights, routing, reduction, drafting, or verification."},
            @"claim_boundary": @"Exact algebra over the retained authentic 68-token top-8 routing trace, all 30 authentic compact expert files, the current runtime's measured effective compact-I/O service rate, and the measured 64-position authentic expert evaluation/preparation medians. It is an optimistic expert-only exclusion: candidate rejection and every nonexpert cost are set to zero. The W=68 compute term extrapolates the validated 64-position per-expert median by four positions and is conditional, but both the observed union threshold and every retained W<=32 window reject 100 tok/s. No candidate verifier, speculative acceptance, or end-to-end token speedup is claimed."
        });
        return validated ? 0 : 1;
    }
}
