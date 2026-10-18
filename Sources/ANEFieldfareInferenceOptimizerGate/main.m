#import <Foundation/Foundation.h>
#import "../ANEFieldfareRuntime/InferenceOptimizer.h"

static void fail(NSString *message) {
    fprintf(stderr, "%s\n", message.UTF8String);
    exit(1);
}

static ANEFieldfareInferenceOptimizer *syntheticOptimizer(
    NSString *name, NSDictionary *architecture) {
    NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"ane-fieldfare-%@-%@", name, NSUUID.UUID.UUIDString]];
    NSError *error = nil;
    if (![NSFileManager.defaultManager createDirectoryAtPath:directory
                                withIntermediateDirectories:YES attributes:nil error:&error])
        fail(error.localizedDescription);
    NSData *json = [NSJSONSerialization dataWithJSONObject:@{@"arch": architecture}
                                                   options:0 error:&error];
    if (!json || ![json writeToFile:[directory stringByAppendingPathComponent:@"manifest.json"]
                            options:NSDataWritingAtomic error:&error])
        fail(error.localizedDescription);
    NSString *optimizerError = nil;
    ANEFieldfareInferenceOptimizer *optimizer =
        [ANEFieldfareInferenceOptimizer optimizerForModelDirectory:directory error:&optimizerError];
    [NSFileManager.defaultManager removeItemAtPath:directory error:nil];
    if (!optimizer) fail(optimizerError ?: @"synthetic optimizer construction failed");
    return optimizer;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        NSString *error = nil;
        ANEFieldfareInferenceOptimizer *optimizer =
            [ANEFieldfareInferenceOptimizer optimizerForModelDirectory:
                [NSString stringWithUTF8String:argv[1]] error:&error];
        if (!optimizer) fail(error ?: @"optimizer construction failed");
        NSDictionary *capabilities = @{
            @"max_position_carrier": @23,
            @"resident_expert_slots": @7,
            @"expert_weight_state_width": @4,
            @"resident_projection_workspace_bytes": @(11ull << 20),
            @"device_attention_prefill": @YES,
            @"weight_state_transport": @YES,
            @"native_route_verification": @YES,
            @"provenance": @"synthetic-capability-contract"
        };
        NSDictionary *overlappedCosts = @{
            @"weight_state_transport_ms": @3.0, @"receiver_lead_time_ms": @4.0
        };
        NSDictionary *plan = [optimizer candidatePlanForPositionCount:97
            availableMemoryBytes:16ull << 30 backendCapabilities:capabilities
            costObservations:overlappedCosts];
        NSDictionary *schedule = plan[@"schedule"];
        if (![schedule[@"group_routes_by_weight_identity"] boolValue] ||
            ![schedule[@"batch_shared_ffn"] boolValue] ||
            ![schedule[@"device_attention"] boolValue] ||
            [schedule[@"position_tile_capacity"] unsignedIntegerValue] != 23 ||
            [schedule[@"expert_cache_slots_per_layer"] unsignedIntegerValue] != 7 ||
            [schedule[@"expert_weight_state_width"] unsignedIntegerValue] != 4 ||
            ![schedule[@"receiver_continuation_prefetch"] boolValue] ||
            ![schedule[@"native_route_verification"] boolValue])
            fail(@"capability-lowered candidate is incomplete");

        NSDictionary *serializedCosts = @{
            @"weight_state_transport_ms": @5.0, @"receiver_lead_time_ms": @4.0
        };
        NSDictionary *serialized = [optimizer candidatePlanForPositionCount:97
            availableMemoryBytes:16ull << 30 backendCapabilities:capabilities
            costObservations:serializedCosts];
        if ([serialized[@"schedule"][@"receiver_continuation_prefetch"] boolValue])
            fail(@"transport exceeding measured lead time was incorrectly overlapped");

        NSDictionary *common = @{
            @"numLayers": @32, @"moeIntermediateSize": @1024,
            @"ffnIntermediate": @4096, @"vocabSize": @128000,
            @"numHeads": @32, @"numKVHeads": @8, @"headDim": @128
        };
        NSMutableDictionary *denseArch = [common mutableCopy];
        [denseArch addEntriesFromDictionary:@{
            @"hiddenSize": @4096, @"numExperts": @1, @"topKExperts": @1
        }];
        ANEFieldfareInferenceOptimizer *dense = syntheticOptimizer(@"dense", denseArch);
        NSDictionary *densePlan = [dense candidatePlanForPositionCount:32
            availableMemoryBytes:8ull << 30 backendCapabilities:capabilities
            costObservations:overlappedCosts];
        if ([densePlan[@"schedule"][@"group_routes_by_weight_identity"] boolValue] ||
            [densePlan[@"schedule"][@"expert_cache_slots_per_layer"] unsignedIntegerValue] != 0)
            fail(@"dense architecture incorrectly received an MoE schedule");

        NSMutableDictionary *moeArch = [common mutableCopy];
        [moeArch addEntriesFromDictionary:@{
            @"hiddenSize": @4096, @"numExperts": @64, @"topKExperts": @2
        }];
        ANEFieldfareInferenceOptimizer *moe = syntheticOptimizer(@"moe", moeArch);
        NSDictionary *moePlan = [moe candidatePlanForPositionCount:32
            availableMemoryBytes:8ull << 30 backendCapabilities:capabilities
            costObservations:overlappedCosts];
        if (![moePlan[@"schedule"][@"group_routes_by_weight_identity"] boolValue] ||
            [moePlan[@"schedule"][@"expert_cache_slots_per_layer"] unsignedIntegerValue] != 7 ||
            [moe.architectureSignature isEqual:optimizer.architectureSignature])
            fail(@"alternate MoE architecture did not receive a distinct grouped schedule");

        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.inference-optimizer-gate.v1",
            @"status": @"validated",
            @"architecture": optimizer.architecture,
            @"architecture_signature": optimizer.architectureSignature,
            @"candidate": plan,
            @"synthetic_architectures_checked": @2,
            @"cross_shape_rules_validated": @YES,
            @"critical_path_branches_checked": @2,
            @"oracle": @"logical shape lowered through explicit backend capabilities and measured costs"
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
            options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);
    }
    return 0;
}
