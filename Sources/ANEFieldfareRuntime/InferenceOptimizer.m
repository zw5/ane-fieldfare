#import "InferenceOptimizer.h"

@interface ANEFieldfareInferenceOptimizer ()
@property(nonatomic, readwrite) NSDictionary<NSString *, NSNumber *> *architecture;
@property(nonatomic, readwrite) NSString *architectureSignature;
@end

@implementation ANEFieldfareInferenceOptimizer

static NSNumber *positiveNumber(NSDictionary *source, NSString *key) {
    id value = source[key];
    if (![value isKindOfClass:NSNumber.class] || [value unsignedLongLongValue] == 0) return nil;
    return value;
}

+ (instancetype)optimizerForModelDirectory:(NSString *)modelDirectory
                                      error:(NSString **)error {
    NSString *path = [modelDirectory stringByAppendingPathComponent:@"manifest.json"];
    NSData *data = [NSData dataWithContentsOfFile:path];
    NSError *jsonError = nil;
    id root = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError] : nil;
    NSDictionary *arch = [root isKindOfClass:NSDictionary.class] ? root[@"arch"] : nil;
    NSArray<NSString *> *required = @[
        @"hiddenSize", @"numLayers", @"numExperts", @"topKExperts",
        @"moeIntermediateSize", @"ffnIntermediate", @"vocabSize",
        @"numHeads", @"numKVHeads", @"headDim"
    ];
    NSMutableDictionary *normalized = [NSMutableDictionary dictionaryWithCapacity:required.count];
    if ([arch isKindOfClass:NSDictionary.class]) {
        for (NSString *key in required) {
            NSNumber *value = positiveNumber(arch, key);
            if (value) normalized[key] = value;
        }
    }
    if (normalized.count != required.count ||
        [normalized[@"topKExperts"] unsignedIntegerValue] >
            [normalized[@"numExperts"] unsignedIntegerValue]) {
        if (error) *error = jsonError.localizedDescription ?: @"model manifest has no valid architecture descriptor";
        return nil;
    }
    ANEFieldfareInferenceOptimizer *optimizer = [self new];
    optimizer.architecture = normalized;
    optimizer.architectureSignature = [NSString stringWithFormat:
        @"shape-v1-h%@-l%@-e%@-k%@-mi%@-si%@-v%@-ah%@-kh%@-hd%@",
        normalized[@"hiddenSize"], normalized[@"numLayers"], normalized[@"numExperts"],
        normalized[@"topKExperts"], normalized[@"moeIntermediateSize"],
        normalized[@"ffnIntermediate"], normalized[@"vocabSize"],
        normalized[@"numHeads"], normalized[@"numKVHeads"], normalized[@"headDim"]];
    return optimizer;
}

- (NSDictionary *)candidatePlanForPositionCount:(NSUInteger)positionCount
                           availableMemoryBytes:(uint64_t)availableMemoryBytes
                            backendCapabilities:(NSDictionary *)capabilities
                               costObservations:(NSDictionary *)observations {
    NSUInteger experts = [self.architecture[@"numExperts"] unsignedIntegerValue];
    NSUInteger topK = [self.architecture[@"topKExperts"] unsignedIntegerValue];
    BOOL hasMoE = experts > 1 && topK < experts;
    BOOL hasSharedFFN = [self.architecture[@"ffnIntermediate"] unsignedIntegerValue] > 0;
    BOOL batched = positionCount > 1;

    // Every numeric bound comes from the backend contract. Measurements select
    // only between semantically equivalent lowerings; the native router remains
    // authoritative, so a continuation miss can cost latency but not correctness.
    NSUInteger carrierCapacity = [capabilities[@"max_position_carrier"] unsignedIntegerValue];
    NSUInteger residentSlots = [capabilities[@"resident_expert_slots"] unsignedIntegerValue];
    NSUInteger stateWidth = [capabilities[@"expert_weight_state_width"] unsignedIntegerValue];
    uint64_t projectionWorkspace = [capabilities[@"resident_projection_workspace_bytes"] unsignedLongLongValue];
    BOOL deviceAttention = batched && [capabilities[@"device_attention_prefill"] boolValue];
    BOOL transport = hasMoE && [capabilities[@"weight_state_transport"] boolValue];
    BOOL nativeVerification = [capabilities[@"native_route_verification"] boolValue];
    double transportMs = [observations[@"weight_state_transport_ms"] doubleValue];
    double receiverLeadMs = [observations[@"receiver_lead_time_ms"] doubleValue];
    BOOL overlapFeasible = transport && nativeVerification && transportMs > 0 &&
        receiverLeadMs >= transportMs;
    NSUInteger tileCapacity = batched ? MIN(positionCount, carrierCapacity) : 0;
    NSUInteger cacheSlots = hasMoE ? MIN(residentSlots, experts) : 0;
    uint64_t projectionBudget = deviceAttention
        ? MIN(projectionWorkspace, availableMemoryBytes) : 0;
    return @{
        @"schema": @"ane-fieldfare.execution-field-plan.v2",
        @"architecture_signature": self.architectureSignature,
        @"applicability": @{ @"phase": @"prefill", @"minimum_positions": @2,
                              @"maximum_positions": @(NSUIntegerMax) },
        @"schedule": @{
            @"backend": @"hybrid",
            @"position_tile_capacity": @(tileCapacity),
            @"group_routes_by_weight_identity": hasMoE && batched ? @YES : @NO,
            @"route_group_capacity": @(cacheSlots),
            @"expert_cache_slots_per_layer": @(cacheSlots),
            @"expert_weight_state_width": @(hasMoE ? stateWidth : 0),
            @"weight_state_transport": transport ? @YES : @NO,
            @"receiver_continuation_prefetch": overlapFeasible ? @YES : @NO,
            @"native_route_verification": nativeVerification ? @YES : @NO,
            @"continuation_miss_effect": @"latency_only",
            @"batch_shared_ffn": hasSharedFFN && batched ? @YES : @NO,
            @"device_attention": deviceAttention ? @YES : @NO,
            @"ane_projection_budget_bytes": @(projectionBudget),
            @"decode_ane_projection": @NO
        },
        @"generation": @{
            @"kind": @"capability_lowered_execution_field",
            @"training_required": @NO,
            @"model_compilation_required": @NO,
            @"capability_provenance": capabilities[@"provenance"] ?: @"unlabelled",
            @"selection_rule": @"feasibility_and_measured_critical_path"
        }
    };
}

- (NSDictionary *)loadAdmittedPlanAtPath:(NSString *)path
                            positionCount:(NSUInteger)positionCount
                                    error:(NSString **)error {
    NSData *data = [NSData dataWithContentsOfFile:path];
    NSError *jsonError = nil;
    NSDictionary *receipt = data
        ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&jsonError] : nil;
    NSDictionary *plan = [receipt isKindOfClass:NSDictionary.class] ? receipt[@"plan"] : nil;
    NSDictionary *applicability = [plan isKindOfClass:NSDictionary.class]
        ? plan[@"applicability"] : nil;
    BOOL valid = [receipt[@"status"] isEqual:@"validated"] &&
        [plan[@"schema"] isEqual:@"ane-fieldfare.execution-field-plan.v2"] &&
        [plan[@"architecture_signature"] isEqual:self.architectureSignature] &&
        [applicability[@"phase"] isEqual:@"prefill"] &&
        positionCount >= [applicability[@"minimum_positions"] unsignedIntegerValue] &&
        positionCount <= [applicability[@"maximum_positions"] unsignedIntegerValue];
    if (!valid) {
        if (error) *error = jsonError.localizedDescription ?: @"cached execution plan was not admitted for this architecture and shape regime";
        return nil;
    }
    return plan;
}

@end
