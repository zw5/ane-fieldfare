#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

static NSString *deviceName(id<MLComputeDeviceProtocol> device) {
    return NSStringFromClass([(id)device class]);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSString *path = argc > 1 ? [NSString stringWithUTF8String:argv[1]] :
            @"/System/Library/PrivateFrameworks/EmbeddingCore.framework/Versions/A/Resources/cross_encoder_v130_ane_8bit_combined.mlmodelc";
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        configuration.functionName = @"main_ctx_128";
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        __block MLComputePlan *plan = nil;
        __block NSError *planError = nil;
        [MLComputePlan loadContentsOfURL:[NSURL fileURLWithPath:path]
                          configuration:configuration
                      completionHandler:^(MLComputePlan *value, NSError *error) {
            plan = value; planError = error; dispatch_semaphore_signal(done);
        }];
        dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC));

        NSMutableDictionary *operatorCounts = [NSMutableDictionary dictionary];
        NSMutableDictionary *preferredCounts = [NSMutableDictionary dictionary];
        NSMutableDictionary *operatorPreferredCounts = [NSMutableDictionary dictionary];
        NSMutableArray *compressedOperators = [NSMutableArray array];
        MLModelStructureProgram *program = plan.modelStructure.program;
        MLModelStructureProgramFunction *function = program.functions[@"main_ctx_128"];
        for (MLModelStructureProgramOperation *operation in function.block.operations) {
            NSString *name = operation.operatorName ?: @"unknown";
            operatorCounts[name] = @([operatorCounts[name] unsignedIntegerValue] + 1);
            MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:operation];
            NSString *preferred = usage.preferredComputeDevice ? deviceName(usage.preferredComputeDevice) : @"none";
            preferredCounts[preferred] = @([preferredCounts[preferred] unsignedIntegerValue] + 1);
            NSString *pair = [NSString stringWithFormat:@"%@ -> %@", name, preferred];
            operatorPreferredCounts[pair] = @([operatorPreferredCounts[pair] unsignedIntegerValue] + 1);
            if ([name.lowercaseString containsString:@"constexpr"] ||
                [name.lowercaseString containsString:@"blockwise"]) {
                NSMutableArray *supported = [NSMutableArray array];
                for (id<MLComputeDeviceProtocol> device in usage.supportedComputeDevices)
                    [supported addObject:deviceName(device)];
                [compressedOperators addObject:@{
                    @"operator": name,
                    @"preferred_device": preferred,
                    @"supported_devices": supported,
                }];
            }
        }
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.compressed-compute-plan.v1",
            @"source_model": path,
            @"configuration": @"CPUAndNeuralEngine; function main_ctx_128",
            @"plan_loaded": @(plan != nil),
            @"error": planError.localizedDescription ?: @"none",
            @"operator_counts": operatorCounts,
            @"preferred_device_counts": preferredCounts,
            @"operator_preferred_device_counts": operatorPreferredCounts,
            @"compressed_operators": compressedOperators,
            @"claim_boundary": @"Core ML compute-plan device assignment for Apple's shipped mixed-FP16/int8 model. The model metadata and MIL establish int8 stored constants; preferred MLNeuralEngineComputeDevice for consuming convs establishes planned ANE execution. Compute-plan output alone does not reveal whether the ANE compiler preserves int8 in its final hardware binary or expands it during compilation.",
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        return plan ? 0 : 1;
    }
}
