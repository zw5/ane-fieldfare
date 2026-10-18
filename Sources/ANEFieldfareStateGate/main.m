#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <mach/mach_time.h>

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

enum { ElementCount = 256 * 64 };

static MLMultiArray *arrayWithValues(const float *values, NSError **error) {
    MLMultiArray *array = [[MLMultiArray alloc] initWithShape:@[@1, @256, @1, @64]
                                                    dataType:MLMultiArrayDataTypeFloat16
                                                       error:error];
    if (!array) return nil;
    __fp16 *storage = array.dataPointer;
    for (NSUInteger i = 0; i < ElementCount; ++i) storage[i] = (__fp16)values[i];
    return array;
}

static NSArray *sampleFromArray(MLMultiArray *array) {
    NSMutableArray *values = [NSMutableArray arrayWithCapacity:8];
    __fp16 *storage = array.dataPointer;
    for (NSUInteger i = 0; i < 8; ++i) [values addObject:@((float)storage[i])];
    return values;
}

static BOOL closeToArray(MLMultiArray *actualArray, const float *expected, float tolerance,
                    float *maxError) {
    BOOL valid = YES;
    __fp16 *storage = actualArray.dataPointer;
    for (NSUInteger i = 0; i < ElementCount; ++i) {
        float actual = (float)storage[i];
        float error = fabsf(actual - expected[i]);
        if (error > *maxError) *maxError = error;
        if (!isfinite(actual) || error > tolerance) valid = NO;
    }
    return valid;
}

static MLMultiArray *runStep(MLModel *model, MLState *state, const float *input,
                        NSError **error, double *milliseconds) {
    MLMultiArray *x = arrayWithValues(input, error);
    if (!x) return nil;
    MLDictionaryFeatureProvider *features =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x": x} error:error];
    if (!features) return nil;
    uint64_t begin = mach_continuous_time();
    id<MLFeatureProvider> output = [model predictionFromFeatures:features
                                                     usingState:state
                                                          error:error];
    uint64_t end = mach_continuous_time();
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    *milliseconds = (double)(end - begin) * timebase.numer / timebase.denom / 1.0e6;
    return output ? [output featureValueForName:@"y"].multiArrayValue : nil;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSString *package = argc > 1 ? @(argv[1]) :
            @".build/state-recurrence/state-recurrence.mlpackage";
        NSURL *packageURL = [NSURL fileURLWithPath:package];
        NSError *error = nil;
        NSURL *compiledURL = [MLModel compileModelAtURL:packageURL error:&error];
        if (!compiledURL) {
            emit(@{@"schema": @"ane-fieldfare.state-recurrence.v1", @"status": @"rejected",
                   @"stage": @"compile", @"error": error.localizedDescription ?: @"unknown",
                   @"package": package,
                   @"claim_boundary": @"Compiler rejection only; no state, numerical, placement, or concurrency claim."});
            return 1;
        }

        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        MLModel *model = [MLModel modelWithContentsOfURL:compiledURL
                                           configuration:configuration error:&error];
        if (!model) {
            emit(@{@"schema": @"ane-fieldfare.state-recurrence.v1", @"status": @"rejected",
                   @"stage": @"load", @"error": error.localizedDescription ?: @"unknown",
                   @"package": package,
                   @"claim_boundary": @"Public Core ML load rejection only; no state, numerical, placement, or concurrency claim."});
            return 1;
        }

        dispatch_semaphore_t planReady = dispatch_semaphore_create(0);
        __block MLComputePlan *plan = nil;
        __block NSError *planError = nil;
        [MLComputePlan loadContentsOfURL:compiledURL configuration:configuration
                      completionHandler:^(MLComputePlan *loaded, NSError *failure) {
            plan = loaded;
            planError = failure;
            dispatch_semaphore_signal(planReady);
        }];
        dispatch_semaphore_wait(planReady, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC));

        NSMutableArray *placement = [NSMutableArray array];
        NSUInteger anePreferred = 0;
        NSUInteger aneSupported = 0;
        NSUInteger operationCount = 0;
        NSDictionary *functions = plan.modelStructure.program.functions;
        for (NSString *functionName in functions) {
            MLModelStructureProgramFunction *function = functions[functionName];
            for (MLModelStructureProgramOperation *operation in function.block.operations) {
                MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:operation];
                NSString *preferred = usage ? NSStringFromClass([usage.preferredComputeDevice class]) : @"none";
                NSMutableArray *supported = [NSMutableArray array];
                for (id device in usage.supportedComputeDevices)
                    [supported addObject:NSStringFromClass([device class])];
                for (NSString *device in supported)
                    if ([device.lowercaseString containsString:@"neuralengine"]) ++aneSupported;
                if ([preferred.lowercaseString containsString:@"neuralengine"]) ++anePreferred;
                ++operationCount;
                [placement addObject:@{@"function": functionName,
                                       @"operator": operation.operatorName ?: @"unknown",
                                       @"preferred": preferred,
                                       @"supported": supported}];
            }
        }

        MLState *stateA = [model newState];
        MLState *stateB = [model newState];
        float *inputs[3] = {calloc(ElementCount, sizeof(float)), calloc(ElementCount, sizeof(float)), calloc(ElementCount, sizeof(float))};
        float *expectedA = calloc(ElementCount, sizeof(float));
        float *expectedB = calloc(ElementCount, sizeof(float));
        for (NSUInteger i = 0; i < ElementCount; ++i) {
            inputs[0][i] = (float)((NSInteger)(i % 9) - 4) * 0.125f;
            inputs[1][i] = (float)((NSInteger)(i % 7) - 3) * 0.25f;
            inputs[2][i] = (float)((NSInteger)(i % 5) - 2) * 0.5f;
        }
        __block float maxError = 0.0f;
        BOOL numerical = YES;
        NSMutableArray *steps = [NSMutableArray array];
        for (NSUInteger i = 0; i < 2; ++i) {
            for (NSUInteger j = 0; j < ElementCount; ++j) expectedA[j] += inputs[i][j];
            double milliseconds = 0;
            MLMultiArray *output = runStep(model, stateA, inputs[i], &error, &milliseconds);
            numerical &= output && closeToArray(output, expectedA, 0.002f, &maxError);
            [steps addObject:@{@"state": @"A", @"input": @(i), @"output_sample": output ? sampleFromArray(output) : @[],
                               @"expected_sample": @[@(expectedA[0]), @(expectedA[1]), @(expectedA[2]), @(expectedA[3])],
                               @"prediction_ms": @(milliseconds)}];
        }
        for (NSUInteger j = 0; j < ElementCount; ++j) expectedB[j] += inputs[2][j];
        double millisecondsB = 0;
        MLMultiArray *outputB = runStep(model, stateB, inputs[2], &error, &millisecondsB);
        numerical &= outputB && closeToArray(outputB, expectedB, 0.002f, &maxError);
        [steps addObject:@{@"state": @"B_reset_control", @"input": @2,
                           @"output_sample": outputB ? sampleFromArray(outputB) : @[],
                           @"expected_sample": @[@(expectedB[0]), @(expectedB[1]), @(expectedB[2]), @(expectedB[3])],
                           @"prediction_ms": @(millisecondsB)}];

        for (NSUInteger j = 0; j < ElementCount; ++j) expectedA[j] += inputs[2][j];
        double millisecondsA3 = 0;
        MLMultiArray *outputA3 = runStep(model, stateA, inputs[2], &error, &millisecondsA3);
        numerical &= outputA3 && closeToArray(outputA3, expectedA, 0.002f, &maxError);
        __block BOOL stateReadback = NO;
        __block NSArray *observedStateSample = @[];
        [stateA getMultiArrayForStateNamed:@"cache" handler:^(MLMultiArray *array) {
            observedStateSample = sampleFromArray(array);
            stateReadback = closeToArray(array, expectedA, 0.002f, &maxError);
        }];
        numerical &= stateReadback;
        [steps addObject:@{@"state": @"A_return", @"input": @2,
                           @"output_sample": outputA3 ? sampleFromArray(outputA3) : @[],
                           @"expected_sample": @[@(expectedA[0]), @(expectedA[1]), @(expectedA[2]), @(expectedA[3])],
                           @"prediction_ms": @(millisecondsA3)}];

        BOOL aneAssigned = anePreferred > 0;
        BOOL validated = numerical && stateReadback && aneAssigned && error == nil;
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.state-recurrence.v1",
            @"status": validated ? @"validated" : @"rejected",
            @"stage": validated ? @"state_recurrence_and_placement" :
                (numerical ? @"compute_plan_ane_assignment" : @"numerical_or_state"),
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"command": @".build/ane-fieldfare-state-gate .build/state-recurrence/state-recurrence.mlpackage",
            @"package": package,
            @"model": @{@"recurrence": @"cache <- cache + x", @"dtype": @"fp16",
                         @"shape": @[@1, @256, @1, @64], @"compiler": @"coremltools 9.0 plus MLModel compileModelAtURL"},
            @"serialization": @"All predictions sharing state A were issued synchronously and serially as required by MLState. State B is an independent reset control.",
            @"steps": steps,
            @"state_a_readback_sample": observedStateSample,
            @"state_readback_matches_oracle": @(stateReadback),
            @"max_abs_error": @(maxError),
            @"state_lifecycle": @{@"status": numerical && stateReadback ? @"validated" : @"rejected",
                                   @"serialized": @YES,
                                   @"independent_reset_state": @YES,
                                   @"return_to_state_a": @YES,
                                   @"explicit_readback": @(stateReadback)},
            @"ane_placement": @{@"status": aneAssigned ? @"validated" : @"rejected",
                                 @"supported_by_compute_plan": @(aneSupported > 0),
                                 @"supported_operation_count": @(aneSupported),
                                 @"preferred_operation_count": @(anePreferred),
                                 @"blocker": aneAssigned ? @"none" : @"MLComputePlan prefers MLCPUComputeDevice for every executable operation, including read_state, conv, add, and write_state, even under MLComputeUnitsCPUAndNeuralEngine."},
            @"compute_plan": @{@"loaded": @(plan != nil),
                                @"error": planError.localizedDescription ?: @"none",
                                @"operation_count": @(operationCount),
                                @"ane_preferred_operations": @(anePreferred),
                                @"operations": placement},
            @"error": error.localizedDescription ?: @"none",
            @"claim_boundary": validated
                ? @"Validates a compiler-authored public Core ML FP16 state buffer whose mutation persists across serialized calls, remains independent across two MLState handles, matches an independent CPU recurrence oracle and explicit state readback, and has at least one operation preferred on the Neural Engine by MLComputePlan. This is a 256-channel by 64-position recurrence, not Gemma KV, private chaining, mutable weight, queue concurrency, or generation."
                : (numerical && stateReadback
                    ? @"Validates the compiler-authored public Core ML state lifecycle, serialized mutation, independent reset handle, return-to-A continuity, explicit state readback, and CPU recurrence oracle. MLComputePlan reports Neural Engine support but prefers CPU for every executable operation, so ANE assignment is rejected. This is not ANE-resident KV, mutable weight, chaining, concurrency, or generation."
                    : @"The named state stage failed. No ANE-resident KV, mutable-weight, chaining, concurrency, or generation claim is made.")
        };
        emit(receipt);
        for (NSUInteger i = 0; i < 3; ++i) free(inputs[i]);
        free(expectedA); free(expectedB);
        return validated ? 0 : 1;
    }
}
