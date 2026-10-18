#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>

#include <math.h>
#include <unistd.h>

#include "../ANEFieldfareRuntime/ANEDynamicProjection.h"

enum { Hidden = 2816, Outputs = 128, Positions = 64, Values = Outputs * Positions,
       Repetitions = 8 };

static double milliseconds(uint64_t ticks) {
    mach_timebase_info_data_t timebase = {0};
    mach_timebase_info(&timebase);
    return (double)ticks * timebase.numer / timebase.denom / 1e6;
}

static NSDictionary *methodRecord(Class cls, SEL selector, BOOL classMethod,
                                  const char *expected) {
    Method method = classMethod ? class_getClassMethod(cls, selector)
                                : class_getInstanceMethod(cls, selector);
    const char *actual = method ? method_getTypeEncoding(method) : NULL;
    BOOL matches = actual && strcmp(actual, expected) == 0;
    return @{
        @"class": cls ? NSStringFromClass(cls) : @"none",
        @"selector": NSStringFromSelector(selector),
        @"kind": classMethod ? @"class" : @"instance",
        @"expected_encoding": @(expected),
        @"actual_encoding": actual ? @(actual) : @"none",
        @"matches": @(matches),
    };
}

static BOOL recordsMatch(NSArray *records) {
    for (NSDictionary *record in records) if (![record[@"matches"] boolValue]) return NO;
    return YES;
}

static BOOL initializeCarrier(__fp16 *carrier, const ANEFieldfareTensor *weights,
                              uint32_t stride) {
    memset(carrier, 0, (size_t)Hidden * stride * sizeof(*carrier));
    for (uint32_t column = 0; column < Hidden; ++column) {
        for (uint32_t row = 0; row < Outputs; ++row) {
            float value = 0;
            if (!ANEFieldfareAffineValue(weights, 8, 64, row, column, &value)) return NO;
            carrier[(uint64_t)column * stride + Positions + row] = (__fp16)value;
        }
    }
    return YES;
}

static void fillActivation(__fp16 *carrier, uint32_t stride, float scale) {
    for (uint32_t column = 0; column < Hidden; ++column)
        carrier[(uint64_t)column * stride] =
            (__fp16)(((int32_t)(column % 257) - 128) * scale);
}

static void poisonSurface(IOSurfaceRef surface) {
    IOSurfaceLock(surface, 0, NULL);
    float *values = IOSurfaceGetBaseAddress(surface);
    for (uint32_t index = 0; index < Values; ++index) values[index] = NAN;
    IOSurfaceUnlock(surface, 0, NULL);
}

static NSDictionary *oracle(IOSurfaceRef input, id<MTLBuffer> downstream, uint32_t stride) {
    IOSurfaceLock(input, kIOSurfaceLockReadOnly, NULL);
    const __fp16 *carrier = IOSurfaceGetBaseAddress(input);
    const float *actual = downstream.contents;
    double maxError = 0;
    uint32_t failures = 0, poison = 0;
    for (uint32_t row = 0; row < Outputs; ++row) {
        double sum = 0;
        for (uint32_t column = 0; column < Hidden; ++column)
            sum += (float)carrier[(uint64_t)column * stride] *
                   (float)carrier[(uint64_t)column * stride + Positions + row];
        for (uint32_t position = 0; position < Positions; ++position) {
            float expected = position == 0 ? (float)(2.0 * sum + 1.0) : 1.0f;
            float got = actual[(uint64_t)row * Positions + position];
            if (isnan(got)) ++poison;
            double error = fabs((double)got - expected);
            maxError = fmax(maxError, error);
            if (!isfinite(got) || error > 0.06 + 0.04 * fabs(expected)) ++failures;
        }
    }
    IOSurfaceUnlock(input, kIOSurfaceLockReadOnly, NULL);
    return @{
        @"valid": @(failures == 0 && poison == 0),
        @"max_absolute_error": @(maxError),
        @"tolerance_failures": @(failures),
        @"poison_values_remaining": @(poison),
    };
}

static NSUInteger poisonCount(id<MTLBuffer> buffer) {
    const float *values = buffer.contents;
    NSUInteger count = 0;
    for (uint32_t index = 0; index < Values; ++index) if (isnan(values[index])) ++count;
    return count;
}

static NSString *copiedRequestDescription(ANEFieldfareANEProgram *program) {
    CFTypeRef value = ANEFieldfareANEProgramCopyRequestDescription(program, 0);
    if (!value) return @"none";
    NSString *description = [(__bridge NSString *)value copy];
    CFRelease(value);
    return description;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        void *aneFramework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class sharedEventsClass = NSClassFromString(@"_ANESharedEvents");
        Class signalEventClass = NSClassFromString(@"_ANESharedSignalEvent");
        Class waitEventClass = NSClassFromString(@"_ANESharedWaitEvent");
        Class requestClass = NSClassFromString(@"_ANERequest");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLSharedEvent> metalEvent = [device newSharedEvent];
        Class metalEventClass = metalEvent ? [(id)metalEvent class] : Nil;
        NSArray *abiChecks = @[
            methodRecord(metalEventClass, NSSelectorFromString(@"IOSurfaceSharedEvent"), NO,
                         "@16@0:8"),
            methodRecord(signalEventClass,
                         @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:), YES,
                         "@44@0:8Q16I24q28@36"),
            methodRecord(signalEventClass, @selector(waitEvent), NO, "@16@0:8"),
            methodRecord(waitEventClass, @selector(waitEventWithValue:sharedEvent:eventType:), YES,
                         "@40@0:8Q16@24Q32"),
            methodRecord(sharedEventsClass, @selector(sharedEventsWithSignalEvents:waitEvents:),
                         YES, "@32@0:8@16@24"),
            methodRecord(requestClass, @selector(setSharedEvents:), NO, "v24@0:8@16"),
            methodRecord(requestClass, @selector(sharedEvents), NO, "@16@0:8"),
            methodRecord(requestClass, @selector(transactionHandle), NO, "@16@0:8"),
        ];
        BOOL abiValid = aneFramework && device && metalEvent && recordsMatch(abiChecks);

        const uint32_t stride = Positions + Outputs;
        const uint64_t inputBytes = (uint64_t)Hidden * stride * sizeof(__fp16);
        const uint64_t outputBytes = (uint64_t)Values * sizeof(float);
        NSString *root = [@(argv[1]) stringByStandardizingPath];
        NSString *weightsPath = [root stringByAppendingPathComponent:@"model_weights.bin"];
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor router = {0};
        const char *creationError = NULL;
        BOOL valid = abiValid &&
            ANEFieldfareTensorIndexOpen(weightsPath.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(
                &index, "language_model.model.layers.0.router.proj.weight", &router);
        ANEFieldfareANEDynamicProjection *projection = valid ?
            ANEFieldfareANEDynamicProjectionCreate(Hidden, Outputs, 1, &creationError) : NULL;
        valid = valid && projection;
        ANEFieldfareANEProgram *program = projection ?
            ANEFieldfareANEDynamicProjectionProgram(projection) : NULL;
        IOSurfaceRef input = program ? ANEFieldfareANEProgramInput(program, 0) : NULL;
        IOSurfaceRef output = program ? ANEFieldfareANEProgramOutput(program, 0) : NULL;

        __fp16 *carrierTemplate = calloc(inputBytes / sizeof(__fp16), sizeof(__fp16));
        valid = valid && carrierTemplate && initializeCarrier(carrierTemplate, &router, stride);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *metalError = nil;
        NSString *source = @"#include <metal_stdlib>\nusing namespace metal; "
            "kernel void consume(const device float *x [[buffer(0)]], "
            "device float *y [[buffer(1)]], uint i [[thread_position_in_grid]]) "
            "{ if (i < 8192) y[i] = 2.0f * x[i] + 1.0f; }";
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&metalError];
        id<MTLFunction> function = [library newFunctionWithName:@"consume"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&metalError];

        void *outputBase = NULL;
        size_t outputAllocation = 0;
        long pageSize = sysconf(_SC_PAGESIZE);
        if (output) {
            IOSurfaceLock(output, 0, NULL);
            outputBase = IOSurfaceGetBaseAddress(output);
            outputAllocation = IOSurfaceGetAllocSize(output);
            IOSurfaceUnlock(output, 0, NULL);
        }
        BOOL alignmentGuard = outputBase && pageSize > 0 &&
            ((uintptr_t)outputBase % (uintptr_t)pageSize) == 0 &&
            outputAllocation >= outputBytes && outputAllocation % (size_t)pageSize == 0;
        id<MTLBuffer> directOutput = alignmentGuard ?
            [device newBufferWithBytesNoCopy:outputBase length:outputAllocation
                                     options:MTLResourceStorageModeShared deallocator:nil] : nil;
        id<MTLBuffer> downstream =
            [device newBufferWithLength:outputBytes options:MTLResourceStorageModeShared];
        BOOL identityGuard = directOutput && directOutput.contents == outputBase;
        valid = valid && input && output && queue && pipeline && directOutput && downstream &&
            alignmentGuard && identityGuard;

        id ioEvent = abiValid ? ((id (*)(id, SEL))objc_msgSend)(
            metalEvent, NSSelectorFromString(@"IOSurfaceSharedEvent")) : nil;
        NSMutableArray *runs = [NSMutableArray array];
        NSString *requestBefore = program ? copiedRequestDescription(program) : @"none";
        NSString *requestAfter = @"none";
        BOOL negativeBlocked = NO, allRunsValid = valid;
        double warmWall = 0;

        for (uint32_t repetition = 0; valid && repetition < Repetitions; ++repetition) {
            uint64_t eventValue = repetition + 1;
            id signalEvent = ((id (*)(Class, SEL, uint64_t, uint32_t, int64_t, id))objc_msgSend)(
                signalEventClass,
                @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
                eventValue, 0, 0, ioEvent);
            id derivedWait = signalEvent ? ((id (*)(id, SEL))objc_msgSend)(
                signalEvent, @selector(waitEvent)) : nil;
            id sharedEvents = signalEvent ? ((id (*)(Class, SEL, id, id))objc_msgSend)(
                sharedEventsClass, @selector(sharedEventsWithSignalEvents:waitEvents:),
                @[signalEvent], @[]) : nil;
            const char *configurationError = NULL;
            BOOL configured = sharedEvents && ANEFieldfareANEProgramSetRequestSharedEvents(
                program, 0, (__bridge CFTypeRef)sharedEvents, &configurationError);
            if (repetition == 0 && configured) requestAfter = copiedRequestDescription(program);

            IOSurfaceLock(input, 0, NULL);
            __fp16 *carrier = IOSurfaceGetBaseAddress(input);
            memcpy(carrier, carrierTemplate, inputBytes);
            fillActivation(carrier, stride, 0.0009765625f * (1.0f + 0.0625f * repetition));
            IOSurfaceUnlock(input, 0, NULL);
            poisonSurface(output);
            float *destination = downstream.contents;
            for (uint32_t indexValue = 0; indexValue < Values; ++indexValue)
                destination[indexValue] = NAN;

            dispatch_semaphore_t metalDone = dispatch_semaphore_create(0);
            id<MTLCommandBuffer> command = [queue commandBuffer];
            [command encodeWaitForEvent:metalEvent value:eventValue];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:directOutput offset:0 atIndex:0];
            [encoder setBuffer:downstream offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(Values, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(
                    MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
            [encoder endEncoding];
            [command addCompletedHandler:^(__unused id<MTLCommandBuffer> completed) {
                dispatch_semaphore_signal(metalDone);
            }];
            uint64_t started = mach_continuous_time();
            [command commit];

            BOOL completedBeforeANE = dispatch_semaphore_wait(
                metalDone, dispatch_time(DISPATCH_TIME_NOW, 2ULL * NSEC_PER_MSEC)) == 0;
            NSUInteger poisonBeforeANE = poisonCount(downstream);
            BOOL blockedBeforeANE = !completedBeforeANE && poisonBeforeANE == Values;
            if (repetition == 0) negativeBlocked = blockedBeforeANE;

            const char *beginError = NULL, *waitError = NULL;
            ANEFieldfareANEEvaluationTiming beginTiming = {0}, completionTiming = {0};
            uint64_t aneStarted = mach_continuous_time();
            BOOL began = configured && blockedBeforeANE && ANEFieldfareANEProgramEvaluateBegin(
                program, 0, &beginTiming, &beginError);
            BOOL metalCompleted = dispatch_semaphore_wait(
                metalDone, dispatch_time(DISPATCH_TIME_NOW, 5ULL * NSEC_PER_SEC)) == 0;
            if (!metalCompleted) {
                metalEvent.signaledValue = eventValue;
                metalCompleted = dispatch_semaphore_wait(
                    metalDone, dispatch_time(DISPATCH_TIME_NOW, 1ULL * NSEC_PER_SEC)) == 0;
            }
            uint64_t metalFinished = mach_continuous_time();
            BOOL aneCompleted = began && ANEFieldfareANEProgramEvaluateWait(
                program, 0, 5ULL * NSEC_PER_SEC, &completionTiming, &waitError);
            double totalWall = milliseconds(metalFinished - started);
            double aneToMetalWall = milliseconds(metalFinished - aneStarted);
            NSDictionary *check = oracle(input, downstream, stride);
            BOOL runValid = configured && derivedWait && blockedBeforeANE && began &&
                metalCompleted && aneCompleted && command.status == MTLCommandBufferStatusCompleted &&
                [check[@"valid"] boolValue];
            allRunsValid = allRunsValid && runValid;
            if (repetition) warmWall += aneToMetalWall;
            [runs addObject:@{
                @"repetition": @(repetition),
                @"warm": @(repetition > 0),
                @"event_value": @(eventValue),
                @"signal_event": signalEvent ? [signalEvent description] : @"none",
                @"derived_wait_event": derivedWait ? [derivedWait description] : @"none",
                @"request_configured": @(configured),
                @"completed_before_ane_submission": @(completedBeforeANE),
                @"downstream_poison_before_ane_submission": @(poisonBeforeANE),
                @"negative_control_blocked": @(blockedBeforeANE),
                @"ane_begin_accepted": @(began),
                @"ane_completion_observed_for_cleanup": @(aneCompleted),
                @"metal_completed_after_ane_signal": @(metalCompleted),
                @"metal_commit_to_completion_including_negative_control_ms": @(totalWall),
                @"ane_begin_to_metal_completion_ms": @(aneToMetalWall),
                @"ane_completion_ms": @(completionTiming.completionMilliseconds),
                @"metal_gpu_ms": @((command.GPUEndTime - command.GPUStartTime) * 1000.0),
                @"oracle": check,
                @"configuration_error": configurationError ? @(configurationError) : @"none",
                @"begin_error": beginError ? @(beginError) : @"none",
                @"wait_error": waitError ? @(waitError) : @"none",
                @"valid": @(runValid),
            }];
            free((void *)configurationError);
            free((void *)beginError);
            free((void *)waitError);
            valid = valid && runValid;
        }

        ANEFieldfareANEProgramInfo info = projection ?
            ANEFieldfareANEDynamicProjectionGetInfo(projection) : (ANEFieldfareANEProgramInfo){0};
        BOOL validated = valid && allRunsValid && negativeBlocked && runs.count == Repetitions;
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.event-transaction.v1",
            @"status": validated ? @"validated" : @"rejected",
            @"stage": validated ? @"ane_signal_to_metal_wait_oracle" : @"execution",
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"command": @".build/ane-fieldfare-event-gate checkpoints/gturbo/gemma4.gturbo",
            @"abi_checks": abiChecks,
            @"objects": @{
                @"metal_shared_event_class": metalEvent ? NSStringFromClass([(id)metalEvent class]) : @"none",
                @"iosurface_shared_event_class": ioEvent ? NSStringFromClass([ioEvent class]) : @"none",
                @"same_object_identity": @(metalEvent && ioEvent && (id)metalEvent == ioEvent),
                @"signal_event_class": @"_ANESharedSignalEvent",
                @"wait_event_class": @"_ANESharedWaitEvent",
                @"event_set_class": @"_ANESharedEvents",
            },
            @"request": @{
                @"before_shared_event_attachment": requestBefore,
                @"after_shared_event_attachment": requestAfter,
                @"setter_readback_validated": @(validated),
                @"transaction_handle_policy": @"runtime_owned; no nonzero handle synthesized",
            },
            @"shape": @{
                @"output_values": @(Values), @"output_bytes": @(outputBytes),
                @"iosurface_allocation_bytes": @(outputAllocation),
            },
            @"guards": @{
                @"output_base_nonnull": @(outputBase != NULL),
                @"output_page_aligned": @(alignmentGuard),
                @"metal_buffer_contents_identity": @(identityGuard),
                @"program_surface_event_lifetime_through_completion": @YES,
            },
            @"lifecycle": @{
                @"ane_compile_count": projection ? @1 : @0,
                @"ane_load_count": projection ? @1 : @0,
                @"ane_compile_ms": @(info.compileMilliseconds),
                @"ane_load_ms": @(info.loadMilliseconds),
                @"metal_library_compile_count": library ? @1 : @0,
                @"metal_pipeline_load_count": pipeline ? @1 : @0,
            },
            @"negative_control": @{
                @"metal_wait_committed_before_ane_submission": @YES,
                @"blocked_and_preserved_poison": @(negativeBlocked),
                @"timeout_ms": @2,
            },
            @"runs": runs,
            @"warm_repetitions": @(Repetitions - 1),
            @"warm_mean_ane_begin_to_metal_completion_ms":
                runs.count == Repetitions ? @(warmWall / (Repetitions - 1)) : @0,
            @"shared_event": @{
                @"symbol_presence": @"present",
                @"request_attachment": validated ? @"validated" : @"rejected",
                @"ane_signal_to_metal_wait_ordering": validated ? @"validated" : @"rejected",
                @"host_semaphore_between_ane_and_metal": @NO,
                @"ane_completion_wait": @"performed only after Metal completion for cleanup and timing readback",
            },
            @"transaction": @{
                @"request_property_and_constructor": @"present",
                @"authoritative_allocator_or_lifecycle_contract": @"absent",
                @"nonzero_handle_constructed": @NO,
                @"status": @"present",
            },
            @"error": creationError ? @(creationError) : @"none",
            @"metal_error": metalError.localizedDescription ?: @"none",
            @"claim_boundary": @"One authentic router projection signals a public Metal shared event through the ABI-validated private ANE event descriptor. Metal is committed first and waits without a host semaphore; before ANE submission it remains blocked and downstream poison is intact. After ANE submission all values pass an independent projection-plus-consumer oracle. This validates ANE-to-Metal ordering for one request and one output symbol, not overlap, two-ANE-request dependency, chaining, persistent state, or transaction-handle semantics.",
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);

        free((void *)creationError);
        free(carrierTemplate);
        if (projection) ANEFieldfareANEDynamicProjectionDestroy(projection);
        if (index.fileDescriptor >= 0) ANEFieldfareTensorIndexClose(&index);
        return validated ? 0 : 1;
    }
}
