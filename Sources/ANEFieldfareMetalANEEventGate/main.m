#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <stdatomic.h>
#import <dlfcn.h>

#include <math.h>
#include <unistd.h>

#include "../ANEFieldfareRuntime/ANEProgram.h"

enum { Channels = 64, Positions = 64, Elements = Channels * Positions, Repetitions = 3 };

static NSDictionary *methodRecord(Class cls, SEL selector, BOOL classMethod,
                                  const char *expected) {
    Method method = classMethod ? class_getClassMethod(cls, selector)
                                : class_getInstanceMethod(cls, selector);
    const char *actual = method ? method_getTypeEncoding(method) : NULL;
    return @{@"class": cls ? NSStringFromClass(cls) : @"none",
             @"selector": NSStringFromSelector(selector),
             @"kind": classMethod ? @"class" : @"instance",
             @"expected_encoding": @(expected),
             @"actual_encoding": actual ? @(actual) : @"none",
             @"matches": @(actual && strcmp(actual, expected) == 0)};
}

static BOOL allMethodsMatch(NSArray *records) {
    for (NSDictionary *record in records) if (![record[@"matches"] boolValue]) return NO;
    return YES;
}

static NSString *programText(void) {
    return @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "  func main<ios18>(tensor<fp16,[1,64,1,64]> x) {\n"
            "    fp16 two=const()[name=string(\"two\"),val=fp16(0x1p+1)];\n"
            "    tensor<fp16,[1,64,1,64]> y=mul(x=x,y=two)[name=string(\"twice\")];\n"
            "  } -> (y);\n"
            "}\n";
}

static void poison(IOSurfaceRef surface) {
    IOSurfaceLock(surface, 0, NULL);
    memset(IOSurfaceGetBaseAddress(surface), 0x7e, IOSurfaceGetAllocSize(surface));
    IOSurfaceUnlock(surface, 0, NULL);
}

static NSUInteger outputPoisonCount(IOSurfaceRef surface) {
    NSUInteger count = 0;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    const uint16_t *values = IOSurfaceGetBaseAddress(surface);
    for (NSUInteger i = 0; values && i < Elements; ++i) if (values[i] == 0x7e7e) ++count;
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return count;
}

static NSDictionary *oracle(IOSurfaceRef input, IOSurfaceRef output, NSUInteger repetition) {
    NSUInteger inputMismatches = 0, outputMismatches = 0, finite = 0;
    double maxError = 0;
    uint64_t checksum = 1469598103934665603ULL;
    IOSurfaceLock(input, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
    const __fp16 *in = IOSurfaceGetBaseAddress(input);
    const __fp16 *out = IOSurfaceGetBaseAddress(output);
    for (NSUInteger i = 0; in && out && i < Elements; ++i) {
        NSUInteger channel = i / Positions, position = i % Positions;
        float expectedInput = (float)((__fp16)((float)channel * 0.03125f +
            (float)position * 0.0078125f - 1.0f + (float)repetition * 0.125f));
        float expectedOutput = (float)((__fp16)(expectedInput * 2.0f));
        float actualInput = (float)in[i], actualOutput = (float)out[i];
        if (fabs((double)actualInput - expectedInput) > 0.001) ++inputMismatches;
        double error = fabs((double)actualOutput - expectedOutput);
        if (error > maxError) maxError = error;
        if (error > 0.001) ++outputMismatches;
        if (isfinite(actualOutput)) ++finite;
        const uint8_t *bytes = (const uint8_t *)&out[i];
        checksum ^= bytes[0]; checksum *= 1099511628211ULL;
        checksum ^= bytes[1]; checksum *= 1099511628211ULL;
    }
    IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(input, kIOSurfaceLockReadOnly, NULL);
    return @{@"valid": @(inputMismatches == 0 && outputMismatches == 0 && finite == Elements),
             @"input_mismatches": @(inputMismatches),
             @"output_mismatches": @(outputMismatches),
             @"finite_outputs": @(finite),
             @"max_absolute_error": @(maxError),
             @"checksum": [NSString stringWithFormat:@"0x%016llx", checksum]};
}

static NSString *requestDescription(ANEFieldfareANEProgram *program) {
    CFTypeRef value = ANEFieldfareANEProgramCopyRequestDescription(program, 0);
    if (!value) return @"none";
    NSString *result = [(__bridge NSString *)value copy];
    CFRelease(value);
    return result;
}

int main(void) {
    @autoreleasepool {
        void *framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                                RTLD_NOW | RTLD_LOCAL);
        Class signalClass = NSClassFromString(@"_ANESharedSignalEvent");
        Class waitClass = NSClassFromString(@"_ANESharedWaitEvent");
        Class eventsClass = NSClassFromString(@"_ANESharedEvents");
        Class requestClass = NSClassFromString(@"_ANERequest");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLSharedEvent> producerEvent = [device newSharedEvent];
        id<MTLSharedEvent> blockerEvent = [device newSharedEvent];
        Class metalEventClass = producerEvent ? [(id)producerEvent class] : Nil;
        NSArray *abi = @[
            methodRecord(metalEventClass, NSSelectorFromString(@"IOSurfaceSharedEvent"), NO, "@16@0:8"),
            methodRecord(signalClass, @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
                         YES, "@44@0:8Q16I24q28@36"),
            methodRecord(signalClass, @selector(waitEvent), NO, "@16@0:8"),
            methodRecord(waitClass, @selector(waitEventWithValue:sharedEvent:eventType:),
                         YES, "@40@0:8Q16@24Q32"),
            methodRecord(eventsClass, @selector(sharedEventsWithSignalEvents:waitEvents:),
                         YES, "@32@0:8@16@24"),
            methodRecord(requestClass, @selector(setSharedEvents:), NO, "v24@0:8@16"),
            methodRecord(requestClass, @selector(sharedEvents), NO, "@16@0:8"),
        ];
        BOOL abiValid = framework && device && producerEvent && blockerEvent && allMethodsMatch(abi);

        const char *creationError = NULL;
        ANEFieldfareANEProgram *program = abiValid ? ANEFieldfareANEProgramCreate(
            programText().UTF8String, Elements * sizeof(__fp16), Elements * sizeof(__fp16),
            1, &creationError) : NULL;
        IOSurfaceRef input = program ? ANEFieldfareANEProgramInput(program, 0) : NULL;
        IOSurfaceRef output = program ? ANEFieldfareANEProgramOutput(program, 0) : NULL;
        void *inputBase = input ? IOSurfaceGetBaseAddress(input) : NULL;
        size_t inputBytes = input ? IOSurfaceGetAllocSize(input) : 0;
        /* Metal accepts this ANE IOSurface allocation at its observed 4 KiB
           mapping granularity even though the host VM page is larger. */
        BOOL aligned = inputBase && (uintptr_t)inputBase % 4096 == 0 &&
            inputBytes % 4096 == 0;
        id<MTLBuffer> inputBuffer = aligned ? [device newBufferWithBytesNoCopy:inputBase
            length:inputBytes options:MTLResourceStorageModeShared deallocator:nil] : nil;
        BOOL pointerIdentity = inputBuffer && inputBuffer.contents == inputBase;

        NSError *metalError = nil;
        NSString *source = @"#include <metal_stdlib>\nusing namespace metal; "
            "kernel void produce(device half *x [[buffer(0)]], constant uint &r [[buffer(1)]], "
            "uint i [[thread_position_in_grid]]) { if (i < 4096) { uint c=i/64, p=i%64; "
            "x[i]=half(float(c)*0.03125f+float(p)*0.0078125f-1.0f+float(r)*0.125f); } }";
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&metalError];
        id<MTLFunction> function = [library newFunctionWithName:@"produce"];
        id<MTLComputePipelineState> pipeline = function ?
            [device newComputePipelineStateWithFunction:function error:&metalError] : nil;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        BOOL valid = program && input && output && inputBuffer && pointerIdentity &&
            library && pipeline && queue;
        id ioProducerEvent = valid ? ((id (*)(id, SEL))objc_msgSend)(
            producerEvent, NSSelectorFromString(@"IOSurfaceSharedEvent")) : nil;
        NSMutableArray *runs = [NSMutableArray array];
        NSString *before = program ? requestDescription(program) : @"none";
        BOOL beforeHasNoSharedEvents = [before containsString:@"sharedEvents=(null)"];
        BOOL afterHasWaitEvent = NO;

        for (NSUInteger repetition = 0; valid && repetition < Repetitions; ++repetition) {
            uint64_t value = repetition + 1;
            id signalTemplate = ((id (*)(Class, SEL, uint64_t, uint32_t, int64_t, id))objc_msgSend)(
                signalClass, @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),
                value, 0, 0, ioProducerEvent);
            id runtimeWait = signalTemplate ? ((id (*)(id, SEL))objc_msgSend)(
                signalTemplate, @selector(waitEvent)) : nil;
            id eventSet = runtimeWait ? ((id (*)(Class, SEL, id, id))objc_msgSend)(
                eventsClass, @selector(sharedEventsWithSignalEvents:waitEvents:), @[], @[runtimeWait]) : nil;
            const char *configurationError = NULL;
            BOOL configured = eventSet && ANEFieldfareANEProgramSetRequestSharedEvents(
                program, 0, (__bridge CFTypeRef)eventSet, &configurationError);
            if (repetition == 0 && configured)
                afterHasWaitEvent = [requestDescription(program) containsString:@"_ANESharedWaitEvent"];

            poison(input); poison(output);
            id<MTLCommandBuffer> command = [queue commandBuffer];
            [command encodeWaitForEvent:blockerEvent value:value];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            uint32_t repetitionValue = (uint32_t)repetition;
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:inputBuffer offset:0 atIndex:0];
            [encoder setBytes:&repetitionValue length:sizeof(repetitionValue) atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(Elements, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,
                   pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
            [encoder endEncoding];
            [command encodeSignalEvent:producerEvent value:value];
            [command commit];

            const char *beginError = NULL, *waitError = NULL;
            ANEFieldfareANEEvaluationTiming beginTiming = {0}, completionTiming = {0};
            BOOL began = configured && ANEFieldfareANEProgramEvaluateBegin(
                program, 0, &beginTiming, &beginError);
            usleep(5000);
            uint32_t inFlightBeforeRelease = ANEFieldfareANEProgramCurrentAsyncRequestsInFlight(program);
            NSUInteger poisonBeforeRelease = outputPoisonCount(output);
            uint64_t producerValueBeforeRelease = producerEvent.signaledValue;
            BOOL metalBlocked = command.status != MTLCommandBufferStatusCompleted;
            BOOL negative = began && metalBlocked && inFlightBeforeRelease == 1 &&
                poisonBeforeRelease == Elements && producerValueBeforeRelease < value;

            blockerEvent.signaledValue = value;
            BOOL aneCompleted = began && ANEFieldfareANEProgramEvaluateWait(
                program, 0, 5ULL * NSEC_PER_SEC, &completionTiming, &waitError);
            [command waitUntilCompleted];
            NSDictionary *check = oracle(input, output, repetition);
            BOOL runValid = configured && runtimeWait && negative && aneCompleted &&
                command.status == MTLCommandBufferStatusCompleted &&
                producerEvent.signaledValue >= value && [check[@"valid"] boolValue];
            [runs addObject:@{
                @"repetition": @(repetition), @"event_value": @(value),
                @"runtime_authored_wait_event_class": runtimeWait ? NSStringFromClass([runtimeWait class]) : @"none",
                @"runtime_authored_wait_event_value": @(value),
                @"request_configured": @(configured), @"ane_begin_accepted": @(began),
                @"negative_control": @{
                    @"delay_ms": @5, @"metal_command_still_blocked": @(metalBlocked),
                    @"ane_in_flight_before_release": @(inFlightBeforeRelease),
                    @"output_poison_values_before_release": @(poisonBeforeRelease),
                    @"producer_event_value_before_release": @(producerValueBeforeRelease),
                    @"passed": @(negative)},
                @"host_released_only_metal_blocker": @YES,
                @"metal_completed": @(command.status == MTLCommandBufferStatusCompleted),
                @"producer_event_value_after_completion": @(producerEvent.signaledValue),
                @"ane_completed_after_event": @(aneCompleted), @"oracle": check,
                @"configuration_error": configurationError ? @(configurationError) : @"none",
                @"begin_error": beginError ? @(beginError) : @"none",
                @"wait_error": waitError ? @(waitError) : @"none", @"valid": @(runValid)}];
            free((void *)configurationError); free((void *)beginError); free((void *)waitError);
            valid = valid && runValid;
        }

        ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);
        BOOL validated = valid && runs.count == Repetitions;
        for (NSDictionary *run in runs) validated &= [run[@"valid"] boolValue];
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.metal-ane-event.v1",
            @"status": validated ? @"validated" : @"rejected",
            @"stage": validated ? @"metal_signal_to_ane_wait_oracle" : @"execution",
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"ane_architecture": @"Apple M5 / H17G",
            @"abi_checks": abi,
            @"objects": @{
                @"metal_producer_event_class": producerEvent ? NSStringFromClass([(id)producerEvent class]) : @"none",
                @"iosurface_shared_event_same_identity": @(producerEvent && ioProducerEvent &&
                    (id)producerEvent == ioProducerEvent),
                @"wait_construction": @"runtime-derived by -[_ANESharedSignalEvent waitEvent]",
                @"event_type": @0, @"input_symbol_index": @0},
            @"request": @{@"before_shared_events_nil": @(beforeHasNoSharedEvents),
                @"after_contains_runtime_wait_event": @(afterHasWaitEvent),
                @"transaction_handle_policy": @"runtime-owned; no nonzero handle synthesized"},
            @"guards": @{@"input_4k_aligned_and_length_multiple": @(aligned),
                @"metal_buffer_contents_matches_ane_input": @(pointerIdentity),
                @"surface_buffer_event_program_lifetime_through_completion": @YES},
            @"lifecycle": @{@"ane_compile_count": program ? @1 : @0,
                @"ane_load_count": program ? @1 : @0,
                @"ane_compile_ms": @(info.compileMilliseconds), @"ane_load_ms": @(info.loadMilliseconds),
                @"metal_library_compile_count": library ? @1 : @0,
                @"metal_pipeline_load_count": pipeline ? @1 : @0},
            @"ordering": @{
                @"submission_order": @"commit blocked Metal producer; submit ANE wait; inspect negative; release Metal blocker; wait for ANE downstream completion",
                @"host_semaphore_between_metal_and_ane": @NO,
                @"host_wait_before_release": @NO,
                @"host_wait_after_release": @"ANE completion, then Metal completion for cleanup"},
            @"runs": runs,
            @"error": creationError ? @(creationError) : @"none",
            @"metal_error": metalError.localizedDescription ?: @"none",
            @"claim_boundary": @"A Metal compute producer writes the directly wrapped ANE input IOSurface and signals an MTLSharedEvent. The mapped ANE request carries a runtime-derived _ANESharedWaitEvent for the same object/value and is submitted while Metal is blocked. During the bounded pre-signal control ANE remains in flight and all output poison is intact; after releasing Metal, all 4,096 input and doubled output values pass independent exact oracles. This validates Metal-to-ANE shared-event ordering for one same-process request on this M5 host, not concurrent surface mutation, transaction handles, cross-process events, or general pipeline speedup."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
            options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        free((void *)creationError);
        if (program) ANEFieldfareANEProgramDestroy(program);
        return validated ? 0 : 1;
    }
}
