#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <sys/sysctl.h>
#import <unistd.h>

#include <math.h>

enum {
    GateWidth = 64,
    GatePositions = 64,
    GateOutputSetCount = 2,
};

static mach_timebase_info_data_t timebaseInfo;

static double milliseconds(uint64_t ticks) {
    return (double)ticks * timebaseInfo.numer / timebaseInfo.denom / 1e6;
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static NSString *errorText(NSError *error) {
    return error ? [NSString stringWithFormat:@"%@ (%@:%ld)",
                    error.localizedDescription ?: @"unknown",
                    error.domain ?: @"unknown", (long)error.code] : @"none";
}

static IOSurfaceRef makeSurface(NSUInteger bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    });
}

static NSData *diagonalWeightBlob(float scale) {
    const NSUInteger payloadBytes = GateWidth * GateWidth * sizeof(__fp16);
    NSMutableData *blob = [NSMutableData dataWithLength:128 + payloadBytes];
    uint8_t *bytes = blob.mutableBytes;
    bytes[0] = 1;
    bytes[4] = 2;
    bytes[64] = 0xef;
    bytes[65] = 0xbe;
    bytes[66] = 0xad;
    bytes[67] = 0xde;
    bytes[68] = 1;
    *(uint32_t *)(bytes + 72) = (uint32_t)payloadBytes;
    *(uint32_t *)(bytes + 80) = 128;
    __fp16 *weights = (__fp16 *)(bytes + 128);
    for (NSUInteger index = 0; index < GateWidth; ++index)
        weights[index * GateWidth + index] = (__fp16)scale;
    return blob;
}

static NSString *recurrenceMIL(void) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "  func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
         "    string fp16_name = const()[name = string(\"fp16_name\"), val = string(\"fp16\")];\n"
         "    tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = fp16_name, x = x)[name = string(\"cast_in\")];\n"
         "    tensor<fp16, [%d, %d, 1, 1]> weight = const()[name = string(\"weight\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
         "    tensor<int32, [2]> strides = const()[name = string(\"strides\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    tensor<int32, [4]> pad = const()[name = string(\"pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "    tensor<int32, [2]> dilations = const()[name = string(\"dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
         "    int32 groups = const()[name = string(\"groups\"), val = int32(1)];\n"
         "    string valid = const()[name = string(\"valid\"), val = string(\"valid\")];\n"
         "    tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = dilations, groups = groups, pad = pad, pad_type = valid, strides = strides, weight = weight, x = x16)[name = string(\"projection\")];\n"
         "    string fp32_name = const()[name = string(\"fp32_name\"), val = string(\"fp32\")];\n"
         "    tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = fp32_name, x = y16)[name = string(\"cast_out\")];\n"
         "  } -> (y);\n"
         "}\n",
         GateWidth, GatePositions, GateWidth, GatePositions,
         GateWidth, GateWidth, GateWidth, GateWidth,
         GateWidth, GatePositions, GateWidth, GatePositions];
}

static void fillInput(IOSurfaceRef surface) {
    IOSurfaceLock(surface, 0, NULL);
    float *values = IOSurfaceGetBaseAddress(surface);
    for (NSUInteger channel = 0; channel < GateWidth; ++channel) {
        for (NSUInteger position = 0; position < GatePositions; ++position) {
            values[channel * GatePositions + position] =
                ((float)((channel * 17 + position * 13) % 97) - 48.0f) / 64.0f;
        }
    }
    IOSurfaceUnlock(surface, 0, NULL);
}

static void poisonOutput(IOSurfaceRef surface) {
    IOSurfaceLock(surface, 0, NULL);
    uint32_t *values = IOSurfaceGetBaseAddress(surface);
    NSUInteger count = GateWidth * GatePositions;
    for (NSUInteger index = 0; index < count; ++index) values[index] = 0x7fc00001U;
    IOSurfaceUnlock(surface, 0, NULL);
}

static NSDictionary *validateOutput(IOSurfaceRef surface, float multiplier) {
    double maxError = 0.0;
    NSUInteger mismatches = 0;
    NSUInteger finiteValues = 0;
    uint64_t checksum = 1469598103934665603ULL;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    const float *values = IOSurfaceGetBaseAddress(surface);
    for (NSUInteger channel = 0; channel < GateWidth; ++channel) {
        for (NSUInteger position = 0; position < GatePositions; ++position) {
            float input = ((float)((channel * 17 + position * 13) % 97) - 48.0f) / 64.0f;
            float expected = (float)((__fp16)((__fp16)input * (__fp16)multiplier));
            float actual = values[channel * GatePositions + position];
            if (isfinite(actual)) finiteValues++;
            double error = isfinite(actual) ? fabs((double)actual - expected) : INFINITY;
            maxError = fmax(maxError, error);
            if (error > 0.004) mismatches++;
            uint32_t bits = 0;
            memcpy(&bits, &actual, sizeof(bits));
            checksum ^= bits;
            checksum *= 1099511628211ULL;
        }
    }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return @{
        @"expected_multiplier": @(multiplier),
        @"max_absolute_error": @(maxError),
        @"mismatches": @(mismatches),
        @"finite_values": @(finiteValues),
        @"checksum": [NSString stringWithFormat:@"0x%016llx", checksum],
    };
}

static NSDictionary *checkRecord(Class cls, SEL selector, BOOL classMethod, const char *encoding) {
    Method method = classMethod ? class_getClassMethod(cls, selector) : class_getInstanceMethod(cls, selector);
    const char *actual = method ? method_getTypeEncoding(method) : NULL;
    return @{
        @"class": cls ? NSStringFromClass(cls) : @"missing",
        @"selector": NSStringFromSelector(selector),
        @"expected_encoding": [NSString stringWithUTF8String:encoding],
        @"actual_encoding": actual ? [NSString stringWithUTF8String:actual] : @"missing",
        @"matches": @(actual && strcmp(actual, encoding) == 0),
    };
}

static BOOL allChecksMatch(NSArray<NSDictionary *> *checks) {
    for (NSDictionary *check in checks) if (![check[@"matches"] boolValue]) return NO;
    return YES;
}

static NSString *machineName(void) {
    size_t size = 0;
    if (sysctlbyname("machdep.cpu.brand_string", NULL, &size, NULL, 0) != 0 || size < 2)
        return @"unknown";
    NSMutableData *storage = [NSMutableData dataWithLength:size];
    if (sysctlbyname("machdep.cpu.brand_string", storage.mutableBytes, &size, NULL, 0) != 0)
        return @"unknown";
    return [NSString stringWithUTF8String:storage.bytes] ?: @"unknown";
}

int main(void) {
    @autoreleasepool {
        mach_timebase_info(&timebaseInfo);
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);

        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class inMemoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        Class clientClass = NSClassFromString(@"_ANEClient");
        Class bufferClass = NSClassFromString(@"_ANEBuffer");
        Class outputSetsClass = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        Class chainingClass = NSClassFromString(@"_ANEChainingRequest");
        Class inputReadyClass = NSClassFromString(@"_ANEInputBuffersReady");
        Class outputEnqueueClass = NSClassFromString(@"_ANEOutputSetEnqueue");

        NSArray<NSDictionary *> *abiChecks = @[
            checkRecord(descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), YES, "@40@0:8@16@24@32"),
            checkRecord(inMemoryModelClass, @selector(inMemoryModelWithDescriptor:), YES, "@24@0:8@16"),
            checkRecord(inMemoryModelClass, @selector(compileWithQoS:options:error:), NO, "B36@0:8I16@20^@28"),
            checkRecord(inMemoryModelClass, @selector(loadWithQoS:options:error:), NO, "B36@0:8I16@20^@28"),
            checkRecord(inMemoryModelClass, @selector(evaluateWithQoS:options:request:error:), NO, "B44@0:8I16@20@28^@36"),
            checkRecord(inMemoryModelClass, @selector(unloadWithQoS:error:), NO, "B28@0:8I16^@20"),
            checkRecord(surfaceClass, @selector(objectWithIOSurface:), YES, "@24@0:8^{__IOSurface=}16"),
            checkRecord(requestClass, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:), YES, "@72@0:8@16@24@32@40@48@56@64"),
            checkRecord(clientClass, @selector(initWithRestrictedAccessAllowed:), NO, "@20@0:8B16"),
            checkRecord(bufferClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:), YES, "@40@0:8@16@24q32"),
            checkRecord(outputSetsClass, @selector(objectWithstatsSurRef:outputBuffer:), YES, "@32@0:8^{__IOSurface=}16@24"),
            checkRecord(chainingClass, @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:), YES, "@88@0:8@16@24@32@40@48@56@64@72@80"),
            checkRecord(chainingClass, @selector(validate), NO, "B16@0:8"),
            checkRecord(inputReadyClass, @selector(inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:), YES, "@44@0:8I16@20@28Q36"),
            checkRecord(inputReadyClass, @selector(validate), NO, "B16@0:8"),
            checkRecord(outputEnqueueClass, @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:), YES, "@40@0:8I16I20Q24B32B36"),
            checkRecord(clientClass, @selector(prepareChainingWithModel:options:chainingReq:qos:error:), NO, "B52@0:8@16@24@32I40^@44"),
            checkRecord(clientClass, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:), NO, "B52@0:8@16@24@32I40^@44"),
            checkRecord(clientClass, @selector(enqueueSetsWithModel:outputSet:options:qos:error:), NO, "B52@0:8@16@24@32I40^@44"),
        ];
        if (!framework || !allChecksMatch(abiChecks)) {
            emit(@{
                @"schema": @"ane-fieldfare.chaining-gate.v1",
                @"status": @"rejected",
                @"stage": @"runtime_abi",
                @"framework_loaded": @(framework != NULL),
                @"abi_checks": abiChecks,
                @"claim_boundary": @"Private selector ABI gate only; no chaining call was made.",
            });
            return 1;
        }

        NSString *milString = recurrenceMIL();
        NSData *mil = [milString dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weights = diagonalWeightBlob(2.0f);
        NSDictionary *weightDictionary = @{
            @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weights}
        };
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, weightDictionary, nil);
        id inMemoryModel = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
            inMemoryModelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        if (!inMemoryModel) {
            emit(@{@"schema": @"ane-fieldfare.chaining-gate.v1", @"status": @"rejected",
                   @"stage": @"model", @"abi_checks": abiChecks,
                   @"claim_boundary": @"Model construction failed before ANE execution."});
            return 1;
        }

        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(inMemoryModel, @selector(hexStringIdentifier));
        NSString *temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *files = NSFileManager.defaultManager;
        [files createDirectoryAtPath:[temporary stringByAppendingPathComponent:@"weights"]
         withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weights writeToFile:[temporary stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *error = nil;
        uint64_t started = mach_absolute_time();
        BOOL compiled = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            inMemoryModel, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        double compileMs = milliseconds(mach_absolute_time() - started);
        if (!compiled) {
            emit(@{@"schema": @"ane-fieldfare.chaining-gate.v1", @"status": @"rejected",
                   @"stage": @"compile", @"detail": errorText(error), @"compile_count": @1,
                   @"load_count": @0, @"abi_checks": abiChecks,
                   @"claim_boundary": @"The exact recurrence MIL construction failed to compile."});
            [files removeItemAtPath:temporary error:nil];
            return 1;
        }

        error = nil;
        started = mach_absolute_time();
        BOOL loaded = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            inMemoryModel, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        double loadMs = milliseconds(mach_absolute_time() - started);
        if (!loaded) {
            emit(@{@"schema": @"ane-fieldfare.chaining-gate.v1", @"status": @"rejected",
                   @"stage": @"load", @"detail": errorText(error), @"compile_count": @1,
                   @"load_count": @1, @"abi_checks": abiChecks,
                   @"claim_boundary": @"The exact recurrence MIL construction compiled but failed to load."});
            [files removeItemAtPath:temporary error:nil];
            return 1;
        }

        const NSUInteger tensorBytes = GateWidth * GatePositions * sizeof(float);
        IOSurfaceRef controlInput = makeSurface(tensorBytes);
        IOSurfaceRef controlOutput = makeSurface(tensorBytes);
        IOSurfaceRef chainInput = makeSurface(tensorBytes);
        IOSurfaceRef chainOutputs[GateOutputSetCount] = {makeSurface(tensorBytes), makeSurface(tensorBytes)};
        IOSurfaceRef stats[GateOutputSetCount] = {makeSurface(4096), makeSurface(4096)};
        BOOL surfacesReady = controlInput && controlOutput && chainInput && chainOutputs[0] &&
            chainOutputs[1] && stats[0] && stats[1];

        NSMutableDictionary *result = [@{
            @"schema": @"ane-fieldfare.chaining-gate.v1",
            @"status": @"rejected",
            @"stage": surfacesReady ? @"control" : @"iosurface_allocation",
            @"compile_count": @1,
            @"load_count": @1,
            @"compile_ms": @(compileMs),
            @"load_ms": @(loadMs),
            @"abi_checks": abiChecks,
            @"shape": @{@"channels": @(GateWidth), @"positions": @(GatePositions)},
        } mutableCopy];

        BOOL controlAccepted = NO;
        NSDictionary *controlOracle = @{};
        BOOL chainRequestValid = NO;
        BOOL inputReadyValid = NO;
        BOOL prepareAccepted = NO;
        BOOL buffersReadyAccepted = NO;
        NSMutableArray *enqueueRecords = [NSMutableArray array];
        NSMutableArray *chainOracles = [NSMutableArray array];

        if (surfacesReady) {
            fillInput(controlInput);
            poisonOutput(controlOutput);
            id wrappedControlInput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), controlInput);
            id wrappedControlOutput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), controlOutput);
            id controlRequest = ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
                requestClass,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wrappedControlInput], @[@0], @[wrappedControlOutput], @[@0], nil, nil, @0);
            error = nil;
            controlAccepted = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                inMemoryModel, @selector(evaluateWithQoS:options:request:error:), 21, @{}, controlRequest, &error);
            controlOracle = validateOutput(controlOutput, 2.0f);
            result[@"control"] = @{
                @"evaluate_accepted": @(controlAccepted),
                @"error": errorText(error),
                @"oracle": controlOracle,
            };

            if (controlAccepted && [controlOracle[@"mismatches"] unsignedIntegerValue] == 0) {
                fillInput(chainInput);
                for (NSUInteger index = 0; index < GateOutputSetCount; ++index)
                    poisonOutput(chainOutputs[index]);

                id wrappedChainInput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                    surfaceClass, @selector(objectWithIOSurface:), chainInput);
                id inputBuffer = ((id (*)(Class, SEL, id, id, int64_t))objc_msgSend)(
                    bufferClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                    wrappedChainInput, @0, 0);
                NSMutableArray *outputSets = [NSMutableArray array];
                for (NSUInteger index = 0; index < GateOutputSetCount; ++index) {
                    id wrappedOutput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                        surfaceClass, @selector(objectWithIOSurface:), chainOutputs[index]);
                    id outputBuffer = ((id (*)(Class, SEL, id, id, int64_t))objc_msgSend)(
                        bufferClass, @selector(bufferWithIOSurfaceObject:symbolIndex:source:),
                        wrappedOutput, @0, 1);
                    id outputSet = ((id (*)(Class, SEL, IOSurfaceRef, id))objc_msgSend)(
                        outputSetsClass, @selector(objectWithstatsSurRef:outputBuffer:),
                        stats[index], @[outputBuffer]);
                    if (outputSet) [outputSets addObject:outputSet];
                }

                id chainingRequest = outputSets.count == GateOutputSetCount ?
                    ((id (*)(Class, SEL, id, id, id, id, id, id, id, id, id))objc_msgSend)(
                        chainingClass,
                        @selector(chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:),
                        @[inputBuffer], outputSets, @[@0], @[@0], @0, nil, @1, @0, @0) : nil;
                chainRequestValid = chainingRequest && ((BOOL (*)(id, SEL))objc_msgSend)(
                    chainingRequest, @selector(validate));

                id inputReady = ((id (*)(Class, SEL, unsigned int, id, id, uint64_t))objc_msgSend)(
                    inputReadyClass,
                    @selector(inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:),
                    0, @[@0], @[@0], 0);
                inputReadyValid = inputReady && ((BOOL (*)(id, SEL))objc_msgSend)(inputReady, @selector(validate));

                id aneModel = ((id (*)(id, SEL))objc_msgSend)(inMemoryModel, @selector(model));
                id chainingClient = ((id (*)(id, SEL, BOOL))objc_msgSend)(
                    [clientClass alloc], @selector(initWithRestrictedAccessAllowed:), YES);

                error = nil;
                if (chainRequestValid && inputReadyValid && aneModel && chainingClient) {
                    prepareAccepted = ((BOOL (*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
                        chainingClient, @selector(prepareChainingWithModel:options:chainingReq:qos:error:),
                        aneModel, @{}, chainingRequest, 21, &error);
                }
                result[@"chaining_request"] = @{
                    @"object_present": @(chainingRequest != nil),
                    @"validate_passed": @(chainRequestValid),
                    @"input_ready_validate_passed": @(inputReadyValid),
                    @"prepare_accepted": @(prepareAccepted),
                    @"prepare_error": errorText(error),
                    @"client_path": @"daemon",
                    @"client_owner": @"fresh_restricted_client",
                    @"model_kind": @"in_memory_mil",
                };

                if (prepareAccepted) {
                    error = nil;
                    buffersReadyAccepted = ((BOOL (*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
                        chainingClient, @selector(buffersReadyWithModel:inputBuffers:options:qos:error:),
                        aneModel, inputReady, @{}, 21, &error);
                    result[@"buffers_ready"] = @{
                        @"accepted": @(buffersReadyAccepted),
                        @"error": errorText(error),
                        @"argument": @"_ANEInputBuffersReady object",
                    };
                }

                if (buffersReadyAccepted) {
                    for (NSUInteger index = 0; index < GateOutputSetCount; ++index) {
                        id enqueue = ((id (*)(Class, SEL, unsigned int, unsigned int, uint64_t, BOOL, BOOL))objc_msgSend)(
                            outputEnqueueClass,
                            @selector(outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:),
                            0, (unsigned int)index, (uint64_t)(index + 1), NO, NO);
                        error = nil;
                        BOOL accepted = enqueue && ((BOOL (*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
                            chainingClient, @selector(enqueueSetsWithModel:outputSet:options:qos:error:),
                            aneModel, enqueue, @{}, 21, &error);
                        [enqueueRecords addObject:@{
                            @"set_index": @(index),
                            @"accepted": @(accepted),
                            @"error": errorText(error),
                            @"order": @"after_buffers_ready",
                        }];
                        if (!accepted) break;
                    }
                }
                if (buffersReadyAccepted && enqueueRecords.count == GateOutputSetCount) {
                    for (NSUInteger attempt = 0; attempt < 2000; ++attempt) {
                        NSDictionary *second = validateOutput(chainOutputs[1], 4.0f);
                        if ([second[@"mismatches"] unsignedIntegerValue] == 0) break;
                        usleep(1000);
                    }
                    [chainOracles addObject:validateOutput(chainOutputs[0], 2.0f)];
                    [chainOracles addObject:validateOutput(chainOutputs[1], 4.0f)];
                }
            }
        }

        BOOL enqueueAccepted = enqueueRecords.count == GateOutputSetCount;
        for (NSDictionary *record in enqueueRecords)
            enqueueAccepted = enqueueAccepted && [record[@"accepted"] boolValue];
        BOOL numerical = chainOracles.count == GateOutputSetCount;
        for (NSDictionary *oracle in chainOracles)
            numerical = numerical && [oracle[@"mismatches"] unsignedIntegerValue] == 0;
        BOOL validated = controlAccepted && [controlOracle[@"mismatches"] unsignedIntegerValue] == 0 &&
            chainRequestValid && inputReadyValid && prepareAccepted && buffersReadyAccepted &&
            enqueueAccepted && numerical;

        result[@"enqueue_sets"] = enqueueRecords;
        result[@"chain_oracles"] = chainOracles;
        result[@"status"] = validated ? @"validated" : @"rejected";
        if (!surfacesReady) result[@"stage"] = @"iosurface_allocation";
        else if (!controlAccepted || [controlOracle[@"mismatches"] unsignedIntegerValue] != 0) result[@"stage"] = @"control";
        else if (!chainRequestValid || !inputReadyValid) result[@"stage"] = @"request_validation";
        else if (!prepareAccepted) result[@"stage"] = @"prepare_chaining";
        else if (!buffersReadyAccepted) result[@"stage"] = @"buffers_ready";
        else if (!enqueueAccepted) result[@"stage"] = @"enqueue_sets";
        else result[@"stage"] = numerical ? @"complete" : @"numerical_oracle";
        result[@"host"] = NSProcessInfo.processInfo.hostName ?: @"unknown";
        result[@"os"] = NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown";
        result[@"ane_architecture"] = machineName();
        result[@"compile_command"] = @"clang -fobjc-arc -O2 -Wall -Wextra -framework Foundation -framework IOSurface -ldl -o .build/ane-fieldfare-chaining-gate Sources/ANEFieldfareChainingGate/main.m";
        result[@"run_command"] = @".build/ane-fieldfare-chaining-gate";
        result[@"exit_status"] = validated ? @0 : @1;
        result[@"claim_boundary"] = validated
            ? @"One compiler-produced and loaded private-ANE program executed a two-step y=2x recurrence through _ANEChainingRequest. The first output matched 2x and the loopback output matched 4x without the host copying the first output into the second input. This validates only the tested single-procedure, two-output-set firmware chaining lifecycle on this host; it does not validate mutable weights, cross-program chaining, Gemma KV state, or generation."
            : @"The ordinary private-ANE y=2x control passed, but the tested _ANEChainingRequest lifecycle did not reach a two-step numerical oracle. Any accepted earlier call is not functional chaining proof.";

        error = nil;
        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            inMemoryModel, @selector(unloadWithQoS:error:), 21, &error);
        [files removeItemAtPath:temporary error:nil];
        if (controlInput) CFRelease(controlInput);
        if (controlOutput) CFRelease(controlOutput);
        if (chainInput) CFRelease(chainInput);
        for (NSUInteger index = 0; index < GateOutputSetCount; ++index) {
            if (chainOutputs[index]) CFRelease(chainOutputs[index]);
            if (stats[index]) CFRelease(stats[index]);
        }

        emit(result);
        return validated ? 0 : 1;
    }
}
