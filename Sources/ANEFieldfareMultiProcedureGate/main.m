#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>

typedef struct {
    IOSurfaceRef surface;
    uint32_t field1;
    int32_t field2;
    int32_t field3;
    uint32_t field4;
} ANEBufferStructProbe;

typedef struct {
    ANEBufferStructProbe buffers[128];
    uint64_t field1;
    uint32_t field2;
    uint32_t field3;
    uint64_t field4;
} ANEMemoryMappingParamsProbe;

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

static id callObject(id object, SEL selector) {
    return object && [object respondsToSelector:selector]
        ? ((id (*)(id, SEL))objc_msgSend)(object, selector) : nil;
}

static uint64_t callUInt64(id object, SEL selector) {
    return object && [object respondsToSelector:selector]
        ? ((uint64_t (*)(id, SEL))objc_msgSend)(object, selector) : 0;
}

static uint64_t checksumValues(const uint8_t *base, NSUInteger count, NSUInteger stride) {
    uint64_t checksum = 1469598103934665603ULL;
    for (NSUInteger index = 0; index < count; ++index) {
        const uint8_t *bytes = base + index * stride;
        for (NSUInteger byte = 0; byte < sizeof(float); ++byte) {
            checksum ^= bytes[byte];
            checksum *= 1099511628211ULL;
        }
    }
    return checksum;
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static NSData *diagonalWeightBlob(NSUInteger width, float scale) {
    const NSUInteger payloadBytes = width * width * sizeof(__fp16);
    NSMutableData *blob = [NSMutableData dataWithLength:128 + payloadBytes];
    uint8_t *bytes = blob.mutableBytes;
    bytes[0] = 1; bytes[4] = 2;
    bytes[64] = 0xef; bytes[65] = 0xbe; bytes[66] = 0xad; bytes[67] = 0xde;
    bytes[68] = 1;
    *(uint32_t *)(bytes + 72) = (uint32_t)payloadBytes;
    *(uint32_t *)(bytes + 80) = 128;
    __fp16 *weights = (__fp16 *)(bytes + 128);
    for (NSUInteger index = 0; index < width; ++index)
        weights[index * width + index] = (__fp16)scale;
    return blob;
}

static NSString *multiFunctionMIL(NSUInteger elements) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "  func main<ios18>(tensor<fp32, [1, %lu, 1, 1]> x0) {\n"
         "    string f16_0=const()[name=string(\"f16_0\"),val=string(\"fp16\")];\n"
         "    tensor<fp16,[1,%lu,1,1]> x16_0=cast(dtype=f16_0,x=x0)[name=string(\"cast_in_0\")];\n"
         "    tensor<fp16,[%lu,%lu,1,1]> w0=const()[name=string(\"w0\"),val=tensor<fp16,[%lu,%lu,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"),offset=uint64(64)))];\n"
         "    tensor<int32,[2]> st0=const()[name=string(\"st0\"),val=tensor<int32,[2]>([1,1])];\n"
         "    tensor<int32,[4]> pd0=const()[name=string(\"pd0\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[2]> dl0=const()[name=string(\"dl0\"),val=tensor<int32,[2]>([1,1])];\n"
         "    int32 gr0=const()[name=string(\"gr0\"),val=int32(1)];\n"
         "    string pt0=const()[name=string(\"pt0\"),val=string(\"valid\")];\n"
         "    tensor<fp16,[1,%lu,1,1]> y16_0=conv(dilations=dl0,groups=gr0,pad=pd0,pad_type=pt0,strides=st0,weight=w0,x=x16_0)[name=string(\"conv_0\")];\n"
         "    string f32_0=const()[name=string(\"f32_0\"),val=string(\"fp32\")];\n"
         "    tensor<fp32,[1,%lu,1,1]> y0=cast(dtype=f32_0,x=y16_0)[name=string(\"cast_out_0\")];\n"
         "  } -> (y0);\n"
         "  func expert_1<ios18>(tensor<fp32, [1, %lu, 1, 1]> x1) {\n"
         "    string f16_1=const()[name=string(\"f16_1\"),val=string(\"fp16\")];\n"
         "    tensor<fp16,[1,%lu,1,1]> x16_1=cast(dtype=f16_1,x=x1)[name=string(\"cast_in_1\")];\n"
         "    tensor<fp16,[%lu,%lu,1,1]> w1=const()[name=string(\"w1\"),val=tensor<fp16,[%lu,%lu,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"),offset=uint64(64)))];\n"
         "    tensor<int32,[2]> st1=const()[name=string(\"st1\"),val=tensor<int32,[2]>([1,1])];\n"
         "    tensor<int32,[4]> pd1=const()[name=string(\"pd1\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[2]> dl1=const()[name=string(\"dl1\"),val=tensor<int32,[2]>([1,1])];\n"
         "    int32 gr1=const()[name=string(\"gr1\"),val=int32(1)];\n"
         "    string pt1=const()[name=string(\"pt1\"),val=string(\"valid\")];\n"
         "    tensor<fp16,[1,%lu,1,1]> twice_1=conv(dilations=dl1,groups=gr1,pad=pd1,pad_type=pt1,strides=st1,weight=w1,x=x16_1)[name=string(\"conv_1\")];\n"
         "    fp16 extra_1=const()[name=string(\"extra_1\"),val=fp16(0x1.8p+0)];\n"
         "    tensor<fp16,[1,%lu,1,1]> y16_1=mul(x=twice_1,y=extra_1)[name=string(\"times_three\")];\n"
         "    string f32_1=const()[name=string(\"f32_1\"),val=string(\"fp32\")];\n"
         "    tensor<fp32,[1,%lu,1,1]> y1=cast(dtype=f32_1,x=y16_1)[name=string(\"cast_out_1\")];\n"
         "  } -> (y1);\n"
         "}\n",
         (unsigned long)elements, (unsigned long)elements,
         (unsigned long)elements, (unsigned long)elements,
         (unsigned long)elements, (unsigned long)elements,
         (unsigned long)elements, (unsigned long)elements,
         (unsigned long)elements, (unsigned long)elements,
         (unsigned long)elements, (unsigned long)elements,
         (unsigned long)elements, (unsigned long)elements,
         (unsigned long)elements, (unsigned long)elements,
         (unsigned long)elements];
}

int main(void) {
    @autoreleasepool {
        const NSUInteger elements = 64;
        const NSUInteger planeStride = 64;
        const NSUInteger bytes = elements * planeStride;
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        Class procedureDataClass = NSClassFromString(@"_ANEProcedureData");
        Class instanceParametersClass = NSClassFromString(@"_ANEModelInstanceParameters");
        Class programProcedureClass = NSClassFromString(@"_ANEProgramProcedurePriv");
        Class weightClass = NSClassFromString(@"_ANEWeight");
        if (!framework || !descriptorClass || !modelClass || !requestClass || !surfaceClass ||
            !procedureDataClass || !instanceParametersClass) {
            emit(@{@"schema": @"ane-fieldfare.two-procedure-execution.v1", @"status": @"rejected",
                   @"stage": @"runtime_surface"});
            return 1;
        }

        NSData *mil = [multiFunctionMIL(elements) dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weightBlob = diagonalWeightBlob(elements, 2.0f);
        NSDictionary *weightFiles = @{@"@model_path/weights/weight.bin":
            @{@"offset": @0, @"data": weightBlob}};
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, weightFiles, nil);
        id model = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        if (!model) {
            emit(@{@"schema": @"ane-fieldfare.two-procedure-execution.v1", @"status": @"rejected",
                   @"stage": @"model_construction"});
            return 1;
        }

        NSString *identifier = callObject(model, @selector(hexStringIdentifier));
        NSString *temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        [NSFileManager.defaultManager createDirectoryAtPath:
            [temporary stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weightBlob writeToFile:[temporary stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSMutableDictionary *compileOptions = [NSMutableDictionary dictionary];
        NSString *procedureNameOption = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_PROC_NAME"];
        if (procedureNameOption.length) compileOptions[@"proc-name"] = procedureNameOption;
        NSError *error = nil;
        BOOL compiled = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, compileOptions, &error);
        if (!compiled) {
            emit(@{@"schema": @"ane-fieldfare.two-procedure-execution.v1", @"status": @"rejected",
                   @"stage": @"compile", @"error": error.localizedDescription ?: @"unknown",
                   @"claim_boundary": @"A two-function MIL program was offered to _ANEInMemoryModel; compilation rejection does not rule out package/AOT multi-function programs."});
            [NSFileManager.defaultManager removeItemAtPath:temporary error:nil];
            return 1;
        }
        NSTimeInterval holdAfterCompile = [NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_HOLD_AFTER_COMPILE"] doubleValue];
        if (holdAfterCompile > 0.0) [NSThread sleepForTimeInterval:holdAfterCompile];

        BOOL loaded = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        id compiledModel = callObject(model, @selector(model));
        NSArray *procedures = callObject(compiledModel, @selector(procedures));
        NSDictionary *attributes = callObject(model, @selector(modelAttributes));
        NSArray *attributeProcedures = attributes[@"ANEFModelDescription"][@"ANEFModelProcedures"];
        NSMutableArray *procedureRecords = [NSMutableArray array];
        for (id procedure in procedures ?: @[]) {
            [procedureRecords addObject:@{
                @"symbol": callObject(procedure, @selector(procedureSymbol)) ?: @"unknown",
                @"weights": @([callObject(procedure, @selector(weightArray)) count]),
            }];
        }
        NSMutableArray *procedureInfoRecords = [NSMutableArray array];
        for (NSUInteger index = 0; index < attributeProcedures.count; ++index) {
            id info = ((id (*)(id, SEL, unsigned int))objc_msgSend)(
                compiledModel, @selector(procedureInfoForProcedureIndex:), (unsigned int)index);
            id inputs = ((id (*)(id, SEL, unsigned int))objc_msgSend)(
                compiledModel, @selector(inputSymbolIndicesForProcedureIndex:), (unsigned int)index);
            id outputsForProcedure = ((id (*)(id, SEL, unsigned int))objc_msgSend)(
                compiledModel, @selector(outputSymbolIndicesForProcedureIndex:), (unsigned int)index);
            [procedureInfoRecords addObject:@{
                @"index": @(index),
                @"info": [info description] ?: @"none",
                @"input_indices": [inputs description] ?: @"none",
                @"output_indices": [outputsForProcedure description] ?: @"none",
            }];
        }
        if (!loaded) {
            emit(@{@"schema": @"ane-fieldfare.two-procedure-execution.v1", @"status": @"rejected",
                   @"stage": @"load", @"compiled": @YES,
                   @"runtime_procedure_count": @(procedures.count),
                   @"attribute_procedure_count": @(attributeProcedures.count),
                   @"error": error.localizedDescription ?: @"unknown"});
            [NSFileManager.defaultManager removeItemAtPath:temporary error:nil];
            return 1;
        }
        id savedBeforeInstance = callObject(model, @selector(saveModelFiles));
        NSTimeInterval holdAfterLoad = [NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_HOLD_AFTER_LOAD"] doubleValue];
        if (holdAfterLoad > 0.0) [NSThread sleepForTimeInterval:holdAfterLoad];

        id client = callObject(model, @selector(sharedConnection));
        NSURL *instanceWeightURL = [NSURL fileURLWithPath:
            [temporary stringByAppendingPathComponent:@"weights/weight.bin"]];
        id weight0 = ((id (*)(Class, SEL, id, id))objc_msgSend)(
            weightClass, @selector(weightWithSymbolAndURL:weightURL:), @"w0", instanceWeightURL);
        id weight1 = ((id (*)(Class, SEL, id, id))objc_msgSend)(
            weightClass, @selector(weightWithSymbolAndURL:weightURL:), @"w1", instanceWeightURL);
        id procedure0 = ((id (*)(Class, SEL, id, id))objc_msgSend)(
            procedureDataClass, @selector(procedureDataWithSymbol:weightArray:), @"main", @[weight0]);
        id procedure1 = ((id (*)(Class, SEL, id, id))objc_msgSend)(
            procedureDataClass, @selector(procedureDataWithSymbol:weightArray:), @"expert_1", @[weight1]);
        NSString *arrayKind = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_INSTANCE_ARRAY_KIND"] ?: @"data";
        NSArray *instanceProcedureArray = @[procedure0, procedure1];
        if ([arrayKind isEqualToString:@"strings"]) {
            instanceProcedureArray = @[@"main", @"expert_1"];
        } else if ([arrayKind isEqualToString:@"private"] && programProcedureClass) {
            id private0 = ((id (*)(id, SEL, id))objc_msgSend)(
                [programProcedureClass alloc], @selector(initWithSymbolName:), @"main");
            id private1 = ((id (*)(id, SEL, id))objc_msgSend)(
                [programProcedureClass alloc], @selector(initWithSymbolName:), @"expert_1");
            instanceProcedureArray = @[private0, private1];
        }
        id instanceParameters = ((id (*)(Class, SEL, id, id))objc_msgSend)(
            instanceParametersClass, @selector(withProcedureData:procedureArray:),
            @"ane-fieldfare-multi-procedure", instanceProcedureArray);
        id instanceModel = callObject(compiledModel, @selector(shallowCopy));
        NSError *baseUnloadError = nil;
        BOOL baseUnloadedBeforeInstance = NO;
        error = nil;
        BOOL probeNewInstance = [NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_PROBE_NEW_INSTANCE"] boolValue];
        BOOL instanceLoaded = probeNewInstance && client && instanceModel && instanceParameters &&
            ((BOOL (*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
                client, @selector(loadModelNewInstance:options:modelInstParams:qos:error:),
                instanceModel, @{}, instanceParameters, 21, &error);
        NSString *instanceLoadError = probeNewInstance
            ? (error.localizedDescription ?: @"none") : @"not_attempted";
        NSString *directBaseSetting = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_DIRECT_BASE"];
        BOOL directBase = !directBaseSetting || directBaseSetting.boolValue;
        BOOL executionReady = instanceLoaded || directBase;
        id activeModel = instanceLoaded ? instanceModel : compiledModel;

        BOOL offsetArena = [NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_OFFSET_ARENA"] boolValue];
        BOOL singleArena = offsetArena && [NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_OFFSET_SINGLE_ARENA"] boolValue];
        const NSUInteger arenaSlots = singleArena ? 5 : 2;
        IOSurfaceRef arena = offsetArena ? makeSurface(bytes * arenaSlots) : NULL;
        IOSurfaceRef input = offsetArena ? arena : makeSurface(bytes);
        IOSurfaceRef outputs[3] = {
            offsetArena ? (singleArena ? arena : makeSurface(bytes * 2)) : makeSurface(bytes),
            offsetArena ? (singleArena ? arena : makeSurface(bytes * 2)) : makeSurface(bytes),
            offsetArena ? (singleArena ? arena : makeSurface(bytes * 2)) : makeSurface(bytes),
        };
        NSUInteger inputOffset = offsetArena ? bytes : 0;
        NSUInteger outputOffsets[3] = {
            offsetArena ? (singleArena ? bytes * 2 : bytes) : 0,
            offsetArena ? (singleArena ? bytes * 3 : bytes) : 0,
            offsetArena ? (singleArena ? bytes * 4 : bytes) : 0,
        };
        if (offsetArena) {
            IOSurfaceLock(arena, 0, NULL);
            memset(IOSurfaceGetBaseAddress(arena), 0xa5, bytes * arenaSlots);
            IOSurfaceUnlock(arena, 0, NULL);
            if (!singleArena) {
                for (NSUInteger index = 0; index < 3; ++index) {
                    IOSurfaceLock(outputs[index], 0, NULL);
                    memset(IOSurfaceGetBaseAddress(outputs[index]), 0xa5, bytes * 2);
                    IOSurfaceUnlock(outputs[index], 0, NULL);
                }
            }
        }
        IOSurfaceLock(input, 0, NULL);
        uint8_t *inputBase = IOSurfaceGetBaseAddress(input);
        if (offsetArena) {
            for (NSUInteger i = 0; i < elements; ++i)
                *(float *)(inputBase + i * planeStride) = (float)i / 16.0f + 5.0f;
        }
        uint8_t *inputBytes = inputBase + inputOffset;
        for (NSUInteger i = 0; i < elements; ++i)
            *(float *)(inputBytes + i * planeStride) = (float)i / 16.0f - 2.0f;
        IOSurfaceUnlock(input, 0, NULL);
        id wrappedInput = offsetArena
            ? ((id (*)(Class, SEL, IOSurfaceRef, id))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:startOffset:), input, @(inputOffset))
            : ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), input);
        id wrappedOutputs[3] = {
            offsetArena ? ((id (*)(Class, SEL, IOSurfaceRef, id))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:startOffset:), outputs[0], @(outputOffsets[0]))
                : ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                    surfaceClass, @selector(objectWithIOSurface:), outputs[0]),
            offsetArena ? ((id (*)(Class, SEL, IOSurfaceRef, id))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:startOffset:), outputs[1], @(outputOffsets[1]))
                : ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                    surfaceClass, @selector(objectWithIOSurface:), outputs[1]),
            offsetArena ? ((id (*)(Class, SEL, IOSurfaceRef, id))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:startOffset:), outputs[2], @(outputOffsets[2]))
                : ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                    surfaceClass, @selector(objectWithIOSurface:), outputs[2]),
        };

        NSMutableArray *runs = [NSMutableArray array];
        BOOL numerical = YES;
        const NSUInteger procedureSequence[3] = {1, 0, 1};
        uint64_t checksums[3] = {0, 0, 0};
        NSUInteger mapCount = 0, unmapCount = 0;
        for (NSUInteger run = 0; run < 3; ++run) {
            NSUInteger procedureIndex = procedureSequence[run];
            id request = ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
                requestClass,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[wrappedInput], @[@(procedureIndex)], @[wrappedOutputs[run]],
                @[@(procedureIndex)], nil, nil, @(procedureIndex));
            BOOL requestValid = ((BOOL (*)(id, SEL))objc_msgSend)(request, @selector(validate));
            id mapper = callObject(activeModel, @selector(mapper));
            BOOL mapperValid = mapper && ((BOOL (*)(id, SEL, id, id))objc_msgSend)(
                mapper, @selector(validateRequest:model:), request, activeModel);
            NSMutableArray *mappingParameters = [NSMutableArray array];
            NSDictionary *mappingTrailer = @{};
            if (offsetArena && mapper && [mapper respondsToSelector:
                    @selector(prepareANEMemoryMappingParams:request:)]) {
                ANEMemoryMappingParamsProbe params = {0};
                ((void (*)(id, SEL, ANEMemoryMappingParamsProbe *, id))objc_msgSend)(
                    mapper, @selector(prepareANEMemoryMappingParams:request:), &params, request);
                for (NSUInteger index = 0; index < 128; ++index) {
                    ANEBufferStructProbe buffer = params.buffers[index];
                    if (buffer.surface || buffer.field1 || buffer.field2 || buffer.field3 || buffer.field4) {
                        [mappingParameters addObject:@{
                            @"index": @(index),
                            @"surface_matches_input": @(buffer.surface == input),
                            @"surface_matches_output": @(buffer.surface == outputs[run]),
                            @"field1": @(buffer.field1), @"field2": @(buffer.field2),
                            @"field3": @(buffer.field3), @"field4": @(buffer.field4),
                        }];
                    }
                }
                mappingTrailer = @{
                    @"field1": @(params.field1), @"field2": @(params.field2),
                    @"field3": @(params.field3), @"field4": @(params.field4),
                };
            }
            error = nil;
            BOOL mapped = directBase
                ? ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
                    model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
                    request, YES, &error)
                : executionReady && ((BOOL (*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
                    client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
                    activeModel, request, YES, &error);
            if (mapped) mapCount++;
            error = nil;
            BOOL evaluated = directBase
                ? ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, request, &error)
                : executionReady && ((BOOL (*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
                    client, @selector(evaluateWithModel:options:request:qos:error:),
                    activeModel, @{}, request, 21, &error);
            double maxError = 0.0;
            NSUInteger mismatches = 0;
            double baseMaxError = 0.0;
            NSUInteger baseMismatches = 0;
            if (evaluated) {
                IOSurfaceLock(outputs[run], kIOSurfaceLockReadOnly, NULL);
                const uint8_t *outputBase = IOSurfaceGetBaseAddress(outputs[run]);
                const uint8_t *actualBytes = outputBase + outputOffsets[run];
                checksums[run] = checksumValues(actualBytes, elements, planeStride);
                for (NSUInteger i = 0; i < elements; ++i) {
                    float actual = *(const float *)(actualBytes + i * planeStride);
                    float source = (float)((__fp16)((float)i / 16.0f - 2.0f));
                    float expected = (float)((__fp16)(source * (procedureIndex == 0 ? 3.0f : 2.0f)));
                    double delta = fabs((double)actual - expected);
                    maxError = fmax(maxError, delta);
                    if (delta > 0.001) mismatches++;
                    float baseActual = *(const float *)(outputBase + i * planeStride);
                    float alternateSource = (float)((__fp16)((float)i / 16.0f + 5.0f));
                    float baseExpected = (float)((__fp16)(alternateSource * (procedureIndex == 0 ? 3.0f : 2.0f)));
                    double baseDelta = fabs((double)baseActual - baseExpected);
                    baseMaxError = fmax(baseMaxError, baseDelta);
                    if (baseDelta > 0.001) baseMismatches++;
                }
                IOSurfaceUnlock(outputs[run], kIOSurfaceLockReadOnly, NULL);
            }
            numerical = numerical && mapped && evaluated && mismatches == 0;
            [runs addObject:@{@"run": @(run), @"procedure_index": @(procedureIndex),
                              @"expected_multiplier": @(procedureIndex == 0 ? 3 : 2),
                              @"mapped": @(mapped),
                              @"request_valid": @(requestValid), @"mapper_valid": @(mapperValid),
                              @"mapper_present": @(mapper != nil),
                              @"prepared_mapping_parameters": mappingParameters,
                              @"prepared_mapping_trailer": mappingTrailer,
                              @"request": [request description] ?: @"none",
                              @"evaluated": @(evaluated), @"mismatches": @(mismatches),
                              @"base_mismatches_for_alternate_input": @(baseMismatches),
                              @"base_max_absolute_error": @(baseMaxError),
                              @"checksum": [NSString stringWithFormat:@"0x%016llx", checksums[run]],
                              @"max_absolute_error": @(maxError),
                              @"error": error.localizedDescription ?: @"none"}];
            if (mapped) {
                if (directBase) ((void (*)(id, SEL, id))objc_msgSend)(
                    model, @selector(unmapIOSurfacesWithRequest:), request);
                else ((void (*)(id, SEL, id, id))objc_msgSend)(
                    client, @selector(unmapIOSurfacesWithModel:request:), activeModel, request);
                unmapCount++;
            }
        }
        BOOL abaRestored = checksums[0] != 0 && checksums[0] == checksums[2];
        numerical = numerical && abaRestored && mapCount == 3 && unmapCount == 3;

        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.two-procedure-execution.v1",
            @"status": numerical ? @"validated" : @"rejected",
            @"compiled": @(compiled), @"loaded": @(loaded),
            @"compile_count": @1, @"load_count": @1,
            @"map_count": @(mapCount), @"unmap_count": @(unmapCount),
            @"a_b_a_checksum_restored": @(abaRestored),
            @"compile_options": compileOptions,
            @"direct_base": @(directBase),
            @"offset_arena_enabled": @(offsetArena),
            @"offset_single_arena": @(singleArena),
            @"offset_argument_interpretation": offsetArena
                ? @"NSNumber value equals the intended byte offset; functional unit rejected/unvalidated"
                : @"not_tested",
            @"arena_bytes": @(offsetArena ? bytes * arenaSlots : 0),
            @"carrier_bytes": @(bytes),
            @"input_start_offset": @(inputOffset),
            @"output_start_offsets": @[@(outputOffsets[0]), @(outputOffsets[1]), @(outputOffsets[2])],
            @"program_handle": @(callUInt64(model, @selector(programHandle))),
            @"base_unloaded_before_instance": @(baseUnloadedBeforeInstance),
            @"base_unload_error": baseUnloadError.localizedDescription ?: @"none",
            @"new_instance_probe_enabled": @(probeNewInstance),
            @"new_instance_loaded": @(instanceLoaded),
            @"new_instance_error": instanceLoadError,
            @"new_instance_program_handle": @(instanceLoaded
                ? callUInt64(instanceModel, @selector(programHandle)) : 0),
            @"instance_parameters": [instanceParameters description] ?: @"none",
            @"instance_array_kind": arrayKind,
            @"instance_array": [instanceProcedureArray description] ?: @"none",
            @"in_memory_local_model_path": [callObject(model, @selector(localModelPath)) description] ?: @"none",
            @"in_memory_model_url": [callObject(model, @selector(modelURL)) description] ?: @"none",
            @"compiled_model_url": [callObject(compiledModel, @selector(modelURL)) description] ?: @"none",
            @"compiled_source_url": [callObject(compiledModel, @selector(sourceURL)) description] ?: @"none",
            @"compiled_cache_identifier": [callObject(compiledModel, @selector(cacheURLIdentifier)) description] ?: @"none",
            @"saved_model_files": [callObject(model, @selector(saveModelFiles)) description] ?: @"none",
            @"saved_before_instance": [savedBeforeInstance description] ?: @"none",
            @"temporary_path": temporary,
            @"runtime_procedure_count": @(procedures.count),
            @"attribute_procedure_count": @(attributeProcedures.count),
            @"attribute_procedures": attributeProcedures ?: @[],
            @"model_attributes": attributes ?: @{},
            @"model_num_inputs": @(((unsigned int (*)(id, SEL))objc_msgSend)(compiledModel, @selector(numInputs))),
            @"model_num_outputs": @(((unsigned int (*)(id, SEL))objc_msgSend)(compiledModel, @selector(numOutputs))),
            @"model_procedures_count": @(((unsigned int (*)(id, SEL))objc_msgSend)(compiledModel, @selector(proceduresCount))),
            @"procedure_info": procedureInfoRecords,
            @"procedures": procedureRecords, @"runs": runs,
            @"claim_boundary": offsetArena
                ? @"The startOffset constructor preserved and displayed the requested NSNumber, and request validation, mapper validation, mapping, and evaluation all succeeded. Nevertheless, ANE read and wrote carrier slot zero exactly and ignored the requested nonzero slot in both distinct-surface and shared-arena constructions. The current direct _ANEInMemoryModel path therefore validates selector acceptance but rejects functional offset-backed views; it does not rule out a different chaining or compiler-authored mapping lifecycle."
                : numerical
                ? @"Compiler-authored procedures 0 and 1 executed as 2x, 3x, 2x behind one compiled and loaded private-ANE artifact and one program handle. All three request mappings and evaluations passed independent numerical oracles and the A checksum was restored. This validates procedure selection only at two tiny functions, not 128 expert-sized procedures or compressed Gemma weights."
                : @"The two-function program compiled and loaded, but both procedure indices did not pass numerical oracles; no multiplexing claim is made."
        };
        emit(receipt);

        if (instanceLoaded) {
            NSError *instanceUnloadError = nil;
            ((BOOL (*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
                client, @selector(unloadModel:options:qos:error:),
                instanceModel, @{}, 21, &instanceUnloadError);
        }
        NSError *unloadError = nil;
        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &unloadError);
        if (offsetArena) {
            if (arena) CFRelease(arena);
            if (!singleArena) {
                for (NSUInteger i = 0; i < 3; ++i) if (outputs[i]) CFRelease(outputs[i]);
            }
        } else {
            if (input) CFRelease(input);
            for (NSUInteger i = 0; i < 3; ++i) if (outputs[i]) CFRelease(outputs[i]);
        }
        if (![NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_KEEP_TEMP"] boolValue])
            [NSFileManager.defaultManager removeItemAtPath:temporary error:nil];
        return numerical ? 0 : 1;
    }
}
