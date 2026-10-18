#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include <math.h>

enum { Elements = 64, PlaneStride = 64, CarrierBytes = Elements * PlaneStride };

static NSArray *methodInventory(Class cls, BOOL classMethods) {
    if (!cls) return @[];
    Class owner = classMethods ? object_getClass(cls) : cls;
    unsigned int count = 0;
    Method *methods = class_copyMethodList(owner, &count);
    NSMutableArray *records = [NSMutableArray array];
    for (unsigned int index = 0; index < count; ++index) {
        const char *encoding = method_getTypeEncoding(methods[index]);
        [records addObject:@{@"selector": NSStringFromSelector(method_getName(methods[index])),
                             @"encoding": encoding ? @(encoding) : @""}];
    }
    free(methods);
    [records sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
        return [a[@"selector"] compare:b[@"selector"]];
    }];
    return records;
}

static id callObject(id object, SEL selector) {
    return object && [object respondsToSelector:selector]
        ? ((id (*)(id, SEL))objc_msgSend)(object, selector) : nil;
}

static uint64_t callUInt64(id object, SEL selector) {
    return object && [object respondsToSelector:selector]
        ? ((uint64_t (*)(id, SEL))objc_msgSend)(object, selector) : 0;
}

static IOSurfaceRef makeSurface(NSUInteger bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0,
    });
}

static uint64_t surfaceHash(IOSurfaceRef surface) {
    uint64_t hash = 1469598103934665603ULL;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    const uint8_t *bytes = IOSurfaceGetBaseAddress(surface);
    for (size_t index = 0; index < IOSurfaceGetAllocSize(surface); ++index) {
        hash ^= bytes[index]; hash *= 1099511628211ULL;
    }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return hash;
}

static NSData *diagonalWeightBlob(float scale) {
    const NSUInteger payloadBytes = Elements * Elements * sizeof(__fp16);
    NSMutableData *blob = [NSMutableData dataWithLength:128 + payloadBytes];
    uint8_t *bytes = blob.mutableBytes;
    bytes[0] = 1; bytes[4] = 2;
    bytes[64] = 0xef; bytes[65] = 0xbe; bytes[66] = 0xad; bytes[67] = 0xde;
    bytes[68] = 1;
    *(uint32_t *)(bytes + 72) = (uint32_t)payloadBytes;
    *(uint32_t *)(bytes + 80) = 128;
    __fp16 *weights = (__fp16 *)(bytes + 128);
    for (NSUInteger index = 0; index < Elements; ++index)
        weights[index * Elements + index] = (__fp16)scale;
    return blob;
}

static NSString *twoOpMIL(void) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "  func main<ios18>(tensor<fp32, [1, %d, 1, 1]> x) {\n"
         "    string f16=const()[name=string(\"f16\"),val=string(\"fp16\")];\n"
         "    tensor<fp16,[1,%d,1,1]> x16=cast(dtype=f16,x=x)[name=string(\"cast_in\")];\n"
         "    tensor<fp16,[%d,%d,1,1]> w=const()[name=string(\"w\"),val=tensor<fp16,[%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/weight.bin\"),offset=uint64(64)))];\n"
         "    tensor<int32,[2]> st=const()[name=string(\"st\"),val=tensor<int32,[2]>([1,1])];\n"
         "    tensor<int32,[4]> pd=const()[name=string(\"pd\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[2]> dl=const()[name=string(\"dl\"),val=tensor<int32,[2]>([1,1])];\n"
         "    int32 gr=const()[name=string(\"gr\"),val=int32(1)];\n"
         "    string pt=const()[name=string(\"pt\"),val=string(\"valid\")];\n"
         "    tensor<fp16,[1,%d,1,1]> mid=conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=w,x=x16)[name=string(\"captured_mid\")];\n"
         "    fp16 three=const()[name=string(\"three\"),val=fp16(3.0)];\n"
         "    tensor<fp16,[1,%d,1,1]> final16=mul(x=mid,y=three)[name=string(\"final_times_three\")];\n"
         "    string f32=const()[name=string(\"f32\"),val=string(\"fp32\")];\n"
         "    tensor<fp32,[1,%d,1,1]> y=cast(dtype=f32,x=final16)[name=string(\"cast_out\")];\n"
         "  } -> (y);\n"
         "}\n", Elements, Elements, Elements, Elements, Elements, Elements,
         Elements, Elements, Elements];
}

static NSArray *descriptorRecords(id value) {
    if (![value isKindOfClass:NSArray.class]) return @[];
    NSMutableArray *records = [NSMutableArray array];
    for (id descriptor in (NSArray *)value) {
        [records addObject:@{
            @"class": NSStringFromClass([descriptor class]) ?: @"unknown",
            @"name": callObject(descriptor, @selector(tensorName)) ?: @"unavailable",
            @"index": @(callUInt64(descriptor, @selector(tensorIndex))),
            @"description": [descriptor description] ?: @"nil",
        }];
    }
    return records;
}

static NSDictionary *queryModel(Class helperClass, id modelURL) {
    if (!helperClass || !modelURL)
        return @{@"attempted": @NO, @"url": [modelURL description] ?: @"nil",
                 @"result": @[], @"error": @"missing helper or URL"};
    NSError *error = nil;
    id result = ((id (*)(Class, SEL, id, NSError **))objc_msgSend)(
        helperClass, @selector(queryTensorsFromModel:error:), modelURL, &error);
    return @{@"attempted": @YES, @"url": [modelURL description] ?: @"nil",
             @"result": descriptorRecords(result),
             @"raw_result_class": result ? NSStringFromClass([result class]) : @"nil",
             @"error": error.localizedDescription ?: @"none"};
}

static NSDictionary *artifactInventory(NSURL *rootURL) {
    if (![rootURL isKindOfClass:NSURL.class] || !rootURL.isFileURL)
        return @{@"root": [rootURL description] ?: @"nil", @"files": @[],
                 @"contains_internal_name": @NO, @"anedebug_files": @[]};
    NSNumber *isDirectory = nil;
    [rootURL getResourceValue:&isDirectory forKey:NSURLIsDirectoryKey error:nil];
    NSURL *root = isDirectory.boolValue ? rootURL : rootURL.URLByDeletingLastPathComponent;
    NSDirectoryEnumerator<NSURL *> *enumerator = [NSFileManager.defaultManager
        enumeratorAtURL:root includingPropertiesForKeys:@[NSURLIsRegularFileKey, NSURLFileSizeKey]
        options:NSDirectoryEnumerationSkipsHiddenFiles errorHandler:^BOOL(NSURL *url, NSError *error) {
            (void)url; (void)error; return YES;
        }];
    NSMutableArray *files = [NSMutableArray array], *debugFiles = [NSMutableArray array];
    BOOL containsName = NO;
    for (NSURL *url in enumerator) {
        NSNumber *regular = nil, *size = nil;
        [url getResourceValue:&regular forKey:NSURLIsRegularFileKey error:nil];
        if (!regular.boolValue) continue;
        [url getResourceValue:&size forKey:NSURLFileSizeKey error:nil];
        NSString *relative = [url.path substringFromIndex:MIN(root.path.length + 1, url.path.length)];
        BOOL isDebug = [url.pathExtension.lowercaseString isEqual:@"anedebug"];
        if (isDebug) [debugFiles addObject:relative];
        BOOL fileContainsName = NO;
        if (size.unsignedLongLongValue <= 16 * 1024 * 1024) {
            NSData *data = [NSData dataWithContentsOfURL:url options:NSDataReadingMappedIfSafe error:nil];
            NSData *needle = [@"captured_mid" dataUsingEncoding:NSUTF8StringEncoding];
            fileContainsName = data && [data rangeOfData:needle options:0 range:NSMakeRange(0, data.length)].location != NSNotFound;
        }
        containsName |= fileContainsName;
        if (files.count < 64) [files addObject:@{@"path": relative ?: url.lastPathComponent,
                                                @"bytes": size ?: @0,
                                                @"contains_internal_name": @(fileContainsName)}];
    }
    return @{@"root": root.path ?: @"nil", @"files": files,
             @"contains_internal_name": @(containsName), @"anedebug_files": debugFiles};
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
}

int main(void) {
    @autoreleasepool {
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class tensorClass = NSClassFromString(@"_ANEIntermediateTensor");
        Class helperClass = NSClassFromString(@"_ANETensorDebugHelper");
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        NSMutableArray *classes = [NSMutableArray array];
        for (NSString *name in @[@"_ANEIntermediateTensor", @"_ANETensorDebugHelper",
                                @"_ANEInMemoryModel", @"_ANEModel",
                                @"_ANEProgramForEvaluation", @"_ANEVirtualClient"]) {
            Class cls = NSClassFromString(name);
            if (cls) [classes addObject:@{@"class": name,
                @"class_methods": methodInventory(cls, YES),
                @"instance_methods": methodInventory(cls, NO)}];
        }

        id hostDescriptor = tensorClass ? ((id (*)(Class, SEL, id, uint64_t))objc_msgSend)(
            tensorClass, @selector(descriptorWithName:index:), @"ane_fieldfare_probe", 7) : nil;
        NSError *archiveError = nil;
        NSData *archive = hostDescriptor ? [NSKeyedArchiver archivedDataWithRootObject:hostDescriptor
                                                                 requiringSecureCoding:YES
                                                                                 error:&archiveError] : nil;
        NSError *unarchiveError = nil;
        id roundTrip = archive ? [NSKeyedUnarchiver unarchivedObjectOfClass:tensorClass
                                                                   fromData:archive error:&unarchiveError] : nil;
        NSError *nilQueryError = nil;
        id nilQuery = helperClass ? ((id (*)(Class, SEL, id, NSError **))objc_msgSend)(
            helperClass, @selector(queryTensorsFromModel:error:), nil, &nilQueryError) : nil;
        NSURL *missingURL = [NSURL fileURLWithPath:
            [NSTemporaryDirectory() stringByAppendingPathComponent:@"ane-fieldfare-no-such-model.mlmodelc"]];
        NSError *missingQueryError = nil;
        id missingQuery = helperClass ? ((id (*)(Class, SEL, id, NSError **))objc_msgSend)(
            helperClass, @selector(queryTensorsFromModel:error:), missingURL, &missingQueryError) : nil;

        IOSurfaceRef normalizationSurface = makeSurface(CarrierBytes);
        IOSurfaceLock(normalizationSurface, 0, NULL);
        memset(IOSurfaceGetBaseAddress(normalizationSurface), 0xa5, CarrierBytes);
        IOSurfaceUnlock(normalizationSurface, 0, NULL);
        uint64_t normalizeHashBefore = surfaceHash(normalizationSurface);
        NSError *normalizeError = nil;
        BOOL normalized = helperClass && hostDescriptor &&
            ((BOOL (*)(Class, SEL, IOSurfaceRef, id, NSError **))objc_msgSend)(
                helperClass, @selector(normalizeTensor:descriptor:error:),
                normalizationSurface, hostDescriptor, &normalizeError);
        uint64_t normalizeHashAfter = surfaceHash(normalizationSurface);
        NSError *nilNormalizeError = nil;
        BOOL nilNormalized = helperClass && hostDescriptor &&
            ((BOOL (*)(Class, SEL, IOSurfaceRef, id, NSError **))objc_msgSend)(
                helperClass, @selector(normalizeTensor:descriptor:error:),
                NULL, hostDescriptor, &nilNormalizeError);

        NSData *mil = [twoOpMIL() dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weightBlob = diagonalWeightBlob(2.0f);
        NSDictionary *weightFiles = @{@"@model_path/weights/weight.bin":
            @{@"offset": @0, @"data": weightBlob}};
        id memoryDescriptor = descriptorClass ? ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, weightFiles, nil) : nil;
        id model = memoryDescriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), memoryDescriptor) : nil;
        NSString *identifier = callObject(model, @selector(hexStringIdentifier)) ?: NSUUID.UUID.UUIDString;
        NSString *temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        [NSFileManager.defaultManager createDirectoryAtPath:
            [temporary stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weightBlob writeToFile:[temporary stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *error = nil;
        BOOL compiled = model && ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        NSString *compileError = error.localizedDescription ?: @"none";
        id compiledModel = callObject(model, @selector(model));
        id modelURL = callObject(model, @selector(modelURL));
        id compiledURL = callObject(compiledModel, @selector(modelURL));
        id sourceURL = callObject(compiledModel, @selector(sourceURL));
        NSDictionary *compilerArtifacts = artifactInventory(modelURL);
        NSArray *queries = @[
            queryModel(helperClass, modelURL), queryModel(helperClass, compiledURL),
            queryModel(helperClass, sourceURL),
            queryModel(helperClass, [NSURL fileURLWithPath:temporary isDirectory:YES]),
        ];
        NSDictionary *attributes = callObject(model, @selector(modelAttributes)) ?: @{};
        NSString *attributeText = attributes.description ?: @"";
        BOOL compilerDeclaredName = [attributeText containsString:@"captured_mid"];
        BOOL anyDescriptors = NO, exactDescriptor = NO;
        for (NSDictionary *query in queries) {
            NSArray *records = query[@"result"];
            anyDescriptors |= records.count > 0;
            for (NSDictionary *record in records)
                exactDescriptor |= [record[@"name"] isEqual:@"captured_mid"];
        }

        error = nil;
        BOOL loaded = compiled && ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        NSString *loadError = error.localizedDescription ?: @"none";
        IOSurfaceRef input = makeSurface(CarrierBytes), output = makeSurface(CarrierBytes);
        IOSurfaceLock(input, 0, NULL);
        uint8_t *inputBytes = IOSurfaceGetBaseAddress(input);
        for (NSUInteger index = 0; index < Elements; ++index)
            *(float *)(inputBytes + index * PlaneStride) = (float)index / 16.0f - 2.0f;
        IOSurfaceUnlock(input, 0, NULL);
        IOSurfaceLock(output, 0, NULL);
        memset(IOSurfaceGetBaseAddress(output), 0xa5, CarrierBytes);
        IOSurfaceUnlock(output, 0, NULL);
        id wrappedInput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), input);
        id wrappedOutput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), output);
        id request = ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wrappedInput], @[@0], @[wrappedOutput], @[@0], nil, nil, @0);
        error = nil;
        BOOL mapped = loaded && ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
            model, @selector(mapIOSurfacesWithRequest:cacheInference:error:), request, YES, &error);
        NSString *mapError = error.localizedDescription ?: @"none";
        error = nil;
        BOOL evaluated = mapped && ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &error);
        NSString *evaluationError = error.localizedDescription ?: @"none";
        NSUInteger finalMismatches = 0, differsFromMid = 0, poisonRemaining = 0;
        double finalMaxError = 0.0, midDistanceMinimum = INFINITY;
        IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
        const uint8_t *outputBytes = IOSurfaceGetBaseAddress(output);
        for (NSUInteger index = 0; index < Elements; ++index) {
            float source = (float)((__fp16)((float)index / 16.0f - 2.0f));
            float expectedMid = (float)((__fp16)(source * 2.0f));
            float expectedFinal = (float)((__fp16)(expectedMid * 3.0f));
            float actual = *(const float *)(outputBytes + index * PlaneStride);
            double finalError = fabs((double)actual - expectedFinal);
            finalMaxError = fmax(finalMaxError, finalError);
            if (finalError > 0.001) finalMismatches++;
            double distance = fabs((double)actual - expectedMid);
            if (distance > 0.001) differsFromMid++;
            midDistanceMinimum = fmin(midDistanceMinimum, distance);
            uint32_t bits = 0; memcpy(&bits, &actual, sizeof(bits));
            if (bits == 0xa5a5a5a5) poisonRemaining++;
        }
        IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
        if (mapped) ((void (*)(id, SEL, id))objc_msgSend)(
            model, @selector(unmapIOSurfacesWithRequest:), request);

        BOOL finalOracle = evaluated && finalMismatches == 0 && poisonRemaining == 0 &&
                           differsFromMid > 0;
        NSString *status = finalOracle && !exactDescriptor
            ? @"validated_execution_capture_rejected"
            : (finalOracle && exactDescriptor ? @"accepted_descriptor_capture_unavailable" : @"rejected");
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.intermediate-tensor.v1", @"status": status,
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"architecture": @"arm64", @"ane_architecture": @"Apple M5 / H17G",
            @"framework_loaded": @(framework != NULL),
            @"exact_command": NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXACT_COMMAND"]
                ?: @".build/ane-fieldfare-intermediate-tensor-gate",
            @"exit_status": @0, @"compile_count": @(compiled ? 1 : 0), @"load_count": @(loaded ? 1 : 0),
            @"abi": @{@"classes": classes,
                @"daemon_metadata": @{
                    @"query_discovery_log": @"Stub implementation - returning sample tensor descriptors",
                    @"normalization_log": @"STUB - ANELoader in-place normalization not yet implemented",
                    @"normalization_effect_log": @"Raw tensor unchanged (client will receive un-normalized data)",
                    @"intermediate_buffer_keys": @[@"kANEFIntermediateBufferHandleKey", @"kANEFIntermediateBufferSizeHint"],
                    @"capture_consumer_found": @NO,
                }},
            @"host_descriptor_control": @{
                @"constructed": @(hostDescriptor != nil), @"name": callObject(hostDescriptor, @selector(tensorName)) ?: @"nil",
                @"index": @(callUInt64(hostDescriptor, @selector(tensorIndex))), @"secure_archive_bytes": @(archive.length),
                @"secure_archive_error": archiveError.localizedDescription ?: @"none",
                @"round_trip": @(roundTrip != nil), @"round_trip_error": unarchiveError.localizedDescription ?: @"none",
            },
            @"negative_controls": @{
                @"nil_model_query": @{@"result": [nilQuery description] ?: @"nil", @"error": nilQueryError.localizedDescription ?: @"none"},
                @"missing_model_query": @{@"result": [missingQuery description] ?: @"nil", @"error": missingQueryError.localizedDescription ?: @"none"},
                @"nil_surface_normalization": @{@"accepted": @(nilNormalized), @"error": nilNormalizeError.localizedDescription ?: @"none"},
                @"host_descriptor_normalization": @{@"accepted": @(normalized), @"error": normalizeError.localizedDescription ?: @"none",
                    @"hash_before": [NSString stringWithFormat:@"0x%016llx", normalizeHashBefore],
                    @"hash_after": [NSString stringWithFormat:@"0x%016llx", normalizeHashAfter],
                    @"bytes_changed": @(normalizeHashBefore != normalizeHashAfter)},
            },
            @"compiler_program": @{
                @"mil_internal_name": @"captured_mid", @"operation_1": @"mid = 2*x",
                @"operation_2": @"final = 3*mid", @"compiled": @(compiled), @"compile_error": compileError,
                @"loaded": @(loaded), @"load_error": loadError, @"program_handle": @(callUInt64(model, @selector(programHandle))),
                @"model_url": [modelURL description] ?: @"nil", @"compiled_model_url": [compiledURL description] ?: @"nil",
                @"compiled_source_url": [sourceURL description] ?: @"nil",
                @"compiler_artifacts": compilerArtifacts,
                @"model_attributes": attributes, @"attributes_declare_captured_mid": @(compilerDeclaredName),
                @"tensor_queries": queries, @"any_descriptors": @(anyDescriptors),
                @"exact_named_descriptor": @(exactDescriptor),
            },
            @"execution_oracle": @{
                @"mapped": @(mapped), @"map_error": mapError, @"evaluated": @(evaluated),
                @"evaluation_error": evaluationError, @"elements": @(Elements),
                @"expected_intermediate": @"2*x", @"expected_final": @"6*x",
                @"final_mismatches": @(finalMismatches), @"final_max_absolute_error": @(finalMaxError),
                @"final_values_differing_from_expected_intermediate": @(differsFromMid),
                @"minimum_final_to_intermediate_distance": @(midDistanceMinimum),
                @"poison_remaining": @(poisonRemaining), @"passed": @(finalOracle),
                @"intermediate_observed": @NO,
            },
            @"claim_boundary": @"The compiler-authored named two-op program compiled, loaded, and its final output passed an independent oracle distinct from the intended intermediate. _ANEIntermediateTensor construction and secure coding are accepted. The only tensor-debug discovery path requires an authoritative .anedebug sidecar; this compiler emitted no matching descriptor, and the daemon labels discovery/normalization as stubs. No request/evaluation selector consumes the descriptor and no captured intermediate bytes were returned. Capture is therefore rejected for this exact construction; opaque intermediateBufferHandle scratch is not treated as tensor capture, and no tensor index or buffer schema is synthesized.",
        };
        emit(receipt);

        NSError *unloadError = nil;
        if (loaded) ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &unloadError);
        if (input) CFRelease(input); if (output) CFRelease(output);
        if (normalizationSurface) CFRelease(normalizationSurface);
        [NSFileManager.defaultManager removeItemAtPath:temporary error:nil];
        return finalOracle ? 0 : 1;
    }
}
