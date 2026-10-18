#define main ANEFieldfareAOTBlockSliceSingleMain
#include "../ANEFieldfareAOTBlockSliceGate/main.m"
#undef main

#import <dlfcn.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <objc/runtime.h>

static double gateMilliseconds(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)(end - start) * timebase.numer / timebase.denom / 1e6;
}

static BOOL hasInstanceABI(Class cls, SEL selector, const char *encoding) {
    Method method = cls ? class_getInstanceMethod(cls, selector) : NULL;
    const char *actual = method ? method_getTypeEncoding(method) : NULL;
    return actual && strcmp(actual, encoding) == 0;
}

static BOOL hasClassABI(Class cls, SEL selector, const char *encoding) {
    Method method = cls ? class_getClassMethod(cls, selector) : NULL;
    const char *actual = method ? method_getTypeEncoding(method) : NULL;
    return actual && strcmp(actual, encoding) == 0;
}

static id objectCall(id object, SEL selector) {
    return object && [object respondsToSelector:selector]
        ? ((id (*)(id, SEL))objc_msgSend)(object, selector) : nil;
}

static NSArray *fileInventory(NSURL *rootURL) {
    if (![rootURL isKindOfClass:NSURL.class] || !rootURL.isFileURL) return @[];
    NSNumber *directory = nil;
    [rootURL getResourceValue:&directory forKey:NSURLIsDirectoryKey error:nil];
    NSString *root = (directory.boolValue ? rootURL : rootURL.URLByDeletingLastPathComponent).path;
    NSDirectoryEnumerator<NSString *> *enumerator =
        [NSFileManager.defaultManager enumeratorAtPath:root];
    NSMutableArray *files = [NSMutableArray array];
    for (NSString *relative in enumerator) {
        NSDictionary *attributes = [enumerator fileAttributes];
        if (![attributes[NSFileType] isEqual:NSFileTypeRegular]) continue;
        [files addObject:@{ @"path": relative,
                            @"bytes": attributes[NSFileSize] ?: @0 }];
    }
    [files sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
        return [a[@"path"] compare:b[@"path"]];
    }];
    return files;
}

static IOSurfaceRef gateSurface(uint64_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0,
    });
}

static BOOL runExpert(id directModel, id client, id compiledModel, BOOL direct,
                      const float *inputValues, const float *expected,
                      NSDictionary **result) {
    Class requestClass = NSClassFromString(@"_ANERequest");
    Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
    NSDictionary *attributes = objectCall(direct ? directModel : compiledModel,
                                           @selector(modelAttributes)) ?: @{};
    NSDictionary *description = attributes[@"ANEFModelDescription"];
    NSNumber *procedure = description[@"kANEFModelProcedureNameToIDMapKey"][@"expert_0"];
    NSArray *records = description[@"ANEFModelProcedures"];
    NSDictionary *record = procedure.unsignedIntegerValue < records.count
        ? records[procedure.unsignedIntegerValue] : nil;
    NSNumber *inputIndex = [record[@"ANEFModelInputSymbolIndexArray"] firstObject];
    NSNumber *outputIndex = [record[@"ANEFModelOutputSymbolIndexArray"] firstObject];
    if (!requestClass || !surfaceClass || !procedure || !inputIndex || !outputIndex) {
        if (result) *result = @{ @"status": @"rejected", @"error": @"procedure metadata incomplete" };
        return NO;
    }
    uint64_t inputBytes = (uint64_t)Hidden * Positions * sizeof(__fp16);
    uint64_t outputBytes = (uint64_t)Hidden * Positions * sizeof(float);
    IOSurfaceRef input = gateSurface(inputBytes), output = gateSurface(outputBytes);
    if (!input || !output || IOSurfaceLock(input, 0, NULL) != kIOReturnSuccess) {
        if (input) CFRelease(input); if (output) CFRelease(output);
        if (result) *result = @{ @"status": @"rejected", @"error": @"IOSurface allocation failed" };
        return NO;
    }
    __fp16 *inputBase = IOSurfaceGetBaseAddress(input);
    for (uint32_t column = 0; column < Hidden; ++column)
        for (uint32_t position = 0; position < Positions; ++position)
            inputBase[(uint64_t)column * Positions + position] = (__fp16)inputValues[column];
    IOSurfaceUnlock(input, 0, NULL);
    IOSurfaceLock(output, 0, NULL);
    memset(IOSurfaceGetBaseAddress(output), 0xa5, outputBytes);
    IOSurfaceUnlock(output, 0, NULL);
    id wrappedInput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        surfaceClass, @selector(objectWithIOSurface:), input);
    id wrappedOutput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        surfaceClass, @selector(objectWithIOSurface:), output);
    id request = wrappedInput && wrappedOutput ?
        ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wrappedInput], @[inputIndex], @[wrappedOutput], @[outputIndex], nil, nil, procedure) : nil;
    NSError *error = nil;
    BOOL mapped = request && (direct
        ? ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
            directModel, @selector(mapIOSurfacesWithRequest:cacheInference:error:), request, YES, &error)
        : ((BOOL (*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
            client, @selector(mapIOSurfacesWithModel:request:cacheInference:error:),
            compiledModel, request, YES, &error));
    uint64_t started = mach_absolute_time();
    BOOL evaluated = mapped && (direct
        ? ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            directModel, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &error)
        : ((BOOL (*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend)(
            client, @selector(evaluateWithModel:options:request:qos:error:),
            compiledModel, @{}, request, 21, &error));
    double evaluationMs = gateMilliseconds(started, mach_absolute_time());
    uint64_t comparisons = 0, finite = 0; double maxError = 0, squareError = 0;
    if (evaluated && IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess) {
        const float *actual = IOSurfaceGetBaseAddress(output);
        for (uint32_t row = 0; row < Hidden; ++row)
            for (uint32_t position = 0; position < Positions; ++position) {
                float value = actual[(uint64_t)row * Positions + position];
                double delta = fabs((double)value - expected[row]);
                maxError = fmax(maxError, delta); squareError += delta * delta;
                finite += isfinite(value); ++comparisons;
            }
        IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
    }
    if (mapped) {
        if (direct) ((void (*)(id, SEL, id))objc_msgSend)(
            directModel, @selector(unmapIOSurfacesWithRequest:), request);
        else ((void (*)(id, SEL, id, id))objc_msgSend)(
            client, @selector(unmapIOSurfacesWithModel:request:), compiledModel, request);
    }
    BOOL valid = evaluated && comparisons == (uint64_t)Hidden * Positions &&
        finite == comparisons && maxError <= 0.08;
    if (result) *result = @{ @"status": valid ? @"validated" : @"rejected",
        @"mapped": @(mapped), @"evaluated": @(evaluated),
        @"comparisons": @(comparisons), @"finite": @(finite),
        @"max_absolute_error": @(maxError),
        @"rmse": @(comparisons ? sqrt(squareError / comparisons) : 0),
        @"evaluation_ms": @(evaluationMs),
        @"error": error.localizedDescription ?: @"none" };
    CFRelease(input); CFRelease(output);
    return valid;
}

static BOOL expertOracleForReload(ANEFieldfareTensor gate, ANEFieldfareTensor up,
                                  ANEFieldfareTensor down, const float *input, float *output) {
    float *g = calloc(Rows, sizeof(float)), *u = calloc(Rows, sizeof(float));
    float *activated = calloc(Rows, sizeof(float));
    BOOL ok = g && u && activated && ANEFieldfareAffineMatVec(&gate, 4, Group, input, g) &&
        ANEFieldfareAffineMatVec(&up, 4, Group, input, u);
    if (ok) for (uint32_t row = 0; row < Rows; ++row) {
        float gf = (float)(__fp16)g[row], uf = (float)(__fp16)u[row];
        float gelu = 0.5f * gf * (1.0f + tanhf(0.7978845608028654f *
            (gf + 0.044715f * gf * gf * gf)));
        activated[row] = (float)(__fp16)(gelu * uf);
    }
    if (ok) ok = ANEFieldfareAffineMatVec(&down, 4, Group, activated, output);
    free(g); free(u); free(activated);
    return ok;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            emit(@{ @"schema": @"ane-fieldfare.compiled-artifact-reload.v1",
                    @"status": @"rejected", @"stage": @"arguments",
                    @"error": @"usage: reload-gate <eight-experts.mlmodelc> <gturbo-root>" });
            return 2;
        }
        NSString *sourceRoot = @(argv[1]), *checkpoint = @(argv[2]);
        NSData *mil = [NSData dataWithContentsOfFile:[sourceRoot stringByAppendingPathComponent:@"model.mil"]];
        NSData *weights = [NSData dataWithContentsOfFile:[sourceRoot stringByAppendingPathComponent:@"weights/weight.bin"]];
        NSDictionary *layout = readJSON([checkpoint stringByAppendingPathComponent:@"packed_experts/layout.json"]);
        NSDictionary *layer = layout[@"layers"][0], *expert = layer[@"experts"][0];
        NSData *blob = [NSData dataWithContentsOfFile:[[checkpoint stringByAppendingPathComponent:@"packed_experts"]
            stringByAppendingPathComponent:layer[@"file"]] options:NSDataReadingMappedIfSafe error:nil];
        ANEFieldfareTensor gate = {0}, up = {0}, down = {0};
        uint64_t base = [expert[@"offset"] unsignedLongLongValue];
        NSDictionary *tensors = expert[@"tensors"];
        float inputValues[Hidden];
        for (uint32_t column = 0; column < Hidden; ++column) inputValues[column] = inputValue(column, 0);
        float *expected = calloc(Hidden, sizeof(float));
        BOOL fixture = mil && weights && blob && expected &&
            resolveMatrix(blob, base, tensors, @"gate", &gate) &&
            resolveMatrix(blob, base, tensors, @"up", &up) &&
            resolveMatrix(blob, base, tensors, @"down", &down) &&
            expertOracleForReload(gate, up, down, inputValues, expected);

        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class compiledModelClass = NSClassFromString(@"_ANEModel");
        Class clientClass = NSClassFromString(@"_ANEClient");
        NSDictionary *abi = @{
            @"descriptor": @(hasClassABI(descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), "@40@0:8@16@24@32")),
            @"compile": @(hasInstanceABI(memoryModelClass, @selector(compileWithQoS:options:error:), "B36@0:8I16@20^@28")),
            @"save": @(hasInstanceABI(memoryModelClass, @selector(saveModelFiles), "@16@0:8")),
            @"model_at_url": @(hasClassABI(compiledModelClass, @selector(modelAtURL:key:), "@32@0:8@16@24")),
            @"model_at_url_with_attributes": @(hasClassABI(compiledModelClass,
                @selector(modelAtURL:key:modelAttributes:), "@40@0:8@16@24@32")),
            @"model_at_url_with_cache": @(hasClassABI(compiledModelClass,
                @selector(modelAtURLWithCacheURLIdentifier:key:cacheURLIdentifier:), "@40@0:8@16@24@32")),
            @"model_with_cache": @(hasClassABI(compiledModelClass,
                @selector(modelWithCacheURLIdentifier:), "@24@0:8@16")),
            @"client_load": @(hasInstanceABI(clientClass, @selector(loadModel:options:qos:error:), "B44@0:8@16@24I32^@36")),
            @"client_map": @(hasInstanceABI(clientClass, @selector(mapIOSurfacesWithModel:request:cacheInference:error:), "B44@0:8@16@24B32^@36")),
            @"client_evaluate": @(hasInstanceABI(clientClass, @selector(evaluateWithModel:options:request:qos:error:), "B52@0:8@16@24@32I40^@44")),
            @"client_unload": @(hasInstanceABI(clientClass, @selector(unloadModel:options:qos:error:), "B44@0:8@16@24I32^@36")),
        };
        BOOL abiValid = fixture && framework;
        for (NSNumber *value in abi.allValues) abiValid &= value.boolValue;
        NSMutableDictionary *receipt = [@{
            @"schema": @"ane-fieldfare.compiled-artifact-reload.v1",
            @"host": sysctlString("kern.hostname"), @"host_chip": sysctlString("machdep.cpu.brand_string"),
            @"os_build": sysctlString("kern.osversion"), @"exact_command": [NSString stringWithFormat:@"%s %s %s", argv[0], argv[1], argv[2]],
            @"abi_guards": abi, @"source": @{ @"path": sourceRoot, @"mil_bytes": @(mil.length), @"weight_bytes": @(weights.length) },
            @"compile_count": @0, @"load_count": @0,
        } mutableCopy];
        if (!abiValid) {
            receipt[@"status"] = @"rejected"; receipt[@"stage"] = @"abi_or_fixture";
            receipt[@"error"] = @"authoritative ABI guard or authentic fixture failed";
            receipt[@"claim_boundary"] = @"No private call was made after an ABI or authentic-fixture precondition failed.";
            emit(receipt); free(expected); return 1;
        }

        NSDictionary *weightMap = @{@"@model_path/weights/weight.bin": @{ @"offset": @0, @"data": weights }};
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, weightMap, nil);
        id memoryModel = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
            memoryModelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        NSString *identifier = objectCall(memoryModel, @selector(hexStringIdentifier));
        NSString *temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier ?: NSUUID.UUID.UUIDString];
        NSFileManager *files = NSFileManager.defaultManager;
        [files createDirectoryAtPath:[temporary stringByAppendingPathComponent:@"weights"]
         withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weights writeToFile:[temporary stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        NSError *error = nil; uint64_t started = mach_absolute_time();
        BOOL compiled = memoryModel && ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            memoryModel, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        double compileMs = gateMilliseconds(started, mach_absolute_time());
        receipt[@"compile_count"] = @(compiled ? 1 : 0);
        receipt[@"cold_compile_ms"] = @(compileMs);
        receipt[@"compile_error"] = error.localizedDescription ?: @"none";
        id compiledA = objectCall(memoryModel, @selector(model));
        NSDictionary *modelAttributesA = objectCall(memoryModel, @selector(modelAttributes)) ?: @{};
        id saveResult = objectCall(memoryModel, @selector(saveModelFiles));
        NSURL *memoryURL = objectCall(memoryModel, @selector(modelURL));
        NSURL *compiledURL = objectCall(compiledA, @selector(modelURL));
        NSURL *sourceURL = objectCall(compiledA, @selector(sourceURL));
        id cacheIdentifier = objectCall(compiledA, @selector(cacheURLIdentifier));
        id derivedCacheIdentifier = objectCall(compiledA, @selector(getCacheURLIdentifier));
        NSString *localPath = objectCall(memoryModel, @selector(localModelPath));
        NSURL *savedURL = [saveResult isKindOfClass:NSURL.class] ? saveResult :
            ([compiledURL isKindOfClass:NSURL.class] ? compiledURL : memoryURL);
        receipt[@"runtime_paths"] = @{
            @"save_model_files": [saveResult description] ?: @"nil",
            @"local_model_path": localPath ?: @"nil", @"memory_model_url": [memoryURL description] ?: @"nil",
            @"compiled_model_url": [compiledURL description] ?: @"nil", @"source_url": [sourceURL description] ?: @"nil",
            @"cache_url_identifier": [cacheIdentifier description] ?: @"nil",
            @"derived_cache_url_identifier": [derivedCacheIdentifier description] ?: @"nil",
        };
        receipt[@"runtime_artifact_files"] = fileInventory(savedURL);
        NSArray *shippedHWXPaths = @[
            @"/System/Library/AssetsV2/com_apple_MobileAsset_UAF_Speech_AutomaticSpeechRecognition/purpose_auto/971265b62793d7941762a4aed871a2aa6310c020.asset/.AssetData/VAD.mlmodelc/model.specialization.bundle/H17G.bundle/context_16/multiprocedure/model.hwx",
            @"/System/Library/AssetsV2/com_apple_MobileAsset_UAF_Speech_AutomaticSpeechRecognition/purpose_auto/971265b62793d7941762a4aed871a2aa6310c020.asset/.AssetData/encoder.mlmodelc/model.specialization.bundle/H17G.bundle/streaming_encoder/multiprocedure/model.hwx",
        ];
        NSMutableArray *shippedHWX = [NSMutableArray array];
        for (NSString *path in shippedHWXPaths) {
            NSDictionary *attributes = [files attributesOfItemAtPath:path error:nil];
            [shippedHWX addObject:@{ @"path": path, @"present": @(attributes != nil),
                                      @"bytes": attributes[NSFileSize] ?: @0 }];
        }
        receipt[@"shipped_multiprocedure_hwx_contract"] = @{
            @"files": shippedHWX,
            @"comparison": @"Apple-shipped multiprocedure bundles contain model.hwx; the in-memory compiler save directory contains net.plist plus data and does not emit model.hwx. Presence of shipped HWX is not proof that this compiler cache can be converted or loaded as HWX."
        };

        error = nil; started = mach_absolute_time();
        BOOL loadedA = compiled && ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            memoryModel, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        double loadAMs = gateMilliseconds(started, mach_absolute_time());
        receipt[@"load_count"] = @(loadedA ? 1 : 0);
        NSDictionary *oracleA = nil;
        BOOL validA = loadedA && runExpert(memoryModel, nil, compiledA, YES, inputValues, expected, &oracleA);

        NSString *copyRoot = [NSTemporaryDirectory() stringByAppendingPathComponent:
            [@"ane-fieldfare-saved-" stringByAppendingString:NSUUID.UUID.UUIDString]];
        NSError *copyError = nil;
        BOOL copied = savedURL.isFileURL && [files copyItemAtURL:savedURL
            toURL:[NSURL fileURLWithPath:copyRoot isDirectory:YES] error:&copyError];
        NSError *unloadAError = nil;
        if (loadedA) ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            memoryModel, @selector(unloadWithQoS:error:), 21, &unloadAError);
        memoryModel = nil; compiledA = nil; descriptor = nil;

        NSURL *reloadURL = [NSURL fileURLWithPath:copyRoot isDirectory:YES];
        id client = clientClass ? ((id (*)(Class, SEL))objc_msgSend)(clientClass, @selector(sharedConnection)) : nil;
        NSString *cacheKey = [derivedCacheIdentifier isKindOfClass:NSString.class]
            ? derivedCacheIdentifier : ([cacheIdentifier isKindOfClass:NSString.class]
                ? cacheIdentifier : identifier);
        NSMutableArray *reloadAttempts = [NSMutableArray array];
        id reloaded = nil; BOOL loadedB = NO, validB = NO; double loadBMs = 0;
        NSDictionary *oracleB = nil; NSError *reloadError = nil;
        NSArray *candidateKinds = @[@"modelAtURL:key:", @"modelAtURL:key:modelAttributes:",
                                    @"modelAtURLWithCacheURLIdentifier:key:cacheURLIdentifier:",
                                    @"modelWithCacheURLIdentifier:hash",
                                    @"modelWithCacheURLIdentifier:path"];
        for (NSString *kind in candidateKinds) {
            id candidate = nil;
            if ([kind isEqualToString:@"modelAtURL:key:"] && copied)
                candidate = ((id (*)(Class, SEL, id, id))objc_msgSend)(
                    compiledModelClass, @selector(modelAtURL:key:), reloadURL, nil);
            else if ([kind isEqualToString:@"modelAtURL:key:modelAttributes:"] && copied)
                candidate = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(compiledModelClass,
                    @selector(modelAtURL:key:modelAttributes:), reloadURL, nil,
                    modelAttributesA);
            else if ([kind isEqualToString:@"modelAtURLWithCacheURLIdentifier:key:cacheURLIdentifier:"] && copied && cacheKey)
                candidate = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(compiledModelClass,
                    @selector(modelAtURLWithCacheURLIdentifier:key:cacheURLIdentifier:), reloadURL, nil, cacheKey);
            else if ([kind isEqualToString:@"modelWithCacheURLIdentifier:hash"] && cacheKey)
                candidate = ((id (*)(Class, SEL, id))objc_msgSend)(
                    compiledModelClass, @selector(modelWithCacheURLIdentifier:), cacheKey);
            else if ([kind isEqualToString:@"modelWithCacheURLIdentifier:path"] && localPath)
                candidate = ((id (*)(Class, SEL, id))objc_msgSend)(
                    compiledModelClass, @selector(modelWithCacheURLIdentifier:), localPath);
            reloadError = nil; started = mach_absolute_time();
            BOOL candidateLoaded = candidate && client &&
                ((BOOL (*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
                    client, @selector(loadModel:options:qos:error:), candidate, @{}, 21, &reloadError);
            double candidateLoadMs = gateMilliseconds(started, mach_absolute_time());
            NSDictionary *candidateOracle = nil;
            BOOL candidateValid = candidateLoaded &&
                runExpert(nil, client, candidate, NO, inputValues, expected, &candidateOracle);
            [reloadAttempts addObject:@{ @"constructor": kind,
                @"created": @(candidate != nil), @"loaded": @(candidateLoaded),
                @"load_ms": @(candidateLoadMs), @"error": reloadError.localizedDescription ?: @"none",
                @"oracle": candidateOracle ?: @{} }];
            if (candidateValid) {
                reloaded = candidate; loadedB = YES; validB = YES; loadBMs = candidateLoadMs;
                oracleB = candidateOracle; break;
            }
            if (candidateLoaded) {
                NSError *candidateUnloadError = nil;
                ((BOOL (*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(client,
                    @selector(unloadModel:options:qos:error:), candidate, @{}, 21, &candidateUnloadError);
            }
        }
        NSError *unloadBError = nil;
        if (loadedB) ((BOOL (*)(id, SEL, id, id, unsigned int, NSError **))objc_msgSend)(
            client, @selector(unloadModel:options:qos:error:), reloaded, @{}, 21, &unloadBError);

        BOOL validated = validA && copied && reloaded && loadedB && validB;
        receipt[@"status"] = validated ? @"validated" : @"rejected";
        receipt[@"stage"] = validated ? @"compile_save_destroy_reload_without_mil_oracle" : @"save_or_reload_lifecycle";
        receipt[@"exit_status"] = @(validated ? 0 : 1);
        receipt[@"load_count"] = @(loadedA + loadedB);
        receipt[@"reload_load_attempt_count"] = @(reloadAttempts.count);
        receipt[@"file_activity_observation"] = @{
            @"fs_usage_attempted": @YES, @"available_without_privilege": @NO,
            @"error": @"fs_usage must be run as root; no privilege escalation was attempted"
        };
        receipt[@"lifecycle"] = @{
            @"cold_load_ms": @(loadAMs), @"cold_oracle": oracleA ?: @{},
            @"save_path_copied": @(copied), @"copy_error": copyError.localizedDescription ?: @"none",
            @"copied_artifact_files": fileInventory(reloadURL),
            @"reload_model_created": @(reloaded != nil), @"reload_without_mil_offered": @YES,
            @"reload_attempts": reloadAttempts,
            @"reload_load_ms": @(loadBMs), @"reload_error": reloadError.localizedDescription ?: @"none",
            @"reload_oracle": oracleB ?: @{},
            @"original_unload_error": unloadAError.localizedDescription ?: @"none",
            @"reload_unload_error": unloadBError.localizedDescription ?: @"none",
        };
        receipt[@"claim_boundary"] = validated
            ? @"A runtime-produced private compiled directory was copied after saveModelFiles, the original in-memory model was unloaded and released, and _ANEModel modelAtURL:key: plus _ANEClient loaded and numerically evaluated expert_0 without receiving MIL or source weights. This validates compile-free reload for this exact M5/macOS build and artifact format; it does not establish cross-OS portability, code-signing policy, indefinite cache stability, or public API support."
            : @"The compiler-produced directory was inventoried and the ABI-authoritative _ANEModel/_ANEClient reload path was attempted without MIL. Failure rejects compile-free reload for this construction; selector presence alone is not a capability claim.";
        emit(receipt);
        [files removeItemAtPath:copyRoot error:nil];
        [files removeItemAtPath:temporary error:nil];
        free(expected);
        return validated ? 0 : 1;
    }
}
