#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>
#import <dlfcn.h>

static NSArray *methodInventory(Class cls) {
    NSMutableArray *records = [NSMutableArray array];
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; ++i) {
        NSString *name = NSStringFromSelector(method_getName(methods[i]));
        NSString *lower = name.lowercaseString;
        if (![lower containsString:@"mutable"] && ![lower containsString:@"weight"] &&
            ![lower containsString:@"procedure"] && ![lower containsString:@"map"] &&
            ![lower containsString:@"sync"] && ![lower containsString:@"unmap"])
            continue;
        const char *encoding = method_getTypeEncoding(methods[i]);
        [records addObject:@{@"selector": name,
                             @"encoding": encoding ? @(encoding) : @"unknown"}];
    }
    free(methods);
    return [records sortedArrayUsingComparator:^NSComparisonResult(id a, id b) {
        return [a[@"selector"] compare:b[@"selector"]];
    }];
}

static NSArray *completeMethodInventory(Class cls) {
    NSMutableArray *records = [NSMutableArray array];
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    for (unsigned int i = 0; i < count; ++i) {
        const char *encoding = method_getTypeEncoding(methods[i]);
        [records addObject:@{
            @"selector": NSStringFromSelector(method_getName(methods[i])),
            @"encoding": encoding ? @(encoding) : @"unknown"
        }];
    }
    free(methods);
    return [records sortedArrayUsingComparator:^NSComparisonResult(id a, id b) {
        return [a[@"selector"] compare:b[@"selector"]];
    }];
}

static NSArray *ivarInventory(Class cls) {
    NSMutableArray *records = [NSMutableArray array];
    unsigned int count = 0;
    Ivar *ivars = class_copyIvarList(cls, &count);
    for (unsigned int i = 0; i < count; ++i) {
        const char *name = ivar_getName(ivars[i]);
        const char *encoding = ivar_getTypeEncoding(ivars[i]);
        NSString *n = name ? @(name) : @"unknown";
        NSString *lower = n.lowercaseString;
        if (![lower containsString:@"mutable"] && ![lower containsString:@"weight"] &&
            ![lower containsString:@"procedure"] && ![lower containsString:@"buffer"])
            continue;
        [records addObject:@{@"name": n,
                             @"encoding": encoding ? @(encoding) : @"unknown",
                             @"offset": @(ivar_getOffset(ivars[i]))}];
    }
    free(ivars);
    return records;
}

static NSDictionary *classInventory(NSString *name) {
    Class cls = NSClassFromString(name);
    if (!cls) return @{@"class": name, @"present": @NO};
    return @{@"class": name, @"present": @YES,
             @"instance_methods": methodInventory(cls),
             @"class_methods": methodInventory(object_getClass(cls)),
             @"ivars": ivarInventory(cls)};
}

static NSDictionary *completeClassInventory(NSString *name) {
    Class cls = NSClassFromString(name);
    if (!cls) return @{@"class": name, @"present": @NO};
    return @{@"class": name, @"present": @YES,
             @"instance_methods": completeMethodInventory(cls),
             @"class_methods": completeMethodInventory(object_getClass(cls)),
             @"ivars": ivarInventory(cls)};
}

static NSArray *classesOwnedByImage(NSString *imageNeedle) {
    int count = objc_getClassList(NULL, 0);
    __unsafe_unretained Class *classes = count > 0
        ? (__unsafe_unretained Class *)calloc((size_t)count, sizeof(*classes)) : NULL;
    int populated = classes ? objc_getClassList(classes, count) : 0;
    NSMutableArray *records = [NSMutableArray array];
    for (int index = 0; index < populated; ++index) {
        const char *image = class_getImageName(classes[index]);
        if (!image || !strstr(image, imageNeedle.UTF8String)) continue;
        [records addObject:completeClassInventory(NSStringFromClass(classes[index]))];
    }
    free(classes);
    return [records sortedArrayUsingComparator:^NSComparisonResult(id a, id b) {
        return [a[@"class"] compare:b[@"class"]];
    }];
}

static NSDictionary *compilerAuthoredPackage(BOOL variableFromFeed) {
    NSString *root = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"ane-fieldfare-prepared-%@", NSUUID.UUID.UUIDString]];
    NSURL *url = [NSURL fileURLWithPath:[root stringByAppendingPathExtension:@"mpsgraphpackage"]];
    NSMutableData *initial = [NSMutableData dataWithLength:64 * sizeof(__fp16)];
    __fp16 *values = initial.mutableBytes;
    for (NSUInteger i = 0; i < 64; ++i) values[i] = 2.0;

    MPSGraph *graph = [MPSGraph new];
    NSArray *shape = @[@1, @64];
    MPSGraphTensor *input = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat16
                                                   name:@"activation"];
    MPSGraphTensor *initializer = variableFromFeed
        ? [graph placeholderWithShape:shape dataType:MPSDataTypeFloat16 name:@"weight_initializer"]
        : nil;
    MPSGraphTensor *variable = variableFromFeed
        ? [graph variableFromTensorWithTensor:initializer name:@"streamed_weight"]
        : [graph variableWithData:initial shape:shape dataType:MPSDataTypeFloat16
                            name:@"streamed_weight"];
    MPSGraphTensor *weight = [graph readVariable:variable name:@"read_streamed_weight"];
    MPSGraphTensor *output = [graph multiplicationWithPrimaryTensor:input
                                                    secondaryTensor:weight
                                                               name:@"weighted_output"];
    MPSGraphShapedType *inputType = [[MPSGraphShapedType alloc] initWithShape:shape
                                                                    dataType:MPSDataTypeFloat16];
    MPSGraphCompilationDescriptor *compile = [MPSGraphCompilationDescriptor new];
    compile.waitForCompilationCompletion = YES;
    id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
    MPSGraphDevice *graphDevice = metalDevice ? [MPSGraphDevice deviceWithMTLDevice:metalDevice] : nil;
    NSMutableDictionary *feeds = [@{input: inputType} mutableCopy];
    if (initializer) feeds[initializer] = inputType;
    MPSGraphExecutable *executable = [graph compileWithDevice:graphDevice
                                                       feeds:feeds
                                               targetTensors:@[output]
                                            targetOperations:nil
                                       compilationDescriptor:compile];
    NSArray *inputTypes = variableFromFeed ? @[inputType, inputType] : @[inputType];
    [executable specializeWithDevice:graphDevice inputTypes:inputTypes
               compilationDescriptor:compile];
    BOOL serialized = NO;
    NSString *exceptionText = @"none";
    @try {
        MPSGraphExecutableSerializationDescriptor *descriptor =
            [MPSGraphExecutableSerializationDescriptor new];
        descriptor.deploymentPlatform = MPSGraphDeploymentPlatformMacOS;
        descriptor.minimumDeploymentTarget = @"26.0";
        [executable serializeToMPSGraphPackageAtURL:url descriptor:descriptor];
        serialized = YES;
    } @catch (NSException *exception) {
        exceptionText = [NSString stringWithFormat:@"%@: %@", exception.name,
                                                   exception.reason ?: @"none"];
    }

    NSString *manifestPath = [url.path stringByAppendingPathComponent:@"manifest.plist"];
    NSDictionary *manifest = [NSDictionary dictionaryWithContentsOfFile:manifestPath] ?: @{};
    id mutableSymbols = manifest[@"Symbols With Mutable Weight"];
    NSMutableArray *fileRecords = [NSMutableArray array];
    NSDirectoryEnumerator *enumerator = [NSFileManager.defaultManager enumeratorAtPath:url.path];
    for (NSString *relative in enumerator) {
        NSDictionary *attributes = [enumerator fileAttributes];
        if ([attributes[NSFileType] isEqual:NSFileTypeRegular])
            [fileRecords addObject:@{@"path": relative, @"bytes": attributes[NSFileSize] ?: @0}];
    }
    NSDictionary *result = @{
        @"construction": variableFromFeed
            ? @"MPSGraph variableFromTensor(weight_initializer) -> readVariable -> elementwise multiply"
            : @"MPSGraph variableWithData -> readVariable -> elementwise multiply",
            @"compiler": @"public MetalPerformanceShadersGraph device specialization and package serializer",
        @"serialized": @(serialized),
        @"manifest_present": @([NSFileManager.defaultManager fileExistsAtPath:manifestPath]),
        @"manifest_keys": [[manifest allKeys] sortedArrayUsingSelector:@selector(compare:)],
        @"mutable_symbol_key_present": @(mutableSymbols != nil),
        @"mutable_symbols": mutableSymbols ?: @[],
        @"resource_storage_mode": manifest[@"Resource Storage Mode"] ?: @"absent",
        @"files": fileRecords,
        @"exception": exceptionText,
        @"private_ane_procedure_symbol": @"absent",
        @"private_ane_mutable_buffer_id": @"absent",
        @"private_ane_prepared_size": @"absent"
    };
    [NSFileManager.defaultManager removeItemAtURL:url error:nil];
    return result;
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

int main(void) {
    @autoreleasepool {
        void *ane = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                          RTLD_NOW | RTLD_LOCAL);
        void *compiler = dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",
                               RTLD_NOW | RTLD_LOCAL);
        NSArray *classes = @[
            classInventory(@"_ANEProcedureData"),
            classInventory(@"_ANEProgramForEvaluation"),
            classInventory(@"_ANEClient"),
            classInventory(@"_ANEModel"),
            classInventory(@"_ANEInMemoryModel")
        ];
        NSArray *compilerClasses = @[
            completeClassInventory(@"_ANECompiler"),
            completeClassInventory(@"_ANEMILCompiler"),
            completeClassInventory(@"_ANEMLIRCompiler"),
            completeClassInventory(@"_ANECoreMLModelCompiler"),
            completeClassInventory(@"_ANEWeight")
        ];
        NSArray *packages = @[compilerAuthoredPackage(NO), compilerAuthoredPackage(YES)];
        BOOL allSerialized = YES, mutableDeclaration = NO, authoritativeTuple = NO;
        for (NSDictionary *package in packages) {
            allSerialized &= [package[@"serialized"] boolValue];
            mutableDeclaration |= [package[@"serialized"] boolValue] &&
                [package[@"mutable_symbol_key_present"] boolValue];
            authoritativeTuple |= ![package[@"private_ane_procedure_symbol"] isEqual:@"absent"] &&
                ![package[@"private_ane_mutable_buffer_id"] isEqual:@"absent"] &&
                ![package[@"private_ane_prepared_size"] isEqual:@"absent"];
        }
        emit(@{
            @"schema": @"ane-fieldfare.prepared-weight-api.v2",
            @"status": @"rejected",
            @"stage": authoritativeTuple ? @"not_executed" : @"compiler_metadata_incomplete",
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"ane_architecture": @"Apple M5 / H17G",
            @"frameworks": @{@"AppleNeuralEngine": @(ane != NULL),
                               @"ANECompiler": @(compiler != NULL)},
            @"runtime_inventory": classes,
            @"compiler_runtime_inventory": compilerClasses,
            @"ane_compiler_owned_classes": classesOwnedByImage(@"ANECompiler"),
            @"compiler_authored_candidates": packages,
            @"package_serialization_accepted": @(allSerialized),
            @"mutable_declaration_accepted": @(mutableDeclaration),
            @"authoritative_private_tuple_complete": @(authoritativeTuple),
            @"map_attempted": @NO,
            @"sync_attempted": @NO,
            @"unmap_attempted": @NO,
            @"numerical_runs": @0,
            @"ordinary_weight_inputs": @0,
            @"compile_count": @2,
            @"load_count": @0,
            @"detail": mutableDeclaration
                ? @"The compiler-authored package records its mutable-symbol field, but does not expose the private ANE procedure symbol, buffer ID, or prepared byte size required for a safe map. The gate therefore rejects before map/sync/unmap and A/B/A."
                : @"The compiler-authored variable package serialized successfully, but its emitted manifest did not declare a mutable-weight symbol or the private ANE procedure symbol, buffer ID, and prepared size. The gate therefore rejects before map/sync/unmap and A/B/A.",
            @"claim_boundary": @"Validates live private API presence and a compiler-authored mutable-variable package construction only. Prepared mutable weights remain rejected because the compiler output does not bridge that package symbol to an _ANEProcedureData buffer ID and prepared size. No IDs, sizes, or payload layouts were guessed; no ordinary request input carried weights and no numerical execution occurred."
        });
        return 1;
    }
}
