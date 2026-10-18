#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>

static void emit(NSDictionary *value) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static NSData *identityBlob(int width) {
    NSUInteger payloadBytes = (NSUInteger)width * width * sizeof(__fp16);
    NSMutableData *blob = [NSMutableData dataWithLength:128 + payloadBytes];
    uint8_t *bytes = blob.mutableBytes;
    bytes[0] = 1;
    bytes[4] = 2;
    bytes[64] = 0xef; bytes[65] = 0xbe; bytes[66] = 0xad; bytes[67] = 0xde;
    bytes[68] = 1;
    *(uint32_t *)(bytes + 72) = (uint32_t)payloadBytes;
    *(uint32_t *)(bytes + 80) = 128;
    __fp16 *weights = (__fp16 *)(bytes + 128);
    for (int row = 0; row < width; row++) weights[row * width + row] = 1.0;
    return blob;
}

static NSString *projectionMIL(int width, int positions) {
    NSString *mode = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_MUTABLE_CONSTRUCTION"] ?: @"buildinfo_hash";
    NSString *marker = @"";
    NSString *prefix = @"";
    if ([mode isEqualToString:@"buildinfo_plain"]) marker = @", {\"TorchImport.mutable_weights\", \"true\"}";
    if ([mode isEqualToString:@"buildinfo_hash"]) marker = @", {\"#TorchImport.mutable_weights\", \"true\"}";
    if ([mode isEqualToString:@"buildinfo_symbol"]) marker = @", {\"TorchImport.mutable_weights\", \"streamed_weight\"}";
    if ([mode isEqualToString:@"buildinfo_hash_symbol"]) marker = @", {\"#TorchImport.mutable_weights\", \"streamed_weight\"}";
    if ([mode isEqualToString:@"buildinfo_symbol_array"]) marker = @", {\"TorchImport.mutable_weights\", \"[streamed_weight]\"}";
    if ([mode isEqualToString:@"buildinfo_hash_symbol_array"]) marker = @", {\"#TorchImport.mutable_weights\", \"[streamed_weight]\"}";
    if ([mode isEqualToString:@"top_comment"]) prefix = @"#TorchImport.mutable_weights\n";
    NSString *body = [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}__MARKER__})]\n"
         "{\n"
         "  func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
         "    string fp16_name = const()[name = string(\"fp16_name\"), val = string(\"fp16\")];\n"
         "    tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = fp16_name, x = x)[name = string(\"cast_in\")];\n"
         "    tensor<fp16, [%d, %d, 1, 1]> weight = const()[name = string(\"streamed_weight\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
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
         width, positions, width, positions,
         width, width, width, width,
         width, positions, width, positions];
    body = [body stringByReplacingOccurrencesOfString:@"__MARKER__" withString:marker];
    return [prefix stringByAppendingString:body];
}

static NSArray *mutableRecords(NSDictionary *attributes) {
    NSDictionary *description = attributes[@"ANEFModelDescription"];
    NSArray *procedures = description[@"ANEFModelProcedures"];
    NSMutableArray *records = [NSMutableArray array];
    for (NSDictionary *procedure in procedures ?: @[]) {
        id info = procedure[@"ANEFModelMutableWeightBufferInfo"] ?: procedure[@"mutableWeightsBuffersInfo"];
        [records addObject:@{
            @"procedure_id": procedure[@"ANEFModelProcedureID"] ?: @(-1),
            @"mutable_info_present": @(info != nil),
            @"mutable_info": info ? [info description] : @"none",
        }];
    }
    return records;
}

int main(void) {
    @autoreleasepool {
        const int width = 64, positions = 64;
        NSString *construction = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_MUTABLE_CONSTRUCTION"] ?: @"buildinfo_hash";
        void *framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        if (!framework || !descriptorClass || !modelClass) {
            emit(@{@"schema": @"ane-fieldfare.mutable-gate.v1", @"status": @"rejected", @"stage": @"runtime_surface"});
            return 1;
        }

        NSData *mil = [projectionMIL(width, positions) dataUsingEncoding:NSUTF8StringEncoding];
        NSData *blob = identityBlob(width);
        NSDictionary *weights = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": blob}};
        NSDictionary *descriptorOptionsDictionary = [construction hasPrefix:@"option_"] ? @{
            [construction containsString:@"hash"] ? @"#TorchImport.mutable_weights" : @"TorchImport.mutable_weights":
                [construction containsString:@"symbol"] ? @"streamed_weight" : @"true"
        } : nil;
        NSData *descriptorOptions = descriptorOptionsDictionary ?
            [NSPropertyListSerialization dataWithPropertyList:descriptorOptionsDictionary
                                                        format:NSPropertyListBinaryFormat_v1_0
                                                       options:0 error:nil] : nil;
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, weights, descriptorOptions);
        id model = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        if (!model) {
            emit(@{@"schema": @"ane-fieldfare.mutable-gate.v1", @"status": @"rejected", @"stage": @"model"});
            return 1;
        }

        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *files = NSFileManager.defaultManager;
        [files createDirectoryAtPath:[temporary stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [blob writeToFile:[temporary stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        NSError *error = nil;
        NSDictionary *compileOptions = [construction hasPrefix:@"compile_option_"] ? @{
            [construction containsString:@"hash"] ? @"#TorchImport.mutable_weights" : @"TorchImport.mutable_weights":
                [construction containsString:@"symbol"] ? @"streamed_weight" : @"true"
        } : @{};
        BOOL compiled = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, compileOptions, &error);
        if (!compiled) {
            emit(@{@"schema": @"ane-fieldfare.mutable-gate.v1", @"status": @"rejected", @"stage": @"compile", @"detail": error.localizedDescription ?: @"unknown", @"construction": construction});
            [files removeItemAtPath:temporary error:nil];
            return 1;
        }
        BOOL loaded = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        NSDictionary *attributes = loaded ? ((id (*)(id, SEL))objc_msgSend)(model, @selector(modelAttributes)) : @{};
        NSArray *records = mutableRecords(attributes ?: @{});
        BOOL declared = NO;
        for (NSDictionary *record in records) declared |= [record[@"mutable_info_present"] boolValue];
        id program = loaded ? ((id (*)(id, SEL))objc_msgSend)(model, @selector(program)) : nil;
        void *mappedPointer = NULL;
        uint64_t mappedSize = 0;
        NSError *mapError = nil;
        BOOL mapped = program && ((BOOL (*)(id, SEL, id, uint64_t, void **, uint64_t *, NSError **))objc_msgSend)(
            program, @selector(mapMutableWeightsBufferDirectForProcedure:bufferID:buffer:size:error:),
            @"main", 0, &mappedPointer, &mappedSize, &mapError);
        emit(@{
            @"schema": @"ane-fieldfare.mutable-gate.v1",
            @"status": declared ? @"accepted" : @"rejected",
            @"stage": loaded ? @"compiler_metadata" : @"load",
            @"construction": construction,
            @"compile_count": @1,
            @"load_count": loaded ? @1 : @0,
            @"procedures": records,
            @"map_attempted": @(program != nil),
            @"map_succeeded": @(mapped),
            @"mapped_size": @(mappedSize),
            @"mapped_pointer_nonnull": @(mappedPointer != NULL),
            @"map_error": mapError.localizedDescription ?: @"none",
            @"detail": loaded ? @"compiler emitted no observable mutable-weight metadata" : (error.localizedDescription ?: @"unknown"),
            @"claim_boundary": @"Compiler construction probe only; accepted does not mean mapped, synchronized, numerically validated, compact, or generation-capable",
        });
        if (mapped) ((BOOL (*)(id, SEL, id, uint64_t))objc_msgSend)(
            program, @selector(unmapMutableWeightsBufferDirectForProcedure:bufferID:), @"main", 0);
        if (loaded) ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &error);
        [files removeItemAtPath:temporary error:nil];
        return declared ? 0 : 1;
    }
}
