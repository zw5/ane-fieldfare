#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

static mach_timebase_info_data_t gTimebase;

static double milliseconds(uint64_t ticks) {
    return (double)ticks * gTimebase.numer / gTimebase.denom / 1e6;
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

static NSString *dynamicProjectionMIL(int width, int positions) {
    int packedPositions = positions + width;
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "  func main<ios18>(tensor<fp32, [1, %d, 1, %d]> packed) {\n"
         "    string fp16_name = const()[name = string(\"fp16_name\"), val = string(\"fp16\")];\n"
         "    tensor<fp16, [1, %d, 1, %d]> packed16 = cast(dtype = fp16_name, x = packed)[name = string(\"cast_in\")];\n"
         "    tensor<int32, [4]> x_begin = const()[name = string(\"x_begin\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "    tensor<int32, [4]> x_size = const()[name = string(\"x_size\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
         "    tensor<fp16, [1, %d, 1, %d]> x = slice_by_size(x = packed16, begin = x_begin, size = x_size)[name = string(\"activation\")];\n"
         "    tensor<int32, [4]> w_begin = const()[name = string(\"w_begin\"), val = tensor<int32, [4]>([0, 0, 0, %d])];\n"
         "    tensor<int32, [4]> w_size = const()[name = string(\"w_size\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
         "    tensor<fp16, [1, %d, 1, %d]> weight_slice = slice_by_size(x = packed16, begin = w_begin, size = w_size)[name = string(\"weight_slice\")];\n"
         "    tensor<int32, [4]> x_shape = const()[name = string(\"x_shape\"), val = tensor<int32, [4]>([1, 1, %d, %d])];\n"
         "    tensor<fp16, [1, 1, %d, %d]> x_matrix = reshape(shape = x_shape, x = x)[name = string(\"x_matrix\")];\n"
         "    tensor<int32, [4]> transpose_perm = const()[name = string(\"transpose_perm\"), val = tensor<int32, [4]>([0, 1, 3, 2])];\n"
         "    tensor<fp16, [1, 1, %d, %d]> x_transposed = transpose(perm = transpose_perm, x = x_matrix)[name = string(\"x_transposed\")];\n"
         "    tensor<int32, [4]> w_shape = const()[name = string(\"w_shape\"), val = tensor<int32, [4]>([1, 1, %d, %d])];\n"
         "    tensor<fp16, [1, 1, %d, %d]> weight = reshape(shape = w_shape, x = weight_slice)[name = string(\"weight\")];\n"
         "    bool no_transpose = const()[name = string(\"no_transpose\"), val = bool(false)];\n"
         "    tensor<fp16, [1, 1, %d, %d]> projected = matmul(transpose_x = no_transpose, transpose_y = no_transpose, x = x_transposed, y = weight)[name = string(\"projection\")];\n"
         "    tensor<fp16, [1, 1, %d, %d]> projected_transposed = transpose(perm = transpose_perm, x = projected)[name = string(\"projected_transposed\")];\n"
         "    tensor<int32, [4]> out_shape = const()[name = string(\"out_shape\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
         "    tensor<fp16, [1, %d, 1, %d]> out16 = reshape(shape = out_shape, x = projected_transposed)[name = string(\"out16\")];\n"
         "    string fp32_name = const()[name = string(\"fp32_name\"), val = string(\"fp32\")];\n"
         "    tensor<fp32, [1, %d, 1, %d]> out = cast(dtype = fp32_name, x = out16)[name = string(\"cast_out\")];\n"
         "  } -> (out);\n"
         "}\n",
         width, packedPositions,
         width, packedPositions,
         width, positions, width, positions,
         positions,
         width, width, width, width,
         width, width,
         width, positions, width, positions,
         positions, width,
         width, width, width, width,
         positions, width,
         width, positions, width, positions,
         width, positions];
}

static void fillPayload(IOSurfaceRef surface, int width, int positions, float scale) {
    IOSurfaceLock(surface, 0, NULL);
    float *values = IOSurfaceGetBaseAddress(surface);
    int stride = positions + width;
    memset(values, 0, (NSUInteger)width * stride * sizeof(float));
    for (int channel = 0; channel < width; ++channel) {
        for (int position = 0; position < positions; ++position) {
            values[channel * stride + position] =
                ((float)((channel * 17 + position * 13) % 101) - 50.0f) / 64.0f;
        }
        values[channel * stride + positions + channel] = scale;
    }
    IOSurfaceUnlock(surface, 0, NULL);
}

static NSDictionary *validateOutput(IOSurfaceRef output, int width, int positions, float scale) {
    double maxError = 0.0;
    NSUInteger mismatches = 0;
    uint64_t checksum = 1469598103934665603ULL;
    IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
    const float *values = IOSurfaceGetBaseAddress(output);
    for (int channel = 0; channel < width; ++channel) {
        for (int position = 0; position < positions; ++position) {
            float input = ((float)((channel * 17 + position * 13) % 101) - 50.0f) / 64.0f;
            float expected = (float)((__fp16)input * (__fp16)scale);
            float actual = values[channel * positions + position];
            double error = fabs((double)actual - expected);
            maxError = fmax(maxError, error);
            if (error > 0.002) mismatches++;
            uint32_t bits = 0;
            memcpy(&bits, &actual, sizeof(bits));
            checksum ^= bits;
            checksum *= 1099511628211ULL;
        }
    }
    IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
    return @{
        @"scale": @(scale),
        @"max_absolute_error": @(maxError),
        @"mismatches": @(mismatches),
        @"checksum": [NSString stringWithFormat:@"0x%016llx", checksum],
    };
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static id callObject(id object, SEL selector) {
    return object && [object respondsToSelector:selector]
        ? ((id (*)(id, SEL))objc_msgSend)(object, selector)
        : nil;
}

static uint64_t callUInt64(id object, SEL selector) {
    return object && [object respondsToSelector:selector]
        ? ((uint64_t (*)(id, SEL))objc_msgSend)(object, selector)
        : 0;
}

static NSDictionary *procedureMetadata(id inMemoryModel) {
    id compiledModel = callObject(inMemoryModel, @selector(model));
    id program = callObject(inMemoryModel, @selector(program));
    NSArray *procedures = callObject(compiledModel, @selector(procedures));
    NSMutableArray *records = [NSMutableArray array];
    for (id procedure in procedures ?: @[]) {
        NSString *symbol = callObject(procedure, @selector(procedureSymbol));
        NSArray *weights = callObject(procedure, @selector(weightArray));
        [records addObject:@{
            @"procedure_symbol": symbol ?: @"unknown",
            @"weight_count": @(weights.count),
            @"mutable_weights_buffer_id": @(callUInt64(procedure, @selector(mutableWeightsBufferID))),
            @"mutable_weights_buffer_size": @(callUInt64(procedure, @selector(mutableWeightsBufferSize))),
        }];
    }
    NSDictionary *attributes = callObject(inMemoryModel, @selector(modelAttributes));
    NSDictionary *description = attributes[@"ANEFModelDescription"];
    NSArray *attributeProcedures = description[@"ANEFModelProcedures"];
    return @{
        @"program_present": @(program != nil),
        @"program_handle": @(callUInt64(inMemoryModel, @selector(programHandle))),
        @"procedure_count": @(procedures.count),
        @"procedures": records,
        @"attribute_procedure_count": @(attributeProcedures.count),
        @"mutable_weight_info_present": @([attributes.description containsString:@"MutableWeight"]),
    };
}

int main(void) {
    @autoreleasepool {
        mach_timebase_info(&gTimebase);
        NSDictionary *environment = NSProcessInfo.processInfo.environment;
        int width = [environment[@"ANE_FIELDFARE_WIDTH"] intValue];
        int positions = [environment[@"ANE_FIELDFARE_POSITIONS"] intValue];
        if (width == 0) width = 64;
        if (positions == 0) positions = 64;
        if (width < 1 || width > 4096 || positions < 1 || positions > 4096) {
            emit(@{@"schema": @"ane-fieldfare.weight-gate.v1", @"status": @"failed", @"stage": @"arguments", @"detail": @"width and positions must each be 1...4096"});
            return 2;
        }
        const int packedPositions = positions + width;
        const NSUInteger inputBytes = (NSUInteger)width * packedPositions * sizeof(float);
        const NSUInteger outputBytes = (NSUInteger)width * positions * sizeof(float);

        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!framework || !descriptorClass || !modelClass || !requestClass || !surfaceClass) {
            emit(@{@"schema": @"ane-fieldfare.weight-gate.v1", @"status": @"failed", @"stage": @"runtime_surface"});
            return 1;
        }

        NSData *mil = [dynamicProjectionMIL(width, positions) dataUsingEncoding:NSUTF8StringEncoding];
        const char *dumpPath = getenv("ANE_FIELDFARE_DUMP_MIL");
        if (dumpPath && dumpPath[0]) {
            [mil writeToFile:[NSString stringWithUTF8String:dumpPath] atomically:YES];
        }
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
        id model = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        if (!model) {
            emit(@{@"schema": @"ane-fieldfare.weight-gate.v1", @"status": @"failed", @"stage": @"model"});
            return 1;
        }

        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *files = NSFileManager.defaultManager;
        [files createDirectoryAtPath:[temporary stringByAppendingPathComponent:@"weights"]
         withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        NSError *error = nil;
        uint64_t start = mach_absolute_time();
        BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        double compileMs = milliseconds(mach_absolute_time() - start);
        if (!ok) {
            emit(@{@"schema": @"ane-fieldfare.weight-gate.v1", @"status": @"failed", @"stage": @"compile", @"detail": error.localizedDescription ?: @"unknown"});
            [files removeItemAtPath:temporary error:nil];
            return 1;
        }

        start = mach_absolute_time();
        ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        double loadMs = milliseconds(mach_absolute_time() - start);
        if (!ok) {
            emit(@{@"schema": @"ane-fieldfare.weight-gate.v1", @"status": @"failed", @"stage": @"load", @"detail": error.localizedDescription ?: @"unknown"});
            [files removeItemAtPath:temporary error:nil];
            return 1;
        }
        NSDictionary *metadata = procedureMetadata(model);

        IOSurfaceRef input = makeSurface(inputBytes);
        IOSurfaceRef output = makeSurface(outputBytes);
        id wrappedInput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(surfaceClass, @selector(objectWithIOSurface:), input);
        id wrappedOutput = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(surfaceClass, @selector(objectWithIOSurface:), output);
        id request = ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wrappedInput], @[@0], @[wrappedOutput], @[@0], nil, nil, @0);

        NSArray<NSNumber *> *scales = @[@1.0f, @2.0f, @1.0f];
        NSMutableArray *runs = [NSMutableArray array];
        for (NSNumber *scaleNumber in scales) {
            float scale = scaleNumber.floatValue;
            fillPayload(input, width, positions, scale);
            IOSurfaceLock(output, 0, NULL);
            memset(IOSurfaceGetBaseAddress(output), 0xA5, outputBytes);
            IOSurfaceUnlock(output, 0, NULL);
            start = mach_absolute_time();
            ok = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &error);
            double evaluationMs = milliseconds(mach_absolute_time() - start);
            if (!ok) break;
            NSMutableDictionary *run = [validateOutput(output, width, positions, scale) mutableCopy];
            run[@"evaluation_ms"] = @(evaluationMs);
            [runs addObject:run];
        }

        BOOL numerical = ok && runs.count == 3;
        for (NSDictionary *run in runs) numerical = numerical && [run[@"mismatches"] unsignedIntegerValue] == 0;
        BOOL returnedToA = numerical && [runs[0][@"checksum"] isEqual:runs[2][@"checksum"]];

        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        [files removeItemAtPath:temporary error:nil];
        CFRelease(input);
        CFRelease(output);

        NSProcessInfo *process = NSProcessInfo.processInfo;
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.weight-gate.v1",
            @"status": numerical && returnedToA ? @"validated" : @"failed",
            @"gate": @"dynamic_weight_runtime_input_control",
            @"shape": @{@"width": @(width), @"positions": @(positions)},
            @"compile_count": @1,
            @"load_count": @1,
            @"compile_ms": @(compileMs),
            @"load_ms": @(loadMs),
            @"compiled_metadata": metadata,
            @"runs": runs,
            @"a_b_a_checksum_restored": @(returnedToA),
            @"host": process.hostName ?: @"unknown",
            @"os": process.operatingSystemVersionString ?: @"unknown",
            @"claim_boundary": @"ANE dynamic-weight projection with weights in an ordinary FP32 input carrier; not compiler-declared mutable weights, compact prepared weights, Gemma expert execution, or generation",
        };
        emit(receipt);
        return numerical && returnedToA ? 0 : 1;
    }
}
