#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

#include <math.h>
#include <string.h>

enum { Width = 64, Positions = 64, GroupSize = 64, GroupsPerRow = Width / GroupSize };

static mach_timebase_info_data_t timebase;

static double milliseconds(uint64_t ticks) {
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)ticks * timebase.numer / timebase.denom / 1e6;
}

static void emit(NSDictionary *value) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static IOSurfaceRef makeSurface(uint64_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    });
}

static bool fp16QuantizedInput(void) {
    const char *value = getenv("ANE_FIELDFARE_COMPACT_Q_DTYPE");
    return value && strcmp(value, "fp16") == 0;
}

static size_t roundUp64(size_t value) {
    return (value + 63u) & ~(size_t)63u;
}

static NSString *compactProjectionMIL(void) {
    NSString *quantizedType = fp16QuantizedInput() ? @"fp16" : @"uint8";
    NSString *quantizedCast = fp16QuantizedInput() ? @"identity(x=q)" : @"cast(dtype=f16,x=q)";
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "  func main<ios18>(tensor<%@,[1,%d,1,%d]> q, tensor<fp16,[1,%d,1,%d]> s, tensor<fp16,[1,%d,1,%d]> b, tensor<fp16,[1,%d,1,%d]> x) {\n"
         "    string f16=const()[name=string(\"f16\"),val=string(\"fp16\")];\n"
         "    tensor<fp16,[1,%d,1,%d]> q16=%@[name=string(\"q16\")];\n"
         "    tensor<int32,[4]> reps=const()[name=string(\"reps\"),val=tensor<int32,[4]>([1,1,1,%d])];\n"
         "    tensor<fp16,[1,%d,1,%d]> se=tile(reps=reps,x=s)[name=string(\"scale_expanded\")];\n"
         "    tensor<fp16,[1,%d,1,%d]> be=tile(reps=reps,x=b)[name=string(\"bias_expanded\")];\n"
         "    tensor<fp16,[1,%d,1,%d]> sw=mul(x=q16,y=se)[name=string(\"scaled_weight\")];\n"
         "    tensor<fp16,[1,%d,1,%d]> wg=add(x=sw,y=be)[name=string(\"biased_weight\")];\n"
         "    tensor<int32,[4]> wr=const()[name=string(\"weight_shape\"),val=tensor<int32,[4]>([%d,%d,1,1])];\n"
         "    tensor<fp16,[%d,%d,1,1]> w=reshape(shape=wr,x=wg)[name=string(\"weight\")];\n"
         "    tensor<int32,[2]> strides=const()[name=string(\"strides\"),val=tensor<int32,[2]>([1,1])];\n"
         "    tensor<int32,[4]> pad=const()[name=string(\"pad\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[2]> dilations=const()[name=string(\"dilations\"),val=tensor<int32,[2]>([1,1])];\n"
         "    int32 groups=const()[name=string(\"groups\"),val=int32(1)];\n"
         "    string valid=const()[name=string(\"valid\"),val=string(\"valid\")];\n"
         "    tensor<fp16,[1,%d,1,%d]> y=conv(dilations=dilations,groups=groups,pad=pad,pad_type=valid,strides=strides,weight=w,x=x)[name=string(\"projection\")];\n"
         "  } -> (y);\n"
         "}\n",
         quantizedType, Width, Width, Width, GroupsPerRow,
         Width, GroupsPerRow, Width, Positions,
         Width, Width, quantizedCast, GroupSize,
         Width, Width,
         Width, Width,
         Width, Width,
         Width, Width,
         Width, Width, Width, Width,
         Width, Positions];
}

static float sourceInput(uint32_t column) {
    return 0.375f * sinf((float)(column + 1) * 0.17f) +
           0.125f * cosf((float)(column + 3) * 0.11f);
}

static uint8_t quantizedValue(uint32_t row, uint32_t column, uint32_t variant) {
    return (uint8_t)((row * (variant ? 5u : 3u) +
                      column * (variant ? 7u : 11u) + variant * 13u) & 15u);
}

static float scaleValue(uint32_t row, uint32_t variant) {
    return 0.0078125f * (float)(1u + ((row + 3u * variant) % 7u));
}

static float biasValue(uint32_t row, uint32_t variant) {
    return ((int)((row + variant) % 5u) - 2) * 0.00390625f;
}

static bool fillSurface(IOSurfaceRef surface, const void *bytes, size_t count) {
    if (!surface || !bytes || IOSurfaceGetAllocSize(surface) < count ||
        IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return false;
    memcpy(IOSurfaceGetBaseAddress(surface), bytes, count);
    IOSurfaceUnlock(surface, 0, NULL);
    return true;
}

static NSArray<NSNumber *> *symbolIndexArray(id rawIndices) {
    if ([rawIndices isKindOfClass:NSArray.class]) return rawIndices;
    if (![rawIndices isKindOfClass:NSIndexSet.class]) return nil;
    NSIndexSet *indices = rawIndices;
    NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:indices.count];
    for (NSUInteger value = indices.firstIndex; value != NSNotFound;
         value = [indices indexGreaterThanIndex:value])
        [values addObject:@(value)];
    return values;
}

static NSArray<NSDictionary *> *liveList(NSDictionary *attributes, NSString *key) {
    NSArray *statuses = attributes[@"NetworkStatusList"];
    NSDictionary *status = [statuses isKindOfClass:NSArray.class] ? statuses.firstObject : nil;
    id value = status[key];
    return [value isKindOfClass:NSArray.class] ? value : nil;
}

static NSArray<NSDictionary *> *layoutReceipt(NSArray<NSDictionary *> *records) {
    NSMutableArray<NSDictionary *> *receipt = [NSMutableArray arrayWithCapacity:records.count];
    for (NSDictionary *record in records) {
        NSString *name = record[@"Symbol"] ?: record[@"Name"] ?: @"unknown";
        uint64_t batchStride = [record[@"BatchStride"] unsignedLongLongValue];
        uint64_t batches = [record[@"Batches"] unsignedLongLongValue];
        [receipt addObject:@{
            @"name": name,
            @"type": record[@"Type"] ?: @"unknown",
            @"bytes": @(batchStride * batches),
            @"batch_stride": @(batchStride),
            @"plane_stride": @([record[@"PlaneStride"] unsignedLongLongValue]),
            @"row_stride": @([record[@"RowStride"] unsignedLongLongValue]),
        }];
    }
    return receipt;
}

static NSDictionary *runVariant(id model,
                                id request,
                                IOSurfaceRef quantized,
                                IOSurfaceRef scales,
                                IOSurfaceRef biases,
                                IOSurfaceRef activation,
                                IOSurfaceRef output,
                                uint32_t variant) {
    const size_t quantizedElementBytes = fp16QuantizedInput() ? sizeof(__fp16) : 1;
    const size_t quantizedRowBytes = roundUp64(Width * quantizedElementBytes);
    const size_t parameterRowBytes = roundUp64(GroupsPerRow * sizeof(__fp16));
    uint8_t q[Width * Width * sizeof(__fp16)] = {0};
    uint8_t s[Width * 64] = {0};
    uint8_t b[Width * 64] = {0};
    __fp16 x[Width * Positions] = {0};
    for (uint32_t row = 0; row < Width; ++row) {
        for (uint32_t column = 0; column < Width; ++column) {
            uint8_t *destination = q + (uint64_t)row * quantizedRowBytes +
                                   column * quantizedElementBytes;
            if (fp16QuantizedInput())
                *(__fp16 *)destination = (__fp16)quantizedValue(row, column, variant);
            else *destination = quantizedValue(row, column, variant);
        }
        *(__fp16 *)(s + (uint64_t)row * parameterRowBytes) = (__fp16)scaleValue(row, variant);
        *(__fp16 *)(b + (uint64_t)row * parameterRowBytes) = (__fp16)biasValue(row, variant);
    }
    for (uint32_t column = 0; column < Width; ++column)
        x[column * Positions] = (__fp16)sourceInput(column);
    uint64_t started = mach_absolute_time();
    size_t quantizedBytes = Width * quantizedRowBytes;
    size_t parameterBytes = Width * parameterRowBytes;
    bool prepared = fillSurface(quantized, q, quantizedBytes) &&
                    fillSurface(scales, s, parameterBytes) &&
                    fillSurface(biases, b, parameterBytes) &&
                    fillSurface(activation, x, sizeof(x));
    double preparationMs = milliseconds(mach_absolute_time() - started);
    if (!prepared) return @{ @"status": @"rejected", @"stage": @"surface_fill" };
    IOSurfaceLock(output, 0, NULL);
    memset(IOSurfaceGetBaseAddress(output), 0, IOSurfaceGetAllocSize(output));
    IOSurfaceUnlock(output, 0, NULL);
    NSError *error = nil;
    started = mach_absolute_time();
    BOOL accepted = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &error);
    double evaluationMs = milliseconds(mach_absolute_time() - started);
    double maximumError = 0.0;
    double squaredError = 0.0;
    uint64_t checksum = 1469598103934665603ull;
    uint32_t finite = 0;
    if (accepted && IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess) {
        const __fp16 *values = IOSurfaceGetBaseAddress(output);
        for (uint32_t row = 0; row < Width; ++row) {
            double reference = 0.0;
            float scale = (float)(__fp16)scaleValue(row, variant);
            float bias = (float)(__fp16)biasValue(row, variant);
            for (uint32_t column = 0; column < Width; ++column) {
                float input = (float)(__fp16)sourceInput(column);
                float weight = quantizedValue(row, column, variant) * scale + bias;
                reference += (double)input * weight;
            }
            float actual = values[(uint64_t)row * Positions];
            double difference = fabs((double)actual - reference);
            maximumError = fmax(maximumError, difference);
            squaredError += difference * difference;
            if (isfinite(actual)) ++finite;
            uint16_t bits = 0;
            memcpy(&bits, &values[(uint64_t)row * Positions], sizeof(bits));
            checksum ^= bits;
            checksum *= 1099511628211ull;
        }
        IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
    }
    bool validated = accepted && finite == Width && maximumError <= 0.02;
    return @{
        @"status": validated ? @"validated" : @"rejected",
        @"stage": accepted ? @"numerical_oracle" : @"evaluate",
        @"variant": @(variant),
        @"preparation_ms": @(preparationMs),
        @"evaluation_ms": @(evaluationMs),
        @"finite_outputs": @(finite),
        @"max_absolute_error": @(maximumError),
        @"rmse": @(sqrt(squaredError / Width)),
        @"checksum": [NSString stringWithFormat:@"0x%016llx", checksum],
        @"error": error.localizedDescription ?: @"none",
    };
}

int main(void) {
    @autoreleasepool {
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!framework || !descriptorClass || !modelClass || !requestClass || !surfaceClass) {
            emit(@{ @"schema": @"ane-fieldfare.compact-projection.v1",
                    @"status": @"rejected", @"stage": @"runtime_surface" });
            return 1;
        }
        NSString *milString = compactProjectionMIL();
        NSData *mil = [milString dataUsingEncoding:NSUTF8StringEncoding];
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
        id model = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        if (!model) {
            emit(@{ @"schema": @"ane-fieldfare.compact-projection.v1",
                    @"status": @"rejected", @"stage": @"model" });
            return 1;
        }
        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(model,
                                                                @selector(hexStringIdentifier));
        NSString *temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *files = NSFileManager.defaultManager;
        [files createDirectoryAtPath:[temporary stringByAppendingPathComponent:@"weights"]
         withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        NSError *error = nil;
        uint64_t started = mach_absolute_time();
        BOOL compiled = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        double compileMs = milliseconds(mach_absolute_time() - started);
        if (!compiled) {
            emit(@{ @"schema": @"ane-fieldfare.compact-projection.v1",
                    @"status": @"rejected", @"stage": @"compile",
                    @"detail": error.localizedDescription ?: @"unknown" });
            [files removeItemAtPath:temporary error:nil];
            return 1;
        }
        started = mach_absolute_time();
        BOOL loaded = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        double loadMs = milliseconds(mach_absolute_time() - started);
        if (!loaded) {
            emit(@{ @"schema": @"ane-fieldfare.compact-projection.v1",
                    @"status": @"rejected", @"stage": @"load",
                    @"detail": error.localizedDescription ?: @"unknown" });
            [files removeItemAtPath:temporary error:nil];
            return 1;
        }
        const uint64_t qBytes = Width * roundUp64(Width *
            (fp16QuantizedInput() ? sizeof(__fp16) : 1));
        const uint64_t scaleBytes = Width * roundUp64(GroupsPerRow * sizeof(__fp16));
        const uint64_t activationBytes = Width * Positions * sizeof(__fp16);
        IOSurfaceRef q = makeSurface(qBytes), s = makeSurface(scaleBytes);
        IOSurfaceRef b = makeSurface(scaleBytes), x = makeSurface(activationBytes);
        IOSurfaceRef y = makeSurface(activationBytes);
        id qObject = q ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), q) : nil;
        id sObject = s ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), s) : nil;
        id bObject = b ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), b) : nil;
        id xObject = x ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), x) : nil;
        id yObject = y ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), y) : nil;
        NSDictionary *modelAttributes = ((id (*)(id, SEL))objc_msgSend)(
            model, @selector(modelAttributes));
        NSArray<NSDictionary *> *inputRecords = liveList(modelAttributes, @"LiveInputList");
        NSArray<NSDictionary *> *outputRecords = liveList(modelAttributes, @"LiveOutputList");
        id backingModel = ((id (*)(id, SEL))objc_msgSend)(model, @selector(model));
        id rawInputIndices = backingModel ? ((id (*)(id, SEL, unsigned int))objc_msgSend)(
            backingModel, @selector(inputSymbolIndicesForProcedureIndex:), 0) : nil;
        id rawOutputIndices = backingModel ? ((id (*)(id, SEL, unsigned int))objc_msgSend)(
            backingModel, @selector(outputSymbolIndicesForProcedureIndex:), 0) : nil;
        NSArray *inputIndices = symbolIndexArray(rawInputIndices);
        NSArray *outputIndices = symbolIndexArray(rawOutputIndices);
        NSDictionary<NSString *, id> *objectsBySymbol = qObject && sObject && bObject && xObject ? @{
            @"q": qObject, @"s": sObject, @"b": bObject, @"x": xObject
        } : nil;
        NSMutableArray *orderedInputs = [NSMutableArray arrayWithCapacity:inputRecords.count];
        NSMutableArray<NSString *> *inputSymbols = [NSMutableArray arrayWithCapacity:inputRecords.count];
        for (NSDictionary *record in inputRecords) {
            NSString *symbol = record[@"Symbol"] ?: record[@"Name"];
            id object = objectsBySymbol[symbol];
            if (!symbol.length || !object) {
                [orderedInputs removeAllObjects];
                [inputSymbols removeAllObjects];
                break;
            }
            [orderedInputs addObject:object];
            [inputSymbols addObject:symbol];
        }
        BOOL compilerContractValid = inputIndices.count == 4 && outputIndices.count == 1 &&
            inputRecords.count == 4 && outputRecords.count == 1 && orderedInputs.count == 4;
        id request = compilerContractValid && yObject ?
            ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
                requestClass,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                /* Request order comes from the compiler's physical LiveInputList;
                 * MIL source-signature order is not a stable request ABI. */
                orderedInputs, inputIndices,
                @[yObject], outputIndices, nil, nil, @0) : nil;
        BOOL requestValid = request && ((BOOL (*)(id, SEL))objc_msgSend)(request, @selector(validate));
        uint64_t requestSurfaceCount = request ?
            ((uint64_t (*)(id, SEL))objc_msgSend)(request, @selector(ioSurfacesCount)) : 0;
        BOOL mapped = request && ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
            model, @selector(mapIOSurfacesWithRequest:cacheInference:error:), request, YES, &error);
        NSMutableArray *runs = [NSMutableArray array];
        if (request) {
            [runs addObject:runVariant(model, request, q, s, b, x, y, 0)];
            [runs addObject:runVariant(model, request, q, s, b, x, y, 1)];
            [runs addObject:runVariant(model, request, q, s, b, x, y, 0)];
        }
        bool numerical = request && runs.count == 3;
        for (NSDictionary *run in runs)
            numerical = numerical && [run[@"status"] isEqual:@"validated"];
        bool restored = numerical && [runs[0][@"checksum"] isEqual:runs[2][@"checksum"]];
        if (mapped) ((void (*)(id, SEL, id))objc_msgSend)(
            model, @selector(unmapIOSurfacesWithRequest:), request);
        ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        [files removeItemAtPath:temporary error:nil];
        if (q) CFRelease(q); if (s) CFRelease(s); if (b) CFRelease(b);
        if (x) CFRelease(x); if (y) CFRelease(y);
        emit(@{
            @"schema": @"ane-fieldfare.compact-projection.v1",
            @"status": numerical && restored ? @"validated" : @"rejected",
            @"stage": numerical ? @"numerical_oracle" : (request ? @"evaluate" : @"request"),
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"shape": @{ @"rows": @(Width), @"columns": @(Width),
                         @"group_size": @(GroupSize), @"positions": @(Positions) },
            @"compiler_input_symbols": inputSymbols,
            @"input_symbol_indices": inputIndices ?: @[],
            @"output_symbol_indices": outputIndices ?: @[],
            @"input_layouts": layoutReceipt(inputRecords ?: @[]),
            @"output_layouts": layoutReceipt(outputRecords ?: @[]),
            @"compiler_contract_valid": @(compilerContractValid),
            @"request_valid": @(requestValid),
            @"request_surface_count": @(requestSurfaceCount),
            @"explicit_surface_mapping": @(mapped),
            @"compile_count": @1,
            @"load_count": @1,
            @"compile_ms": @(compileMs),
            @"load_ms": @(loadMs),
            @"compact_runtime_input_bytes": @(qBytes + 2 * scaleBytes),
            @"quantized_input_dtype": fp16QuantizedInput() ? @"fp16" : @"uint8",
            @"fp16_weight_carrier_bytes": @(2 * Width * Width),
            @"runs": runs,
            @"a_b_a_checksum_restored": @(restored),
            @"error": numerical ? @"none" : (error.localizedDescription ?: @"unknown"),
            @"claim_boundary": @"Dynamic uint8 quantized weights plus FP16 per-group affine parameters are expanded inside one compiled and loaded ANE program and validated A/B/A against an independent CPU oracle. This is an unpacked 8-bit runtime-input construction, not packed uint4, compiler-prepared mutable weights, an authentic Gemma expert, or generation."
        });
        return numerical && restored ? 0 : 1;
    }
}
