#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../ANEFieldfareRuntime/RoutedLayer.h"
#include "../ANEFieldfareRuntime/TensorIndex.h"

enum {
    Hidden = 2816,
    Intermediate = 704,
    Positions = 64,
    GroupSize = 64,
    ExpertCount = 128,
    ExpertStride = 3358720,
    GateWeightOffset = 0,
    GateScaleOffset = 991232,
    GateBiasOffset = 1053184,
    UpWeightOffset = 1115136,
    UpScaleOffset = 2106368,
    UpBiasOffset = 2168320,
    DownWeightOffset = 2230272,
    DownScaleOffset = 3221504,
    DownBiasOffset = 3283456,
};

typedef struct {
    __strong NSString *name;
    uint64_t bytes;
    uint64_t batchStride;
    uint64_t batches;
    uint64_t planeStride;
    uint64_t rowStride;
    uint64_t planes;
    IOSurfaceRef surface;
    __strong id object;
} SurfaceLayout;

typedef struct {
    const uint8_t *weights;
    const uint8_t *scales;
    const uint8_t *biases;
    uint32_t rows;
    uint32_t columns;
} PackedMatrix;

typedef struct {
    PackedMatrix gate;
    PackedMatrix up;
    PackedMatrix down;
    SurfaceLayout *qgu;
    SurfaceLayout *sgu;
    SurfaceLayout *bgu;
    SurfaceLayout *qd;
    SurfaceLayout *sd;
    SurfaceLayout *bd;
} FillContext;

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
    if (!bytes || bytes > NSUIntegerMax) return NULL;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    });
}

static NSString *expertMIL(void) {
    return @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "  func main<ios18>(tensor<uint8,[1,1408,1,2816]> qgu, tensor<fp16,[1,1408,1,44]> sgu, tensor<fp16,[1,1408,1,44]> bgu, tensor<uint8,[1,2816,1,704]> qd, tensor<fp16,[1,2816,1,11]> sd, tensor<fp16,[1,2816,1,11]> bd, tensor<fp16,[1,2816,1,64]> x) {\n"
        "    string f16=const()[name=string(\"f16\"),val=string(\"fp16\")];\n"
        "    tensor<fp16,[1,1408,1,2816]> qgu16=cast(dtype=f16,x=qgu)[name=string(\"qgu16\")];\n"
        "    tensor<int32,[4]> qgshape=const()[name=string(\"qgshape\"),val=tensor<int32,[4]>([1,1408,44,64])];\n"
        "    tensor<int32,[4]> pgshape=const()[name=string(\"pgshape\"),val=tensor<int32,[4]>([1,1408,44,1])];\n"
        "    tensor<fp16,[1,1408,44,64]> qgur=reshape(shape=qgshape,x=qgu16)[name=string(\"qgur\")];\n"
        "    tensor<fp16,[1,1408,44,1]> sgur=reshape(shape=pgshape,x=sgu)[name=string(\"sgur\")];\n"
        "    tensor<fp16,[1,1408,44,1]> bgur=reshape(shape=pgshape,x=bgu)[name=string(\"bgur\")];\n"
        "    tensor<fp16,[1,1408,44,64]> wgus=mul(x=qgur,y=sgur)[name=string(\"wgus\")];\n"
        "    tensor<fp16,[1,1408,44,64]> wgu4=add(x=wgus,y=bgur)[name=string(\"wgu4\")];\n"
        "    tensor<int32,[4]> sg=const()[name=string(\"sg\"),val=tensor<int32,[4]>([1408,2816,1,1])];\n"
        "    tensor<fp16,[1408,2816,1,1]> wgu=reshape(shape=sg,x=wgu4)[name=string(\"wgu\")];\n"
        "    tensor<int32,[2]> strides=const()[name=string(\"strides\"),val=tensor<int32,[2]>([1,1])];\n"
        "    tensor<int32,[4]> pad=const()[name=string(\"pad\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
        "    tensor<int32,[2]> dilations=const()[name=string(\"dilations\"),val=tensor<int32,[2]>([1,1])];\n"
        "    int32 groups=const()[name=string(\"groups\"),val=int32(1)];\n"
        "    string valid=const()[name=string(\"valid\"),val=string(\"valid\")];\n"
        "    tensor<fp16,[1,1408,1,64]> gu=conv(dilations=dilations,groups=groups,pad=pad,pad_type=valid,strides=strides,weight=wgu,x=x)[name=string(\"gu\")];\n"
        "    tensor<int32,[4]> gb=const()[name=string(\"gb\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
        "    tensor<int32,[4]> ub=const()[name=string(\"ub\"),val=tensor<int32,[4]>([0,704,0,0])];\n"
        "    tensor<int32,[4]> gs=const()[name=string(\"gs\"),val=tensor<int32,[4]>([1,704,1,64])];\n"
        "    tensor<fp16,[1,704,1,64]> gate=slice_by_size(x=gu,begin=gb,size=gs)[name=string(\"gate\")];\n"
        "    tensor<fp16,[1,704,1,64]> up=slice_by_size(x=gu,begin=ub,size=gs)[name=string(\"up\")];\n"
        "    tensor<fp16,[1,704,1,64]> gate2=mul(x=gate,y=gate)[name=string(\"gate2\")];\n"
        "    tensor<fp16,[1,704,1,64]> gate3=mul(x=gate2,y=gate)[name=string(\"gate3\")];\n"
        "    fp16 ck=const()[name=string(\"ck\"),val=fp16(0.044715)];\n"
        "    tensor<fp16,[1,704,1,64]> cubic=mul(x=gate3,y=ck)[name=string(\"cubic\")];\n"
        "    tensor<fp16,[1,704,1,64]> inner=add(x=gate,y=cubic)[name=string(\"inner\")];\n"
        "    fp16 tk=const()[name=string(\"tk\"),val=fp16(0.7978845608028654)];\n"
        "    tensor<fp16,[1,704,1,64]> scaled=mul(x=inner,y=tk)[name=string(\"scaled\")];\n"
        "    tensor<fp16,[1,704,1,64]> th=tanh(x=scaled)[name=string(\"th\")];\n"
        "    fp16 one=const()[name=string(\"one\"),val=fp16(1.0)];\n"
        "    fp16 half=const()[name=string(\"half\"),val=fp16(0.5)];\n"
        "    tensor<fp16,[1,704,1,64]> plus=add(x=th,y=one)[name=string(\"plus\")];\n"
        "    tensor<fp16,[1,704,1,64]> hg=mul(x=gate,y=half)[name=string(\"hg\")];\n"
        "    tensor<fp16,[1,704,1,64]> gelu=mul(x=hg,y=plus)[name=string(\"gelu\")];\n"
        "    tensor<fp16,[1,704,1,64]> act=mul(x=gelu,y=up)[name=string(\"act\")];\n"
        "    tensor<fp16,[1,2816,1,704]> qd16=cast(dtype=f16,x=qd)[name=string(\"qd16\")];\n"
        "    tensor<int32,[4]> qdshape=const()[name=string(\"qdshape\"),val=tensor<int32,[4]>([1,2816,11,64])];\n"
        "    tensor<int32,[4]> pdshape=const()[name=string(\"pdshape\"),val=tensor<int32,[4]>([1,2816,11,1])];\n"
        "    tensor<fp16,[1,2816,11,64]> qdr=reshape(shape=qdshape,x=qd16)[name=string(\"qdr\")];\n"
        "    tensor<fp16,[1,2816,11,1]> sdr=reshape(shape=pdshape,x=sd)[name=string(\"sdr\")];\n"
        "    tensor<fp16,[1,2816,11,1]> bdr=reshape(shape=pdshape,x=bd)[name=string(\"bdr\")];\n"
        "    tensor<fp16,[1,2816,11,64]> wds=mul(x=qdr,y=sdr)[name=string(\"wds\")];\n"
        "    tensor<fp16,[1,2816,11,64]> wd4=add(x=wds,y=bdr)[name=string(\"wd4\")];\n"
        "    tensor<int32,[4]> sdown=const()[name=string(\"sdown\"),val=tensor<int32,[4]>([2816,704,1,1])];\n"
        "    tensor<fp16,[2816,704,1,1]> wd=reshape(shape=sdown,x=wd4)[name=string(\"wd\")];\n"
        "    tensor<fp16,[1,2816,1,64]> out16=conv(dilations=dilations,groups=groups,pad=pad,pad_type=valid,strides=strides,weight=wd,x=act)[name=string(\"down\")];\n"
        "    string f32=const()[name=string(\"f32\"),val=string(\"fp32\")];\n"
        "    tensor<fp32,[1,2816,1,64]> out=cast(dtype=f32,x=out16)[name=string(\"out\")];\n"
        "  } -> (out);\n"
        "}\n";
}

static NSArray<NSNumber *> *symbolIndexArray(id rawIndices) {
    if ([rawIndices isKindOfClass:NSArray.class]) return rawIndices;
    if (![rawIndices isKindOfClass:NSIndexSet.class]) return nil;
    NSIndexSet *indices = rawIndices;
    NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:indices.count];
    for (NSUInteger value = indices.firstIndex; value != NSNotFound;
         value = [indices indexGreaterThanIndex:value]) [values addObject:@(value)];
    return values;
}

static NSArray<NSDictionary *> *liveList(NSDictionary *attributes, NSString *key) {
    NSArray *statuses = attributes[@"NetworkStatusList"];
    NSDictionary *status = [statuses isKindOfClass:NSArray.class] ? statuses.firstObject : nil;
    id value = status[key];
    return [value isKindOfClass:NSArray.class] ? value : nil;
}

static bool initializeLayout(SurfaceLayout *layout,
                             NSDictionary *record,
                             Class surfaceClass) {
    if (!layout || !record || !surfaceClass) return false;
    NSString *name = record[@"Symbol"] ?: record[@"Name"];
    uint64_t batchStride = [record[@"BatchStride"] unsignedLongLongValue];
    uint64_t batches = [record[@"Batches"] unsignedLongLongValue];
    uint64_t bytes = batchStride * batches;
    IOSurfaceRef surface = makeSurface(bytes);
    id object = surface ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        surfaceClass, @selector(objectWithIOSurface:), surface) : nil;
    if (!name.length || !bytes || !surface || !object) {
        if (surface) CFRelease(surface);
        return false;
    }
    *layout = (SurfaceLayout){
        .name = name,
        .bytes = bytes,
        .batchStride = batchStride,
        .batches = batches,
        .planeStride = [record[@"PlaneStride"] unsignedLongLongValue],
        .rowStride = [record[@"RowStride"] unsignedLongLongValue],
        .planes = [record[@"PlaneCount"] unsignedLongLongValue],
        .surface = surface,
        .object = object,
    };
    return true;
}

static SurfaceLayout *findLayout(SurfaceLayout *layouts, uint32_t count, NSString *name) {
    for (uint32_t index = 0; index < count; ++index)
        if ([layouts[index].name isEqualToString:name]) return &layouts[index];
    return NULL;
}

static PackedMatrix packedMatrix(const uint8_t *base,
                                 uint32_t rows,
                                 uint32_t columns,
                                 uint32_t weightOffset,
                                 uint32_t scaleOffset,
                                 uint32_t biasOffset) {
    return (PackedMatrix){base + weightOffset, base + scaleOffset, base + biasOffset,
                          rows, columns};
}

static void fillMatrixRow(PackedMatrix matrix,
                          uint32_t row,
                          SurfaceLayout *quantized,
                          SurfaceLayout *scales,
                          SurfaceLayout *biases,
                          uint32_t destinationRow) {
    uint8_t *qBase = IOSurfaceGetBaseAddress(quantized->surface);
    uint8_t *sBase = IOSurfaceGetBaseAddress(scales->surface);
    uint8_t *bBase = IOSurfaceGetBaseAddress(biases->surface);
    uint8_t *q = qBase + (uint64_t)destinationRow * quantized->planeStride;
    __fp16 *s = (__fp16 *)(sBase + (uint64_t)destinationRow * scales->planeStride);
    __fp16 *b = (__fp16 *)(bBase + (uint64_t)destinationRow * biases->planeStride);
    uint64_t rowLinear = (uint64_t)row * matrix.columns;
    const uint8_t *packed = matrix.weights + rowLinear / 2;
    for (uint32_t byte = 0; byte < matrix.columns / 2; ++byte) {
        uint8_t value = packed[byte];
        q[2 * byte] = value & 0x0f;
        q[2 * byte + 1] = value >> 4;
    }
    uint32_t groups = matrix.columns / GroupSize;
    uint64_t firstGroup = rowLinear / GroupSize;
    for (uint32_t group = 0; group < groups; ++group) {
        s[group] = (__fp16)ANEFieldfareBF16(matrix.scales + 2 * (firstGroup + group));
        b[group] = (__fp16)ANEFieldfareBF16(matrix.biases + 2 * (firstGroup + group));
    }
}

static void fillRowTask(void *rawContext, size_t task) {
    FillContext *context = rawContext;
    if (task < 2 * Intermediate) {
        bool isUp = task >= Intermediate;
        uint32_t row = (uint32_t)(isUp ? task - Intermediate : task);
        fillMatrixRow(isUp ? context->up : context->gate, row,
                      context->qgu, context->sgu, context->bgu, (uint32_t)task);
    } else {
        uint32_t row = (uint32_t)(task - 2 * Intermediate);
        fillMatrixRow(context->down, row,
                      context->qd, context->sd, context->bd, row);
    }
}

static bool fillExpert(const uint8_t *expertBase,
                       const float input[Hidden],
                       SurfaceLayout *layouts,
                       uint32_t layoutCount,
                       double *elapsed) {
    SurfaceLayout *qgu = findLayout(layouts, layoutCount, @"qgu");
    SurfaceLayout *sgu = findLayout(layouts, layoutCount, @"sgu");
    SurfaceLayout *bgu = findLayout(layouts, layoutCount, @"bgu");
    SurfaceLayout *qd = findLayout(layouts, layoutCount, @"qd");
    SurfaceLayout *sd = findLayout(layouts, layoutCount, @"sd");
    SurfaceLayout *bd = findLayout(layouts, layoutCount, @"bd");
    SurfaceLayout *x = findLayout(layouts, layoutCount, @"x");
    if (!qgu || !sgu || !bgu || !qd || !sd || !bd || !x) return false;
    SurfaceLayout *all[] = {qgu, sgu, bgu, qd, sd, bd, x};
    uint32_t locked = 0;
    for (; locked < sizeof(all) / sizeof(all[0]); ++locked)
        if (IOSurfaceLock(all[locked]->surface, 0, NULL) != kIOReturnSuccess) break;
    if (locked != sizeof(all) / sizeof(all[0])) {
        while (locked) IOSurfaceUnlock(all[--locked]->surface, 0, NULL);
        return false;
    }
    uint64_t started = mach_absolute_time();
    memset(IOSurfaceGetBaseAddress(x->surface), 0, (size_t)x->bytes);
    uint8_t *xBase = IOSurfaceGetBaseAddress(x->surface);
    for (uint32_t column = 0; column < Hidden; ++column)
        *(__fp16 *)(xBase + (uint64_t)column * x->planeStride) = (__fp16)input[column];
    FillContext context = {
        .gate = packedMatrix(expertBase, Intermediate, Hidden,
                             GateWeightOffset, GateScaleOffset, GateBiasOffset),
        .up = packedMatrix(expertBase, Intermediate, Hidden,
                           UpWeightOffset, UpScaleOffset, UpBiasOffset),
        .down = packedMatrix(expertBase, Hidden, Intermediate,
                             DownWeightOffset, DownScaleOffset, DownBiasOffset),
        .qgu = qgu, .sgu = sgu, .bgu = bgu, .qd = qd, .sd = sd, .bd = bd,
    };
    dispatch_apply_f(2 * Intermediate + Hidden,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &context, fillRowTask);
    if (elapsed) *elapsed = milliseconds(mach_absolute_time() - started);
    for (uint32_t index = 0; index < sizeof(all) / sizeof(all[0]); ++index)
        IOSurfaceUnlock(all[index]->surface, 0, NULL);
    return true;
}

static bool rawBF16(const ANEFieldfareTensor *tensor, uint32_t count, float *output) {
    if (!tensor || !output || tensor->weightBytes != (uint64_t)count * 2 ||
        tensor->scaleBytes || tensor->biasBytes) return false;
    for (uint32_t index = 0; index < count; ++index)
        output[index] = ANEFieldfareBF16(tensor->weights + (uint64_t)index * 2);
    return true;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        mach_timebase_info(&timebase);
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *weightsPath = [root stringByAppendingPathComponent:@"model_weights.bin"];
        NSString *packedPath = [root stringByAppendingPathComponent:@"packed_experts/layer_00.bin"];
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor embedding = {0}, norm = {0};
        ANEFieldfareRoutedLayer *layer = NULL;
        float *residual = calloc(Hidden, sizeof(float));
        float *normScale = calloc(Hidden, sizeof(float));
        float *input = calloc(Hidden, sizeof(float));
        float *expected = calloc(Hidden, sizeof(float));
        float *actual = calloc(Hidden, sizeof(float));
        int packedDescriptor = open(packedPath.fileSystemRepresentation, O_RDONLY | O_CLOEXEC);
        struct stat packedStatus = {0};
        bool valid = residual && normScale && input && expected && actual &&
            packedDescriptor >= 0 && fstat(packedDescriptor, &packedStatus) == 0 &&
            packedStatus.st_size == (off_t)((uint64_t)ExpertCount * ExpertStride) &&
            ANEFieldfareTensorIndexOpen(weightsPath.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.embed_tokens.weight", &embedding) &&
            ANEFieldfareTensorIndexFind(&index,
                "language_model.model.layers.0.pre_feedforward_layernorm_2.weight", &norm) &&
            ANEFieldfareAffineRow(&embedding, 4, GroupSize, 2, residual) &&
            rawBF16(&norm, Hidden, normScale) &&
            ANEFieldfareRMSNorm(residual, normScale, Hidden, 1e-6f, input) &&
            ANEFieldfareRoutedLayerOpen(root.fileSystemRepresentation, 0, &layer);
        const uint8_t *packedBytes = valid ? mmap(NULL, (size_t)packedStatus.st_size,
            PROT_READ, MAP_PRIVATE, packedDescriptor, 0) : NULL;
        valid = valid && packedBytes && packedBytes != MAP_FAILED;
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        valid = valid && framework && descriptorClass && modelClass && requestClass && surfaceClass;
        NSString *milString = expertMIL();
        NSData *mil = [milString dataUsingEncoding:NSUTF8StringEncoding];
        NSString *dumpPath = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_DUMP_MIL"];
        if (dumpPath.length) [mil writeToFile:dumpPath atomically:YES];
        id descriptor = valid ? ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil) : nil;
        id model = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        valid = valid && model;
        NSString *identifier = valid ? ((id (*)(id, SEL))objc_msgSend)(
            model, @selector(hexStringIdentifier)) : nil;
        NSString *temporary = identifier ? [NSTemporaryDirectory() stringByAppendingPathComponent:identifier] : nil;
        NSFileManager *files = NSFileManager.defaultManager;
        if (temporary) {
            [files createDirectoryAtPath:[temporary stringByAppendingPathComponent:@"weights"]
             withIntermediateDirectories:YES attributes:nil error:nil];
            [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        }
        NSError *error = nil;
        uint64_t started = mach_absolute_time();
        BOOL compiled = valid && ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        double compileMs = milliseconds(mach_absolute_time() - started);
        started = mach_absolute_time();
        BOOL loaded = compiled && ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        double loadMs = milliseconds(mach_absolute_time() - started);
        NSDictionary *attributes = loaded ? ((id (*)(id, SEL))objc_msgSend)(
            model, @selector(modelAttributes)) : nil;
        NSArray<NSDictionary *> *inputRecords = liveList(attributes, @"LiveInputList");
        NSArray<NSDictionary *> *outputRecords = liveList(attributes, @"LiveOutputList");
        SurfaceLayout inputLayouts[7] = {0};
        SurfaceLayout outputLayout = {0};
        valid = loaded && inputRecords.count == 7 && outputRecords.count == 1;
        for (uint32_t i = 0; valid && i < 7; ++i)
            valid = initializeLayout(&inputLayouts[i], inputRecords[i], surfaceClass);
        valid = valid && initializeLayout(&outputLayout, outputRecords[0], surfaceClass);
        NSMutableArray *inputObjects = [NSMutableArray arrayWithCapacity:7];
        for (uint32_t i = 0; valid && i < 7; ++i) [inputObjects addObject:inputLayouts[i].object];
        id backingModel = loaded ? ((id (*)(id, SEL))objc_msgSend)(model, @selector(model)) : nil;
        NSArray *inputIndices = symbolIndexArray(backingModel ?
            ((id (*)(id, SEL, unsigned int))objc_msgSend)(
                backingModel, @selector(inputSymbolIndicesForProcedureIndex:), 0) : nil);
        NSArray *outputIndices = symbolIndexArray(backingModel ?
            ((id (*)(id, SEL, unsigned int))objc_msgSend)(
                backingModel, @selector(outputSymbolIndicesForProcedureIndex:), 0) : nil);
        valid = valid && inputIndices.count == 7 && outputIndices.count == 1;
        id request = valid ? ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            inputObjects, inputIndices, @[outputLayout.object], outputIndices, nil, nil, @0) : nil;
        BOOL mapped = request && ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
            model, @selector(mapIOSurfacesWithRequest:cacheInference:error:), request, YES, &error);
        NSMutableArray *runs = [NSMutableArray array];
        uint32_t sequence[] = {0, 1, 0};
        double maximumError = 0.0;
        for (uint32_t run = 0; mapped && run < 3; ++run) {
            uint32_t expert = sequence[run];
            double preparationMs = 0.0;
            bool runValid = ANEFieldfareRoutedLayerCPUExpert(layer, expert, input, expected) &&
                fillExpert(packedBytes + (uint64_t)expert * ExpertStride, input,
                           inputLayouts, 7, &preparationMs);
            started = mach_absolute_time();
            BOOL accepted = runValid && ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &error);
            double evaluationMs = milliseconds(mach_absolute_time() - started);
            double runMaximum = 0.0, squared = 0.0, checksum = 0.0;
            uint32_t finite = 0;
            if (accepted && IOSurfaceLock(outputLayout.surface, kIOSurfaceLockReadOnly, NULL) ==
                              kIOReturnSuccess) {
                const uint8_t *bytes = IOSurfaceGetBaseAddress(outputLayout.surface);
                for (uint32_t row = 0; row < Hidden; ++row) {
                    actual[row] = *(const float *)(bytes + (uint64_t)row * outputLayout.planeStride);
                    double difference = fabs((double)actual[row] - expected[row]);
                    runMaximum = fmax(runMaximum, difference);
                    squared += difference * difference;
                    checksum += actual[row] * (row + 1.0);
                    if (isfinite(actual[row])) ++finite;
                    runValid = runValid && difference <= 0.05 + 0.03 * fabs(expected[row]);
                }
                IOSurfaceUnlock(outputLayout.surface, kIOSurfaceLockReadOnly, NULL);
            } else runValid = false;
            maximumError = fmax(maximumError, runMaximum);
            [runs addObject:@{
                @"expert": @(expert),
                @"status": runValid ? @"validated" : @"rejected",
                @"preparation_ms": @(preparationMs),
                @"evaluation_ms": @(evaluationMs),
                @"finite_outputs": @(finite),
                @"max_absolute_error": @(runMaximum),
                @"rmse": @(sqrt(squared / Hidden)),
                @"checksum": @(checksum),
                @"error": error.localizedDescription ?: @"none",
            }];
            valid = valid && runValid;
        }
        bool restored = valid && runs.count == 3 && [runs[0][@"checksum"] isEqual:runs[2][@"checksum"]];
        uint64_t dynamicInputBytes = 0;
        NSMutableArray *layoutReceipt = [NSMutableArray array];
        for (uint32_t i = 0; i < 7; ++i) {
            dynamicInputBytes += inputLayouts[i].bytes;
            if (inputLayouts[i].name) [layoutReceipt addObject:@{
                @"name": inputLayouts[i].name,
                @"bytes": @(inputLayouts[i].bytes),
                @"plane_stride": @(inputLayouts[i].planeStride),
                @"row_stride": @(inputLayouts[i].rowStride),
            }];
        }
        if (mapped) ((void (*)(id, SEL, id))objc_msgSend)(
            model, @selector(unmapIOSurfacesWithRequest:), request);
        if (loaded) ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        if (temporary) [files removeItemAtPath:temporary error:nil];
        for (uint32_t i = 0; i < 7; ++i)
            if (inputLayouts[i].surface) CFRelease(inputLayouts[i].surface);
        if (outputLayout.surface) CFRelease(outputLayout.surface);
        emit(@{
            @"schema": @"ane-fieldfare.compact-expert.v1",
            @"status": valid && restored ? @"validated" : @"rejected",
            @"stage": mapped ? @"numerical_oracle" :
                (loaded ? @"surface_mapping" : (compiled ? @"load" : @"compile")),
            @"host": NSProcessInfo.processInfo.hostName ?: @"unknown",
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString ?: @"unknown",
            @"shape": @{ @"hidden": @(Hidden), @"intermediate": @(Intermediate),
                         @"positions": @(Positions), @"group_size": @(GroupSize) },
            @"compile_count": compiled ? @1 : @0,
            @"load_count": loaded ? @1 : @0,
            @"compile_ms": @(compileMs),
            @"load_ms": @(loadMs),
            @"input_layouts": layoutReceipt,
            @"dynamic_input_bytes_per_bank": @(dynamicInputBytes),
            @"prior_fp16_carrier_bytes_per_bank": @((uint64_t)Hidden *
                                                     (Positions + 3 * Intermediate) * 2),
            @"output_bytes_per_bank": @(outputLayout.bytes),
            @"runs": runs,
            @"max_absolute_error": @(maximumError),
            @"a_b_a_checksum_restored": @(restored),
            @"error": error.localizedDescription ?: @"none",
            @"claim_boundary": @"An authentic compact checkpoint expert executes gate/up, GELU product, and down through dynamic uint8 codes plus row-padded FP16 per-group affine parameters. Dequantization occurs inside one compiled and loaded ANE graph and A/B/A passes an independent compact CPU oracle. This is unpacked uint8 runtime-input execution, not packed uint4, compiler-prepared mutable weights, routed top-8, a full layer, or generation."
        });
        if (packedBytes && packedBytes != MAP_FAILED)
            munmap((void *)packedBytes, (size_t)packedStatus.st_size);
        if (packedDescriptor >= 0) close(packedDescriptor);
        ANEFieldfareRoutedLayerClose(layer);
        ANEFieldfareTensorIndexClose(&index);
        free(residual); free(normScale); free(input); free(expected); free(actual);
        return valid && restored ? 0 : 1;
    }
}
