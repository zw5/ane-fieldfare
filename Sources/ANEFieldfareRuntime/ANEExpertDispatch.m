#import "ANEExpertDispatch.h"

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <mach/mach_time.h>

#include <fcntl.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    Hidden = ANEFieldfareGemma4HiddenSize,
    Intermediate = ANEFieldfareGemma4ExpertIntermediateSize,
    ExpertCount = ANEFieldfareGemma4ExpertCount,
    Positions = 64,
    QuantGroupSize = 64,
    ExpertStride = 3358720,
    CarrierStride = Positions + 3 * Intermediate,
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

_Static_assert(Hidden % QuantGroupSize == 0, "hidden width must align to quant groups");
_Static_assert(Intermediate % QuantGroupSize == 0,
               "expert intermediate width must align to quant groups");

typedef struct {
    const uint8_t *weights;
    const uint8_t *scales;
    const uint8_t *biases;
    uint32_t rows;
    uint32_t columns;
} PackedMatrix;

struct ANEFieldfareANEExpertDispatcher {
    ANEFieldfareANEProgram *program;
    int descriptor;
    int prefetchDescriptor;
    uint64_t prefetchRequests, prefetchHints, prefetchBytes, prefetchFailures;
    double prefetchWaitMilliseconds;
    const uint8_t *packedBytes;
    uint64_t packedByteCount;
    uint64_t dispatchCount;
    double totalPreparationMilliseconds;
    double totalEvaluationMilliseconds;
    uint32_t lastBank;
    char *lastError;
};

static mach_timebase_info_data_t timebase;

static double milliseconds(uint64_t ticks) {
    return (double)ticks * timebase.numer / timebase.denom / 1e6;
}

static NSString *expertMIL(void) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "  func main<ios18>(tensor<fp16,[1,%d,1,%d]> packed) {\n"
         "    tensor<fp16,[1,2816,1,2176]> p=identity(x=packed)[name=string(\"carrier\")];\n"
         "    tensor<int32,[4]> pm=const()[name=string(\"pm\"),val=tensor<int32,[4]>([0,1,3,2])];\n"
         "    bool no=const()[name=string(\"no\"),val=bool(false)];\n"
         "    tensor<int32,[4]> xb=const()[name=string(\"xb\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[4]> xs=const()[name=string(\"xs\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
         "    tensor<fp16,[1,%d,1,%d]> x=slice_by_size(x=p,begin=xb,size=xs)[name=string(\"x\")];\n"
         "    tensor<int32,[4]> xr=const()[name=string(\"xr\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n"
         "    tensor<fp16,[1,1,%d,%d]> x2=reshape(shape=xr,x=x)[name=string(\"x2\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> xt=transpose(perm=pm,x=x2)[name=string(\"xt\")];\n"
         "    tensor<int32,[4]> ws=const()[name=string(\"ws\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
         "    tensor<int32,[4]> gb=const()[name=string(\"gb\"),val=tensor<int32,[4]>([0,0,0,%d])];\n"
         "    tensor<int32,[4]> ub=const()[name=string(\"ub\"),val=tensor<int32,[4]>([0,0,0,%d])];\n"
         "    tensor<int32,[4]> db=const()[name=string(\"db\"),val=tensor<int32,[4]>([0,0,0,%d])];\n"
         "    tensor<fp16,[1,%d,1,%d]> gw=slice_by_size(x=p,begin=gb,size=ws)[name=string(\"gw\")];\n"
         "    tensor<fp16,[1,%d,1,%d]> uw=slice_by_size(x=p,begin=ub,size=ws)[name=string(\"uw\")];\n"
         "    tensor<fp16,[1,%d,1,%d]> dw=slice_by_size(x=p,begin=db,size=ws)[name=string(\"dw\")];\n"
         "    tensor<int32,[4]> wr=const()[name=string(\"wr\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n"
         "    tensor<fp16,[1,1,%d,%d]> g2=reshape(shape=wr,x=gw)[name=string(\"g2\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> u2=reshape(shape=wr,x=uw)[name=string(\"u2\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> gate=matmul(transpose_x=no,transpose_y=no,x=xt,y=g2)[name=string(\"gate\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> up=matmul(transpose_x=no,transpose_y=no,x=xt,y=u2)[name=string(\"up\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> gate2=mul(x=gate,y=gate)[name=string(\"gate2\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> gate3=mul(x=gate2,y=gate)[name=string(\"gate3\")];\n"
         "    fp16 ck=const()[name=string(\"ck\"),val=fp16(0.044715)];\n"
         "    tensor<fp16,[1,1,%d,%d]> cubic=mul(x=gate3,y=ck)[name=string(\"cubic\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> inner=add(x=gate,y=cubic)[name=string(\"inner\")];\n"
         "    fp16 tk=const()[name=string(\"tk\"),val=fp16(0.7978845608028654)];\n"
         "    tensor<fp16,[1,1,%d,%d]> scaled=mul(x=inner,y=tk)[name=string(\"scaled\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> th=tanh(x=scaled)[name=string(\"th\")];\n"
         "    fp16 one=const()[name=string(\"one\"),val=fp16(1.0)];\n"
         "    fp16 half=const()[name=string(\"half\"),val=fp16(0.5)];\n"
         "    tensor<fp16,[1,1,%d,%d]> plus=add(x=th,y=one)[name=string(\"plus\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> hg=mul(x=gate,y=half)[name=string(\"hg\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> gelu=mul(x=hg,y=plus)[name=string(\"gelu\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> act=mul(x=gelu,y=up)[name=string(\"act\")];\n"
         "    tensor<int32,[4]> dr=const()[name=string(\"dr\"),val=tensor<int32,[4]>([1,1,%d,%d])];\n"
         "    tensor<fp16,[1,1,%d,%d]> d2=reshape(shape=dr,x=dw)[name=string(\"d2\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> dt=transpose(perm=pm,x=d2)[name=string(\"dt\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> o2=matmul(transpose_x=no,transpose_y=no,x=act,y=dt)[name=string(\"down\")];\n"
         "    tensor<fp16,[1,1,%d,%d]> ot=transpose(perm=pm,x=o2)[name=string(\"ot\")];\n"
         "    tensor<int32,[4]> os=const()[name=string(\"os\"),val=tensor<int32,[4]>([1,%d,1,%d])];\n"
         "    tensor<fp16,[1,%d,1,%d]> or=reshape(shape=os,x=ot)[name=string(\"or\")];\n"
         "    string f32=const()[name=string(\"f32\"),val=string(\"fp32\")];\n"
         "    tensor<fp32,[1,%d,1,%d]> out=cast(dtype=f32,x=or)[name=string(\"cast_out\")];\n"
         "  } -> (out);\n"
         "}\n",
         Hidden, CarrierStride,
         Hidden, Positions, Hidden, Positions,
         Hidden, Positions, Hidden, Positions, Positions, Hidden,
         Hidden, Intermediate,
         Positions, Positions + Intermediate, Positions + 2 * Intermediate,
         Hidden, Intermediate, Hidden, Intermediate, Hidden, Intermediate,
         Hidden, Intermediate, Hidden, Intermediate, Hidden, Intermediate,
         Positions, Intermediate, Positions, Intermediate,
         Positions, Intermediate, Positions, Intermediate,
         Positions, Intermediate, Positions, Intermediate,
         Positions, Intermediate, Positions, Intermediate,
         Positions, Intermediate, Positions, Intermediate,
         Positions, Intermediate, Positions, Intermediate,
         Hidden, Intermediate, Hidden, Intermediate, Intermediate, Hidden,
         Positions, Hidden, Hidden, Positions,
         Hidden, Positions, Hidden, Positions, Hidden, Positions];
}

static PackedMatrix matrix(const uint8_t *base, uint32_t rows, uint32_t columns,
                           uint32_t weightOffset, uint32_t scaleOffset, uint32_t biasOffset) {
    return (PackedMatrix){base + weightOffset, base + scaleOffset, base + biasOffset, rows, columns};
}

static void expandGroup(const uint8_t *weights,
                        float scale,
                        float bias,
                        __fp16 output[QuantGroupSize]) {
    for (uint32_t byte = 0; byte < QuantGroupSize / 2; ++byte) {
        uint8_t packed = weights[byte];
        output[2 * byte] = (__fp16)((packed & 0x0f) * scale + bias);
        output[2 * byte + 1] = (__fp16)((packed >> 4) * scale + bias);
    }
}

static void expandRow(PackedMatrix value, uint32_t row, __fp16 *output) {
    uint64_t rowStart = (uint64_t)row * value.columns;
    uint64_t firstGroup = rowStart / QuantGroupSize;
    uint32_t groups = value.columns / QuantGroupSize;
    const uint8_t *weights = value.weights + rowStart / 2;
    for (uint32_t group = 0; group < groups; ++group) {
        float scale = ANEFieldfareBF16(value.scales + (firstGroup + group) * 2);
        float bias = ANEFieldfareBF16(value.biases + (firstGroup + group) * 2);
        expandGroup(weights + (uint64_t)group * QuantGroupSize / 2,
                    scale, bias, output + (uint64_t)group * QuantGroupSize);
    }
}

enum {
    PairRowsPerBlock = 16,
    DownRowsPerBlock = 32,
    PairBlockCount = (Intermediate + PairRowsPerBlock - 1) / PairRowsPerBlock,
    DownBlockCount = (Hidden + DownRowsPerBlock - 1) / DownRowsPerBlock,
};

static void fillTransposedPairBlock(__fp16 *carrier,
                                    PackedMatrix gate,
                                    PackedMatrix up,
                                    uint32_t block) {
    uint32_t rowBase = block * PairRowsPerBlock;
    uint32_t rowCount = MIN(PairRowsPerBlock, Intermediate - rowBase);
    __fp16 gateTile[PairRowsPerBlock][Hidden];
    __fp16 upTile[PairRowsPerBlock][Hidden];
    for (uint32_t row = 0; row < rowCount; ++row) {
        expandRow(gate, rowBase + row, gateTile[row]);
        expandRow(up, rowBase + row, upTile[row]);
    }
    for (uint32_t column = 0; column < Hidden; ++column) {
        __fp16 *destination = carrier + (uint64_t)column * CarrierStride + Positions + rowBase;
        for (uint32_t row = 0; row < rowCount; ++row) {
            destination[row] = gateTile[row][column];
            destination[Intermediate + row] = upTile[row][column];
        }
    }
}

static void fillTransposedPair(__fp16 *carrier, PackedMatrix gate, PackedMatrix up) {
    dispatch_apply(PairBlockCount,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) {
        fillTransposedPairBlock(carrier, gate, up, (uint32_t)block);
    });
}

static void fillDownBlock(__fp16 *carrier, PackedMatrix down, uint32_t block) {
    uint32_t start = block * DownRowsPerBlock;
    uint32_t end = MIN(start + DownRowsPerBlock, Hidden);
    for (uint32_t row = start; row < end; ++row) {
        __fp16 *destination = carrier + (uint64_t)row * CarrierStride + Positions + 2 * Intermediate;
        expandRow(down, row, destination);
    }
}

static void fillDown(__fp16 *carrier, PackedMatrix down) {
    dispatch_apply(DownBlockCount,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) {
        fillDownBlock(carrier, down, (uint32_t)block);
    });
}

static void fillInput(__fp16 *carrier, const float *input) {
    for (uint32_t column = 0; column < Hidden; ++column) {
        memset(carrier + (uint64_t)column * CarrierStride, 0,
               Positions * sizeof(__fp16));
        carrier[(uint64_t)column * CarrierStride] = (__fp16)input[column];
    }
}

static bool fillCarrier(ANEFieldfareANEExpertDispatcher *dispatcher,
                        uint32_t bank,
                        uint32_t expert,
                        const float *input) {
    IOSurfaceRef surface = ANEFieldfareANEProgramInput(dispatcher->program, bank);
    if (!surface || IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return false;
    __fp16 *carrier = IOSurfaceGetBaseAddress(surface);
    bool valid = carrier && IOSurfaceGetAllocSize(surface) >=
        (size_t)Hidden * CarrierStride * sizeof(__fp16);
    if (valid) {
        fillInput(carrier, input);
        const uint8_t *base = dispatcher->packedBytes + (uint64_t)expert * ExpertStride;
        PackedMatrix gate = matrix(base, Intermediate, Hidden,
                                   GateWeightOffset, GateScaleOffset, GateBiasOffset);
        PackedMatrix up = matrix(base, Intermediate, Hidden,
                                 UpWeightOffset, UpScaleOffset, UpBiasOffset);
        PackedMatrix down = matrix(base, Hidden, Intermediate,
                                   DownWeightOffset, DownScaleOffset, DownBiasOffset);
        fillTransposedPair(carrier, gate, up);
        fillDown(carrier, down);
    }
    IOSurfaceUnlock(surface, 0, NULL);
    return valid;
}

static bool fillCarrierBatch(ANEFieldfareANEExpertDispatcher *dispatcher,
                             const uint32_t experts[ANEFieldfareGemma4TopK],
                             const float *input) {
    IOSurfaceRef surfaces[ANEFieldfareGemma4TopK] = {0};
    __fp16 *carriers[ANEFieldfareGemma4TopK] = {0};
    PackedMatrix gates[ANEFieldfareGemma4TopK] = {0};
    PackedMatrix ups[ANEFieldfareGemma4TopK] = {0};
    PackedMatrix downs[ANEFieldfareGemma4TopK] = {0};
    uint32_t locked = 0;
    bool valid = true;
    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
        surfaces[rank] = ANEFieldfareANEProgramInput(dispatcher->program, rank);
        if (!surfaces[rank] || IOSurfaceLock(surfaces[rank], 0, NULL) != kIOReturnSuccess) {
            valid = false;
            break;
        }
        ++locked;
        carriers[rank] = IOSurfaceGetBaseAddress(surfaces[rank]);
        if (!carriers[rank] || IOSurfaceGetAllocSize(surfaces[rank]) <
                (size_t)Hidden * CarrierStride * sizeof(__fp16)) {
            valid = false;
            break;
        }
        const uint8_t *base = dispatcher->packedBytes + (uint64_t)experts[rank] * ExpertStride;
        gates[rank] = matrix(base, Intermediate, Hidden,
                             GateWeightOffset, GateScaleOffset, GateBiasOffset);
        ups[rank] = matrix(base, Intermediate, Hidden,
                           UpWeightOffset, UpScaleOffset, UpBiasOffset);
        downs[rank] = matrix(base, Hidden, Intermediate,
                             DownWeightOffset, DownScaleOffset, DownBiasOffset);
    }
    if (valid) {
        __fp16 **carrierValues = carriers;
        PackedMatrix *gateValues = gates;
        PackedMatrix *upValues = ups;
        PackedMatrix *downValues = downs;
        dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
        dispatch_apply(ANEFieldfareGemma4TopK, queue, ^(size_t rank) {
            fillInput(carrierValues[rank], input);
        });
        dispatch_apply(ANEFieldfareGemma4TopK * PairBlockCount, queue, ^(size_t task) {
            uint32_t rank = (uint32_t)task / PairBlockCount;
            uint32_t block = (uint32_t)task % PairBlockCount;
            fillTransposedPairBlock(carrierValues[rank], gateValues[rank], upValues[rank], block);
        });
        dispatch_apply(ANEFieldfareGemma4TopK * DownBlockCount, queue, ^(size_t task) {
            uint32_t rank = (uint32_t)task / DownBlockCount;
            uint32_t block = (uint32_t)task % DownBlockCount;
            fillDownBlock(carrierValues[rank], downValues[rank], block);
        });
    }
    for (uint32_t rank = 0; rank < locked; ++rank)
        IOSurfaceUnlock(surfaces[rank], 0, NULL);
    return valid;
}

static void setError(ANEFieldfareANEExpertDispatcher *dispatcher, const char *message) {
    free(dispatcher->lastError);
    dispatcher->lastError = message ? strdup(message) : NULL;
}

bool ANEFieldfareANEExpertDispatcherSetPackedLayerPath(
    ANEFieldfareANEExpertDispatcher *dispatcher,
    const char *packedLayerPath,
    const char **errorMessage) {
    if (errorMessage) *errorMessage = NULL;
    if (!dispatcher || !packedLayerPath) {
        if (errorMessage) *errorMessage = strdup("invalid packed layer replacement");
        return false;
    }
    int descriptor = open(packedLayerPath, O_RDONLY | O_CLOEXEC);
    struct stat status = {0};
    bool valid = descriptor >= 0 && fstat(descriptor, &status) == 0 &&
        status.st_size == (off_t)((uint64_t)ExpertStride * ExpertCount);
    uint64_t byteCount = valid ? (uint64_t)status.st_size : 0;
    const uint8_t *bytes = valid
        ? mmap(NULL, (size_t)byteCount, PROT_READ, MAP_PRIVATE, descriptor, 0) : NULL;
    valid = valid && bytes != MAP_FAILED;
    if (!valid) {
        if (bytes && bytes != MAP_FAILED) munmap((void *)bytes, (size_t)byteCount);
        if (descriptor >= 0) close(descriptor);
        setError(dispatcher, "packed layer replacement failed validation");
        if (errorMessage) *errorMessage = strdup(dispatcher->lastError);
        return false;
    }

    const uint8_t *oldBytes = dispatcher->packedBytes;
    uint64_t oldByteCount = dispatcher->packedByteCount;
    int oldDescriptor = dispatcher->descriptor;
    dispatcher->packedBytes = bytes;
    dispatcher->packedByteCount = byteCount;
    dispatcher->descriptor = descriptor;
    setError(dispatcher, NULL);
    if (oldBytes && oldByteCount) munmap((void *)oldBytes, (size_t)oldByteCount);
    if (oldDescriptor >= 0) close(oldDescriptor);
    return true;
}

ANEFieldfareANEExpertDispatcher *ANEFieldfareANEExpertDispatcherCreate(
    const char *packedLayerPath,
    const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!packedLayerPath) return NULL;
        mach_timebase_info(&timebase);
        ANEFieldfareANEExpertDispatcher *dispatcher = calloc(1, sizeof(*dispatcher));
        if (!dispatcher) return NULL;
        dispatcher->descriptor = -1;
        dispatcher->prefetchDescriptor = -1;
        dispatcher->descriptor = open(packedLayerPath, O_RDONLY | O_CLOEXEC);
        struct stat status = {0};
        bool valid = dispatcher->descriptor >= 0 && fstat(dispatcher->descriptor, &status) == 0 &&
            status.st_size == (off_t)((uint64_t)ExpertStride * ExpertCount);
        if (valid) {
            dispatcher->packedByteCount = (uint64_t)status.st_size;
            dispatcher->packedBytes = mmap(NULL, (size_t)dispatcher->packedByteCount,
                                           PROT_READ, MAP_PRIVATE, dispatcher->descriptor, 0);
            valid = dispatcher->packedBytes != MAP_FAILED;
            if (!valid) dispatcher->packedBytes = NULL;
        }
        const char *programError = NULL;
        if (valid) {
            NSString *mil = expertMIL();
            dispatcher->program = ANEFieldfareANEProgramCreate(
                mil.UTF8String,
                (uint64_t)Hidden * CarrierStride * sizeof(__fp16),
                (uint64_t)Hidden * Positions * sizeof(float),
                ANEFieldfareGemma4TopK, &programError);
            valid = dispatcher->program != NULL;
        }
        if (!valid) {
            if (errorMessage) *errorMessage = strdup(programError ?: "expert dispatcher creation failed");
            free((void *)programError);
            ANEFieldfareANEExpertDispatcherDestroy(dispatcher);
            return NULL;
        }
        free((void *)programError);
        return dispatcher;
    }
}

ANEFieldfareANEExpertDispatcher *ANEFieldfareANEExpertDispatcherCreateDeviceCarrier(uint32_t batchExperts, const char **error) {
    @autoreleasepool {
        if (error) *error = NULL;
        if (batchExperts != 1 && batchExperts != 4) {
            if (error) *error = strdup("device expert batch must be 1 or 4");
            return NULL;
        }
        mach_timebase_info(&timebase);
        ANEFieldfareANEExpertDispatcher *d = calloc(1, sizeof(*d));
        if (!d) return NULL;
        d->descriptor = -1; d->prefetchDescriptor = -1;
        NSMutableString *mil = [expertMIL() mutableCopy];
        // Power-of-two input scaling avoids ANE's loss of tiny products.
        // Undo it before nonlinear gate evaluation and after the down product;
        // the up branch carries the scale through GELU(gate) * up.
        [mil replaceOccurrencesOfString:@"> gate=matmul(" withString:@"> gate_scaled=matmul(" options:0 range:NSMakeRange(0, mil.length)];
        [mil replaceOccurrencesOfString:@"    tensor<fp16,[1,1,64,704]> gate2=" withString:@"    tensor<fp16,[1,1,64,704]> gate=mul(x=gate_scaled,y=fp16(0.015625))[name=string(\"gate_unscale\")];\n    tensor<fp16,[1,1,64,704]> gate2=" options:0 range:NSMakeRange(0, mil.length)];
        [mil replaceOccurrencesOfString:@"> o2=matmul(" withString:@"> o2_scaled=matmul(" options:0 range:NSMakeRange(0, mil.length)];
        [mil replaceOccurrencesOfString:@"    tensor<fp16,[1,1,2816,64]> ot=" withString:@"    tensor<fp16,[1,1,64,2816]> o2=mul(x=o2_scaled,y=fp16(0.015625))[name=string(\"down_unscale\")];\n    tensor<fp16,[1,1,2816,64]> ot=" options:0 range:NSMakeRange(0, mil.length)];
        if (batchExperts > 1) {
            // Preserve each expert's D/F matrix axes while introducing the
            // leading batch dimension. External channels flatten batch and D.
            [mil replaceOccurrencesOfString:@"[1,2816,1," withString:@"[1,11264,1,"
                options:0 range:NSMakeRange(0, mil.length)];
            [mil replaceOccurrencesOfString:@"[1,1," withString:@"[4,1,"
                options:0 range:NSMakeRange(0, mil.length)];
        }
        d->program = ANEFieldfareANEProgramCreate(mil.UTF8String,
            (uint64_t)batchExperts * Hidden * CarrierStride * sizeof(__fp16),
            (uint64_t)batchExperts * Hidden * Positions * sizeof(float), 2, error);
        if (!d->program) { ANEFieldfareANEExpertDispatcherDestroy(d); return NULL; }
        return d;
    }
}
IOSurfaceRef ANEFieldfareANEExpertDispatcherInput(ANEFieldfareANEExpertDispatcher *d, uint32_t bank) {
    return d ? ANEFieldfareANEProgramInput(d->program, bank) : NULL;
}
IOSurfaceRef ANEFieldfareANEExpertDispatcherOutput(ANEFieldfareANEExpertDispatcher *d, uint32_t bank) {
    return d ? ANEFieldfareANEProgramOutput(d->program, bank) : NULL;
}
bool ANEFieldfareANEExpertDispatcherSetOutputEvent(ANEFieldfareANEExpertDispatcher *d,
    uint32_t bank, CFTypeRef event, uint64_t value, const char **error) {
    return d && ANEFieldfareANEProgramSetOutputSignalEvent(d->program, bank, event, 0, value, error);
}
bool ANEFieldfareANEExpertDispatcherEvaluateDeviceCarrier(ANEFieldfareANEExpertDispatcher *d,
    uint32_t bank, const char **error) {
    ANEFieldfareANEEvaluationTiming timing = {0};
    bool valid = d && ANEFieldfareANEProgramEvaluate(d->program, bank, 5 * NSEC_PER_SEC, &timing, error);
    // The device-carrier scheduler serializes submissions, including counters.
    if (valid) { ++d->dispatchCount; d->totalEvaluationMilliseconds += timing.completionMilliseconds; }
    return valid;
}

bool ANEFieldfareANEExpertDispatcherCall(
    void *rawDispatcher,
    uint32_t expert,
    const float input[ANEFieldfareGemma4HiddenSize],
    float output[ANEFieldfareGemma4HiddenSize]) {
    return ANEFieldfareANEExpertDispatcherCallPositions(rawDispatcher, expert, 1, input, output);
}

bool ANEFieldfareANEExpertDispatcherCallPositions(
    void *rawDispatcher, uint32_t expert, uint32_t count,
    const float *input, float *output) {
    ANEFieldfareANEExpertDispatcher *dispatcher = rawDispatcher;
    if (!dispatcher || !dispatcher->packedBytes || expert >= ExpertCount || !count || count > Positions || !input || !output)
        return false;
    uint32_t bank = (uint32_t)(dispatcher->dispatchCount & 1);
    uint64_t started = mach_absolute_time();
    if (!fillCarrier(dispatcher, bank, expert, input)) {
        setError(dispatcher, "failed to fill expert carrier");
        return false;
    }
    if (count > 1) {
        IOSurfaceRef carrier = ANEFieldfareANEProgramInput(dispatcher->program, bank);
        if (!carrier || IOSurfaceLock(carrier, 0, NULL) != kIOReturnSuccess) {
            setError(dispatcher, "failed to lock position carrier");
            return false;
        }
        __fp16 *values = IOSurfaceGetBaseAddress(carrier);
        bool valid = values && IOSurfaceGetAllocSize(carrier) >=
            (size_t)Hidden * CarrierStride * sizeof(__fp16);
        if (valid) {
            for (uint32_t column = 0; column < Hidden; ++column)
                for (uint32_t position = 1; position < count; ++position)
                    values[(uint64_t)column * CarrierStride + position] =
                        (__fp16)input[(uint64_t)position * Hidden + column];
        }
        IOSurfaceUnlock(carrier, 0, NULL);
        if (!valid) { setError(dispatcher, "invalid position carrier size"); return false; }
    }
    dispatcher->totalPreparationMilliseconds += milliseconds(mach_absolute_time() - started);
    ANEFieldfareANEEvaluationTiming timing = {0};
    const char *evaluationError = NULL;
    bool valid = ANEFieldfareANEProgramEvaluate(
        dispatcher->program, bank, 5 * NSEC_PER_SEC, &timing, &evaluationError);
    if (!valid) {
        setError(dispatcher, evaluationError ?: "ANE expert evaluation failed");
        free((void *)evaluationError);
        return false;
    }
    free((void *)evaluationError);
    IOSurfaceRef surface = ANEFieldfareANEProgramOutput(dispatcher->program, bank);
    valid = surface && IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess;
    if (valid) {
        const float *values = IOSurfaceGetBaseAddress(surface);
        valid = values && IOSurfaceGetAllocSize(surface) >= (size_t)Hidden * Positions * sizeof(float);
        for (uint32_t row = 0; valid && row < Hidden; ++row)
            for (uint32_t position = 0; valid && position < count; ++position) {
                uint64_t offset = (uint64_t)position * Hidden + row;
                output[offset] = values[(uint64_t)row * Positions + position];
                valid = isfinite(output[offset]);
            }
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    }
    if (!valid) {
        setError(dispatcher, "invalid ANE expert output surface");
        return false;
    }
    setError(dispatcher, NULL);
    dispatcher->totalEvaluationMilliseconds += timing.completionMilliseconds;
    dispatcher->lastBank = bank;
    ++dispatcher->dispatchCount;
    return true;
}

bool ANEFieldfareANEExpertDispatcherCallBatch(
    void *rawDispatcher,
    const uint32_t experts[ANEFieldfareGemma4TopK],
    const float input[ANEFieldfareGemma4HiddenSize],
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize]) {
    ANEFieldfareANEExpertDispatcher *dispatcher = rawDispatcher;
    if (!dispatcher || !dispatcher->packedBytes || !experts || !input || !outputs ||
        ANEFieldfareANEProgramGetInfo(dispatcher->program).requestBankCount < ANEFieldfareGemma4TopK) return false;
    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
        if (experts[rank] >= ExpertCount) return false;
    uint64_t started = mach_absolute_time();
    if (!fillCarrierBatch(dispatcher, experts, input)) {
        setError(dispatcher, "failed to fill batched expert carriers"); return false;
    }
    dispatcher->totalPreparationMilliseconds += milliseconds(mach_absolute_time() - started);

    ANEFieldfareANEEvaluationTiming timings[ANEFieldfareGemma4TopK] = {0};
    const char *errors[ANEFieldfareGemma4TopK] = {0};
    ANEFieldfareANEEvaluationTiming *timingValues = timings;
    const char **errorValues = errors;
    __block _Atomic int evaluationFailures = 0;
    dispatch_apply(ANEFieldfareGemma4TopK,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t rank) {
        if (!ANEFieldfareANEProgramEvaluate(dispatcher->program, (uint32_t)rank,
                                            5 * NSEC_PER_SEC, &timingValues[rank], &errorValues[rank]))
            atomic_fetch_add_explicit(&evaluationFailures, 1, memory_order_relaxed);
    });
    if (atomic_load_explicit(&evaluationFailures, memory_order_relaxed)) {
        const char *message = "batched ANE expert evaluation failed";
        for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank)
            if (errors[rank]) { message = errors[rank]; break; }
        setError(dispatcher, message);
        for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) free((void *)errors[rank]);
        return false;
    }
    bool valid = true;
    for (uint32_t rank = 0; rank < ANEFieldfareGemma4TopK; ++rank) {
        free((void *)errors[rank]);
        IOSurfaceRef surface = ANEFieldfareANEProgramOutput(dispatcher->program, rank);
        valid = valid && surface &&
            IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess;
        if (!valid) break;
        const float *values = IOSurfaceGetBaseAddress(surface);
        valid = values && IOSurfaceGetAllocSize(surface) >= (size_t)Hidden * Positions * sizeof(float);
        for (uint32_t row = 0; valid && row < Hidden; ++row) {
            outputs[rank][row] = values[(uint64_t)row * Positions];
            valid = isfinite(outputs[rank][row]);
        }
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        dispatcher->totalEvaluationMilliseconds += timings[rank].completionMilliseconds;
    }
    if (!valid) { setError(dispatcher, "invalid batched ANE expert output"); return false; }
    dispatcher->dispatchCount += ANEFieldfareGemma4TopK;
    dispatcher->lastBank = ANEFieldfareGemma4TopK - 1;
    setError(dispatcher, NULL);
    return true;
}

void ANEFieldfareANEExpertDispatcherWaitForLayerPrefetch(
    ANEFieldfareANEExpertDispatcher *dispatcher) {
    if (!dispatcher || dispatcher->prefetchDescriptor < 0) return;
    uint64_t start = mach_absolute_time();
    close(dispatcher->prefetchDescriptor);
    dispatcher->prefetchDescriptor = -1;
    dispatcher->prefetchWaitMilliseconds += milliseconds(mach_absolute_time() - start);
}

bool ANEFieldfareANEExpertDispatcherPrefetchLayerExperts(
    ANEFieldfareANEExpertDispatcher *dispatcher, const char *path,
    const uint32_t *experts, uint32_t count) {
    if (!dispatcher || !path || !experts || !count || count > ExpertCount) return false;
    for (uint32_t i = 0; i < count; ++i)
        if (experts[i] >= ExpertCount) return false;
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    struct stat status = {0};
    if (descriptor < 0) { ++dispatcher->prefetchFailures; return false; }
    if (fstat(descriptor, &status) || !S_ISREG(status.st_mode) ||
        status.st_size < (off_t)ExpertCount * ExpertStride) {
        close(descriptor);
        ++dispatcher->prefetchFailures;
        return false;
    }
    ANEFieldfareANEExpertDispatcherWaitForLayerPrefetch(dispatcher);
    dispatcher->prefetchDescriptor = descriptor;
    ++dispatcher->prefetchRequests;
    bool valid = true;
    for (uint32_t i = 0; i < count; ++i) {
        struct radvisory advice = {.ra_offset = (off_t)experts[i] * ExpertStride,
                                    .ra_count = ExpertStride};
        if (fcntl(descriptor, F_RDADVISE, &advice) == 0) {
            ++dispatcher->prefetchHints;
            dispatcher->prefetchBytes += ExpertStride;
        } else {
            ++dispatcher->prefetchFailures;
            valid = false;
        }
    }
    return valid;
}

ANEFieldfareANEExpertDispatchInfo ANEFieldfareANEExpertDispatcherGetInfo(
    const ANEFieldfareANEExpertDispatcher *dispatcher) {
    if (!dispatcher) return (ANEFieldfareANEExpertDispatchInfo){0};
    return (ANEFieldfareANEExpertDispatchInfo){
        .program = ANEFieldfareANEProgramGetInfo(dispatcher->program),
        .dispatchCount = dispatcher->dispatchCount,
        .totalPreparationMilliseconds = dispatcher->totalPreparationMilliseconds,
        .totalEvaluationMilliseconds = dispatcher->totalEvaluationMilliseconds,
        .lastBank = dispatcher->lastBank,
        .layerPrefetchRequests = dispatcher->prefetchRequests,
        .layerPrefetchExpertHints = dispatcher->prefetchHints,
        .layerPrefetchHintBytes = dispatcher->prefetchBytes,
        .layerPrefetchFailures = dispatcher->prefetchFailures,
        .layerPrefetchWaitMilliseconds = dispatcher->prefetchWaitMilliseconds,
    };
}

const char *ANEFieldfareANEExpertDispatcherLastError(
    const ANEFieldfareANEExpertDispatcher *dispatcher) {
    return dispatcher ? dispatcher->lastError : NULL;
}

void ANEFieldfareANEExpertDispatcherDestroy(ANEFieldfareANEExpertDispatcher *dispatcher) {
    if (!dispatcher) return;
    ANEFieldfareANEExpertDispatcherWaitForLayerPrefetch(dispatcher);
    ANEFieldfareANEProgramDestroy(dispatcher->program);
    if (dispatcher->packedBytes && dispatcher->packedByteCount)
        munmap((void *)dispatcher->packedBytes, (size_t)dispatcher->packedByteCount);
    if (dispatcher->descriptor >= 0) close(dispatcher->descriptor);
    free(dispatcher->lastError);
    free(dispatcher);
}
