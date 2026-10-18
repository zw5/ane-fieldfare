#import "ANEDynamicProjection.h"

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <mach/mach_time.h>

enum { Positions = 64 };

struct ANEFieldfareANEDynamicProjection {
    ANEFieldfareANEProgram *program;
    uint32_t inputWidth;
    uint32_t outputWidth;
};

static NSString *projectionMIL(uint32_t input, uint32_t output) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n"
         "  func main<ios18>(tensor<fp16,[1,%u,1,%u]> p) {\n"
         "    tensor<int32,[4]> xb = const()[name=string(\"xb\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
         "    tensor<int32,[4]> xs = const()[name=string(\"xs\"),val=tensor<int32,[4]>([1,%u,1,%u])];\n"
         "    tensor<fp16,[1,%u,1,%u]> x = slice_by_size(x=p,begin=xb,size=xs)[name=string(\"x\")];\n"
         "    tensor<int32,[4]> wb = const()[name=string(\"wb\"),val=tensor<int32,[4]>([0,0,0,%u])];\n"
         "    tensor<int32,[4]> ws = const()[name=string(\"ws\"),val=tensor<int32,[4]>([1,%u,1,%u])];\n"
         "    tensor<fp16,[1,%u,1,%u]> w = slice_by_size(x=p,begin=wb,size=ws)[name=string(\"w\")];\n"
         "    tensor<int32,[4]> xr = const()[name=string(\"xr\"),val=tensor<int32,[4]>([1,1,%u,%u])];\n"
         "    tensor<fp16,[1,1,%u,%u]> x2 = reshape(shape=xr,x=x)[name=string(\"x2\")];\n"
         "    tensor<int32,[4]> pm = const()[name=string(\"pm\"),val=tensor<int32,[4]>([0,1,3,2])];\n"
         "    tensor<fp16,[1,1,%u,%u]> xt = transpose(perm=pm,x=x2)[name=string(\"xt\")];\n"
         "    tensor<int32,[4]> wr = const()[name=string(\"wr\"),val=tensor<int32,[4]>([1,1,%u,%u])];\n"
         "    tensor<fp16,[1,1,%u,%u]> w2 = reshape(shape=wr,x=w)[name=string(\"w2\")];\n"
         "    bool no = const()[name=string(\"no\"),val=bool(false)];\n"
         "    tensor<fp16,[1,1,%u,%u]> y2 = matmul(transpose_x=no,transpose_y=no,x=xt,y=w2)[name=string(\"projection\")];\n"
         "    tensor<fp16,[1,1,%u,%u]> yt = transpose(perm=pm,x=y2)[name=string(\"yt\")];\n"
         "    tensor<int32,[4]> yr = const()[name=string(\"yr\"),val=tensor<int32,[4]>([1,%u,1,%u])];\n"
         "    tensor<fp16,[1,%u,1,%u]> y16 = reshape(shape=yr,x=yt)[name=string(\"y16\")];\n"
         "    string f32 = const()[name=string(\"f32\"),val=string(\"fp32\")];\n"
         "    tensor<fp32,[1,%u,1,%u]> out = cast(dtype=f32,x=y16)[name=string(\"out\")];\n"
         "  } -> (out);\n"
         "}\n",
         input, Positions + output,
         input, Positions,
         input, Positions,
         Positions,
         input, output,
         input, output,
         input, Positions,
         input, Positions,
         Positions, input,
         input, output,
         input, output,
         Positions, output,
         output, Positions,
         output, Positions,
         output, Positions,
         output, Positions];
}

ANEFieldfareANEDynamicProjection *ANEFieldfareANEDynamicProjectionCreate(
    uint32_t inputWidth, uint32_t outputWidth, uint32_t requestBanks,
    const char **errorMessage) {
    if (!inputWidth || !outputWidth) return NULL;
    NSString *mil = projectionMIL(inputWidth, outputWidth);
    uint64_t inputBytes = (uint64_t)inputWidth * (Positions + outputWidth) * sizeof(__fp16);
    uint64_t outputBytes = (uint64_t)outputWidth * Positions * sizeof(float);
    ANEFieldfareANEProgram *program = ANEFieldfareANEProgramCreate(
        mil.UTF8String, inputBytes, outputBytes, requestBanks, errorMessage);
    if (!program) return NULL;
    ANEFieldfareANEDynamicProjection *projection = calloc(1, sizeof(*projection));
    projection->program = program;
    projection->inputWidth = inputWidth;
    projection->outputWidth = outputWidth;
    return projection;
}

bool ANEFieldfareANEDynamicProjectionEvaluate(
    ANEFieldfareANEDynamicProjection *projection, uint32_t bank, const float *input,
    const ANEFieldfareTensor *weights, uint32_t weightBits, uint32_t groupSize,
    float *output, double *preparationMilliseconds,
    ANEFieldfareANEEvaluationTiming *timing, const char **errorMessage) {
    if (!projection || !input || !weights || !output ||
        weights->rows != projection->outputWidth || weights->columns != projection->inputWidth)
        return false;
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    uint64_t started = mach_absolute_time();
    IOSurfaceRef carrierSurface = ANEFieldfareANEProgramInput(projection->program, bank);
    IOSurfaceLock(carrierSurface, 0, NULL);
    __fp16 *carrier = IOSurfaceGetBaseAddress(carrierSurface);
    uint32_t stride = Positions + projection->outputWidth;
    memset(carrier, 0, (size_t)projection->inputWidth * stride * sizeof(*carrier));
    dispatch_apply(projection->inputWidth,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t column) {
        carrier[column * stride] = (__fp16)input[column];
        for (uint32_t row = 0; row < projection->outputWidth; ++row) {
            float value = 0;
            ANEFieldfareAffineValue(weights, weightBits, groupSize, row, (uint32_t)column, &value);
            carrier[column * stride + Positions + row] = (__fp16)value;
        }
    });
    IOSurfaceUnlock(carrierSurface, 0, NULL);
    uint64_t prepared = mach_absolute_time();
    if (preparationMilliseconds)
        *preparationMilliseconds = (double)(prepared - started) * timebase.numer / timebase.denom / 1e6;
    if (!ANEFieldfareANEProgramEvaluate(
            projection->program, bank, 5ULL * NSEC_PER_SEC, timing, errorMessage)) return false;
    IOSurfaceRef outputSurface = ANEFieldfareANEProgramOutput(projection->program, bank);
    IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL);
    const float *values = IOSurfaceGetBaseAddress(outputSurface);
    for (uint32_t row = 0; row < projection->outputWidth; ++row)
        output[row] = values[(uint64_t)row * Positions];
    IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);
    return true;
}

ANEFieldfareANEProgramInfo ANEFieldfareANEDynamicProjectionGetInfo(
    ANEFieldfareANEDynamicProjection *projection) {
    return projection ? ANEFieldfareANEProgramGetInfo(projection->program)
                      : (ANEFieldfareANEProgramInfo){0};
}

ANEFieldfareANEProgram *ANEFieldfareANEDynamicProjectionProgram(
    ANEFieldfareANEDynamicProjection *projection) {
    return projection ? projection->program : NULL;
}

void ANEFieldfareANEDynamicProjectionDestroy(ANEFieldfareANEDynamicProjection *projection) {
    if (!projection) return;
    ANEFieldfareANEProgramDestroy(projection->program);
    free(projection);
}
