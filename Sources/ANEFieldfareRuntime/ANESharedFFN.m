#import "ANESharedFFN.h"
#import <Foundation/Foundation.h>
#include "TensorIndex.h"
#include "Dequant.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct ANEFieldfareANESharedFFN {
    ANEFieldfareANEProgram *program;
    ANEFieldfareANESharedFFNInfo info;
};

static NSString *ffnMIL(uint32_t d, uint32_t f, uint32_t positions) {
    NSMutableString *s = [NSMutableString stringWithString:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [s appendFormat:@"func main<ios18>(tensor<fp16,[1,%u,1,64]> x) {\n", d];
    [s appendString:@"string pt=const()[name=string(\"pt\"),val=string(\"valid\")];\n"
        "tensor<int32,[2]> st=const()[name=string(\"st\"),val=tensor<int32,[2]>([1,1])];\n"
        "tensor<int32,[4]> pd=const()[name=string(\"pd\"),val=tensor<int32,[4]>([0,0,0,0])];\n"
        "tensor<int32,[2]> dl=const()[name=string(\"dl\"),val=tensor<int32,[2]>([1,1])];\n"
        "int32 gr=const()[name=string(\"gr\"),val=int32(1)];\n"];
    uint32_t matrixIndex = 0;
    for (NSString *name in @[@"gate", @"up", @"down"]) {
        BOOL down = [name isEqualToString:@"down"];
        uint32_t rows = down ? d : f, cols = down ? f : d;
        [s appendFormat:@"tensor<fp16,[%u,%u,1,1]> %@W=const()[name=string(\"%@W\"),val=tensor<fp16,[%u,%u,1,1]>(BLOBFILE(path=string(\"@model_path/weights/ffn.bin\"),offset=uint64(%llu)))];\n",
            rows, cols, name, name, rows, cols, (unsigned long long)(64 + matrixIndex++ * ((uint64_t)d * f * 2 + 64))];
    }
    for (NSString *name in @[@"gate", @"up"]) {
        [s appendFormat:@"tensor<fp16,[1,%u,1,64]> %@=conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=%@W,x=x)[name=string(\"%@\")];\n", f, name, name, name];
    }
    [s appendString:@"string mode=const()[name=string(\"mode\"),val=string(\"TANH_APPROXIMATION\")];\n"];
    [s appendFormat:@"tensor<fp16,[1,%u,1,64]> activated=gelu(x=gate,mode=mode)[name=string(\"activated\")];\n", f];
    [s appendFormat:@"tensor<fp16,[1,%u,1,64]> a=mul(x=activated,y=up)[name=string(\"a\")];\n", f];
    [s appendFormat:@"tensor<fp16,[1,%u,1,64]> y=conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=downW,x=a)[name=string(\"y\")];\n} -> (y);\n}\n", d];
    if (positions != 64) [s replaceOccurrencesOfString:@",1,64]>"
        withString:[NSString stringWithFormat:@",1,%u]>", positions]
        options:0 range:NSMakeRange(0, s.length)];
    return s;
}

ANEFieldfareANESharedFFN *ANEFieldfareANESharedFFNCreate(
    const char *path, uint32_t layer, uint32_t d, uint32_t f, uint32_t bits, uint32_t positions, const char **error) {
    @autoreleasepool {
        if (error) *error = NULL;
        if (!path || d != 2816 || f != 2112 || layer >= 30 || (bits != 4 && bits != 8) || (positions != 64 && positions != 128 && positions != 512 && positions != 1024)) {
            if (error) *error = strdup("unsupported shared FFN shape");
            return NULL;
        }
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        if (!ANEFieldfareTensorIndexOpen(path, &index)) {
            if (error) *error = strdup("cannot open shared FFN checkpoint index");
            return NULL;
        }
        const char *projections[] = {"gate", "up", "down"};
        const uint64_t elements = (uint64_t)d * f;
        const uint64_t matrixBytes = elements * sizeof(__fp16);
        const uint64_t combinedSize = 64 + 3 * (64 + matrixBytes);
        uint8_t *combined = calloc(combinedSize, 1);
        bool valid = combined != NULL;
        if (valid) { *(uint32_t *)combined = 3; *(uint32_t *)(combined + 4) = 2; }
        for (uint32_t m = 0; m < 3 && valid; ++m) {
            char name[160];
            snprintf(name, sizeof(name), "language_model.model.layers.%u.mlp.%s_proj.weight", layer, projections[m]);
            ANEFieldfareTensor tensor = {0};
            valid = ANEFieldfareTensorIndexFind(&index, name, &tensor) &&
                tensor.rows == (m == 2 ? d : f) && tensor.columns == (m == 2 ? f : d) &&
                tensor.weightBytes == elements * bits / 8 &&
                tensor.scaleBytes == elements / 64 * 2 && tensor.biasBytes == elements / 64 * 2;
            if (!valid) break;
            uint64_t offset = 64 + m * (64 + matrixBytes);
            uint8_t *chunk = combined + offset;
            *(uint32_t *)chunk = 0xDEADBEEF;
            *(uint32_t *)(chunk + 4) = 1;
            *(uint64_t *)(chunk + 8) = matrixBytes;
            *(uint64_t *)(chunk + 16) = offset + 64;
            __fp16 *out = (__fp16 *)(chunk + 64);
            // Validate metadata once per quantization group, then expand the
            // complete matrix through the independently tested NEON primitive.
            for (uint64_t g = 0; g < elements / 64; ++g) {
                float scale = ANEFieldfareBF16(tensor.scales + g * 2);
                float bias = ANEFieldfareBF16(tensor.biases + g * 2);
                float end = ((1u << bits) - 1) * scale + bias;
                if (!isfinite(scale) || !isfinite(bias) || !isfinite(end) ||
                    fabsf(bias) > 65504 || fabsf(end) > 65504) { valid = false; break; }
            }
            if (!valid) break;
            if (bits == 4) {
                ANEFieldfareDequantGroupsFP16(tensor.weights, tensor.scales, tensor.biases,
                                             elements / 64, out);
            } else {
                for (uint64_t g = 0; g < elements / 64; ++g) {
                    float scale = ANEFieldfareBF16(tensor.scales + g * 2);
                    float bias = ANEFieldfareBF16(tensor.biases + g * 2);
                    for (uint32_t c = 0; c < 64; ++c)
                        out[g * 64 + c] = (__fp16)(tensor.weights[g * 64 + c] * scale + bias);
                }
            }
        }
        ANEFieldfareANESharedFFN *ffn = valid ? calloc(1, sizeof(*ffn)) : NULL;
        if (ffn) {
            ffn->program = ANEFieldfareANEProgramCreateWithConstantWeights(
                ffnMIL(d, f, positions).UTF8String, "ffn.bin", combined, combinedSize,
                (uint64_t)d * positions * 2, (uint64_t)d * positions * 2, 1, error);
            if (ffn->program) {
                ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(ffn->program);
                ffn->info = (ANEFieldfareANESharedFFNInfo){
                    .constantBytes = combinedSize, .compileCount = 1, .loadCount = 1,
                    .compileMilliseconds = info.compileMilliseconds, .loadMilliseconds = info.loadMilliseconds};
            } else { free(ffn); ffn = NULL; }
        }
        free(combined);
        ANEFieldfareTensorIndexClose(&index);
        if (!ffn && error && !*error) *error = strdup("shared FFN constant preparation failed");
        return ffn;
    }
}

IOSurfaceRef ANEFieldfareANESharedFFNInput(ANEFieldfareANESharedFFN *f) {
    return f ? ANEFieldfareANEProgramInput(f->program, 0) : NULL;
}
IOSurfaceRef ANEFieldfareANESharedFFNOutput(ANEFieldfareANESharedFFN *f) {
    return f ? ANEFieldfareANEProgramOutput(f->program, 0) : NULL;
}
bool ANEFieldfareANESharedFFNBegin(ANEFieldfareANESharedFFN *f, const char **error) {
    return f && ANEFieldfareANEProgramEvaluateBegin(f->program, 0, NULL, error);
}
bool ANEFieldfareANESharedFFNSetOutputEvent(ANEFieldfareANESharedFFN *f,
    CFTypeRef event, uint64_t value, const char **error) {
    return f && ANEFieldfareANEProgramSetOutputSignalEvent(f->program, 0, event, 0, value, error);
}
bool ANEFieldfareANESharedFFNWait(ANEFieldfareANESharedFFN *f, uint64_t timeout, const char **error) {
    ANEFieldfareANEEvaluationTiming t = {0};
    bool valid = f && ANEFieldfareANEProgramEvaluateWait(f->program, 0, timeout, &t, error);
    if (valid) { ++f->info.evaluations; f->info.evaluationMilliseconds += t.completionMilliseconds; }
    return valid;
}
ANEFieldfareANESharedFFNInfo ANEFieldfareANESharedFFNGetInfo(ANEFieldfareANESharedFFN *f) {
    return f ? f->info : (ANEFieldfareANESharedFFNInfo){0};
}
void ANEFieldfareANESharedFFNDestroy(ANEFieldfareANESharedFFN *f) {
    if (f) { ANEFieldfareANEProgramDestroy(f->program); free(f); }
}
