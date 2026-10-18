#import <Foundation/Foundation.h>
#include "ANESharedFFN.h"
#include "TensorIndex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, const char **argv) { @autoreleasepool {
    if (argc < 2) { fprintf(stderr, "usage: shared-ffn model_weights.bin [layer]\n"); return 2; }
    const unsigned D=2816, F=2112, layer=argc>2?(unsigned)atoi(argv[2]):0;
    const char *error=NULL;
    ANEFieldfareANESharedFFN *ffn=ANEFieldfareANESharedFFNCreate(argv[1],layer,D,F,4,64,&error);
    if (!ffn) { fprintf(stderr,"ANE creation rejected: %s\n",error?:"unknown"); free((void*)error); return 3; }
    ANEFieldfareTensorIndex index={.fileDescriptor=-1};
    ANEFieldfareTensor matrices[3];
    const char *names[]={"gate","up","down"};
    bool valid=ANEFieldfareTensorIndexOpen(argv[1],&index);
    for (unsigned m=0;m<3 && valid;m++) {
        char name[160]; snprintf(name,sizeof(name),"language_model.model.layers.%u.mlp.%s_proj.weight",layer,names[m]);
        valid=ANEFieldfareTensorIndexFind(&index,name,&matrices[m]);
    }
    float *x=calloc(D,sizeof(float)), *gate=calloc(F,sizeof(float)), *up=calloc(F,sizeof(float)), *expected=calloc(D,sizeof(float));
    IOSurfaceRef input=ANEFieldfareANESharedFFNInput(ffn), output=ANEFieldfareANESharedFFNOutput(ffn);
    double maximum=0, squared=0, energy=0;
    size_t comparisons=0, nonfinite=0;
    for(unsigned sample=0;sample<3 && valid;sample++) {
        IOSurfaceLock(input,0,NULL);
        __fp16 *in=IOSurfaceGetBaseAddress(input);
        memset(in,0,D*64*2);
        for(unsigned d=0;d<D;d++) {
            x[d]=(float)(__fp16)(sin((d+1)*(sample+1)*0.173)*(sample+1)*0.25);
            in[d*64+sample]=(__fp16)x[d];
        }
        IOSurfaceUnlock(input,0,NULL);
        valid=ANEFieldfareAffineMatVec(&matrices[0],4,64,x,gate) && ANEFieldfareAffineMatVec(&matrices[1],4,64,x,up);
        for(unsigned f=0;f<F;f++) { double g=gate[f]; gate[f]=(float)(0.5*g*(1+tanh(0.7978845608028654*(g+0.044715*g*g*g)))*up[f]); }
        valid=valid && ANEFieldfareAffineMatVec(&matrices[2],4,64,gate,expected);
        valid=valid && ANEFieldfareANESharedFFNBegin(ffn,&error) && ANEFieldfareANESharedFFNWait(ffn,5000000000ull,&error);
        IOSurfaceLock(output,kIOSurfaceLockReadOnly,NULL);
        const __fp16 *out=IOSurfaceGetBaseAddress(output);
        for(unsigned d=0;d<D;d++) {
            double actual=out[d*64+sample], e=fabs(actual-expected[d]);
            nonfinite+=!isfinite(actual); maximum=fmax(maximum,e); squared+=e*e; energy+=(double)expected[d]*expected[d]; comparisons++;
        }
        IOSurfaceUnlock(output,kIOSurfaceLockReadOnly,NULL);
    }
    ANEFieldfareANESharedFFNInfo before=ANEFieldfareANESharedFFNGetInfo(ffn);
    const unsigned repetitions=30;
    for(unsigned i=0;i<repetitions && valid;i++) valid=ANEFieldfareANESharedFFNBegin(ffn,&error) && ANEFieldfareANESharedFFNWait(ffn,5000000000ull,&error);
    ANEFieldfareANESharedFFNInfo info=ANEFieldfareANESharedFFNGetInfo(ffn);
    double relative=sqrt(squared/fmax(energy,1e-30));
    valid=valid && !nonfinite && relative<0.01;
    printf("{\"status\":\"%s\",\"layer\":%u,\"comparisons\":%zu,\"nonfinite\":%zu,\"max_absolute_error\":%.9g,\"relative_l2_error\":%.9g,\"warm_evaluation_ms\":%.6f,\"compile_ms\":%.3f,\"load_ms\":%.3f,\"constant_bytes\":%llu,\"compile_count\":%u,\"load_count\":%u,\"evaluations\":%llu,\"claim_boundary\":\"Checkpoint shared FFN compared with independent affine CPU matvec and FP64 GELU; component accuracy and scalar request latency only.\"}\n",valid?"validated":"rejected",layer,comparisons,nonfinite,maximum,relative,(info.evaluationMilliseconds-before.evaluationMilliseconds)/repetitions,info.compileMilliseconds,info.loadMilliseconds,(unsigned long long)info.constantBytes,info.compileCount,info.loadCount,(unsigned long long)info.evaluations);
    if(error) { fprintf(stderr,"%s\n",error); free((void*)error); }
    ANEFieldfareTensorIndexClose(&index); ANEFieldfareANESharedFFNDestroy(ffn);
    free(x);free(gate);free(up);free(expected);
    return valid?0:1;
} }
