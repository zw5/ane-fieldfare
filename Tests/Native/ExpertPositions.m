#import <Foundation/Foundation.h>
#include "ANEExpertDispatch.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) { @autoreleasepool {
    const size_t D=2816, F=704, stride=3358720;
    char path[]="/tmp/ane-position-fixture.XXXXXX";
    int fd=mkstemp(path);
    if(fd<0) return 2;
    uint8_t *bytes=calloc(stride,1);
    const size_t woff[]={0,1115136,2230272}, soff[]={991232,2106368,3221504};
    for(size_t m=0;m<3;m++) {
        for(size_t g=0;g<D*F/64;g++) { bytes[soff[m]+g*2]=0x80; bytes[soff[m]+g*2+1]=0x3f; }
        size_t columns=m==2?F:D;
        for(size_t r=0;r<F;r++) { size_t linear=r*columns+r; bytes[woff[m]+linear/2] |= (linear%2?16:1); }
    }
    bool valid=ftruncate(fd,(off_t)stride*128)==0 && pwrite(fd,bytes,stride,0)==(ssize_t)stride;
    free(bytes); close(fd);
    const char *error=NULL;
    ANEFieldfareANEExpertDispatcher *dispatcher=valid?ANEFieldfareANEExpertDispatcherCreate(path,&error):NULL;
    if(!dispatcher) { fprintf(stderr,"creation rejected: %s\n",error?:"fixture failure");free((void*)error);unlink(path);return 3; }
    float *input=calloc(D*64,sizeof(float)), *output=calloc(D*64,sizeof(float));
    if(!input||!output) return 4;
    uint32_t counts[]={1,7,64,1}; size_t compared=0, mismatches=0; double maxError=0;
    for(size_t arm=0;arm<4;arm++) {
        uint32_t count=counts[arm];
        for(size_t p=0;p<count;p++) for(size_t d=0;d<D;d++) input[p*D+d]=(float)((int)(p%13)-6)*0.01f+(float)(d%7)*0.001f;
        for(size_t i=0;i<D*count;i++) output[i]=NAN;
        valid=ANEFieldfareANEExpertDispatcherCallPositions(dispatcher,0,count,input,output);
        if(!valid) {fprintf(stderr,"evaluation rejected: %s\n",ANEFieldfareANEExpertDispatcherLastError(dispatcher));break;}
        for(size_t p=0;p<count;p++) for(size_t d=0;d<D;d++) {
            double x=(float)(__fp16)input[p*D+d];
            double expected=d<F ? x*(0.5*x*(1+tanh(0.7978845608028654*(x+0.044715*x*x*x)))) : 0;
            double e=fabs(output[p*D+d]-expected);
            if(e>maxError)maxError=e;
            mismatches+=!isfinite(output[p*D+d])||e>0.0003;
            compared++;
        }
    }
    uint32_t hints[]={0,1};
    bool hinted=ANEFieldfareANEExpertDispatcherPrefetchLayerExperts(dispatcher,path,hints,2);
    ANEFieldfareANEExpertDispatcherWaitForLayerPrefetch(dispatcher);
    uint32_t invalidExpert=128;
    bool rejected=!ANEFieldfareANEExpertDispatcherPrefetchLayerExperts(dispatcher,path,&invalidExpert,1) &&
        !ANEFieldfareANEExpertDispatcherCallPositions(dispatcher,0,65,input,output);
    ANEFieldfareANEExpertDispatchInfo info=ANEFieldfareANEExpertDispatcherGetInfo(dispatcher);
    printf("{\"status\":\"%s\",\"comparisons\":%zu,\"mismatches\":%zu,\"maximum_absolute_error\":%.12g,\"dispatches\":%llu,\"storage_hint_accepted\":%s,\"invalid_requests_rejected\":%s,\"compile_count\":1,\"load_count\":1,\"claim_boundary\":\"Synthetic diagonal compact experts validate actual ANE multi-position execution and A/B/A carrier reuse, not checkpoint generation or throughput. Storage hints are call acceptance only.\"}\n", valid&&!mismatches&&rejected?"validated":"rejected",compared,mismatches,maxError,(unsigned long long)info.dispatchCount,hinted?"true":"false",rejected?"true":"false");
    ANEFieldfareANEExpertDispatcherDestroy(dispatcher); free(input);free(output);unlink(path);
    return valid&&!mismatches&&rejected?0:1;
} }
