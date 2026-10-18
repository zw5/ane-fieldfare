#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <fcntl.h>
#include <mach/mach_time.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// This isolated gate deliberately compiles the production expert-program
// description into its own executable. It does not add a production API merely
// to expose the program's private carrier layout.
#include "../ANEFieldfareRuntime/ANEExpertDispatch.m"

static double gateMilliseconds(uint64_t ticks) {
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    return (double)ticks * tb.numer / tb.denom / 1e6;
}

static NSString *kernelSource(void) {
    return @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "constant uint H=2816, I=704, P=64, S=2176;\n"
    "constant uint GO=0, GS=991232, GB=1053184;\n"
    "constant uint UO=1115136, US=2106368, UB=2168320;\n"
    "constant uint DO=2230272, DS=3221504, DB=3283456;\n"
    "float bf(device const uchar *p,uint o){uint b=uint(p[o])|(uint(p[o+1])<<8);return as_type<float>(b<<16);}\n"
    "half q(device const uchar *p,uint wo,uint so,uint bo,uint linear){"
    "uint packed=p[wo+(linear>>1)];uint code=(linear&1)?(packed>>4):(packed&15);"
    "uint g=linear>>6;return half(float(code)*bf(p,so+2*g)+bf(p,bo+2*g));}\n"
    "kernel void expand(device const uchar *all [[buffer(0)]],device half *carrier [[buffer(1)]],"
    "constant uint &expert [[buffer(2)]],constant uint &batch [[buffer(3)]],uint n [[thread_position_in_grid]]){"
    "if(n>=H*S)return;device const uchar *p=all+expert*3358720;uint c=n/S,x=n-c*S;"
    "if(x<P){carrier[n]=(x==0)?half((float(int((c*17+batch*13)%101)-50))/512.0f):half(0);return;}"
    "if(x<P+I){uint row=x-P;carrier[n]=q(p,GO,GS,GB,row*H+c);return;}"
    "if(x<P+2*I){uint row=x-P-I;carrier[n]=q(p,UO,US,UB,row*H+c);return;}"
    "uint col=x-P-2*I;carrier[n]=q(p,DO,DS,DB,c*I+col);}\n"
    "kernel void expand8(device const uchar *all[[buffer(0)]],device half*c0[[buffer(1)]],device half*c1[[buffer(2)]],"
    "device half*c2[[buffer(3)]],device half*c3[[buffer(4)]],device half*c4[[buffer(5)]],device half*c5[[buffer(6)]],"
    "device half*c6[[buffer(7)]],device half*c7[[buffer(8)]],constant uint*experts[[buffer(9)]],"
    "constant uint&batch[[buffer(10)]],uint z[[thread_position_in_grid]]){uint E=H*S,b=z/E,n=z-b*E;if(b>=8)return;"
    "device half*carrier=b==0?c0:b==1?c1:b==2?c2:b==3?c3:b==4?c4:b==5?c5:b==6?c6:c7;"
    "device const uchar*p=all+experts[b]*3358720;uint c=n/S,x=n-c*S;"
    "if(x<P){carrier[n]=(x==0)?half((float(int((c*17+batch*13)%101)-50))/512.0f):half(0);return;}"
    "if(x<P+I){uint row=x-P;carrier[n]=q(p,GO,GS,GB,row*H+c);return;}"
    "if(x<P+2*I){uint row=x-P-I;carrier[n]=q(p,UO,US,UB,row*H+c);return;}"
    "uint col=x-P-2*I;carrier[n]=q(p,DO,DS,DB,c*I+col);}\n";
}

static void expectedInput(float *input, uint32_t batch) {
    for (uint32_t c=0;c<Hidden;++c)
        input[c]=(float)((int32_t)((c*17+batch*13)%101)-50)/512.0f;
}

static bool validateOutput(ANEFieldfareRoutedLayer *layer, ANEFieldfareANEProgram *program,
                           uint32_t bank, uint32_t expert, uint32_t batch,
                           double *maximum, double *rmse) {
    float *input=malloc(Hidden*sizeof(float)), *expected=malloc(Hidden*sizeof(float));
    if(!input||!expected){free(input);free(expected);return false;}
    expectedInput(input,batch);
    bool valid=ANEFieldfareRoutedLayerCPUExpert(layer,expert,input,expected);
    IOSurfaceRef surface=ANEFieldfareANEProgramOutput(program,bank);
    valid=valid&&surface&&IOSurfaceLock(surface,kIOSurfaceLockReadOnly,NULL)==kIOReturnSuccess;
    double squared=0;*maximum=0;
    if(valid){const float *actual=IOSurfaceGetBaseAddress(surface);
        for(uint32_t c=0;valid&&c<Hidden;++c){double d=fabs((double)actual[(uint64_t)c*Positions]-expected[c]);
            *maximum=fmax(*maximum,d);squared+=d*d;
            valid=isfinite(actual[(uint64_t)c*Positions])&&d<=0.05+0.03*fabs(expected[c]);}
        IOSurfaceUnlock(surface,kIOSurfaceLockReadOnly,NULL);}
    *rmse=sqrt(squared/Hidden);free(input);free(expected);return valid;
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc<2){fprintf(stderr,"usage: %s <model-directory> [repetitions]\n",argv[0]);return 2;}
    uint32_t repetitions=argc>2?(uint32_t)strtoul(argv[2],NULL,10):10;if(!repetitions)repetitions=1;
    NSString *root=[[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
    NSString *packed=[root stringByAppendingPathComponent:@"packed_experts/layer_00.bin"];
    int fd=open(packed.fileSystemRepresentation,O_RDONLY|O_CLOEXEC);struct stat st={0};
    bool valid=fd>=0&&fstat(fd,&st)==0&&st.st_size==(off_t)((uint64_t)ExpertStride*ExpertCount);
    void *mapping=valid?mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0):MAP_FAILED;
    valid=valid&&mapping!=MAP_FAILED;
    const char *aneError=NULL;
    ANEFieldfareANEProgram *program=valid?ANEFieldfareANEProgramCreate(expertMIL().UTF8String,
        (uint64_t)Hidden*CarrierStride*sizeof(__fp16),(uint64_t)Hidden*Positions*sizeof(float),8,&aneError):NULL;
    valid=valid&&program;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError *error=nil;
    id<MTLLibrary> library=[device newLibraryWithSource:kernelSource() options:nil error:&error];
    id<MTLFunction> function=[library newFunctionWithName:@"expand"];
    id<MTLComputePipelineState> pipeline=function?[device newComputePipelineStateWithFunction:function error:&error]:nil;
    id<MTLFunction> fusedFunction=[library newFunctionWithName:@"expand8"];
    id<MTLComputePipelineState> fusedPipeline=fusedFunction?[device newComputePipelineStateWithFunction:fusedFunction error:&error]:nil;
    bool fused=getenv("ANE_FIELDFARE_GPU_DEQUANT_FUSED8")!=NULL;
    id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLSharedEvent> ready=[device newSharedEvent];
    id<MTLBuffer> compact=valid?[device newBufferWithBytesNoCopy:mapping length:(NSUInteger)st.st_size
        options:MTLResourceStorageModeShared deallocator:nil]:nil;
    id<MTLBuffer> carriers[8]={nil};
    for(uint32_t b=0;valid&&b<8;++b){IOSurfaceRef s=ANEFieldfareANEProgramInput(program,b);
        void *base=s?IOSurfaceGetBaseAddress(s):NULL;size_t bytes=s?IOSurfaceGetAllocSize(s):0;
        carriers[b]=base?[device newBufferWithBytesNoCopy:base length:bytes options:MTLResourceStorageModeShared deallocator:nil]:nil;
        valid=carriers[b]&&carriers[b].contents==base;}
    valid=valid&&device&&library&&pipeline&&fusedPipeline&&queue&&ready&&compact;
    const uint32_t experts[8]={0,17,31,43,57,71,103,127};
    double totalPipelineMs=0,totalMetalGpuMs=0,totalANECompletionMs=0;
    double worstMax=0,worstRMSE=0;uint64_t compactBytes=(uint64_t)ExpertStride*8;
    for(uint32_t rep=0;valid&&rep<=repetitions;++rep){
        uint64_t eventValue=rep+1;id<MTLCommandBuffer> command=[queue commandBuffer];
        if(fused){id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];[e setComputePipelineState:fusedPipeline];
            [e setBuffer:compact offset:0 atIndex:0];for(uint32_t b=0;b<8;++b)[e setBuffer:carriers[b] offset:0 atIndex:b+1];
            [e setBytes:experts length:sizeof(experts) atIndex:9];[e setBytes:&rep length:sizeof(uint32_t) atIndex:10];
            [e dispatchThreads:MTLSizeMake((NSUInteger)Hidden*CarrierStride*8,1,1)
                threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,fusedPipeline.maxTotalThreadsPerThreadgroup),1,1)];[e endEncoding];
        }else for(uint32_t b=0;b<8;++b){id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
            [e setComputePipelineState:pipeline];[e setBuffer:compact offset:0 atIndex:0];[e setBuffer:carriers[b] offset:0 atIndex:1];
            [e setBytes:&experts[b] length:sizeof(uint32_t) atIndex:2];[e setBytes:&rep length:sizeof(uint32_t) atIndex:3];
            [e dispatchThreads:MTLSizeMake((NSUInteger)Hidden*CarrierStride,1,1)
                threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,pipeline.maxTotalThreadsPerThreadgroup),1,1)];[e endEncoding];}
        [command encodeSignalEvent:ready value:eventValue];
        for(uint32_t b=0;b<8&&valid;++b){const char *eventError=NULL;
            valid=ANEFieldfareANEProgramSetInputWaitEvent(program,b,(__bridge CFTypeRef)ready,eventValue,&eventError);
            if(!valid&&eventError)fprintf(stderr,"event: %s\n",eventError);free((void*)eventError);}
        uint64_t started=mach_absolute_time();[command commit];
        ANEFieldfareANEEvaluationTiming timings[8]={0};
        for(uint32_t b=0;b<8&&valid;++b){const char *beginError=NULL;
            valid=ANEFieldfareANEProgramEvaluateBegin(program,b,&timings[b],&beginError);
            if(!valid&&beginError)fprintf(stderr,"begin: %s\n",beginError);free((void*)beginError);}
        for(uint32_t b=0;b<8&&valid;++b){const char *waitError=NULL;
            valid=ANEFieldfareANEProgramEvaluateWait(program,b,5*NSEC_PER_SEC,&timings[b],&waitError);
            if(!valid&&waitError)fprintf(stderr,"wait: %s\n",waitError);free((void*)waitError);}
        [command waitUntilCompleted];double elapsed=gateMilliseconds(mach_absolute_time()-started);
        valid=valid&&command.status==MTLCommandBufferStatusCompleted;
        if(rep){totalPipelineMs+=elapsed;
            if(command.GPUEndTime>=command.GPUStartTime)totalMetalGpuMs+=(command.GPUEndTime-command.GPUStartTime)*1000.0;
            for(uint32_t b=0;b<8;++b)totalANECompletionMs+=timings[b].completionMilliseconds;}
    }
    // Open the CPU oracle only after timing, so its page activity cannot improve the measured compact-source path.
    ANEFieldfareRoutedLayer *layer=NULL;
    if(valid)valid=ANEFieldfareRoutedLayerOpen(root.fileSystemRepresentation,0,&layer);
    if(valid){uint32_t last=repetitions;for(uint32_t b=0;b<8&&valid;++b){double m=0,r=0;
        valid=validateOutput(layer,program,b,experts[b],last,&m,&r);worstMax=fmax(worstMax,m);worstRMSE=fmax(worstRMSE,r);}}
    double pipelineWall=totalPipelineMs/repetitions,metalGpu=totalMetalGpuMs/repetitions;
    ANEFieldfareANEProgramInfo info=ANEFieldfareANEProgramGetInfo(program);
    printf("{\"schema\":\"ane-fieldfare.gpu-dequant-ane-carrier.v1\",\"status\":\"%s\","
      "\"mode\":\"%s\",\"repetitions\":%u,\"banks\":8,\"compile_count\":1,\"load_count\":1,"
      "\"compact_source_bytes_per_batch\":%llu,\"fp16_carrier_bytes_per_batch\":%llu,"
      "\"cpu_fp16_preparation_or_copy\":false,\"pipeline_wall_ms_per_batch\":%.6f,"
      "\"metal_gpu_ms_per_batch\":%.6f,\"summed_ane_completion_ms_per_batch\":%.6f,"
      "\"max_absolute_error\":%.9g,\"rmse\":%.9g,\"request_banks\":%u,"
      "\"ane_compile_ms\":%.6f,\"ane_load_ms\":%.6f,\"error\":\"%s\"}\n",
      valid?"validated":"rejected",fused?"fused8":"eight-encoders",repetitions,compactBytes,
      (unsigned long long)((uint64_t)Hidden*CarrierStride*sizeof(__fp16)*8),pipelineWall,metalGpu,
      totalANECompletionMs/repetitions,worstMax,worstRMSE,info.requestBankCount,
      info.compileMilliseconds,info.loadMilliseconds,error.localizedDescription.UTF8String?:aneError?:"none");
    if(layer)ANEFieldfareRoutedLayerClose(layer);if(program)ANEFieldfareANEProgramDestroy(program);
    if(mapping!=MAP_FAILED)munmap(mapping,(size_t)st.st_size);if(fd>=0)close(fd);free((void*)aneError);
    return valid?0:1;
}}
