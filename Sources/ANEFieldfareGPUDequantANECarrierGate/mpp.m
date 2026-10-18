#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../ANEFieldfareRuntime/TensorIndex.h"
#include "../ANEFieldfareRuntime/RoutedLayer.h"

enum { Hidden=2816, Intermediate=704, ExpertStride=3358720,
       GateWeightOffset=0, GateScaleOffset=991232, GateBiasOffset=1053184,
       UpWeightOffset=1115136, UpScaleOffset=2106368, UpBiasOffset=2168320,
       DownWeightOffset=2230272, DownScaleOffset=3221504, DownBiasOffset=3283456 };

static double ms(uint64_t ticks){mach_timebase_info_data_t t;mach_timebase_info(&t);return(double)ticks*t.numer/t.denom/1e6;}

static void encodeMPP(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> pso,
                      id<MTLBuffer> source,NSUInteger wo,NSUInteger so,NSUInteger bo,
                      id<MTLBuffer> x,id<MTLBuffer> y,uint32_t M,uint32_t N,uint32_t K){
    [e setComputePipelineState:pso];[e setBuffer:source offset:wo atIndex:0];
    [e setBuffer:source offset:so atIndex:1];[e setBuffer:source offset:bo atIndex:2];
    [e setBuffer:x offset:0 atIndex:3];[e setBuffer:y offset:0 atIndex:4];
    [e setBytes:&M length:4 atIndex:5];[e setBytes:&N length:4 atIndex:6];[e setBytes:&K length:4 atIndex:7];
    [e dispatchThreadgroups:MTLSizeMake((N+31)/32,(M+63)/64,1)
        threadsPerThreadgroup:MTLSizeMake(pso.threadExecutionWidth*4,1,1)];[e endEncoding];
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc<2)return 2;NSString *root=[[NSString stringWithUTF8String:argv[1]]stringByStandardizingPath];
    NSString *path=[root stringByAppendingPathComponent:@"packed_experts/layer_00.bin"];
    int fd=open(path.fileSystemRepresentation,O_RDONLY|O_CLOEXEC);struct stat st={0};
    bool valid=fd>=0&&fstat(fd,&st)==0;void *map=valid?mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0):MAP_FAILED;
    valid=valid&&map!=MAP_FAILED;id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    NSString *metal=[NSString stringWithContentsOfFile:@"Sources/ANEFieldfareHotpath/Metal/TensorCore/tensorops.metal"
        encoding:NSUTF8StringEncoding error:nil];MTLCompileOptions *options=[MTLCompileOptions new];
    options.languageVersion=MTLLanguageVersion4_0;NSError *error=nil;
    id<MTLLibrary> library=[device newLibraryWithSource:metal options:options error:&error];
    id<MTLFunction> function=[library newFunctionWithName:@"mpp_prefill_affine_threadgroup_f16"];
    id<MTLComputePipelineState> pso=function?[device newComputePipelineStateWithFunction:function error:&error]:nil;
    NSString *activationSource=@"#include <metal_stdlib>\nusing namespace metal; "
      "kernel void swiglu(device half *g[[buffer(0)]],device const half *u[[buffer(1)]],constant uint& count[[buffer(2)]],uint i[[thread_position_in_grid]]){"
      "if(i<count){float x=float(g[i]);float gelu=.5f*x*(1.f+tanh(.7978845608f*(x+.044715f*x*x*x)));g[i]=half(gelu*float(u[i]));}}";
    id<MTLLibrary> activationLibrary=[device newLibraryWithSource:activationSource options:nil error:&error];
    id<MTLFunction> activationFunction=[activationLibrary newFunctionWithName:@"swiglu"];
    id<MTLComputePipelineState> activationPSO=activationFunction?[device newComputePipelineStateWithFunction:activationFunction error:&error]:nil;
    id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLBuffer> source=valid?[device newBufferWithBytesNoCopy:map
        length:st.st_size options:MTLResourceStorageModeShared deallocator:nil]:nil;valid=valid&&pso&&queue&&source;
    valid=valid&&activationPSO;
    const uint32_t counts[5]={4,8,16,32,64};NSMutableArray *records=[NSMutableArray array];
    for(uint32_t ci=0;valid&&ci<5;++ci){uint32_t M=counts[ci],N=Intermediate,K=Hidden;
        id<MTLBuffer>x=[device newBufferWithLength:(NSUInteger)M*K*sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer>gate=[device newBufferWithLength:(NSUInteger)M*N*sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer>up=[device newBufferWithLength:(NSUInteger)M*N*sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer>y=[device newBufferWithLength:(NSUInteger)M*Hidden*sizeof(__fp16) options:MTLResourceStorageModeShared];
        __fp16 *xp=x.contents;for(uint32_t m=0;m<M;++m)for(uint32_t k=0;k<K;++k)
            xp[(uint64_t)m*K+k]=(__fp16)((float)((int32_t)((k*17+m*13)%101)-50)/512.0f);
        double totalWall=0,totalGpu=0;const uint32_t reps=20;
        for(uint32_t r=0;r<=reps;++r){id<MTLCommandBuffer>c=[queue commandBuffer];
            encodeMPP([c computeCommandEncoder],pso,source,GateWeightOffset,GateScaleOffset,GateBiasOffset,x,gate,M,N,K);
            encodeMPP([c computeCommandEncoder],pso,source,UpWeightOffset,UpScaleOffset,UpBiasOffset,x,up,M,N,K);
            id<MTLComputeCommandEncoder>a=[c computeCommandEncoder];[a setComputePipelineState:activationPSO];
            [a setBuffer:gate offset:0 atIndex:0];[a setBuffer:up offset:0 atIndex:1];
            uint32_t activationCount=M*Intermediate;[a setBytes:&activationCount length:4 atIndex:2];
            [a dispatchThreads:MTLSizeMake((NSUInteger)M*Intermediate,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[a endEncoding];
            encodeMPP([c computeCommandEncoder],pso,source,DownWeightOffset,DownScaleOffset,DownBiasOffset,gate,y,M,Hidden,Intermediate);
            uint64_t start=mach_absolute_time();[c commit];[c waitUntilCompleted];double wall=ms(mach_absolute_time()-start);
            if(r){totalWall+=wall;if(c.GPUEndTime>=c.GPUStartTime)totalGpu+=(c.GPUEndTime-c.GPUStartTime)*1000.0;}
            valid=c.status==MTLCommandBufferStatusCompleted;}
        ANEFieldfareRoutedLayer *layer=NULL;valid=valid&&ANEFieldfareRoutedLayerOpen(root.fileSystemRepresentation,0,&layer);
        float *in=malloc(K*sizeof(float)),*expected=malloc(Hidden*sizeof(float));__fp16 *actual=y.contents;
        double maximum=0,squared=0;uint64_t compared=0;
        for(uint32_t m=0;valid&&m<M;++m){for(uint32_t k=0;k<K;++k)in[k]=(float)xp[(uint64_t)m*K+k];
            valid=ANEFieldfareRoutedLayerCPUExpert(layer,0,in,expected);
            for(uint32_t n=0;valid&&n<Hidden;++n){double d=fabs((double)(float)actual[(uint64_t)m*Hidden+n]-expected[n]);
                maximum=fmax(maximum,d);squared+=d*d;++compared;valid=d<=0.05+0.03*fabs(expected[n]);}}
        ANEFieldfareRoutedLayerClose(layer);free(in);free(expected);[records addObject:@{@"positions":@(M),@"wall_ms":@(totalWall/reps),
            @"gpu_ms":@(totalGpu/reps),@"positions_per_second":@(M*1000.0/(totalWall/reps)),
            @"max_absolute_error":@(maximum),@"rmse":@(sqrt(squared/compared)),@"valid":@(valid)}];}
    NSDictionary *receipt=@{@"schema":@"ane-fieldfare.mpp-authentic-expert-affine.v1",@"status":valid?@"validated":@"rejected",
      @"kernel":@"MPP gate + MPP up + Metal SwiGLU + MPP down",@"matrix":@{@"gate_up":@"704x2816",@"down":@"2816x704",@"quantization":@"affine int4 group64"},
      @"measurements":records,@"error":error.localizedDescription?:@"none",
      @"claim_boundary":@"One complete authentic expert at positions 4/8/16/32/64: gate, up, SwiGLU, and down. This is the repository MPP tensor-op QMM, unlike the scalar-GEMV grouped expert kernel. Routing, top-8 reduction, and ANE carrier construction are excluded."};
    NSData *json=[NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
    fwrite(json.bytes,1,json.length,stdout);fputc('\n',stdout);if(map!=MAP_FAILED)munmap(map,st.st_size);if(fd>=0)close(fd);return valid?0:1;
}}
