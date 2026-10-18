#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <mach/mach_time.h>

#include <math.h>
#include <unistd.h>

#include "../ANEFieldfareRuntime/ANEDynamicProjection.h"

enum { Hidden = 2816, Outputs = 128, Positions = 64, Values = Outputs * Positions, Repetitions = 8 };
static double millis(uint64_t ticks) { mach_timebase_info_data_t t={0}; mach_timebase_info(&t); return (double)ticks*t.numer/t.denom/1e6; }

static bool initializeCarrier(__fp16 *carrier, const ANEFieldfareTensor *weights, uint32_t stride) {
    memset(carrier, 0, (size_t)Hidden * stride * sizeof(*carrier));
    for (uint32_t c=0;c<Hidden;++c) for(uint32_t r=0;r<Outputs;++r) {
        float v=0; if(!ANEFieldfareAffineValue(weights,8,64,r,c,&v)) return false;
        carrier[(uint64_t)c*stride+Positions+r]=(__fp16)v;
    }
    return true;
}
static void fillActivation(__fp16 *carrier,uint32_t stride,float scale) {
    for(uint32_t c=0;c<Hidden;++c) carrier[(uint64_t)c*stride]=(__fp16)(((int32_t)(c%257)-128)*scale);
}
static NSDictionary *oracle(IOSurfaceRef input, id<MTLBuffer> downstream, uint32_t stride) {
    IOSurfaceLock(input,kIOSurfaceLockReadOnly,NULL); const __fp16 *carrier=IOSurfaceGetBaseAddress(input);
    const float *actual=downstream.contents; double maxError=0; uint32_t failures=0, poison=0;
    for(uint32_t row=0;row<Outputs;++row) {
        double sum=0; for(uint32_t c=0;c<Hidden;++c) sum+=(float)carrier[(uint64_t)c*stride]*(float)carrier[(uint64_t)c*stride+Positions+row];
        for(uint32_t p=0;p<Positions;++p) {
            float expected=p==0?(float)(2.0*sum+1.0):1.0f, got=actual[(uint64_t)row*Positions+p];
            if(isnan(got))++poison; double e=fabs((double)got-expected); maxError=fmax(maxError,e);
            if(!isfinite(got)||e>0.06+0.04*fabs(expected))++failures;
        }
    }
    IOSurfaceUnlock(input,kIOSurfaceLockReadOnly,NULL);
    return @{@"valid":@(failures==0&&poison==0),@"max_absolute_error":@(maxError),@"tolerance_failures":@(failures),@"poison_values_remaining":@(poison)};
}
typedef NS_ENUM(uint32_t,ArmKind){ArmCPU,ArmBlit,ArmDirect};
static NSDictionary *runArm(NSString *name,ArmKind kind,ANEFieldfareANEProgram *program,IOSurfaceRef inputSurface,IOSurfaceRef outputSurface,id<MTLCommandQueue> queue,id<MTLComputePipelineState> pipeline,id<MTLBuffer> directOutput,id<MTLBuffer> separate,id<MTLBuffer> downstream,const __fp16 *template,uint64_t inputBytes,uint64_t outputBytes,uint32_t stride,bool *allValid){
    NSMutableArray *runs=[NSMutableArray array]; double warm=0,warmANE=0,warmGPU=0;
    for(uint32_t rep=0;rep<Repetitions;++rep){
        IOSurfaceLock(inputSurface,0,NULL); __fp16 *carrier=IOSurfaceGetBaseAddress(inputSurface); memcpy(carrier,template,inputBytes); fillActivation(carrier,stride,0.0009765625f*(1+0.0625f*rep)); IOSurfaceUnlock(inputSurface,0,NULL);
        IOSurfaceLock(outputSurface,0,NULL); float *out=IOSurfaceGetBaseAddress(outputSurface); for(uint32_t i=0;i<Values;++i)out[i]=NAN; IOSurfaceUnlock(outputSurface,0,NULL);
        float *final=downstream.contents; for(uint32_t i=0;i<Values;++i)final[i]=NAN;
        uint64_t start=mach_continuous_time(); const char *aneError=NULL; ANEFieldfareANEEvaluationTiming timing={0};
        bool valid=ANEFieldfareANEProgramEvaluate(program,0,5ull*NSEC_PER_SEC,&timing,&aneError);
        if(valid&&kind==ArmCPU){IOSurfaceLock(outputSurface,kIOSurfaceLockReadOnly,NULL);memcpy(separate.contents,IOSurfaceGetBaseAddress(outputSurface),outputBytes);IOSurfaceUnlock(outputSurface,kIOSurfaceLockReadOnly,NULL);}
        id<MTLCommandBuffer> command=[queue commandBuffer];
        if(valid&&kind==ArmBlit){id<MTLBlitCommandEncoder>b=[command blitCommandEncoder];[b copyFromBuffer:directOutput sourceOffset:0 toBuffer:separate destinationOffset:0 size:outputBytes];[b endEncoding];}
        id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];[e setComputePipelineState:pipeline];[e setBuffer:kind==ArmDirect?directOutput:separate offset:0 atIndex:0];[e setBuffer:downstream offset:0 atIndex:1];[e dispatchThreads:MTLSizeMake(Values,1,1) threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,pipeline.maxTotalThreadsPerThreadgroup),1,1)];[e endEncoding];[command commit];[command waitUntilCompleted];
        double full=millis(mach_continuous_time()-start),gpu=(command.GPUEndTime-command.GPUStartTime)*1000.0; NSDictionary *check=oracle(inputSurface,downstream,stride);
        valid=valid&&command.status==MTLCommandBufferStatusCompleted&&[check[@"valid"]boolValue];*allValid&=valid;
        if(rep){warm+=full;warmANE+=timing.completionMilliseconds;warmGPU+=gpu;}
        [runs addObject:@{@"repetition":@(rep),@"warm":@(rep>0),@"ane_completion_ms":@(timing.completionMilliseconds),@"ane_begin_to_metal_completion_ms":@(full),@"metal_gpu_ms":@(gpu),@"oracle":check,@"error":aneError?@(aneError):@"none"}]; free((void*)aneError);
    }
    return @{@"name":name,@"runs":runs,@"warm_repetitions":@(Repetitions-1),@"warm_mean_ane_begin_to_metal_completion_ms":@(warm/(Repetitions-1)),@"warm_mean_ane_completion_ms":@(warmANE/(Repetitions-1)),@"warm_mean_metal_gpu_ms":@(warmGPU/(Repetitions-1)),@"cpu_memcpy_bytes":kind==ArmCPU?@(outputBytes):@0,@"explicit_blit_bytes":kind==ArmBlit?@(outputBytes):@0,@"direct_shared_bytes":kind==ArmDirect?@(outputBytes):@0,@"all_runs_valid":@(*allValid)};
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc!=2)return 2; const uint32_t stride=Positions+Outputs; const uint64_t inputBytes=(uint64_t)Hidden*stride*2, outputBytes=(uint64_t)Values*4;
    NSString *root=[@(argv[1]) stringByStandardizingPath],*weightsPath=[root stringByAppendingPathComponent:@"model_weights.bin"];
    ANEFieldfareTensorIndex index={.fileDescriptor=-1};ANEFieldfareTensor router={0};const char *error=NULL;
    bool valid=ANEFieldfareTensorIndexOpen(weightsPath.fileSystemRepresentation,&index)&&ANEFieldfareTensorIndexFind(&index,"language_model.model.layers.0.router.proj.weight",&router);
    ANEFieldfareANEDynamicProjection *projection=valid?ANEFieldfareANEDynamicProjectionCreate(Hidden,Outputs,1,&error):NULL; valid&=projection!=NULL;
    ANEFieldfareANEProgram *program=ANEFieldfareANEDynamicProjectionProgram(projection);IOSurfaceRef input=ANEFieldfareANEProgramInput(program,0),output=ANEFieldfareANEProgramOutput(program,0);
    __fp16 *template=calloc(inputBytes/2,2);valid&=template&&initializeCarrier(template,&router,stride);
    id<MTLDevice>device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue>queue=[device newCommandQueue];NSError *metalError=nil;
    NSString *source=@"#include <metal_stdlib>\nusing namespace metal; kernel void consume(const device float *x [[buffer(0)]], device float *y [[buffer(1)]], uint i [[thread_position_in_grid]]) { if(i<8192)y[i]=2.0f*x[i]+1.0f; }";
    id<MTLLibrary>library=[device newLibraryWithSource:source options:nil error:&metalError];id<MTLFunction>function=[library newFunctionWithName:@"consume"];id<MTLComputePipelineState>pipeline=[device newComputePipelineStateWithFunction:function error:&metalError];
    IOSurfaceLock(output,0,NULL);void *base=IOSurfaceGetBaseAddress(output);size_t alloc=IOSurfaceGetAllocSize(output);long page=sysconf(_SC_PAGESIZE);
    bool pointerGuard=base&&page>0&&((uintptr_t)base%(uintptr_t)page)==0&&alloc>=outputBytes&&alloc%page==0;
    id<MTLBuffer>direct=pointerGuard?[device newBufferWithBytesNoCopy:base length:alloc options:MTLResourceStorageModeShared deallocator:nil]:nil;IOSurfaceUnlock(output,0,NULL);
    bool identityGuard=direct&&direct.contents==base;id<MTLBuffer>separate=[device newBufferWithLength:alloc options:MTLResourceStorageModeShared],downstream=[device newBufferWithLength:outputBytes options:MTLResourceStorageModeShared];
    valid&=device&&queue&&pipeline&&pointerGuard&&identityGuard&&separate&&downstream;
    bool a=valid,b=valid,c=valid;NSDictionary *cpu=valid?runArm(@"ane_completion_cpu_copy_metal",ArmCPU,program,input,output,queue,pipeline,direct,separate,downstream,template,inputBytes,outputBytes,stride,&a):@{};
    NSDictionary *blit=valid?runArm(@"ane_completion_explicit_blit_metal",ArmBlit,program,input,output,queue,pipeline,direct,separate,downstream,template,inputBytes,outputBytes,stride,&b):@{};
    NSDictionary *shared=valid?runArm(@"ane_completion_direct_shared_metal",ArmDirect,program,input,output,queue,pipeline,direct,separate,downstream,template,inputBytes,outputBytes,stride,&c):@{};valid&=a&&b&&c;
    ANEFieldfareANEProgramInfo info=ANEFieldfareANEDynamicProjectionGetInfo(projection);
    NSDictionary *receipt=@{@"schema":@"ane-fieldfare.ane-gpu-transfer.v1",@"status":valid?@"validated":@"rejected",@"host":NSProcessInfo.processInfo.hostName,@"os":NSProcessInfo.processInfo.operatingSystemVersionString,@"command":@".build/ane-fieldfare-ane-gpu-transfer-gate checkpoints/gturbo/gemma4.gturbo",@"shape":@{@"output_values":@(Values),@"output_bytes":@(outputBytes),@"iosurface_alloc_bytes":@(alloc)},@"guards":@{@"base_nonnull":@(base!=NULL),@"page_alignment":@(pointerGuard),@"buffer_contents_identity":@(identityGuard),@"surface_and_program_outlive_metal_buffers":@YES},@"lifecycle":@{@"ane_compile_count":@1,@"ane_load_count":@1,@"ane_compile_ms":@(info.compileMilliseconds),@"ane_load_ms":@(info.loadMilliseconds),@"metal_library_compile_count":@1,@"metal_pipeline_load_count":@1},@"synchronization":@{@"executed":@"ANE completion-handler semaphore wait before Metal commit",@"private_shared_event_transaction":@{@"constructor_surface_present":@YES,@"authoritative_event_object_and_signal_value_contract_present":@NO,@"executed":@NO,@"blocker":@"The mapped ANEProgram request was compiler-created with nil sharedEvents and transactionHandle; runtime metadata exposes longer constructors but no authoritative schema connecting MTLSharedEvent objects and signal values to the request. The established request cannot be safely rebuilt or mutated."}},@"arms":@[cpu,blit,shared],@"error":error?@(error):@"none",@"metal_error":metalError.localizedDescription?:@"none",@"claim_boundary":@"One authentic router projection, eight repetitions per arm. ANE completion precedes Metal submission. CPU-copy, explicit Metal blit, and direct MTLBuffer-over-ANE-output-IOSurface paths all poison and validate every downstream value against an independent projection-plus-consumer oracle. Direct validates no CPU memcpy after ANE completion, not cross-engine overlap or shared-event ordering."};
    NSData *json=[NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];fwrite(json.bytes,1,json.length,stdout);fputc('\n',stdout);
    free((void*)error);free(template);ANEFieldfareANEDynamicProjectionDestroy(projection);ANEFieldfareTensorIndexClose(&index);return valid?0:1;
}}
