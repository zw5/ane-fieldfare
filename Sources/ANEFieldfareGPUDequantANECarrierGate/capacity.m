#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <math.h>
#include <stdatomic.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Isolated ownership extension: compile the exact retained implementation into
// this gate so its one loaded model and metadata-authoritative request lifecycle
// are visible without changing the production eight-bank wrapper.
#include "../ANEFieldfareRuntime/ANEProgram.m"
#define milliseconds expertMilliseconds
#define timebase expertTimebase
#include "../ANEFieldfareRuntime/ANEExpertDispatch.m"
#undef timebase
#undef milliseconds

typedef struct { dispatch_semaphore_t semaphore; _Atomic bool completed; } ExtraCompletion;

static NSString *overlapKernelSource(void){return @"#include <metal_stdlib>\nusing namespace metal;"
"constant uint H=2816,I=704,P=64,S=2176;constant uint GO=0,GS=991232,GB=1053184;"
"constant uint UO=1115136,US=2106368,UB=2168320;constant uint DO=2230272,DS=3221504,DB=3283456;"
"float bf(device const uchar*p,uint o){uint b=uint(p[o])|(uint(p[o+1])<<8);return as_type<float>(b<<16);}"
"half q(device const uchar*p,uint wo,uint so,uint bo,uint n){uint v=p[wo+(n>>1)],c=(n&1)?v>>4:v&15,g=n>>6;"
"return half(float(c)*bf(p,so+2*g)+bf(p,bo+2*g));}"
"kernel void expand8(device const uchar*a[[buffer(0)]],device half*c0[[buffer(1)]],device half*c1[[buffer(2)]],"
"device half*c2[[buffer(3)]],device half*c3[[buffer(4)]],device half*c4[[buffer(5)]],device half*c5[[buffer(6)]],"
"device half*c6[[buffer(7)]],device half*c7[[buffer(8)]],constant uint*e[[buffer(9)]],constant uint&batch[[buffer(10)]],"
"uint z[[thread_position_in_grid]]){uint E=H*S,b=z/E,n=z-b*E;if(b>=8)return;device half*c=b==0?c0:b==1?c1:b==2?c2:b==3?c3:b==4?c4:b==5?c5:b==6?c6:c7;"
"device const uchar*p=a+e[b]*3358720;uint col=n/S,x=n-col*S;if(x<P){c[n]=x?half(0):half((float(int((col*17+batch*13)%101)-50))/512.f);return;}"
"if(x<P+I){c[n]=q(p,GO,GS,GB,(x-P)*H+col);return;}if(x<P+2*I){c[n]=q(p,UO,US,UB,(x-P-I)*H+col);return;}"
"c[n]=q(p,DO,DS,DB,col*I+x-P-2*I);}";}

static bool setRequestWait(id request,id<MTLSharedEvent> event,uint64_t value){
    id io=((id(*)(id,SEL))objc_msgSend)(event,NSSelectorFromString(@"IOSurfaceSharedEvent"));
    Class signalClass=NSClassFromString(@"_ANESharedSignalEvent"),eventsClass=NSClassFromString(@"_ANESharedEvents");
    id signal=io?((id(*)(Class,SEL,uint64_t,uint32_t,int64_t,id))objc_msgSend)(signalClass,
        @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),value,0,0,io):nil;
    id wait=signal?((id(*)(id,SEL))objc_msgSend)(signal,@selector(waitEvent)):nil;
    id set=wait?((id(*)(Class,SEL,id,id))objc_msgSend)(eventsClass,@selector(sharedEventsWithSignalEvents:waitEvents:),@[],@[wait]):nil;
    if(set)((void(*)(id,SEL,id))objc_msgSend)(request,@selector(setSharedEvents:),set);return set!=nil;
}

static id<MTLCommandBuffer> prepareBatch(id<MTLCommandQueue> queue,id<MTLComputePipelineState> pso,
    id<MTLBuffer> compact,id<MTLBuffer> __strong carriers[8],const uint32_t experts[8],uint32_t batch,
    id<MTLSharedEvent> event,uint64_t value){
    id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pso];[encoder setBuffer:compact offset:0 atIndex:0];
    for(uint32_t b=0;b<8;++b)[encoder setBuffer:carriers[b] offset:0 atIndex:b+1];
    [encoder setBytes:experts length:8*4 atIndex:9];[encoder setBytes:&batch length:4 atIndex:10];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)Hidden*CarrierStride*8,1,1)
        threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,pso.maxTotalThreadsPerThreadgroup),1,1)];[encoder endEncoding];
    [command encodeSignalEvent:event value:value];return command;
}

static void makeInput(float *input,uint32_t batch){
    for(uint32_t c=0;c<Hidden;++c)input[c]=(float)((int32_t)((c*17+batch*13)%101)-50)/512.0f;
}

static bool fillSurface(IOSurfaceRef surface,const uint8_t *base,const float *input){
    if(!surface||IOSurfaceLock(surface,0,NULL)!=kIOReturnSuccess)return false;
    __fp16 *carrier=IOSurfaceGetBaseAddress(surface);bool valid=carrier&&IOSurfaceGetAllocSize(surface)>=(size_t)Hidden*CarrierStride*2;
    if(valid){memset(carrier,0x7e,(size_t)Hidden*CarrierStride*2);fillInput(carrier,input);
        fillTransposedPair(carrier,matrix(base,Intermediate,Hidden,GateWeightOffset,GateScaleOffset,GateBiasOffset),
            matrix(base,Intermediate,Hidden,UpWeightOffset,UpScaleOffset,UpBiasOffset));
        fillDown(carrier,matrix(base,Hidden,Intermediate,DownWeightOffset,DownScaleOffset,DownBiasOffset));}
    IOSurfaceUnlock(surface,0,NULL);return valid;
}

static bool oracle(ANEFieldfareRoutedLayer *layer,IOSurfaceRef output,uint32_t expert,const float *input,
                   double *worst){
    float *expected=malloc(Hidden*sizeof(float));if(!expected)return false;
    bool valid=ANEFieldfareRoutedLayerCPUExpert(layer,expert,input,expected)&&output&&
        IOSurfaceLock(output,kIOSurfaceLockReadOnly,NULL)==kIOReturnSuccess;
    if(valid){const float *actual=IOSurfaceGetBaseAddress(output);for(uint32_t c=0;valid&&c<Hidden;++c){
        double d=fabs((double)actual[(uint64_t)c*Positions]-expected[c]);*worst=fmax(*worst,d);
        valid=isfinite(actual[(uint64_t)c*Positions])&&d<=0.05+0.03*fabs(expected[c]);}
        IOSurfaceUnlock(output,kIOSurfaceLockReadOnly,NULL);}free(expected);return valid;
}

int main(int argc,const char *argv[]){@autoreleasepool{
    if(argc!=2)return 2;NSString *root=[[NSString stringWithUTF8String:argv[1]]stringByStandardizingPath];
    NSString *packed=[root stringByAppendingPathComponent:@"packed_experts/layer_00.bin"];
    int fd=open(packed.fileSystemRepresentation,O_RDONLY|O_CLOEXEC);struct stat st={0};
    bool valid=fd>=0&&fstat(fd,&st)==0&&st.st_size==(off_t)((uint64_t)ExpertStride*ExpertCount);
    uint8_t *bytes=valid?mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0):MAP_FAILED;valid=valid&&bytes!=MAP_FAILED;
    const char *createError=NULL;ANEFieldfareANEProgram *program=valid?ANEFieldfareANEProgramCreate(
        expertMIL().UTF8String,(uint64_t)Hidden*CarrierStride*2,(uint64_t)Hidden*Positions*4,8,&createError):NULL;
    valid=valid&&program;Class requestClass=NSClassFromString(@"_ANERequest"),surfaceClass=NSClassFromString(@"_ANEIOSurfaceObject");
    __strong id requests[8]={nil};IOSurfaceRef inputs[8]={0},outputs[8]={0};ExtraCompletion completions[8]={0};
    NSError *mapError=nil;uint32_t mapped=0;
    for(uint32_t x=0;valid&&x<8;++x){inputs[x]=makeSurface((uint64_t)Hidden*CarrierStride*2);
        outputs[x]=makeSurface((uint64_t)Hidden*Positions*4);
        id input=inputs[x]?((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(surfaceClass,@selector(objectWithIOSurface:),inputs[x]):nil;
        id output=outputs[x]?((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(surfaceClass,@selector(objectWithIOSurface:),outputs[x]):nil;
        requests[x]=input&&output?((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(requestClass,
          @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
          @[input],@[@0],@[output],@[@0],nil,nil,@0):nil;
        completions[x].semaphore=dispatch_semaphore_create(0);ExtraCompletion *completion=&completions[x];
        if(requests[x])((void(*)(id,SEL,id))objc_msgSend)(requests[x],@selector(setCompletionHandler:),^{
            atomic_store_explicit(&completion->completed,true,memory_order_release);dispatch_semaphore_signal(completion->semaphore);});
        BOOL ok=requests[x]&&((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(program->model,
            @selector(mapIOSurfacesWithRequest:cacheInference:error:),requests[x],YES,&mapError);
        valid=valid&&ok;if(ok)++mapped;
    }
    ANEFieldfareRoutedLayer *layer=NULL;valid=valid&&ANEFieldfareRoutedLayerOpen(root.fileSystemRepresentation,0,&layer);
    const uint32_t experts[16]={0,17,31,43,57,71,103,127,1,19,37,53,67,83,101,125};
    NSMutableArray *runs=[NSMutableArray array];double worst=0;
    const uint32_t widths[2]={9,16};
    for(uint32_t run=0;valid&&run<2;++run){uint32_t count=widths[run];float input[Hidden];makeInput(input,run+1);
        for(uint32_t b=0;b<count&&valid;++b){IOSurfaceRef in=b<8?ANEFieldfareANEProgramInput(program,b):inputs[b-8];
            IOSurfaceRef out=b<8?ANEFieldfareANEProgramOutput(program,b):outputs[b-8];
            IOSurfaceLock(out,0,NULL);memset(IOSurfaceGetBaseAddress(out),0x7e,IOSurfaceGetAllocSize(out));IOSurfaceUnlock(out,0,NULL);
            valid=fillSurface(in,bytes+(uint64_t)experts[b]*ExpertStride,input);}
        uint32_t accepted=0;
        for(uint32_t b=0;b<count&&valid;++b){NSError *submitError=nil;BOOL ok=false;
            if(b<8){const char *e=NULL;ANEFieldfareANEEvaluationTiming t={0};ok=ANEFieldfareANEProgramEvaluateBegin(program,b,&t,&e);free((void*)e);}
            else{atomic_store_explicit(&completions[b-8].completed,false,memory_order_relaxed);
                ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(program->model,
                    @selector(evaluateWithQoS:options:request:error:),21,@{},requests[b-8],&submitError);}
            if(ok)++accepted;else valid=false;}
        for(uint32_t b=0;b<count&&valid;++b){bool ok=false;if(b<8){const char *e=NULL;ANEFieldfareANEEvaluationTiming t={0};
                ok=ANEFieldfareANEProgramEvaluateWait(program,b,5*NSEC_PER_SEC,&t,&e);free((void*)e);}
            else ok=dispatch_semaphore_wait(completions[b-8].semaphore,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC))==0&&
                atomic_load_explicit(&completions[b-8].completed,memory_order_acquire);valid=valid&&ok;}
        for(uint32_t b=0;b<count&&valid;++b)valid=oracle(layer,b<8?ANEFieldfareANEProgramOutput(program,b):outputs[b-8],experts[b],input,&worst);
        [runs addObject:@{@"requested":@(count),@"mapped_total":@(8+mapped),@"submissions_accepted":@(accepted),
          @"all_completed":@(accepted==count),@"all_numerical_oracles":@(valid)}];}
    // With sixteen independent mapped carriers, A occupies the retained eight
    // banks while Metal prepares B directly into the eight dynamic IOSurfaces.
    // B is submitted before the Metal signal and is held by runtime-authored
    // wait events; there is no host semaphore between producer and consumer.
    BOOL overlapValid=valid;double overlapWorst=0,overlapWallMs=0,overlapGpuMs=0;
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError *metalError=nil;
    id<MTLLibrary> library=[device newLibraryWithSource:overlapKernelSource() options:nil error:&metalError];
    id<MTLFunction> function=[library newFunctionWithName:@"expand8"];
    id<MTLComputePipelineState> pso=function?[device newComputePipelineStateWithFunction:function error:&metalError]:nil;
    id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLSharedEvent> ready=[device newSharedEvent];
    id<MTLBuffer> compact=[device newBufferWithBytesNoCopy:bytes length:(NSUInteger)st.st_size options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> bcarriers[8]={nil};for(uint32_t b=0;b<8&&overlapValid;++b){IOSurfaceLock(inputs[b],0,NULL);
        memset(IOSurfaceGetBaseAddress(inputs[b]),0x7e,IOSurfaceGetAllocSize(inputs[b]));IOSurfaceUnlock(inputs[b],0,NULL);
        void *base=IOSurfaceGetBaseAddress(inputs[b]);
        bcarriers[b]=[device newBufferWithBytesNoCopy:base length:IOSurfaceGetAllocSize(inputs[b]) options:MTLResourceStorageModeShared deallocator:nil];
        overlapValid=bcarriers[b]&&bcarriers[b].contents==base;}
    overlapValid=overlapValid&&pso&&queue&&ready&&compact;
    uint32_t batchA=4,batchB=5;float inputA[Hidden],inputB[Hidden];makeInput(inputA,batchA);makeInput(inputB,batchB);
    for(uint32_t b=0;b<8&&overlapValid;++b){overlapValid=fillSurface(ANEFieldfareANEProgramInput(program,b),
        bytes+(uint64_t)experts[b]*ExpertStride,inputA);IOSurfaceLock(outputs[b],0,NULL);
        memset(IOSurfaceGetBaseAddress(outputs[b]),0x7e,IOSurfaceGetAllocSize(outputs[b]));IOSurfaceUnlock(outputs[b],0,NULL);}
    id ioEvent=overlapValid?((id(*)(id,SEL))objc_msgSend)(ready,NSSelectorFromString(@"IOSurfaceSharedEvent")):nil;
    Class signalClass=NSClassFromString(@"_ANESharedSignalEvent"),eventsClass=NSClassFromString(@"_ANESharedEvents");uint64_t eventValue=1;
    for(uint32_t b=0;b<8&&overlapValid;++b){id signal=((id(*)(Class,SEL,uint64_t,uint32_t,int64_t,id))objc_msgSend)(signalClass,
        @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:),eventValue,0,0,ioEvent);
        id wait=signal?((id(*)(id,SEL))objc_msgSend)(signal,@selector(waitEvent)):nil;
        id eventSet=wait?((id(*)(Class,SEL,id,id))objc_msgSend)(eventsClass,@selector(sharedEventsWithSignalEvents:waitEvents:),@[],@[wait]):nil;
        if(eventSet)((void(*)(id,SEL,id))objc_msgSend)(requests[b],@selector(setSharedEvents:),eventSet);else overlapValid=false;}
    id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pso];[encoder setBuffer:compact offset:0 atIndex:0];
    for(uint32_t b=0;b<8;++b)[encoder setBuffer:bcarriers[b] offset:0 atIndex:b+1];[encoder setBytes:experts+8 length:8*4 atIndex:9];
    [encoder setBytes:&batchB length:4 atIndex:10];[encoder dispatchThreads:MTLSizeMake((NSUInteger)Hidden*CarrierStride*8,1,1)
        threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256,pso.maxTotalThreadsPerThreadgroup),1,1)];[encoder endEncoding];
    [command encodeSignalEvent:ready value:eventValue];
    uint64_t overlapStarted=mach_absolute_time();uint32_t aAccepted=0,bAccepted=0;
    for(uint32_t b=0;b<8&&overlapValid;++b){const char *e=NULL;ANEFieldfareANEEvaluationTiming t={0};
        BOOL ok=ANEFieldfareANEProgramEvaluateBegin(program,b,&t,&e);free((void*)e);if(ok)++aAccepted;else overlapValid=false;}
    [command commit];
    for(uint32_t b=0;b<8&&overlapValid;++b){atomic_store_explicit(&completions[b].completed,false,memory_order_relaxed);NSError *e=nil;
        BOOL ok=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(program->model,@selector(evaluateWithQoS:options:request:error:),21,@{},requests[b],&e);
        if(ok)++bAccepted;else overlapValid=false;}
    for(uint32_t b=0;b<8&&overlapValid;++b){const char *e=NULL;ANEFieldfareANEEvaluationTiming t={0};
        overlapValid=ANEFieldfareANEProgramEvaluateWait(program,b,5*NSEC_PER_SEC,&t,&e);free((void*)e);}
    for(uint32_t b=0;b<8&&overlapValid;++b)overlapValid=dispatch_semaphore_wait(completions[b].semaphore,
        dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC))==0&&atomic_load_explicit(&completions[b].completed,memory_order_acquire);
    [command waitUntilCompleted];overlapWallMs=milliseconds(mach_absolute_time()-overlapStarted);
    if(command.GPUEndTime>=command.GPUStartTime)overlapGpuMs=(command.GPUEndTime-command.GPUStartTime)*1000.0;
    overlapValid=overlapValid&&command.status==MTLCommandBufferStatusCompleted&&aAccepted==8&&bAccepted==8;
    for(uint32_t b=0;b<8&&overlapValid;++b)overlapValid=oracle(layer,ANEFieldfareANEProgramOutput(program,b),experts[b],inputA,&overlapWorst)&&
        oracle(layer,outputs[b],experts[b+8],inputB,&overlapWorst);
    // Steady state: one untimed A completion, then twenty alternating B/A
    // completions. Each next ring is prepared and submitted behind an event
    // before the current ring is waited, so GPU preparation overlaps current
    // ANE consumption without reusing a live carrier.
    BOOL steadyValid=overlapValid;id<MTLBuffer> acarriers[8]={nil};
    for(uint32_t b=0;b<8&&steadyValid;++b){IOSurfaceRef s=ANEFieldfareANEProgramInput(program,b);void *base=IOSurfaceGetBaseAddress(s);
        acarriers[b]=[device newBufferWithBytesNoCopy:base length:IOSurfaceGetAllocSize(s) options:MTLResourceStorageModeShared deallocator:nil];
        steadyValid=acarriers[b]&&acarriers[b].contents==base;}
    uint64_t baseEvent=100;id<MTLCommandBuffer> initial=prepareBatch(queue,pso,compact,acarriers,experts,10,ready,baseEvent);
    [initial commit];[initial waitUntilCompleted];steadyValid=steadyValid&&initial.status==MTLCommandBufferStatusCompleted;
    uint64_t submittedAt[21]={0},completedAt[21]={0};ANEFieldfareANEEvaluationTiming baseTimings[8]={0};
    submittedAt[0]=mach_absolute_time();for(uint32_t b=0;b<8&&steadyValid;++b){const char *e=NULL;
        steadyValid=ANEFieldfareANEProgramEvaluateBegin(program,b,&baseTimings[b],&e);free((void*)e);}
    double steadyMetalGpu=0,steadyMetalSignalWall=0;uint32_t current=0;
    for(uint32_t batch=1;batch<=20&&steadyValid;++batch){bool targetExtra=(batch&1)!=0;id<MTLBuffer> __strong *target=targetExtra?bcarriers:acarriers;
        const uint32_t *targetExperts=targetExtra?experts+8:experts;id<MTLCommandBuffer> prep=prepareBatch(queue,pso,compact,target,targetExperts,10+batch,ready,baseEvent+batch);
        for(uint32_t b=0;b<8&&steadyValid;++b)steadyValid=setRequestWait(targetExtra?requests[b]:program->requests[b],ready,baseEvent+batch);
        __block _Atomic uint64_t prepCompletedAt=0;[prep addCompletedHandler:^(id<MTLCommandBuffer> value){(void)value;
            atomic_store_explicit(&prepCompletedAt,mach_absolute_time(),memory_order_release);}];
        uint64_t prepCommittedAt=mach_absolute_time();[prep commit];submittedAt[batch]=mach_absolute_time();
        if(targetExtra){for(uint32_t b=0;b<8&&steadyValid;++b){atomic_store_explicit(&completions[b].completed,false,memory_order_relaxed);NSError *e=nil;
            steadyValid=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(program->model,@selector(evaluateWithQoS:options:request:error:),21,@{},requests[b],&e);}}
        else for(uint32_t b=0;b<8&&steadyValid;++b){const char *e=NULL;ANEFieldfareANEEvaluationTiming t={0};
            steadyValid=ANEFieldfareANEProgramEvaluateBegin(program,b,&t,&e);free((void*)e);}
        if(current&1){for(uint32_t b=0;b<8&&steadyValid;++b)steadyValid=dispatch_semaphore_wait(completions[b].semaphore,
            dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC))==0;}
        else for(uint32_t b=0;b<8&&steadyValid;++b){const char *e=NULL;ANEFieldfareANEEvaluationTiming t={0};
            steadyValid=ANEFieldfareANEProgramEvaluateWait(program,b,5*NSEC_PER_SEC,&t,&e);free((void*)e);}
        completedAt[current]=mach_absolute_time();[prep waitUntilCompleted];steadyValid=steadyValid&&prep.status==MTLCommandBufferStatusCompleted;
        uint64_t prepDone=atomic_load_explicit(&prepCompletedAt,memory_order_acquire);
        if(prepDone>=prepCommittedAt)steadyMetalSignalWall+=milliseconds(prepDone-prepCommittedAt);
        if(prep.GPUEndTime>=prep.GPUStartTime)steadyMetalGpu+=(prep.GPUEndTime-prep.GPUStartTime)*1000.0;current=batch;}
    if(steadyValid){for(uint32_t b=0;b<8&&steadyValid;++b){if(current&1)steadyValid=dispatch_semaphore_wait(completions[b].semaphore,
            dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC))==0;else{const char *e=NULL;ANEFieldfareANEEvaluationTiming t={0};
            steadyValid=ANEFieldfareANEProgramEvaluateWait(program,b,5*NSEC_PER_SEC,&t,&e);free((void*)e);}}completedAt[20]=mach_absolute_time();}
    double cadenceMs=steadyValid?milliseconds(completedAt[20]-completedAt[0])/20.0:0;
    double aneLatencySum=0;for(uint32_t b=1;b<=20;++b)if(completedAt[b]>=submittedAt[b])aneLatencySum+=milliseconds(completedAt[b]-submittedAt[b]);
    // Oracle replay of the same twenty deterministic batches is intentionally
    // outside the cadence interval so host CPU reference work cannot inflate it.
    double steadyWorst=0;for(uint32_t batch=1;batch<=20&&steadyValid;++batch){bool extra=(batch&1)!=0;
        id<MTLBuffer> __strong *target=extra?bcarriers:acarriers;const uint32_t *targetExperts=extra?experts+8:experts;
        for(uint32_t b=0;b<8;++b){IOSurfaceRef s=extra?inputs[b]:ANEFieldfareANEProgramInput(program,b);IOSurfaceLock(s,0,NULL);
            memset(IOSurfaceGetBaseAddress(s),0x7e,IOSurfaceGetAllocSize(s));IOSurfaceUnlock(s,0,NULL);}
        id<MTLCommandBuffer> prep=prepareBatch(queue,pso,compact,target,targetExperts,10+batch,ready,1000+batch);[prep commit];[prep waitUntilCompleted];
        if(extra){for(uint32_t b=0;b<8&&steadyValid;++b){atomic_store_explicit(&completions[b].completed,false,memory_order_relaxed);NSError *e=nil;
            steadyValid=((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(program->model,@selector(evaluateWithQoS:options:request:error:),21,@{},requests[b],&e);}
            for(uint32_t b=0;b<8&&steadyValid;++b)steadyValid=dispatch_semaphore_wait(completions[b].semaphore,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC))==0;}
        else{for(uint32_t b=0;b<8&&steadyValid;++b){const char *e=NULL;ANEFieldfareANEEvaluationTiming t={0};steadyValid=ANEFieldfareANEProgramEvaluateBegin(program,b,&t,&e);free((void*)e);}
            for(uint32_t b=0;b<8&&steadyValid;++b){const char *e=NULL;ANEFieldfareANEEvaluationTiming t={0};steadyValid=ANEFieldfareANEProgramEvaluateWait(program,b,5*NSEC_PER_SEC,&t,&e);free((void*)e);}}
        float replayInput[Hidden];makeInput(replayInput,10+batch);for(uint32_t b=0;b<8&&steadyValid;++b)steadyValid=oracle(layer,
            extra?outputs[b]:ANEFieldfareANEProgramOutput(program,b),targetExperts[b],replayInput,&steadyWorst);}
    NSDictionary *receipt=@{@"schema":@"ane-fieldfare.dynamic-request-bank-capacity.v1",
      @"status":valid&&overlapValid&&steadyValid?@"validated":@"rejected",@"compile_count":@1,@"load_count":@1,
      @"base_wrapper_banks":@8,@"extra_requests_mapped_on_same_model":@(mapped),@"total_mapped_requests":@(8+mapped),
      @"runs":runs,@"maximum_absolute_error":@(worst),@"double_buffer_overlap":@{
        @"status":overlapValid?@"validated":@"rejected",@"a_submitted_before_metal_commit":@YES,
        @"b_submitted_before_metal_signal":@YES,@"host_semaphore_between_metal_and_ane":@NO,
        @"a_accepted":@(aAccepted),@"b_accepted":@(bAccepted),@"wall_ms":@(overlapWallMs),
        @"metal_gpu_ms":@(overlapGpuMs),@"maximum_absolute_error":@(overlapWorst)},
      @"steady_state_double_buffer":@{@"status":steadyValid?@"validated":@"rejected",@"warmup_batches":@1,
        @"timed_batches":@20,@"alternating_rings":@YES,@"cadence_ms_per_batch":@(cadenceMs),
        @"mean_request_submit_to_completion_including_event_wait_ms":@(aneLatencySum/20.0),
        @"mean_metal_commit_to_event_signal_ms":@(steadyMetalSignalWall/20.0),@"mean_metal_gpu_ms":@(steadyMetalGpu/20.0),
        @"single_ring_reference_ms":@6.388425,@"gain_percent":@((6.388425-cadenceMs)/6.388425*100.0),
        @"oracle_replay_batches":@20,@"oracle_outputs":@160,@"maximum_absolute_error":@(steadyWorst)},
      @"mapping_error":mapError.localizedDescription?:@"none",
      @"creation_error":createError?@(createError):@"none",
      @"claim_boundary":@"One ANEProgram compile/load and one _ANEInMemoryModel. Eight additional requests use the exact retained IOSurface wrapper, request metadata, completion handler, map, evaluate, and unmap lifecycle. Tests submit 9 then 16 poisoned-and-refilled carriers before waiting and check every output against an independent CPU expert oracle."};
    NSData *json=[NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
    fwrite(json.bytes,1,json.length,stdout);fputc('\n',stdout);
    for(uint32_t x=0;x<mapped;++x){((void(*)(id,SEL,id))objc_msgSend)(program->model,@selector(unmapIOSurfacesWithRequest:),requests[x]);
        ((void(*)(id,SEL,id))objc_msgSend)(requests[x],@selector(setCompletionHandler:),nil);}
    if(layer)ANEFieldfareRoutedLayerClose(layer);if(program)ANEFieldfareANEProgramDestroy(program);
    for(uint32_t x=0;x<8;++x){if(inputs[x])CFRelease(inputs[x]);if(outputs[x])CFRelease(outputs[x]);}
    if(bytes!=MAP_FAILED)munmap(bytes,(size_t)st.st_size);if(fd>=0)close(fd);free((void*)createError);return valid&&overlapValid&&steadyValid?0:1;
}}
