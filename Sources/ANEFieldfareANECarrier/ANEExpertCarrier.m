#import "ANEExpertCarrier.h"

#import <Foundation/Foundation.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <objc/runtime.h>

enum {
    Hidden = 2816,
    Intermediate = 704,
    Banks = 2,
    BatchExperts = 4,
};

struct ANEExpertCarrier {
    uint32_t lanes;
    id model;
    id requests[Banks];
    IOSurfaceRef inputs[Banks];
    IOSurfaceRef outputs[Banks];
    dispatch_semaphore_t completed[Banks];
    BOOL mapped[Banks];
    NSString *temporaryPath;
    ANEExpertCarrierInfo info;
};

static mach_timebase_info_data_t timebase;

static double milliseconds(uint64_t ticks) {
    return (double)ticks * timebase.numer / timebase.denom / 1e6;
}

static const char *retainError(NSString *message) {
    return strdup(message.length ? message.UTF8String : "unknown ANE runtime error");
}

static bool methodHasEncoding(Class cls, SEL selector, const char *expected) {
    Method method = class_getInstanceMethod(cls, selector);
    const char *actual = method ? method_getTypeEncoding(method) : NULL;
    return actual && strcmp(actual, expected) == 0;
}

static IOSurfaceRef makeSurface(uint64_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    });
}

static NSString *expertMIL(uint32_t Positions) {
    uint32_t CarrierStride = Positions + 3 * Intermediate;
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

static NSString *carrierMIL(uint32_t lanes) {
    NSMutableString *mil = [expertMIL(lanes) mutableCopy];
    NSRange all = NSMakeRange(0, mil.length);
    // Power-of-two input scaling avoids ANE's loss of tiny products. Undo it
    // before the nonlinear gate and after the down product; the up branch
    // carries the scale through GELU(gate) * up.
    [mil replaceOccurrencesOfString:@"> gate=matmul(" withString:@"> gate_scaled=matmul(" options:0 range:all];
    all = NSMakeRange(0, mil.length);
    [mil replaceOccurrencesOfString:[NSString stringWithFormat:@"    tensor<fp16,[1,1,%u,704]> gate2=", lanes]
                         withString:[NSString stringWithFormat:@"    tensor<fp16,[1,1,%u,704]> gate=mul(x=gate_scaled,y=fp16(0.015625))[name=string(\"gate_unscale\")];\n    tensor<fp16,[1,1,%u,704]> gate2=", lanes, lanes]
                            options:0 range:all];
    all = NSMakeRange(0, mil.length);
    [mil replaceOccurrencesOfString:@"> o2=matmul(" withString:@"> o2_scaled=matmul(" options:0 range:all];
    all = NSMakeRange(0, mil.length);
    [mil replaceOccurrencesOfString:[NSString stringWithFormat:@"    tensor<fp16,[1,1,2816,%u]> ot=", lanes]
                         withString:[NSString stringWithFormat:@"    tensor<fp16,[1,1,%u,2816]> o2=mul(x=o2_scaled,y=fp16(0.015625))[name=string(\"down_unscale\")];\n    tensor<fp16,[1,1,2816,%u]> ot=", lanes, lanes]
                            options:0 range:all];
    // Keep each expert's D/F matrix axes and add a leading batch dimension.
    // External channels flatten batch and D.
    all = NSMakeRange(0, mil.length);
    [mil replaceOccurrencesOfString:@"[1,2816,1," withString:@"[1,11264,1," options:0 range:all];
    all = NSMakeRange(0, mil.length);
    [mil replaceOccurrencesOfString:@"[1,1," withString:@"[4,1," options:0 range:all];
    return mil;
}

ANEExpertCarrier *ANEExpertCarrierCreate(uint32_t lanes, const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (lanes < 64 || lanes > 512 || lanes % 32) {
            if (errorMessage) *errorMessage = retainError(@"carrier lanes must be a multiple of 32 in 64...512");
            return NULL;
        }
        mach_timebase_info(&timebase);
        void *framework = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!framework || !descriptorClass || !modelClass || !requestClass || !surfaceClass) {
            if (errorMessage) *errorMessage = retainError(@"private ANE runtime surface unavailable");
            return NULL;
        }
        NSData *mil = [carrierMIL(lanes) dataUsingEncoding:NSUTF8StringEncoding];
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, @{}, nil);
        id model = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        if (!model) {
            if (errorMessage) *errorMessage = retainError(@"ANE model construction failed");
            return NULL;
        }
        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *files = NSFileManager.defaultManager;
        [files createDirectoryAtPath:[temporary stringByAppendingPathComponent:@"weights"]
         withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];

        ANEExpertCarrier *carrier = calloc(1, sizeof(*carrier));
        carrier->lanes = lanes;
        carrier->model = model;
        carrier->temporaryPath = temporary;
        NSError *error = nil;
        uint64_t started = mach_absolute_time();
        BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        carrier->info.compileMilliseconds = milliseconds(mach_absolute_time() - started);
        if (!ok) {
            if (errorMessage) *errorMessage = retainError(error.localizedDescription ?: @"ANE compile failed");
            ANEExpertCarrierDestroy(carrier);
            return NULL;
        }
        carrier->info.compileCount += 1;
        started = mach_absolute_time();
        ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
        carrier->info.loadMilliseconds = milliseconds(mach_absolute_time() - started);
        if (!ok) {
            if (errorMessage) *errorMessage = retainError(error.localizedDescription ?: @"ANE load failed");
            carrier->model = nil;
            ANEExpertCarrierDestroy(carrier);
            return NULL;
        }
        carrier->info.loadCount += 1;
        uint64_t inputBytes = (uint64_t)BatchExperts * Hidden * (lanes + 3 * Intermediate) * sizeof(__fp16);
        uint64_t outputBytes = (uint64_t)BatchExperts * Hidden * lanes * sizeof(float);
        for (uint32_t bank = 0; bank < Banks; ++bank) {
            carrier->inputs[bank] = makeSurface(inputBytes);
            carrier->outputs[bank] = makeSurface(outputBytes);
            id input = carrier->inputs[bank] ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), carrier->inputs[bank]) : nil;
            id output = carrier->outputs[bank] ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), carrier->outputs[bank]) : nil;
            carrier->requests[bank] = input && output ?
                ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
                    requestClass,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[input], @[@0], @[output], @[@0], nil, nil, @0) : nil;
            dispatch_semaphore_t completed = dispatch_semaphore_create(0);
            carrier->completed[bank] = completed;
            if (carrier->requests[bank]) ((void (*)(id, SEL, id))objc_msgSend)(
                carrier->requests[bank], @selector(setCompletionHandler:), ^{ dispatch_semaphore_signal(completed); });
            carrier->mapped[bank] = carrier->requests[bank] &&
                ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
                    model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
                    carrier->requests[bank], YES, &error);
            if (!carrier->mapped[bank]) {
                if (errorMessage) *errorMessage = retainError(error.localizedDescription ?: @"ANE IOSurface map failed");
                ANEExpertCarrierDestroy(carrier);
                return NULL;
            }
        }
        return carrier;
    }
}

IOSurfaceRef ANEExpertCarrierInput(ANEExpertCarrier *carrier, uint32_t bank) {
    return carrier && bank < Banks ? carrier->inputs[bank] : NULL;
}

IOSurfaceRef ANEExpertCarrierOutput(ANEExpertCarrier *carrier, uint32_t bank) {
    return carrier && bank < Banks ? carrier->outputs[bank] : NULL;
}

bool ANEExpertCarrierSetOutputEvent(ANEExpertCarrier *carrier, uint32_t bank,
                                    CFTypeRef metalSharedEvent, uint64_t value,
                                    const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!carrier || bank >= Banks || !metalSharedEvent || !value) {
            if (errorMessage) *errorMessage = retainError(@"invalid output signal-event arguments");
            return false;
        }
        id metalEvent = (__bridge id)metalSharedEvent;
        id request = carrier->requests[bank];
        SEL ioSurfaceEventSelector = NSSelectorFromString(@"IOSurfaceSharedEvent");
        Class signalClass = NSClassFromString(@"_ANESharedSignalEvent");
        Class eventsClass = NSClassFromString(@"_ANESharedEvents");
        SEL signalSelector = @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:);
        SEL eventsSelector = @selector(sharedEventsWithSignalEvents:waitEvents:);
        if (!methodHasEncoding([metalEvent class], ioSurfaceEventSelector, "@16@0:8") ||
            !signalClass || !methodHasEncoding(object_getClass(signalClass), signalSelector,
                                               "@44@0:8Q16I24q28@36") ||
            !eventsClass || !methodHasEncoding(object_getClass(eventsClass), eventsSelector,
                                               "@32@0:8@16@24") ||
            !methodHasEncoding([request class], @selector(setSharedEvents:), "v24@0:8@16") ||
            !methodHasEncoding([request class], @selector(sharedEvents), "@16@0:8")) {
            if (errorMessage) *errorMessage = retainError(@"output signal-event ABI mismatch");
            return false;
        }
        id ioSurfaceEvent = ((id (*)(id, SEL))objc_msgSend)(metalEvent, ioSurfaceEventSelector);
        id signalEvent = ioSurfaceEvent ?
            ((id (*)(Class, SEL, uint64_t, uint32_t, int64_t, id))objc_msgSend)(
                signalClass, signalSelector, value, 0, 0, ioSurfaceEvent) : nil;
        id events = signalEvent ? ((id (*)(Class, SEL, id, id))objc_msgSend)(
            eventsClass, eventsSelector, @[signalEvent], @[]) : nil;
        if (!events) {
            if (errorMessage) *errorMessage = retainError(@"output signal-event construction failed");
            return false;
        }
        ((void (*)(id, SEL, id))objc_msgSend)(request, @selector(setSharedEvents:), events);
        if (((id (*)(id, SEL))objc_msgSend)(request, @selector(sharedEvents)) != events) {
            if (errorMessage) *errorMessage = retainError(@"shared-event request readback mismatch");
            return false;
        }
        return true;
    }
}

bool ANEExpertCarrierEvaluate(ANEExpertCarrier *carrier, uint32_t bank, const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!carrier || bank >= Banks) {
            if (errorMessage) *errorMessage = retainError(@"invalid ANE request bank");
            return false;
        }
        uint64_t started = mach_absolute_time();
        NSError *error = nil;
        BOOL accepted = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            carrier->model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, carrier->requests[bank], &error);
        if (!accepted) {
            if (errorMessage) *errorMessage = retainError(
                error.localizedDescription ?: @"ANE evaluate submission failed");
            return false;
        }
        if (dispatch_semaphore_wait(carrier->completed[bank],
                                    dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC))) {
            if (errorMessage) *errorMessage = retainError(@"ANE completion timeout");
            return false;
        }
        carrier->info.evaluations += 1;
        carrier->info.evaluationMilliseconds += milliseconds(mach_absolute_time() - started);
        return true;
    }
}

ANEExpertCarrierInfo ANEExpertCarrierGetInfo(const ANEExpertCarrier *carrier) {
    return carrier ? carrier->info : (ANEExpertCarrierInfo){0};
}

void ANEExpertCarrierDestroy(ANEExpertCarrier *carrier) {
    if (!carrier) return;
    @autoreleasepool {
        NSError *error = nil;
        for (uint32_t bank = 0; bank < Banks; ++bank) {
            if (carrier->mapped[bank]) ((void (*)(id, SEL, id))objc_msgSend)(
                carrier->model, @selector(unmapIOSurfacesWithRequest:), carrier->requests[bank]);
            if (carrier->requests[bank]) ((void (*)(id, SEL, id))objc_msgSend)(
                carrier->requests[bank], @selector(setCompletionHandler:), nil);
        }
        if (carrier->model) ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            carrier->model, @selector(unloadWithQoS:error:), 21, &error);
        if (carrier->temporaryPath)
            [NSFileManager.defaultManager removeItemAtPath:carrier->temporaryPath error:nil];
        for (uint32_t bank = 0; bank < Banks; ++bank) {
            if (carrier->inputs[bank]) CFRelease(carrier->inputs[bank]);
            if (carrier->outputs[bank]) CFRelease(carrier->outputs[bank]);
            carrier->requests[bank] = nil;
            carrier->completed[bank] = nil;
        }
        carrier->model = nil;
        carrier->temporaryPath = nil;
        free(carrier);
    }
}
