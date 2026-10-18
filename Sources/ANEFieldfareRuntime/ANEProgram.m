#import "ANEProgram.h"

#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <stdatomic.h>

typedef struct {
    dispatch_semaphore_t semaphore;
    _Atomic uint64_t completedAt;
    _Atomic uint32_t completionOrdinal;
    uint64_t startedAt;
    double enqueueMilliseconds;
    _Atomic bool pending;
} CompletionState;

typedef enum : uint32_t {
    BankIdle = 0,
    BankOrdinaryEvaluationInFlight,
    BankMappingMutation,
    BankCatalogOwnedIdle,
    BankCatalogEvaluationInFlight,
} BankOperationState;

struct ANEFieldfareANEProgram {
    id model;
    id requests[8];
    id configuredPerfStats[8];
    IOSurfaceRef perfStatsSurfaces[8];
    IOSurfaceRef inputs[8];
    IOSurfaceRef outputs[8];
    CompletionState completions[8];
    // Catalog ownership lasts from successful construction through full
    // unmap, preventing the original request or a second catalog from racing
    // on the shared input/output surfaces.
    _Atomic uint32_t bankOperationState[8];
    _Atomic uint32_t currentAsyncRequestsInFlight;
    _Atomic uint32_t maximumAsyncRequestsInFlight;
    _Atomic uint32_t completionCounter;
    _Atomic uint32_t sharedRequestSetCount;
    _Atomic bool destroyRequested;
    BOOL mapped[8];
    NSString *temporaryPath;
    ANEFieldfareANEProgramInfo info;
};

struct ANEFieldfareANESharedSurfaceRequestSet {
    ANEFieldfareANEProgram *program;
    uint32_t sourceBank;
    id requests[128];
    BOOL mapped[128];
    uint32_t requestCount;
    uint32_t mappedCount;
    bool ownsBank;
    CompletionState completion;
    _Atomic bool inFlight;
    _Atomic uint32_t completionCounter;
};

static mach_timebase_info_data_t timebase;
static _Atomic uint32_t programCreateOrdinal;

static double milliseconds(uint64_t ticks) {
    return (double)ticks * timebase.numer / timebase.denom / 1e6;
}

static IOSurfaceRef makeSurface(uint64_t bytes) {
    if (!bytes || bytes > NSUIntegerMax) return NULL;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0,
    });
}

static uint64_t compilerCarrierBytes(NSDictionary *entry) {
    uint64_t batchStride = [entry[@"BatchStride"] unsignedLongLongValue];
    uint64_t batches = MAX(1ULL, [entry[@"Batches"] unsignedLongLongValue]);
    return batchStride && batches <= UINT64_MAX / batchStride ? batchStride * batches : 0;
}

static NSUInteger firstCompilerIndex(id indices) {
    if ([indices isKindOfClass:NSIndexSet.class]) return [(NSIndexSet *)indices firstIndex];
    if ([indices isKindOfClass:NSArray.class] && [indices count])
        return [[indices firstObject] unsignedIntegerValue];
    return NSNotFound;
}

static NSUInteger compilerIndexCount(id indices) {
    if ([indices isKindOfClass:NSIndexSet.class]) return [(NSIndexSet *)indices count];
    if ([indices isKindOfClass:NSArray.class]) return [(NSArray *)indices count];
    return 0;
}

typedef struct {
    uint32_t procedureIndex;
    NSUInteger inputIndex;
    NSUInteger outputIndex;
    uint64_t inputBytes;
    uint64_t outputBytes;
} CompilerProcedureMetadata;

static bool resolveCompilerProcedureMetadata(id model, NSString *name,
                                             CompilerProcedureMetadata *metadata,
                                             NSString **failure) {
    NSDictionary *attributes = ((id (*)(id, SEL))objc_msgSend)(
        model, @selector(modelAttributes));
    NSDictionary *nameToID = attributes[@"ANEFModelDescription"]
                                       [@"kANEFModelProcedureNameToIDMapKey"];
    NSArray *networkStatus = attributes[@"NetworkStatusList"];
    NSNumber *identifier = nameToID[name];
    NSDictionary *status = nil;
    for (NSDictionary *candidate in networkStatus)
        if ([candidate[@"Name"] isEqual:name]) { status = candidate; break; }
    id backingModel = ((id (*)(id, SEL))objc_msgSend)(model, @selector(model));
    id inputIndices = identifier ? ((id (*)(id, SEL, unsigned int))objc_msgSend)(
        backingModel, @selector(inputSymbolIndicesForProcedureIndex:),
        identifier.unsignedIntValue) : nil;
    id outputIndices = identifier ? ((id (*)(id, SEL, unsigned int))objc_msgSend)(
        backingModel, @selector(outputSymbolIndicesForProcedureIndex:),
        identifier.unsignedIntValue) : nil;
    NSArray *liveInputs = status[@"LiveInputList"];
    NSArray *liveOutputs = status[@"LiveOutputList"];
    uint64_t inputBytes = compilerCarrierBytes(liveInputs.firstObject);
    uint64_t outputBytes = compilerCarrierBytes(liveOutputs.firstObject);
    NSUInteger inputIndex = firstCompilerIndex(inputIndices);
    NSUInteger outputIndex = firstCompilerIndex(outputIndices);
    if (!identifier || !status || compilerIndexCount(inputIndices) != 1 ||
        compilerIndexCount(outputIndices) != 1 || liveInputs.count != 1 ||
        liveOutputs.count != 1 || inputIndex == NSNotFound ||
        outputIndex == NSNotFound || !inputBytes || !outputBytes) {
        if (failure) *failure = identifier
            ? @"compiler procedure carrier metadata is incomplete or not single-carrier"
            : @"compiler procedure name is absent from the loaded artifact";
        return false;
    }
    if (metadata) *metadata = (CompilerProcedureMetadata){
        .procedureIndex = identifier.unsignedIntValue,
        .inputIndex = inputIndex,
        .outputIndex = outputIndex,
        .inputBytes = inputBytes,
        .outputBytes = outputBytes,
    };
    return true;
}

static void setCompletionHandler(ANEFieldfareANEProgram *program, uint32_t bank,
                                 id request) {
    CompletionState *completion = &program->completions[bank];
    void (^handler)(void) = ^{
        atomic_store_explicit(&completion->completedAt, mach_absolute_time(), memory_order_release);
        atomic_store_explicit(&completion->completionOrdinal,
                              atomic_fetch_add_explicit(&program->completionCounter, 1,
                                                        memory_order_relaxed) + 1,
                              memory_order_release);
        atomic_fetch_sub_explicit(&program->currentAsyncRequestsInFlight, 1,
                                  memory_order_relaxed);
        dispatch_semaphore_signal(completion->semaphore);
    };
    if (request) ((void (*)(id, SEL, id))objc_msgSend)(
        request, @selector(setCompletionHandler:), handler);
}

static const char *retainError(NSString *message) {
    return message.length ? strdup(message.UTF8String) : strdup("unknown ANE runtime error");
}

static bool methodHasEncoding(Class cls, SEL selector, const char *expected) {
    Method method = class_getInstanceMethod(cls, selector);
    const char *actual = method ? method_getTypeEncoding(method) : NULL;
    return actual && strcmp(actual, expected) == 0;
}

bool ANEFieldfareANEProgramSetRequestSharedEvents(
    ANEFieldfareANEProgram *program, uint32_t bank, CFTypeRef sharedEvents,
    const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!program || bank >= program->info.requestBankCount || !program->requests[bank] ||
            !sharedEvents) {
            if (errorMessage) *errorMessage = retainError(@"invalid shared-event request arguments");
            return false;
        }
        Class cls = [program->requests[bank] class];
        if (!methodHasEncoding(cls, @selector(setSharedEvents:), "v24@0:8@16") ||
            !methodHasEncoding(cls, @selector(sharedEvents), "@16@0:8")) {
            if (errorMessage) *errorMessage = retainError(@"shared-event request ABI mismatch");
            return false;
        }
        id object = (__bridge id)sharedEvents;
        ((void (*)(id, SEL, id))objc_msgSend)(
            program->requests[bank], @selector(setSharedEvents:), object);
        id readback = ((id (*)(id, SEL))objc_msgSend)(
            program->requests[bank], @selector(sharedEvents));
        if (readback != object) {
            if (errorMessage) *errorMessage = retainError(@"shared-event request readback mismatch");
            return false;
        }
        return true;
    }
}

bool ANEFieldfareANEProgramSetOutputSignalEvent(
    ANEFieldfareANEProgram *program, uint32_t bank, CFTypeRef metalSharedEvent,
    uint32_t outputSymbolIndex, uint64_t signalValue, const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!metalSharedEvent || !signalValue) {
            if (errorMessage) *errorMessage = retainError(@"invalid output signal-event arguments");
            return false;
        }
        id metalEvent = (__bridge id)metalSharedEvent;
        SEL ioSurfaceEventSelector = NSSelectorFromString(@"IOSurfaceSharedEvent");
        Class signalClass = NSClassFromString(@"_ANESharedSignalEvent");
        Class eventsClass = NSClassFromString(@"_ANESharedEvents");
        SEL signalSelector = @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:);
        SEL eventsSelector = @selector(sharedEventsWithSignalEvents:waitEvents:);
        if (!methodHasEncoding([metalEvent class], ioSurfaceEventSelector, "@16@0:8") ||
            !signalClass || !methodHasEncoding(object_getClass(signalClass), signalSelector,
                                               "@44@0:8Q16I24q28@36") ||
            !eventsClass || !methodHasEncoding(object_getClass(eventsClass), eventsSelector,
                                               "@32@0:8@16@24")) {
            if (errorMessage) *errorMessage = retainError(@"output signal-event ABI mismatch");
            return false;
        }
        id ioSurfaceEvent = ((id (*)(id, SEL))objc_msgSend)(metalEvent, ioSurfaceEventSelector);
        id signalEvent = ioSurfaceEvent ?
            ((id (*)(Class, SEL, uint64_t, uint32_t, int64_t, id))objc_msgSend)(
                signalClass, signalSelector, signalValue, outputSymbolIndex, 0, ioSurfaceEvent) : nil;
        id events = signalEvent ? ((id (*)(Class, SEL, id, id))objc_msgSend)(
            eventsClass, eventsSelector, @[signalEvent], @[]) : nil;
        if (!events) {
            if (errorMessage) *errorMessage = retainError(@"output signal-event construction failed");
            return false;
        }
        return ANEFieldfareANEProgramSetRequestSharedEvents(
            program, bank, (__bridge CFTypeRef)events, errorMessage);
    }
}

bool ANEFieldfareANEProgramSetInputWaitEvent(
    ANEFieldfareANEProgram *program, uint32_t bank, CFTypeRef metalSharedEvent,
    uint64_t waitValue, const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!metalSharedEvent || !waitValue) {
            if (errorMessage) *errorMessage = retainError(@"invalid input wait-event arguments");
            return false;
        }
        id metalEvent = (__bridge id)metalSharedEvent;
        SEL ioSurfaceEventSelector = NSSelectorFromString(@"IOSurfaceSharedEvent");
        Class signalClass = NSClassFromString(@"_ANESharedSignalEvent");
        Class eventsClass = NSClassFromString(@"_ANESharedEvents");
        SEL signalSelector = @selector(signalEventWithValue:symbolIndex:eventType:sharedEvent:);
        SEL waitSelector = @selector(waitEvent);
        SEL eventsSelector = @selector(sharedEventsWithSignalEvents:waitEvents:);
        if (!methodHasEncoding([metalEvent class], ioSurfaceEventSelector, "@16@0:8") ||
            !signalClass || !methodHasEncoding(object_getClass(signalClass), signalSelector,
                                               "@44@0:8Q16I24q28@36") ||
            !methodHasEncoding(signalClass, waitSelector, "@16@0:8") ||
            !eventsClass || !methodHasEncoding(object_getClass(eventsClass), eventsSelector,
                                               "@32@0:8@16@24")) {
            if (errorMessage) *errorMessage = retainError(@"input wait-event ABI mismatch");
            return false;
        }
        id ioSurfaceEvent = ((id (*)(id, SEL))objc_msgSend)(metalEvent, ioSurfaceEventSelector);
        id signalTemplate = ioSurfaceEvent ?
            ((id (*)(Class, SEL, uint64_t, uint32_t, int64_t, id))objc_msgSend)(
                signalClass, signalSelector, waitValue, 0, 0, ioSurfaceEvent) : nil;
        id waitEvent = signalTemplate ? ((id (*)(id, SEL))objc_msgSend)(
            signalTemplate, waitSelector) : nil;
        id events = waitEvent ? ((id (*)(Class, SEL, id, id))objc_msgSend)(
            eventsClass, eventsSelector, @[], @[waitEvent]) : nil;
        if (!events) {
            if (errorMessage) *errorMessage = retainError(@"input wait-event construction failed");
            return false;
        }
        return ANEFieldfareANEProgramSetRequestSharedEvents(
            program, bank, (__bridge CFTypeRef)events, errorMessage);
    }
}

CFTypeRef ANEFieldfareANEProgramCopyRequestDescription(
    ANEFieldfareANEProgram *program, uint32_t bank) {
    @autoreleasepool {
        if (!program || bank >= program->info.requestBankCount || !program->requests[bank]) return NULL;
        NSString *description = [program->requests[bank] description];
        return description ? CFBridgingRetain([description copy]) : NULL;
    }
}

// Kept out of ANEProgram.h on purpose. The retained AOT rebind gates declare
// this evidence-only symbol locally; product code selects compiler procedures
// through the ownership-checked shared-surface request catalog below.
bool ANEFieldfareANEProgramRebindRequestBankToProcedureEvidence(
    ANEFieldfareANEProgram *program, uint32_t bank, const char *procedureName,
    const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!program || bank >= program->info.requestBankCount ||
            !program->model || !program->requests[bank] ||
            !program->inputs[bank] || !program->outputs[bank] ||
            !procedureName) {
            if (errorMessage) *errorMessage = retainError(
                @"invalid evidence rebind arguments");
            return false;
        }
        uint32_t expectedBankState = BankIdle;
        if (!atomic_compare_exchange_strong_explicit(
                &program->bankOperationState[bank], &expectedBankState,
                BankMappingMutation,
                memory_order_acq_rel, memory_order_relaxed)) {
            if (errorMessage) *errorMessage = retainError(
                @"request bank is not idle for evidence rebind");
            return false;
        }

        NSString *name = [NSString stringWithUTF8String:procedureName];
        CompilerProcedureMetadata metadata = {0};
        NSString *failure = nil;
        uint64_t inputBytes = IOSurfaceGetWidth(program->inputs[bank]);
        uint64_t outputBytes = IOSurfaceGetWidth(program->outputs[bank]);
        if (!name || !resolveCompilerProcedureMetadata(
                program->model, name, &metadata, &failure) ||
            metadata.inputBytes != inputBytes ||
            metadata.outputBytes != outputBytes) {
            atomic_store_explicit(&program->bankOperationState[bank], BankIdle,
                                  memory_order_release);
            if (errorMessage) *errorMessage = retainError(
                failure ?: @"evidence rebind carrier size mismatch");
            return false;
        }

        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        id input = surfaceClass ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), program->inputs[bank]) : nil;
        id output = surfaceClass ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), program->outputs[bank]) : nil;
        id replacement = requestClass && input && output ?
            ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
                requestClass,
                @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                @[input], @[@(metadata.inputIndex)], @[output],
                @[@(metadata.outputIndex)], nil, program->configuredPerfStats[bank],
                @(metadata.procedureIndex)) : nil;
        if (!replacement) {
            atomic_store_explicit(&program->bankOperationState[bank], BankIdle,
                                  memory_order_release);
            if (errorMessage) *errorMessage = retainError(
                @"evidence rebind request construction failed");
            return false;
        }

        id original = program->requests[bank];
        BOOL originalMapped = program->mapped[bank];
        NSError *error = nil;
        if (originalMapped) ((void (*)(id, SEL, id))objc_msgSend)(
            program->model, @selector(unmapIOSurfacesWithRequest:), original);
        setCompletionHandler(program, bank, replacement);
        BOOL mapped = ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
            program->model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
            replacement, YES, &error);
        if (!mapped) {
            ((void (*)(id, SEL, id))objc_msgSend)(
                replacement, @selector(setCompletionHandler:), nil);
            NSError *restoreError = nil;
            BOOL restored = !originalMapped ||
                ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
                    program->model,
                    @selector(mapIOSurfacesWithRequest:cacheInference:error:),
                    original, YES, &restoreError);
            program->mapped[bank] = originalMapped && restored;
            atomic_store_explicit(&program->bankOperationState[bank], BankIdle,
                                  memory_order_release);
            if (errorMessage) *errorMessage = retainError(restored
                ? (error.localizedDescription ?: @"evidence rebind map failed")
                : [NSString stringWithFormat:@"evidence rebind and rollback failed: %@; %@",
                    error.localizedDescription ?: @"map failed",
                    restoreError.localizedDescription ?: @"rollback failed"]);
            return false;
        }

        ((void (*)(id, SEL, id))objc_msgSend)(
            original, @selector(setCompletionHandler:), nil);
        program->requests[bank] = replacement;
        program->mapped[bank] = YES;
        atomic_store_explicit(&program->bankOperationState[bank], BankIdle,
                              memory_order_release);
        return true;
    }
}

static void releaseSharedRequestSetProgram(ANEFieldfareANEProgram *program) {
    uint32_t previous = atomic_fetch_sub_explicit(
        &program->sharedRequestSetCount, 1, memory_order_acq_rel);
    if (previous == 1 && atomic_load_explicit(
            &program->destroyRequested, memory_order_acquire))
        ANEFieldfareANEProgramDestroy(program);
}

ANEFieldfareANESharedSurfaceRequestSet *
ANEFieldfareANEProgramCreateSharedSurfaceRequestSet(
    ANEFieldfareANEProgram *program, uint32_t sourceBank,
    const char *const *procedureNames, uint32_t procedureCount,
    uint32_t *mappedBeforeFailure,
    const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (mappedBeforeFailure) *mappedBeforeFailure = 0;
        if (!program || sourceBank >= program->info.requestBankCount ||
            !program->model || !program->inputs[sourceBank] ||
            !program->outputs[sourceBank] || !procedureNames ||
            !procedureCount || procedureCount > 128) {
            if (errorMessage) *errorMessage = retainError(
                @"invalid shared-surface request-set arguments");
            return NULL;
        }
        if (atomic_load_explicit(&program->destroyRequested, memory_order_acquire)) {
            if (errorMessage) *errorMessage = retainError(
                @"ANE program destruction is pending");
            return NULL;
        }
        uint32_t expectedBankState = BankIdle;
        if (!atomic_compare_exchange_strong_explicit(
                &program->bankOperationState[sourceBank], &expectedBankState,
                BankCatalogOwnedIdle,
                memory_order_acq_rel, memory_order_relaxed)) {
            if (errorMessage) *errorMessage = retainError(
                @"shared source bank is not idle");
            return NULL;
        }
        // From this point through set destruction, program teardown defers.
        // As with the rest of ANEProgram, the initial API call and a concurrent
        // destroy require external ownership synchronization.
        atomic_fetch_add_explicit(&program->sharedRequestSetCount, 1,
                                  memory_order_acq_rel);
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!requestClass || !surfaceClass) {
            atomic_store_explicit(&program->bankOperationState[sourceBank], BankIdle,
                                  memory_order_release);
            releaseSharedRequestSetProgram(program);
            if (errorMessage) *errorMessage = retainError(
                @"shared-surface request classes are unavailable");
            return NULL;
        }
        ANEFieldfareANESharedSurfaceRequestSet *set = calloc(1, sizeof(*set));
        if (!set) {
            atomic_store_explicit(&program->bankOperationState[sourceBank], BankIdle,
                                  memory_order_release);
            releaseSharedRequestSetProgram(program);
            return NULL;
        }
        set->program = program;
        set->sourceBank = sourceBank;
        set->ownsBank = true;
        set->requestCount = procedureCount;
        set->completion.semaphore = dispatch_semaphore_create(0);
        uint64_t inputBytes = IOSurfaceGetWidth(program->inputs[sourceBank]);
        uint64_t outputBytes = IOSurfaceGetWidth(program->outputs[sourceBank]);
        id input = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), program->inputs[sourceBank]);
        id output = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), program->outputs[sourceBank]);
        NSString *failure = nil;
        for (uint32_t index = 0; index < procedureCount; ++index) {
            NSString *name = procedureNames[index]
                ? [NSString stringWithUTF8String:procedureNames[index]] : nil;
            CompilerProcedureMetadata metadata = {0};
            if (!name || !resolveCompilerProcedureMetadata(
                    program->model, name, &metadata, &failure) ||
                metadata.inputBytes != inputBytes ||
                metadata.outputBytes != outputBytes) {
                if (!failure) failure = @"shared-surface procedure carrier size mismatch";
                break;
            }
            id request = input && output ?
                ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
                    requestClass,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[input], @[@(metadata.inputIndex)], @[output],
                    @[@(metadata.outputIndex)], nil, nil,
                    @(metadata.procedureIndex)) : nil;
            if (!request) { failure = @"shared-surface request construction failed"; break; }
            set->requests[index] = request;
            CompletionState *completion = &set->completion;
            ANEFieldfareANESharedSurfaceRequestSet *rawSet = set;
            void (^handler)(void) = ^{
                atomic_store_explicit(&completion->completedAt, mach_absolute_time(),
                                      memory_order_release);
                atomic_store_explicit(&completion->completionOrdinal,
                    atomic_fetch_add_explicit(&rawSet->completionCounter, 1,
                                              memory_order_relaxed) + 1,
                    memory_order_release);
                dispatch_semaphore_signal(completion->semaphore);
            };
            ((void (*)(id, SEL, id))objc_msgSend)(
                request, @selector(setCompletionHandler:), handler);
            NSError *mapError = nil;
            BOOL mapped = ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
                program->model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
                request, YES, &mapError);
            if (!mapped) {
                failure = mapError.localizedDescription ?:
                    @"shared-surface request map failed";
                ((void (*)(id, SEL, id))objc_msgSend)(
                    request, @selector(setCompletionHandler:), nil);
                set->requests[index] = nil;
                break;
            }
            set->mapped[index] = YES;
            ++set->mappedCount;
            if (mappedBeforeFailure) *mappedBeforeFailure = set->mappedCount;
        }
        if (set->mappedCount != procedureCount) {
            uint32_t ignored = 0;
            ANEFieldfareANESharedSurfaceRequestSetUnmapAll(set, &ignored, NULL);
            if (errorMessage) *errorMessage = retainError(
                failure ?: @"shared-surface request-set construction failed");
            free(set);
            releaseSharedRequestSetProgram(program);
            return NULL;
        }
        return set;
    }
}

uint32_t ANEFieldfareANESharedSurfaceRequestSetMappedCount(
    ANEFieldfareANESharedSurfaceRequestSet *set) {
    return set ? set->mappedCount : 0;
}

CFTypeRef ANEFieldfareANESharedSurfaceRequestSetCopyRequestDescription(
    ANEFieldfareANESharedSurfaceRequestSet *set, uint32_t requestIndex) {
    if (!set || requestIndex >= set->requestCount || !set->requests[requestIndex])
        return NULL;
    NSString *description = [set->requests[requestIndex] description];
    return description ? CFBridgingRetain([description copy]) : NULL;
}

bool ANEFieldfareANESharedSurfaceRequestSetEvaluate(
    ANEFieldfareANESharedSurfaceRequestSet *set, uint32_t requestIndex,
    uint64_t timeoutNanoseconds, ANEFieldfareANEEvaluationTiming *timing,
    const char **errorMessage) {
    ANEFieldfareANEEvaluationTiming beginTiming = {0};
    if (!ANEFieldfareANESharedSurfaceRequestSetEvaluateBegin(
            set, requestIndex, &beginTiming, errorMessage)) return false;
    bool completed = ANEFieldfareANESharedSurfaceRequestSetEvaluateWait(
        set, timeoutNanoseconds, timing, errorMessage);
    if (timing) timing->enqueueMilliseconds = beginTiming.enqueueMilliseconds;
    return completed;
}

bool ANEFieldfareANESharedSurfaceRequestSetEvaluateBegin(
    ANEFieldfareANESharedSurfaceRequestSet *set, uint32_t requestIndex,
    ANEFieldfareANEEvaluationTiming *timing, const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!set || requestIndex >= set->requestCount ||
            !set->mapped[requestIndex] || !set->ownsBank) {
            if (errorMessage) *errorMessage = retainError(
                @"invalid shared-surface request selection");
            return false;
        }
        bool expected = false;
        if (!atomic_compare_exchange_strong_explicit(
                &set->inFlight, &expected, true,
                memory_order_acq_rel, memory_order_relaxed)) {
            if (errorMessage) *errorMessage = retainError(
                @"shared-surface request set is already in flight");
            return false;
        }
        uint32_t expectedBankState = BankCatalogOwnedIdle;
        if (!atomic_compare_exchange_strong_explicit(
                &set->program->bankOperationState[set->sourceBank],
                &expectedBankState, BankCatalogEvaluationInFlight,
                memory_order_acq_rel,
                memory_order_relaxed)) {
            atomic_store_explicit(&set->inFlight, false, memory_order_release);
            if (errorMessage) *errorMessage = retainError(
                @"shared carrier bank is not owned and idle");
            return false;
        }
        CompletionState *completion = &set->completion;
        atomic_store_explicit(&completion->pending, true, memory_order_release);
        atomic_store_explicit(&completion->completedAt, 0, memory_order_relaxed);
        atomic_store_explicit(&completion->completionOrdinal, 0, memory_order_relaxed);
        completion->startedAt = mach_absolute_time();
        NSError *error = nil;
        BOOL accepted = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            set->program->model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, set->requests[requestIndex], &error);
        uint64_t enqueued = mach_absolute_time();
        completion->enqueueMilliseconds = milliseconds(enqueued - completion->startedAt);
        if (!accepted) {
            atomic_store_explicit(&completion->pending, false, memory_order_release);
            atomic_store_explicit(&set->inFlight, false, memory_order_release);
            atomic_store_explicit(
                &set->program->bankOperationState[set->sourceBank],
                BankCatalogOwnedIdle,
                memory_order_release);
            if (errorMessage) *errorMessage = retainError(
                error.localizedDescription ?: @"shared-surface evaluation submission failed");
            return false;
        }
        if (timing) *timing = (ANEFieldfareANEEvaluationTiming){
            .enqueueMilliseconds = completion->enqueueMilliseconds,
            .inFlightAfterEnqueue = 1,
        };
        return true;
    }
}

bool ANEFieldfareANESharedSurfaceRequestSetEvaluateWait(
    ANEFieldfareANESharedSurfaceRequestSet *set, uint64_t timeoutNanoseconds,
    ANEFieldfareANEEvaluationTiming *timing, const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!set || !timeoutNanoseconds ||
            !atomic_load_explicit(&set->inFlight, memory_order_acquire)) {
            if (errorMessage) *errorMessage = retainError(
                @"shared-surface request set is not in flight");
            return false;
        }
        CompletionState *completion = &set->completion;
        bool completed = dispatch_semaphore_wait(
            completion->semaphore,
            dispatch_time(DISPATCH_TIME_NOW, timeoutNanoseconds)) == 0;
        if (!completed) {
            if (errorMessage) *errorMessage = retainError(
                @"shared-surface evaluation completion timeout");
            return false;
        }
        uint64_t completedAt = atomic_load_explicit(
            &completion->completedAt, memory_order_acquire);
        uint32_t ordinal = atomic_load_explicit(
            &completion->completionOrdinal, memory_order_acquire);
        atomic_store_explicit(&completion->pending, false, memory_order_release);
        atomic_store_explicit(&set->inFlight, false, memory_order_release);
        atomic_store_explicit(
            &set->program->bankOperationState[set->sourceBank],
            BankCatalogOwnedIdle,
            memory_order_release);
        if (timing) *timing = (ANEFieldfareANEEvaluationTiming){
            .enqueueMilliseconds = completion->enqueueMilliseconds,
            .completionMilliseconds = completedAt >= completion->startedAt
                ? milliseconds(completedAt - completion->startedAt) : 0,
            .completionOrdinal = ordinal,
            .inFlightAfterEnqueue = 1,
        };
        return true;
    }
}

bool ANEFieldfareANESharedSurfaceRequestSetUnmapAll(
    ANEFieldfareANESharedSurfaceRequestSet *set, uint32_t *unmappedCount,
    const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (unmappedCount) *unmappedCount = 0;
        if (!set || atomic_load_explicit(&set->inFlight, memory_order_acquire)) {
            if (errorMessage) *errorMessage = retainError(
                @"cannot unmap an in-flight shared-surface request set");
            return false;
        }
        if (!set->ownsBank) return true;
        uint32_t expectedBankState = BankCatalogOwnedIdle;
        if (!atomic_compare_exchange_strong_explicit(
                &set->program->bankOperationState[set->sourceBank],
                &expectedBankState, BankMappingMutation, memory_order_acq_rel,
                memory_order_relaxed)) {
            if (errorMessage) *errorMessage = retainError(
                @"shared carrier bank is not owned and idle for unmap");
            return false;
        }
        uint32_t count = 0;
        for (uint32_t index = 0; index < set->requestCount; ++index) {
            if (!set->mapped[index]) continue;
            ((void (*)(id, SEL, id))objc_msgSend)(
                set->requests[index], @selector(setCompletionHandler:), nil);
            ((void (*)(id, SEL, id))objc_msgSend)(
                set->program->model, @selector(unmapIOSurfacesWithRequest:),
                set->requests[index]);
            set->mapped[index] = NO;
            ++count;
        }
        set->mappedCount -= count;
        set->ownsBank = false;
        atomic_store_explicit(
            &set->program->bankOperationState[set->sourceBank], BankIdle,
            memory_order_release);
        if (unmappedCount) *unmappedCount = count;
        return true;
    }
}

void ANEFieldfareANESharedSurfaceRequestSetDestroy(
    ANEFieldfareANESharedSurfaceRequestSet *set) {
    if (!set) return;
    uint32_t ignored = 0;
    if (!ANEFieldfareANESharedSurfaceRequestSetUnmapAll(set, &ignored, NULL))
        return; // Completion handler still owns the allocation; leak is safer.
    ANEFieldfareANEProgram *program = set->program;
    free(set);
    releaseSharedRequestSetProgram(program);
}

void *ANEFieldfareANEWeightBlobCreate(uint64_t elementCount,
                                      uint64_t *blobByteCount,
                                      void **payload) {
    return ANEFieldfareANEWeightBlobCreateWithBytes(elementCount * 2, blobByteCount, payload);
}

void *ANEFieldfareANEWeightBlobCreateWithBytes(uint64_t payloadBytes,
                                               uint64_t *blobByteCount,
                                               void **payload) {
    if (!payloadBytes) return NULL;
    uint64_t total = 128 + payloadBytes;
    uint8_t *blob = calloc(total, 1);
    if (!blob) return NULL;
    // File header. Chunk header follows at offset 64, which is the offset the
    // MIL BLOBFILE reference declares. Payload begins 64 bytes after that.
    blob[0] = 0x01;
    blob[4] = 0x02;
    uint8_t *chunk = blob + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    // chunk+8 is a uint32 payload-size field and chunk+16 is the absolute
    // offset of the payload within the blob. The prior-art reference hardcodes
    // byte 10 to 0x08, which is that size field frozen at 0x00080000 for its
    // own 512-channel fixture; it has no numerical oracle, so the constant went
    // unnoticed there. Both fields are written explicitly here.
    *(uint32_t *)(chunk + 8) = (uint32_t)payloadBytes;
    *(uint32_t *)(chunk + 16) = 128;
    if (payload) *payload = chunk + 64;
    if (blobByteCount) *blobByteCount = total;
    return blob;
}

static ANEFieldfareANEProgram *createProgram(const char *milUTF8,
                                             NSDictionary<NSString *, NSData *> *weightFiles,
                                             uint64_t inputBytes,
                                             uint64_t outputBytes,
                                             uint32_t requestBanks,
                                             NSArray<NSString *> *procedureNames,
                                             const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!milUTF8 || requestBanks < 1 || requestBanks > 8) {
            if (errorMessage) *errorMessage = retainError(@"invalid ANE program arguments");
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
        NSData *mil = [[NSString stringWithUTF8String:milUTF8] dataUsingEncoding:NSUTF8StringEncoding];
        NSString *dumpPath = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_DUMP_MIL"];
        if (dumpPath.length) [mil writeToFile:dumpPath atomically:YES];
        NSMutableDictionary *weights = [NSMutableDictionary dictionary];
        for (NSString *name in weightFiles) {
            NSString *path = [@"@model_path/weights/" stringByAppendingString:name];
            weights[path] = @{@"offset": @0, @"data": weightFiles[name]};
        }
        id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:), mil, weights, nil);
        id model = descriptor ? ((id (*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor) : nil;
        if (!model) {
            if (errorMessage) *errorMessage = retainError(@"ANE model construction failed");
            return NULL;
        }
        const char *perfMaskValue = getenv("ANE_FIELDFARE_PERF_STATS_MASK");
        uint32_t requestedPerfMask = perfMaskValue && *perfMaskValue
            ? (uint32_t)strtoull(perfMaskValue, NULL, 0) : 0;
        if (requestedPerfMask)
            ((void (*)(id, SEL, uint32_t))objc_msgSend)(
                model, @selector(setPerfStatsMask:), requestedPerfMask);
        NSString *queueValue = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_QUEUE_DEPTH"];
        if (queueValue.length) {
            NSInteger queueDepth = queueValue.integerValue;
            if (queueDepth >= 1 && queueDepth <= 127)
                ((void (*)(id, SEL, int8_t))objc_msgSend)(model, @selector(setQueueDepth:), (int8_t)queueDepth);
        }
        NSString *identifier = ((id (*)(id, SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *temporary = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        NSFileManager *files = NSFileManager.defaultManager;
        [files createDirectoryAtPath:[temporary stringByAppendingPathComponent:@"weights"]
         withIntermediateDirectories:YES attributes:nil error:nil];
        [mil writeToFile:[temporary stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        for (NSString *name in weightFiles) {
            NSString *destination = [temporary stringByAppendingPathComponent:
                [@"weights/" stringByAppendingString:name]];
            if (![weightFiles[name] writeToFile:destination atomically:YES]) {
                if (errorMessage) *errorMessage = retainError(@"ANE weight blob write failed");
                [files removeItemAtPath:temporary error:nil];
                return NULL;
            }
        }
        NSError *error = nil;
        uint64_t started = mach_absolute_time();
        BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &error);
        double compileMs = milliseconds(mach_absolute_time() - started);
        if (!ok) {
            if (errorMessage) *errorMessage = retainError(error.localizedDescription ?: @"ANE compile failed");
            [files removeItemAtPath:temporary error:nil];
            return NULL;
        }
        NSMutableDictionary *loadOptions = [NSMutableDictionary dictionary];
        const char *fixedPool = getenv("ANE_FIELDFARE_MEMORY_POOL_ID");
        const char *poolModulus = getenv("ANE_FIELDFARE_MEMORY_POOL_MODULUS");
        if ((fixedPool && *fixedPool) || (poolModulus && *poolModulus)) {
            long long pool = fixedPool && *fixedPool ? strtoll(fixedPool, NULL, 0) : 0;
            long long modulus = poolModulus && *poolModulus ? strtoll(poolModulus, NULL, 0) : 0;
            if (modulus > 0) pool = atomic_fetch_add_explicit(
                &programCreateOrdinal, 1, memory_order_relaxed) % (uint32_t)modulus;
            NSString *__unsafe_unretained *symbol = (NSString *__unsafe_unretained *)
                dlsym(framework, "kANEFMemoryPoolIDKey");
            NSString *key = symbol && *symbol ? *symbol : @"kANEFMemoryPoolIDKey";
            loadOptions[key] = @(pool);
        }
        const char *keepWired = getenv("ANE_FIELDFARE_KEEP_MODEL_MEMORY_WIRED");
        if (keepWired && *keepWired) {
            NSString *__unsafe_unretained *symbol = (NSString *__unsafe_unretained *)
                dlsym(framework, "kANEFKeepModelMemoryWiredKey");
            NSString *key = symbol && *symbol ? *symbol : @"kANEFKeepModelMemoryWiredKey";
            loadOptions[key] = @(strtol(keepWired, NULL, 0) != 0);
        }
        started = mach_absolute_time();
        ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, loadOptions, &error);
        double loadMs = milliseconds(mach_absolute_time() - started);
        if (!ok) {
            if (errorMessage) *errorMessage = retainError(error.localizedDescription ?: @"ANE load failed");
            [files removeItemAtPath:temporary error:nil];
            return NULL;
        }
        // Loading reconstructs the evaluation program and resets this property
        // on the current runtime. Apply the requested mask again before request
        // construction/mapping, then let the gate verify exact readback.
        if (requestedPerfMask)
            ((void (*)(id, SEL, uint32_t))objc_msgSend)(
                model, @selector(setPerfStatsMask:), requestedPerfMask);

        ANEFieldfareANEProgram *program = calloc(1, sizeof(*program));
        program->model = model;
        program->temporaryPath = temporary;
        program->info = (ANEFieldfareANEProgramInfo){
            .compileMilliseconds = compileMs,
            .loadMilliseconds = loadMs,
            .reportedQueueDepth = ((int8_t (*)(id, SEL))objc_msgSend)(model, @selector(queueDepth)),
            .requestBankCount = requestBanks,
        };
        for (uint32_t bank = 0; bank < requestBanks; ++bank) {
            uint32_t procedureIndex = 0;
            NSUInteger inputIndex = 0, outputIndex = 0;
            uint64_t bankInputBytes = inputBytes, bankOutputBytes = outputBytes;
            if (procedureNames.count) {
                NSString *name = procedureNames[bank];
                CompilerProcedureMetadata metadata = {0};
                NSString *failure = nil;
                if (!resolveCompilerProcedureMetadata(model, name, &metadata, &failure)) {
                    if (errorMessage) *errorMessage = retainError(failure);
                    ANEFieldfareANEProgramDestroy(program);
                    return NULL;
                }
                procedureIndex = metadata.procedureIndex;
                inputIndex = metadata.inputIndex;
                outputIndex = metadata.outputIndex;
                bankInputBytes = metadata.inputBytes;
                bankOutputBytes = metadata.outputBytes;
            }
            program->inputs[bank] = makeSurface(bankInputBytes);
            program->outputs[bank] = makeSurface(bankOutputBytes);
            id input = program->inputs[bank] ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), program->inputs[bank]) : nil;
            id output = program->outputs[bank] ? ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:), program->outputs[bank]) : nil;
            // _ANERequest validation expects a collection. The runtime ships an
            // authoritative stats-IOSurface wrapper; use its ordinary -new
            // construction and let the gate inspect the resulting object rather
            // than fabricating a buffer layout or counter schema.
            Class perfSurfaceClass = NSClassFromString(@"_ANEPerformanceStatsIOSurface");
            program->perfStatsSurfaces[bank] = requestedPerfMask ? makeSurface(1024 * 1024) : NULL;
            id perfSurface = requestedPerfMask && perfSurfaceClass && program->perfStatsSurfaces[bank]
                ? ((id (*)(Class, SEL, IOSurfaceRef, int64_t))objc_msgSend)(
                    perfSurfaceClass, @selector(objectWithIOSurface:statType:),
                    program->perfStatsSurfaces[bank], (int64_t)requestedPerfMask) : nil;
            id requestedPerfStats = perfSurface ? @[perfSurface] :
                (requestedPerfMask ? [NSMutableArray array] : nil);
            program->configuredPerfStats[bank] = requestedPerfStats;
            program->requests[bank] = input && output ?
                ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
                    requestClass,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
                    @[input], @[@(inputIndex)], @[output], @[@(outputIndex)], nil,
                    requestedPerfStats,
                    @(procedureIndex)) : nil;
            program->completions[bank].semaphore = dispatch_semaphore_create(0);
            setCompletionHandler(program, bank, program->requests[bank]);
            program->mapped[bank] = program->requests[bank] &&
                ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
                    model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
                    program->requests[bank], YES, &error);
            if (!program->mapped[bank]) {
                if (errorMessage) *errorMessage = retainError(error.localizedDescription ?: @"ANE IOSurface map failed");
                ANEFieldfareANEProgramDestroy(program);
                return NULL;
            }
        }
        return program;
    }
}

ANEFieldfareANEProgram *ANEFieldfareANEProgramCreate(const char *milUTF8,
                                                      uint64_t inputBytes,
                                                      uint64_t outputBytes,
                                                      uint32_t requestBanks,
                                                      const char **errorMessage) {
    return createProgram(milUTF8, @{}, inputBytes, outputBytes, requestBanks, nil, errorMessage);
}

ANEFieldfareANEProgram *ANEFieldfareANEProgramCreateWithConstantWeights(
    const char *milUTF8,
    const char *blobName,
    const void *blobBytes,
    uint64_t blobByteCount,
    uint64_t inputBytes,
    uint64_t outputBytes,
    uint32_t requestBanks,
    const char **errorMessage) {
    @autoreleasepool {
        if (!blobName || !blobBytes || !blobByteCount) {
            if (errorMessage) *errorMessage = retainError(@"missing constant weight blob");
            return NULL;
        }
        NSData *blob = [NSData dataWithBytesNoCopy:(void *)blobBytes
                                            length:(NSUInteger)blobByteCount
                                      freeWhenDone:NO];
        NSString *name = [NSString stringWithUTF8String:blobName];
        return createProgram(milUTF8, @{name: blob}, inputBytes, outputBytes,
                             requestBanks, nil, errorMessage);
    }
}

ANEFieldfareANEProgram *ANEFieldfareANEProgramCreateWithTwoProceduresAndConstantWeights(
    const char *milUTF8,
    const char *blobName0,
    const void *blobBytes0,
    uint64_t blobByteCount0,
    const char *blobName1,
    const void *blobBytes1,
    uint64_t blobByteCount1,
    const char *procedureName0,
    const char *procedureName1,
    const char **errorMessage) {
    @autoreleasepool {
        if (!blobName0 || !blobBytes0 || !blobByteCount0 ||
            !blobName1 || !blobBytes1 || !blobByteCount1 ||
            !procedureName0 || !procedureName1) {
            if (errorMessage) *errorMessage = retainError(@"missing two-procedure program data");
            return NULL;
        }
        NSString *weightName0 = [NSString stringWithUTF8String:blobName0];
        NSString *weightName1 = [NSString stringWithUTF8String:blobName1];
        NSData *weight0 = [NSData dataWithBytesNoCopy:(void *)blobBytes0
                                                length:(NSUInteger)blobByteCount0
                                          freeWhenDone:NO];
        NSData *weight1 = [NSData dataWithBytesNoCopy:(void *)blobBytes1
                                                length:(NSUInteger)blobByteCount1
                                          freeWhenDone:NO];
        NSArray *procedures = @[
            [NSString stringWithUTF8String:procedureName0],
            [NSString stringWithUTF8String:procedureName1],
        ];
        return createProgram(milUTF8, @{weightName0: weight0, weightName1: weight1},
                             0, 0, 2, procedures, errorMessage);
    }
}

ANEFieldfareANEProgram *ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
    const char *milUTF8,
    const char *blobName,
    const void *blobBytes,
    uint64_t blobByteCount,
    const char *const *procedureNames,
    uint32_t procedureCount,
    const char **errorMessage) {
    @autoreleasepool {
        if (!milUTF8 || !blobName || !blobBytes || !blobByteCount ||
            !procedureNames || !procedureCount || procedureCount > 8) {
            if (errorMessage) *errorMessage = retainError(@"invalid multi-procedure program data");
            return NULL;
        }
        NSMutableArray<NSString *> *procedures = [NSMutableArray arrayWithCapacity:procedureCount];
        for (uint32_t index = 0; index < procedureCount; ++index) {
            if (!procedureNames[index] || !procedureNames[index][0]) {
                if (errorMessage) *errorMessage = retainError(@"missing compiler procedure name");
                return NULL;
            }
            [procedures addObject:[NSString stringWithUTF8String:procedureNames[index]]];
        }
        NSData *blob = [NSData dataWithBytesNoCopy:(void *)blobBytes
                                            length:(NSUInteger)blobByteCount
                                      freeWhenDone:NO];
        NSString *name = [NSString stringWithUTF8String:blobName];
        return createProgram(milUTF8, @{name: blob}, 0, 0, procedureCount,
                             procedures, errorMessage);
    }
}

ANEFieldfareANEProgramInfo ANEFieldfareANEProgramGetInfo(ANEFieldfareANEProgram *program) {
    return program ? program->info : (ANEFieldfareANEProgramInfo){0};
}

IOSurfaceRef ANEFieldfareANEProgramInput(ANEFieldfareANEProgram *program, uint32_t bank) {
    return program && bank < program->info.requestBankCount ? program->inputs[bank] : NULL;
}

IOSurfaceRef ANEFieldfareANEProgramOutput(ANEFieldfareANEProgram *program, uint32_t bank) {
    return program && bank < program->info.requestBankCount ? program->outputs[bank] : NULL;
}

bool ANEFieldfareANEProgramEvaluate(ANEFieldfareANEProgram *program,
                                    uint32_t bank,
                                    uint64_t timeoutNanoseconds,
                                    ANEFieldfareANEEvaluationTiming *timing,
                                    const char **errorMessage) {
    ANEFieldfareANEEvaluationTiming beginTiming = {0};
    if (!ANEFieldfareANEProgramEvaluateBegin(program, bank, &beginTiming, errorMessage)) return false;
    bool completed = ANEFieldfareANEProgramEvaluateWait(
        program, bank, timeoutNanoseconds, timing, errorMessage);
    if (timing) timing->enqueueMilliseconds = beginTiming.enqueueMilliseconds;
    return completed;
}

bool ANEFieldfareANEProgramEvaluateBegin(ANEFieldfareANEProgram *program,
                                         uint32_t bank,
                                         ANEFieldfareANEEvaluationTiming *timing,
                                         const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!program || bank >= program->info.requestBankCount) {
            if (errorMessage) *errorMessage = retainError(@"invalid ANE request bank");
            return false;
        }
        uint32_t expectedState = BankIdle;
        if (!atomic_compare_exchange_strong_explicit(&program->bankOperationState[bank],
                                                     &expectedState,
                                                     BankOrdinaryEvaluationInFlight,
                                                     memory_order_acq_rel,
                                                     memory_order_relaxed)) {
            NSString *reason = expectedState == BankMappingMutation
                ? @"ANE request bank mapping is being mutated"
                : (expectedState == BankCatalogOwnedIdle
                    ? @"ANE request bank is owned by a shared-surface catalog"
                    : (expectedState == BankCatalogEvaluationInFlight
                        ? @"shared-surface request selection already in flight"
                        : @"ANE request bank already in flight"));
            if (errorMessage) *errorMessage = retainError(reason);
            return false;
        }
        CompletionState *completion = &program->completions[bank];
        atomic_store_explicit(&completion->pending, true, memory_order_release);
        atomic_store_explicit(&completion->completedAt, 0, memory_order_relaxed);
        atomic_store_explicit(&completion->completionOrdinal, 0, memory_order_relaxed);
        completion->startedAt = mach_absolute_time();
        uint32_t inFlight = atomic_fetch_add_explicit(
            &program->currentAsyncRequestsInFlight, 1, memory_order_relaxed) + 1;
        uint32_t observedMaximum = atomic_load_explicit(
            &program->maximumAsyncRequestsInFlight, memory_order_relaxed);
        while (inFlight > observedMaximum && !atomic_compare_exchange_weak_explicit(
                   &program->maximumAsyncRequestsInFlight, &observedMaximum, inFlight,
                   memory_order_relaxed, memory_order_relaxed)) {}
        NSError *error = nil;
        BOOL accepted = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            program->model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, program->requests[bank], &error);
        uint64_t enqueued = mach_absolute_time();
        completion->enqueueMilliseconds = milliseconds(enqueued - completion->startedAt);
        if (!accepted) {
            atomic_fetch_sub_explicit(&program->currentAsyncRequestsInFlight, 1,
                                      memory_order_relaxed);
            atomic_store_explicit(&completion->pending, false, memory_order_release);
            atomic_store_explicit(&program->bankOperationState[bank], BankIdle,
                                  memory_order_release);
            if (errorMessage) *errorMessage = retainError(
                error.localizedDescription ?: @"ANE evaluate submission failed");
            return false;
        }
        if (timing) *timing = (ANEFieldfareANEEvaluationTiming){
            .enqueueMilliseconds = completion->enqueueMilliseconds,
            .inFlightAfterEnqueue = atomic_load_explicit(
                &program->currentAsyncRequestsInFlight, memory_order_relaxed),
        };
        return true;
    }
}

bool ANEFieldfareANEProgramEvaluateWait(ANEFieldfareANEProgram *program,
                                        uint32_t bank,
                                        uint64_t timeoutNanoseconds,
                                        ANEFieldfareANEEvaluationTiming *timing,
                                        const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!program || bank >= program->info.requestBankCount) {
            if (errorMessage) *errorMessage = retainError(@"invalid ANE request bank");
            return false;
        }
        CompletionState *completion = &program->completions[bank];
        if (!atomic_load_explicit(&completion->pending, memory_order_acquire)) {
            if (errorMessage) *errorMessage = retainError(@"ANE request bank is not in flight");
            return false;
        }
        bool completed = dispatch_semaphore_wait(
            completion->semaphore, dispatch_time(DISPATCH_TIME_NOW, timeoutNanoseconds)) == 0;
        if (!completed) {
            if (errorMessage) *errorMessage = retainError(@"ANE completion timeout");
            return false;
        }
        uint64_t completedAt = atomic_load_explicit(&completion->completedAt, memory_order_acquire);
        uint32_t ordinal = atomic_load_explicit(&completion->completionOrdinal, memory_order_acquire);
        atomic_store_explicit(&completion->pending, false, memory_order_release);
        atomic_store_explicit(&program->bankOperationState[bank], BankIdle,
                              memory_order_release);
        if (timing) *timing = (ANEFieldfareANEEvaluationTiming){
            .enqueueMilliseconds = completion->enqueueMilliseconds,
            .completionMilliseconds = completedAt >= completion->startedAt
                ? milliseconds(completedAt - completion->startedAt) : 0,
            .completionOrdinal = ordinal,
            .inFlightAfterEnqueue = 0,
        };
        return true;
    }
}

uint32_t ANEFieldfareANEProgramCurrentAsyncRequestsInFlight(ANEFieldfareANEProgram *program) {
    return program ? atomic_load_explicit(&program->currentAsyncRequestsInFlight,
                                          memory_order_relaxed) : 0;
}

uint32_t ANEFieldfareANEProgramMaximumAsyncRequestsInFlight(ANEFieldfareANEProgram *program) {
    return program ? atomic_load_explicit(&program->maximumAsyncRequestsInFlight,
                                          memory_order_relaxed) : 0;
}

uint32_t ANEFieldfareANEProgramPerformanceStatsMask(ANEFieldfareANEProgram *program) {
    return program && program->model
        ? ((uint32_t (*)(id, SEL))objc_msgSend)(program->model, @selector(perfStatsMask)) : 0;
}

bool ANEFieldfareANEProgramSetPerformanceStatsMask(ANEFieldfareANEProgram *program,
                                                   uint32_t mask,
                                                   const char **errorMessage) {
    @autoreleasepool {
        if (errorMessage) *errorMessage = NULL;
        if (!program || !program->model) {
            if (errorMessage) *errorMessage = retainError(@"ANE program is unavailable");
            return false;
        }
        ((void (*)(id, SEL, uint32_t))objc_msgSend)(
            program->model, @selector(setPerfStatsMask:), mask);
        if (ANEFieldfareANEProgramPerformanceStatsMask(program) != mask) {
            if (errorMessage) *errorMessage = retainError(@"ANE perfStatsMask readback mismatch");
            return false;
        }
        return true;
    }
}

CFTypeRef ANEFieldfareANEProgramCopyRequestPerformanceStats(
    ANEFieldfareANEProgram *program, uint32_t bank) {
    if (!program || bank >= program->info.requestBankCount || !program->requests[bank]) return NULL;
    id value = ((id (*)(id, SEL))objc_msgSend)(program->requests[bank], @selector(perfStats));
    return value ? CFRetain((__bridge CFTypeRef)value) : NULL;
}

CFTypeRef ANEFieldfareANEProgramCopyRequestPerformanceStatsArray(
    ANEFieldfareANEProgram *program, uint32_t bank) {
    if (!program || bank >= program->info.requestBankCount || !program->requests[bank]) return NULL;
    id value = ((id (*)(id, SEL))objc_msgSend)(program->requests[bank], @selector(perfStatsArray));
    return value ? CFRetain((__bridge CFTypeRef)value) : NULL;
}

CFTypeRef ANEFieldfareANEProgramCopyConfiguredPerformanceStats(
    ANEFieldfareANEProgram *program, uint32_t bank) {
    if (!program || bank >= program->info.requestBankCount) return NULL;
    id value = program->configuredPerfStats[bank];
    return value ? CFRetain((__bridge CFTypeRef)value) : NULL;
}

void ANEFieldfareANEProgramDestroy(ANEFieldfareANEProgram *program) {
    if (!program) return;
    if (atomic_load_explicit(&program->sharedRequestSetCount,
                             memory_order_acquire)) {
        atomic_store_explicit(&program->destroyRequested, true,
                              memory_order_release);
        return;
    }
    @autoreleasepool {
        NSError *error = nil;
        for (uint32_t bank = 0; bank < program->info.requestBankCount; ++bank) {
            if (program->mapped[bank]) ((void (*)(id, SEL, id))objc_msgSend)(
                program->model, @selector(unmapIOSurfacesWithRequest:), program->requests[bank]);
            if (program->requests[bank]) ((void (*)(id, SEL, id))objc_msgSend)(
                program->requests[bank], @selector(setCompletionHandler:), nil);
        }
        if (program->model) ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            program->model, @selector(unloadWithQoS:error:), 21, &error);
        if (program->temporaryPath)
            [NSFileManager.defaultManager removeItemAtPath:program->temporaryPath error:nil];
        for (uint32_t bank = 0; bank < program->info.requestBankCount; ++bank) {
            if (program->inputs[bank]) CFRelease(program->inputs[bank]);
            if (program->outputs[bank]) CFRelease(program->outputs[bank]);
            if (program->perfStatsSurfaces[bank]) CFRelease(program->perfStatsSurfaces[bank]);
        }
        free(program);
    }
}
