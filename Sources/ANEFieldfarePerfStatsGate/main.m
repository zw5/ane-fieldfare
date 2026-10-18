#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <math.h>
#include <dlfcn.h>

#include "../ANEFieldfareRuntime/ANEProgram.h"

static double milliseconds(uint64_t ticks) {
    mach_timebase_info_data_t info = {0};
    mach_timebase_info(&info);
    return (double)ticks * info.numer / info.denom / 1e6;
}

static NSDictionary *rawStatsSurfaceSnapshot(id wrapper) {
    id statsObject = ((id (*)(id, SEL))objc_msgSend)(wrapper, @selector(stats));
    IOSurfaceRef surface = (__bridge IOSurfaceRef)statsObject;
    if (!surface) return @{ @"surface_present": @NO };
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    const uint8_t *bytes = IOSurfaceGetBaseAddress(surface);
    size_t size = IOSurfaceGetAllocSize(surface), nonzero = 0, lastNonzero = 0;
    uint64_t fnv = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        if (bytes[index]) { ++nonzero; lastNonzero = index; }
        fnv = (fnv ^ bytes[index]) * 1099511628211ULL;
    }
    size_t prefixSize = MIN((size_t)256, size);
    NSData *prefix = [NSData dataWithBytes:bytes length:prefixSize];
    NSData *allData = [NSData dataWithBytes:bytes length:size];
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    NSMutableDictionary *record = [@{
        @"surface_present": @YES, @"allocated_bytes": @(size),
        @"nonzero_bytes": @(nonzero),
        @"last_nonzero_offset": nonzero ? @(lastNonzero) : NSNull.null,
        @"fnv1a64": [NSString stringWithFormat:@"0x%016llx", fnv],
        @"first_256_bytes_base64": [prefix base64EncodedStringWithOptions:0]
    } mutableCopy];
    @try {
        Class statsClass = NSClassFromString(@"_ANEPerformanceStats");
        id decoded = ((id (*)(Class, SEL, id))objc_msgSend)(
            statsClass, @selector(decodeRawStatsData:), allData);
        record[@"decodeRawStatsData"] = decoded ? @{
            @"present": @YES, @"class": NSStringFromClass([decoded class]),
            @"description": [decoded description] ?: @"nil",
            @"hardware_execution_raw": [decoded respondsToSelector:@selector(hwExecutionTime)]
                ? @(((uint64_t (*)(id, SEL))objc_msgSend)(decoded, @selector(hwExecutionTime))) : NSNull.null,
            @"performance_counters_class": [decoded respondsToSelector:@selector(performanceCounters)] &&
                ((id (*)(id, SEL))objc_msgSend)(decoded, @selector(performanceCounters))
                    ? NSStringFromClass([((id (*)(id, SEL))objc_msgSend)(decoded,
                        @selector(performanceCounters)) class]) : @"nil"
        } : @{ @"present": @NO };
    } @catch (NSException *exception) {
        record[@"decodeRawStatsData"] = @{
            @"present": @NO, @"exception": exception.reason ?: @"unknown" };
    }
    return record;
}

static NSDictionary *snapshotObject(CFTypeRef retained) {
    if (!retained) return @{ @"present": @NO, @"class": @"nil" };
    id value = CFBridgingRelease(retained);
    NSString *className = NSStringFromClass([value class]);
    NSMutableDictionary *record = [@{ @"present": @YES, @"class": className } mutableCopy];
    if ([value isKindOfClass:NSArray.class]) {
        NSArray *array = value;
        record[@"count"] = @(array.count);
        NSMutableArray *items = [NSMutableArray array];
        for (id item in array) {
            if ([item isKindOfClass:NSNumber.class])
                [items addObject:@{ @"class": NSStringFromClass([item class]), @"number": item }];
            else if ([item isKindOfClass:NSData.class]) {
                NSData *data = item;
                [items addObject:@{ @"class": NSStringFromClass([item class]),
                                    @"byte_count": @(data.length),
                                    @"base64": [data base64EncodedStringWithOptions:0] }];
            } else if ([item isKindOfClass:NSDictionary.class])
                [items addObject:@{ @"class": NSStringFromClass([item class]), @"dictionary": item }];
            else if ([item respondsToSelector:@selector(statType)] &&
                     [item respondsToSelector:@selector(stats)]) {
                int64_t statType = ((int64_t (*)(id, SEL))objc_msgSend)(item, @selector(statType));
                id stats = ((id (*)(id, SEL))objc_msgSend)(item, @selector(stats));
                [items addObject:@{ @"class": NSStringFromClass([item class]),
                                    @"stat_type_raw": @(statType),
                                    @"stats_present": @(stats != nil),
                                    @"stats_class": stats ? NSStringFromClass([stats class]) : @"nil",
                                    @"raw_surface": rawStatsSurfaceSnapshot(item),
                                    @"description": [item description] ?: @"nil" }];
            }
            else
                [items addObject:@{ @"class": NSStringFromClass([item class]),
                                    @"description": [item description] ?: @"nil" }];
        }
        record[@"items"] = items;
    } else if ([value isKindOfClass:NSNumber.class]) record[@"number"] = value;
    else if ([value isKindOfClass:NSData.class]) {
        NSData *data = value;
        record[@"byte_count"] = @(data.length);
        record[@"base64"] = [data base64EncodedStringWithOptions:0];
    } else record[@"description"] = [value description] ?: @"nil";
    return record;
}

static NSString *milForShape(uint32_t channels, uint32_t positions) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-version\", \"9.0\"}})]\n"
         "{\n  func main<ios18>(tensor<fp16,[1,%u,1,%u]> x) {\n"
         "    fp16 z = const()[name=string(\"z\"),val=fp16(0)];\n"
         "    tensor<fp16,[1,%u,1,%u]> y = maximum(x=x,y=z)[name=string(\"work\")];\n"
         "    string f32 = const()[name=string(\"f32\"),val=string(\"fp32\")];\n"
         "    tensor<fp32,[1,%u,1,%u]> out = cast(dtype=f32,x=y)[name=string(\"out\")];\n"
         "  } -> (out);\n}\n", channels, positions, channels, positions,
         channels, positions];
}

static bool prepare(ANEFieldfareANEProgram *program,
                    uint32_t channels,
                    uint32_t positions,
                    uint32_t iteration) {
    IOSurfaceRef inputSurface = ANEFieldfareANEProgramInput(program, 0);
    IOSurfaceRef outputSurface = ANEFieldfareANEProgramOutput(program, 0);
    if (!inputSurface || !outputSurface) return false;
    IOSurfaceLock(inputSurface, 0, NULL);
    __fp16 *input = IOSurfaceGetBaseAddress(inputSurface);
    uint64_t count = (uint64_t)channels * positions;
    for (uint64_t index = 0; index < count; ++index)
        input[index] = (__fp16)(((int32_t)((index + iteration * 17) % 257) - 128) / 256.0f);
    IOSurfaceUnlock(inputSurface, 0, NULL);
    IOSurfaceLock(outputSurface, 0, NULL);
    float *output = IOSurfaceGetBaseAddress(outputSurface);
    for (uint64_t index = 0; index < count; ++index) output[index] = NAN;
    IOSurfaceUnlock(outputSurface, 0, NULL);
    return true;
}

static NSDictionary *validate(ANEFieldfareANEProgram *program,
                              uint32_t channels,
                              uint32_t positions) {
    IOSurfaceRef inputSurface = ANEFieldfareANEProgramInput(program, 0);
    IOSurfaceRef outputSurface = ANEFieldfareANEProgramOutput(program, 0);
    IOSurfaceLock(inputSurface, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL);
    const __fp16 *input = IOSurfaceGetBaseAddress(inputSurface);
    const float *output = IOSurfaceGetBaseAddress(outputSurface);
    uint64_t count = (uint64_t)channels * positions, poison = 0, failures = 0;
    double maxError = 0;
    for (uint64_t index = 0; index < count; ++index) {
        if (isnan(output[index])) ++poison;
        float expected = fmaxf((float)input[index], 0.0f);
        double error = fabs((double)output[index] - expected);
        if (!isfinite(output[index]) || error > 0.0001) ++failures;
        if (error > maxError) maxError = error;
    }
    IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(inputSurface, kIOSurfaceLockReadOnly, NULL);
    return @{ @"valid": @(failures == 0 && poison == 0),
              @"elements": @(count), @"failures": @(failures),
              @"poison_remaining": @(poison), @"max_absolute_error": @(maxError) };
}

static NSDictionary *runShape(NSString *name,
                              uint32_t channels,
                              uint32_t positions,
                              uint32_t mask,
                              bool *overall) {
    NSString *maskString = [NSString stringWithFormat:@"0x%08x", mask];
    setenv("ANE_FIELDFARE_PERF_STATS_MASK", maskString.UTF8String, 1);
    NSString *mil = milForShape(channels, positions);
    uint64_t elements = (uint64_t)channels * positions;
    const char *creationError = NULL;
    ANEFieldfareANEProgram *program = ANEFieldfareANEProgramCreate(
        mil.UTF8String, elements * sizeof(__fp16), elements * sizeof(float), 1,
        &creationError);
    bool valid = program != NULL;
    uint32_t maskReadback = ANEFieldfareANEProgramPerformanceStatsMask(program);
    NSDictionary *zeroPerf = snapshotObject(
        ANEFieldfareANEProgramCopyRequestPerformanceStats(program, 0));
    NSDictionary *zeroArray = snapshotObject(
        ANEFieldfareANEProgramCopyRequestPerformanceStatsArray(program, 0));
    NSDictionary *configured = snapshotObject(
        ANEFieldfareANEProgramCopyConfiguredPerformanceStats(program, 0));
    NSMutableArray *runs = [NSMutableArray array];
    NSMutableArray *countProgression = [NSMutableArray array];
    double wallSum = 0, enqueueSum = 0, completionSum = 0;
    for (uint32_t iteration = 0; valid && iteration < 17; ++iteration) {
        valid = prepare(program, channels, positions, iteration);
        uint64_t wallStarted = mach_absolute_time();
        ANEFieldfareANEEvaluationTiming timing = {0};
        const char *error = NULL;
        bool evaluated = valid && ANEFieldfareANEProgramEvaluate(
            program, 0, 5ULL * NSEC_PER_SEC, &timing, &error);
        double wall = milliseconds(mach_absolute_time() - wallStarted);
        wallSum += wall;
        enqueueSum += timing.enqueueMilliseconds;
        completionSum += timing.completionMilliseconds;
        NSDictionary *oracle = evaluated ? validate(program, channels, positions) : @{};
        NSDictionary *perf = snapshotObject(
            ANEFieldfareANEProgramCopyRequestPerformanceStats(program, 0));
        NSDictionary *array = snapshotObject(
            ANEFieldfareANEProgramCopyRequestPerformanceStatsArray(program, 0));
        NSNumber *count = array[@"count"] ?: perf[@"count"] ?: @(-1);
        [countProgression addObject:count];
        valid = valid && evaluated && [oracle[@"valid"] boolValue];
        [runs addObject:@{
            @"iteration": @(iteration + 1), @"wall_ms": @(wall),
            @"enqueue_ms": @(timing.enqueueMilliseconds),
            @"completion_ms": @(timing.completionMilliseconds),
            @"oracle": oracle, @"perfStats": perf, @"perfStatsArray": array,
            @"error": error ? [NSString stringWithUTF8String:error] : @"none"
        }];
        free((void *)error);
    }
    const char *setError = NULL;
    bool resetAccepted = ANEFieldfareANEProgramSetPerformanceStatsMask(program, 0, &setError);
    NSDictionary *afterResetPerf = snapshotObject(
        ANEFieldfareANEProgramCopyRequestPerformanceStats(program, 0));
    NSDictionary *afterResetArray = snapshotObject(
        ANEFieldfareANEProgramCopyRequestPerformanceStatsArray(program, 0));
    CFTypeRef invalidBank = ANEFieldfareANEProgramCopyRequestPerformanceStatsArray(program, 99);
    bool invalidBankRejected = invalidBank == NULL;
    if (invalidBank) CFRelease(invalidBank);
    ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);
    valid = valid && maskReadback == mask && resetAccepted && invalidBankRejected;
    *overall = *overall && valid;
    NSDictionary *result = @{
        @"name": name, @"shape": @{ @"channels": @(channels), @"positions": @(positions),
                                       @"elements": @(elements) },
        @"requested_mask": @(mask), @"requested_mask_hex": maskString,
        @"mask_readback_after_load": @(maskReadback),
        @"zero_work": @{ @"configured_perf_stats": configured,
                          @"perfStats": zeroPerf, @"perfStatsArray": zeroArray },
        @"one_work": runs.count ? runs[0] : @{},
        @"repeated_work": @{ @"evaluation_count": @(runs.count), @"runs": runs,
                               @"reported_count_progression": countProgression,
                               @"mean_wall_ms": runs.count ? @(wallSum / runs.count) : @0,
                               @"mean_enqueue_ms": runs.count ? @(enqueueSum / runs.count) : @0,
                               @"mean_completion_ms": runs.count ? @(completionSum / runs.count) : @0 },
        @"reset_control": @{ @"set_mask_zero_accepted": @(resetAccepted),
                              @"mask_readback": @(ANEFieldfareANEProgramPerformanceStatsMask(program)),
                              @"perfStats": afterResetPerf, @"perfStatsArray": afterResetArray,
                              @"error": setError ? [NSString stringWithUTF8String:setError] : @"none" },
        @"negative_control": @{ @"invalid_bank_returns_null": @(invalidBankRejected) },
        @"lifecycle": @{ @"compile_count": @1, @"load_count": @1,
                          @"compile_ms": @(info.compileMilliseconds), @"load_ms": @(info.loadMilliseconds) },
        @"status": valid ? @"validated_execution" : @"rejected",
        @"error": creationError ? [NSString stringWithUTF8String:creationError] : @"none"
    };
    free((void *)creationError); free((void *)setError);
    ANEFieldfareANEProgramDestroy(program);
    return result;
}

static NSDictionary *methodRecord(Class cls, SEL selector, bool classMethod) {
    Method method = classMethod ? class_getClassMethod(cls, selector) : class_getInstanceMethod(cls, selector);
    const char *encoding = method ? method_getTypeEncoding(method) : NULL;
    return @{ @"class": NSStringFromClass(cls) ?: @"nil",
              @"selector": NSStringFromSelector(selector),
              @"kind": classMethod ? @"class" : @"instance",
              @"present": @(method != NULL),
              @"encoding": encoding ? [NSString stringWithUTF8String:encoding] : @"absent" };
}

static NSArray *allMethods(Class cls, bool classMethods) {
    unsigned int count = 0;
    Method *methods = class_copyMethodList(classMethods ? object_getClass(cls) : cls, &count);
    NSMutableArray *records = [NSMutableArray array];
    for (unsigned int index = 0; index < count; ++index) {
        const char *encoding = method_getTypeEncoding(methods[index]);
        [records addObject:@{
            @"selector": NSStringFromSelector(method_getName(methods[index])),
            @"encoding": encoding ? [NSString stringWithUTF8String:encoding] : @"unknown",
            @"kind": classMethods ? @"class" : @"instance"
        }];
    }
    free(methods);
    return records;
}

int main(void) {
    @autoreleasepool {
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
               RTLD_NOW | RTLD_LOCAL);
        Class model = NSClassFromString(@"_ANEInMemoryModel");
        Class request = NSClassFromString(@"_ANERequest");
        Class performanceStats = NSClassFromString(@"_ANEPerformanceStats");
        Class performanceStatsSurface = NSClassFromString(@"_ANEPerformanceStatsIOSurface");
        unsigned int classCount = 0;
        Class *runtimeClasses = objc_copyClassList(&classCount);
        NSMutableArray *perfClasses = [NSMutableArray array];
        NSMutableArray *configurationOwners = [NSMutableArray array];
        for (unsigned int index = 0; index < classCount; ++index) {
            NSString *className = NSStringFromClass(runtimeClasses[index]);
            NSString *lower = className.lowercaseString;
            if ([lower containsString:@"perf"] || [lower containsString:@"stat"] ||
                [lower containsString:@"event"])
                [perfClasses addObject:className];
            if (class_getClassMethod(runtimeClasses[index], @selector(configurationForPerfStatsMask:)) ||
                class_getInstanceMethod(runtimeClasses[index], @selector(configurationForPerfStatsMask:)))
                [configurationOwners addObject:className];
        }
        free(runtimeClasses);
        [perfClasses sortUsingSelector:@selector(compare:)];
        NSArray *abi = @[
            methodRecord(model, @selector(perfStatsMask), false),
            methodRecord(model, @selector(setPerfStatsMask:), false),
            methodRecord(request, @selector(perfStats), false),
            methodRecord(request, @selector(perfStatsArray), false),
            methodRecord(request, @selector(setPerfStats:), false),
            methodRecord(request, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:), true)
        ];
        bool valid = model && request;
        NSDictionary *small = runShape(@"relu_64x64", 64, 64, 1, &valid);
        NSDictionary *large = runShape(@"relu_2816x64", 2816, 64, 1, &valid);
        unsetenv("ANE_FIELDFARE_PERF_STATS_MASK");
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.perf-stats.v1",
            @"status": valid ? @"validated_execution_stats_schema_unavailable" : @"rejected",
            @"host": NSProcessInfo.processInfo.hostName,
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
            @"abi_inventory": abi, @"runtime_perf_stat_event_classes": perfClasses,
            @"configuration_for_perf_stats_mask_owners": configurationOwners,
            @"performance_stats_class_inventory": @{
                @"_ANEPerformanceStats": @{
                    @"instance_methods": allMethods(performanceStats, false),
                    @"class_methods": allMethods(performanceStats, true)
                },
                @"_ANEPerformanceStatsIOSurface": @{
                    @"instance_methods": allMethods(performanceStatsSurface, false),
                    @"class_methods": allMethods(performanceStatsSurface, true)
                }
            },
            @"shapes": @[small, large],
            @"procedure_event_exports": @[
                @"ZinComputeProgramGetProcedureEvents(ZinComputeProgram*, ZinComputeProgramProcedure*, unsigned long long*, unsigned long long*, unsigned long long*)",
                @"ZinComputeProgramRTGetProcedureEvents(RtProcedureGraphNode const*, span<unsigned long long>&, span<unsigned long long>&, span<unsigned long long>&)",
                @"ZinComputeProgramRTGetProcedureLatencyStats(RtProcedureGraphNode const*, ZinProcedureLatencyStats*)",
                @"ZinComputeProgramSupportsSummaryPerformanceStats(ident_command const*, bool*)",
                @"ZinComputeProgramSupportsSummaryPerformanceStats(ident_command const*, bool&)"
            ],
            @"classification": @{
                @"plumbing": @"validated for mask readback, request collection acceptance, evaluation, and reset calls",
                @"counter_schema": @"request collection contains one runtime-authored _ANEPerformanceStatsIOSurface, but its 1 MiB raw surface remained byte-zero after 17 evaluations for both shapes",
                @"units": @"unknown; hwExecutionTime has uint64 ABI and a constructor name mentions NS, but no populated value validates nanoseconds",
                @"scope": @"request-local collection object plus model-level uint32 mask; hardware sampling scope unknown",
                @"correlation": @"unavailable: raw stats surfaces remained zero, so only independent enqueue, completion, and wall timings are reportable",
                @"field_meanings": @"none assigned from names or binary exports"
            },
            @"claim_boundary": @"Exact Objective-C ABI plus two independently oracle-validated shapes, each using one compiled and loaded program, with zero-work, one-work, seventeen-work, mask-reset, and invalid-bank controls. Request performance-statistics objects are retained and serialized only by observed runtime class and value. Procedure-event export names are presence evidence only. No field meaning, unit, hardware-counter scope, or latency correlation is claimed without populated numeric records."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        return valid ? 0 : 1;
    }
}
