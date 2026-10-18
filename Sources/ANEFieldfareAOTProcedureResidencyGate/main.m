#define main ANEFieldfareAOTBlockSliceSingleMain
#include "../ANEFieldfareAOTBlockSliceGate/main.m"
#undef main

#import "../ANEFieldfareAOTBlockSliceGate/RebindEvidence.h"

#import <mach/mach_time.h>
#import <stdatomic.h>

enum { ProcedureCount = 96, TouchCount = 8, MaximumScalingHandles = 4 };

static double elapsedMilliseconds(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)(end - start) * timebase.numer / timebase.denom / 1e6;
}

static BOOL expertOracle(ANEFieldfareTensor gate, ANEFieldfareTensor up,
                         ANEFieldfareTensor down, const float *input, float *output) {
    float *g = calloc(Rows, sizeof(float)), *u = calloc(Rows, sizeof(float));
    float *activated = calloc(Rows, sizeof(float));
    BOOL ok = g && u && activated && ANEFieldfareAffineMatVec(&gate, 4, Group, input, g) &&
        ANEFieldfareAffineMatVec(&up, 4, Group, input, u);
    if (ok) for (uint32_t row = 0; row < Rows; ++row) {
        float gf = (float)(__fp16)g[row], uf = (float)(__fp16)u[row];
        float gelu = 0.5f * gf * (1.0f + tanhf(0.7978845608028654f *
            (gf + 0.044715f * gf * gf * gf)));
        activated[row] = (float)(__fp16)(gelu * uf);
    }
    if (ok) ok = ANEFieldfareAffineMatVec(&down, 4, Group, activated, output);
    free(g); free(u); free(activated);
    return ok;
}

static BOOL compareExpertOutput(IOSurfaceRef surface, const float *expected,
                                uint64_t *comparisons, uint64_t *finite,
                                double *maxError, double *squareError) {
    if (!surface || !expected ||
        IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess)
        return NO;
    const float *actual = IOSurfaceGetBaseAddress(surface);
    for (uint32_t row = 0; row < Hidden; ++row)
        for (uint32_t position = 0; position < Positions; ++position) {
            float value = actual[(uint64_t)row * Positions + position];
            double delta = fabs((double)value - expected[row]);
            *maxError = fmax(*maxError, delta);
            *squareError += delta * delta;
            *finite += isfinite(value);
            ++*comparisons;
        }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return YES;
}

typedef struct {
    BOOL hostValid;
    BOOL taskValid;
    uint64_t wired;
    uint64_t purgeable;
    uint64_t compressor;
    uint64_t freeAndInactive;
    uint64_t physFootprint;
    uint64_t taskCompressed;
    uint64_t taskPurgeableResident;
    uint64_t residentSize;
} ResidencySample;

static ResidencySample residencySample(void) {
    vm_statistics64_data_t vm = {0};
    mach_msg_type_number_t count = 0;
    vm_size_t page = 0;
    BOOL pageOK = host_page_size(mach_host_self(), &page) == KERN_SUCCESS;
    BOOL hostOK = NO;
    for (uint32_t attempt = 0; pageOK && attempt < 4; ++attempt) {
        count = HOST_VM_INFO64_COUNT;
        hostOK = host_statistics64(mach_host_self(), HOST_VM_INFO64,
            (host_info64_t)&vm, &count) == KERN_SUCCESS;
        if (hostOK) break;
        if (attempt < 3) usleep(1000);
    }
    task_vm_info_data_t task = {0};
    count = TASK_VM_INFO_COUNT;
    BOOL taskOK = task_info(mach_task_self(), TASK_VM_INFO,
        (task_info_t)&task, &count) == KERN_SUCCESS;
    return (ResidencySample){
        .hostValid = hostOK,
        .taskValid = taskOK,
        .wired = hostOK ? (uint64_t)vm.wire_count * page : 0,
        .purgeable = hostOK ? (uint64_t)vm.purgeable_count * page : 0,
        .compressor = hostOK ? (uint64_t)vm.compressor_page_count * page : 0,
        .freeAndInactive = hostOK ? (uint64_t)(vm.free_count + vm.inactive_count +
            vm.speculative_count) * page : 0,
        .physFootprint = taskOK ? task.phys_footprint : 0,
        .taskCompressed = taskOK ? task.compressed : 0,
        .taskPurgeableResident = taskOK ? task.purgeable_volatile_resident : 0,
        .residentSize = taskOK ? task.resident_size : 0,
    };
}

static NSDictionary *sampleDictionary(ResidencySample sample, double millisecondsValue) {
    return @{
        @"elapsed_ms": @(millisecondsValue),
        @"host_sample_valid": @(sample.hostValid),
        @"task_sample_valid": @(sample.taskValid),
        @"system_wired_bytes": @(sample.wired),
        @"system_purgeable_bytes": @(sample.purgeable),
        @"system_compressor_bytes": @(sample.compressor),
        @"system_free_inactive_speculative_bytes": @(sample.freeAndInactive),
        @"process_phys_footprint_bytes": @(sample.physFootprint),
        @"process_compressed_bytes": @(sample.taskCompressed),
        @"process_purgeable_resident_bytes": @(sample.taskPurgeableResident),
        @"process_resident_size_bytes": @(sample.residentSize),
    };
}

static NSArray *settledSeries(const uint32_t *delaysMilliseconds, uint32_t count) {
    NSMutableArray *series = [NSMutableArray arrayWithCapacity:count];
    uint64_t started = mach_absolute_time();
    for (uint32_t index = 0; index < count; ++index) {
        if (delaysMilliseconds[index]) usleep(delaysMilliseconds[index] * 1000);
        [series addObject:sampleDictionary(residencySample(),
            elapsedMilliseconds(started, mach_absolute_time()))];
    }
    return series;
}

static double medianMetric(NSArray *series, NSString *key) {
    NSString *validityKey = [key hasPrefix:@"system_"] ? @"host_sample_valid" : @"task_sample_valid";
    double *values = calloc(series.count, sizeof(*values));
    NSUInteger count = 0;
    for (NSDictionary *sample in series)
        if ([sample[validityKey] boolValue]) values[count++] = [sample[key] doubleValue];
    if (!count) { free(values); return 0; }
    qsort(values, count, sizeof(*values), compareDouble);
    double result = count & 1 ? values[count / 2]
        : 0.5 * (values[count / 2 - 1] + values[count / 2]);
    free(values);
    return result;
}

static double madMetric(NSArray *series, NSString *key) {
    NSString *validityKey = [key hasPrefix:@"system_"] ? @"host_sample_valid" : @"task_sample_valid";
    double median = medianMetric(series, key);
    double *deviations = calloc(series.count, sizeof(*deviations));
    NSUInteger count = 0;
    for (NSDictionary *sample in series)
        if ([sample[validityKey] boolValue])
            deviations[count++] = fabs([sample[key] doubleValue] - median);
    if (!count) { free(deviations); return 0; }
    qsort(deviations, count, sizeof(*deviations), compareDouble);
    double result = count & 1 ? deviations[count / 2]
        : 0.5 * (deviations[count / 2 - 1] + deviations[count / 2]);
    free(deviations);
    return result;
}

static NSDictionary *metricSummary(NSArray *series, NSString *key) {
    NSString *validityKey = [key hasPrefix:@"system_"] ? @"host_sample_valid" : @"task_sample_valid";
    double minimum = INFINITY, maximum = -INFINITY;
    uint32_t count = 0;
    for (NSDictionary *sample in series) if ([sample[validityKey] boolValue]) {
        double value = [sample[key] doubleValue];
        minimum = fmin(minimum, value); maximum = fmax(maximum, value); ++count;
    }
    return @{@"median": @(medianMetric(series, key)), @"mad": @(madMetric(series, key)),
             @"min": @(count ? minimum : 0), @"max": @(count ? maximum : 0)};
}

static NSDictionary *seriesSummary(NSArray *series) {
    uint32_t hostValid = 0, taskValid = 0;
    for (NSDictionary *sample in series) {
        hostValid += [sample[@"host_sample_valid"] boolValue];
        taskValid += [sample[@"task_sample_valid"] boolValue];
    }
    return @{
        @"sample_count": @(series.count),
        @"elapsed_ms": @([series.lastObject[@"elapsed_ms"] doubleValue]),
        @"host_valid": @(hostValid), @"host_invalid": @(series.count - hostValid),
        @"task_valid": @(taskValid), @"task_invalid": @(series.count - taskValid),
        @"system_wired_bytes": metricSummary(series, @"system_wired_bytes"),
        @"system_compressor_bytes": metricSummary(series, @"system_compressor_bytes"),
        @"system_purgeable_bytes": metricSummary(series, @"system_purgeable_bytes"),
        @"process_phys_footprint_bytes": metricSummary(series, @"process_phys_footprint_bytes"),
        @"process_compressed_bytes": metricSummary(series, @"process_compressed_bytes"),
        @"process_purgeable_resident_bytes": metricSummary(series, @"process_purgeable_resident_bytes"),
    };
}

typedef struct {
    atomic_bool stop;
    uint64_t started;
} CompileTrace;

static void startCompileTrace(CompileTrace *trace, NSMutableArray *samples,
                              dispatch_group_t group) {
    trace->started = mach_absolute_time();
    atomic_init(&trace->stop, false);
    dispatch_group_async(group, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        while (!atomic_load_explicit(&trace->stop, memory_order_acquire)) {
            [samples addObject:sampleDictionary(residencySample(),
                elapsedMilliseconds(trace->started, mach_absolute_time()))];
            usleep(100000);
        }
        [samples addObject:sampleDictionary(residencySample(),
            elapsedMilliseconds(trace->started, mach_absolute_time()))];
    });
}

static NSArray *stopCompileTrace(CompileTrace *trace, NSMutableArray *samples,
                                 dispatch_group_t group) {
    atomic_store_explicit(&trace->stop, true, memory_order_release);
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    return [samples copy];
}

static NSArray *compactCompileTrace(NSArray *raw) {
    NSMutableArray *compact = [NSMutableArray array];
    double nextMillisecond = 0;
    for (NSDictionary *sample in raw) {
        double elapsed = [sample[@"elapsed_ms"] doubleValue];
        if (!compact.count || elapsed >= nextMillisecond) {
            [compact addObject:sample];
            nextMillisecond = floor(elapsed / 1000.0 + 1.0) * 1000.0;
        }
    }
    NSDictionary *last = raw.lastObject;
    if (last && compact.lastObject != last) [compact addObject:last];
    return compact;
}

static BOOL prepareOracles(NSString *checkpoint, const uint32_t experts[TouchCount],
                           float input[Hidden], float *expected[TouchCount]) {
    NSDictionary *layout = readJSON([checkpoint stringByAppendingPathComponent:
        @"packed_experts/layout.json"]);
    NSDictionary *layer = [layout[@"layers"] firstObject];
    NSData *blob = [NSData dataWithContentsOfFile:
        [[checkpoint stringByAppendingPathComponent:@"packed_experts"]
            stringByAppendingPathComponent:layer[@"file"] ?: @""]
        options:NSDataReadingMappedIfSafe error:nil];
    for (uint32_t column = 0; column < Hidden; ++column)
        input[column] = inputValue(column, 0);
    BOOL valid = blob && layer;
    for (uint32_t index = 0; index < TouchCount && valid; ++index) {
        NSDictionary *record = nil;
        for (NSDictionary *candidate in layer[@"experts"])
            if ([candidate[@"expert"] unsignedIntValue] == experts[index]) {
                record = candidate; break;
            }
        ANEFieldfareTensor gate = {0}, up = {0}, down = {0};
        uint64_t base = [record[@"offset"] unsignedLongLongValue];
        NSDictionary *tensors = record[@"tensors"];
        expected[index] = calloc(Hidden, sizeof(float));
        valid &= record && expected[index] &&
            resolveMatrix(blob, base, tensors, @"gate", &gate) &&
            resolveMatrix(blob, base, tensors, @"up", &up) &&
            resolveMatrix(blob, base, tensors, @"down", &down) &&
            expertOracle(gate, up, down, input, expected[index]);
    }
    return valid;
}

static void fillInput(ANEFieldfareANEProgram *program, uint32_t bank, const float *input) {
    IOSurfaceRef surface = ANEFieldfareANEProgramInput(program, bank);
    if (!surface || IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return;
    __fp16 *values = IOSurfaceGetBaseAddress(surface);
    for (uint32_t column = 0; column < Hidden; ++column)
        for (uint32_t position = 0; position < Positions; ++position)
            values[(uint64_t)column * Positions + position] = (__fp16)input[column];
    IOSurfaceUnlock(surface, 0, NULL);
}

static NSDictionary *touchProcedures(ANEFieldfareANEProgram *program,
                                     const uint32_t experts[TouchCount],
                                     float *expected[TouchCount],
                                     const float *input) {
    NSMutableArray *touches = [NSMutableArray array];
    BOOL valid = program != NULL;
    uint64_t totalComparisons = 0, totalFinite = 0;
    double maximumError = 0, squareError = 0;
    fillInput(program, 0, input);
    for (uint32_t index = 0; index < TouchCount && valid; ++index) {
        char procedure[32] = {0};
        snprintf(procedure, sizeof(procedure), "expert_%u", experts[index]);
        const char *rebindError = NULL, *evaluateError = NULL;
        uint64_t rebindStarted = mach_absolute_time();
        BOOL rebound = ANEFieldfareANEProgramRebindRequestBankToProcedureEvidence(
            program, 0, procedure, &rebindError);
        double rebindMs = elapsedMilliseconds(rebindStarted, mach_absolute_time());
        IOSurfaceRef output = ANEFieldfareANEProgramOutput(program, 0);
        if (output && IOSurfaceLock(output, 0, NULL) == kIOReturnSuccess) {
            memset(IOSurfaceGetBaseAddress(output), 0x7e, IOSurfaceGetAllocSize(output));
            IOSurfaceUnlock(output, 0, NULL);
        }
        ANEFieldfareANEEvaluationTiming timing = {0};
        BOOL evaluated = rebound && ANEFieldfareANEProgramEvaluate(
            program, 0, 5ULL * NSEC_PER_SEC, &timing, &evaluateError);
        uint64_t comparisonsBefore = totalComparisons, finiteBefore = totalFinite;
        double errorBefore = maximumError;
        BOOL compared = evaluated && compareExpertOutput(output, expected[index],
            &totalComparisons, &totalFinite, &maximumError, &squareError);
        uint64_t expectedComparisons = (uint64_t)Hidden * Positions;
        BOOL touchValid = rebound && evaluated && compared &&
            totalComparisons - comparisonsBefore == expectedComparisons &&
            totalFinite - finiteBefore == expectedComparisons && maximumError <= 0.08;
        [touches addObject:@{
            @"expert": @(experts[index]), @"procedure": @(procedure),
            @"rebind_ms": @(rebindMs), @"completion_ms": @(timing.completionMilliseconds),
            @"comparisons": @(totalComparisons - comparisonsBefore),
            @"finite": @(totalFinite - finiteBefore),
            @"max_absolute_error_through_touch": @(maximumError),
            @"max_absolute_error_before_touch": @(errorBefore),
            @"rebind_error": rebindError ? @(rebindError) : @"none",
            @"evaluate_error": evaluateError ? @(evaluateError) : @"none",
            @"valid": @(touchValid),
        }];
        free((void *)rebindError); free((void *)evaluateError);
        valid &= touchValid;
    }
    return @{
        @"valid": @(valid), @"touches": touches,
        @"comparisons": @(totalComparisons), @"finite": @(totalFinite),
        @"max_absolute_error": @(maximumError),
        @"rmse": @(totalComparisons ? sqrt(squareError / totalComparisons) : 0),
        @"tolerance": @0.08,
    };
}

static ANEFieldfareANEProgram *createHandle(NSString *mil, NSData *weights,
                                            uint32_t identity,
                                            ANEFieldfareANEProgramInfo *info,
                                            NSString **errorText) {
    NSString *identityMIL = identity == 0 ? mil : [mil stringByReplacingOccurrencesOfString:
        @"gate_x_0_begin_0" withString:[NSString stringWithFormat:@"gate_x_0_begin_0_h%u", identity]];
    uint32_t experts[8] = {0, 13, 27, 40, 54, 67, 81, 95};
    char storage[8][32] = {{0}}; const char *names[8] = {0};
    for (uint32_t bank = 0; bank < 8; ++bank) {
        snprintf(storage[bank], sizeof(storage[bank]), "expert_%u", experts[bank]);
        names[bank] = storage[bank];
    }
    const char *error = NULL;
    ANEFieldfareANEProgram *program =
        ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
            identityMIL.UTF8String, "weight.bin", weights.bytes, weights.length,
            names, 8, &error);
    if (program && info) *info = ANEFieldfareANEProgramGetInfo(program);
    if (errorText) *errorText = error ? @(error) : @"none";
    free((void *)error);
    return program;
}

static NSDictionary *phaseSummary(NSArray *baseline, NSArray *loaded,
                                  NSArray *touched, NSArray *destroyed,
                                  uint64_t compactBytes, uint64_t fp16Bytes) {
    NSString *wired = @"system_wired_bytes", *phys = @"process_phys_footprint_bytes";
    double baselineWired = medianMetric(baseline, wired);
    double loadedWired = medianMetric(loaded, wired);
    double touchedWired = medianMetric(touched, wired);
    double destroyedWired = medianMetric(destroyed, wired);
    double pooledNoise = fmax(madMetric(baseline, wired),
        fmax(madMetric(loaded, wired), madMetric(destroyed, wired)));
    double loadDelta = touchedWired - baselineWired;
    double releaseDelta = touchedWired - destroyedWired;
    BOOL directionsAgree = loadDelta > 0 && releaseDelta > 0;
    BOOL separatedFromObservedNoise = directionsAgree &&
        fmin(loadDelta, releaseDelta) > 3.0 * fmax(1.0, pooledNoise);
    double evidenceBytes = directionsAgree ? sqrt(loadDelta * releaseDelta) : 0;
    double compactDistance = evidenceBytes > 0 ? fabs(log(evidenceBytes / compactBytes)) : INFINITY;
    double fp16Distance = evidenceBytes > 0 ? fabs(log(evidenceBytes / fp16Bytes)) : INFINITY;
    NSString *nearest = !separatedFromObservedNoise ? @"unresolved_at_observed_noise" :
        (compactDistance < fp16Distance ? @"closer_to_stored_compact_bytes" :
                                         @"closer_to_logical_fp16_bytes");
    return @{
        @"median_system_wired": @{
            @"baseline": @(baselineWired), @"post_load": @(loadedWired),
            @"post_touch": @(touchedWired), @"post_destroy": @(destroyedWired),
            @"post_touch_minus_baseline": @(loadDelta),
            @"post_touch_minus_post_destroy": @(releaseDelta),
            @"maximum_phase_mad": @(pooledNoise),
        },
        @"median_process_phys_footprint": @{
            @"baseline": @(medianMetric(baseline, phys)),
            @"post_load": @(medianMetric(loaded, phys)),
            @"post_touch": @(medianMetric(touched, phys)),
            @"post_destroy": @(medianMetric(destroyed, phys)),
        },
        @"hypothesis_scale": @{
            @"stored_compact_bytes": @(compactBytes), @"logical_fp16_bytes": @(fp16Bytes),
            @"bidirectional_geometric_mean_evidence_bytes": @(evidenceBytes),
            @"load_and_release_directions_agree": @(directionsAgree),
            @"separated_from_phase_mad": @(separatedFromObservedNoise),
            @"log_distance_to_compact": @(isfinite(compactDistance) ? compactDistance : -1),
            @"log_distance_to_fp16": @(isfinite(fp16Distance) ? fp16Distance : -1),
            @"classification": nearest,
            @"classification_rule": @"Classify only when both load and release are positive and their smaller magnitude exceeds three times the largest phase median absolute deviation; then report the nearer byte hypothesis by logarithmic distance.",
        },
        @"accounting_boundary": @"System wired, purgeable, and compressor counts include unrelated host activity. Process phys_footprint, compressed, purgeable, and resident_size are task ledgers and do not identify ANE physical pages. The classification is controlled lifecycle evidence, not byte-exact physical residency proof.",
    };
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3 || argc > 4) {
            emit(reject(@"arguments",
                @"usage: residency-gate <96-experts.mlmodelc> <gturbo-root> [scaling-handles-2...4]", nil));
            return 2;
        }
        NSString *modelRoot = [@(argv[1]) stringByStandardizingPath];
        NSString *checkpoint = [@(argv[2]) stringByStandardizingPath];
        uint32_t scalingRequested = argc == 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : 2;
        scalingRequested = MAX(2u, MIN(MaximumScalingHandles, scalingRequested));
        NSData *milData = [NSData dataWithContentsOfFile:
            [modelRoot stringByAppendingPathComponent:@"model.mil"]];
        NSData *weights = [NSData dataWithContentsOfFile:
            [modelRoot stringByAppendingPathComponent:@"weights/weight.bin"]
            options:NSDataReadingMappedIfSafe error:nil];
        NSString *mil = milData ? [[NSString alloc] initWithData:milData
            encoding:NSUTF8StringEncoding] : nil;
        NSUInteger functions = 0;
        for (uint32_t expert = 0; expert < ProcedureCount; ++expert)
            functions += [mil containsString:[NSString stringWithFormat:@"func expert_%u<ios18>", expert]];
        if (!mil || !weights || functions != ProcedureCount ||
            occurrences(mil, @"constexpr_blockwise_shift_scale") != 99 * ProcedureCount) {
            emit(reject(@"artifact_inspection",
                @"expected one artifact with expert_0...expert_95 and 9,504 uint4 constexpr slices",
                @{@"function_count": @(functions), @"stored_weight_blob_bytes": @(weights.length)}));
            return 1;
        }
        uint32_t touchedExperts[TouchCount] = {0, 13, 27, 40, 54, 67, 81, 95};
        float input[Hidden] = {0}; float *expected[TouchCount] = {0};
        BOOL valid = prepareOracles(checkpoint, touchedExperts, input, expected);
        if (!valid) {
            emit(reject(@"checkpoint_oracle", @"authentic compact checkpoint oracle setup failed", nil));
            for (uint32_t i = 0; i < TouchCount; ++i) free(expected[i]);
            return 1;
        }
        uint64_t fp16Bytes = (uint64_t)ProcedureCount * 3 * Rows * Hidden * 2;
        const uint32_t baselineDelays[] = {0, 200, 200, 200, 200, 200, 200};
        const uint32_t settledDelays[] = {0, 250, 500, 1000, 2000, 2000, 4000};
        NSArray *baseline = settledSeries(baselineDelays, 7);

        CompileTrace trace = {0}; dispatch_group_t traceGroup = dispatch_group_create();
        NSMutableArray *traceSamples = [NSMutableArray array];
        startCompileTrace(&trace, traceSamples, traceGroup);
        ANEFieldfareANEProgramInfo primaryInfo = {0}; NSString *primaryError = nil;
        ANEFieldfareANEProgram *primary = createHandle(mil, weights, 0, &primaryInfo, &primaryError);
        NSArray *rawCompileLoadTrace = stopCompileTrace(&trace, traceSamples, traceGroup);
        NSArray *compileLoadTrace = compactCompileTrace(rawCompileLoadTrace);
        valid &= primary != NULL;
        NSArray *loadedSeries = primary ? settledSeries(settledDelays, 7) : @[];
        NSDictionary *touchResult = primary ? touchProcedures(primary, touchedExperts, expected, input)
                                             : @{@"valid": @NO};
        valid &= [touchResult[@"valid"] boolValue];
        NSArray *touchedSeries = primary ? settledSeries(settledDelays, 7) : @[];
        if (primary) ANEFieldfareANEProgramDestroy(primary);
        NSArray *destroyedSeries = settledSeries(settledDelays, 7);
        NSDictionary *controlledSummary = phaseSummary(baseline, loadedSeries, touchedSeries,
            destroyedSeries, weights.length, fp16Bytes);

        // Separate scaling phase. It cannot contaminate the primary handle's
        // load/touch/unload comparison. Descriptors are made distinct by a
        // metadata-only MIL name salt; constant payload bytes remain identical,
        // so any driver-level sharing is measured rather than assumed absent.
        ResidencySample beforeScaling = residencySample();
        uint64_t conservativeNextHandle = MAX(fp16Bytes,
            (uint64_t)llabs((long long)(medianMetric(touchedSeries, @"system_wired_bytes") -
                                        medianMetric(baseline, @"system_wired_bytes"))));
        uint64_t available = beforeScaling.freeAndInactive;
        uint32_t safeRequested = scalingRequested;
        while (safeRequested > 2 && conservativeNextHandle * safeRequested > available)
            --safeRequested;
        BOOL scalingSafe = conservativeNextHandle * 2 <= available;
        ANEFieldfareANEProgram *scaling[MaximumScalingHandles] = {0};
        NSMutableArray *scalingLoads = [NSMutableArray array];
        uint32_t scalingLoaded = 0;
        NSString *scalingFailure = scalingSafe ? @"none" : @"insufficient free+inactive+speculative bytes for two logical-FP16 handle budgets";
        if (scalingSafe) for (uint32_t index = 0; index < safeRequested; ++index) {
            ResidencySample before = residencySample();
            ANEFieldfareANEProgramInfo info = {0}; NSString *error = nil;
            scaling[index] = createHandle(mil, weights, index + 1, &info, &error);
            ResidencySample immediate = residencySample();
            const uint32_t shortSettle[] = {250, 750, 1000};
            NSArray *settled = scaling[index] ? settledSeries(shortSettle, 3) : @[];
            [scalingLoads addObject:@{
                @"ordinal": @(index + 1), @"loaded": @(scaling[index] != NULL),
                @"compile_ms": @(info.compileMilliseconds), @"load_ms": @(info.loadMilliseconds),
                @"immediate_wired_delta_bytes": @((int64_t)immediate.wired - (int64_t)before.wired),
                @"settled_median_wired_delta_bytes": @((int64_t)medianMetric(settled, @"system_wired_bytes") - (int64_t)before.wired),
                @"immediate_phys_delta_bytes": @((int64_t)immediate.physFootprint - (int64_t)before.physFootprint),
                @"settled_median_phys_delta_bytes": @((int64_t)medianMetric(settled, @"process_phys_footprint_bytes") - (int64_t)before.physFootprint),
                @"settled_validity": @{ @"host_invalid": @([seriesSummary(settled)[@"host_invalid"] unsignedIntValue]),
                    @"task_invalid": @([seriesSummary(settled)[@"task_invalid"] unsignedIntValue]) },
                @"error": error ?: @"none",
            }];
            if (!scaling[index]) { scalingFailure = error ?: @"unknown"; break; }
            ++scalingLoaded;
        }
        NSDictionary *scalingOracle = scalingLoaded ?
            touchProcedures(scaling[scalingLoaded - 1], touchedExperts, expected, input) :
            @{@"valid": @NO, @"reason": @"no scaling handle loaded"};
        if (scalingSafe) valid &= scalingLoaded >= 2 && [scalingOracle[@"valid"] boolValue];
        NSMutableArray *scalingDestroys = [NSMutableArray array];
        for (uint32_t remaining = scalingLoaded; remaining > 0; --remaining) {
            ResidencySample before = residencySample();
            ANEFieldfareANEProgramDestroy(scaling[remaining - 1]);
            const uint32_t shortSettle[] = {250, 750, 1000};
            NSArray *settled = settledSeries(shortSettle, 3);
            [scalingDestroys addObject:@{@"destroyed_ordinal": @(remaining),
                @"settled_median_wired_delta_bytes": @((int64_t)medianMetric(settled, @"system_wired_bytes") - (int64_t)before.wired),
                @"settled_median_phys_delta_bytes": @((int64_t)medianMetric(settled, @"process_phys_footprint_bytes") - (int64_t)before.physFootprint),
                @"settled_validity": @{ @"host_invalid": @([seriesSummary(settled)[@"host_invalid"] unsignedIntValue]),
                    @"task_invalid": @([seriesSummary(settled)[@"task_invalid"] unsignedIntValue]) }}];
        }

        NSString *command = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXACT_COMMAND"];
        if (!command.length) command = [NSString stringWithFormat:@"%s %s %s %u",
            argv[0], argv[1], argv[2], scalingRequested];
        BOOL primaryNumerical = [touchResult[@"valid"] boolValue];
        BOOL validated = valid && primaryNumerical && (!scalingSafe || scalingLoaded >= 2);
        emit(@{
            @"schema": @"ane-fieldfare.aot-procedure-residency.v1",
            @"status": validated ? @"validated" : @"rejected",
            @"stage": validated ? @"controlled_load_touch_unload_residency" : @"load_touch_unload_or_oracle",
            @"host_chip": sysctlString("machdep.cpu.brand_string"),
            @"os_build": sysctlString("kern.osversion"), @"ane_architecture": @"H17G",
            @"exact_command": command, @"exit_status": @(validated ? 0 : 1),
            @"artifact": @{
                @"compiled_path": modelRoot, @"compiler_named_procedure_count": @(ProcedureCount),
                @"constexpr_uint4_slice_count": @(99 * ProcedureCount),
                @"stored_compact_weight_blob_bytes": @(weights.length),
                @"logical_fp16_weight_bytes": @(fp16Bytes),
                @"logical_fp16_over_stored": @((double)fp16Bytes / weights.length),
                @"compiled_total_bytes": @(directoryBytes(modelRoot)),
            },
            @"primary_lifecycle": @{
                @"compile_count": primary ? @1 : @0, @"load_count": primary ? @1 : @0,
                @"compile_ms": @(primaryInfo.compileMilliseconds),
                @"load_ms": @(primaryInfo.loadMilliseconds), @"error": primaryError ?: @"none",
                @"baseline_summary": seriesSummary(baseline),
                @"compile_load_memory_summary": seriesSummary(rawCompileLoadTrace),
                @"trace_boundary_labels": @{
                    @"call_started_ms": @0,
                    @"raw_sampler_period_ms": @100,
                    @"raw_sample_count": @(rawCompileLoadTrace.count),
                    @"retained_trace_sample_count": @(compileLoadTrace.count),
                    @"compiler_reported_compile_end_ms": @(primaryInfo.compileMilliseconds),
                    @"compiler_reported_load_end_ms": @(primaryInfo.compileMilliseconds + primaryInfo.loadMilliseconds),
                    @"note": @"ANEProgram creation is one synchronous host call. Memory was sampled at 100 ms and compacted to approximately one retained point per second; the trace is labelled using runtime-reported compile and load durations rather than pretending there was a host-visible split call.",
                },
                @"post_load_settled_summary": seriesSummary(loadedSeries),
                @"wide_procedure_touch": touchResult,
                @"post_touch_settled_summary": seriesSummary(touchedSeries),
                @"post_destroy_settled_summary": seriesSummary(destroyedSeries),
            },
            @"controlled_residency_evidence": controlledSummary,
            @"scaling_phase": @{
                @"separate_from_primary_control": @YES,
                @"requested_handles": @(scalingRequested), @"safe_requested_handles": @(safeRequested),
                @"loaded_simultaneous_handles": @(scalingLoaded),
                @"available_free_inactive_speculative_before_bytes": @(available),
                @"conservative_per_handle_budget_bytes": @(conservativeNextHandle),
                @"safe_to_attempt_two": @(scalingSafe),
                @"safety_rule": @"Require free+inactive+speculative memory to cover one logical-FP16 byte budget per simultaneous handle; reduce requested count but never attempt fewer than two.",
                @"descriptor_identity": @"Distinct compiler descriptor metadata name per handle; identical authentic constant payload bytes. Constant-page sharing or deduplication is therefore possible and not ruled out.",
                @"loads": scalingLoads, @"last_handle_wide_procedure_oracle": scalingOracle,
                @"reverse_destroy_series": scalingDestroys, @"failure": scalingFailure,
            },
            @"numerical_result": @{
                @"oracle": @"ANEFieldfareAffineMatVec directly over authentic compact checkpoint uint4 codes and BF16 scale/bias with compiler-graph FP16 boundaries",
                @"touched_experts": @[@0, @13, @27, @40, @54, @67, @81, @95],
                @"primary": touchResult, @"scaling_last_handle": scalingOracle,
            },
            @"claim_boundary": validated
                ? @"One authentic 96-procedure block-64 expert artifact was observed through repeated settled memory samples before creation, across a 100 ms compile/load trace, after load, after exact evaluations spanning experts 0 through 95, and after destroy. Every touched procedure passed the independent compact-checkpoint oracle. A separate bounded phase also loaded at least two simultaneously live, metadata-distinct handles and oracle-checked the last handle. System wired/compressor/purgeable values remain noisy host-wide ledgers and process values remain accounting ledgers; the receipt classifies lifecycle evidence relative to compact and FP16 byte hypotheses but does not assert byte-exact physical residency or lack of driver deduplication."
                : @"The controlled artifact lifecycle or exact wide-procedure oracle failed, or two-handle scaling was safe but did not load. No residency classification is promoted.",
        });
        for (uint32_t index = 0; index < TouchCount; ++index) free(expected[index]);
        return validated ? 0 : 1;
    }
}
