#define main ANEFieldfareAOTBlockSliceReferenceMain
#include "../ANEFieldfareAOTBlockSliceGate/main.m"
#undef main

#import <mach/mach_time.h>

static double selectionMilliseconds(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)(end - start) * timebase.numer / timebase.denom / 1e6;
}

static BOOL selectionOracle(ANEFieldfareTensor gate, ANEFieldfareTensor up,
                            ANEFieldfareTensor down, const float *input,
                            float *output) {
    float *g = calloc(Rows, sizeof(float));
    float *u = calloc(Rows, sizeof(float));
    float *activated = calloc(Rows, sizeof(float));
    BOOL ok = g && u && activated &&
        ANEFieldfareAffineMatVec(&gate, 4, Group, input, g) &&
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

static BOOL expectedForExpert(NSData *blob, NSDictionary *layer,
                              uint32_t expert, const float *input,
                              float *expected) {
    NSDictionary *record = nil;
    for (NSDictionary *candidate in layer[@"experts"])
        if ([candidate[@"expert"] unsignedIntValue] == expert) {
            record = candidate;
            break;
        }
    NSDictionary *tensors = record[@"tensors"];
    uint64_t base = [record[@"offset"] unsignedLongLongValue];
    ANEFieldfareTensor gate = {0}, up = {0}, down = {0};
    return record && resolveMatrix(blob, base, tensors, @"gate", &gate) &&
        resolveMatrix(blob, base, tensors, @"up", &up) &&
        resolveMatrix(blob, base, tensors, @"down", &down) &&
        selectionOracle(gate, up, down, input, expected);
}

static BOOL fillInput(IOSurfaceRef surface, const float *input) {
    if (!surface || IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return NO;
    __fp16 *base = IOSurfaceGetBaseAddress(surface);
    for (uint32_t column = 0; column < Hidden; ++column)
        for (uint32_t position = 0; position < Positions; ++position)
            base[(uint64_t)column * Positions + position] = (__fp16)input[column];
    IOSurfaceUnlock(surface, 0, NULL);
    return YES;
}

static BOOL poisonOutput(IOSurfaceRef surface) {
    if (!surface || IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return NO;
    float *base = IOSurfaceGetBaseAddress(surface);
    for (uint64_t index = 0; index < (uint64_t)Hidden * Positions; ++index)
        base[index] = NAN;
    IOSurfaceUnlock(surface, 0, NULL);
    return YES;
}

static BOOL compareSelection(IOSurfaceRef surface, const float *expected,
                             uint64_t *comparisons, uint64_t *finite,
                             double *maximumError) {
    if (!surface || IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess)
        return NO;
    const float *actual = IOSurfaceGetBaseAddress(surface);
    for (uint32_t row = 0; row < Hidden; ++row)
        for (uint32_t position = 0; position < Positions; ++position) {
            float value = actual[(uint64_t)row * Positions + position];
            *finite += isfinite(value);
            *maximumError = fmax(*maximumError, fabs((double)value - expected[row]));
            ++*comparisons;
        }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc == 2 && !strcmp(argv[1], "--rejection-smoke")) {
            const char *name = "expert_0";
            const char *createError = NULL;
            uint32_t mapped = UINT32_MAX;
            ANEFieldfareANESharedSurfaceRequestSet *set =
                ANEFieldfareANEProgramCreateSharedSurfaceRequestSet(
                    NULL, 0, &name, 1, &mapped, &createError);
            const char *beginError = NULL;
            BOOL beginRejected = !ANEFieldfareANESharedSurfaceRequestSetEvaluateBegin(
                NULL, 0, NULL, &beginError);
            const char *unmapError = NULL;
            BOOL unmapRejected = !ANEFieldfareANESharedSurfaceRequestSetUnmapAll(
                NULL, NULL, &unmapError);
            BOOL valid = !set && mapped == 0 && createError &&
                beginRejected && beginError && unmapRejected && unmapError;
            emit(@{@"schema": @"ane-fieldfare.aot-shared-surface-rejection-smoke.v1",
                @"status": valid ? @"validated" : @"rejected",
                @"mapped_before_failure": @(mapped),
                @"null_create_rejected": @(!set),
                @"null_begin_rejected": @(beginRejected),
                @"null_unmap_rejected": @(unmapRejected)});
            free((void *)createError); free((void *)beginError);
            free((void *)unmapError);
            return valid ? 0 : 1;
        }
        if (argc != 3) {
            emit(reject(@"arguments",
                @"usage: shared-surface-selection-gate <layer.mlmodelc> <gturbo-root>", nil));
            return 2;
        }
        NSString *modelRoot = @(argv[1]);
        NSString *checkpoint = @(argv[2]);
        NSData *milData = [NSData dataWithContentsOfFile:
            [modelRoot stringByAppendingPathComponent:@"model.mil"]];
        NSData *weights = [NSData dataWithContentsOfFile:
            [modelRoot stringByAppendingPathComponent:@"weights/weight.bin"]];
        NSString *mil = milData ? [[NSString alloc] initWithData:milData
            encoding:NSUTF8StringEncoding] : nil;
        BOOL valid = mil && weights;
        for (uint32_t expert = 0; expert < 64 && valid; ++expert)
            valid &= [mil containsString:[NSString stringWithFormat:
                @"func expert_%u<ios18>", expert]];

        NSDictionary *layout = readJSON(
            [checkpoint stringByAppendingPathComponent:@"packed_experts/layout.json"]);
        NSDictionary *layer = layout[@"layers"][0];
        NSData *blob = [NSData dataWithContentsOfFile:
            [[checkpoint stringByAppendingPathComponent:@"packed_experts"]
                stringByAppendingPathComponent:layer[@"file"]]
            options:NSDataReadingMappedIfSafe error:nil];
        float input[Hidden];
        for (uint32_t column = 0; column < Hidden; ++column)
            input[column] = inputValue(column, 0);
        uint32_t oracleExperts[8] = {0};
        float *expected[8] = {0};
        for (uint32_t index = 0; index < 8; ++index)
            oracleExperts[index] = index * 63 / 7;
        for (uint32_t index = 0; index < 8 && valid; ++index) {
            expected[index] = calloc(Hidden, sizeof(float));
            valid &= expected[index] && blob && expectedForExpert(
                blob, layer, oracleExperts[index], input, expected[index]);
        }

        const char *createError = NULL;
        char initialNameStorage[8][32] = {{0}};
        const char *initialNames[8] = {0};
        for (uint32_t bank = 0; bank < 8; ++bank) {
            snprintf(initialNameStorage[bank], sizeof(initialNameStorage[bank]),
                     "expert_%u", oracleExperts[bank]);
            initialNames[bank] = initialNameStorage[bank];
        }
        uint64_t createStarted = mach_absolute_time();
        ANEFieldfareANEProgram *program = valid ?
            ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
                mil.UTF8String, "weight.bin", weights.bytes, weights.length,
                initialNames, 8, &createError) : NULL;
        double createMilliseconds = selectionMilliseconds(
            createStarted, mach_absolute_time());
        valid &= program != NULL;
        for (uint32_t bank = 0; bank < 8 && valid; ++bank)
            valid &= fillInput(ANEFieldfareANEProgramInput(program, bank), input);
        IOSurfaceRef sharedInput = ANEFieldfareANEProgramInput(program, 0);
        IOSurfaceRef sharedOutput = ANEFieldfareANEProgramOutput(program, 0);

        const char *smallNames[2] = {"expert_0", "expert_63"};
        const char *smallError = NULL;
        uint32_t smallMapAttempt = 0;
        uint64_t smallMapStarted = mach_absolute_time();
        ANEFieldfareANESharedSurfaceRequestSet *small = valid ?
            ANEFieldfareANEProgramCreateSharedSurfaceRequestSet(
                program, 0, smallNames, 2, &smallMapAttempt, &smallError) : NULL;
        double smallMapMilliseconds = selectionMilliseconds(
            smallMapStarted, mach_absolute_time());
        uint32_t smallMapped = ANEFieldfareANESharedSurfaceRequestSetMappedCount(small);
        NSMutableArray *descriptions = [NSMutableArray array];
        for (uint32_t index = 0; index < smallMapped; ++index) {
            CFTypeRef raw = ANEFieldfareANESharedSurfaceRequestSetCopyRequestDescription(
                small, index);
            [descriptions addObject:raw ? CFBridgingRelease(raw) : @"unavailable"];
        }
        valid &= small && smallMapped == 2;
        const char *ownedEvaluateError = NULL;
        ANEFieldfareANEEvaluationTiming forbiddenTiming = {0};
        BOOL originalEvaluateRejectedWhileOwned = small &&
            !ANEFieldfareANEProgramEvaluateBegin(
                program, 0, &forbiddenTiming, &ownedEvaluateError);
        uint32_t competingMapAttempt = 0;
        const char *competingCatalogError = NULL;
        ANEFieldfareANESharedSurfaceRequestSet *competing = small ?
            ANEFieldfareANEProgramCreateSharedSurfaceRequestSet(
                program, 0, smallNames, 2, &competingMapAttempt,
                &competingCatalogError) : NULL;
        BOOL competingCatalogRejectedWhileOwned = !competing &&
            competingMapAttempt == 0;
        valid &= originalEvaluateRejectedWhileOwned &&
            competingCatalogRejectedWhileOwned;
        uint32_t smallSequence[3] = {0, 1, 0};
        uint32_t smallOracle[3] = {0, 7, 0};
        NSMutableArray *smallTimings = [NSMutableArray array];
        uint64_t comparisons = 0, finite = 0;
        double maximumError = 0;
        for (uint32_t step = 0; step < 3 && valid; ++step) {
            valid &= poisonOutput(sharedOutput);
            ANEFieldfareANEEvaluationTiming timing = {0};
            const char *error = NULL;
            uint64_t started = mach_absolute_time();
            BOOL evaluated = ANEFieldfareANESharedSurfaceRequestSetEvaluate(
                small, smallSequence[step], 5000000000ull, &timing, &error);
            double wall = selectionMilliseconds(started, mach_absolute_time());
            [smallTimings addObject:@{@"expert": @(smallSequence[step] ? 63 : 0),
                @"wall_ms": @(wall), @"enqueue_ms": @(timing.enqueueMilliseconds),
                @"completion_ms": @(timing.completionMilliseconds)}];
            if (!evaluated && !smallError) smallError = error;
            valid &= evaluated && compareSelection(sharedOutput,
                expected[smallOracle[step]], &comparisons, &finite, &maximumError);
            free((void *)error);
        }
        uint32_t smallUnmapped = 0;
        const char *smallUnmapError = NULL;
        valid &= small && ANEFieldfareANESharedSurfaceRequestSetUnmapAll(
            small, &smallUnmapped, &smallUnmapError) && smallUnmapped == 2;

        // Production-shaped scaling gate: every compiler procedure is mapped
        // over each of eight distinct physical carrier pairs. This creates 512
        // request objects but no additional program handle or IOSurface.
        char largeNameStorage[64][32] = {{0}};
        const char *largeNames[64] = {0};
        for (uint32_t expert = 0; expert < 64; ++expert) {
            snprintf(largeNameStorage[expert], sizeof(largeNameStorage[expert]),
                     "expert_%u", expert);
            largeNames[expert] = largeNameStorage[expert];
        }
        const char *largeErrors[8] = {0};
        uint32_t largeMapAttempts[8] = {0};
        ANEFieldfareANESharedSurfaceRequestSet *large[8] = {0};
        MemorySample beforeLargeMap = memorySample();
        uint64_t largeMapStarted = mach_absolute_time();
        uint32_t largeMapped = 0;
        for (uint32_t bank = 0; bank < 8 && valid; ++bank) {
            large[bank] = ANEFieldfareANEProgramCreateSharedSurfaceRequestSet(
                program, bank, largeNames, 64, &largeMapAttempts[bank],
                &largeErrors[bank]);
            valid &= large[bank] != NULL && largeMapAttempts[bank] == 64;
            largeMapped += ANEFieldfareANESharedSurfaceRequestSetMappedCount(large[bank]);
        }
        double largeMapMilliseconds = selectionMilliseconds(
            largeMapStarted, mach_absolute_time());
        MemorySample afterLargeMap = memorySample();
        valid &= largeMapped == 512;

        ANEFieldfareANEEvaluationTiming largeTimings[8] = {0};
        const char *largeEvaluationErrors[8] = {0};
        uint64_t concurrentStarted = mach_absolute_time();
        for (uint32_t bank = 0; bank < 8 && valid; ++bank) {
            valid &= poisonOutput(ANEFieldfareANEProgramOutput(program, bank));
            valid &= ANEFieldfareANESharedSurfaceRequestSetEvaluateBegin(
                large[bank], oracleExperts[bank], &largeTimings[bank],
                &largeEvaluationErrors[bank]);
        }
        for (uint32_t bank = 0; bank < 8 && valid; ++bank)
            valid &= ANEFieldfareANESharedSurfaceRequestSetEvaluateWait(
                large[bank], 5000000000ull, &largeTimings[bank],
                &largeEvaluationErrors[bank]);
        double concurrentWallMilliseconds = selectionMilliseconds(
            concurrentStarted, mach_absolute_time());
        for (uint32_t bank = 0; bank < 8 && valid; ++bank)
            valid &= compareSelection(ANEFieldfareANEProgramOutput(program, bank),
                expected[bank], &comparisons, &finite, &maximumError);

        uint32_t largeUnmapped = 0;
        const char *largeUnmapErrors[8] = {0};
        for (uint32_t bank = 0; bank < 8 && valid; ++bank) {
            uint32_t count = 0;
            valid &= ANEFieldfareANESharedSurfaceRequestSetUnmapAll(
                large[bank], &count, &largeUnmapErrors[bank]);
            largeUnmapped += count;
        }
        valid &= largeUnmapped == 512;

        // Unmapping the selected-request catalogs must not disturb the original
        // request that owns the carrier pair.
        ANEFieldfareANEEvaluationTiming originalTimings[8] = {0};
        const char *originalErrors[8] = {0};
        BOOL originalSurvived = valid;
        for (uint32_t bank = 0; bank < 8 && originalSurvived; ++bank) {
            originalSurvived &= poisonOutput(ANEFieldfareANEProgramOutput(program, bank));
            originalSurvived &= ANEFieldfareANEProgramEvaluateBegin(
                program, bank, &originalTimings[bank], &originalErrors[bank]);
        }
        for (uint32_t bank = 0; bank < 8 && originalSurvived; ++bank)
            originalSurvived &= ANEFieldfareANEProgramEvaluateWait(
                program, bank, 5000000000ull, &originalTimings[bank],
                &originalErrors[bank]);
        for (uint32_t bank = 0; bank < 8 && originalSurvived; ++bank)
            originalSurvived &= compareSelection(
                ANEFieldfareANEProgramOutput(program, bank), expected[bank],
                &comparisons, &finite, &maximumError);
        valid &= originalSurvived;
        uint64_t expectedComparisons = (uint64_t)(3 + 8 + 8) * Hidden * Positions;
        valid &= comparisons == expectedComparisons && finite == comparisons &&
            maximumError <= 0.08;

        NSMutableArray *largeTimingReceipt = [NSMutableArray array];
        NSMutableArray *largeMapAttemptReceipt = [NSMutableArray array];
        NSMutableArray *largeErrorReceipt = [NSMutableArray array];
        NSMutableArray *originalErrorReceipt = [NSMutableArray array];
        for (uint32_t bank = 0; bank < 8; ++bank) {
            [largeTimingReceipt addObject:@{
                @"bank": @(bank), @"expert": @(oracleExperts[bank]),
                @"enqueue_ms": @(largeTimings[bank].enqueueMilliseconds),
                @"completion_ms": @(largeTimings[bank].completionMilliseconds),
            }];
            [largeMapAttemptReceipt addObject:@(largeMapAttempts[bank])];
            const char *largeFailure = largeErrors[bank] ?: largeEvaluationErrors[bank];
            [largeErrorReceipt addObject:@(largeFailure ?: "none")];
            [originalErrorReceipt addObject:@(originalErrors[bank] ?: "none")];
        }
        NSString *command = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXACT_COMMAND"];
        if (!command.length) command = [NSString stringWithFormat:@"%s %s %s",
            argv[0], argv[1], argv[2]];

        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.aot-shared-surface-selection.v1",
            @"status": valid ? @"validated" : @"rejected",
            @"host_chip": sysctlString("machdep.cpu.brand_string"),
            @"os_build": sysctlString("kern.osversion"),
            @"ane_architecture": @"H17G", @"exact_command": command,
            @"exit_status": @(valid ? 0 : 1),
            @"artifact": modelRoot,
            @"lifecycle": @{
                @"program_handles": @(program ? 1 : 0), @"compile_count": @(program ? 1 : 0),
                @"load_count": @(program ? 1 : 0), @"initial_request_mappings": @(program ? 8 : 0),
                @"physical_input_surfaces": @(program ? 8 : 0),
                @"physical_output_surfaces": @(program ? 8 : 0),
                @"small_shared_request_maps": @(smallMapped),
                @"small_shared_request_unmaps": @(smallUnmapped),
                @"original_evaluate_rejected_while_catalog_owned":
                    @(originalEvaluateRejectedWhileOwned),
                @"competing_catalog_rejected_while_owned":
                    @(competingCatalogRejectedWhileOwned),
                @"scaled_shared_request_maps": @(largeMapped),
                @"scaled_shared_request_unmaps": @(largeUnmapped),
                @"scaled_map_attempts_per_bank": largeMapAttemptReceipt,
                @"total_request_objects_constructed": @(8 + smallMapped + largeMapped),
                @"peak_simultaneously_mapped_request_objects": @(8 + largeMapped),
                @"shared_input_surface_preserved": @(sharedInput == ANEFieldfareANEProgramInput(program, 0)),
                @"shared_output_surface_preserved": @(sharedOutput == ANEFieldfareANEProgramOutput(program, 0)),
                @"original_request_survived_unmaps": @(originalSurvived),
            },
            @"memory": @{
                @"task_footprint_before_512_maps_bytes": @(beforeLargeMap.footprint),
                @"task_footprint_after_512_maps_bytes": @(afterLargeMap.footprint),
                @"task_footprint_delta_bytes": @((int64_t)afterLargeMap.footprint -
                                                   (int64_t)beforeLargeMap.footprint),
                @"system_wired_before_bytes": @(beforeLargeMap.wired),
                @"system_wired_after_bytes": @(afterLargeMap.wired),
                @"system_wired_delta_bytes": @((int64_t)afterLargeMap.wired -
                                                 (int64_t)beforeLargeMap.wired),
            },
            @"timing": @{
                @"program_create_ms": @(createMilliseconds),
                @"small_map_ms": @(smallMapMilliseconds),
                @"scaled_map_ms": @(largeMapMilliseconds),
                @"small_selection": smallTimings,
                @"eight_begin_then_wait_wall_ms": @(concurrentWallMilliseconds),
                @"eight_begin_then_wait": largeTimingReceipt,
            },
            @"concurrency_boundary": @{
                @"host_submissions_before_first_wait": @8,
                @"distinct_surface_pairs": @8,
                @"hardware_parallelism_measured": @NO,
            },
            @"request_descriptions": descriptions,
            @"numerical_result": @{
                @"poisoned_before_each_evaluation": @YES,
                @"comparisons": @(comparisons), @"finite": @(finite),
                @"maximum_absolute_error": @(maximumError), @"tolerance": @0.08,
            },
            @"errors": @{
                @"program": @(createError ?: "none"), @"small_map_or_evaluate": @(smallError ?: "none"),
                @"small_unmap": @(smallUnmapError ?: "none"),
                @"owned_original_evaluate": @(ownedEvaluateError ?: "none"),
                @"owned_competing_catalog": @(competingCatalogError ?: "none"),
                @"scaled_map_or_evaluate_per_bank": largeErrorReceipt,
                @"original_per_bank": originalErrorReceipt,
            },
            @"claim_boundary": @"One loaded authentic 64-procedure layer artifact was tested with compiler-derived procedure IDs. The gate first maps two requests over one shared surface pair for A/B/A, then maps all 64 procedure identities over each of eight existing carrier pairs: 512 request objects, one handle, eight input and eight output IOSurfaces. It submits eight distinct experts before waiting and validates every output against the checkpoint after poisoning. This validates mapped procedure selection and host-side submission overlap, not hardware parallel execution, multi-layer residency, or generation.",
        };
        emit(receipt);

        for (uint32_t bank = 0; bank < 8; ++bank)
            ANEFieldfareANESharedSurfaceRequestSetDestroy(large[bank]);
        ANEFieldfareANESharedSurfaceRequestSetDestroy(competing);
        ANEFieldfareANESharedSurfaceRequestSetDestroy(small);
        ANEFieldfareANEProgramDestroy(program);
        for (uint32_t index = 0; index < 8; ++index) free(expected[index]);
        free((void *)createError); free((void *)smallError);
        free((void *)smallUnmapError);
        free((void *)ownedEvaluateError); free((void *)competingCatalogError);
        for (uint32_t bank = 0; bank < 8; ++bank) {
            free((void *)largeErrors[bank]);
            free((void *)largeEvaluationErrors[bank]);
            free((void *)largeUnmapErrors[bank]);
            free((void *)originalErrors[bank]);
        }
        return valid ? 0 : 1;
    }
}
