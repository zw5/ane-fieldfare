#define main ANEFieldfareAOTBlockSliceSingleMain
#include "main.m"
#undef main

#import "RebindEvidence.h"

#import <mach/mach_time.h>

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
    for (uint32_t row = 0; row < Hidden; ++row) {
        for (uint32_t position = 0; position < Positions; ++position) {
            float value = actual[(uint64_t)row * Positions + position];
            double delta = fabs((double)value - expected[row]);
            *maxError = fmax(*maxError, delta);
            *squareError += delta * delta;
            *finite += isfinite(value);
            ++*comparisons;
        }
    }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3 && argc != 4) {
            emit(reject(@"arguments", @"usage: multiprocedure-gate <experts.mlmodelc> <gturbo-root> [function-count]", nil));
            return 2;
        }
        NSString *modelRoot = @(argv[1]), *checkpoint = @(argv[2]);
        uint32_t functionCount = argc == 4 ? (uint32_t)strtoul(argv[3], NULL, 10) : 8;
        if (functionCount < 8 || functionCount > 128) {
            emit(reject(@"arguments", @"function-count must be in 8...128", nil));
            return 2;
        }
        NSData *milData = [NSData dataWithContentsOfFile:[modelRoot stringByAppendingPathComponent:@"model.mil"]];
        NSData *weights = [NSData dataWithContentsOfFile:[modelRoot stringByAppendingPathComponent:@"weights/weight.bin"]];
        NSString *mil = milData ? [[NSString alloc] initWithData:milData encoding:NSUTF8StringEncoding] : nil;
        NSMutableArray *functionNames = [NSMutableArray array];
        for (uint32_t index = 0; index < functionCount; ++index) {
            NSString *name = [NSString stringWithFormat:@"expert_%u", index];
            if ([mil containsString:[NSString stringWithFormat:@"func %@<ios18>", name]])
                [functionNames addObject:name];
        }
        if (!mil || !weights || functionNames.count != functionCount ||
            occurrences(mil, @"constexpr_blockwise_shift_scale") != 99 * functionCount) {
            emit(reject(@"artifact_inspection", @"compiled artifact lacks the declared named exact experts", nil));
            return 1;
        }
        NSDictionary *layout = readJSON([checkpoint stringByAppendingPathComponent:@"packed_experts/layout.json"]);
        NSDictionary *layer = layout[@"layers"][0];
        NSData *blob = [NSData dataWithContentsOfFile:[[checkpoint stringByAppendingPathComponent:@"packed_experts"]
            stringByAppendingPathComponent:layer[@"file"]] options:NSDataReadingMappedIfSafe error:nil];
        float input[Hidden]; for (uint32_t column = 0; column < Hidden; ++column) input[column] = inputValue(column, 0);
        uint32_t sampledExperts[8] = {0};
        for (uint32_t bank = 0; bank < 8; ++bank)
            sampledExperts[bank] = bank * (functionCount - 1) / 7;
        float *expected[8] = {0}; BOOL valid = blob != nil;
        for (uint32_t index = 0; index < 8 && valid; ++index) {
            NSDictionary *record = nil;
            for (NSDictionary *candidate in layer[@"experts"])
                if ([candidate[@"expert"] unsignedIntValue] == sampledExperts[index]) {
                    record = candidate; break;
                }
            NSDictionary *tensors = record[@"tensors"];
            uint64_t base = [record[@"offset"] unsignedLongLongValue];
            ANEFieldfareTensor gate = {0}, up = {0}, down = {0};
            expected[index] = calloc(Hidden, sizeof(float));
            valid &= expected[index] && resolveMatrix(blob, base, tensors, @"gate", &gate) &&
                resolveMatrix(blob, base, tensors, @"up", &up) &&
                resolveMatrix(blob, base, tensors, @"down", &down) &&
                expertOracle(gate, up, down, input, expected[index]);
        }
        char nameStorage[8][32] = {{0}};
        const char *names[8] = {0};
        for (uint32_t bank = 0; bank < 8; ++bank) {
            snprintf(nameStorage[bank], sizeof(nameStorage[bank]),
                     "expert_%u", sampledExperts[bank]);
            names[bank] = nameStorage[bank];
        }
        MemorySample before = memorySample(); const char *createError = NULL;
        ANEFieldfareANEProgram *program = valid ?
            ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
                mil.UTF8String, "weight.bin", weights.bytes, weights.length, names, 8, &createError) : NULL;
        BOOL programLoaded = program != NULL;
        valid &= program != NULL;
        MemorySample loaded = memorySample();
        ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);
        for (uint32_t bank = 0; bank < 8 && valid; ++bank) {
            IOSurfaceRef surface = ANEFieldfareANEProgramInput(program, bank);
            valid &= surface && IOSurfaceLock(surface, 0, NULL) == kIOReturnSuccess;
            if (!valid) break;
            __fp16 *base = IOSurfaceGetBaseAddress(surface);
            for (uint32_t column = 0; column < Hidden; ++column)
                for (uint32_t position = 0; position < Positions; ++position)
                    base[(uint64_t)column * Positions + position] = (__fp16)input[column];
            IOSurfaceUnlock(surface, 0, NULL);
        }
        double sequentialWall = 0, concurrentWall = 0;
        if (valid) {
            uint64_t start = mach_absolute_time();
            for (uint32_t bank = 0; bank < 8 && valid; ++bank) {
                ANEFieldfareANEEvaluationTiming timing = {0}; const char *error = NULL;
                valid &= ANEFieldfareANEProgramEvaluate(program, bank, 5000000000ull, &timing, &error);
                free((void *)error);
            }
            sequentialWall = elapsedMilliseconds(start, mach_absolute_time());
        }
        ANEFieldfareANEEvaluationTiming timings[8] = {0};
        if (valid) {
            uint64_t start = mach_absolute_time();
            for (uint32_t bank = 0; bank < 8 && valid; ++bank) {
                const char *error = NULL;
                valid &= ANEFieldfareANEProgramEvaluateBegin(program, bank, &timings[bank], &error);
                free((void *)error);
            }
            for (uint32_t bank = 0; bank < 8 && valid; ++bank) {
                const char *error = NULL;
                valid &= ANEFieldfareANEProgramEvaluateWait(program, bank, 5000000000ull,
                                                             &timings[bank], &error);
                free((void *)error);
            }
            concurrentWall = elapsedMilliseconds(start, mach_absolute_time());
        }
        uint64_t comparisons = 0, finite = 0; double maxError = 0, squareError = 0;
        for (uint32_t bank = 0; bank < 8 && valid; ++bank) {
            valid &= compareExpertOutput(ANEFieldfareANEProgramOutput(program, bank),
                                         expected[bank], &comparisons, &finite,
                                         &maxError, &squareError);
        }
        BOOL initialNumerical = valid && comparisons == (uint64_t)8 * Hidden * Positions &&
            finite == comparisons && maxError <= 0.08;

        // Exercise the production operation the artifact topology needs: one
        // idle mapped bank is rebound by compiler name across distant experts,
        // without reconstructing the loaded model or allocating new carriers.
        IOSurfaceRef originalInput = program ? ANEFieldfareANEProgramInput(program, 0) : NULL;
        IOSurfaceRef originalOutput = program ? ANEFieldfareANEProgramOutput(program, 0) : NULL;
        uint32_t rebindExperts[3] = {0, functionCount - 1, 0};
        uint32_t rebindOracleIndices[3] = {0, 7, 0};
        uint64_t rebindComparisons = 0, rebindFinite = 0;
        double rebindMaxError = 0, rebindSquareError = 0;
        uint32_t successfulRebinds = 0;
        double rebindLatencyMilliseconds[3] = {0};
        double rebindLatencySum = 0, rebindLatencyMaximum = 0;
        BOOL surfacesPreserved = originalInput && originalOutput;
        NSString *rebindFailure = @"none";
        for (uint32_t step = 0; step < 3 && valid; ++step) {
            char procedure[32] = {0};
            snprintf(procedure, sizeof(procedure), "expert_%u", rebindExperts[step]);
            const char *error = NULL;
            uint64_t rebindStarted = mach_absolute_time();
            BOOL rebound = ANEFieldfareANEProgramRebindRequestBankToProcedureEvidence(
                program, 0, procedure, &error);
            rebindLatencyMilliseconds[step] = elapsedMilliseconds(
                rebindStarted, mach_absolute_time());
            rebindLatencySum += rebindLatencyMilliseconds[step];
            rebindLatencyMaximum = fmax(
                rebindLatencyMaximum, rebindLatencyMilliseconds[step]);
            if (!rebound) rebindFailure = error ? @(error) : @"unknown rebind error";
            free((void *)error);
            valid &= rebound;
            if (!valid) break;
            ++successfulRebinds;
            surfacesPreserved &= ANEFieldfareANEProgramInput(program, 0) == originalInput &&
                ANEFieldfareANEProgramOutput(program, 0) == originalOutput;
            ANEFieldfareANEEvaluationTiming timing = {0};
            error = NULL;
            valid &= ANEFieldfareANEProgramEvaluate(
                program, 0, 5000000000ull, &timing, &error);
            if (error) rebindFailure = @(error);
            free((void *)error);
            if (valid) valid &= compareExpertOutput(
                ANEFieldfareANEProgramOutput(program, 0),
                expected[rebindOracleIndices[step]], &rebindComparisons,
                &rebindFinite, &rebindMaxError, &rebindSquareError);
        }

        // Submission owns the bank until its completion is consumed, even if
        // the hardware happens to finish before this host call. Rebind must
        // reject that state without disturbing the outstanding request.
        BOOL inFlightRejected = NO;
        NSString *inFlightError = @"none";
        ANEFieldfareANEEvaluationTiming finalTiming = {0};
        if (valid) {
            const char *error = NULL;
            valid &= ANEFieldfareANEProgramEvaluateBegin(
                program, 0, &finalTiming, &error);
            free((void *)error);
            if (valid) {
                error = NULL;
                inFlightRejected = !ANEFieldfareANEProgramRebindRequestBankToProcedureEvidence(
                    program, 0, "expert_0", &error);
                inFlightError = error ? @(error) : @"none";
                free((void *)error);
                error = NULL;
                valid &= ANEFieldfareANEProgramEvaluateWait(
                    program, 0, 5000000000ull, &finalTiming, &error);
                free((void *)error);
            }
        }
        BOOL rebindNumerical = valid && successfulRebinds == 3 && surfacesPreserved &&
            inFlightRejected &&
            rebindComparisons == (uint64_t)3 * Hidden * Positions &&
            rebindFinite == rebindComparisons && rebindMaxError <= 0.08;
        BOOL numerical = initialNumerical && rebindNumerical;
        if (program) ANEFieldfareANEProgramDestroy(program);
        MemorySample destroyed = memorySample();
        for (uint32_t index = 0; index < 8; ++index) free(expected[index]);
        NSString *command = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXACT_COMMAND"];
        if (!command.length) command = argc == 4
            ? [NSString stringWithFormat:@"%s %s %s %s", argv[0], argv[1], argv[2], argv[3]]
            : [NSString stringWithFormat:@"%s %s %s", argv[0], argv[1], argv[2]];
        emit(@{
            @"schema": @"ane-fieldfare.aot-block-slice-many-procedure.v1",
            @"status": numerical ? @"validated" : @"rejected",
            @"stage": numerical ? @"one_handle_dynamic_named_procedure_rebind_oracle" : @"load_evaluate_rebind_or_oracle",
            @"host_chip": sysctlString("machdep.cpu.brand_string"), @"os_build": sysctlString("kern.osversion"),
            @"exact_command": command, @"exit_status": @(numerical ? 0 : 1), @"error": @(createError ?: "none"),
            @"artifact": @{@"compiled_path": modelRoot, @"function_names": functionNames,
                @"stored_weight_blob_bytes": @(weights.length),
                @"constexpr_slice_count": @(99 * functionCount),
                @"sampled_experts": @[@(sampledExperts[0]), @(sampledExperts[1]),
                    @(sampledExperts[2]), @(sampledExperts[3]), @(sampledExperts[4]),
                    @(sampledExperts[5]), @(sampledExperts[6]), @(sampledExperts[7])],
                @"compiled_total_bytes": @(directoryBytes(modelRoot))},
            @"lifecycle": @{@"program_create_attempt_count": @1,
                @"program_handles": @(programLoaded ? 1 : 0),
                @"compile_count": @(programLoaded ? 1 : 0),
                @"load_count": @(programLoaded ? 1 : 0),
                @"compiler_named_procedure_count": @(functionCount),
                @"mapped_sample_request_count": @(programLoaded ? 8 : 0),
                @"successful_idle_bank_rebind_count": @(successfulRebinds),
                @"rebind_sequence": @[@(rebindExperts[0]), @(rebindExperts[1]), @(rebindExperts[2])],
                @"initial_request_map_count": @8,
                @"rebind_request_unmap_count": @(successfulRebinds),
                @"rebind_request_map_count": @(successfulRebinds),
                @"model_recompile_count_during_rebind": @0,
                @"model_reload_count_during_rebind": @0,
                @"in_flight_rebind_rejected": @(inFlightRejected),
                @"in_flight_rebind_error": inFlightError,
                @"final_completion_ordinal": @(finalTiming.completionOrdinal),
                @"request_bank_count": @(info.requestBankCount),
                @"procedure_ids": @"resolved internally from kANEFModelProcedureNameToIDMapKey; none synthesized"},
            @"timing_ms": @{@"compile": @(info.compileMilliseconds), @"load": @(info.loadMilliseconds),
                @"sequential_eight_wall": @(sequentialWall), @"concurrent_begin_then_wait_wall": @(concurrentWall),
                @"concurrent_speedup": @(concurrentWall ? sequentialWall / concurrentWall : 0),
                @"dynamic_rebind_per_step": @[@(rebindLatencyMilliseconds[0]),
                    @(rebindLatencyMilliseconds[1]), @(rebindLatencyMilliseconds[2])],
                @"dynamic_rebind_mean": @(successfulRebinds
                    ? rebindLatencySum / successfulRebinds : 0),
                @"dynamic_rebind_max": @(rebindLatencyMaximum),
                @"compile_load_excluded": @YES},
            @"numerical_result": @{@"comparisons": @(comparisons), @"finite": @(finite),
                @"max_absolute_error": @(maxError), @"rmse": @(comparisons ? sqrt(squareError/comparisons) : 0),
                @"tolerance": @0.08,
                @"rebind_comparisons": @(rebindComparisons),
                @"rebind_finite": @(rebindFinite),
                @"rebind_max_absolute_error": @(rebindMaxError),
                @"rebind_rmse": @(rebindComparisons ? sqrt(rebindSquareError/rebindComparisons) : 0),
                @"rebind_error": rebindFailure,
                @"carrier_iosurfaces_preserved": @(surfacesPreserved)},
            @"memory": @{@"phys_before": @(before.footprint), @"phys_one_handle_loaded": @(loaded.footprint),
                @"phys_after_destroy": @(destroyed.footprint),
                @"phys_load_delta": @((int64_t)loaded.footprint - (int64_t)before.footprint),
                @"wired_before": @(before.wired), @"wired_one_handle_loaded": @(loaded.wired),
                @"wired_after_destroy": @(destroyed.wired),
                @"wired_load_delta": @((int64_t)loaded.wired - (int64_t)before.wired),
                @"interpretation": programLoaded
                    ? @"One compiled weight blob and one loaded model handle are directly observed. Physical constant sharing and exactly one hardware table entry remain inferences; these noisy ledgers do not prove them."
                    : @"No loaded model handle was observed. Memory deltas around a rejected construction do not establish compilation, loading, or residency."},
            @"claim_boundary": numerical
                ? @"One compiler-authored artifact and one loaded private-ANE handle exposed the declared authentic expert procedure population. Eight procedures spanning that population were mapped from compiler metadata and matched independent compact-checkpoint oracles. One idle request bank was then unmapped and remapped by compiler name across experts 0, the final declared expert, and 0 while retaining its carrier IOSurfaces; all three results matched authentic checkpoint oracles, with no model compile or load. An in-flight rebind was rejected. This validates dynamic named-procedure request remapping within one loaded artifact, not physical residency of every constant, routed generation, or throughput."
                : @"The one-handle many-procedure construction failed its compiler metadata, lifecycle, or numerical oracle; no expanded multi-procedure capability is claimed."
        });
        free((void *)createError);
        return numerical ? 0 : 1;
    }
}
