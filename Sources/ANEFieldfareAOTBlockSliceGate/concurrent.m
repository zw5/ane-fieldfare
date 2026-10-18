#define main ANEFieldfareAOTBlockSliceSingleMain
#include "main.m"
#undef main

#import <mach/mach_time.h>

static double wallMilliseconds(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)(end - start) * timebase.numer / timebase.denom / 1e6;
}

static BOOL fullExpertOracle(ANEFieldfareTensor gate, ANEFieldfareTensor up,
                             ANEFieldfareTensor down, const float *input, float *output) {
    float *g = calloc(Rows, sizeof(float)), *u = calloc(Rows, sizeof(float));
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

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            emit(reject(@"arguments", @"usage: concurrent-gate <compiled-artifact-directory> <gturbo-root>", nil));
            return 2;
        }
        NSString *artifactRoot = @(argv[1]), *checkpoint = @(argv[2]);
        NSDictionary *layout = readJSON([checkpoint stringByAppendingPathComponent:@"packed_experts/layout.json"]);
        NSDictionary *layer = layout[@"layers"][0];
        NSString *blobPath = [[checkpoint stringByAppendingPathComponent:@"packed_experts"]
                              stringByAppendingPathComponent:layer[@"file"]];
        NSData *blob = [NSData dataWithContentsOfFile:blobPath options:NSDataReadingMappedIfSafe error:nil];
        ANEFieldfareANEProgram *programs[8] = {0};
        float *expected[8] = {0};
        NSMutableArray *experts = [NSMutableArray array];
        float input[Hidden];
        for (uint32_t column = 0; column < Hidden; ++column) input[column] = inputValue(column, 0);
        MemorySample before = memorySample();
        BOOL valid = blob != nil;
        double compileSum = 0, loadSum = 0;
        for (uint32_t expertIndex = 0; expertIndex < 8 && valid; ++expertIndex) {
            NSDictionary *record = layer[@"experts"][expertIndex], *tensors = record[@"tensors"];
            uint64_t base = [record[@"offset"] unsignedLongLongValue];
            ANEFieldfareTensor gate = {0}, up = {0}, down = {0};
            expected[expertIndex] = calloc(Hidden, sizeof(float));
            valid &= resolveMatrix(blob, base, tensors, @"gate", &gate) &&
                resolveMatrix(blob, base, tensors, @"up", &up) &&
                resolveMatrix(blob, base, tensors, @"down", &down) && expected[expertIndex] &&
                fullExpertOracle(gate, up, down, input, expected[expertIndex]);
            NSString *modelRoot = [artifactRoot stringByAppendingPathComponent:
                [NSString stringWithFormat:@"expert-l0-e%u.mlmodelc", expertIndex]];
            NSData *milData = [NSData dataWithContentsOfFile:[modelRoot stringByAppendingPathComponent:@"model.mil"]];
            NSData *weights = [NSData dataWithContentsOfFile:[modelRoot stringByAppendingPathComponent:@"weights/weight.bin"]];
            NSString *mil = milData ? [[NSString alloc] initWithData:milData encoding:NSUTF8StringEncoding] : nil;
            const char *error = NULL;
            programs[expertIndex] = valid && mil && weights ? ANEFieldfareANEProgramCreateWithConstantWeights(
                mil.UTF8String, "weight.bin", weights.bytes, weights.length,
                (uint64_t)Hidden * Positions * 2, (uint64_t)Hidden * Positions * 4, 1, &error) : NULL;
            valid &= programs[expertIndex] != NULL;
            if (programs[expertIndex]) {
                ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(programs[expertIndex]);
                compileSum += info.compileMilliseconds; loadSum += info.loadMilliseconds;
                IOSurfaceRef surface = ANEFieldfareANEProgramInput(programs[expertIndex], 0);
                IOSurfaceLock(surface, 0, NULL); __fp16 *baseAddress = IOSurfaceGetBaseAddress(surface);
                for (uint32_t column = 0; column < Hidden; ++column)
                    for (uint32_t position = 0; position < Positions; ++position)
                        baseAddress[(uint64_t)column * Positions + position] = (__fp16)input[column];
                IOSurfaceUnlock(surface, 0, NULL);
            }
            [experts addObject:@{@"expert": @(expertIndex), @"create_error": @(error ?: "none")}];
            free((void *)error);
        }
        MemorySample loaded = memorySample();
        double sequentialWall = 0;
        if (valid) {
            uint64_t start = mach_absolute_time();
            for (uint32_t index = 0; index < 8 && valid; ++index) {
                ANEFieldfareANEEvaluationTiming timing = {0}; const char *error = NULL;
                valid &= ANEFieldfareANEProgramEvaluate(programs[index], 0, 5000000000ull, &timing, &error);
                free((void *)error);
            }
            sequentialWall = wallMilliseconds(start, mach_absolute_time());
        }
        ANEFieldfareANEEvaluationTiming timings[8] = {0};
        double concurrentWall = 0;
        if (valid) {
            uint64_t start = mach_absolute_time();
            for (uint32_t index = 0; index < 8 && valid; ++index) {
                const char *error = NULL;
                valid &= ANEFieldfareANEProgramEvaluateBegin(programs[index], 0, &timings[index], &error);
                free((void *)error);
            }
            for (uint32_t index = 0; index < 8 && valid; ++index) {
                const char *error = NULL;
                valid &= ANEFieldfareANEProgramEvaluateWait(programs[index], 0, 5000000000ull,
                                                             &timings[index], &error);
                free((void *)error);
            }
            concurrentWall = wallMilliseconds(start, mach_absolute_time());
        }
        uint64_t comparisons = 0, finite = 0; double maxError = 0, squareError = 0;
        for (uint32_t index = 0; index < 8 && valid; ++index) {
            IOSurfaceRef surface = ANEFieldfareANEProgramOutput(programs[index], 0);
            if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) { valid = NO; break; }
            const float *actual = IOSurfaceGetBaseAddress(surface);
            for (uint32_t row = 0; row < Hidden; ++row) for (uint32_t position = 0; position < Positions; ++position) {
                float value = actual[(uint64_t)row * Positions + position];
                double delta = fabs((double)value - expected[index][row]);
                maxError = fmax(maxError, delta); squareError += delta * delta;
                finite += isfinite(value); ++comparisons;
            }
            IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        }
        BOOL numerical = valid && comparisons == (uint64_t)8 * Hidden * Positions &&
            finite == comparisons && maxError <= 0.08;
        for (uint32_t index = 0; index < 8; ++index) {
            if (programs[index]) ANEFieldfareANEProgramDestroy(programs[index]);
            free(expected[index]);
        }
        MemorySample destroyed = memorySample();
        NSString *command = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXACT_COMMAND"];
        if (!command.length) command = [NSString stringWithFormat:@"%s %s %s", argv[0], argv[1], argv[2]];
        emit(@{
            @"schema": @"ane-fieldfare.aot-block-slice-concurrent.v1",
            @"status": numerical ? @"validated" : @"rejected",
            @"stage": numerical ? @"eight_independent_programs_concurrent_oracle" : @"load_evaluate_or_oracle",
            @"host_chip": sysctlString("machdep.cpu.brand_string"), @"os_build": sysctlString("kern.osversion"),
            @"exact_command": command, @"exit_status": @(numerical ? 0 : 1),
            @"compile_count": @8, @"load_count": @8, @"expert_count": @8, @"experts": experts,
            @"program_structure": @"eight independent single-procedure resident programs; not a multi-procedure artifact",
            @"timing_ms": @{@"compile_sum": @(compileSum), @"load_sum": @(loadSum),
                @"sequential_eight_wall": @(sequentialWall), @"concurrent_begin_then_wait_wall": @(concurrentWall),
                @"concurrent_speedup": @(concurrentWall ? sequentialWall / concurrentWall : 0),
                @"compile_load_excluded": @YES},
            @"numerical_result": @{@"comparisons": @(comparisons), @"finite": @(finite),
                @"max_absolute_error": @(maxError), @"rmse": @(comparisons ? sqrt(squareError/comparisons) : 0),
                @"tolerance": @0.08, @"same_input_replicated_across_64_positions": @YES},
            @"memory": @{@"phys_before": @(before.footprint), @"phys_eight_loaded": @(loaded.footprint),
                @"phys_after_destroy": @(destroyed.footprint),
                @"phys_load_delta": @((int64_t)loaded.footprint - (int64_t)before.footprint),
                @"wired_before": @(before.wired), @"wired_eight_loaded": @(loaded.wired),
                @"wired_after_destroy": @(destroyed.wired),
                @"wired_load_delta": @((int64_t)loaded.wired - (int64_t)before.wired),
                @"note": @"Wired is a noisy system ledger; process phys_footprint is not proof of compressed physical residency."},
            @"claim_boundary": numerical
                ? @"Eight authentic layer-0 experts were compiled and loaded as eight independent private-ANE programs, submitted with all eight begins before waits, and all 1,441,792 outputs matched independent compact-checkpoint oracles. This validates bounded concurrent submission and residency for these eight programs, not parallel hardware execution, an eight-procedure program, routed scheduling, or generation speedup."
                : @"The eight-program construction failed its lifecycle or numerical oracle; no concurrency capability is claimed."
        });
        return numerical ? 0 : 1;
    }
}
