#define main ANEFieldfareAOTBlockSliceSingleMain
#include "../ANEFieldfareAOTBlockSliceGate/main.m"
#undef main

#import <mach/mach_time.h>

static double layerMilliseconds(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom) mach_timebase_info(&timebase);
    return (double)(end - start) * timebase.numer / timebase.denom / 1e6;
}

static BOOL layerExpertOracle(ANEFieldfareTensor gate, ANEFieldfareTensor up,
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

static ANEFieldfareANEProgram *createEight(NSString *root, uint32_t first,
                                           ANEFieldfareANEProgramInfo *info,
                                           const char **error) {
    NSData *milData = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"model.mil"]];
    NSData *weights = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"weights/weight.bin"]];
    NSString *mil = milData ? [[NSString alloc] initWithData:milData encoding:NSUTF8StringEncoding] : nil;
    const char *names[8]; char storage[8][32];
    for (uint32_t i = 0; i < 8; ++i) {
        snprintf(storage[i], sizeof(storage[i]), "expert_%u", first + i);
        names[i] = storage[i];
        if (![mil containsString:[NSString stringWithUTF8String:storage[i]]]) return NULL;
    }
    ANEFieldfareANEProgram *program = ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
        mil.UTF8String, "weight.bin", weights.bytes, weights.length, names, 8, error);
    if (program && info) *info = ANEFieldfareANEProgramGetInfo(program);
    return program;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) return 2;
        NSString *roots[2] = {@(argv[1]), @(argv[2])}, *checkpoint = @(argv[3]);
        NSDictionary *layout = readJSON([checkpoint stringByAppendingPathComponent:@"packed_experts/layout.json"]);
        NSDictionary *layer = layout[@"layers"][0];
        NSData *blob = [NSData dataWithContentsOfFile:[[checkpoint stringByAppendingPathComponent:@"packed_experts"]
            stringByAppendingPathComponent:layer[@"file"]] options:NSDataReadingMappedIfSafe error:nil];
        float input[Hidden]; for (uint32_t column = 0; column < Hidden; ++column) input[column] = inputValue(column, 0);
        float *expected[16] = {0}; BOOL valid = blob != nil;
        for (uint32_t index = 0; index < 16 && valid; ++index) {
            NSDictionary *record = layer[@"experts"][index], *tensors = record[@"tensors"];
            uint64_t base = [record[@"offset"] unsignedLongLongValue];
            ANEFieldfareTensor gate = {0}, up = {0}, down = {0};
            expected[index] = calloc(Hidden, sizeof(float));
            valid &= expected[index] && resolveMatrix(blob, base, tensors, @"gate", &gate) &&
                resolveMatrix(blob, base, tensors, @"up", &up) &&
                resolveMatrix(blob, base, tensors, @"down", &down) &&
                layerExpertOracle(gate, up, down, input, expected[index]);
        }
        MemorySample before = memorySample(); const char *errors[2] = {0};
        ANEFieldfareANEProgramInfo infos[2] = {0};
        ANEFieldfareANEProgram *programs[2] = {
            valid ? createEight(roots[0], 0, &infos[0], &errors[0]) : NULL, NULL};
        MemorySample oneLoaded = memorySample();
        programs[1] = programs[0] ? createEight(roots[1], 8, &infos[1], &errors[1]) : NULL;
        valid &= programs[0] && programs[1]; MemorySample twoLoaded = memorySample();
        for (uint32_t p = 0; p < 2 && valid; ++p) for (uint32_t bank = 0; bank < 8; ++bank) {
            IOSurfaceRef surface = ANEFieldfareANEProgramInput(programs[p], bank);
            valid &= surface && IOSurfaceLock(surface, 0, NULL) == kIOReturnSuccess;
            if (!valid) break;
            __fp16 *base = IOSurfaceGetBaseAddress(surface);
            for (uint32_t column = 0; column < Hidden; ++column)
                for (uint32_t position = 0; position < Positions; ++position)
                    base[(uint64_t)column * Positions + position] = (__fp16)input[column];
            IOSurfaceUnlock(surface, 0, NULL);
        }
        double serial = 0, concurrent = 0;
        if (valid) {
            uint64_t start = mach_absolute_time();
            for (uint32_t p = 0; p < 2 && valid; ++p) for (uint32_t bank = 0; bank < 8; ++bank) {
                ANEFieldfareANEEvaluationTiming timing = {0}; const char *error = NULL;
                valid &= ANEFieldfareANEProgramEvaluate(programs[p], bank, 5000000000ull, &timing, &error); free((void *)error);
            }
            serial = layerMilliseconds(start, mach_absolute_time());
        }
        ANEFieldfareANEEvaluationTiming timings[2][8] = {0};
        if (valid) {
            uint64_t start = mach_absolute_time();
            for (uint32_t p = 0; p < 2 && valid; ++p) for (uint32_t bank = 0; bank < 8; ++bank) {
                const char *error = NULL;
                valid &= ANEFieldfareANEProgramEvaluateBegin(programs[p], bank, &timings[p][bank], &error); free((void *)error);
            }
            for (uint32_t p = 0; p < 2 && valid; ++p) for (uint32_t bank = 0; bank < 8; ++bank) {
                const char *error = NULL;
                valid &= ANEFieldfareANEProgramEvaluateWait(programs[p], bank, 5000000000ull, &timings[p][bank], &error); free((void *)error);
            }
            concurrent = layerMilliseconds(start, mach_absolute_time());
        }
        uint64_t comparisons = 0, finite = 0; double maxError = 0, squareError = 0;
        for (uint32_t p = 0; p < 2 && valid; ++p) for (uint32_t bank = 0; bank < 8; ++bank) {
            IOSurfaceRef surface = ANEFieldfareANEProgramOutput(programs[p], bank);
            if (!surface || IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) { valid = NO; break; }
            const float *actual = IOSurfaceGetBaseAddress(surface); uint32_t expert = p * 8 + bank;
            for (uint32_t row = 0; row < Hidden; ++row) for (uint32_t position = 0; position < Positions; ++position) {
                float value = actual[(uint64_t)row * Positions + position];
                double delta = fabs((double)value - expected[expert][row]); maxError = fmax(maxError, delta);
                squareError += delta * delta; finite += isfinite(value); ++comparisons;
            }
            IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
        }
        BOOL numerical = valid && comparisons == (uint64_t)16 * Hidden * Positions && finite == comparisons && maxError <= 0.08;
        for (uint32_t p = 0; p < 2; ++p) if (programs[p]) ANEFieldfareANEProgramDestroy(programs[p]);
        MemorySample destroyed = memorySample(); for (uint32_t i = 0; i < 16; ++i) free(expected[i]);
        emit(@{@"schema":@"ane-fieldfare.aot-layer-dual-handle.v1", @"status":numerical?@"validated":@"rejected",
            @"exit_status":@(numerical?0:1), @"errors":@[@(errors[0]?:"none"), @(errors[1]?:"none")],
            @"lifecycle":@{@"program_handles":@2,@"compile_count":@2,@"load_count":@2,@"procedures_per_handle":@8,@"total_procedures":@16},
            @"timing_ms":@{@"compile_handle_0":@(infos[0].compileMilliseconds),@"load_handle_0":@(infos[0].loadMilliseconds),
                @"compile_handle_1":@(infos[1].compileMilliseconds),@"load_handle_1":@(infos[1].loadMilliseconds),
                @"serial_sixteen":@(serial),@"concurrent_cross_handle_begin_then_wait":@(concurrent),@"speedup":@(concurrent?serial/concurrent:0)},
            @"numerical_result":@{@"comparisons":@(comparisons),@"finite":@(finite),@"max_absolute_error":@(maxError),
                @"rmse":@(comparisons?sqrt(squareError/comparisons):0),@"tolerance":@0.08},
            @"memory":@{@"phys_before":@(before.footprint),@"phys_one_handle":@(oneLoaded.footprint),@"phys_two_handles":@(twoLoaded.footprint),
                @"phys_first_delta":@((int64_t)oneLoaded.footprint-(int64_t)before.footprint),
                @"phys_second_delta":@((int64_t)twoLoaded.footprint-(int64_t)oneLoaded.footprint),
                @"wired_before":@(before.wired),@"wired_one_handle":@(oneLoaded.wired),@"wired_two_handles":@(twoLoaded.wired),@"phys_after_destroy":@(destroyed.footprint)},
            @"claim_boundary":numerical?@"Two independently compiled and loaded eight-procedure handles remained resident together, selected authentic experts 0 through 15, accepted all sixteen submissions before waits, and passed independent numerical oracles. This does not prove hardware-parallel execution or a firmware program-table entry count.":@"The dual-handle lifecycle, submission, or numerical oracle rejected."});
        for (uint32_t p = 0; p < 2; ++p) free((void *)errors[p]); return numerical ? 0 : 1;
    }
}
