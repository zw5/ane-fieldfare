#include "../ANEFieldfareRuntime/ANEProgram.m"

#define main ANEFieldfareAOTBlockSliceSingleMain
#include "../ANEFieldfareAOTBlockSliceGate/main.m"
#undef main

static BOOL capacityExpertOracle(ANEFieldfareTensor gate, ANEFieldfareTensor up,
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

static BOOL executeHandle(ANEFieldfareANEProgram *program, const float *input,
                          float *expected[8], uint64_t *comparisons, uint64_t *finite,
                          double *maxError, double *squareError, const char **failure) {
    for (uint32_t bank = 0; bank < 8; ++bank) {
        IOSurfaceRef surface = ANEFieldfareANEProgramInput(program, bank);
        if (!surface || IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return NO;
        __fp16 *base = IOSurfaceGetBaseAddress(surface);
        for (uint32_t column = 0; column < Hidden; ++column)
            for (uint32_t position = 0; position < Positions; ++position)
                base[(uint64_t)column * Positions + position] = (__fp16)input[column];
        IOSurfaceUnlock(surface, 0, NULL);
        ANEFieldfareANEEvaluationTiming timing = {0}; const char *error = NULL;
        if (!ANEFieldfareANEProgramEvaluate(program, bank, 5000000000ull, &timing, &error)) {
            if (failure) *failure = error; else free((void *)error);
            return NO;
        }
    }
    for (uint32_t bank = 0; bank < 8; ++bank) {
        IOSurfaceRef surface = ANEFieldfareANEProgramOutput(program, bank);
        if (!surface || IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return NO;
        const float *actual = IOSurfaceGetBaseAddress(surface);
        for (uint32_t row = 0; row < Hidden; ++row) for (uint32_t position = 0; position < Positions; ++position) {
            float value = actual[(uint64_t)row * Positions + position];
            double delta = fabs((double)value - expected[bank][row]);
            *maxError = fmax(*maxError, delta); *squareError += delta * delta;
            *finite += isfinite(value); ++*comparisons;
        }
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    }
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) return 2;
        NSString *modelRoot = @(argv[1]), *checkpoint = @(argv[2]);
        NSData *milData = [NSData dataWithContentsOfFile:[modelRoot stringByAppendingPathComponent:@"model.mil"]];
        NSData *weights = [NSData dataWithContentsOfFile:[modelRoot stringByAppendingPathComponent:@"weights/weight.bin"]];
        NSString *mil = milData ? [[NSString alloc] initWithData:milData encoding:NSUTF8StringEncoding] : nil;
        const char *names[8] = {"expert_0","expert_1","expert_2","expert_3","expert_4","expert_5","expert_6","expert_7"};
        NSDictionary *layout = readJSON([checkpoint stringByAppendingPathComponent:@"packed_experts/layout.json"]);
        NSDictionary *layer = layout[@"layers"][0];
        NSData *blob = [NSData dataWithContentsOfFile:[[checkpoint stringByAppendingPathComponent:@"packed_experts"]
            stringByAppendingPathComponent:layer[@"file"]] options:NSDataReadingMappedIfSafe error:nil];
        float input[Hidden]; for (uint32_t column = 0; column < Hidden; ++column) input[column] = inputValue(column, 0);
        float *expected[8] = {0}; BOOL valid = mil && weights && blob;
        for (uint32_t index = 0; index < 8 && valid; ++index) {
            NSDictionary *record = layer[@"experts"][index], *tensors = record[@"tensors"];
            uint64_t base = [record[@"offset"] unsignedLongLongValue];
            ANEFieldfareTensor gate = {0}, up = {0}, down = {0}; expected[index] = calloc(Hidden, sizeof(float));
            valid &= expected[index] && resolveMatrix(blob, base, tensors, @"gate", &gate) &&
                resolveMatrix(blob, base, tensors, @"up", &up) && resolveMatrix(blob, base, tensors, @"down", &down) &&
                capacityExpertOracle(gate, up, down, input, expected[index]);
        }
        enum { MaximumHandles = 128 }; const uint64_t footprintLimit = 2ull * 1024 * 1024 * 1024;
        ANEFieldfareANEProgram *programs[MaximumHandles] = {0}; uint32_t accepted = 0, attempts = 0;
        double totalCompileMs = 0, totalLoadMs = 0; const char *rejection = NULL;
        MemorySample before = memorySample(), peak = before; NSString *stopReason = @"maximum_handle_guard";
        while (valid && accepted < MaximumHandles) {
            if (peak.footprint >= footprintLimit) { stopReason = @"process_phys_footprint_limit"; break; }
            ++attempts; const char *error = NULL;
            ANEFieldfareANEProgram *program = ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
                mil.UTF8String, "weight.bin", weights.bytes, weights.length, names, 8, &error);
            if (!program) { rejection = error; stopReason = @"private_runtime_rejection"; break; }
            programs[accepted++] = program; ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);
            totalCompileMs += info.compileMilliseconds; totalLoadMs += info.loadMilliseconds; peak = memorySample();
            fprintf(stderr, "accepted_handles=%u phys_footprint=%llu\n", accepted, peak.footprint);
        }
        uint32_t peakAccepted = accepted, releaseCount = 0; BOOL recoveryAttempted = rejection != NULL, recoveryAccepted = NO;
        double releaseUnloadMs = 0; MemorySample afterRelease = peak, afterRecovery = peak; const char *recoveryError = NULL;
        uint64_t comparisons = 0, finite = 0; double maxError = 0, squareError = 0; const char *executionError = NULL;
        if (valid && accepted) valid &= executeHandle(programs[0], input, expected, &comparisons, &finite, &maxError, &squareError, &executionError);
        else valid = NO;
        if (valid && recoveryAttempted && accepted) {
            uint32_t releaseStart = accepted > 1 ? 1 : 0;
            releaseCount = MIN(8u, accepted - releaseStart); uint64_t releaseStarted = mach_absolute_time();
            for (uint32_t index = releaseStart; index < releaseStart + releaseCount; ++index) {
                for (uint32_t bank = 0; bank < 8; ++bank) if (programs[index]->mapped[bank]) {
                    ((void (*)(id, SEL, id))objc_msgSend)(programs[index]->model,
                        @selector(unmapIOSurfacesWithRequest:), programs[index]->requests[bank]);
                    programs[index]->mapped[bank] = NO;
                }
                NSError *error = nil;
                ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(programs[index]->model,
                    @selector(unloadWithQoS:error:), 21, &error);
            }
            releaseUnloadMs = milliseconds(mach_absolute_time() - releaseStarted); afterRelease = memorySample(); ++attempts;
            ANEFieldfareANEProgram *recovered = ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
                mil.UTF8String, "weight.bin", weights.bytes, weights.length, names, 8, &recoveryError);
            if (recovered) {
                programs[accepted++] = recovered; recoveryAccepted = YES; ANEFieldfareANEProgramInfo recoveredInfo = ANEFieldfareANEProgramGetInfo(recovered);
                totalCompileMs += recoveredInfo.compileMilliseconds; totalLoadMs += recoveredInfo.loadMilliseconds;
            }
            afterRecovery = memorySample();
        }
        if (valid && recoveryAccepted) valid &= executeHandle(programs[accepted - 1], input, expected, &comparisons, &finite, &maxError, &squareError, &executionError);
        BOOL numerical = valid && recoveryAccepted && comparisons == (uint64_t)2 * 8 * Hidden * Positions &&
            finite == comparisons && maxError <= 0.08;
        for (uint32_t index = accepted; index > 0; --index) ANEFieldfareANEProgramDestroy(programs[index - 1]);
        MemorySample destroyed = memorySample(); for (uint32_t i = 0; i < 8; ++i) free(expected[i]);
        emit(@{@"schema":@"ane-fieldfare.aot-layer-handle-capacity.v1",@"status":numerical?@"validated":@"rejected",
            @"exit_status":@(numerical?0:1),@"stop_reason":stopReason,@"rejection":@(rejection?:"none"),@"execution_error":@(executionError?:"none"),
            @"capacity":@{@"peak_simultaneously_loaded_handles":@(peakAccepted),@"retained_handle_objects_after_recovery":@(accepted),@"creation_attempts":@(attempts),@"procedures_per_handle":@8,
                @"compiler_procedures_at_peak":@(peakAccepted*8),@"successful_compile_count":@(peakAccepted+(recoveryAccepted?1:0)),@"successful_load_count":@(peakAccepted+(recoveryAccepted?1:0)),
                @"footprint_limit_bytes":@(footprintLimit),@"maximum_handle_guard":@(MaximumHandles)},
            @"release_recovery":@{@"attempted":@(recoveryAttempted),@"unloaded_retained_handle_count":@(releaseCount),
                @"destroyed_handle_count_before_retry":@0,@"release_unload_ms":@(releaseUnloadMs),@"fresh_handle_after_release_accepted":@(recoveryAccepted),
                @"recovery_error":@(recoveryError?:"none"),@"phys_after_release":@(afterRelease.footprint),@"phys_after_recovery":@(afterRecovery.footprint)},
            @"timing_ms":@{@"total_compile":@(totalCompileMs),@"total_load":@(totalLoadMs),
                @"mean_compile":@(accepted?totalCompileMs/accepted:0),@"mean_load":@(accepted?totalLoadMs/accepted:0)},
            @"memory":@{@"phys_before":@(before.footprint),@"phys_at_stop":@(peak.footprint),
                @"phys_delta":@((int64_t)peak.footprint-(int64_t)before.footprint),@"phys_after_teardown":@(destroyed.footprint),
                @"wired_before":@(before.wired),@"wired_at_stop":@(peak.wired),@"wired_after_teardown":@(destroyed.wired)},
            @"numerical_result":@{@"pre_release_oracle_handle_index":@0,@"post_release_oracle_handle_index":@(accepted-1),@"post_release_oracle_is_fresh_recovered_handle":@(recoveryAccepted),@"comparisons":@(comparisons),
                @"finite":@(finite),@"max_absolute_error":@(maxError),@"rmse":@(comparisons?sqrt(squareError/comparisons):0),@"tolerance":@0.08},
            @"claim_boundary":@"Counts successful private compile/load handles and their eight compiler procedures separately until the first rejection or 2 GiB process-footprint stop. After rejection, retained handles are unloaded without destruction and a fresh compile/load is retried. Handle zero executes before release and the fresh recovered handle executes after release. The observed bound does not identify the underlying weighted resource."});
        free((void *)rejection); free((void *)recoveryError); free((void *)executionError); return numerical ? 0 : 1;
    }
}
