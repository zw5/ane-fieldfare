#include "../ANEFieldfareRuntime/ANEProgram.m"
#include <sys/clonefile.h>
#include <unistd.h>

#define main ANEFieldfareAOTBlockSliceSingleMain
#include "../ANEFieldfareAOTBlockSliceGate/main.m"
#undef main

static double reloadMs(uint64_t start, uint64_t end) { return milliseconds(end - start); }

static BOOL reloadOracle(ANEFieldfareTensor gate, ANEFieldfareTensor up,
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
    free(g); free(u); free(activated); return ok;
}

static BOOL evaluateReloaded(ANEFieldfareANEProgram *program, const float *input,
                             float *expected[8], uint64_t *comparisons, uint64_t *finite,
                             double *maxError, double *squareError, const char **failure) {
    for (uint32_t bank = 0; bank < 8; ++bank) {
        IOSurfaceRef surface = program->inputs[bank];
        if (!surface || IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return NO;
        __fp16 *base = IOSurfaceGetBaseAddress(surface);
        for (uint32_t column = 0; column < Hidden; ++column)
            for (uint32_t position = 0; position < Positions; ++position)
                base[(uint64_t)column * Positions + position] = (__fp16)input[column];
        IOSurfaceUnlock(surface, 0, NULL);
        ANEFieldfareANEEvaluationTiming timing = {0}; const char *error = NULL;
        if (!ANEFieldfareANEProgramEvaluate(program, bank, 5000000000ull, &timing, &error)) {
            if (failure) *failure = error; else free((void *)error); return NO;
        }
    }
    for (uint32_t bank = 0; bank < 8; ++bank) {
        IOSurfaceRef surface = program->outputs[bank];
        if (!surface || IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess)
            return NO;
        const float *actual = IOSurfaceGetBaseAddress(surface);
        for (uint32_t row = 0; row < Hidden; ++row)
            for (uint32_t position = 0; position < Positions; ++position) {
                float value = actual[(uint64_t)row * Positions + position];
                double delta = fabs((double)value - expected[bank][row]);
                *maxError = fmax(*maxError, delta); *squareError += delta * delta;
                *finite += isfinite(value); ++*comparisons;
            }
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    }
    return YES;
}

static BOOL expectedExperts(NSData *blob, NSDictionary *layer, uint32_t first,
                            const float *input, float *expected[8]) {
    for (uint32_t bank = 0; bank < 8; ++bank) {
        NSDictionary *record = layer[@"experts"][first + bank];
        NSDictionary *tensors = record[@"tensors"];
        ANEFieldfareTensor gate = {0}, up = {0}, down = {0};
        uint64_t base = [record[@"offset"] unsignedLongLongValue];
        expected[bank] = calloc(Hidden, sizeof(float));
        if (!expected[bank] ||
            !resolveMatrix(blob, base, tensors, @"gate", &gate) ||
            !resolveMatrix(blob, base, tensors, @"up", &up) ||
            !resolveMatrix(blob, base, tensors, @"down", &down) ||
            !reloadOracle(gate, up, down, input, expected[bank])) return NO;
    }
    return YES;
}

static BOOL installStateFile(NSString *source, NSString *destination, NSData *fallback,
                             NSString **mode, NSError **error) {
    unlink(destination.fileSystemRepresentation);
    if (clonefile(source.fileSystemRepresentation, destination.fileSystemRepresentation, 0) == 0) {
        if (mode) *mode = @"apfs_clonefile";
        return YES;
    }
    BOOL wrote = [fallback writeToURL:[NSURL fileURLWithPath:destination] options:0 error:error];
    if (wrote && mode) *mode = @"buffered_write_fallback";
    return wrote;
}

static BOOL transportWeights(ANEFieldfareANEProgram *program, NSString *sourcePath,
                             NSData *weights,
                             double *unmapMs, double *unloadMs,
                             double *writeMs, double *reloadMsOut,
                             double *remapMs, NSString **installMode,
                             const char **failure) {
    NSError *error = nil;
    uint64_t start = mach_absolute_time();
    for (uint32_t bank = 0; bank < 8; ++bank) {
        if (program->mapped[bank]) {
            ((void (*)(id, SEL, id))objc_msgSend)(
                program->model, @selector(unmapIOSurfacesWithRequest:),
                program->requests[bank]);
            program->mapped[bank] = NO;
        }
    }
    *unmapMs = reloadMs(start, mach_absolute_time());

    // This in-memory construction deletes its staging directory during unload.
    // Patch the idle, unmapped program first; the subsequent unload/reload is
    // the visibility boundary for the baked constants.
    start = mach_absolute_time();
    NSString *weightPath = [program->temporaryPath
        stringByAppendingPathComponent:@"weights/weight.bin"];
    NSString *dataPath = [program->temporaryPath stringByAppendingPathComponent:@"data"];
    NSError *weightError = nil, *dataError = nil;
    [NSFileManager.defaultManager createDirectoryAtPath:
        [program->temporaryPath stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:&weightError];
    BOOL wroteWeight = installStateFile(sourcePath, weightPath, weights,
        installMode, &weightError);
    BOOL dataPresent = [NSFileManager.defaultManager fileExistsAtPath:dataPath];
    NSString *ignoredMode = nil;
    BOOL wroteData = !dataPresent || installStateFile(sourcePath, dataPath, weights,
        &ignoredMode, &dataError);
    *writeMs = reloadMs(start, mach_absolute_time());
    if (!wroteWeight || !wroteData) {
        if (failure) *failure = retainError([NSString stringWithFormat:
            @"compiled weight-state write failed temporary=%@ weight_present=%d "
             "weight_error=%@ data_present=%d data_error=%@",
            program->temporaryPath,
            [NSFileManager.defaultManager fileExistsAtPath:weightPath],
            weightError.localizedDescription ?: @"none", dataPresent,
            dataError.localizedDescription ?: @"none"]);
        return NO;
    }

    start = mach_absolute_time();
    BOOL unloaded = ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        program->model, @selector(unloadWithQoS:error:), 21, &error);
    *unloadMs = reloadMs(start, mach_absolute_time());
    if (!unloaded) {
        if (failure) *failure = retainError(error.localizedDescription ?: @"unload failed");
        return NO;
    }

    start = mach_absolute_time();
    BOOL loaded = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        program->model, @selector(loadWithQoS:options:error:), 21, @{}, &error);
    *reloadMsOut = reloadMs(start, mach_absolute_time());
    if (!loaded) {
        if (failure) *failure = retainError(error.localizedDescription ?: @"reload failed");
        return NO;
    }

    start = mach_absolute_time();
    BOOL mapped = YES;
    for (uint32_t bank = 0; bank < 8 && mapped; ++bank) {
        mapped = ((BOOL (*)(id, SEL, id, BOOL, NSError **))objc_msgSend)(
            program->model, @selector(mapIOSurfacesWithRequest:cacheInference:error:),
            program->requests[bank], YES, &error);
        program->mapped[bank] = mapped;
    }
    *remapMs = reloadMs(start, mach_absolute_time());
    if (!mapped && failure)
        *failure = retainError(error.localizedDescription ?: @"remap failed");
    return mapped;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) return 2;
        NSString *artifactA = @(argv[1]);
        NSString *artifactB = @(argv[2]);
        NSString *checkpoint = @(argv[3]);
        NSString *weightAPath = [artifactA stringByAppendingPathComponent:@"weights/weight.bin"];
        NSString *weightBPath = [artifactB stringByAppendingPathComponent:@"weights/weight.bin"];
        NSData *milData = [NSData dataWithContentsOfFile:
            [artifactA stringByAppendingPathComponent:@"model.mil"]];
        NSData *weightsA = [NSData dataWithContentsOfFile:weightAPath];
        NSData *weightsB = [NSData dataWithContentsOfFile:weightBPath];
        NSString *mil = milData ? [[NSString alloc] initWithData:milData
            encoding:NSUTF8StringEncoding] : nil;
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
        float *expectedA[8] = {0}, *expectedB[8] = {0};
        BOOL valid = mil && weightsA && weightsB &&
            weightsA.length == weightsB.length && blob &&
            expectedExperts(blob, layer, 0, input, expectedA) &&
            expectedExperts(blob, layer, 8, input, expectedB);

        const char *names[8] = {
            "expert_0", "expert_1", "expert_2", "expert_3",
            "expert_4", "expert_5", "expert_6", "expert_7"
        };
        const char *createError = NULL, *executionError = NULL;
        ANEFieldfareANEProgram *program = valid ?
            ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
                mil.UTF8String, "weight.bin", weightsA.bytes, weightsA.length,
                names, 8, &createError) : NULL;
        valid &= program != NULL;
        ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);
        uint64_t comparisons = 0, finite = 0;
        double maxError = 0, squareError = 0;
        NSMutableArray *states = [NSMutableArray array];

        if (valid) {
            valid = evaluateReloaded(program, input, expectedA, &comparisons,
                &finite, &maxError, &squareError, &executionError);
            [states addObject:@{@"state": @"A", @"oracle": @(valid),
                @"expert_identities": @"0...7"}];
        }

        NSArray *transport = @[
            @{@"state": @"B", @"expert_identities": @"8...15",
              @"source_path": weightBPath, @"weights": weightsB,
              @"expected": [NSValue valueWithPointer:expectedB]},
            @{@"state": @"A", @"expert_identities": @"0...7",
              @"source_path": weightAPath, @"weights": weightsA,
              @"expected": [NSValue valueWithPointer:expectedA]},
        ];
        for (NSDictionary *next in transport) {
            if (!valid) break;
            double unmap = 0, unload = 0, write = 0, reload = 0, remap = 0;
            NSString *installMode = nil;
            valid = transportWeights(program, next[@"source_path"], next[@"weights"],
                &unmap, &unload, &write, &reload, &remap, &installMode, &executionError);
            float **expected = [next[@"expected"] pointerValue];
            if (valid) valid = evaluateReloaded(program, input, expected,
                &comparisons, &finite, &maxError, &squareError, &executionError);
            [states addObject:@{@"state": next[@"state"],
                @"expert_identities": next[@"expert_identities"],
                @"oracle": @(valid), @"unmap_ms": @(unmap),
                @"weight_install_mode": installMode ?: @"none",
                @"unload_ms": @(unload), @"weight_write_ms": @(write),
                @"reload_ms": @(reload), @"remap_ms": @(remap),
                @"transport_wall_ms": @(unmap + unload + write + reload + remap)}];
        }

        BOOL numerical = valid && comparisons == 3ull * 8 * Hidden * Positions &&
            finite == comparisons && maxError <= 0.08;
        NSString *command = [NSString stringWithFormat:
            @".build/ane-fieldfare-aot-weight-state-transport-gate %@ %@ %@",
            artifactA, artifactB, checkpoint];
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.aot-weight-state-transport.v1",
            @"status": numerical ? @"validated" : @"rejected",
            @"exit_status": numerical ? @0 : @1,
            @"host_chip": sysctlString("machdep.cpu.brand_string"),
            @"os_build": sysctlString("kern.osversion"),
            @"ane_architecture": @"H17G",
            @"exact_command": command,
            @"source_states": @{ @"A": artifactA, @"B": artifactB },
            @"lifecycle": @{ @"compile_count": @1, @"initial_load_count": @1,
                @"weight_state_transports": @2, @"recompile_count_during_transport": @0,
                @"states": states },
            @"initial_timing_ms": @{ @"compile": @(info.compileMilliseconds),
                @"load": @(info.loadMilliseconds) },
            @"numerical_result": @{ @"oracle_sequence": @"A/B/A",
                @"comparisons": @(comparisons), @"finite": @(finite),
                @"max_absolute_error": @(maxError),
                @"rmse": @(comparisons ? sqrt(squareError / comparisons) : 0),
                @"tolerance": @0.08 },
            @"create_error": @(createError ?: "none"),
            @"execution_error": @(executionError ?: "none"),
            @"claim_boundary": @"One compiler-authored eight-procedure program transports between two authentic compact checkpoint weight states and restores the first state in an A/B/A oracle without recompilation. This validates weight-state transport for one fixed layer-0 graph shape; it does not establish cross-layer graph identity, parallel bank reload, full-layer routing, or end-to-end generation throughput."
        };
        emit(receipt);
        if (program) ANEFieldfareANEProgramDestroy(program);
        for (uint32_t i = 0; i < 8; ++i) {
            free(expectedA[i]); free(expectedB[i]);
        }
        free((void *)createError); free((void *)executionError);
        return numerical ? 0 : 1;
    }
}
