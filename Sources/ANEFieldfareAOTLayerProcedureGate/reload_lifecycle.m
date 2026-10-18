#include "../ANEFieldfareRuntime/ANEProgram.m"

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
        if (!surface || IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return NO;
        const float *actual = IOSurfaceGetBaseAddress(surface);
        for (uint32_t row = 0; row < Hidden; ++row) for (uint32_t position = 0; position < Positions; ++position) {
            float value = actual[(uint64_t)row * Positions + position]; double delta = fabs((double)value - expected[bank][row]);
            *maxError = fmax(*maxError, delta); *squareError += delta * delta; *finite += isfinite(value); ++*comparisons;
        }
        IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    }
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) return 2; NSString *modelRoot = @(argv[1]), *checkpoint = @(argv[2]);
        NSData *milData = [NSData dataWithContentsOfFile:[modelRoot stringByAppendingPathComponent:@"model.mil"]];
        NSData *weights = [NSData dataWithContentsOfFile:[modelRoot stringByAppendingPathComponent:@"weights/weight.bin"]];
        NSString *mil = milData ? [[NSString alloc] initWithData:milData encoding:NSUTF8StringEncoding] : nil;
        const char *names[8] = {"expert_0","expert_1","expert_2","expert_3","expert_4","expert_5","expert_6","expert_7"};
        NSDictionary *layout = readJSON([checkpoint stringByAppendingPathComponent:@"packed_experts/layout.json"]), *layer = layout[@"layers"][0];
        NSData *blob = [NSData dataWithContentsOfFile:[[checkpoint stringByAppendingPathComponent:@"packed_experts"]
            stringByAppendingPathComponent:layer[@"file"]] options:NSDataReadingMappedIfSafe error:nil];
        float input[Hidden]; for (uint32_t column = 0; column < Hidden; ++column) input[column] = inputValue(column, 0);
        float *expected[8] = {0}; BOOL valid = mil && weights && blob;
        for (uint32_t index = 0; index < 8 && valid; ++index) {
            NSDictionary *record = layer[@"experts"][index], *tensors = record[@"tensors"];
            ANEFieldfareTensor gate={0},up={0},down={0}; expected[index]=calloc(Hidden,sizeof(float)); uint64_t base=[record[@"offset"] unsignedLongLongValue];
            valid &= expected[index] && resolveMatrix(blob,base,tensors,@"gate",&gate) && resolveMatrix(blob,base,tensors,@"up",&up) &&
                resolveMatrix(blob,base,tensors,@"down",&down) && reloadOracle(gate,up,down,input,expected[index]);
        }
        const char *createError=NULL; MemorySample before=memorySample();
        ANEFieldfareANEProgram *program=valid?ANEFieldfareANEProgramCreateWithProceduresAndConstantWeights(
            mil.UTF8String,"weight.bin",weights.bytes,weights.length,names,8,&createError):NULL;
        valid &= program != NULL; MemorySample initiallyLoaded=memorySample(); ANEFieldfareANEProgramInfo info=ANEFieldfareANEProgramGetInfo(program);
        Method loadMethod=class_getInstanceMethod([program->model class],@selector(loadWithQoS:options:error:));
        Method unloadMethod=class_getInstanceMethod([program->model class],@selector(unloadWithQoS:error:));
        Method mapMethod=class_getInstanceMethod([program->model class],@selector(mapIOSurfacesWithRequest:cacheInference:error:));
        const char *loadEncoding=loadMethod?method_getTypeEncoding(loadMethod):NULL;
        const char *unloadEncoding=unloadMethod?method_getTypeEncoding(unloadMethod):NULL;
        const char *mapEncoding=mapMethod?method_getTypeEncoding(mapMethod):NULL;
        valid &= loadMethod && unloadMethod && mapMethod;
        uint64_t comparisons=0,finite=0;double maxError=0,squareError=0;const char *executionError=NULL;
        valid &= evaluateReloaded(program,input,expected,&comparisons,&finite,&maxError,&squareError,&executionError);
        const uint32_t cycles=10; double unloadTotal=0,reloadTotal=0,remapTotal=0; NSMutableArray *cycleRecords=[NSMutableArray array];
        NSDictionary *metadataBefore=valid?((id(*)(id,SEL))objc_msgSend)(program->model,@selector(modelAttributes)):nil;
        NSUInteger procedureCountBefore=[metadataBefore[@"ANEFModelDescription"][@"kANEFModelProcedureNameToIDMapKey"] count];
        for(uint32_t cycle=0;cycle<cycles&&valid;++cycle){NSError *error=nil;uint64_t start=mach_absolute_time();
            for(uint32_t bank=0;bank<8;++bank)if(program->mapped[bank]){((void(*)(id,SEL,id))objc_msgSend)(program->model,@selector(unmapIOSurfacesWithRequest:),program->requests[bank]);program->mapped[bank]=NO;}
            double unmap=reloadMs(start,mach_absolute_time());start=mach_absolute_time();
            BOOL unloaded=((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(program->model,@selector(unloadWithQoS:error:),21,&error);
            double unload=reloadMs(start,mach_absolute_time());MemorySample unloadedMemory=memorySample();start=mach_absolute_time();
            BOOL loaded=unloaded&&((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(program->model,@selector(loadWithQoS:options:error:),21,@{},&error);
            double reload=reloadMs(start,mach_absolute_time());start=mach_absolute_time();BOOL mapped=loaded;
            for(uint32_t bank=0;bank<8&&mapped;++bank){mapped=((BOOL(*)(id,SEL,id,BOOL,NSError**))objc_msgSend)(program->model,@selector(mapIOSurfacesWithRequest:cacheInference:error:),program->requests[bank],YES,&error);program->mapped[bank]=mapped;}
            double remap=reloadMs(start,mach_absolute_time());MemorySample reloadedMemory=memorySample();valid&=unloaded&&loaded&&mapped;
            if(!valid&& !executionError)executionError=retainError(error.localizedDescription?:@"reload lifecycle failed");
            if(valid)valid&=evaluateReloaded(program,input,expected,&comparisons,&finite,&maxError,&squareError,&executionError);
            unloadTotal+=unload;reloadTotal+=reload;remapTotal+=remap;[cycleRecords addObject:@{@"cycle":@(cycle+1),@"unmap_ms":@(unmap),@"unload_ms":@(unload),
                @"reload_ms":@(reload),@"remap_ms":@(remap),@"phys_unloaded":@(unloadedMemory.footprint),@"phys_reloaded":@(reloadedMemory.footprint),@"oracle":@(valid)}];}
        NSDictionary *metadataAfter=valid?((id(*)(id,SEL))objc_msgSend)(program->model,@selector(modelAttributes)):nil;
        NSUInteger procedureCountAfter=[metadataAfter[@"ANEFModelDescription"][@"kANEFModelProcedureNameToIDMapKey"] count];
        BOOL numerical=valid&&comparisons==(uint64_t)(cycles+1)*8*Hidden*Positions&&finite==comparisons&&maxError<=0.08&&procedureCountBefore==8&&procedureCountAfter==8;
        if(program)ANEFieldfareANEProgramDestroy(program);MemorySample destroyed=memorySample();for(uint32_t i=0;i<8;++i)free(expected[i]);
        emit(@{@"schema":@"ane-fieldfare.aot-layer-reload-lifecycle.v1",@"status":numerical?@"validated":@"rejected",@"exit_status":@(numerical?0:1),
            @"abi":@{@"load_selector":@"loadWithQoS:options:error:",@"load_encoding":@(loadEncoding?:"missing"),@"unload_selector":@"unloadWithQoS:error:",
                @"unload_encoding":@(unloadEncoding?:"missing"),@"map_selector":@"mapIOSurfacesWithRequest:cacheInference:error:",@"map_encoding":@(mapEncoding?:"missing")},
            @"lifecycle":@{@"compile_count":@1,@"initial_load_count":@1,@"unload_count":@(cycles),@"reload_same_object_count":@(cycles),
                @"recompile_count_during_cycles":@0,@"procedure_count_before":@(procedureCountBefore),@"procedure_count_after":@(procedureCountAfter),@"cycles":cycleRecords},
            @"initial_timing_ms":@{@"compile":@(info.compileMilliseconds),@"load":@(info.loadMilliseconds)},
            @"cycle_timing_ms":@{@"unload_total":@(unloadTotal),@"reload_total":@(reloadTotal),@"remap_total":@(remapTotal),
                @"unload_mean":@(unloadTotal/cycles),@"reload_mean":@(reloadTotal/cycles),@"remap_mean":@(remapTotal/cycles)},
            @"memory":@{@"phys_before":@(before.footprint),@"phys_initially_loaded":@(initiallyLoaded.footprint),@"phys_after_destroy":@(destroyed.footprint)},
            @"numerical_result":@{@"oracle_passes":@(cycles+1),@"comparisons":@(comparisons),@"finite":@(finite),@"max_absolute_error":@(maxError),
                @"rmse":@(comparisons?sqrt(squareError/comparisons):0),@"tolerance":@0.08},@"create_error":@(createError?:"none"),@"execution_error":@(executionError?:"none"),
            @"claim_boundary":@"One already-compiled private in-memory model object is unloaded, reloaded, and remapped ten times without re-offering MIL or weights or invoking compile. Compiler procedure metadata remains present and every cycle passes all eight authentic expert oracles. Immediate footprint samples do not prove reclamation."});
        free((void*)createError);free((void*)executionError);return numerical?0:1;
    }
}
