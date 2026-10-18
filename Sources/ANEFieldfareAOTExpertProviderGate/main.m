#define main ANEFieldfareAOTBlockSliceReferenceMain
#include "../ANEFieldfareAOTBlockSliceGate/main.m"
#undef main

#include "../ANEFieldfareRuntime/AOTExpertRuntime.h"

static BOOL providerExpertOracle(ANEFieldfareTensor gate, ANEFieldfareTensor up,
                                 ANEFieldfareTensor down, const float *input,
                                 float *output) {
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
        if (argc != 3) return 2;
        NSString *root = @(argv[1]), *checkpoint = @(argv[2]);
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
        float *expected[8] = {0}; bool valid = blob != nil;
        for (uint32_t expert = 0; expert < 8 && valid; ++expert) {
            NSDictionary *record = layer[@"experts"][expert];
            NSDictionary *tensors = record[@"tensors"];
            uint64_t base = [record[@"offset"] unsignedLongLongValue];
            ANEFieldfareTensor gate = {0}, up = {0}, down = {0};
            expected[expert] = calloc(Hidden, sizeof(float));
            valid = expected[expert] &&
                resolveMatrix(blob, base, tensors, @"gate", &gate) &&
                resolveMatrix(blob, base, tensors, @"up", &up) &&
                resolveMatrix(blob, base, tensors, @"down", &down) &&
                providerExpertOracle(gate, up, down, input, expected[expert]);
        }
        ANEFieldfareAOTExpertShape shape = {
            Hidden, Rows, ANEFieldfareGemma4ExpertCount, 8, 1, 16,
            ANEFieldfareAOTCapabilityCompilerNamedProcedures |
                ANEFieldfareAOTCapabilityCompletionGenerations,
            16,
        };
        const char *error = NULL;
        ANEFieldfareAOTExpertRuntime *runtime = valid ?
            ANEFieldfareAOTExpertRuntimeCreateFromArtifactRoot(
                root.UTF8String, shape, &error) : NULL;
        uint32_t experts[8] = {0,1,2,3,4,5,6,7};
        float (*outputs)[Hidden] = calloc(8, sizeof(*outputs));
        valid = valid && runtime && outputs;
        for (uint32_t run = 0; run < 2 && valid; ++run)
            valid = ANEFieldfareAOTExpertRuntimeCallTop8(
                runtime, 0, experts, input, outputs, 10000000000ull);
        uint64_t comparisons = 0, finite = 0; double maximumError = 0;
        for (uint32_t expert = 0; expert < 8 && valid; ++expert)
            for (uint32_t column = 0; column < Hidden; ++column) {
                double delta = fabs((double)outputs[expert][column] - expected[expert][column]);
                maximumError = fmax(maximumError, delta);
                finite += isfinite(outputs[expert][column]); ++comparisons;
            }
        ANEFieldfareAOTExpertRuntimeInfo info =
            ANEFieldfareAOTExpertRuntimeGetInfo(runtime);
        valid = valid && comparisons == (uint64_t)8 * Hidden &&
            finite == comparisons && maximumError <= 0.08 &&
            info.procedureLoadCount == 8 && info.procedureCacheHitCount == 8 &&
            info.submissionCount == 16 && info.completionCount == 16 &&
            info.lastBankBase == 8 && info.lastCompletedGeneration == 1 &&
            !info.failedClosed;
        // The retained fixture proves only IDs 0...7. An unavailable identity
        // must reject without submitting any additional work.
        uint32_t unavailable[8] = {0,1,2,3,4,5,6,8};
        bool unavailableRejected = runtime && !ANEFieldfareAOTExpertRuntimeCallTop8(
            runtime, 0, unavailable, input, outputs, 10000000000ull);
        ANEFieldfareAOTExpertRuntimeInfo after =
            ANEFieldfareAOTExpertRuntimeGetInfo(runtime);
        unavailableRejected &= after.submissionCount == info.submissionCount;
        valid &= unavailableRejected;
        printf("{\"schema\":\"ane-fieldfare.aot-expert-provider.v1\"," 
               "\"status\":\"%s\",\"compile_count\":1,\"load_count\":1,"
               "\"loaded_layer_handles\":1,\"compiler_named_procedures\":8,"
               "\"logical_request_banks\":16,\"physical_ane_requests\":8,"
               "\"a_b_bank_bases\":[0,8],\"production_double_buffering\":false,"
               "\"submissions\":%llu,"
               "\"completions\":%llu,\"comparisons\":%llu,\"finite\":%llu,"
               "\"maximum_absolute_error\":%.12g,\"tolerance\":0.08,"
               "\"unavailable_expert_rejected_before_submit\":%s,"
               "\"claim_boundary\":\"Actual provider validated on one retained layer-0 handle and eight physical ANE requests for compiler-authored expert_0 through expert_7. Logical bases 0 and 8 execute synchronously over the same completed physical requests and do not prove production double buffering or overlap. Unavailable identities fail closed; no 128-expert or 30-layer artifact is claimed\"}\n",
               valid ? "validated" : "rejected",
               (unsigned long long)info.submissionCount,
               (unsigned long long)info.completionCount,
               (unsigned long long)comparisons, (unsigned long long)finite,
               maximumError, unavailableRejected ? "true" : "false");
        ANEFieldfareAOTExpertRuntimeDestroy(runtime);
        for (uint32_t expert = 0; expert < 8; ++expert) free(expected[expert]);
        free(outputs); free((void *)error);
        return valid ? 0 : 1;
    }
}
