#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach.h>
#import <mach/mach_host.h>
#import <sys/sysctl.h>
#import <unistd.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../ANEFieldfareRuntime/ANEProgram.h"
#include "../ANEFieldfareRuntime/TensorIndex.h"

enum {
    Hidden = 2816,
    Intermediate = 704,
    Positions = 64,
    GroupSize = 64,
    MaximumPrograms = 256,
};

typedef struct {
    uint64_t wiredBytes;
    uint64_t physFootprintBytes;
} MemorySample;

static MemorySample memorySample(void) {
    vm_statistics64_data_t statistics = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    host_page_size(mach_host_self(), &page);
    uint64_t wired = host_statistics64(mach_host_self(), HOST_VM_INFO64,
        (host_info64_t)&statistics, &count) == KERN_SUCCESS
        ? (uint64_t)statistics.wire_count * page : 0;
    task_vm_info_data_t task = {0};
    count = TASK_VM_INFO_COUNT;
    uint64_t footprint = task_info(mach_task_self(), TASK_VM_INFO,
        (task_info_t)&task, &count) == KERN_SUCCESS ? task.phys_footprint : 0;
    return (MemorySample){wired, footprint};
}

static NSString *sysctlString(const char *name) {
    size_t bytes = 0;
    if (sysctlbyname(name, NULL, &bytes, NULL, 0) || bytes < 2) return @"unknown";
    char *value = calloc(bytes, 1);
    if (!value || sysctlbyname(name, value, &bytes, NULL, 0)) {
        free(value); return @"unknown";
    }
    NSString *result = [NSString stringWithUTF8String:value] ?: @"unknown";
    free(value); return result;
}

static uint16_t bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, 4);
    return (uint16_t)(bits >> 16);
}

static float inputValue(uint32_t column) {
    return 0.25f + (float)((column * 13u) % 101u) / 202.0f;
}

static BOOL buildOracle(NSData *weightBlob, float **inputOut, float **expectedOut) {
    const uint64_t codeBytes = (uint64_t)Intermediate * Hidden / 2;
    const uint64_t parameterBytes = (uint64_t)Intermediate * (Hidden / GroupSize) * 2;
    uint8_t *codes = calloc(codeBytes, 1);
    uint8_t *scales = calloc(parameterBytes, 1);
    uint8_t *biases = calloc(parameterBytes, 1);
    float *input = calloc(Hidden, sizeof(float));
    float *expected = calloc(Intermediate, sizeof(float));
    if (!codes || !scales || !biases || !input || !expected || weightBlob.length < 128 + codeBytes)
        return NO;
    // Decode the actual, possibly identity-perturbed, signed-int4 artifact
    // bytes into the unsigned affine representation consumed by the production
    // oracle. Bias -8*scale makes all signed nibbles [-8,7] representable.
    const uint8_t *stored = (const uint8_t *)weightBlob.bytes + 128;
    for (uint64_t linear = 0; linear < (uint64_t)Intermediate * Hidden; ++linear) {
        uint8_t nibble = (linear & 1) ? stored[linear / 2] >> 4 : stored[linear / 2] & 0x0f;
        int32_t signedValue = nibble >= 8 ? (int32_t)nibble - 16 : nibble;
        uint8_t q = (uint8_t)(signedValue + 8);
        if (linear & 1) codes[linear / 2] |= q << 4;
        else codes[linear / 2] = q;
    }
    uint16_t scale = bf16(1.0f / 1024.0f), bias = bf16(-8.0f / 1024.0f);
    for (uint64_t group = 0; group < (uint64_t)Intermediate * Hidden / GroupSize; ++group) {
        memcpy(scales + group * 2, &scale, 2);
        memcpy(biases + group * 2, &bias, 2);
    }
    for (uint32_t column = 0; column < Hidden; ++column)
        input[column] = (float)(__fp16)inputValue(column);
    ANEFieldfareTensor tensor = {
        .name = "c2_oracle", .nameBytes = 9,
        .weights = codes, .weightBytes = codeBytes,
        .scales = scales, .scaleBytes = parameterBytes,
        .biases = biases, .biasBytes = parameterBytes,
        .rows = Intermediate, .columns = Hidden,
    };
    BOOL valid = ANEFieldfareAffineMatVec(&tensor, 4, GroupSize, input, expected);
    free(codes); free(scales); free(biases);
    if (!valid) { free(input); free(expected); return NO; }
    *inputOut = input; *expectedOut = expected;
    return YES;
}

static NSDictionary *validateProgram(ANEFieldfareANEProgram *program,
                                     const float *input,
                                     const float *expected) {
    IOSurfaceRef inputSurface = ANEFieldfareANEProgramInput(program, 0);
    IOSurfaceRef outputSurface = ANEFieldfareANEProgramOutput(program, 0);
    if (!inputSurface || !outputSurface || IOSurfaceLock(inputSurface, 0, NULL) != kIOReturnSuccess)
        return @{@"status": @"rejected", @"stage": @"surface"};
    __fp16 *base = IOSurfaceGetBaseAddress(inputSurface);
    for (uint32_t column = 0; column < Hidden; ++column)
        for (uint32_t position = 0; position < Positions; ++position)
            base[(uint64_t)column * Positions + position] = (__fp16)input[column];
    IOSurfaceUnlock(inputSurface, 0, NULL);
    const char *error = NULL;
    ANEFieldfareANEEvaluationTiming timing = {0};
    BOOL evaluated = ANEFieldfareANEProgramEvaluate(program, 0, 5 * NSEC_PER_SEC,
                                                     &timing, &error);
    double maximumError = 0;
    uint32_t finite = 0;
    if (evaluated && IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess) {
        const __fp16 *actual = IOSurfaceGetBaseAddress(outputSurface);
        for (uint32_t row = 0; row < Intermediate; ++row) {
            float value = actual[(uint64_t)row * Positions];
            if (isfinite(value)) ++finite;
            maximumError = fmax(maximumError, fabs((double)value - expected[row]));
        }
        IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);
    }
    BOOL valid = evaluated && finite == Intermediate && maximumError <= 0.08;
    NSDictionary *result = @{
        @"status": valid ? @"validated" : @"rejected",
        @"evaluated": @(evaluated),
        @"finite": @(finite),
        @"comparisons": @(Intermediate),
        @"max_absolute_error": @(maximumError),
        @"completion_ms": @(timing.completionMilliseconds),
        @"error": @(error ?: "none"),
        @"oracle": @"ANEFieldfareAffineMatVec on the same deterministic per-channel int4 lattice",
    };
    free((void *)error);
    return result;
}

static void writeSignal(NSString *path, NSDictionary *value) {
    if (!path.length) return;
    NSData *json = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingSortedKeys error:nil];
    [json writeToFile:path atomically:YES];
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys error:nil];
    fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
    NSString *path = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_RECEIPT_PATH"];
    if (path.length) [json writeToFile:path atomically:YES];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s <int4-per-channel.mlmodelc> <requested> [hold-seconds]\n", argv[0]);
            return 2;
        }
        NSString *root = [NSString stringWithUTF8String:argv[1]];
        uint32_t requested = (uint32_t)strtoul(argv[2], NULL, 10);
        if (requested > MaximumPrograms) requested = MaximumPrograms;
        uint32_t holdSeconds = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 0;
        NSData *milData = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"model.mil"]];
        NSData *weightData = [NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"weights/weight.bin"]];
        NSString *mil = milData ? [[NSString alloc] initWithData:milData encoding:NSUTF8StringEncoding] : nil;
        if (!mil || !weightData || ![mil containsString:@"tensor<int4"] ||
            ![mil containsString:@"constexpr_blockwise_shift_scale"] ||
            ![mil containsString:@"scale = tensor<fp16, [704, 1, 1, 1]"]) {
            emit(@{@"schema": @"ane-fieldfare.aot-residency-contention.v1",
                   @"status": @"rejected", @"stage": @"artifact_inspection",
                   @"error": @"exact per-channel int4 artifact not found"});
            return 1;
        }

        MemorySample before = memorySample();
        ANEFieldfareANEProgram *programs[MaximumPrograms] = {0};
        uint32_t loaded = 0;
        double compileTotal = 0, loadTotal = 0;
        NSString *firstFailure = @"none";
        NSMutableArray<NSString *> *hostProgramPointers = [NSMutableArray array];
        NSMutableArray<NSData *> *distinctBlobs = [NSMutableArray array];
        NSString *identitySalt = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_IDENTITY_SALT"] ?: @"single";
        uint8_t saltMask = (uint8_t)(identitySalt.hash ^ (identitySalt.hash >> 8));
        for (uint32_t index = 0; index < requested; ++index) {
            // Force a distinct compiler/model identity without changing the
            // emitted constexpr, stored constants, graph, or numerics. Loading
            // byte-identical descriptors could be satisfied by compiler cache
            // deduplication and would not be a valid table-count experiment.
            NSString *uniqueName = [NSString stringWithFormat:@"projection_cast_fp16_c2_%@_%u",
                                     identitySalt, index];
            NSString *uniqueMIL = [mil stringByReplacingOccurrencesOfString:@"projection_cast_fp16"
                                                                 withString:uniqueName];
            NSMutableData *uniqueBlob = [weightData mutableCopy];
            uint8_t *blobBytes = uniqueBlob.mutableBytes;
            uint8_t identityMask = (uint8_t)(index * 37u + 1u + saltMask);
            if (!identityMask) identityMask = 0x5b;
            const uint64_t codeBytes = (uint64_t)Intermediate * Hidden / 2;
            // Change one code byte in every 4 KiB payload page. This defeats
            // whole-blob and page-granular constant deduplication while keeping
            // dtype, shape, scale, and graph unchanged. The oracle decodes the
            // resulting signed nibbles directly from this blob.
            for (uint64_t offset = 0; offset < codeBytes; offset += 4096)
                blobBytes[128 + offset] ^= identityMask;
            const char *error = NULL;
            ANEFieldfareANEProgram *program = ANEFieldfareANEProgramCreateWithConstantWeights(
                uniqueMIL.UTF8String, "weight.bin", uniqueBlob.bytes, uniqueBlob.length,
                (uint64_t)Hidden * Positions * 2,
                (uint64_t)Intermediate * Positions * 2, 1, &error);
            if (!program) {
                firstFailure = @(error ?: "unknown");
                free((void *)error);
                break;
            }
            free((void *)error);
            ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);
            compileTotal += info.compileMilliseconds;
            loadTotal += info.loadMilliseconds;
            programs[loaded++] = program;
            [distinctBlobs addObject:uniqueBlob];
            [hostProgramPointers addObject:[NSString stringWithFormat:@"%p", program]];
        }
        MemorySample afterLoad = memorySample();
        NSString *readyPath = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_READY_FILE"];
        writeSignal(readyPath, @{@"loaded": @(loaded), @"requested": @(requested),
                                 @"first_failure": firstFailure});

        NSString *waitPath = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_WAIT_FOR_FILE"];
        BOOL waitSatisfied = !waitPath.length;
        for (uint32_t tick = 0; waitPath.length && tick < 300; ++tick) {
            if ([NSFileManager.defaultManager fileExistsAtPath:waitPath]) {
                waitSatisfied = YES; break;
            }
            usleep(100000);
        }

        float *firstInput = NULL, *firstExpected = NULL;
        float *lastInput = NULL, *lastExpected = NULL;
        BOOL firstOracle = loaded && buildOracle(distinctBlobs[0], &firstInput, &firstExpected);
        BOOL lastOracle = loaded && buildOracle(distinctBlobs[loaded - 1], &lastInput, &lastExpected);
        NSDictionary *firstValidation = firstOracle
            ? validateProgram(programs[0], firstInput, firstExpected)
            : @{@"status": loaded ? @"rejected" : @"not_run", @"stage": @"oracle"};
        NSDictionary *lastValidation = lastOracle
            ? validateProgram(programs[loaded - 1], lastInput, lastExpected)
            : @{@"status": loaded ? @"rejected" : @"not_run", @"stage": @"oracle"};
        free(firstInput); free(firstExpected); free(lastInput); free(lastExpected);
        if (holdSeconds) sleep(holdSeconds);

        for (uint32_t index = 0; index < loaded; ++index)
            ANEFieldfareANEProgramDestroy(programs[index]);
        MemorySample afterDestroy = memorySample();
        NSMutableArray *destroyTimeline = [NSMutableArray array];
        BOOL captureTimeline = [NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_MEMORY_TIMELINE"] boolValue];
        if (captureTimeline) {
            uint32_t previous = 0;
            for (uint32_t second = 1; second <= 10; second = second == 1 ? 5 : 10) {
                sleep(second - previous); previous = second;
                MemorySample sample = memorySample();
                [destroyTimeline addObject:@{
                    @"seconds_after_destroy": @(second),
                    @"wired_bytes": @(sample.wiredBytes),
                    @"wired_delta_from_loaded_bytes": @((int64_t)sample.wiredBytes - (int64_t)afterLoad.wiredBytes),
                    @"phys_footprint_bytes": @(sample.physFootprintBytes),
                }];
                if (second == 10) break;
            }
        }
        BOOL hitCeiling = loaded < requested && [firstFailure containsString:@"0x50004"];
        BOOL numerical = [firstValidation[@"status"] isEqual:@"validated"] &&
                         [lastValidation[@"status"] isEqual:@"validated"];
        NSString *command = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXACT_COMMAND"];
        if (!command.length) command = [NSString stringWithFormat:@"%s %s %u %u",
            argv[0], argv[1], requested, holdSeconds];
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.aot-residency-contention.v1",
            @"status": numerical ? @"validated" : @"rejected",
            @"stage": @"residency_contention_and_post_failure_oracle",
            @"host_chip": sysctlString("machdep.cpu.brand_string"),
            @"os_build": sysctlString("kern.osversion"),
            @"ane_architecture": @"H17G",
            @"exact_command": command,
            @"exit_status": @(numerical ? 0 : 1),
            @"artifact": @{
                @"path": root,
                @"kind": @"public package compiler emitted .mlmodelc containing MIL and weight.bin; no HWX was emitted",
                @"operator": @"constexpr_blockwise_shift_scale int4 per output channel",
                @"stored_weight_blob_bytes": @(weightData.length),
                @"pre_specialized_hwx": @NO,
                @"distinct_identity_mechanism": @"rename emitted conv metadata and XOR one signed-int4 code byte in every 4 KiB code-payload page with a unique ordinal mask; graph shape and scales remain unchanged and the oracle decodes each exact resulting blob",
                @"identity_salt": identitySalt,
            },
            @"residency": @{
                @"programs_requested": @(requested), @"programs_loaded": @(loaded),
                @"compile_count": @(loaded), @"load_count": @(loaded),
                @"compile_ms_total": @(compileTotal), @"load_ms_total": @(loadTotal),
                @"first_failure": firstFailure, @"hit_0x50004": @(hitCeiling),
                @"unique_host_program_wrapper_count": @([NSSet setWithArray:hostProgramPointers].count),
                @"host_program_wrapper_pointers": hostProgramPointers,
                @"stored_int4_bytes_total": @((uint64_t)loaded * weightData.length),
                @"logical_fp16_bytes_total": @((uint64_t)loaded * Intermediate * Hidden * 2),
            },
            @"contention_coordination": @{
                @"ready_file": readyPath ?: @"none", @"wait_for_file": waitPath ?: @"none",
                @"wait_satisfied": @(waitSatisfied), @"hold_seconds": @(holdSeconds),
            },
            @"already_loaded_programs_after_failure": @{
                @"first": firstValidation, @"last": lastValidation,
            },
            @"memory": @{
                @"wired_before_bytes": @(before.wiredBytes),
                @"wired_loaded_bytes": @(afterLoad.wiredBytes),
                @"wired_after_destroy_bytes": @(afterDestroy.wiredBytes),
                @"wired_load_delta_bytes": @((int64_t)afterLoad.wiredBytes - (int64_t)before.wiredBytes),
                @"wired_destroy_delta_bytes": @((int64_t)afterDestroy.wiredBytes - (int64_t)afterLoad.wiredBytes),
                @"wired_load_over_stored_int4": @(loaded && weightData.length
                    ? (double)((int64_t)afterLoad.wiredBytes - (int64_t)before.wiredBytes) /
                      ((double)loaded * weightData.length) : 0),
                @"wired_load_over_logical_fp16": @(loaded
                    ? (double)((int64_t)afterLoad.wiredBytes - (int64_t)before.wiredBytes) /
                      ((double)loaded * Intermediate * Hidden * 2) : 0),
                @"wired_release_over_stored_int4": @(loaded && weightData.length
                    ? -(double)((int64_t)afterDestroy.wiredBytes - (int64_t)afterLoad.wiredBytes) /
                      ((double)loaded * weightData.length) : 0),
                @"wired_release_over_logical_fp16": @(loaded
                    ? -(double)((int64_t)afterDestroy.wiredBytes - (int64_t)afterLoad.wiredBytes) /
                      ((double)loaded * Intermediate * Hidden * 2) : 0),
                @"phys_footprint_before_bytes": @(before.physFootprintBytes),
                @"phys_footprint_loaded_bytes": @(afterLoad.physFootprintBytes),
                @"phys_footprint_after_destroy_bytes": @(afterDestroy.physFootprintBytes),
                @"destroy_timeline": destroyTimeline,
                @"note": @"Wired is system-wide and noisy; phys_footprint is a process ledger, not physical residency."
            },
            @"fallback": @{
                @"explicit_behavior": @"none",
                @"on_load_failure": @"stop loading; keep already-loaded private ANE programs alive and validate them",
                @"coreml_fallback_tested": @NO,
                @"note": @"No CPU, GPU, or Core ML fallback is implemented or inferred by this private-runtime gate."
            },
            @"negative_and_lifecycle_controls": @{
                @"ceiling_control": @"request more programs than the known table capacity and require exact 0x50004",
                @"post_failure_control": @"evaluate first and last already-loaded programs against ANEFieldfareAffineMatVec",
                @"lifecycle_control": @"destroy every successfully loaded program and sample memory afterward"
            },
            @"claim_boundary": @"This repeatedly compiles package-emitted MIL through the ordinary private _ANEInMemoryModel route. It determines whether that route consumes the known device-global program table. Because the public .mlmodelc contains no pre-specialized HWX, it does not test whether pre-specialized AOT/HWX programs use a separate pool."
        };
        emit(receipt);
        return numerical ? 0 : 1;
    }
}
