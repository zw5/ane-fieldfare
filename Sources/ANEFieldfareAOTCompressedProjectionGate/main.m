#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach.h>
#import <mach/mach_host.h>
#import <sys/stat.h>
#import <sys/sysctl.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../ANEFieldfareRuntime/ANEProgram.h"
#include "../ANEFieldfareRuntime/TensorIndex.h"

enum { Hidden = 2816, Intermediate = 704, Positions = 64, GroupSize = 64 };

typedef struct {
    uint64_t wiredBytes;
    uint64_t physFootprintBytes;
} MemorySample;

static NSString *sysctlString(const char *name) {
    size_t bytes = 0;
    if (sysctlbyname(name, NULL, &bytes, NULL, 0) != 0 || bytes < 2) return @"unknown";
    char *value = calloc(bytes, 1);
    if (!value || sysctlbyname(name, value, &bytes, NULL, 0) != 0) {
        free(value);
        return @"unknown";
    }
    NSString *result = [NSString stringWithUTF8String:value] ?: @"unknown";
    free(value);
    return result;
}

static MemorySample memorySample(void) {
    vm_statistics64_data_t statistics = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    uint64_t page = 0;
    host_page_size(mach_host_self(), (vm_size_t *)&page);
    uint64_t wired = host_statistics64(mach_host_self(), HOST_VM_INFO64,
        (host_info64_t)&statistics, &count) == KERN_SUCCESS
        ? (uint64_t)statistics.wire_count * page : 0;
    task_vm_info_data_t task = {0};
    count = TASK_VM_INFO_COUNT;
    uint64_t footprint = task_info(mach_task_self(), TASK_VM_INFO,
        (task_info_t)&task, &count) == KERN_SUCCESS ? task.phys_footprint : 0;
    return (MemorySample){wired, footprint};
}

static uint16_t bf16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float inputValue(uint32_t column) {
    // Positive inputs make the missing-bias negative control decisively wrong.
    return 0.25f + (float)((column * 13u) % 101u) / 202.0f;
}

static int32_t signedCode(uint32_t row, uint32_t column, uint32_t bits) {
    if (bits == 4) return (int32_t)((row * 5u + column * 3u) % 15u) - 7;
    if (column % GroupSize == 0) return -127;
    if (column % GroupSize == 1) return 127;
    return (int32_t)((row * 29u + column * 17u) % 255u) - 127;
}

static NSDictionary *reject(NSString *stage, NSString *error, NSDictionary *extra) {
    NSMutableDictionary *receipt = [@{
        @"schema": @"ane-fieldfare.aot-compressed-projection.v1",
        @"status": @"rejected",
        @"stage": stage,
        @"error": error ?: @"unknown",
        @"compile_count": @0,
        @"load_count": @0,
        @"numerical_result": @"not_run",
        @"claim_boundary": @"The named stage failed. This does not establish ANE execution, compressed resident storage, fidelity, fetch bandwidth, or a general compiler limitation."
    } mutableCopy];
    [receipt addEntriesFromDictionary:extra ?: @{}];
    return receipt;
}

static NSString *deviceName(id<MLComputeDeviceProtocol> device) {
    return device ? NSStringFromClass([(id)device class]) : @"none";
}

static NSDictionary *computePlanAtPath(NSString *path) {
    MLModelConfiguration *configuration = [MLModelConfiguration new];
    configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    configuration.functionName = @"main";
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block MLComputePlan *plan = nil;
    __block NSError *planError = nil;
    [MLComputePlan loadContentsOfURL:[NSURL fileURLWithPath:path]
                      configuration:configuration
                  completionHandler:^(MLComputePlan *value, NSError *error) {
        plan = value;
        planError = error;
        dispatch_semaphore_signal(done);
    }];
    long wait = dispatch_semaphore_wait(done,
        dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC));
    NSMutableArray *operations = [NSMutableArray array];
    MLModelStructureProgramFunction *function = plan.modelStructure.program.functions[@"main"];
    for (MLModelStructureProgramOperation *operation in function.block.operations) {
        MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:operation];
        NSMutableArray *supported = [NSMutableArray array];
        for (id<MLComputeDeviceProtocol> device in usage.supportedComputeDevices)
            [supported addObject:deviceName(device)];
        [operations addObject:@{
            @"operator": operation.operatorName ?: @"unknown",
            @"preferred_device": deviceName(usage.preferredComputeDevice),
            @"supported_devices": supported,
        }];
    }
    return @{
        @"status": plan ? @"accepted" : @"rejected",
        @"configuration": @"CPUAndNeuralEngine; function main",
        @"timed_out": @(wait != 0),
        @"error": planError.localizedDescription ?: @"none",
        @"operations": operations,
        @"claim_boundary": @"A compute plan records Core ML's preferred assignment. It is not private-ANE loading or proof that a prediction executed on that device."
    };
}

static void emit(NSDictionary *receipt) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
    const char *path = getenv("ANE_FIELDFARE_RECEIPT_PATH");
    if (path) [json writeToFile:[NSString stringWithUTF8String:path] atomically:YES];
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            emit(reject(@"arguments", @"usage: gate <projection-int4.mlmodelc>", nil));
            return 2;
        }
        NSString *root = [NSString stringWithUTF8String:argv[1]];
        NSString *milPath = [root stringByAppendingPathComponent:@"model.mil"];
        NSString *weightPath = [root stringByAppendingPathComponent:@"weights/weight.bin"];
        NSData *milData = [NSData dataWithContentsOfFile:milPath];
        NSData *weightData = [NSData dataWithContentsOfFile:weightPath];
        NSString *mil = milData ? [[NSString alloc] initWithData:milData
                                                        encoding:NSUTF8StringEncoding] : nil;
        BOOL constexprPresent = [mil containsString:@"constexpr_blockwise_shift_scale"];
        BOOL int4Present = [mil containsString:@"tensor<int4"];
        BOOL int8Present = [mil containsString:@"tensor<int8"];
        uint32_t bits = int4Present ? 4 : (int8Present ? 8 : 0);
        uint32_t storedBlockSize = [mil containsString:@"[704, 1, 1, 1]"] ? Hidden : GroupSize;
        NSDictionary *computePlan = computePlanAtPath(root);
        NSDictionary *artifact = @{
            @"mlmodelc": root,
            @"mil_bytes": @(milData.length),
            @"stored_weight_blob_bytes": @(weightData.length),
            @"source_fp16_weight_bytes": @((uint64_t)Intermediate * Hidden * 2),
            @"constexpr_blockwise_shift_scale_present": @(constexprPresent),
            @"int4_present": @(int4Present),
            @"int8_present": @(int8Present),
            @"compute_plan": computePlan,
        };
        if (!mil || !weightData || !constexprPresent || !bits) {
            emit(reject(@"artifact_inspection", @"missing model.mil, weight.bin, or exact int4/int8 constexpr spelling", artifact));
            return 1;
        }

        const uint64_t weightBytes = bits == 4
            ? (uint64_t)Intermediate * Hidden / 2 : (uint64_t)Intermediate * Hidden;
        const uint64_t parameterBytes = (uint64_t)Intermediate * (Hidden / GroupSize) * 2;
        uint8_t *codes = calloc(weightBytes, 1);
        uint8_t *scales = calloc(parameterBytes, 1);
        uint8_t *biases = calloc(parameterBytes, 1);
        uint8_t *negativeBiases = calloc(parameterBytes, 1);
        float *input = calloc(Hidden, sizeof(float));
        float *expected = calloc(Intermediate, sizeof(float));
        float *negativeExpected = calloc(Intermediate, sizeof(float));
        if (!codes || !scales || !biases || !negativeBiases || !input || !expected || !negativeExpected) {
            emit(reject(@"oracle_allocation", @"allocation failed", artifact));
            return 1;
        }
        for (uint32_t row = 0; row < Intermediate; ++row) {
            for (uint32_t column = 0; column < Hidden; ++column) {
                uint64_t linear = (uint64_t)row * Hidden + column;
                uint8_t q = (uint8_t)(signedCode(row, column, bits) + (bits == 4 ? 7 : 127));
                if (bits == 8) codes[linear] = q;
                else if (linear & 1) codes[linear / 2] |= q << 4;
                else codes[linear / 2] = q;
            }
        }
        float scale = bits == 4 ? 1.0f / 1024.0f : 1.0f / 16384.0f;
        float bias = -(bits == 4 ? 7.0f : 127.0f) * scale;
        const uint16_t scaleBF16 = bf16(scale);
        const uint16_t biasBF16 = bf16(bias);
        for (uint64_t group = 0; group < (uint64_t)Intermediate * Hidden / GroupSize; ++group) {
            memcpy(scales + group * 2, &scaleBF16, 2);
            memcpy(biases + group * 2, &biasBF16, 2);
        }
        for (uint32_t column = 0; column < Hidden; ++column)
            input[column] = (float)(__fp16)inputValue(column);
        ANEFieldfareTensor tensor = {
            .name = "c1_projection", .nameBytes = 13,
            .weights = codes, .weightBytes = weightBytes,
            .scales = scales, .scaleBytes = parameterBytes,
            .biases = biases, .biasBytes = parameterBytes,
            .rows = Intermediate, .columns = Hidden,
        };
        ANEFieldfareTensor negativeTensor = tensor;
        negativeTensor.biases = negativeBiases;
        BOOL oracleReady = ANEFieldfareAffineMatVec(&tensor, bits, GroupSize, input, expected);
        BOOL negativeReady = ANEFieldfareAffineMatVec(&negativeTensor, bits, GroupSize, input, negativeExpected);
        if (!oracleReady || !negativeReady) {
            emit(reject(@"oracle_construction", @"ANEFieldfareAffineMatVec rejected the tensor", artifact));
            return 1;
        }

        MemorySample before = memorySample();
        const char *error = NULL;
        ANEFieldfareANEProgram *program = ANEFieldfareANEProgramCreateWithConstantWeights(
            mil.UTF8String, "weight.bin", weightData.bytes, weightData.length,
            (uint64_t)Hidden * Positions * 2,
            (uint64_t)Intermediate * Positions * 2, 1, &error);
        MemorySample loadedMemory = memorySample();
        if (!program) {
            NSMutableDictionary *extra = [artifact mutableCopy];
            extra[@"host_chip"] = sysctlString("machdep.cpu.brand_string");
            extra[@"os_build"] = sysctlString("kern.osversion");
            emit(reject(@"private_ane_compile_or_load", @(error ?: "unknown"), extra));
            free((void *)error);
            return 1;
        }
        free((void *)error);
        ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);

        IOSurfaceRef inputSurface = ANEFieldfareANEProgramInput(program, 0);
        IOSurfaceRef outputSurface = ANEFieldfareANEProgramOutput(program, 0);
        IOSurfaceLock(inputSurface, 0, NULL);
        __fp16 *inputBase = IOSurfaceGetBaseAddress(inputSurface);
        for (uint32_t column = 0; column < Hidden; ++column)
            for (uint32_t position = 0; position < Positions; ++position)
                inputBase[(uint64_t)column * Positions + position] = (__fp16)input[column];
        IOSurfaceUnlock(inputSurface, 0, NULL);

        const uint32_t warmups = 2, timed = 7;
        BOOL evaluated = YES;
        double bestMs = 0.0, meanMs = 0.0;
        NSString *evaluationError = @"none";
        for (uint32_t iteration = 0; iteration < warmups + timed && evaluated; ++iteration) {
            ANEFieldfareANEEvaluationTiming timing = {0};
            const char *runError = NULL;
            evaluated = ANEFieldfareANEProgramEvaluate(program, 0, 5000000000ull,
                                                        &timing, &runError);
            if (!evaluated) evaluationError = @(runError ?: "unknown");
            free((void *)runError);
            if (iteration >= warmups && evaluated) {
                meanMs += timing.completionMilliseconds;
                if (bestMs == 0.0 || timing.completionMilliseconds < bestMs)
                    bestMs = timing.completionMilliseconds;
            }
        }
        if (evaluated) meanMs /= timed;

        double maximumError = 0.0, squaredError = 0.0;
        double negativeMaximumError = 0.0;
        uint64_t finite = 0, comparisons = 0;
        if (evaluated && IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess) {
            const __fp16 *actual = IOSurfaceGetBaseAddress(outputSurface);
            for (uint32_t row = 0; row < Intermediate; ++row) {
                for (uint32_t position = 0; position < Positions; ++position) {
                    float value = actual[(uint64_t)row * Positions + position];
                    double delta = fabs((double)value - expected[row]);
                    maximumError = fmax(maximumError, delta);
                    squaredError += delta * delta;
                    negativeMaximumError = fmax(negativeMaximumError,
                        fabs((double)value - negativeExpected[row]));
                    if (isfinite(value)) ++finite;
                    ++comparisons;
                }
            }
            IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        }
        // The independent negative oracle omits the signed-int4 bias. It must
        // be decisively incompatible with the hardware output.
        BOOL negativeControlRejected = negativeMaximumError > 1.0;
        BOOL numerical = evaluated && comparisons == (uint64_t)Intermediate * Positions &&
            finite == comparisons && maximumError <= 0.08 && negativeControlRejected;

        ANEFieldfareANEProgramDestroy(program);
        MemorySample afterDestroy = memorySample();
        double storedGBps = bestMs > 0.0
            ? ((double)weightData.length / 1e9) / (bestMs / 1000.0) : 0.0;
        NSString *command = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXACT_COMMAND"];
        if (!command.length) command = [NSString stringWithFormat:@"%s %s", argv[0], argv[1]];
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.aot-compressed-projection.v1",
            @"status": numerical ? @"validated" : @"rejected",
            @"stage": numerical ? @"private_ane_numerical_oracle" : @"evaluate_or_oracle",
            @"host_chip": sysctlString("machdep.cpu.brand_string"),
            @"os_build": sysctlString("kern.osversion"),
            @"ane_architecture": @"H17G",
            @"exact_command": command,
            @"exit_status": @(numerical ? 0 : 1),
            @"compile_count": @1,
            @"load_count": @1,
            @"evaluate_count": @(evaluated ? warmups + timed : 0),
            @"artifact": artifact,
            @"shape": @{@"input_channels": @(Hidden), @"output_channels": @(Intermediate),
                         @"positions": @(Positions), @"block_size": @(storedBlockSize), @"bits": @(bits)},
            @"compile_ms": @(info.compileMilliseconds),
            @"load_ms": @(info.loadMilliseconds),
            @"best_completion_ms": @(bestMs),
            @"mean_completion_ms": @(meanMs),
            @"stored_byte_equivalent_gbps": @(storedGBps),
            @"bandwidth_note": @"Stored weight.bin bytes divided by end-to-end warm completion time. This is an equivalent rate, not a hardware fetch counter and not proof that compiled resident storage remains compressed.",
            @"numerical_result": @{
                @"oracle": [NSString stringWithFormat:@"ANEFieldfareAffineMatVec, unsigned int%u plus BF16 scale and bias equivalent to the signed AOT source", bits],
                @"comparisons": @(comparisons), @"finite": @(finite),
                @"max_absolute_error": @(maximumError),
                @"rmse": @(comparisons ? sqrt(squaredError / comparisons) : 0.0),
                @"tolerance": @0.08,
            },
            @"negative_control": @{
                @"construction": @"same codes and scale but zero bias instead of the signed-quantization bias",
                @"rejected": @(negativeControlRejected),
                @"max_absolute_error": @(negativeMaximumError), @"required": @"> 1.0",
            },
            @"memory": @{
                @"wired_before_bytes": @(before.wiredBytes),
                @"wired_loaded_bytes": @(loadedMemory.wiredBytes),
                @"wired_after_destroy_bytes": @(afterDestroy.wiredBytes),
                @"wired_load_delta_bytes": @((int64_t)loadedMemory.wiredBytes - (int64_t)before.wiredBytes),
                @"wired_destroy_delta_bytes": @((int64_t)afterDestroy.wiredBytes - (int64_t)loadedMemory.wiredBytes),
                @"phys_footprint_before_bytes": @(before.physFootprintBytes),
                @"phys_footprint_loaded_bytes": @(loadedMemory.physFootprintBytes),
                @"phys_footprint_after_destroy_bytes": @(afterDestroy.physFootprintBytes),
                @"note": @"Wired is a noisy system-wide snapshot; phys_footprint is a process ledger, not physical residency."
            },
            @"error": evaluationError,
            @"claim_boundary": numerical
                ? @"Validated that the exact AOT-emitted compressed constexpr MIL can be compiled, loaded, and evaluated through the private ANE runtime with an independent production affine oracle. The .mlmodelc stores compressed constants. This does not prove the ANE-resident representation remains compressed, and the equivalent bandwidth is not a hardware fetch counter."
                : @"The artifact was accepted by earlier stages but failed private-ANE evaluation or its numerical/negative-control oracle; no ANE-compute or compressed-residency claim is made."
        };
        emit(receipt);
        free(codes); free(scales); free(biases); free(negativeBiases);
        free(input); free(expected); free(negativeExpected);
        return numerical ? 0 : 1;
    }
}
