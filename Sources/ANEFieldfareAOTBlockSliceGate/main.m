#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach.h>
#import <mach/mach_host.h>
#import <sys/sysctl.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../ANEFieldfareRuntime/ANEProgram.h"
#include "../ANEFieldfareRuntime/TensorIndex.h"

enum { Hidden = 2816, Rows = 704, Positions = 64, Group = 64, Groups = 44 };

static int compareDouble(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return (a > b) - (a < b);
}

typedef struct { uint64_t wired, footprint; } MemorySample;

static MemorySample memorySample(void) {
    vm_statistics64_data_t statistics = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    uint64_t page = 0;
    host_page_size(mach_host_self(), (vm_size_t *)&page);
    uint64_t wired = host_statistics64(mach_host_self(), HOST_VM_INFO64,
        (host_info64_t)&statistics, &count) == KERN_SUCCESS
        ? (uint64_t)statistics.wire_count * page : 0;
    task_vm_info_data_t task = {0}; count = TASK_VM_INFO_COUNT;
    uint64_t footprint = task_info(mach_task_self(), TASK_VM_INFO,
        (task_info_t)&task, &count) == KERN_SUCCESS ? task.phys_footprint : 0;
    return (MemorySample){wired, footprint};
}

static NSString *sysctlString(const char *name) {
    size_t size = 0;
    if (sysctlbyname(name, NULL, &size, NULL, 0) || size < 2) return @"unknown";
    char *value = calloc(size, 1);
    if (!value || sysctlbyname(name, value, &size, NULL, 0)) { free(value); return @"unknown"; }
    NSString *result = [NSString stringWithUTF8String:value] ?: @"unknown";
    free(value);
    return result;
}

static NSDictionary *readJSON(NSString *path) {
    NSData *data = [NSData dataWithContentsOfFile:path];
    return data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
}

static uint64_t directoryBytes(NSString *root) {
    uint64_t total = 0;
    NSDirectoryEnumerator *items = [NSFileManager.defaultManager enumeratorAtPath:root];
    for (NSString *relative in items) {
        (void)relative;
        NSDictionary *attributes = [items fileAttributes];
        if ([attributes[NSFileType] isEqual:NSFileTypeRegular])
            total += [attributes[NSFileSize] unsignedLongLongValue];
    }
    return total;
}

static NSUInteger occurrences(NSString *text, NSString *needle) {
    NSUInteger count = 0, cursor = 0;
    while (cursor < text.length) {
        NSRange range = [text rangeOfString:needle options:0
                                      range:NSMakeRange(cursor, text.length - cursor)];
        if (range.location == NSNotFound) break;
        ++count;
        cursor = NSMaxRange(range);
    }
    return count;
}

static BOOL checkedTensor(NSData *blob, uint64_t base, NSDictionary *record,
                          const uint8_t **pointer, uint64_t *bytes) {
    uint64_t offset = [record[@"offset"] unsignedLongLongValue];
    uint64_t size = [record[@"size"] unsignedLongLongValue];
    if (base > UINT64_MAX - offset || base + offset > blob.length ||
        size > blob.length - (base + offset)) return NO;
    *pointer = (const uint8_t *)blob.bytes + base + offset;
    *bytes = size;
    return YES;
}

static BOOL resolveMatrix(NSData *blob, uint64_t base, NSDictionary *tensors,
                          NSString *name, ANEFieldfareTensor *matrix) {
    NSDictionary *weight = tensors[name];
    NSArray *shape = weight[@"shape"];
    if ([weight[@"bits"] intValue] != 4 || shape.count != 2) return NO;
    const uint8_t *codes = NULL, *scales = NULL, *biases = NULL;
    uint64_t codeBytes = 0, scaleBytes = 0, biasBytes = 0;
    if (!checkedTensor(blob, base, weight, &codes, &codeBytes) ||
        !checkedTensor(blob, base, tensors[[name stringByAppendingString:@"_scales"]],
                       &scales, &scaleBytes) ||
        !checkedTensor(blob, base, tensors[[name stringByAppendingString:@"_biases"]],
                       &biases, &biasBytes)) return NO;
    uint32_t rows = [shape[0] unsignedIntValue], columns = [shape[1] unsignedIntValue];
    if (!rows || !columns || columns % Group || codeBytes != (uint64_t)rows * columns / 2 ||
        scaleBytes != (uint64_t)rows * (columns / Group) * 2 || biasBytes != scaleBytes) return NO;
    *matrix = (ANEFieldfareTensor){.name=name.UTF8String, .nameBytes=(uint32_t)name.length,
        .weights=codes, .weightBytes=codeBytes, .scales=scales, .scaleBytes=scaleBytes,
        .biases=biases, .biasBytes=biasBytes, .rows=rows, .columns=columns};
    return YES;
}

static void emit(NSDictionary *value) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static NSDictionary *reject(NSString *stage, NSString *error, NSDictionary *extra) {
    NSMutableDictionary *result = [@{
        @"schema": @"ane-fieldfare.aot-block-slice.v1", @"status": @"rejected",
        @"stage": stage, @"error": error ?: @"unknown", @"compile_count": @0,
        @"load_count": @0, @"numerical_result": @"not_run",
        @"claim_boundary": @"The named stage failed. It does not validate authentic checkpoint fidelity, private-ANE execution, timing, or compressed residency."
    } mutableCopy];
    [result addEntriesFromDictionary:extra ?: @{}];
    return result;
}

static float inputValue(uint32_t column, uint32_t position) {
    int32_t value = (int32_t)((column * 17u + position * 29u) % 193u) - 96;
    return (float)(__fp16)((float)value / 2048.0f);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 5) {
            emit(reject(@"arguments", @"usage: gate <model.mlmodelc> <gturbo-root> <layer> <expert>", nil));
            return 2;
        }
        NSString *modelRoot = @(argv[1]), *checkpoint = @(argv[2]);
        NSInteger layerIndex = atoi(argv[3]), expertIndex = atoi(argv[4]);
        NSString *milPath = [modelRoot stringByAppendingPathComponent:@"model.mil"];
        NSString *weightPath = [modelRoot stringByAppendingPathComponent:@"weights/weight.bin"];
        NSData *milData = [NSData dataWithContentsOfFile:milPath];
        NSData *weightData = [NSData dataWithContentsOfFile:weightPath];
        NSString *mil = milData ? [[NSString alloc] initWithData:milData encoding:NSUTF8StringEncoding] : nil;
        NSUInteger constexprCount = occurrences(mil, @"constexpr_blockwise_shift_scale");
        NSUInteger convolutionCount = occurrences(mil, @"quantized_group_");
        NSDictionary *artifact = @{
            @"compiled_path": modelRoot, @"compiled_total_bytes": @(directoryBytes(modelRoot)),
            @"mil_bytes": @(milData.length), @"stored_weight_blob_bytes": @(weightData.length),
            @"constexpr_int4_slice_count": @(constexprCount),
            @"quantized_group_name_occurrences": @(convolutionCount),
            @"uint4_present": @([mil containsString:@"tensor<uint4"]),
        };
        BOOL relaxedExpert = constexprCount == 50 || constexprCount == 25 ||
            constexprCount == 14 || constexprCount == 3;
        BOOL fullExpert = constexprCount == 99 || relaxedExpert;
        if (!mil || !weightData || (constexprCount != Groups && !fullExpert) ||
            ![mil containsString:@"tensor<uint4"]) {
            emit(reject(@"artifact_inspection", @"expected exactly 44 projection or 99 full-expert uint4 constexpr slices", artifact));
            return 1;
        }

        NSDictionary *layout = readJSON([checkpoint stringByAppendingPathComponent:@"packed_experts/layout.json"]);
        NSDictionary *layer = nil, *expert = nil;
        for (NSDictionary *candidate in layout[@"layers"])
            if ([candidate[@"layer"] integerValue] == layerIndex) { layer = candidate; break; }
        for (NSDictionary *candidate in layer[@"experts"])
            if ([candidate[@"expert"] integerValue] == expertIndex) { expert = candidate; break; }
        NSString *blobPath = [[checkpoint stringByAppendingPathComponent:@"packed_experts"]
                              stringByAppendingPathComponent:layer[@"file"] ?: @""];
        NSData *blob = [NSData dataWithContentsOfFile:blobPath options:NSDataReadingMappedIfSafe error:nil];
        NSDictionary *tensors = expert[@"tensors"];
        uint64_t base = [expert[@"offset"] unsignedLongLongValue];
        ANEFieldfareTensor gateMatrix = {0}, upMatrix = {0}, downMatrix = {0};
        BOOL checkpointReady = blob && resolveMatrix(blob, base, tensors, @"gate", &gateMatrix) &&
            gateMatrix.rows == Rows && gateMatrix.columns == Hidden;
        if (fullExpert) checkpointReady &= resolveMatrix(blob, base, tensors, @"up", &upMatrix) &&
            resolveMatrix(blob, base, tensors, @"down", &downMatrix) &&
            upMatrix.rows == Rows && upMatrix.columns == Hidden &&
            downMatrix.rows == Hidden && downMatrix.columns == Rows;
        if (!checkpointReady) {
            emit(reject(@"checkpoint", @"authentic gate tensor contract mismatch", artifact));
            return 1;
        }

        float *inputs = calloc((uint64_t)Positions * Hidden, sizeof(float));
        uint32_t outputRows = fullExpert ? Hidden : Rows;
        float *expected = calloc((uint64_t)Positions * outputRows, sizeof(float));
        float *negative = calloc(Rows, sizeof(float));
        uint8_t *zeroBias = calloc(gateMatrix.biasBytes, 1);
        if (!inputs || !expected || !negative || !zeroBias) {
            emit(reject(@"oracle_allocation", @"allocation failed", artifact)); return 1;
        }
        ANEFieldfareTensor negativeMatrix = gateMatrix;
        negativeMatrix.biases = zeroBias;
        BOOL oracleReady = YES;
        float *gateValues = fullExpert ? calloc(Rows, sizeof(float)) : NULL;
        float *upValues = fullExpert ? calloc(Rows, sizeof(float)) : NULL;
        float *activated = fullExpert ? calloc(Rows, sizeof(float)) : NULL;
        for (uint32_t position = 0; position < Positions; ++position) {
            float *input = inputs + (uint64_t)position * Hidden;
            for (uint32_t column = 0; column < Hidden; ++column)
                input[column] = inputValue(column, position);
            if (!fullExpert) {
                oracleReady &= ANEFieldfareAffineMatVec(&gateMatrix, 4, Group, input,
                    expected + (uint64_t)position * Rows);
            } else {
                oracleReady &= ANEFieldfareAffineMatVec(&gateMatrix, 4, Group, input, gateValues) &&
                    ANEFieldfareAffineMatVec(&upMatrix, 4, Group, input, upValues);
                for (uint32_t row = 0; row < Rows; ++row) {
                    float g = (float)(__fp16)gateValues[row], u = (float)(__fp16)upValues[row];
                    float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608028654f *
                        (g + 0.044715f * g * g * g)));
                    activated[row] = (float)(__fp16)(gelu * u);
                }
                oracleReady &= ANEFieldfareAffineMatVec(&downMatrix, 4, Group, activated,
                    expected + (uint64_t)position * Hidden);
            }
        }
        oracleReady &= ANEFieldfareAffineMatVec(&negativeMatrix, 4, Group, inputs, negative);
        if (!oracleReady) { emit(reject(@"oracle", @"independent compact oracle rejected", artifact)); return 1; }

        MemorySample beforeCreate = memorySample();
        const char *createError = NULL;
        ANEFieldfareANEProgram *program = ANEFieldfareANEProgramCreateWithConstantWeights(
            mil.UTF8String, "weight.bin", weightData.bytes, weightData.length,
            (uint64_t)Hidden * Positions * 2, (uint64_t)outputRows * Positions * 4, 1, &createError);
        if (!program) {
            NSMutableDictionary *extra = [artifact mutableCopy];
            extra[@"host_chip"] = sysctlString("machdep.cpu.brand_string");
            extra[@"os_build"] = sysctlString("kern.osversion");
            emit(reject(@"private_ane_compile_or_load", @(createError ?: "unknown"), extra));
            free((void *)createError); return 1;
        }
        free((void *)createError);
        MemorySample afterLoad = memorySample();
        ANEFieldfareANEProgramInfo info = ANEFieldfareANEProgramGetInfo(program);
        IOSurfaceRef inputSurface = ANEFieldfareANEProgramInput(program, 0);
        IOSurfaceLock(inputSurface, 0, NULL);
        __fp16 *inputBase = IOSurfaceGetBaseAddress(inputSurface);
        for (uint32_t column = 0; column < Hidden; ++column)
            for (uint32_t position = 0; position < Positions; ++position)
                inputBase[(uint64_t)column * Positions + position] =
                    (__fp16)inputs[(uint64_t)position * Hidden + column];
        IOSurfaceUnlock(inputSurface, 0, NULL);

        const uint32_t warmups = 2, timed = 9;
        BOOL evaluated = YES;
        double bestMs = 0, meanMs = 0;
        double completionSamples[9] = {0};
        NSString *runErrorText = @"none";
        for (uint32_t iteration = 0; iteration < warmups + timed && evaluated; ++iteration) {
            ANEFieldfareANEEvaluationTiming timing = {0}; const char *runError = NULL;
            evaluated = ANEFieldfareANEProgramEvaluate(program, 0, 5000000000ull, &timing, &runError);
            if (!evaluated) runErrorText = @(runError ?: "unknown");
            free((void *)runError);
            if (iteration >= warmups && evaluated) {
                completionSamples[iteration - warmups] = timing.completionMilliseconds;
                meanMs += timing.completionMilliseconds;
                if (!bestMs || timing.completionMilliseconds < bestMs) bestMs = timing.completionMilliseconds;
            }
        }
        if (evaluated) meanMs /= timed;
        if (evaluated) qsort(completionSamples, timed, sizeof(double), compareDouble);
        double medianMs = evaluated ? completionSamples[timed / 2] : 0;

        IOSurfaceRef outputSurface = ANEFieldfareANEProgramOutput(program, 0);
        uint64_t comparisons = 0, finite = 0; double maxError = 0, squareError = 0, negativeDelta = 0;
        double expectedSquare = 0, actualSquare = 0, dotProduct = 0, expectedMax = 0, actualMax = 0;
        if (evaluated && IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess) {
            const float *actual = IOSurfaceGetBaseAddress(outputSurface);
            for (uint32_t row = 0; row < outputRows; ++row) for (uint32_t position = 0; position < Positions; ++position) {
                float value = actual[(uint64_t)row * Positions + position];
                double delta = fabs((double)value - expected[(uint64_t)position * outputRows + row]);
                double expectedValue = expected[(uint64_t)position * outputRows + row];
                maxError = fmax(maxError, delta); squareError += delta * delta;
                expectedSquare += expectedValue * expectedValue;
                actualSquare += (double)value * value;
                dotProduct += expectedValue * value;
                expectedMax = fmax(expectedMax, fabs(expectedValue));
                actualMax = fmax(actualMax, fabs((double)value));
                if (!fullExpert && position == 0)
                    negativeDelta = fmax(negativeDelta, fabs((double)value - negative[row]));
                if (fullExpert) negativeDelta = fmax(negativeDelta, fabs((double)value));
                finite += isfinite(value); ++comparisons;
            }
            IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        }
        BOOL negativeRejected = fullExpert ? negativeDelta > 1e-5 : negativeDelta > 0.1;
        BOOL numerical = evaluated && comparisons == (uint64_t)outputRows * Positions && finite == comparisons &&
            maxError <= 0.08 && negativeRejected;
        MemorySample afterEvaluate = memorySample();
        ANEFieldfareANEProgramDestroy(program);
        MemorySample afterDestroy = memorySample();

        NSString *command = NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_EXACT_COMMAND"];
        if (!command.length) command = [NSString stringWithFormat:@"%s %s %s %s %s", argv[0], argv[1], argv[2], argv[3], argv[4]];
        emit(@{
            @"schema": @"ane-fieldfare.aot-block-slice.v1", @"status": numerical ? @"validated" : @"rejected",
            @"stage": numerical ? @"private_ane_checkpoint_oracle" : @"evaluate_or_oracle",
            @"host_chip": sysctlString("machdep.cpu.brand_string"), @"os_build": sysctlString("kern.osversion"),
            @"ane_architecture": @"H17G", @"exact_command": command, @"exit_status": @(numerical ? 0 : 1),
            @"compile_count": @1, @"load_count": @1, @"evaluate_count": @(evaluated ? warmups + timed : 0),
            @"source": @{@"checkpoint": checkpoint, @"layer": @(layerIndex), @"expert": @(expertIndex),
                          @"projection": relaxedExpert ? @"requantized_full_expert" :
                              (fullExpert ? @"full_expert" : @"gate")},
            @"shape": @{@"rows": @(outputRows), @"columns": @(Hidden), @"positions": @(Positions),
                         @"group_size": @(Group), @"slice_count": @(constexprCount)},
            @"artifact": artifact,
            @"stored_bytes": @{@"checkpoint_codes": @(gateMatrix.weightBytes + upMatrix.weightBytes + downMatrix.weightBytes),
                                @"checkpoint_scales": @(gateMatrix.scaleBytes + upMatrix.scaleBytes + downMatrix.scaleBytes),
                                @"checkpoint_biases": @(gateMatrix.biasBytes + upMatrix.biasBytes + downMatrix.biasBytes),
                                @"checkpoint_projection_total": @(gateMatrix.weightBytes + gateMatrix.scaleBytes + gateMatrix.biasBytes + upMatrix.weightBytes + upMatrix.scaleBytes + upMatrix.biasBytes + downMatrix.weightBytes + downMatrix.scaleBytes + downMatrix.biasBytes),
                                @"compiled_weight_blob": @(weightData.length),
                                @"fp16_equivalent": @((uint64_t)(fullExpert ? 3 : 1) * Rows * Hidden * 2)},
            @"timing_ms": @{@"compile": @(info.compileMilliseconds), @"load": @(info.loadMilliseconds),
                             @"warm_best_completion": @(bestMs), @"warm_median_completion": @(medianMs),
                             @"warm_mean_completion": @(meanMs),
                             @"warm_completion_samples_sorted": @[@(completionSamples[0]), @(completionSamples[1]),
                                 @(completionSamples[2]), @(completionSamples[3]), @(completionSamples[4]),
                                 @(completionSamples[5]), @(completionSamples[6]), @(completionSamples[7]),
                                 @(completionSamples[8])],
                             @"compile_load_excluded_from_warm": @YES},
            @"memory": @{
                @"wired_before_create": @(beforeCreate.wired), @"wired_after_load": @(afterLoad.wired),
                @"wired_after_evaluate": @(afterEvaluate.wired), @"wired_after_destroy": @(afterDestroy.wired),
                @"wired_load_delta": @((int64_t)afterLoad.wired - (int64_t)beforeCreate.wired),
                @"phys_footprint_before_create": @(beforeCreate.footprint),
                @"phys_footprint_after_load": @(afterLoad.footprint),
                @"phys_footprint_after_evaluate": @(afterEvaluate.footprint),
                @"phys_footprint_after_destroy": @(afterDestroy.footprint),
                @"phys_footprint_load_delta": @((int64_t)afterLoad.footprint - (int64_t)beforeCreate.footprint),
                @"note": @"Wired is one noisy system-wide sample; process phys_footprint is an accounting ledger, not proof of ANE physical residency."
            },
            @"numerical_result": @{@"oracle": @"ANEFieldfareAffineMatVec directly over authentic packed codes and BF16 scale/bias",
                @"comparisons": @(comparisons), @"finite": @(finite), @"max_absolute_error": @(maxError),
                @"rmse": @(comparisons ? sqrt(squareError / comparisons) : 0),
                @"exact_output_rms": @(comparisons ? sqrt(expectedSquare / comparisons) : 0),
                @"normalized_rmse": @(expectedSquare ? sqrt(squareError / expectedSquare) : 0),
                @"cosine_similarity": @(expectedSquare && actualSquare ? dotProduct / sqrt(expectedSquare * actualSquare) : 0),
                @"exact_output_max_absolute": @(expectedMax), @"actual_output_max_absolute": @(actualMax),
                @"tolerance": @0.08},
            @"negative_control": @{@"construction": fullExpert ? @"all-zero output oracle" : @"same codes and scales with all checkpoint biases zeroed",
                @"rejected": @(negativeRejected), @"max_absolute_distance": @(negativeDelta),
                @"required": fullExpert ? @"> 1e-5" : @"> 0.1"},
            @"error": runErrorText,
            @"claim_boundary": numerical
                ? (fullExpert
                    ? (relaxedExpert
                        ? @"One authentic Gemma expert was deterministically requantized without calibration or training to coarser per-output-channel affine uint4 slices, executed gate/up/GELU/down through private ANE, and compared to the exact compact-checkpoint oracle over distinct position inputs. This is a relaxed-fidelity performance point, not exact weight preservation, compressed physical residency, routed execution, or generation quality."
                        : @"One authentic Gemma expert executed gate, up, tanh-approximate GELU product, and down as 99 compiler-authored per-output-channel uint4 constexpr convolutions over exact 64-input slices, checkpoint BF16 affine corrections, and explicit sums. It loaded and evaluated through the private ANE runtime and matched an independent compact-checkpoint oracle. This validates one resident expert, not compressed ANE residency, multi-procedure packaging, routed top-8, or end-to-end speedup.")
                    : @"One authentic 704x2816 Gemma gate projection was represented deterministically as 44 compiler-authored per-output-channel uint4 constexpr convolutions over 64-input slices, with the checkpoint BF16 affine bias applied to each slice input sum and an explicit balanced sum. The compiled artifact loaded and evaluated through the private ANE runtime and matched an independent compact-checkpoint oracle. This validates this one projection and below-FP16 stored constants; it does not prove compressed ANE residency, a complete expert, two procedures, or an end-to-end speedup.")
                : @"The artifact did not clear private ANE evaluation and its independent authentic-checkpoint oracle; no capability or performance promotion is claimed."
        });
        free(inputs); free(expected); free(negative); free(zeroBias);
        free(gateValues); free(upValues); free(activated);
        return numerical ? 0 : 1;
    }
}
