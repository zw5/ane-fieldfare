#import <Foundation/Foundation.h>

#include <fcntl.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include "../ANEFieldfareRuntime/ANEExpertDispatch.h"

enum { ExpertStride = 3358720, ExpertCount = 128, RotationCount = 8 };
static const uint64_t LayerBytes = (uint64_t)ExpertStride * ExpertCount;

typedef struct {
    bool completed;
    double wallMs;
    double preparationMs;
    double evaluationMs;
    uint64_t outputHash;
} DispatchMeasurement;

static uint64_t ticks(void) { return mach_absolute_time(); }

static double milliseconds(uint64_t begin, uint64_t end) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)(end - begin) * tb.numer / tb.denom / 1e6;
}

static NSString *sysctlString(const char *name) {
    size_t size = 0;
    if (sysctlbyname(name, NULL, &size, NULL, 0) || !size) return @"unknown";
    char *value = calloc(size + 1, 1);
    if (!value || sysctlbyname(name, value, &size, NULL, 0)) {
        free(value);
        return @"unknown";
    }
    NSString *result = [NSString stringWithUTF8String:value] ?: @"unknown";
    free(value);
    return result;
}

static uint64_t hashFloats(const float *values, size_t count) {
    const uint8_t *bytes = (const uint8_t *)values;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count * sizeof(float); ++index)
        hash = (hash ^ bytes[index]) * UINT64_C(1099511628211);
    return hash;
}

static uint64_t residentPagesForExpert(NSString *path, uint32_t expert) {
    int fd = open(path.fileSystemRepresentation, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return UINT64_MAX;
    off_t offset = (off_t)((uint64_t)expert * ExpertStride);
    void *mapping = mmap(NULL, ExpertStride, PROT_READ, MAP_PRIVATE, fd, offset);
    close(fd);
    if (mapping == MAP_FAILED) return UINT64_MAX;
    size_t pages = (ExpertStride + (size_t)getpagesize() - 1) / (size_t)getpagesize();
    unsigned char *state = calloc(pages, 1);
    uint64_t resident = UINT64_MAX;
    if (state && mincore(mapping, ExpertStride, (char *)state) == 0) {
        resident = 0;
        for (size_t page = 0; page < pages; ++page)
            resident += (state[page] & MINCORE_INCORE) != 0;
    }
    free(state);
    munmap(mapping, ExpertStride);
    return resident;
}

static bool copyPhysicalNoCache(NSString *source, NSString *destination,
                                uint64_t *allocatedBytes) {
    int input = open(source.fileSystemRepresentation, O_RDONLY | O_CLOEXEC);
    int output = open(destination.fileSystemRepresentation,
                      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    void *buffer = NULL;
    const size_t chunkBytes = 4 * 1024 * 1024;
    bool valid = input >= 0 && output >= 0 &&
        posix_memalign(&buffer, 16384, chunkBytes) == 0 &&
        fcntl(input, F_NOCACHE, 1) == 0 && fcntl(output, F_NOCACHE, 1) == 0;
    for (uint64_t offset = 0; valid && offset < LayerBytes;) {
        size_t wanted = (size_t)MIN((uint64_t)chunkBytes, LayerBytes - offset);
        ssize_t got = pread(input, buffer, wanted, (off_t)offset);
        size_t written = 0;
        while (got == (ssize_t)wanted && written < wanted) {
            ssize_t put = pwrite(output, (uint8_t *)buffer + written,
                                 wanted - written, (off_t)(offset + written));
            if (put <= 0) { valid = false; break; }
            written += (size_t)put;
        }
        valid = valid && got == (ssize_t)wanted && written == wanted;
        if (valid) offset += wanted;
    }
    valid = valid && fsync(output) == 0;
    struct stat status = {0};
    valid = valid && fstat(output, &status) == 0 && status.st_size == (off_t)LayerBytes;
    if (allocatedBytes) *allocatedBytes = valid ? (uint64_t)status.st_blocks * 512 : 0;
    // A clone or sparse fixture is not an acceptable cold-storage control.
    valid = valid && (uint64_t)status.st_blocks * 512 >= LayerBytes * 9 / 10;
    free(buffer);
    if (input >= 0) close(input);
    if (output >= 0) close(output);
    return valid;
}

static DispatchMeasurement measureCall(ANEFieldfareANEExpertDispatcher *dispatcher,
                                       uint32_t expert, const float *input, float *output) {
    ANEFieldfareANEExpertDispatchInfo before =
        ANEFieldfareANEExpertDispatcherGetInfo(dispatcher);
    uint64_t begin = ticks();
    bool completed = ANEFieldfareANEExpertDispatcherCall(dispatcher, expert, input, output);
    uint64_t end = ticks();
    ANEFieldfareANEExpertDispatchInfo after =
        ANEFieldfareANEExpertDispatcherGetInfo(dispatcher);
    return (DispatchMeasurement){
        .completed = completed,
        .wallMs = milliseconds(begin, end),
        .preparationMs = after.totalPreparationMilliseconds - before.totalPreparationMilliseconds,
        .evaluationMs = after.totalEvaluationMilliseconds - before.totalEvaluationMilliseconds,
        .outputHash = completed ? hashFloats(output, ANEFieldfareGemma4HiddenSize) : 0,
    };
}

static NSDictionary *measurementJSON(DispatchMeasurement value) {
    return @{
        @"completed": @(value.completed),
        @"wall_ms": @(value.wallMs),
        @"preparation_ms": @(value.preparationMs),
        @"evaluation_ms": @(value.evaluationMs),
        @"output_fnv1a64": [NSString stringWithFormat:@"0x%016llx", value.outputHash],
    };
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2 || argc > 3) {
            fprintf(stderr, "usage: %s <model-directory> [scratch-directory]\n", argv[0]);
            return 2;
        }
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *scratch = argc == 3
            ? [[NSString stringWithUTF8String:argv[2]] stringByStandardizingPath]
            : [[[NSFileManager defaultManager] currentDirectoryPath]
                stringByAppendingPathComponent:@".build"];
        NSString *source = [root stringByAppendingPathComponent:@"packed_experts/layer_00.bin"];
        char directoryTemplate[PATH_MAX];
        snprintf(directoryTemplate, sizeof(directoryTemplate), "%s/b2-expert-residency.XXXXXX",
                 scratch.fileSystemRepresentation);
        char *createdDirectory = mkdtemp(directoryTemplate);
        NSString *temporaryDirectory = createdDirectory
            ? [NSString stringWithUTF8String:createdDirectory] : nil;
        NSString *copy = [temporaryDirectory stringByAppendingPathComponent:@"layer_00.bin"];

        uint64_t allocatedBytes = 0;
        bool copied = temporaryDirectory && copyPhysicalNoCache(source, copy, &allocatedBytes);
        uint64_t initialTargetPages = copied ? residentPagesForExpert(copy, 0) : UINT64_MAX;
        bool targetInitiallyCold = initialTargetPages == 0;

        setenv("ANE_FIELDFARE_MMAP_EXPERT_READS", "1", 1);
        unsetenv("ANE_FIELDFARE_PREFETCH_EXPERTS");
        unsetenv("ANE_FIELDFARE_LAYER_PREFETCH");
        const char *createError = NULL;
        ANEFieldfareANEExpertDispatcher *dispatcher = copied
            ? ANEFieldfareANEExpertDispatcherCreate(copy.fileSystemRepresentation, &createError)
            : NULL;
        unsetenv("ANE_FIELDFARE_MMAP_EXPERT_READS");

        float input[ANEFieldfareGemma4HiddenSize];
        float warmupOutput[ANEFieldfareGemma4HiddenSize] = {0};
        float coldOutput[ANEFieldfareGemma4HiddenSize] = {0};
        float warmOutputs[RotationCount][ANEFieldfareGemma4HiddenSize] = {{0}};
        float rotatingOutputs[RotationCount][ANEFieldfareGemma4HiddenSize] = {{0}};
        float oracle[ANEFieldfareGemma4HiddenSize] = {0};
        for (uint32_t index = 0; index < ANEFieldfareGemma4HiddenSize; ++index)
            input[index] = (float)((int32_t)(index % 97) - 48) / 64.0f;

        bool warmupCompleted = dispatcher &&
            ANEFieldfareANEExpertDispatcherCall(dispatcher, 126, input, warmupOutput) &&
            ANEFieldfareANEExpertDispatcherCall(dispatcher, 126, input, warmupOutput);
        ANEFieldfareANEExpertDispatchInfo transportInfo = dispatcher
            ? ANEFieldfareANEExpertDispatcherGetInfo(dispatcher)
            : (ANEFieldfareANEExpertDispatchInfo){0};
        uint64_t targetPagesBefore = dispatcher ? residentPagesForExpert(copy, 0) : UINT64_MAX;
        bool validColdStart = warmupCompleted && targetInitiallyCold && targetPagesBefore == 0 &&
                              !transportInfo.explicitReadEnabled;

        DispatchMeasurement cold = {0};
        uint64_t targetPagesAfterCold = UINT64_MAX;
        bool invalidColdnessRejected = false;
        DispatchMeasurement warm[RotationCount] = {0};
        if (validColdStart) {
            cold = measureCall(dispatcher, 0, input, coldOutput);
            targetPagesAfterCold = residentPagesForExpert(copy, 0);
            // This is the negative control: applying the cold admission rule
            // to the now-resident range must reject it before the warm call.
            invalidColdnessRejected = targetPagesAfterCold != UINT64_MAX &&
                                      targetPagesAfterCold > 0;
            for (uint32_t run = 0; run < RotationCount && cold.completed; ++run)
                warm[run] = measureCall(dispatcher, 0, input, warmOutputs[run]);
        }

        const uint32_t rotatingExperts[RotationCount] = {1, 17, 33, 49, 65, 81, 97, 113};
        uint64_t rotatingPagesBefore[RotationCount] = {0};
        DispatchMeasurement rotating[RotationCount] = {0};
        bool everyRotatingRangeCold = true;
        for (uint32_t run = 0; run < RotationCount && cold.completed; ++run) {
            rotatingPagesBefore[run] = residentPagesForExpert(copy, rotatingExperts[run]);
            everyRotatingRangeCold &= rotatingPagesBefore[run] == 0;
            if (!everyRotatingRangeCold) break;
            rotating[run] = measureCall(dispatcher, rotatingExperts[run], input,
                                        rotatingOutputs[run]);
        }

        bool repeatedExact = cold.completed;
        double warmPreparationTotal = 0, warmWallTotal = 0;
        double rotatingPreparationTotal = 0, rotatingWallTotal = 0;
        for (uint32_t run = 0; run < RotationCount; ++run) {
            repeatedExact &= warm[run].completed &&
                memcmp(coldOutput, warmOutputs[run], sizeof(coldOutput)) == 0;
            warmPreparationTotal += warm[run].preparationMs;
            warmWallTotal += warm[run].wallMs;
            rotatingPreparationTotal += rotating[run].preparationMs;
            rotatingWallTotal += rotating[run].wallMs;
        }

        // Independent numerical oracle runs only after timing, and against the
        // original installation vnode so it cannot warm the copied fixture.
        ANEFieldfareRoutedLayer *cpuLayer = NULL;
        bool oracleValid = ANEFieldfareRoutedLayerOpen(root.fileSystemRepresentation, 0, &cpuLayer);
        double maximumError = 0, rmse = 0;
        uint64_t compared = 0;
        uint32_t oracleExperts[RotationCount + 1] = {0, 1, 17, 33, 49, 65, 81, 97, 113};
        for (uint32_t run = 0; oracleValid && run < RotationCount + 1; ++run) {
            const float *actual = run == 0 ? coldOutput : rotatingOutputs[run - 1];
            oracleValid = ANEFieldfareRoutedLayerCPUExpert(cpuLayer, oracleExperts[run], input, oracle);
            for (uint32_t column = 0; oracleValid && column < ANEFieldfareGemma4HiddenSize; ++column) {
                double difference = fabs((double)actual[column] - oracle[column]);
                maximumError = fmax(maximumError, difference);
                rmse += difference * difference;
                ++compared;
                oracleValid = isfinite(actual[column]) &&
                    difference <= 0.05 + 0.03 * fabs(oracle[column]);
            }
        }
        if (compared) rmse = sqrt(rmse / compared);

        NSMutableArray *warmJSON = [NSMutableArray array];
        NSMutableArray *rotatingJSON = [NSMutableArray array];
        for (uint32_t run = 0; run < RotationCount; ++run) {
            [warmJSON addObject:measurementJSON(warm[run])];
            [rotatingJSON addObject:@{
                @"expert": @(rotatingExperts[run]),
                @"resident_pages_before": @(rotatingPagesBefore[run]),
                @"measurement": measurementJSON(rotating[run]),
            }];
        }

        bool validated = copied && validColdStart && cold.completed &&
            targetPagesAfterCold > 0 && invalidColdnessRejected && repeatedExact &&
            everyRotatingRangeCold && oracleValid;
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.expert-residency.v1",
            @"gate": @"B2",
            @"status": validated ? @"validated" : @"rejected",
            @"host": NSProcessInfo.processInfo.hostName,
            @"host_chip": sysctlString("machdep.cpu.brand_string"),
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
            @"ane_architecture": @"not independently exposed by consumed private runtime",
            @"exact_command": [NSString stringWithFormat:
                @"Scripts/with_benchmark_lock.sh .build/ane-fieldfare-transport-benchmarks expert-residency %@ .build > results/expert-residency-cold-warm-m5-macos27.json", root],
            @"exit_status": @(validated ? 0 : 1),
            @"compile_count": @(dispatcher ? 1 : 0),
            @"load_count": @(dispatcher ? 1 : 0),
            @"transport": @{
                @"mode": @"mmap_demand_opt_out_from_default_explicit_pread",
                @"mmap_environment": @"ANE_FIELDFARE_MMAP_EXPERT_READS=1",
                @"explicit_reads_enabled": @(transportInfo.explicitReadEnabled),
                @"prefetch_disabled": @YES,
            },
            @"physical_fixture": @{
                @"kind": @"bounded full-layer pwrite copy with F_NOCACHE on source and destination; no clonefile",
                @"logical_bytes": @(LayerBytes),
                @"allocated_bytes": @(allocatedBytes),
                @"allocation_at_least_90_percent": @((BOOL)(allocatedBytes >= LayerBytes * 9 / 10)),
                @"deleted_after_run": @YES,
            },
            @"cold_target": @{
                @"expert": @0,
                @"resident_pages_before_dispatcher": @(initialTargetPages),
                @"resident_pages_before": @(targetPagesBefore),
                @"resident_pages_after": @(targetPagesAfterCold),
                @"measurement": measurementJSON(cold),
            },
            @"warm_same_expert": @{
                @"first_immediate": measurementJSON(warm[0]),
                @"runs": warmJSON,
                @"mean_preparation_ms": @(warmPreparationTotal / RotationCount),
                @"mean_wall_ms": @(warmWallTotal / RotationCount),
                @"byte_exact_to_cold_output": @(repeatedExact),
            },
            @"rotating_cold_experts": @{
                @"runs": rotatingJSON,
                @"every_range_zero_pages_before": @(everyRotatingRangeCold),
                @"mean_preparation_ms": @(rotatingPreparationTotal / RotationCount),
                @"mean_wall_ms": @(rotatingWallTotal / RotationCount),
            },
            @"reconciliation": @{
                @"historical_repeated_ms": @0.851,
                @"historical_rotating_ms": @1.44,
                @"measured_cold_target_over_warm_mean_preparation":
                    @(cold.preparationMs / (warmPreparationTotal / RotationCount)),
                @"measured_rotating_cold_over_warm_mean_preparation":
                    @((rotatingPreparationTotal / RotationCount) /
                      (warmPreparationTotal / RotationCount)),
                @"fixed_prep_breakdown_evaluation_completed": @YES,
                @"interpretation": @"The historical direction is retested with physical page-residency admission. This gate attributes only the measured mmap preparation difference; wall time also includes ANE evaluation.",
            },
            @"numerical_result": @{
                @"independent_cpu_experts_checked": @(RotationCount + 1),
                @"values_compared": @(compared),
                @"max_absolute_error": @(maximumError),
                @"rmse": @(rmse),
                @"tolerance": @"abs_error <= 0.05 + 0.03*abs(cpu)",
                @"validated": @(oracleValid),
            },
            @"negative_control": @{
                @"construction": @"reapply zero-resident-page cold admission after target dispatch",
                @"resident_pages_observed": @(targetPagesAfterCold),
                @"invalid_coldness_rejected": @(invalidColdnessRejected),
            },
            @"claim_boundary": @"Validates cold-versus-immediately-warm preparation and wall timing only for one F_NOCACHE physical copy, mmap demand paging, layer 0, one target expert, and eight widely spaced rotating experts. Every admitted cold range has a zero-page mincore proof, and outputs pass an independent compact CPU expert oracle. It does not attribute default explicit-pread performance, generalize to end-to-end token throughput, or separate SSD service from page-fault and carrier-fill costs inside preparation.",
            @"error": @(createError ?: ANEFieldfareANEExpertDispatcherLastError(dispatcher) ?: "none"),
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted |
                                                               NSJSONWritingSortedKeys error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);

        if (cpuLayer) ANEFieldfareRoutedLayerClose(cpuLayer);
        if (dispatcher) ANEFieldfareANEExpertDispatcherDestroy(dispatcher);
        free((void *)createError);
        if (copy) unlink(copy.fileSystemRepresentation);
        if (temporaryDirectory) rmdir(temporaryDirectory.fileSystemRepresentation);
        return validated ? 0 : 1;
    }
}
