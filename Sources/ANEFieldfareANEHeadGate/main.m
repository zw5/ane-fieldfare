#import <Foundation/Foundation.h>

#include <float.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

#include "../ANEFieldfareRuntime/ANETiledHead.h"
#include "../ANEFieldfareRuntime/Decode.h"

enum {
    DefaultTileRows = 1472,
    DefaultFixtureToken = 2,
    DefaultRequestBanks = 2,
    WeightBits = 4,
    GroupSize = 64,
};

typedef struct {
    uint64_t physicalFootprint;
    uint64_t residentBytes;
    uint64_t peakResidentBytes;
} MemorySnapshot;

static mach_timebase_info_data_t gTimebase;

static double elapsedMilliseconds(uint64_t start, uint64_t end) {
    if (!gTimebase.denom) mach_timebase_info(&gTimebase);
    return (double)(end - start) * gTimebase.numer / gTimebase.denom / 1e6;
}

static MemorySnapshot memorySnapshot(void) {
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t result = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    if (result != KERN_SUCCESS) return (MemorySnapshot){UINT64_MAX, UINT64_MAX, UINT64_MAX};
    return (MemorySnapshot){info.phys_footprint, info.resident_size, info.resident_size_peak};
}

static NSDictionary *memoryDictionary(MemorySnapshot snapshot) {
    return @{
        @"physical_footprint_bytes": snapshot.physicalFootprint == UINT64_MAX
            ? NSNull.null : @(snapshot.physicalFootprint),
        @"resident_bytes": snapshot.residentBytes == UINT64_MAX
            ? NSNull.null : @(snapshot.residentBytes),
        @"peak_resident_bytes": snapshot.peakResidentBytes == UINT64_MAX
            ? NSNull.null : @(snapshot.peakResidentBytes),
    };
}

static void emit(NSDictionary *value) {
    NSError *error = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys
                                                     error:&error];
    if (!json) {
        fprintf(stderr, "receipt serialization failed: %s\n", error.localizedDescription.UTF8String);
        return;
    }
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

static NSString *runtimeCommand(int argc, const char *argv[]) {
    NSMutableArray<NSString *> *parts = [NSMutableArray arrayWithCapacity:(NSUInteger)argc];
    for (int index = 0; index < argc; ++index) {
        NSString *value = [NSString stringWithUTF8String:argv[index]] ?: @"";
        NSString *escaped = [value stringByReplacingOccurrencesOfString:@"'" withString:@"'\\''"];
        [parts addObject:[NSString stringWithFormat:@"'%@'", escaped]];
    }
    return [parts componentsJoinedByString:@" "];
}

static BOOL parseUnsigned(const char *text, uint32_t minimum, uint32_t maximum,
                          uint32_t *value) {
    if (!text || !value || !*text) return NO;
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 10);
    if (*end || parsed < minimum || parsed > maximum) return NO;
    *value = (uint32_t)parsed;
    return YES;
}

static NSString *statusString(BOOL value) {
    return value ? @"validated" : @"rejected";
}

static NSString *sysctlString(const char *name) {
    size_t bytes = 0;
    if (sysctlbyname(name, NULL, &bytes, NULL, 0) != 0 || bytes < 2) return @"unknown";
    char *value = calloc(bytes, 1);
    if (!value) return @"unknown";
    NSString *result = sysctlbyname(name, value, &bytes, NULL, 0) == 0
        ? [NSString stringWithUTF8String:value] : @"unknown";
    free(value);
    return result ?: @"unknown";
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2 || argc > 5) {
            emit(@{
                @"schema": @"ane-fieldfare.ane-tiled-head.v1",
                @"status": @"rejected",
                @"stage": @"arguments",
                @"usage": @"ane-fieldfare-ane-head-gate MODEL_DIR [TILE_ROWS] [FIXTURE_TOKEN] [REQUEST_BANKS]",
            });
            return 2;
        }
        uint32_t tileRows = DefaultTileRows;
        uint32_t fixtureToken = DefaultFixtureToken;
        uint32_t requestBanks = DefaultRequestBanks;
        if ((argc >= 3 && (!parseUnsigned(argv[2], 64, 16384, &tileRows) || tileRows % 64)) ||
            (argc >= 4 && !parseUnsigned(argv[3], 0, ANEFieldfareGemmaVocabularySize - 1,
                                         &fixtureToken)) ||
            (argc >= 5 && !parseUnsigned(argv[4], 1, 8, &requestBanks))) {
            emit(@{@"schema": @"ane-fieldfare.ane-tiled-head.v1", @"status": @"rejected",
                   @"stage": @"arguments",
                   @"error": @"invalid tile rows, fixture token, or request-bank count"});
            return 2;
        }

        NSString *modelRoot = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *weightsPath = [modelRoot stringByAppendingPathComponent:@"model_weights.bin"];
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor embedding = {0};
        float *hidden = calloc(ANEFieldfareGemmaHiddenSize, sizeof(*hidden));
        float *cpuLogits = calloc(ANEFieldfareGemmaVocabularySize, sizeof(*cpuLogits));
        float *aneLogits = calloc(ANEFieldfareGemmaVocabularySize, sizeof(*aneLogits));
        float *tileOutput = calloc((uint64_t)tileRows * requestBanks, sizeof(*tileOutput));
        BOOL checkpointValid = hidden && cpuLogits && aneLogits && tileOutput &&
            ANEFieldfareTensorIndexOpen(weightsPath.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(&index, "language_model.model.embed_tokens.weight", &embedding) &&
            embedding.rows == ANEFieldfareGemmaVocabularySize &&
            embedding.columns == ANEFieldfareGemmaHiddenSize &&
            ANEFieldfareScaledEmbedding(&index, fixtureToken, hidden) &&
            ANEFieldfareFinalRMSNorm(&index, 1e-6f, hidden);

        uint32_t cpuToken = 0;
        float cpuWinningLogit = 0;
        double cpuIterations[3] = {0};
        double cpuMilliseconds = DBL_MAX;
        BOOL cpuValid = checkpointValid;
        for (uint32_t iteration = 0; cpuValid && iteration < 3; ++iteration) {
            uint64_t cpuStart = mach_absolute_time();
            cpuValid = ANEFieldfareTiedGreedy(
                &index, hidden, tileRows, cpuLogits, &cpuToken, &cpuWinningLogit);
            cpuIterations[iteration] = elapsedMilliseconds(cpuStart, mach_absolute_time());
            cpuMilliseconds = fmin(cpuMilliseconds, cpuIterations[iteration]);
        }
        if (!cpuValid) cpuMilliseconds = 0;

        MemorySnapshot beforeCreate = memorySnapshot();
        const char *errorMessage = NULL;
        uint64_t createStart = mach_absolute_time();
        ANEFieldfareANETiledHead *head = cpuValid ? ANEFieldfareANETiledHeadCreate(
            ANEFieldfareGemmaHiddenSize, tileRows, GroupSize, requestBanks, &errorMessage) : NULL;
        double createWallMilliseconds = elapsedMilliseconds(createStart, mach_absolute_time());
        MemorySnapshot afterCreate = memorySnapshot();
        ANEFieldfareANETiledHeadInfo info = ANEFieldfareANETiledHeadGetInfo(head);

        double preparationMilliseconds = 0;
        double enqueueMilliseconds = 0;
        double evaluationMilliseconds = 0;
        double readbackMilliseconds = 0;
        double warmupMilliseconds = 0;
        __block _Atomic uint32_t tilesCompleted = 0;
        uint32_t aneToken = 0;
        float aneWinningRaw = -FLT_MAX;
        ANEFieldfareANEHeadTileTiming warmupTiming = {0};
        uint32_t firstTileRows = MIN(tileRows, embedding.rows);
        uint64_t warmupStart = mach_absolute_time();
        BOOL warmupValid = head && ANEFieldfareANETiledHeadEvaluateTile(
            head, 0, hidden, &embedding, 0, firstTileRows, tileOutput,
            &warmupTiming, &errorMessage);
        warmupMilliseconds = elapsedMilliseconds(warmupStart, mach_absolute_time());

        uint64_t aneStart = mach_absolute_time();
        __block _Atomic bool evaluationFailed = !warmupValid;
        __block const char *evaluationError = NULL;
        NSLock *errorLock = [[NSLock alloc] init];
        ANEFieldfareANEHeadTileTiming *bankTimings = calloc(
            requestBanks, sizeof(*bankTimings));
        dispatch_queue_t queues[8] = {0};
        dispatch_group_t group = dispatch_group_create();
        for (uint32_t bank = 0; bank < requestBanks; ++bank)
            queues[bank] = dispatch_queue_create(
                "com.openai.ane-fieldfare.head-bank", DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
        uint32_t tileIndex = 0;
        for (uint32_t firstRow = 0; firstRow < embedding.rows; firstRow += tileRows, ++tileIndex) {
            uint32_t remaining = embedding.rows - firstRow;
            uint32_t rowCount = tileRows < remaining ? tileRows : remaining;
            uint32_t bank = tileIndex % requestBanks;
            dispatch_group_async(group, queues[bank], ^{
                if (atomic_load_explicit(&evaluationFailed, memory_order_relaxed)) return;
                ANEFieldfareANEHeadTileTiming tileTiming = {0};
                const char *tileError = NULL;
                float *bankOutput = tileOutput + (uint64_t)bank * tileRows;
                BOOL ok = ANEFieldfareANETiledHeadEvaluateTile(
                    head, bank, hidden, &embedding, firstRow, rowCount, bankOutput,
                    &tileTiming, &tileError);
                if (ok) {
                    for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
                        float raw = bankOutput[localRow];
                        if (!isfinite(raw)) {
                            ok = NO;
                            break;
                        }
                        uint32_t token = firstRow + localRow;
                        aneLogits[token] = raw;
                    }
                }
                if (!ok) {
                    atomic_store_explicit(&evaluationFailed, true, memory_order_relaxed);
                    [errorLock lock];
                    if (!evaluationError)
                        evaluationError = strdup(tileError ?: "ANE tiled-head evaluation failed");
                    [errorLock unlock];
                } else {
                    bankTimings[bank].preparationMilliseconds += tileTiming.preparationMilliseconds;
                    bankTimings[bank].enqueueMilliseconds += tileTiming.enqueueMilliseconds;
                    bankTimings[bank].evaluationMilliseconds += tileTiming.evaluationMilliseconds;
                    bankTimings[bank].readbackMilliseconds += tileTiming.readbackMilliseconds;
                    atomic_fetch_add_explicit(&tilesCompleted, 1, memory_order_relaxed);
                }
                free((void *)tileError);
            });
        }
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        for (uint32_t bank = 0; bank < requestBanks; ++bank) {
            preparationMilliseconds += bankTimings[bank].preparationMilliseconds;
            enqueueMilliseconds += bankTimings[bank].enqueueMilliseconds;
            evaluationMilliseconds += bankTimings[bank].evaluationMilliseconds;
            readbackMilliseconds += bankTimings[bank].readbackMilliseconds;
        }
        BOOL aneValid = !atomic_load_explicit(&evaluationFailed, memory_order_relaxed);
        for (uint32_t token = 0; aneValid && token < embedding.rows; ++token) {
            float raw = aneLogits[token];
            if (raw > aneWinningRaw) {
                aneWinningRaw = raw;
                aneToken = token;
            }
            aneLogits[token] = 30.0f * tanhf(raw / 30.0f);
        }
        double aneWallMilliseconds = elapsedMilliseconds(aneStart, mach_absolute_time());
        float aneWinningLogit = 30.0f * tanhf(aneWinningRaw / 30.0f);
        MemorySnapshot afterRun = memorySnapshot();

        double squareError = 0;
        float maximumAbsoluteError = 0;
        uint32_t toleranceFailures = 0;
        uint32_t nonfiniteLogits = 0;
        for (uint32_t token = 0; cpuValid && aneValid && token < embedding.rows; ++token) {
            float cpu = cpuLogits[token];
            float ane = aneLogits[token];
            if (!isfinite(cpu) || !isfinite(ane)) {
                nonfiniteLogits++;
                continue;
            }
            float error = fabsf(cpu - ane);
            maximumAbsoluteError = fmaxf(maximumAbsoluteError, error);
            squareError += (double)error * error;
            float tolerance = 0.50f + 0.03f * fabsf(cpu);
            if (error > tolerance) toleranceFailures++;
        }
        double rootMeanSquareError = embedding.rows
            ? sqrt(squareError / embedding.rows) : 0;
        BOOL top1Matches = cpuValid && aneValid && cpuToken == aneToken;
        BOOL logitsMatch = cpuValid && aneValid && nonfiniteLogits == 0 && toleranceFailures == 0;
        BOOL numericalValidation = top1Matches && logitsMatch;
        double speedup = numericalValidation && aneWallMilliseconds > 0
            ? cpuMilliseconds / aneWallMilliseconds : 0;
        BOOL performancePromoted = numericalValidation && speedup >= 1.10;
        uint64_t tiles = ((uint64_t)embedding.rows + tileRows - 1) / tileRows;
        uint64_t fullCodeCarrierBytes = (uint64_t)embedding.rows * embedding.columns * sizeof(__fp16);
        uint64_t fullParameterCarrierBytes =
            (uint64_t)embedding.rows * (embedding.columns / GroupSize) * 2 * sizeof(__fp16);
        uint64_t compactHeadBytes = embedding.weightBytes + embedding.scaleBytes + embedding.biasBytes;
        const char *reportedError = errorMessage ?: evaluationError;
        NSString *failure = reportedError
            ? [NSString stringWithUTF8String:reportedError]
            : (checkpointValid ? @"none" : @"checkpoint fixture construction failed");

        NSProcessInfo *process = NSProcessInfo.processInfo;
        NSString *buildCommand = @"xcrun clang -O3 -Wall -Wextra -fobjc-arc -fblocks -framework Foundation -framework IOSurface -ldl Sources/ANEFieldfareRuntime/TensorIndex.c Sources/ANEFieldfareRuntime/Decode.c Sources/ANEFieldfareRuntime/ANEProgram.m Sources/ANEFieldfareRuntime/ANETiledHead.m Sources/ANEFieldfareANEHeadGate/main.m -o .build/ane-fieldfare-ane-head-gate";
        emit(@{
            @"schema": @"ane-fieldfare.ane-tiled-head.v1",
            @"status": statusString(numericalValidation),
            @"stage": numericalValidation ? @"full_vocabulary_tied_head_oracle" : @"ane_tiled_head",
            @"exit_status": numericalValidation ? @0 : @1,
            @"host": process.hostName ?: @"unknown",
            @"os": process.operatingSystemVersionString ?: @"unknown",
            @"host_chip": sysctlString("machdep.cpu.brand_string"),
            @"ane_architecture": @"not independently exposed by the consumed private runtime surface",
            @"commands": @{
                @"build": buildCommand,
                @"run": runtimeCommand(argc, argv),
            },
            @"checkpoint": @{
                @"path": weightsPath,
                @"embedding_name": @"language_model.model.embed_tokens.weight",
                @"shape": @[@(embedding.rows), @(embedding.columns)],
                @"quantization": @{@"bits": @(WeightBits), @"group_size": @(GroupSize),
                                    @"scale": @"BF16", @"bias": @"BF16"},
                @"compact_head_bytes": @(compactHeadBytes),
            },
            @"fixture": @{
                @"kind": @"authentic checkpoint embedding row followed by final RMSNorm",
                @"token": @(fixtureToken),
                @"note": @"Deterministic numerical fixture, not a completed decoder hidden state.",
            },
            @"graph": @{
                @"tile_rows": @(tileRows),
                @"tile_count": @(tiles),
                @"tiles_completed": @(tilesCompleted),
                @"ane_batch_lanes": @64,
                @"rows_per_lane": @(tileRows / 64),
                @"request_banks": @(info.requestBanks),
                @"input_bytes_per_bank": @(info.inputBytesPerBank),
                @"output_bytes_per_bank": @(info.outputBytesPerBank),
                @"full_scan_unpacked_code_bytes": @(fullCodeCarrierBytes),
                @"full_scan_scale_bias_bytes": @(fullParameterCarrierBytes),
                @"dequantization": @"ANE graph: fp16(q_code) * fp16(BF16 scale) + fp16(BF16 bias)",
            },
            @"private_ane_lifecycle": @{
                @"compile_count": head ? @1 : @0,
                @"load_count": head ? @1 : @0,
                @"program_reused_for_all_tiles": @((BOOL)(head != NULL)),
            },
            @"timing_ms": @{
                @"cpu_current_decode_head": @(cpuMilliseconds),
                @"cpu_current_decode_head_iterations": @[
                    @(cpuIterations[0]), @(cpuIterations[1]), @(cpuIterations[2])],
                @"ane_create_wall": @(createWallMilliseconds),
                @"ane_compile": @(info.program.compileMilliseconds),
                @"ane_load": @(info.program.loadMilliseconds),
                @"ane_full_scan_wall": @(aneWallMilliseconds),
                @"ane_warmup_tile_wall": @(warmupMilliseconds),
                @"ane_host_preparation_sum": @(preparationMilliseconds),
                @"ane_enqueue_sum": @(enqueueMilliseconds),
                @"ane_evaluation_completion_sum": @(evaluationMilliseconds),
                @"ane_readback_sum": @(readbackMilliseconds),
            },
            @"memory": @{
                @"before_create": memoryDictionary(beforeCreate),
                @"after_create": memoryDictionary(afterCreate),
                @"after_run": memoryDictionary(afterRun),
            },
            @"oracle": @{
                @"cpu_token": @(cpuToken),
                @"ane_token": @(aneToken),
                @"top1_matches": @(top1Matches),
                @"cpu_winning_softcapped_logit": @(cpuWinningLogit),
                @"ane_winning_softcapped_logit": @(aneWinningLogit),
                @"maximum_absolute_softcapped_logit_error": @(maximumAbsoluteError),
                @"root_mean_square_softcapped_logit_error": @(rootMeanSquareError),
                @"tolerance": @"abs_error <= 0.50 + 0.03*abs(cpu_softcapped_logit)",
                @"tolerance_failures": @(toleranceFailures),
                @"nonfinite_logits": @(nonfiniteLogits),
                @"all_logits_within_tolerance": @(logitsMatch),
            },
            @"performance": @{
                @"speedup_vs_current_cpu_head": @(speedup),
                @"promotion_threshold": @"at least 1.10x versus best of three warm CPU scans",
                @"status": performancePromoted ? @"promoted" : @"rejected",
                @"full_vocabulary_executed": @((BOOL)(atomic_load_explicit(
                    &tilesCompleted, memory_order_relaxed) == tiles)),
                @"cold_compile_load_excluded_from_speedup": @YES,
            },
            @"error": failure,
            @"claim_boundary": numericalValidation
                ? (performancePromoted
                    ? @"One compiled and loaded private-ANE tiled head scanned all 262,144 authentic tied-embedding rows, performed affine dequantization and projection in the ANE graph, matched every softcapped CPU-oracle logit within the declared tolerance, preserved greedy top-1, and beat the current CPU tied-head wall time for this deterministic checkpoint fixture. This does not validate a decoder-produced hidden state or end-to-end generation speedup."
                    : @"One compiled and loaded private-ANE tiled head scanned all 262,144 authentic tied-embedding rows, performed affine dequantization and projection in the ANE graph, matched every softcapped CPU-oracle logit within the declared tolerance, and preserved greedy top-1. It did not beat the current CPU tied-head wall time for this fixture, so it is a validated capability and a rejected performance replacement.")
                : @"The full-vocabulary private-ANE tiled tied-head construction did not pass its independent CPU numerical oracle; no performance replacement or generated token is claimed.",
        });

        free((void *)errorMessage);
        free((void *)evaluationError);
        free(bankTimings);
        ANEFieldfareANETiledHeadDestroy(head);
        if (index.fileDescriptor >= 0) ANEFieldfareTensorIndexClose(&index);
        free(tileOutput);
        free(aneLogits);
        free(cpuLogits);
        free(hidden);
        return numericalValidation ? 0 : 1;
    }
}
