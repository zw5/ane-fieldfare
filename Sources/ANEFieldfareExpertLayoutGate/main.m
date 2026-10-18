#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <mach/mach_time.h>

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// A4 is deliberately isolated from ANEExpertDispatch.m. This gate reproduces
// its exact source-read and carrier-write order against one authentic expert,
// then compares it with a bounded, in-memory repack. It is a hot/replay layout
// experiment, not a cold-storage throughput claim.
enum {
    Hidden = 2816,
    Intermediate = 704,
    Positions = 64,
    Group = 64,
    GroupsWide = Hidden / Group,
    GroupsNarrow = Intermediate / Group,
    GroupCodeBytes = Group / 2,
    RecordBytes = GroupCodeBytes + 2 + 2,
    PairRowsPerBlock = 16,
    DownRowsPerBlock = 32,
    PairBlocks = Intermediate / PairRowsPerBlock,
    DownBlocks = Hidden / DownRowsPerBlock,
    ExpertStride = 3358720,
    PayloadBytes = 3345408,
    CarrierStride = Positions + 3 * Intermediate,
    CarrierElements = Hidden * CarrierStride,
    GateWeightOffset = 0,
    GateScaleOffset = 991232,
    GateBiasOffset = 1053184,
    UpWeightOffset = 1115136,
    UpScaleOffset = 2106368,
    UpBiasOffset = 2168320,
    DownWeightOffset = 2230272,
    DownScaleOffset = 3221504,
    DownBiasOffset = 3283456,
};

_Static_assert(Intermediate % PairRowsPerBlock == 0, "pair blocks must tile exactly");
_Static_assert(Hidden % DownRowsPerBlock == 0, "down blocks must tile exactly");
_Static_assert(3 * 1115136 == PayloadBytes, "three matrices make the payload");

typedef struct {
    const uint8_t *weights;
    const uint8_t *scales;
    const uint8_t *biases;
    uint32_t columns;
} SplitMatrix;

static mach_timebase_info_data_t gTimebase;

static inline float bf16(const uint8_t *p) {
    uint32_t word = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

static inline void expandCodes(const uint8_t *codes, float scale, float bias, __fp16 *out) {
    for (uint32_t byte = 0; byte < GroupCodeBytes; ++byte) {
        uint8_t packed = codes[byte];
        out[2 * byte] = (__fp16)((packed & 15) * scale + bias);
        out[2 * byte + 1] = (__fp16)((packed >> 4) * scale + bias);
    }
}

static void expandSplitRow(SplitMatrix matrix, uint32_t row, __fp16 *out) {
    uint32_t groups = matrix.columns / Group;
    uint64_t firstGroup = (uint64_t)row * groups;
    const uint8_t *weights = matrix.weights + firstGroup * GroupCodeBytes;
    for (uint32_t group = 0; group < groups; ++group) {
        float scale = bf16(matrix.scales + 2 * (firstGroup + group));
        float bias = bf16(matrix.biases + 2 * (firstGroup + group));
        expandCodes(weights + (uint64_t)group * GroupCodeBytes, scale, bias,
                    out + (uint64_t)group * Group);
    }
}

static void fillSplitPairBlock(__fp16 *carrier, SplitMatrix gate, SplitMatrix up,
                               uint32_t block) {
    uint32_t rowBase = block * PairRowsPerBlock;
    __fp16 gateTile[PairRowsPerBlock][Hidden];
    __fp16 upTile[PairRowsPerBlock][Hidden];
    for (uint32_t row = 0; row < PairRowsPerBlock; ++row) {
        expandSplitRow(gate, rowBase + row, gateTile[row]);
        expandSplitRow(up, rowBase + row, upTile[row]);
    }
    for (uint32_t column = 0; column < Hidden; ++column) {
        __fp16 *destination = carrier + (uint64_t)column * CarrierStride + Positions + rowBase;
        for (uint32_t row = 0; row < PairRowsPerBlock; ++row) {
            destination[row] = gateTile[row][column];
            destination[Intermediate + row] = upTile[row][column];
        }
    }
}

static void fillSplitDownBlock(__fp16 *carrier, SplitMatrix down, uint32_t block) {
    uint32_t start = block * DownRowsPerBlock;
    for (uint32_t row = start; row < start + DownRowsPerBlock; ++row) {
        __fp16 *destination = carrier + (uint64_t)row * CarrierStride + Positions + 2 * Intermediate;
        expandSplitRow(down, row, destination);
    }
}

static void fillSplit(__fp16 *carrier, const uint8_t *base) {
    SplitMatrix gate = {base + GateWeightOffset, base + GateScaleOffset,
                        base + GateBiasOffset, Hidden};
    SplitMatrix up = {base + UpWeightOffset, base + UpScaleOffset,
                      base + UpBiasOffset, Hidden};
    SplitMatrix down = {base + DownWeightOffset, base + DownScaleOffset,
                        base + DownBiasOffset, Intermediate};
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(PairBlocks, queue, ^(size_t block) {
        fillSplitPairBlock(carrier, gate, up, (uint32_t)block);
    });
    dispatch_apply(DownBlocks, queue, ^(size_t block) {
        fillSplitDownBlock(carrier, down, (uint32_t)block);
    });
}

static void appendRowRecords(uint8_t **destination, SplitMatrix matrix, uint32_t row) {
    uint32_t groups = matrix.columns / Group;
    uint64_t first = (uint64_t)row * groups;
    for (uint32_t group = 0; group < groups; ++group) {
        memcpy(*destination, matrix.weights + (first + group) * GroupCodeBytes, GroupCodeBytes);
        memcpy(*destination + GroupCodeBytes, matrix.scales + 2 * (first + group), 2);
        memcpy(*destination + GroupCodeBytes + 2, matrix.biases + 2 * (first + group), 2);
        *destination += RecordBytes;
    }
}

static bool repackFillOrder(const uint8_t *base, uint8_t *ordered) {
    SplitMatrix gate = {base + GateWeightOffset, base + GateScaleOffset,
                        base + GateBiasOffset, Hidden};
    SplitMatrix up = {base + UpWeightOffset, base + UpScaleOffset,
                      base + UpBiasOffset, Hidden};
    SplitMatrix down = {base + DownWeightOffset, base + DownScaleOffset,
                        base + DownBiasOffset, Intermediate};
    uint8_t *cursor = ordered;
    for (uint32_t block = 0; block < PairBlocks; ++block) {
        uint32_t rowBase = block * PairRowsPerBlock;
        for (uint32_t row = 0; row < PairRowsPerBlock; ++row)
            appendRowRecords(&cursor, gate, rowBase + row);
        for (uint32_t row = 0; row < PairRowsPerBlock; ++row)
            appendRowRecords(&cursor, up, rowBase + row);
    }
    for (uint32_t block = 0; block < DownBlocks; ++block) {
        uint32_t rowBase = block * DownRowsPerBlock;
        for (uint32_t row = 0; row < DownRowsPerBlock; ++row)
            appendRowRecords(&cursor, down, rowBase + row);
    }
    return (size_t)(cursor - ordered) == PayloadBytes;
}

static void expandRecordRow(const uint8_t *records, uint32_t groups, __fp16 *out) {
    for (uint32_t group = 0; group < groups; ++group) {
        const uint8_t *record = records + (uint64_t)group * RecordBytes;
        expandCodes(record, bf16(record + GroupCodeBytes),
                    bf16(record + GroupCodeBytes + 2), out + (uint64_t)group * Group);
    }
}

static void fillOrderedPairBlock(__fp16 *carrier, const uint8_t *ordered, uint32_t block) {
    const uint64_t rowBytes = GroupsWide * RecordBytes;
    const uint64_t blockBytes = 2 * PairRowsPerBlock * rowBytes;
    const uint8_t *gate = ordered + (uint64_t)block * blockBytes;
    const uint8_t *up = gate + PairRowsPerBlock * rowBytes;
    uint32_t rowBase = block * PairRowsPerBlock;
    __fp16 gateTile[PairRowsPerBlock][Hidden];
    __fp16 upTile[PairRowsPerBlock][Hidden];
    for (uint32_t row = 0; row < PairRowsPerBlock; ++row) {
        expandRecordRow(gate + row * rowBytes, GroupsWide, gateTile[row]);
        expandRecordRow(up + row * rowBytes, GroupsWide, upTile[row]);
    }
    for (uint32_t column = 0; column < Hidden; ++column) {
        __fp16 *destination = carrier + (uint64_t)column * CarrierStride + Positions + rowBase;
        for (uint32_t row = 0; row < PairRowsPerBlock; ++row) {
            destination[row] = gateTile[row][column];
            destination[Intermediate + row] = upTile[row][column];
        }
    }
}

static void fillOrderedDownBlock(__fp16 *carrier, const uint8_t *ordered, uint32_t block) {
    const uint64_t pairBytes = (uint64_t)PairBlocks * 2 * PairRowsPerBlock *
                               GroupsWide * RecordBytes;
    const uint64_t rowBytes = GroupsNarrow * RecordBytes;
    const uint8_t *rows = ordered + pairBytes +
                          (uint64_t)block * DownRowsPerBlock * rowBytes;
    uint32_t start = block * DownRowsPerBlock;
    for (uint32_t row = 0; row < DownRowsPerBlock; ++row) {
        __fp16 *destination = carrier + (uint64_t)(start + row) * CarrierStride +
                              Positions + 2 * Intermediate;
        expandRecordRow(rows + row * rowBytes, GroupsNarrow, destination);
    }
}

static void fillOrdered(__fp16 *carrier, const uint8_t *ordered) {
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply(PairBlocks, queue, ^(size_t block) {
        fillOrderedPairBlock(carrier, ordered, (uint32_t)block);
    });
    dispatch_apply(DownBlocks, queue, ^(size_t block) {
        fillOrderedDownBlock(carrier, ordered, (uint32_t)block);
    });
}

static int compareDouble(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static volatile uint64_t gSink;

static NSDictionary *benchmarkInterleaved(void (^currentFill)(__fp16 *), __fp16 *currentCarrier,
                                          void (^orderedFill)(__fp16 *), __fp16 *orderedCarrier,
                                          uint32_t iterations) {
    double *current = calloc(iterations, sizeof(double));
    double *ordered = calloc(iterations, sizeof(double));
    currentFill(currentCarrier);
    orderedFill(orderedCarrier);
    for (uint32_t i = 0; i < iterations; ++i) {
        bool orderedFirst = (i & 1) != 0;
        for (uint32_t pass = 0; pass < 2; ++pass) {
            bool runOrdered = orderedFirst ^ (pass != 0);
            uint64_t start = mach_absolute_time();
            if (runOrdered) orderedFill(orderedCarrier); else currentFill(currentCarrier);
            uint64_t end = mach_absolute_time();
            double ms = (double)(end - start) * gTimebase.numer / gTimebase.denom / 1e6;
            (runOrdered ? ordered : current)[i] = ms;
            gSink ^= ((uint16_t *)(runOrdered ? orderedCarrier : currentCarrier))[i % CarrierElements];
        }
    }
    qsort(current, iterations, sizeof(double), compareDouble);
    qsort(ordered, iterations, sizeof(double), compareDouble);
    double currentMedian = current[iterations / 2], orderedMedian = ordered[iterations / 2];
    NSDictionary *(^summary)(double *) = ^NSDictionary *(double *values) {
        double median = values[iterations / 2];
        return @{ @"median_ms": @(median), @"p10_ms": @(values[iterations / 10]),
                  @"p90_ms": @(values[(iterations * 9) / 10]),
                  @"logical_source_MB_per_s": @((double)PayloadBytes / 1e6 / (median / 1e3)) };
    };
    NSDictionary *result = @{ @"current_split_planes": summary(current),
                              @"proposed_fill_order": summary(ordered),
                              @"speedup": @(currentMedian / orderedMedian),
                              @"iterations_per_arm": @(iterations),
                              @"schedule": @"paired alternating order" };
    free(current); free(ordered);
    return result;
}

static uint64_t fnv1a(const void *bytes, size_t length) {
    const uint8_t *p = bytes;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < length; ++i) hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <layer_XX.bin> [expert] [iterations]\n", argv[0]);
            return 2;
        }
        uint32_t expert = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 0;
        uint32_t iterations = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 31;
        if (expert >= 128 || iterations < 5) return 2;
        int fd = open(argv[1], O_RDONLY);
        struct stat st = {0};
        if (fd < 0 || fstat(fd, &st) || (uint64_t)st.st_size < (uint64_t)(expert + 1) * ExpertStride) {
            perror("open/stat layer");
            if (fd >= 0) close(fd);
            return 1;
        }
        const uint8_t *mapping = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
        const uint8_t *base = mapping + (uint64_t)expert * ExpertStride;
        uint8_t *ordered = aligned_alloc(16384, ExpertStride);
        __fp16 *splitCarrier = aligned_alloc(16384, CarrierElements * sizeof(__fp16));
        __fp16 *orderedCarrier = aligned_alloc(16384, CarrierElements * sizeof(__fp16));
        __fp16 *negativeCarrier = aligned_alloc(16384, CarrierElements * sizeof(__fp16));
        if (!ordered || !splitCarrier || !orderedCarrier || !negativeCarrier) return 1;
        memset(ordered, 0, ExpertStride);
        memset(splitCarrier, 0, CarrierElements * sizeof(__fp16));
        memset(orderedCarrier, 0, CarrierElements * sizeof(__fp16));
        memset(negativeCarrier, 0, CarrierElements * sizeof(__fp16));
        if (!repackFillOrder(base, ordered)) return 1;

        fillSplit(splitCarrier, base);
        fillOrdered(orderedCarrier, ordered);
        size_t carrierBytes = CarrierElements * sizeof(__fp16);
        bool byteExact = memcmp(splitCarrier, orderedCarrier, carrierBytes) == 0;
        uint64_t splitHash = fnv1a(splitCarrier, carrierBytes);
        uint64_t orderedHash = fnv1a(orderedCarrier, carrierBytes);

        // Negative control: replace the first group's codes and affine
        // parameters with a deliberately large, deterministic group. A single
        // nibble flip can legitimately round to the same FP16 value when the
        // authentic scale is tiny, so that would be a weak oracle.
        uint8_t saved[RecordBytes];
        memcpy(saved, ordered, RecordBytes);
        memset(ordered, 0xff, GroupCodeBytes);
        ordered[GroupCodeBytes] = 0x40;      // BF16 2.0 in file byte order
        ordered[GroupCodeBytes + 1] = 0x00;
        ordered[GroupCodeBytes + 2] = 0x00; // BF16 0.0 bias
        ordered[GroupCodeBytes + 3] = 0x00;
        fillOrdered(negativeCarrier, ordered);
        memcpy(ordered, saved, RecordBytes);
        bool negativeDetected = memcmp(splitCarrier, negativeCarrier, carrierBytes) != 0;
        uint64_t negativeHash = fnv1a(negativeCarrier, carrierBytes);

        mach_timebase_info(&gTimebase);
        NSDictionary *pairedTiming = benchmarkInterleaved(
            ^(__fp16 *carrier) { fillSplit(carrier, base); }, splitCarrier,
            ^(__fp16 *carrier) { fillOrdered(carrier, ordered); }, orderedCarrier, iterations);
        NSDictionary *splitTiming = pairedTiming[@"current_split_planes"];
        NSDictionary *orderedTiming = pairedTiming[@"proposed_fill_order"];
        double splitMs = [splitTiming[@"median_ms"] doubleValue];
        double orderedMs = [orderedTiming[@"median_ms"] doubleValue];
        bool validated = byteExact && negativeDetected;

        NSOperatingSystemVersion version = NSProcessInfo.processInfo.operatingSystemVersion;
        NSString *os = [NSString stringWithFormat:@"%ld.%ld.%ld",
                        version.majorVersion, version.minorVersion, version.patchVersion];
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.expert-layout.v1",
            @"gate": @"A4",
            @"host": NSProcessInfo.processInfo.hostName,
            @"os_version": os,
            @"ane_architecture": @"not_observed_host_only_layout_gate",
            @"command": [NSString stringWithFormat:@".build/ane-fieldfare-expert-layout-gate %s %u %u",
                         argv[1], expert, iterations],
            @"exit_status": @(validated ? 0 : 1),
            @"compile_count": @0,
            @"load_count": @0,
            @"fixture": @{ @"layer": @(argv[1]), @"expert": @(expert),
                            @"expert_stride_bytes": @(ExpertStride),
                            @"meaningful_payload_bytes": @(PayloadBytes),
                            @"padding_bytes": @(ExpertStride - PayloadBytes) },
            @"source_trace": @{
                @"current": @{
                    @"physical_layout": @"matrix-major; within each matrix: all codes, all BF16 scales, all BF16 biases",
                    @"inner_group_order": @"32 code bytes, jump to 2-byte scale plane, jump to 2-byte bias plane, jump back to next code group",
                    @"simultaneous_streams_per_worker": @3,
                    @"pair_schedule": @"44 parallel 16-row blocks; each block expands 16 gate rows then 16 up rows",
                    @"down_schedule": @"88 parallel 32-row blocks",
                    @"group_source_bytes": @36,
                },
                @"proposed": @{
                    @"physical_layout": @"dispatcher-block-major; each row contains 36-byte group records: 32 code + 2 scale + 2 bias",
                    @"simultaneous_streams_per_worker": @1,
                    @"pair_block_bytes": @(2 * PairRowsPerBlock * GroupsWide * RecordBytes),
                    @"down_block_bytes": @(DownRowsPerBlock * GroupsNarrow * RecordBytes),
                    @"total_bytes_unchanged": @YES,
                },
            },
            @"numerical_result": @{
                @"byte_exact_entire_carrier": @(byteExact),
                @"current_carrier_fnv1a64": [NSString stringWithFormat:@"%016llx", splitHash],
                @"proposed_carrier_fnv1a64": [NSString stringWithFormat:@"%016llx", orderedHash],
                @"negative_control_replaced_first_group_detected": @(negativeDetected),
                @"negative_carrier_fnv1a64": [NSString stringWithFormat:@"%016llx", negativeHash],
            },
            @"hot_replay_benchmark": @{
                @"iterations": @(iterations), @"current_split_planes": splitTiming,
                @"proposed_fill_order": orderedTiming,
                @"speedup": @(splitMs / orderedMs),
                @"schedule": pairedTiming[@"schedule"],
                @"stored_byte_reduction": @0,
                @"disk_MB_per_s": [NSNull null],
            },
            @"claim_status": validated ? @"validated" : @"rejected",
            @"status": validated ? @"validated" : @"rejected",
            @"optimization_decision": validated && splitMs / orderedMs < 1.02
                ? @"rejected_no_material_hot_win" : @"accepted_for_cold_followup_only",
            @"claim_boundary": @"Validates byte-equivalent repacking and measures hot/replay host expansion for one authentic expert. It does not establish a cold SSD win: the fixture is mmap-backed, repacked in memory, warmed before timing, and may reside in UBC. No production layout or 12 GB installation was modified.",
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted |
                                                               NSJSONWritingSortedKeys error:nil];
        fwrite(json.bytes, 1, json.length, stdout);
        fputc('\n', stdout);

        free(ordered); free(splitCarrier); free(orderedCarrier); free(negativeCarrier);
        munmap((void *)mapping, st.st_size); close(fd);
        return validated ? 0 : 1;
    }
}
