#import <Foundation/Foundation.h>

#include <fcntl.h>
#include <mach/mach_time.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include "../ANEFieldfareRuntime/ANEExpertDispatch.h"

enum { ExpertStride = 3358720, ExpertCount = 128, TestPositions = 4 };
static const uint64_t LayerBytes = (uint64_t)ExpertStride * ExpertCount;

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
    if (!value || sysctlbyname(name, value, &size, NULL, 0)) { free(value); return @"unknown"; }
    NSString *result = [NSString stringWithUTF8String:value] ?: @"unknown";
    free(value); return result;
}

static uint64_t checksum(const float *values, size_t count) {
    uint64_t hash = 1469598103934665603ULL;
    const uint8_t *bytes = (const uint8_t *)values;
    for (size_t index = 0; index < count * sizeof(float); ++index) {
        hash ^= bytes[index]; hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t residentPages(NSString *path) {
    int fd = open(path.fileSystemRepresentation, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return UINT64_MAX;
    void *mapping = mmap(NULL, LayerBytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) return UINT64_MAX;
    size_t pages = (LayerBytes + (uint64_t)getpagesize() - 1) / (uint64_t)getpagesize();
    char *state = calloc(pages, 1);
    uint64_t resident = UINT64_MAX;
    if (state && mincore(mapping, LayerBytes, state) == 0) {
        resident = 0;
        for (size_t i = 0; i < pages; ++i) resident += (state[i] & MINCORE_INCORE) != 0;
    }
    free(state); munmap(mapping, LayerBytes); return resident;
}

static bool copyNoCache(NSString *source, NSString *destination) {
    int in = open(source.fileSystemRepresentation, O_RDONLY | O_CLOEXEC);
    int out = open(destination.fileSystemRepresentation,
                   O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    void *buffer = NULL;
    const size_t chunk = 4 * 1024 * 1024;
    bool valid = in >= 0 && out >= 0 && posix_memalign(&buffer, 16384, chunk) == 0 &&
        fcntl(in, F_NOCACHE, 1) == 0 && fcntl(out, F_NOCACHE, 1) == 0;
    for (uint64_t offset = 0; valid && offset < LayerBytes;) {
        size_t wanted = (size_t)MIN((uint64_t)chunk, LayerBytes - offset);
        ssize_t got = pread(in, buffer, wanted, (off_t)offset);
        ssize_t put = got > 0 ? pwrite(out, buffer, (size_t)got, (off_t)offset) : -1;
        valid = got == (ssize_t)wanted && put == got;
        if (valid) offset += (uint64_t)got;
    }
    valid = valid && fsync(out) == 0;
    free(buffer); if (in >= 0) close(in); if (out >= 0) close(out);
    return valid;
}

static ANEFieldfareANEExpertDispatcher *createDispatcher(NSString *path, bool explicitReads,
                                                         NSString **error) {
    if (explicitReads) unsetenv("ANE_FIELDFARE_MMAP_EXPERT_READS");
    else setenv("ANE_FIELDFARE_MMAP_EXPERT_READS", "1", 1);
    const char *message = NULL;
    ANEFieldfareANEExpertDispatcher *dispatcher =
        ANEFieldfareANEExpertDispatcherCreate(path.fileSystemRepresentation, &message);
    if (!dispatcher && error) *error = message ? [NSString stringWithUTF8String:message] : @"unknown";
    free((void *)message);
    unsetenv("ANE_FIELDFARE_MMAP_EXPERT_READS");
    return dispatcher;
}

static NSDictionary *timedBatch(NSString *path, bool explicitReads, const float *input) {
    NSString *createError = nil;
    ANEFieldfareANEExpertDispatcher *dispatcher = createDispatcher(path, explicitReads, &createError);
    if (!dispatcher) return @{ @"status": @"rejected", @"reason": createError ?: @"create failed" };
    uint64_t resident = residentPages(path);
    uint32_t experts[ANEFieldfareGemma4TopK] = {0,1,2,3,4,5,6,7};
    float outputs[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize] = {{0}};
    uint64_t begin = ticks();
    bool valid = ANEFieldfareANEExpertDispatcherCallBatch(dispatcher, experts, input, outputs);
    double elapsed = milliseconds(begin, ticks());
    ANEFieldfareANEExpertDispatchInfo info = ANEFieldfareANEExpertDispatcherGetInfo(dispatcher);
    NSString *failure = valid ? nil : [NSString stringWithUTF8String:
        ANEFieldfareANEExpertDispatcherLastError(dispatcher) ?: "unknown"];
    uint64_t hash = checksum(&outputs[0][0], ANEFieldfareGemma4TopK * ANEFieldfareGemma4HiddenSize);
    ANEFieldfareANEExpertDispatcherDestroy(dispatcher);
    if (!valid || resident != 0)
        return @{ @"status": @"rejected", @"reason": failure ?: @"cache contamination",
                  @"resident_pages_before": @(resident) };
    return @{ @"status": @"validated", @"transport": explicitReads ? @"explicit_pread" : @"mmap_demand",
              @"resident_pages_before": @(resident), @"wall_milliseconds": @(elapsed),
              @"logical_expert_bytes": @((uint64_t)ExpertStride * ANEFieldfareGemma4TopK),
              @"explicit_read_bytes": @(info.explicitReadBytes),
              @"explicit_read_worker_milliseconds": @(info.explicitReadMilliseconds),
              @"output_fnv1a64": [NSString stringWithFormat:@"0x%016llx", hash] };
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) { fprintf(stderr, "usage: %s <model-directory>\n", argv[0]); return 2; }
        NSString *root = [NSString stringWithUTF8String:argv[1]];
        NSString *layer0 = [root stringByAppendingPathComponent:@"packed_experts/layer_00.bin"];
        float input[ANEFieldfareGemma4HiddenSize];
        for (uint32_t i = 0; i < ANEFieldfareGemma4HiddenSize; ++i)
            input[i] = (float)((int32_t)(i % 97) - 48) / 256.0f;

        NSString *controlError = nil, *explicitError = nil;
        ANEFieldfareANEExpertDispatcher *control = createDispatcher(layer0, false, &controlError);
        ANEFieldfareANEExpertDispatcher *explicit = createDispatcher(layer0, true, &explicitError);
        bool valid = control && explicit;
        const uint32_t positions = TestPositions;
        float positionInputs[TestPositions][ANEFieldfareGemma4HiddenSize];
        float controlPositions[TestPositions][ANEFieldfareGemma4HiddenSize];
        float explicitPositions[TestPositions][ANEFieldfareGemma4HiddenSize];
        for (uint32_t p = 0; p < TestPositions; ++p)
            for (uint32_t i = 0; i < ANEFieldfareGemma4HiddenSize; ++i)
                positionInputs[p][i] = input[i] * (1.0f + (float)p / 16.0f);
        bool controlPositionOK = valid && ANEFieldfareANEExpertDispatcherCallPositions(
            control, 0, positions, &positionInputs[0][0], &controlPositions[0][0]);
        bool explicitPositionOK = valid && ANEFieldfareANEExpertDispatcherCallPositions(
            explicit, 0, positions, &positionInputs[0][0], &explicitPositions[0][0]);
        bool positionExact = controlPositionOK && explicitPositionOK &&
            memcmp(controlPositions, explicitPositions, sizeof(controlPositions)) == 0;

        uint32_t experts[ANEFieldfareGemma4TopK] = {0,1,2,3,4,5,6,7};
        float controlBatch[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize] = {{0}};
        float explicitBatch[ANEFieldfareGemma4TopK][ANEFieldfareGemma4HiddenSize] = {{0}};
        bool controlBatchOK = valid && ANEFieldfareANEExpertDispatcherCallBatch(
            control, experts, input, controlBatch);
        bool explicitBatchOK = valid && ANEFieldfareANEExpertDispatcherCallBatch(
            explicit, experts, input, explicitBatch);
        bool batchExact = controlBatchOK && explicitBatchOK &&
            memcmp(controlBatch, explicitBatch, sizeof(controlBatch)) == 0;
        ANEFieldfareANEExpertDispatchInfo explicitInfo =
            ANEFieldfareANEExpertDispatcherGetInfo(explicit);

        // The file validates at switch time, then is shortened under the open
        // descriptor.  Expert 127 must fail with an actual partial-read/EOF;
        // falling back to the still-mapped bytes would make this control pass.
        char corruptTemplate[] = ".build/a3-short-read-XXXXXX";
        int corruptFD = mkstemp(corruptTemplate);
        bool negativeRejected = false;
        NSString *negativeError = @"setup failed";
        if (corruptFD >= 0 && ftruncate(corruptFD, (off_t)LayerBytes) == 0) {
            close(corruptFD); corruptFD = -1;
            const char *switchError = NULL;
            bool switched = ANEFieldfareANEExpertDispatcherSetPackedLayerPath(
                explicit, corruptTemplate, &switchError);
            free((void *)switchError);
            if (switched && truncate(corruptTemplate,
                    (off_t)((uint64_t)127 * ExpertStride + ExpertStride / 2)) == 0) {
                float ignored[ANEFieldfareGemma4HiddenSize] = {0};
                negativeRejected = !ANEFieldfareANEExpertDispatcherCall(explicit, 127, input, ignored);
                negativeError = [NSString stringWithUTF8String:
                    ANEFieldfareANEExpertDispatcherLastError(explicit) ?: "none"];
            }
        }
        if (corruptFD >= 0) close(corruptFD);
        unlink(corruptTemplate);

        ANEFieldfareANEExpertDispatcherDestroy(control);
        ANEFieldfareANEExpertDispatcherDestroy(explicit);

        // Four physical cold copies provide an order-balanced comparison.
        char tempTemplate[] = ".build/a3-cold-XXXXXX";
        char *tempPath = mkdtemp(tempTemplate);
        NSMutableArray *timings = [NSMutableArray array];
        bool coldValid = tempPath != NULL;
        const bool modes[4] = {false, true, true, false};
        for (uint32_t arm = 0; coldValid && arm < 4; ++arm) {
            NSString *source = [root stringByAppendingPathComponent:
                [NSString stringWithFormat:@"packed_experts/layer_%02u.bin", arm + 2]];
            NSString *destination = [[NSString stringWithUTF8String:tempPath]
                stringByAppendingPathComponent:[NSString stringWithFormat:@"arm_%u.bin", arm]];
            coldValid = copyNoCache(source, destination) && residentPages(destination) == 0;
            if (coldValid) [timings addObject:timedBatch(destination, modes[arm], input)];
            unlink(destination.fileSystemRepresentation);
        }
        if (tempPath) rmdir(tempPath);
        for (NSDictionary *timing in timings)
            coldValid = coldValid && [timing[@"status"] isEqual:@"validated"];
        double mmapTotal = 0, explicitTotal = 0;
        uint32_t mmapCount = 0, explicitCount = 0;
        for (NSDictionary *timing in timings) {
            if ([timing[@"transport"] isEqual:@"mmap_demand"]) {
                mmapTotal += [timing[@"wall_milliseconds"] doubleValue]; ++mmapCount;
            } else if ([timing[@"transport"] isEqual:@"explicit_pread"]) {
                explicitTotal += [timing[@"wall_milliseconds"] doubleValue]; ++explicitCount;
            }
        }
        double mmapMean = mmapCount ? mmapTotal / mmapCount : 0;
        double explicitMean = explicitCount ? explicitTotal / explicitCount : 0;

        valid = valid && positionExact && batchExact && negativeRejected &&
            [negativeError containsString:@"unexpected EOF"] &&
            explicitInfo.explicitReadEnabled &&
            explicitInfo.explicitReadBytes == (uint64_t)ExpertStride * 9 && coldValid && timings.count == 4;
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.explicit-read.v1", @"gate": @"A3",
            @"status": valid ? @"validated" : @"rejected",
            @"claim_status": valid ? @"validated" : @"rejected",
            @"host": NSProcessInfo.processInfo.hostName,
            @"host_chip": sysctlString("machdep.cpu.brand_string"),
            @"os_version": NSProcessInfo.processInfo.operatingSystemVersionString,
            @"os_build": sysctlString("kern.osversion"), @"ane_architecture": sysctlString("hw.target"),
            @"exact_command": [NSString stringWithFormat:@".build/ane-fieldfare-explicit-read-gate %@ > results/explicit-read-m5-macos27.json", root],
            @"benchmark_lock": @".build/ane-fieldfare-benchmark.lock directory acquired before command and removed afterward",
            @"exit_status": @(valid ? 0 : 1), @"compile_count": @6, @"load_count": @6,
            @"position_oracle": @{ @"positions": @(positions), @"bit_exact": @(positionExact),
                @"control_checksum": [NSString stringWithFormat:@"0x%016llx", checksum(&controlPositions[0][0], positions * ANEFieldfareGemma4HiddenSize)],
                @"explicit_checksum": [NSString stringWithFormat:@"0x%016llx", checksum(&explicitPositions[0][0], positions * ANEFieldfareGemma4HiddenSize)] },
            @"batch_oracle": @{ @"experts": @[@0,@1,@2,@3,@4,@5,@6,@7], @"bit_exact": @(batchExact),
                @"control_checksum": [NSString stringWithFormat:@"0x%016llx", checksum(&controlBatch[0][0], ANEFieldfareGemma4TopK * ANEFieldfareGemma4HiddenSize)],
                @"explicit_checksum": [NSString stringWithFormat:@"0x%016llx", checksum(&explicitBatch[0][0], ANEFieldfareGemma4TopK * ANEFieldfareGemma4HiddenSize)] },
            @"explicit_read_accounting": @{ @"bytes": @(explicitInfo.explicitReadBytes),
                @"expected_bytes": @((uint64_t)ExpertStride * 9),
                @"worker_milliseconds": @(explicitInfo.explicitReadMilliseconds) },
            @"short_read_negative_control": @{ @"status": negativeRejected ? @"rejected" : @"accepted",
                @"construction": @"full-size sparse layer validated, then truncated midway through expert 127",
                @"error": negativeError },
            @"cold_order_balanced_timings": timings,
            @"cold_timing_summary": @{
                @"mmap_demand_mean_milliseconds": @(mmapMean),
                @"explicit_pread_mean_milliseconds": @(explicitMean),
                @"explicit_speedup": @(explicitMean > 0 ? mmapMean / explicitMean : 0),
                @"mmap_effective_logical_mb_per_second": @(mmapMean > 0 ?
                    (double)(ExpertStride * ANEFieldfareGemma4TopK) / mmapMean / 1000.0 : 0),
                @"explicit_effective_logical_mb_per_second": @(explicitMean > 0 ?
                    (double)(ExpertStride * ANEFieldfareGemma4TopK) / explicitMean / 1000.0 : 0)
            },
            @"coldness_oracle": @"F_NOCACHE physical copies, no APFS clones; mincore zero pages immediately before each timed CallBatch",
            @"numerical_result": valid ? @"CallPositions(4) and CallBatch(top-8) are bit-exact against the mmap production control" : @"oracle failed",
            @"negative_control": @"A post-validation truncated file must produce explicit pread unexpected EOF and must not fall back to mmap",
            @"claim_boundary": @"Validates opt-in complete-expert explicit reads, rank-owned top-8 buffers, exact ANE outputs, short-read failure, and representative cold CallBatch timing. Timing includes storage, dequant/carrier fill, and ANE evaluation and is not a CPU-cost attribution. It does not yet validate byte-identical end-to-end generated text."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        return valid ? 0 : 1;
    }
}
