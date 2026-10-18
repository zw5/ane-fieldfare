// Cold expert-transport ceiling gate.
//
// Each arm consumes a distinct packed layer.  Before timing, mincore(2) must
// report that none of the file's pages are resident.  This is deliberately a
// strict gate: a warm number is not relabelled as storage throughput.

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include <fcntl.h>
#include <mach/mach_time.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <unistd.h>

enum { ExpertCount = 128, ExpertStride = 3358720 };
static const uint64_t ExpectedLayerBytes = (uint64_t)ExpertCount * ExpertStride;
static volatile uint64_t sink;

typedef NS_ENUM(uint32_t, Method) {
    MethodMMapDemand,
    MethodMMapWillNeed,
    MethodPRead,
    MethodPReadNoCache,
    MethodDispatchIO,
};

static uint64_t nowTicks(void) { return mach_absolute_time(); }

static double secondsBetween(uint64_t begin, uint64_t end) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)(end - begin) * tb.numer / tb.denom / 1e9;
}

static NSString *sysctlString(const char *name) {
    size_t size = 0;
    if (sysctlbyname(name, NULL, &size, NULL, 0) || size == 0) return @"unknown";
    char *value = calloc(size + 1, 1);
    if (!value || sysctlbyname(name, value, &size, NULL, 0)) { free(value); return @"unknown"; }
    NSString *result = [NSString stringWithUTF8String:value] ?: @"unknown";
    free(value);
    return result;
}

static const char *methodName(Method method) {
    switch (method) {
        case MethodMMapDemand: return "mmap_demand_fault";
        case MethodMMapWillNeed: return "mmap_madv_willneed";
        case MethodPRead: return "pread";
        case MethodPReadNoCache: return "pread_f_nocache";
        case MethodDispatchIO: return "dispatch_io";
    }
}

// A fixed permutation prevents a sequential scan from being mistaken for the
// expert-aligned access pattern while still touching every expert exactly once.
static uint32_t expertForTask(uint32_t task) { return (task * 73u + 19u) & 127u; }

static uint64_t sampledChecksum(const uint8_t *bytes, size_t count) {
    uint64_t sum = 0;
    for (size_t offset = 0; offset < count; offset += 4096) sum += bytes[offset];
    if (count) sum += bytes[count - 1];
    return sum;
}

static uint64_t residentPages(const void *mapping, size_t length) {
    size_t pageSize = (size_t)getpagesize();
    size_t pages = (length + pageSize - 1) / pageSize;
    char *status = calloc(pages, 1);
    if (!status || mincore((void *)mapping, length, status)) { free(status); return UINT64_MAX; }
    uint64_t resident = 0;
    for (size_t page = 0; page < pages; ++page) resident += (status[page] & MINCORE_INCORE) != 0;
    free(status);
    return resident;
}

static NSDictionary *rejectedArm(Method method, uint32_t concurrency, NSString *file,
                                  uint64_t resident, NSString *reason) {
    return @{ @"method": @(methodName(method)), @"concurrency": @(concurrency),
              @"file": file.lastPathComponent, @"status": @"rejected",
              @"resident_pages_before": @(resident), @"reason": reason };
}

static NSDictionary *runMMap(Method method, uint32_t concurrency, NSString *file, int fd,
                             const uint8_t *mapping, uint64_t residentBefore) {
    (void)fd;
    uint8_t **scratch = calloc(concurrency, sizeof(*scratch));
    for (uint32_t i = 0; i < concurrency; ++i) scratch[i] = malloc(ExpertStride);
    for (uint32_t i = 0; i < concurrency; ++i) if (!scratch[i]) {
        for (uint32_t j = 0; j < concurrency; ++j) free(scratch[j]); free(scratch);
        return rejectedArm(method, concurrency, file, residentBefore, @"scratch allocation failed");
    }
    __block _Atomic uint32_t next = 0;
    __block _Atomic uint64_t checksum = 0;
    uint64_t begin = nowTicks();
    dispatch_apply(concurrency, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t worker) {
        for (;;) {
            uint32_t task = atomic_fetch_add_explicit(&next, 1, memory_order_relaxed);
            if (task >= ExpertCount) break;
            uint32_t expert = expertForTask(task);
            const uint8_t *source = mapping + (uint64_t)expert * ExpertStride;
            if (method == MethodMMapWillNeed) madvise((void *)source, ExpertStride, MADV_WILLNEED);
            memcpy(scratch[worker], source, ExpertStride);
            atomic_fetch_add_explicit(&checksum, sampledChecksum(scratch[worker], ExpertStride),
                                      memory_order_relaxed);
        }
    });
    double elapsed = secondsBetween(begin, nowTicks());
    sink ^= atomic_load_explicit(&checksum, memory_order_relaxed);
    for (uint32_t i = 0; i < concurrency; ++i) free(scratch[i]); free(scratch);
    double mbps = (double)ExpectedLayerBytes / elapsed / 1e6;
    return @{ @"method": @(methodName(method)), @"concurrency": @(concurrency),
              @"file": file.lastPathComponent, @"status": @"accepted",
              @"resident_pages_before": @(residentBefore), @"seconds": @(elapsed),
              @"bytes": @(ExpectedLayerBytes), @"mb_per_second": @(mbps),
              @"checksum": @(atomic_load(&checksum)) };
}

static NSDictionary *runPRead(Method method, uint32_t concurrency, NSString *file, int fd,
                              uint64_t residentBefore) {
    if (method == MethodPReadNoCache && fcntl(fd, F_NOCACHE, 1) != 0)
        return rejectedArm(method, concurrency, file, residentBefore, @"F_NOCACHE failed");
    uint8_t **scratch = calloc(concurrency, sizeof(*scratch));
    for (uint32_t i = 0; i < concurrency; ++i) scratch[i] = malloc(ExpertStride);
    for (uint32_t i = 0; i < concurrency; ++i) if (!scratch[i]) {
        for (uint32_t j = 0; j < concurrency; ++j) free(scratch[j]); free(scratch);
        return rejectedArm(method, concurrency, file, residentBefore, @"scratch allocation failed");
    }
    __block _Atomic uint32_t next = 0;
    __block _Atomic uint32_t failures = 0;
    __block _Atomic uint64_t checksum = 0;
    uint64_t begin = nowTicks();
    dispatch_apply(concurrency, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t worker) {
        for (;;) {
            uint32_t task = atomic_fetch_add_explicit(&next, 1, memory_order_relaxed);
            if (task >= ExpertCount) break;
            uint32_t expert = expertForTask(task);
            ssize_t got = pread(fd, scratch[worker], ExpertStride, (off_t)((uint64_t)expert * ExpertStride));
            if (got != ExpertStride) { atomic_fetch_add(&failures, 1); continue; }
            atomic_fetch_add(&checksum, sampledChecksum(scratch[worker], ExpertStride));
        }
    });
    double elapsed = secondsBetween(begin, nowTicks());
    sink ^= atomic_load(&checksum);
    for (uint32_t i = 0; i < concurrency; ++i) free(scratch[i]); free(scratch);
    if (atomic_load(&failures))
        return rejectedArm(method, concurrency, file, residentBefore, @"short or failed pread");
    return @{ @"method": @(methodName(method)), @"concurrency": @(concurrency),
              @"file": file.lastPathComponent, @"status": @"accepted",
              @"resident_pages_before": @(residentBefore),
              @"seconds": @(elapsed), @"bytes": @(ExpectedLayerBytes),
              @"mb_per_second": @((double)ExpectedLayerBytes / elapsed / 1e6),
              @"checksum": @(atomic_load(&checksum)) };
}

static NSDictionary *runDispatchIO(uint32_t concurrency, NSString *file, int fd,
                                   uint64_t residentBefore) {
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_io_t channel = dispatch_io_create(DISPATCH_IO_RANDOM, fd, queue, ^(int error) {
        (void)error;
    });
    if (!channel) return rejectedArm(MethodDispatchIO, concurrency, file, residentBefore,
                                     @"dispatch_io_create failed");
    dispatch_io_set_low_water(channel, ExpertStride);
    dispatch_io_set_high_water(channel, ExpertStride);
    __block _Atomic uint32_t failures = 0;
    __block _Atomic uint64_t checksum = 0;
    uint64_t begin = nowTicks();
    for (uint32_t first = 0; first < ExpertCount; first += concurrency) {
        dispatch_group_t wave = dispatch_group_create();
        uint32_t last = MIN(first + concurrency, ExpertCount);
        for (uint32_t task = first; task < last; ++task) {
            uint32_t expert = expertForTask(task);
            dispatch_group_enter(wave);
            dispatch_io_read(channel, (off_t)((uint64_t)expert * ExpertStride), ExpertStride, queue,
                             ^(bool done, dispatch_data_t data, int error) {
                if (error) atomic_fetch_add(&failures, 1);
                dispatch_data_apply(data, ^bool(dispatch_data_t region, size_t offset,
                                                const void *buffer, size_t size) {
                    (void)region; (void)offset;
                    atomic_fetch_add(&checksum, sampledChecksum(buffer, size));
                    return true;
                });
                if (done) dispatch_group_leave(wave);
            });
        }
        dispatch_group_wait(wave, DISPATCH_TIME_FOREVER);
    }
    dispatch_io_close(channel, 0);
    double elapsed = secondsBetween(begin, nowTicks());
    sink ^= atomic_load(&checksum);
    if (atomic_load(&failures))
        return rejectedArm(MethodDispatchIO, concurrency, file, residentBefore,
                           @"dispatch_io read failed");
    return @{ @"method": @(methodName(MethodDispatchIO)), @"concurrency": @(concurrency),
              @"file": file.lastPathComponent, @"status": @"accepted",
              @"resident_pages_before": @(residentBefore), @"seconds": @(elapsed),
              @"bytes": @(ExpectedLayerBytes),
              @"mb_per_second": @((double)ExpectedLayerBytes / elapsed / 1e6),
              @"checksum": @(atomic_load(&checksum)) };
}

static NSDictionary *runArm(Method method, uint32_t concurrency, NSString *file) {
    int fd = open(file.fileSystemRepresentation, O_RDONLY | O_CLOEXEC);
    struct stat st = {0};
    if (fd < 0 || fstat(fd, &st) || st.st_size != (off_t)ExpectedLayerBytes) {
        if (fd >= 0) close(fd);
        return rejectedArm(method, concurrency, file, UINT64_MAX, @"missing or wrong-size layer");
    }
    const uint8_t *mapping = mmap(NULL, ExpectedLayerBytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) { close(fd); return rejectedArm(method, concurrency, file, UINT64_MAX,
                                                               @"mmap failed"); }
    uint64_t resident = residentPages(mapping, ExpectedLayerBytes);
    NSDictionary *result;
    if (resident != 0) {
        result = rejectedArm(method, concurrency, file, resident,
                             resident == UINT64_MAX ? @"mincore failed" : @"cache contamination");
    } else if (method == MethodMMapDemand || method == MethodMMapWillNeed) {
        result = runMMap(method, concurrency, file, fd, mapping, resident);
    } else if (method == MethodDispatchIO) {
        result = runDispatchIO(concurrency, file, fd, resident);
    } else {
        result = runPRead(method, concurrency, file, fd, resident);
    }
    munmap((void *)mapping, ExpectedLayerBytes);
    close(fd);
    return result;
}

static NSDictionary *runDD(NSString *file) {
    int fd = open(file.fileSystemRepresentation, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return @{ @"status": @"rejected", @"reason": @"open failed" };
    const void *mapping = mmap(NULL, ExpectedLayerBytes, PROT_READ, MAP_PRIVATE, fd, 0);
    uint64_t resident = mapping == MAP_FAILED ? UINT64_MAX : residentPages(mapping, ExpectedLayerBytes);
    if (mapping != MAP_FAILED) munmap((void *)mapping, ExpectedLayerBytes);
    close(fd);
    if (resident != 0)
        return @{ @"status": @"rejected", @"file": file.lastPathComponent,
                  @"resident_pages_before": @(resident), @"reason": @"cache contamination" };
    NSString *input = [@"if=" stringByAppendingString:file];
    int reportPipe[2];
    if (pipe(reportPipe))
        return @{ @"status": @"rejected", @"file": file.lastPathComponent,
                  @"resident_pages_before": @(resident), @"reason": @"pipe failed" };
    uint64_t begin = nowTicks();
    pid_t child = fork();
    if (child == 0) {
        close(reportPipe[0]);
        dup2(reportPipe[1], STDERR_FILENO);
        close(reportPipe[1]);
        execl("/bin/dd", "dd", input.fileSystemRepresentation, "of=/dev/null", "bs=3358720",
              (char *)NULL);
        _exit(127);
    }
    close(reportPipe[1]);
    NSMutableData *reportData = [NSMutableData data];
    uint8_t reportBuffer[4096];
    for (;;) {
        ssize_t count = read(reportPipe[0], reportBuffer, sizeof(reportBuffer));
        if (count <= 0) break;
        [reportData appendBytes:reportBuffer length:(NSUInteger)count];
    }
    close(reportPipe[0]);
    int status = 0;
    if (child < 0 || waitpid(child, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status))
        return @{ @"status": @"rejected", @"file": file.lastPathComponent,
                  @"resident_pages_before": @(resident), @"reason": @"dd failed" };
    double wrapperElapsed = secondsBetween(begin, nowTicks());
    NSString *report = [[NSString alloc] initWithData:reportData encoding:NSUTF8StringEncoding] ?: @"";
    NSRegularExpression *pattern = [NSRegularExpression
        regularExpressionWithPattern:@"transferred in ([0-9.]+) secs" options:0 error:nil];
    NSTextCheckingResult *match = [pattern firstMatchInString:report options:0
                                                       range:NSMakeRange(0, report.length)];
    if (!match || match.numberOfRanges < 2)
        return @{ @"status": @"rejected", @"file": file.lastPathComponent,
                  @"resident_pages_before": @(resident), @"reason": @"could not parse dd timer",
                  @"dd_report": report };
    double elapsed = [[report substringWithRange:[match rangeAtIndex:1]] doubleValue];
    return @{ @"status": @"accepted", @"file": file.lastPathComponent,
              @"resident_pages_before": @(resident), @"seconds": @(elapsed),
              @"wrapper_seconds": @(wrapperElapsed), @"dd_report": report,
              @"bytes": @(ExpectedLayerBytes),
              @"mb_per_second": @((double)ExpectedLayerBytes / elapsed / 1e6),
              @"command": [NSString stringWithFormat:@"/bin/dd if=%@ of=/dev/null bs=%d",
                            file, ExpertStride] };
}

// Build a physically written, cache-bypassing copy corpus when the host does
// not permit the privileged system-wide purge operation.  This is not an APFS
// clone: every byte passes through aligned F_NOCACHE reads and writes, and the
// benchmark still refuses the result unless mincore independently sees zero
// resident destination pages.
static int prepareColdCorpus(NSString *sourceRoot, NSString *destinationRoot) {
    NSFileManager *manager = NSFileManager.defaultManager;
    NSString *packed = [destinationRoot stringByAppendingPathComponent:@"packed_experts"];
    NSError *directoryError = nil;
    if (![manager createDirectoryAtPath:packed withIntermediateDirectories:YES
                              attributes:nil error:&directoryError]) {
        fprintf(stderr, "cold corpus directory: %s\n", directoryError.localizedDescription.UTF8String);
        return 1;
    }
    void *buffer = NULL;
    const size_t chunk = 4 * 1024 * 1024;
    if (posix_memalign(&buffer, 16384, chunk)) return 1;
    for (uint32_t layer = 2; layer < 30; ++layer) {
        NSString *name = [NSString stringWithFormat:@"layer_%02u.bin", layer];
        NSString *source = [[sourceRoot stringByAppendingPathComponent:@"packed_experts"]
                            stringByAppendingPathComponent:name];
        NSString *destination = [packed stringByAppendingPathComponent:name];
        int in = open(source.fileSystemRepresentation, O_RDONLY | O_CLOEXEC);
        int out = open(destination.fileSystemRepresentation,
                       O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (in < 0 || out < 0 || fcntl(in, F_NOCACHE, 1) || fcntl(out, F_NOCACHE, 1)) {
            if (in >= 0) close(in); if (out >= 0) close(out); free(buffer); return 1;
        }
        uint64_t copied = 0;
        while (copied < ExpectedLayerBytes) {
            size_t wanted = (size_t)MIN((uint64_t)chunk, ExpectedLayerBytes - copied);
            ssize_t got = pread(in, buffer, wanted, (off_t)copied);
            ssize_t put = got > 0 ? pwrite(out, buffer, (size_t)got, (off_t)copied) : -1;
            if (got != (ssize_t)wanted || put != got) {
                close(in); close(out); free(buffer); return 1;
            }
            copied += (uint64_t)got;
        }
        if (fsync(out)) { close(in); close(out); free(buffer); return 1; }
        close(in); close(out);
    }
    free(buffer);
    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc == 4 && !strcmp(argv[1], "--prepare-cold-corpus"))
            return prepareColdCorpus([NSString stringWithUTF8String:argv[2]],
                                     [NSString stringWithUTF8String:argv[3]]);
        if (argc != 2) { fprintf(stderr, "usage: %s <model-directory>\n", argv[0]); return 2; }
        NSString *root = [NSString stringWithUTF8String:argv[1]];
        NSString *(^layer)(uint32_t) = ^NSString *(uint32_t index) {
            return [root stringByAppendingPathComponent:
                    [NSString stringWithFormat:@"packed_experts/layer_%02u.bin", index]];
        };
        // Three cold sequential controls absorb ordinary SSD/run variance; the
        // maximum is the conservative upper-bound control.  The 25 sweep cells
        // use layers 26..2, so no cell benefits from an earlier arm.
        NSArray *ddControls = @[runDD(layer(29)), runDD(layer(28)), runDD(layer(27))];
        NSDictionary *dd = nil;
        for (NSDictionary *candidate in ddControls)
            if ([candidate[@"status"] isEqual:@"accepted"] &&
                (!dd || [candidate[@"mb_per_second"] doubleValue] > [dd[@"mb_per_second"] doubleValue]))
                dd = candidate;
        if (!dd) dd = ddControls.firstObject;
        NSMutableArray *rawArms = [NSMutableArray array];
        const uint32_t concurrencies[] = {1, 2, 4, 8, 16};
        uint32_t nextLayer = 26;
        for (Method method = MethodMMapDemand; method <= MethodDispatchIO; ++method)
            for (uint32_t i = 0; i < 5; ++i)
                [rawArms addObject:runArm(method, concurrencies[i], layer(nextLayer--))];

        double ceiling = [dd[@"status"] isEqual:@"accepted"] ? [dd[@"mb_per_second"] doubleValue] : 0;
        NSDictionary *current = nil;
        BOOL sweepValid = ceiling > 0;
        NSMutableArray *arms = [NSMutableArray array];
        NSMutableArray *ceilingViolations = [NSMutableArray array];
        for (NSDictionary *rawArm in rawArms) {
            NSMutableDictionary *arm = [rawArm mutableCopy];
            if (![arm[@"status"] isEqual:@"accepted"]) sweepValid = NO;
            double rate = [arm[@"mb_per_second"] doubleValue];
            if (ceiling > 0 && rate > ceiling) {
                [ceilingViolations addObject:@{ @"method": arm[@"method"],
                                                 @"concurrency": arm[@"concurrency"],
                                                 @"mb_per_second": @(rate) }];
                arm[@"status"] = @"rejected";
                arm[@"reason"] = @"reported rate exceeds cold sequential dd ceiling; number rejected as cache-contaminated";
                [arm removeObjectForKey:@"mb_per_second"];
            }
            if ([arm[@"method"] isEqual:@"mmap_demand_fault"] &&
                [arm[@"concurrency"] unsignedIntValue] == 8) current = arm;
            [arms addObject:arm];
        }
        double analogueFraction = ceiling > 0 ? [current[@"mb_per_second"] doubleValue] / ceiling : 0;
        double documentedLowFraction = ceiling > 0 ? 371.0 / ceiling : 0;
        double documentedHighFraction = ceiling > 0 ? 490.0 / ceiling : 0;
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.transport-ceiling.v1",
            @"status": sweepValid ? @"validated" : @"rejected",
            @"gate": @"A1",
            @"host": NSProcessInfo.processInfo.hostName,
            @"host_chip": sysctlString("machdep.cpu.brand_string"),
            @"os_version": NSProcessInfo.processInfo.operatingSystemVersionString,
            @"os_build": sysctlString("kern.osversion"),
            @"ane_architecture": sysctlString("hw.target"),
            @"exact_command": [NSString stringWithFormat:@"%@.build/ane-fieldfare-transport-ceiling-gate %@",
                NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_COLD_RESET_PREFIX"] ?: @"", root],
            @"exit_status": @(sweepValid ? 0 : 1),
            @"expert_stride_bytes": @(ExpertStride),
            @"experts_per_layer": @(ExpertCount),
            @"access_order": @"expert-aligned full-layer fixed permutation (task*73+19 mod 128)",
            @"coldness_oracle": @"mincore resident_pages_before == 0 on a distinct layer per arm",
            @"cold_corpus_preparation": NSProcessInfo.processInfo.environment[@"ANE_FIELDFARE_COLD_CORPUS_NOTE"]
                ?: @"original model layers; no preparation declared",
            @"sequential_dd_ceiling": dd,
            @"sequential_dd_controls": ddControls,
            @"arms": arms,
            @"current_path_documented_mb_per_second": @{ @"low": @371, @"high": @490,
                @"provenance": @"docs/EXECUTION_GATES.md current generation-path range" },
            @"current_path_fraction_of_dd": @{ @"low": @(documentedLowFraction),
                                                @"high": @(documentedHighFraction) },
            @"current_path_fraction_of_dd_primary": @(documentedHighFraction),
            @"current_path_fraction_primary_definition": @"490 MB/s, the high end of the documented current generation-path range, divided by the cold dd ceiling",
            @"mmap_demand_concurrency_8_microbenchmark_fraction_of_dd": @(analogueFraction),
            @"microbenchmark_boundary": @"The full-layer memcpy sweep is an access-method comparison, not a substitute for measured generation-path disk throughput.",
            @"ceiling_violations": ceilingViolations,
            @"negative_control": @"Any nonzero pre-run resident-page count rejects that arm; any arm above the cold dd ceiling rejects the sweep.",
            @"compile_count": @0,
            @"load_count": @0,
            @"numerical_result": @"not applicable; byte transport only, checksums prevent elision",
            @"claim_status": sweepValid ? @"validated" : @"rejected",
            @"claim_boundary": sweepValid
                ? @"Validates the accepted cold packed-expert transport cells on this host for the tested methods and concurrency only. Cells above the dd ceiling are explicitly rejected and carry no accepted rate. This does not measure dequantization, ANE evaluation, token throughput, or warmed UBC reuse."
                : @"The cold transport sweep is rejected because at least one arm was cache-contaminated, failed, or exceeded the sequential cold-dd control."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        return sweepValid ? 0 : 1;
    }
}
