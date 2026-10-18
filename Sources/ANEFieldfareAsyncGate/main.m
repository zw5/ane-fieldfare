#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <objc/runtime.h>

#include <math.h>

#include "../ANEFieldfareRuntime/ANEDynamicProjection.h"

enum { Hidden = 2816, Outputs = 128, Banks = 8, Positions = 64 };

static double milliseconds(uint64_t ticks) {
    mach_timebase_info_data_t info = {0};
    mach_timebase_info(&info);
    return (double)ticks * info.numer / info.denom / 1e6;
}

static bool prepareBank(ANEFieldfareANEProgram *program,
                        uint32_t bank,
                        const ANEFieldfareTensor *weights,
                        float scale,
                        float expected[Outputs]) {
    IOSurfaceRef inputSurface = ANEFieldfareANEProgramInput(program, bank);
    IOSurfaceRef outputSurface = ANEFieldfareANEProgramOutput(program, bank);
    if (!inputSurface || !outputSurface || IOSurfaceLock(inputSurface, 0, NULL) != kIOReturnSuccess)
        return false;
    __fp16 *carrier = IOSurfaceGetBaseAddress(inputSurface);
    const uint32_t stride = Positions + Outputs;
    memset(carrier, 0, (size_t)Hidden * stride * sizeof(*carrier));
    for (uint32_t column = 0; column < Hidden; ++column) {
        float input = scale * (sinf((float)(column + 1) * 0.013f) * 0.125f);
        carrier[(uint64_t)column * stride] = (__fp16)input;
        for (uint32_t row = 0; row < Outputs; ++row) {
            float weight = 0;
            if (!ANEFieldfareAffineValue(weights, 8, 64, row, column, &weight)) {
                IOSurfaceUnlock(inputSurface, 0, NULL);
                return false;
            }
            carrier[(uint64_t)column * stride + Positions + row] = (__fp16)weight;
        }
    }
    for (uint32_t row = 0; row < Outputs; ++row) {
        double sum = 0;
        for (uint32_t column = 0; column < Hidden; ++column)
            sum += (float)carrier[(uint64_t)column * stride] *
                   (float)carrier[(uint64_t)column * stride + Positions + row];
        expected[row] = (float)sum;
    }
    IOSurfaceUnlock(inputSurface, 0, NULL);
    if (IOSurfaceLock(outputSurface, 0, NULL) != kIOReturnSuccess) return false;
    float *output = IOSurfaceGetBaseAddress(outputSurface);
    for (uint32_t index = 0; index < Outputs * Positions; ++index) output[index] = NAN;
    IOSurfaceUnlock(outputSurface, 0, NULL);
    return true;
}

static NSDictionary *validateBank(ANEFieldfareANEProgram *program,
                                  uint32_t bank,
                                  const float expected[Outputs]) {
    IOSurfaceRef surface = ANEFieldfareANEProgramOutput(program, bank);
    if (!surface || IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess)
        return @{ @"bank": @(bank), @"valid": @NO, @"error": @"output lock failed" };
    const float *output = IOSurfaceGetBaseAddress(surface);
    double maxError = 0, squared = 0;
    uint32_t failures = 0, poisonRemaining = 0, nonfinite = 0;
    for (uint32_t row = 0; row < Outputs; ++row) {
        float value = output[(uint64_t)row * Positions];
        if (isnan(value)) ++poisonRemaining;
        if (!isfinite(value)) { ++nonfinite; ++failures; continue; }
        double error = fabs((double)value - expected[row]);
        if (error > maxError) maxError = error;
        squared += error * error;
        if (error > 0.025 + 0.02 * fabs(expected[row])) ++failures;
        for (uint32_t position = 1; position < Positions; ++position) {
            float inactive = output[(uint64_t)row * Positions + position];
            if (isnan(inactive)) ++poisonRemaining;
            if (!isfinite(inactive) || fabsf(inactive) > 0.001f) ++failures;
        }
    }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return @{
        @"bank": @(bank), @"valid": @(failures == 0 && poisonRemaining == 0),
        @"max_absolute_error": @(maxError), @"rmse": @(sqrt(squared / Outputs)),
        @"tolerance_failures": @(failures), @"poison_values_remaining": @(poisonRemaining),
        @"nonfinite_active_outputs": @(nonfinite)
    };
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) return 2;
        NSString *root = [[NSString stringWithUTF8String:argv[1]] stringByStandardizingPath];
        NSString *path = [root stringByAppendingPathComponent:@"model_weights.bin"];
        ANEFieldfareTensorIndex index = {.fileDescriptor = -1};
        ANEFieldfareTensor router = {0};
        const char *creationError = NULL;
        bool valid = ANEFieldfareTensorIndexOpen(path.fileSystemRepresentation, &index) &&
            ANEFieldfareTensorIndexFind(
                &index, "language_model.model.layers.0.router.proj.weight", &router);
        ANEFieldfareANEDynamicProjection *projection = valid
            ? ANEFieldfareANEDynamicProjectionCreate(Hidden, Outputs, Banks, &creationError) : NULL;
        valid = valid && projection;
        ANEFieldfareANEProgram *program = ANEFieldfareANEDynamicProjectionProgram(projection);
        float expected[Banks][Outputs] = {{0}};
        for (uint32_t bank = 0; valid && bank < Banks; ++bank)
            valid = prepareBank(program, bank, &router, 1.0f + 0.0625f * bank, expected[bank]);

        NSMutableArray *serialRuns = [NSMutableArray array];
        uint64_t serialStarted = mach_absolute_time();
        for (uint32_t bank = 0; valid && bank < Banks; ++bank) {
            ANEFieldfareANEEvaluationTiming timing = {0};
            const char *error = NULL;
            bool accepted = ANEFieldfareANEProgramEvaluate(
                program, bank, 5ULL * NSEC_PER_SEC, &timing, &error);
            NSDictionary *oracle = accepted ? validateBank(program, bank, expected[bank]) : @{};
            valid = valid && accepted && [oracle[@"valid"] boolValue];
            [serialRuns addObject:@{
                @"bank": @(bank), @"accepted": @(accepted),
                @"enqueue_ms": @(timing.enqueueMilliseconds),
                @"completion_ms": @(timing.completionMilliseconds),
                @"completion_ordinal": @(timing.completionOrdinal),
                @"oracle": oracle,
                @"error": error ? [NSString stringWithUTF8String:error] : @"none"
            }];
            free((void *)error);
        }
        double serialWall = milliseconds(mach_absolute_time() - serialStarted);

        for (uint32_t bank = 0; valid && bank < Banks; ++bank)
            valid = prepareBank(program, bank, &router, 1.0f + 0.0625f * bank, expected[bank]);
        NSMutableArray *submissions = [NSMutableArray array];
        ANEFieldfareANEEvaluationTiming begins[Banks] = {{0}}, completions[Banks] = {{0}};
        uint64_t asyncStarted = mach_absolute_time();
        for (uint32_t bank = 0; valid && bank < Banks; ++bank) {
            const char *error = NULL;
            bool accepted = ANEFieldfareANEProgramEvaluateBegin(program, bank, &begins[bank], &error);
            valid = valid && accepted;
            [submissions addObject:@{
                @"bank": @(bank), @"accepted": @(accepted),
                @"enqueue_ms": @(begins[bank].enqueueMilliseconds),
                @"current_in_flight_after_return": @(
                    ANEFieldfareANEProgramCurrentAsyncRequestsInFlight(program)),
                @"error": error ? [NSString stringWithUTF8String:error] : @"none"
            }];
            free((void *)error);
        }
        NSMutableArray *completionRuns = [NSMutableArray array];
        for (uint32_t bank = 0; valid && bank < Banks; ++bank) {
            const char *error = NULL;
            bool completed = ANEFieldfareANEProgramEvaluateWait(
                program, bank, 5ULL * NSEC_PER_SEC, &completions[bank], &error);
            NSDictionary *oracle = completed ? validateBank(program, bank, expected[bank]) : @{};
            valid = valid && completed && [oracle[@"valid"] boolValue];
            [completionRuns addObject:@{
                @"bank": @(bank), @"completed": @(completed),
                @"completion_ms": @(completions[bank].completionMilliseconds),
                @"completion_ordinal": @(completions[bank].completionOrdinal),
                @"oracle": oracle,
                @"error": error ? [NSString stringWithUTF8String:error] : @"none"
            }];
            free((void *)error);
        }
        double asyncWall = milliseconds(mach_absolute_time() - asyncStarted);
        NSArray *completionOrder = [completionRuns sortedArrayUsingComparator:
            ^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
                return [a[@"completion_ordinal"] compare:b[@"completion_ordinal"]];
            }];

        // A bank remains logically in flight until its completion is consumed,
        // so a second begin on the same bank must fail even if the hardware has
        // already called the completion handler.
        prepareBank(program, 0, &router, 1.0f, expected[0]);
        const char *negativeError = NULL, *duplicateError = NULL, *negativeWaitError = NULL;
        ANEFieldfareANEEvaluationTiming negativeBegin = {0}, negativeWait = {0};
        bool firstAccepted = ANEFieldfareANEProgramEvaluateBegin(
            program, 0, &negativeBegin, &negativeError);
        bool duplicateAccepted = ANEFieldfareANEProgramEvaluateBegin(
            program, 0, NULL, &duplicateError);
        bool negativeCompleted = firstAccepted && ANEFieldfareANEProgramEvaluateWait(
            program, 0, 5ULL * NSEC_PER_SEC, &negativeWait, &negativeWaitError);
        bool duplicateRejected = firstAccepted && !duplicateAccepted && negativeCompleted;
        valid = valid && duplicateRejected;

        bool timeoutObserved = false, timeoutRecovered = false;
        uint32_t timeoutAttempt = 0;
        NSMutableArray *timeoutAttempts = [NSMutableArray array];
        for (uint32_t attempt = 0; valid && !timeoutObserved && attempt < Banks; ++attempt) {
            uint32_t bank = attempt;
            prepareBank(program, bank, &router, 0.75f + 0.03125f * bank, expected[bank]);
            const char *beginError = NULL, *timeoutError = NULL, *recoveryError = NULL;
            ANEFieldfareANEEvaluationTiming begin = {0}, recovered = {0};
            bool began = ANEFieldfareANEProgramEvaluateBegin(program, bank, &begin, &beginError);
            bool immediate = began && ANEFieldfareANEProgramEvaluateWait(
                program, bank, 0, NULL, &timeoutError);
            bool recoveredOK = immediate || ANEFieldfareANEProgramEvaluateWait(
                program, bank, 5ULL * NSEC_PER_SEC, &recovered, &recoveryError);
            [timeoutAttempts addObject:@{
                @"attempt": @(attempt), @"bank": @(bank), @"began": @(began),
                @"zero_timeout_completed": @(immediate), @"recovered": @(recoveredOK),
                @"timeout_error": timeoutError ? [NSString stringWithUTF8String:timeoutError] : @"none",
                @"recovery_error": recoveryError ? [NSString stringWithUTF8String:recoveryError] : @"none"
            }];
            if (began && !immediate) {
                timeoutObserved = true; timeoutRecovered = recoveredOK; timeoutAttempt = attempt;
            }
            valid = valid && began && recoveredOK;
            free((void *)beginError); free((void *)timeoutError); free((void *)recoveryError);
        }
        valid = valid && timeoutObserved && timeoutRecovered;

        ANEFieldfareANEProgramInfo info = ANEFieldfareANEDynamicProjectionGetInfo(projection);
        Class sharedEventClass = NSClassFromString(@"_ANESharedEvent");
        Class transactionClass = NSClassFromString(@"_ANETransaction");
        NSDictionary *receipt = @{
            @"schema": @"ane-fieldfare.async-shared-event.v1",
            @"status": valid ? @"validated" : @"rejected",
            @"host": NSProcessInfo.processInfo.hostName,
            @"os": NSProcessInfo.processInfo.operatingSystemVersionString,
            @"shape": @{ @"input": @(Hidden), @"output": @(Outputs),
                          @"positions": @(Positions), @"banks": @(Banks) },
            @"lifecycle": @{
                @"compile_count": @1, @"load_count": @1,
                @"compile_ms": @(info.compileMilliseconds), @"load_ms": @(info.loadMilliseconds),
                @"mapped_request_banks": @(info.requestBankCount),
                @"reported_queue_depth": @(info.reportedQueueDepth)
            },
            @"serial": @{ @"wall_ms": @(serialWall), @"runs": serialRuns },
            @"async": @{
                @"wall_ms": @(asyncWall), @"submissions": submissions,
                @"completions_by_bank": completionRuns,
                @"completion_order": [completionOrder valueForKey:@"bank"],
                @"maximum_current_async_requests_in_flight": @(
                    ANEFieldfareANEProgramMaximumAsyncRequestsInFlight(program)),
                @"speedup_vs_serial": asyncWall > 0 ? @(serialWall / asyncWall) : @0
            },
            @"negative_control": @{
                @"construction": @"duplicate_begin_on_same_mapped_bank",
                @"first_begin_accepted": @(firstAccepted),
                @"duplicate_begin_accepted": @(duplicateAccepted),
                @"duplicate_rejected": @(duplicateRejected),
                @"first_error": negativeError ? [NSString stringWithUTF8String:negativeError] : @"none",
                @"duplicate_error": duplicateError ? [NSString stringWithUTF8String:duplicateError] : @"none",
                @"completion_error": negativeWaitError ? [NSString stringWithUTF8String:negativeWaitError] : @"none"
            },
            @"timeout": @{
                @"zero_nanosecond_timeout_observed": @(timeoutObserved),
                @"recovered_with_five_second_wait": @(timeoutRecovered),
                @"successful_attempt": @(timeoutAttempt), @"attempts": timeoutAttempts
            },
            @"shared_event_transaction": @{
                @"shared_event_class_present": @(sharedEventClass != Nil),
                @"transaction_class_present": @(transactionClass != Nil),
                @"authoritative_constructor_and_signal_wait_contract_present": @NO,
                @"construction_attempted": @NO,
                @"decision": @"not_attempted_without_authoritative_constructor_and_signal_wait_contract"
            },
            @"error": creationError ? [NSString stringWithUTF8String:creationError] : @"none",
            @"claim_boundary": @"One compiled and loaded authentic 2816-to-128 router projection used eight independently mapped IOSurface request banks. Outputs were poisoned before serial and async runs; all active and inactive lanes passed independent CPU oracles after completion. Begin submitted requests without waiting, currentAsyncRequestsInFlight measured actual accepted outstanding requests, completion callbacks supplied ordering, a zero-timeout recovered cleanly, and duplicate begin failed closed. Runtime class presence alone did not authorize constructing shared-event or transaction objects, so no shared-event signal/wait claim is made."
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                         error:nil];
        fwrite(json.bytes, 1, json.length, stdout); fputc('\n', stdout);
        free((void *)creationError); free((void *)negativeError); free((void *)duplicateError);
        free((void *)negativeWaitError);
        ANEFieldfareANEDynamicProjectionDestroy(projection);
        ANEFieldfareTensorIndexClose(&index);
        return valid ? 0 : 1;
    }
}
