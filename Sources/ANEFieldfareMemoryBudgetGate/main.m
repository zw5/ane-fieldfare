#import <Foundation/Foundation.h>

#include "../ANEFieldfareRuntime/MemoryBudget.h"

static void emit(NSDictionary *value) {
    NSData *json = [NSJSONSerialization dataWithJSONObject:value
                                                   options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                                                     error:nil];
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
}

int main(void) {
    @autoreleasepool {
        ANEFieldfareMemoryPolicy policy = ANEFieldfareDefaultMemoryPolicy();
        const uint64_t target = policy.targetBytes;
        const uint64_t hardLimit = policy.hardLimitBytes;
        const uint64_t headroom = policy.requiredHeadroomBytes;
        uint64_t footprint = ANEFieldfarePhysicalFootprint();
        NSArray *cases = @[
            @{@"name": @"inside_target", @"footprint": @(target - headroom), @"request": @0,
              @"expected": @(ANEFieldfareMemoryDecisionAdmit)},
            @{@"name": @"target_tolerance", @"footprint": @(target - headroom), @"request": @1,
              @"expected": @(ANEFieldfareMemoryDecisionAdmitUnderPressure)},
            @{@"name": @"hard_limit_boundary", @"footprint": @(hardLimit - headroom), @"request": @0,
              @"expected": @(ANEFieldfareMemoryDecisionAdmitUnderPressure)},
            @{@"name": @"hard_limit_exceeded", @"footprint": @(hardLimit - headroom), @"request": @1,
              @"expected": @(ANEFieldfareMemoryDecisionReject)},
            @{@"name": @"request_overflow", @"footprint": @1, @"request": @(UINT64_MAX),
              @"expected": @(ANEFieldfareMemoryDecisionReject)},
            @{@"name": @"measurement_failure", @"footprint": @(UINT64_MAX), @"request": @0,
              @"expected": @(ANEFieldfareMemoryDecisionMeasurementFailure)},
        ];
        NSMutableArray *results = [NSMutableArray array];
        BOOL unitClean = YES;
        for (NSDictionary *test in cases) {
            ANEFieldfareMemoryDecision actual = ANEFieldfareMemoryBudgetDecision(
                policy, [test[@"footprint"] unsignedLongLongValue],
                [test[@"request"] unsignedLongLongValue]);
            BOOL matches = actual == [test[@"expected"] integerValue];
            unitClean &= matches;
            [results addObject:@{@"name": test[@"name"], @"decision": @(actual), @"matches": @(matches)}];
        }
        ANEFieldfareMemoryDecision liveDecision = ANEFieldfareMemoryBudgetDecision(policy, footprint, 0);
        BOOL liveAllows = liveDecision <= ANEFieldfareMemoryDecisionAdmitUnderPressure;
        BOOL passed = unitClean && liveAllows;
        emit(@{
            @"schema": @"ane-fieldfare.memory-budget.v2",
            @"status": passed ? @"validated" : @"rejected",
            @"target_bytes": @(target),
            @"hard_limit_bytes": @(hardLimit),
            @"required_headroom_bytes": @(headroom),
            @"live_physical_footprint_bytes": @(footprint),
            @"live_decision": @(liveDecision),
            @"live_admission_allowed": @(liveAllows),
            @"unit_cases": results,
            @"claim_boundary": @"Validates a 2 GiB operating target, a 2.25 GiB fail-closed safety limit, overflow-safe admission arithmetic, and live physical-footprint sampling; it does not validate a model run or peak footprint between samples",
        });
        return passed ? 0 : 1;
    }
}
