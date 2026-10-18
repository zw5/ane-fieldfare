#include "MemoryBudget.h"

#include <limits.h>
#include <mach/mach.h>

uint64_t ANEFieldfarePhysicalFootprint(void) {
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t result = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    return result == KERN_SUCCESS ? info.phys_footprint : UINT64_MAX;
}

ANEFieldfareMemoryPolicy ANEFieldfareDefaultMemoryPolicy(void) {
    return (ANEFieldfareMemoryPolicy){
        .targetBytes = ANEFieldfareMemoryTargetBytes,
        .hardLimitBytes = ANEFieldfareMemoryHardLimitBytes,
        .requiredHeadroomBytes = ANEFieldfareMinimumHeadroomBytes,
    };
}

static bool sumFits(uint64_t a, uint64_t b, uint64_t c, uint64_t limit) {
    if (a > limit || b > limit - a) return false;
    a += b;
    return c <= limit - a;
}

ANEFieldfareMemoryDecision ANEFieldfareMemoryBudgetDecision(
    ANEFieldfareMemoryPolicy policy,
    uint64_t footprintBytes,
    uint64_t requestedResidentBytes) {
    if (footprintBytes == UINT64_MAX)
        return ANEFieldfareMemoryDecisionMeasurementFailure;
    if (policy.targetBytes == 0 || policy.hardLimitBytes < policy.targetBytes)
        return ANEFieldfareMemoryDecisionReject;
    if (sumFits(footprintBytes, requestedResidentBytes,
                policy.requiredHeadroomBytes, policy.targetBytes))
        return ANEFieldfareMemoryDecisionAdmit;
    if (sumFits(footprintBytes, requestedResidentBytes,
                policy.requiredHeadroomBytes, policy.hardLimitBytes))
        return ANEFieldfareMemoryDecisionAdmitUnderPressure;
    return ANEFieldfareMemoryDecisionReject;
}

bool ANEFieldfareMemoryBudgetAllows(uint64_t footprintBytes,
                                    uint64_t requestedResidentBytes,
                                    uint64_t requiredHeadroomBytes) {
    ANEFieldfareMemoryPolicy policy = ANEFieldfareDefaultMemoryPolicy();
    policy.requiredHeadroomBytes = requiredHeadroomBytes;
    ANEFieldfareMemoryDecision decision = ANEFieldfareMemoryBudgetDecision(
        policy, footprintBytes, requestedResidentBytes);
    return decision == ANEFieldfareMemoryDecisionAdmit ||
           decision == ANEFieldfareMemoryDecisionAdmitUnderPressure;
}
