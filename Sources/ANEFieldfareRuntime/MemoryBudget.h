#pragma once

#include <stdbool.h>
#include <stdint.h>

#define ANEFieldfareMemoryTargetBytes (2ULL * 1024ULL * 1024ULL * 1024ULL)
#define ANEFieldfareMemoryHardLimitBytes (2304ULL * 1024ULL * 1024ULL)
#define ANEFieldfareMinimumHeadroomBytes (64ULL * 1024ULL * 1024ULL)

typedef struct {
    uint64_t targetBytes;
    uint64_t hardLimitBytes;
    uint64_t requiredHeadroomBytes;
} ANEFieldfareMemoryPolicy;

typedef enum {
    ANEFieldfareMemoryDecisionAdmit = 0,
    ANEFieldfareMemoryDecisionAdmitUnderPressure = 1,
    ANEFieldfareMemoryDecisionReject = 2,
    ANEFieldfareMemoryDecisionMeasurementFailure = 3,
} ANEFieldfareMemoryDecision;

uint64_t ANEFieldfarePhysicalFootprint(void);
ANEFieldfareMemoryPolicy ANEFieldfareDefaultMemoryPolicy(void);
ANEFieldfareMemoryDecision ANEFieldfareMemoryBudgetDecision(
    ANEFieldfareMemoryPolicy policy,
    uint64_t footprintBytes,
    uint64_t requestedResidentBytes);
bool ANEFieldfareMemoryBudgetAllows(uint64_t footprintBytes,
                                    uint64_t requestedResidentBytes,
                                    uint64_t requiredHeadroomBytes);
