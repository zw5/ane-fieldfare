#pragma once

#include "ANEProgram.h"
#include "TensorIndex.h"

typedef struct ANEFieldfareANEDynamicProjection ANEFieldfareANEDynamicProjection;

ANEFieldfareANEDynamicProjection *ANEFieldfareANEDynamicProjectionCreate(
    uint32_t inputWidth,
    uint32_t outputWidth,
    uint32_t requestBanks,
    const char **errorMessage);
bool ANEFieldfareANEDynamicProjectionEvaluate(
    ANEFieldfareANEDynamicProjection *projection,
    uint32_t bank,
    const float *input,
    const ANEFieldfareTensor *weights,
    uint32_t weightBits,
    uint32_t groupSize,
    float *output,
    double *preparationMilliseconds,
    ANEFieldfareANEEvaluationTiming *timing,
    const char **errorMessage);
ANEFieldfareANEProgramInfo ANEFieldfareANEDynamicProjectionGetInfo(
    ANEFieldfareANEDynamicProjection *projection);
ANEFieldfareANEProgram *ANEFieldfareANEDynamicProjectionProgram(
    ANEFieldfareANEDynamicProjection *projection);
void ANEFieldfareANEDynamicProjectionDestroy(ANEFieldfareANEDynamicProjection *projection);
