#pragma once

#include "../ANEFieldfareRuntime/ANEProgram.h"

// Evidence-only mutation primitive used by the retained AOT gates. It is
// deliberately absent from ANEProgram.h: production selection uses the
// ownership-checked shared-surface request catalog instead of mutating a
// mapped request in place.
bool ANEFieldfareANEProgramRebindRequestBankToProcedureEvidence(
    ANEFieldfareANEProgram *program, uint32_t bank, const char *procedureName,
    const char **errorMessage);
