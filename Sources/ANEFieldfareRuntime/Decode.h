#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "TensorIndex.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ANEFieldfareGemmaHiddenSize = 2816,
    ANEFieldfareGemmaVocabularySize = 262144,
};

/* Dequantizes one tied embedding row and applies Gemma's BF16-rounded
 * sqrt(hidden_size) input scaling. `output` must hold embedding.columns
 * floats. */
bool ANEFieldfareScaledEmbedding(const ANEFieldfareTensorIndex *index,
                                 uint32_t token,
                                 float *output);

/* Applies the checkpoint's final Gemma RMSNorm in place. */
bool ANEFieldfareFinalRMSNorm(const ANEFieldfareTensorIndex *index,
                             float epsilon,
                             float *hidden);

/* Streams the tied embedding matrix one bounded row tile at a time. Logits are
 * softcapped as 30*tanh(logit/30); ties resolve to the lowest token ID. When
 * logits is NULL only the argmax and winning softcapped logit are retained. */
bool ANEFieldfareTiedGreedy(const ANEFieldfareTensorIndex *index,
                           const float *normalizedHidden,
                           uint32_t rowsPerTile,
                           float *logits,
                           uint32_t *token,
                           float *winningLogit);

/* Convenience endpoint: copies hidden, final-normalizes it, then selects from
 * the tied softcapped head without allocating a vocabulary-sized buffer. */
bool ANEFieldfareDecodeGreedy(const ANEFieldfareTensorIndex *index,
                             const float *hidden,
                             uint32_t rowsPerTile,
                             uint32_t *token,
                             float *winningLogit);

#ifdef __cplusplus
}
#endif
