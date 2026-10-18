#include "Decode.h"

#include <dispatch/dispatch.h>
#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *EmbeddingName = "language_model.model.embed_tokens.weight";
static const char *FinalNormName = "language_model.model.norm.weight";

enum { MaximumHeadWorkers = 64 };

typedef struct {
    const ANEFieldfareTensor *embedding;
    const float *hidden;
    float *logits;
    uint32_t rowsPerTile;
    uint32_t tileCount;
    uint32_t workerCount;
    double bestRaw[MaximumHeadWorkers];
    uint32_t bestToken[MaximumHeadWorkers];
    _Atomic bool valid;
} TiedHeadContext;

static void tiedHeadWorker(void *rawContext, size_t workerValue) {
    TiedHeadContext *context = rawContext;
    uint32_t worker = (uint32_t)workerValue;
    uint32_t firstTile = (uint32_t)((uint64_t)context->tileCount * worker /
                                    context->workerCount);
    uint32_t finalTile = (uint32_t)((uint64_t)context->tileCount * (worker + 1) /
                                    context->workerCount);
    double bestRaw = -DBL_MAX;
    uint32_t bestToken = UINT32_MAX;
    for (uint32_t tileIndex = firstTile; tileIndex < finalTile; ++tileIndex) {
        uint32_t first = tileIndex * context->rowsPerTile;
        uint32_t remaining = context->embedding->rows - first;
        uint32_t count = context->rowsPerTile < remaining ? context->rowsPerTile : remaining;
        for (uint32_t candidate = first; candidate < first + count; ++candidate) {
            double sum = 0.0;
            if (!ANEFieldfareAffineRowDot(context->embedding, 4, 64, candidate,
                                          context->hidden, &sum)) {
                atomic_store_explicit(&context->valid, false, memory_order_relaxed);
                return;
            }
            if (context->logits)
                context->logits[candidate] = 30.0f * tanhf((float)sum / 30.0f);
            if (sum > bestRaw) {
                bestRaw = sum;
                bestToken = candidate;
            }
        }
    }
    context->bestRaw[worker] = bestRaw;
    context->bestToken[worker] = bestToken;
}

static float bf16Rounded(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool rawBF16(const ANEFieldfareTensor *tensor, uint32_t count, float *output) {
    if (!tensor || !output || tensor->weightBytes != (uint64_t)count * 2 ||
        tensor->scaleBytes || tensor->biasBytes) return false;
    for (uint32_t i = 0; i < count; ++i) output[i] = ANEFieldfareBF16(tensor->weights + 2 * i);
    return true;
}

bool ANEFieldfareScaledEmbedding(const ANEFieldfareTensorIndex *index,
                                 uint32_t token,
                                 float *output) {
    ANEFieldfareTensor embedding = {0};
    if (!index || !output ||
        !ANEFieldfareTensorIndexFind(index, EmbeddingName, &embedding) ||
        embedding.rows != ANEFieldfareGemmaVocabularySize ||
        embedding.columns != ANEFieldfareGemmaHiddenSize || token >= embedding.rows ||
        !ANEFieldfareAffineRow(&embedding, 4, 64, token, output)) return false;
    /* Gemma 4 materializes sqrt(hidden_size) as a buffer and casts it to the
     * BF16 embedding dtype before multiplication. For width 2816 this is
     * exactly 53.0, rather than the FP32 value 53.065996.... */
    const float scale = bf16Rounded(sqrtf((float)embedding.columns));
    for (uint32_t i = 0; i < embedding.columns; ++i) output[i] *= scale;
    return true;
}

bool ANEFieldfareFinalRMSNorm(const ANEFieldfareTensorIndex *index,
                             float epsilon,
                             float *hidden) {
    ANEFieldfareTensor norm = {0};
    if (!index || !hidden || !(epsilon > 0.0f) || !isfinite(epsilon) ||
        !ANEFieldfareTensorIndexFind(index, FinalNormName, &norm) ||
        norm.rows != ANEFieldfareGemmaHiddenSize || norm.columns != 0) return false;
    float weights[ANEFieldfareGemmaHiddenSize];
    if (!rawBF16(&norm, ANEFieldfareGemmaHiddenSize, weights)) return false;
    double squareSum = 0.0;
    for (uint32_t i = 0; i < ANEFieldfareGemmaHiddenSize; ++i)
        squareSum += (double)hidden[i] * hidden[i];
    float inverseRMS = 1.0f / sqrtf((float)(squareSum / ANEFieldfareGemmaHiddenSize) + epsilon);
    if (!isfinite(inverseRMS)) return false;
    for (uint32_t i = 0; i < ANEFieldfareGemmaHiddenSize; ++i)
        hidden[i] *= inverseRMS * weights[i];
    return true;
}

bool ANEFieldfareTiedGreedy(const ANEFieldfareTensorIndex *index,
                           const float *hidden,
                           uint32_t rowsPerTile,
                           float *logits,
                           uint32_t *token,
                           float *winningLogit) {
    ANEFieldfareTensor embedding = {0};
    if (!index || !hidden || !token || !winningLogit || !rowsPerTile ||
        !ANEFieldfareTensorIndexFind(index, EmbeddingName, &embedding) ||
        embedding.rows != ANEFieldfareGemmaVocabularySize ||
        embedding.columns != ANEFieldfareGemmaHiddenSize) return false;
    if (rowsPerTile > embedding.rows) rowsPerTile = embedding.rows;
    uint32_t tileCount = (embedding.rows + rowsPerTile - 1) / rowsPerTile;
    long activeCPUs = sysconf(_SC_NPROCESSORS_ONLN);
    uint32_t workerCount = activeCPUs > 0 ? (uint32_t)activeCPUs : 1;
    if (workerCount > MaximumHeadWorkers) workerCount = MaximumHeadWorkers;
    if (workerCount > tileCount) workerCount = tileCount;
    TiedHeadContext context = {
        .embedding = &embedding,
        .hidden = hidden,
        .logits = logits,
        .rowsPerTile = rowsPerTile,
        .tileCount = tileCount,
        .workerCount = workerCount,
    };
    atomic_init(&context.valid, true);
    dispatch_apply_f(workerCount,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &context, tiedHeadWorker);
    if (!atomic_load_explicit(&context.valid, memory_order_relaxed)) return false;
    uint32_t bestToken = 0;
    double bestRaw = -DBL_MAX;
    for (uint32_t worker = 0; worker < workerCount; ++worker) {
        /* Workers own ascending, non-overlapping token ranges. Strict greater-
         * than therefore preserves the original lowest-ID tie break. */
        if (context.bestRaw[worker] > bestRaw) {
            bestRaw = context.bestRaw[worker];
            bestToken = context.bestToken[worker];
        }
    }
    *token = bestToken;
    *winningLogit = 30.0f * tanhf((float)bestRaw / 30.0f);
    return isfinite(bestRaw) && isfinite(*winningLogit);
}

bool ANEFieldfareDecodeGreedy(const ANEFieldfareTensorIndex *index,
                             const float *hidden,
                             uint32_t rowsPerTile,
                             uint32_t *token,
                             float *winningLogit) {
    if (!hidden) return false;
    float normalized[ANEFieldfareGemmaHiddenSize];
    memcpy(normalized, hidden, sizeof(normalized));
    return ANEFieldfareFinalRMSNorm(index, 1e-6f, normalized) &&
        ANEFieldfareTiedGreedy(index, normalized, rowsPerTile, NULL, token, winningLogit);
}
