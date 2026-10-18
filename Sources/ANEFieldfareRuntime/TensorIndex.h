#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int fileDescriptor;
    const uint8_t *bytes;
    uint64_t fileBytes;
    uint64_t headerBytes;
    uint64_t tensorCount;
} ANEFieldfareTensorIndex;

typedef struct {
    const char *name;
    uint32_t nameBytes;
    const uint8_t *weights;
    uint64_t weightBytes;
    const uint8_t *scales;
    uint64_t scaleBytes;
    const uint8_t *biases;
    uint64_t biasBytes;
    uint32_t rows;
    uint32_t columns;
} ANEFieldfareTensor;

bool ANEFieldfareTensorIndexOpen(const char *path, ANEFieldfareTensorIndex *index);
void ANEFieldfareTensorIndexClose(ANEFieldfareTensorIndex *index);
bool ANEFieldfareTensorIndexTensorAt(const ANEFieldfareTensorIndex *index,
                                     uint64_t position,
                                     ANEFieldfareTensor *tensor);
bool ANEFieldfareTensorIndexFind(const ANEFieldfareTensorIndex *index,
                                 const char *name,
                                 ANEFieldfareTensor *tensor);

float ANEFieldfareBF16(const uint8_t *bytes);
bool ANEFieldfareAffineValue(const ANEFieldfareTensor *matrix,
                             uint32_t bits,
                             uint32_t groupSize,
                             uint32_t row,
                             uint32_t column,
                             float *value);
bool ANEFieldfareAffineMatVec(const ANEFieldfareTensor *matrix,
                              uint32_t bits,
                              uint32_t groupSize,
                              const float *input,
                              float *output);
bool ANEFieldfareAffineRow(const ANEFieldfareTensor *matrix,
                           uint32_t bits,
                           uint32_t groupSize,
                           uint32_t row,
                           float *output);

// Fuses row dequantization with a FP64 dot product. This is numerically
// equivalent to ANEFieldfareAffineRow followed by the decoder head's dot loop,
// but avoids materializing and rereading a 2816-float temporary for every
// vocabulary row.
bool ANEFieldfareAffineRowDot(const ANEFieldfareTensor *matrix,
                              uint32_t bits,
                              uint32_t groupSize,
                              uint32_t row,
                              const float *input,
                              double *output);
