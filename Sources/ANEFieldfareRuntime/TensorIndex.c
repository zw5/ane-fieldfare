#include "TensorIndex.h"

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    HeaderPrefixBytes = 24,
    TensorRecordBytes = 72,
};

typedef struct {
    const ANEFieldfareTensor *matrix;
    const float *input;
    float *output;
    uint32_t bits;
    uint32_t groupSize;
    uint32_t groupsPerRow;
} AffineRowContext;

static void affineMatVecRow(void *rawContext, size_t rowValue) {
    const AffineRowContext *context = rawContext;
    const ANEFieldfareTensor *matrix = context->matrix;
    uint32_t row = (uint32_t)rowValue;
    float sum = 0;
    uint64_t rowLinear = (uint64_t)row * matrix->columns;
    uint64_t rowGroup = (uint64_t)row * context->groupsPerRow;
    for (uint32_t group = 0; group < context->groupsPerRow; ++group) {
        float scale = ANEFieldfareBF16(matrix->scales + (rowGroup + group) * 2);
        float bias = ANEFieldfareBF16(matrix->biases + (rowGroup + group) * 2);
        uint32_t begin = group * context->groupSize;
        uint32_t end = begin + context->groupSize;
        for (uint32_t column = begin; column < end; ++column) {
            uint64_t linear = rowLinear + column;
            uint32_t q = context->bits == 8 ? matrix->weights[linear]
                : ((linear & 1) ? matrix->weights[linear >> 1] >> 4
                                : matrix->weights[linear >> 1] & 0x0f);
            float weight = q * scale + bias;
            sum += context->input[column] * weight;
        }
    }
    context->output[row] = sum;
}

static uint32_t read32(const uint8_t *bytes) {
    uint32_t value = 0;
    memcpy(&value, bytes, sizeof(value));
    return value;
}

static uint64_t read64(const uint8_t *bytes) {
    uint64_t value = 0;
    memcpy(&value, bytes, sizeof(value));
    return value;
}

static bool rangeFits(uint64_t offset, uint64_t length, uint64_t limit) {
    return offset <= limit && length <= limit - offset;
}

bool ANEFieldfareTensorIndexOpen(const char *path, ANEFieldfareTensorIndex *index) {
    if (!path || !index) return false;
    *index = (ANEFieldfareTensorIndex){.fileDescriptor = -1};
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return false;
    struct stat status = {0};
    if (fstat(descriptor, &status) != 0 || status.st_size < HeaderPrefixBytes) {
        close(descriptor);
        return false;
    }
    uint64_t fileBytes = (uint64_t)status.st_size;
    const uint8_t *bytes = mmap(NULL, (size_t)fileBytes, PROT_READ, MAP_PRIVATE, descriptor, 0);
    if (bytes == MAP_FAILED) {
        close(descriptor);
        return false;
    }
    uint64_t headerBytes = read64(bytes);
    uint64_t declaredFileBytes = read64(bytes + 8);
    uint64_t tensorCount = read64(bytes + 16);
    if (headerBytes > fileBytes || declaredFileBytes != fileBytes - headerBytes ||
        tensorCount > (headerBytes - HeaderPrefixBytes) / TensorRecordBytes) {
        munmap((void *)bytes, (size_t)fileBytes);
        close(descriptor);
        return false;
    }
    *index = (ANEFieldfareTensorIndex){descriptor, bytes, fileBytes, headerBytes, tensorCount};
    return true;
}

void ANEFieldfareTensorIndexClose(ANEFieldfareTensorIndex *index) {
    if (!index) return;
    if (index->bytes && index->fileBytes) munmap((void *)index->bytes, (size_t)index->fileBytes);
    if (index->fileDescriptor >= 0) close(index->fileDescriptor);
    *index = (ANEFieldfareTensorIndex){.fileDescriptor = -1};
}

bool ANEFieldfareTensorIndexTensorAt(const ANEFieldfareTensorIndex *index,
                                     uint64_t position,
                                     ANEFieldfareTensor *tensor) {
    if (!index || !tensor || position >= index->tensorCount) return false;
    const uint8_t *record = index->bytes + HeaderPrefixBytes + position * TensorRecordBytes;
    uint64_t nameOffset = read32(record);
    // The low 16 bits carry the byte length. The upper bits describe the raw
    // payload kind for unquantized tensors and are not part of the name.
    uint64_t nameBytes = read32(record + 4) & 0xffffu;
    uint64_t weightOffset = read64(record + 8);
    uint64_t weightBytes = read64(record + 16);
    uint64_t scaleOffset = read64(record + 40);
    uint64_t scaleBytes = read64(record + 48);
    uint64_t biasOffset = read64(record + 56);
    uint64_t biasBytes = read64(record + 64);
    if (!rangeFits(nameOffset, nameBytes, index->headerBytes) ||
        !rangeFits(weightOffset, weightBytes, index->fileBytes) ||
        (scaleBytes && !rangeFits(scaleOffset, scaleBytes, index->fileBytes)) ||
        (biasBytes && !rangeFits(biasOffset, biasBytes, index->fileBytes))) return false;
    *tensor = (ANEFieldfareTensor){
        .name = (const char *)index->bytes + nameOffset,
        .nameBytes = (uint32_t)nameBytes,
        .weights = index->bytes + weightOffset,
        .weightBytes = weightBytes,
        .scales = scaleBytes ? index->bytes + scaleOffset : NULL,
        .scaleBytes = scaleBytes,
        .biases = biasBytes ? index->bytes + biasOffset : NULL,
        .biasBytes = biasBytes,
        .rows = read32(record + 24),
        .columns = read32(record + 28),
    };
    return true;
}

bool ANEFieldfareTensorIndexFind(const ANEFieldfareTensorIndex *index,
                                 const char *name,
                                 ANEFieldfareTensor *tensor) {
    if (!index || !name || !tensor) return false;
    size_t length = strlen(name);
    for (uint64_t position = 0; position < index->tensorCount; ++position) {
        ANEFieldfareTensor candidate = {0};
        if (!ANEFieldfareTensorIndexTensorAt(index, position, &candidate)) return false;
        if (candidate.nameBytes == length && memcmp(candidate.name, name, length) == 0) {
            *tensor = candidate;
            return true;
        }
    }
    return false;
}

float ANEFieldfareBF16(const uint8_t *bytes) {
    uint16_t word = 0;
    memcpy(&word, bytes, sizeof(word));
    uint32_t bits = (uint32_t)word << 16;
    float value = 0;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

bool ANEFieldfareAffineValue(const ANEFieldfareTensor *matrix,
                             uint32_t bits,
                             uint32_t groupSize,
                             uint32_t row,
                             uint32_t column,
                             float *value) {
    if (!matrix || !value || row >= matrix->rows || column >= matrix->columns ||
        (bits != 4 && bits != 8) || !groupSize || matrix->columns % groupSize) return false;
    uint64_t elements = (uint64_t)matrix->rows * matrix->columns;
    uint64_t groups = elements / groupSize;
    uint64_t expectedWeights = bits == 4 ? (elements + 1) / 2 : elements;
    if (matrix->weightBytes != expectedWeights || matrix->scaleBytes != groups * 2 ||
        matrix->biasBytes != groups * 2) return false;
    uint64_t linear = (uint64_t)row * matrix->columns + column;
    uint32_t q = bits == 8 ? matrix->weights[linear]
        : ((linear & 1) ? matrix->weights[linear >> 1] >> 4
                        : matrix->weights[linear >> 1] & 0x0f);
    uint64_t group = (uint64_t)row * (matrix->columns / groupSize) + column / groupSize;
    *value = q * ANEFieldfareBF16(matrix->scales + group * 2) +
             ANEFieldfareBF16(matrix->biases + group * 2);
    return true;
}

bool ANEFieldfareAffineMatVec(const ANEFieldfareTensor *matrix,
                              uint32_t bits,
                              uint32_t groupSize,
                              const float *input,
                              float *output) {
    if (!matrix || !input || !output || !matrix->rows || !matrix->columns ||
        (bits != 4 && bits != 8) || !groupSize || matrix->columns % groupSize) return false;
    uint64_t elements = (uint64_t)matrix->rows * matrix->columns;
    uint64_t groups = elements / groupSize;
    uint64_t expectedWeights = bits == 4 ? (elements + 1) / 2 : elements;
    if (matrix->weightBytes != expectedWeights || matrix->scaleBytes != groups * 2 ||
        matrix->biasBytes != groups * 2) return false;
    AffineRowContext context = {
        .matrix = matrix,
        .input = input,
        .output = output,
        .bits = bits,
        .groupSize = groupSize,
        .groupsPerRow = matrix->columns / groupSize,
    };
    if (matrix->rows >= 256) {
        dispatch_apply_f(matrix->rows,
                         dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                         &context, affineMatVecRow);
    } else {
        for (uint32_t row = 0; row < matrix->rows; ++row)
            affineMatVecRow(&context, row);
    }
    return true;
}

bool ANEFieldfareAffineRow(const ANEFieldfareTensor *matrix,
                           uint32_t bits,
                           uint32_t groupSize,
                           uint32_t row,
                           float *output) {
    if (!matrix || !output || row >= matrix->rows || !matrix->columns ||
        (bits != 4 && bits != 8) || !groupSize || matrix->columns % groupSize) return false;
    uint64_t elements = (uint64_t)matrix->rows * matrix->columns;
    uint64_t groups = elements / groupSize;
    uint64_t expectedWeights = bits == 4 ? (elements + 1) / 2 : elements;
    if (matrix->weightBytes != expectedWeights || matrix->scaleBytes != groups * 2 ||
        matrix->biasBytes != groups * 2) return false;
    uint32_t groupsPerRow = matrix->columns / groupSize;
    uint64_t rowLinear = (uint64_t)row * matrix->columns;
    uint64_t rowGroup = (uint64_t)row * groupsPerRow;
    for (uint32_t group = 0; group < groupsPerRow; ++group) {
        float scale = ANEFieldfareBF16(matrix->scales + (rowGroup + group) * 2);
        float bias = ANEFieldfareBF16(matrix->biases + (rowGroup + group) * 2);
        uint32_t begin = group * groupSize;
        uint32_t end = begin + groupSize;
        for (uint32_t column = begin; column < end; ++column) {
            uint64_t linear = rowLinear + column;
            uint32_t q = bits == 8 ? matrix->weights[linear]
                : ((linear & 1) ? matrix->weights[linear >> 1] >> 4
                                : matrix->weights[linear >> 1] & 0x0f);
            output[column] = q * scale + bias;
        }
    }
    return true;
}

bool ANEFieldfareAffineRowDot(const ANEFieldfareTensor *matrix,
                              uint32_t bits,
                              uint32_t groupSize,
                              uint32_t row,
                              const float *input,
                              double *output) {
    if (!matrix || !input || !output || row >= matrix->rows || !matrix->columns ||
        (bits != 4 && bits != 8) || !groupSize || matrix->columns % groupSize) return false;
    uint64_t elements = (uint64_t)matrix->rows * matrix->columns;
    uint64_t groups = elements / groupSize;
    uint64_t expectedWeights = bits == 4 ? (elements + 1) / 2 : elements;
    if (matrix->weightBytes != expectedWeights || matrix->scaleBytes != groups * 2 ||
        matrix->biasBytes != groups * 2) return false;
    uint32_t groupsPerRow = matrix->columns / groupSize;
    uint64_t rowLinear = (uint64_t)row * matrix->columns;
    uint64_t rowGroup = (uint64_t)row * groupsPerRow;
    double sum = 0.0;
    for (uint32_t group = 0; group < groupsPerRow; ++group) {
        float scale = ANEFieldfareBF16(matrix->scales + (rowGroup + group) * 2);
        float bias = ANEFieldfareBF16(matrix->biases + (rowGroup + group) * 2);
        uint32_t begin = group * groupSize;
        uint32_t end = begin + groupSize;
        for (uint32_t column = begin; column < end; ++column) {
            uint64_t linear = rowLinear + column;
            uint32_t q = bits == 8 ? matrix->weights[linear]
                : ((linear & 1) ? matrix->weights[linear >> 1] >> 4
                                : matrix->weights[linear >> 1] & 0x0f);
            float weight = q * scale + bias;
            sum += (double)input[column] * weight;
        }
    }
    *output = sum;
    return true;
}
