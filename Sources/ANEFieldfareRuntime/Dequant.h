#pragma once
#include <stddef.h>
#include <stdint.h>

// Each group is 64 low-nibble-first unsigned INT4 codes (32 bytes),
// one little-endian BF16 scale, and one little-endian BF16 affine bias.
// The caller provides groups * 64 output elements and valid input spans.
void ANEFieldfareDequantGroupsFP16(const uint8_t *weights,
                                  const uint8_t *scales,
                                  const uint8_t *biases,
                                  size_t groups,
                                  __fp16 *output);
