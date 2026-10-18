// New implementation of the surviving resident-projection call contract.
// This is not a recovered copy of the missing historical Dequant.c.
#include "Dequant.h"
#include <string.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static float bf16(const uint8_t *bytes) {
    uint32_t word = ((uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8)) << 16;
    float value;
    memcpy(&value, &word, sizeof(value));
    return value;
}

void ANEFieldfareDequantGroupsFP16(const uint8_t *weights,
                                  const uint8_t *scales,
                                  const uint8_t *biases,
                                  size_t groups,
                                  __fp16 *output) {
    for (size_t group = 0; group < groups; ++group) {
        float scale = bf16(scales + group * 2);
        float bias = bf16(biases + group * 2);
#if defined(__aarch64__)
        const float32x4_t s = vdupq_n_f32(scale), b = vdupq_n_f32(bias);
        for (size_t chunk = 0; chunk < 4; ++chunk) {
            uint8x8_t packed = vld1_u8(weights + group * 32 + chunk * 8);
            uint8x8x2_t codes = vzip_u8(vand_u8(packed, vdup_n_u8(15)),
                                       vshr_n_u8(packed, 4));
            for (size_t half = 0; half < 2; ++half) {
                uint16x8_t q = vmovl_u8(codes.val[half]);
                float32x4_t lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(q)));
                float32x4_t hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(q)));
                vst1_f16(output + group * 64 + chunk * 16 + half * 8,
                         vcvt_f16_f32(vaddq_f32(vmulq_f32(lo, s), b)));
                vst1_f16(output + group * 64 + chunk * 16 + half * 8 + 4,
                         vcvt_f16_f32(vaddq_f32(vmulq_f32(hi, s), b)));
            }
        }
#else
        for (size_t element = 0; element < 64; ++element) {
            uint8_t packed = weights[group * 32 + element / 2];
            unsigned q = element % 2 ? packed >> 4 : packed & 15;
            output[group * 64 + element] = (__fp16)(q * scale + bias);
        }
#endif
    }
}
