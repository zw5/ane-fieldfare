#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int4 — MLX `affine` 4-bit dequant.
//
// Layout (per row of length N):
//   W       : N/2 bytes. Low nibble of byte k = component 2k (unsigned 0..15),
//             high nibble = component 2k+1.
//   scales  : N/64 BF16, one per group of 64.
//   biases  : N/64 BF16, one per group of 64.
//   value   : w[i] = float(nibble[i]) * scale[i/64] + bias[i/64].
//
// Affine factoring for GEMV (sum over a group of 64):
//   sum_k (q_k * s + b) * x_k = s * sum_k(q_k * x_k) + b * sum_k x_k
// so scale and bias each cost one mul + one FMA per group instead of per
// element; the per-element inner loop keeps the scalar path's FMA count.
// ============================================================================

constant constexpr uint kGroupSize = 64;
constant uint FC_INT4_M [[function_constant(20)]];
constant uint FC_INT4_N [[function_constant(21)]];
constant bool FC_INT4_USE_FC [[function_constant(22)]];
constant uint FC_INT4_QKV_MQ [[function_constant(23)]];
constant uint FC_INT4_QKV_MKV [[function_constant(24)]];
constant uint FC_INT4_QKV_N [[function_constant(25)]];
constant bool FC_INT4_QKV_USE_FC [[function_constant(26)]];
constant uint FC_INT4_ROWS_R [[function_constant(27)]];
constant uint FC_INT4_ROWS_K [[function_constant(28)]];
constant bool FC_INT4_ROWS_USE_FC [[function_constant(29)]];
constant uint FC_INT4_ROWS_WPS [[function_constant(32)]];
// Set for the matrix kernel to read weight row ids[o] for output o.
constant bool FC_INT4_ROWS_LISTED [[function_constant(33)]];

static inline uint int4_fc_m(constant uint& M) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_M)) ? FC_INT4_M : M;
}

static inline uint int4_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_USE_FC) &&
            FC_INT4_USE_FC &&
            is_function_constant_defined(FC_INT4_N)) ? FC_INT4_N : N;
}

static inline uint int4_qkv_fc_mq(constant uint& Mq) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MQ)) ? FC_INT4_QKV_MQ : Mq;
}

static inline uint int4_qkv_fc_mkv(constant uint& Mkv) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_MKV)) ? FC_INT4_QKV_MKV : Mkv;
}

static inline uint int4_qkv_fc_n(constant uint& N) {
    return (is_function_constant_defined(FC_INT4_QKV_USE_FC) &&
            FC_INT4_QKV_USE_FC &&
            is_function_constant_defined(FC_INT4_QKV_N)) ? FC_INT4_QKV_N : N;
}

inline uint nib_lo(uint8_t b) { return uint(b & 0x0F); }
inline uint nib_hi(uint8_t b) { return uint(b >> 4); }


kernel void embed_lookup_int4(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/2] nibbles
    device const bfloat*  scales    [[buffer(1)]],   // [V, D/64] BF16
    device const bfloat*  biases    [[buffer(2)]],   // [V, D/64] BF16
    device half*          out       [[buffer(3)]],   // [D] FP16
    constant uint&        token_id  [[buffer(4)]],
    constant uint&        D         [[buffer(5)]],
    constant float&       out_scale [[buffer(6)]],   // pass 1.0 to disable
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint groups_per_row = D / kGroupSize;
    device const uint8_t* row_q = table  + uint(token_id) * (D / 2u);
    device const bfloat*  row_s = scales + uint(token_id) * groups_per_row;
    device const bfloat*  row_b = biases + uint(token_id) * groups_per_row;
    uint8_t byte = row_q[gid >> 1];
    uint    q    = (gid & 1u) ? uint(byte >> 4) : uint(byte & 0xFu);
    float   s    = float(row_s[gid / kGroupSize]);
    float   b    = float(row_b[gid / kGroupSize]);
    out[gid] = half((float(q) * s + b) * out_scale);
}

// y[m] = sum_{n} W[m, n] * x[n]. One-SIMD-per-row variant: 32 threads
// cooperate on a single output row, each handling 2 elements per group of 64
// (one byte → two nibbles). simd_sum reduces across the group; lane 0 writes.
//
// Requires N % 64 == 0 (per group of 64). Validated at the wrapper.
// Each threadgroup handles eight consecutive rows, one SIMD per row. The
// larger work unit gives the scheduler enough independent rows while sharing
// the L1-cached input-vector reads.
static inline void dequant_int4_gemv_simd_body(
    device const uint8_t* W,
    device const bfloat*  scales,
    device const bfloat*  biases,
    device const half*    x,
    device half*          y,
    uint                  M,
    uint                  N,
    uint                  rows_per_tg,
    uint                  tg_idx,
    uint                  sg_idx,
    uint                  lane
) {
    const uint row = tg_idx * rows_per_tg + sg_idx;
    if (row >= M) return;
    const uint n_groups  = N / kGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W      + uint(row) * row_bytes;
    device const bfloat*  s_row = scales + uint(row) * n_groups;
    device const bfloat*  b_row = biases + uint(row) * n_groups;

    float acc = 0.0f;
    // The vectorized row path reads
    // weights a uint (4 bytes = 8 nibbles) and x as half4 in 4-group (128-byte)
    // blocks, with a scalar byte-per-lane remainder. Within a block the 32
    // lanes split 8-per-group, each handling 8 contiguous elements of one
    // 64-element group, so the affine factoring s·Σqx + b·Σx is preserved
    // (simd_sum aggregates; s/b are constant within a group). Aligned: row
    // stride N/2 and weightsOffset are multiples of 4; x is
    // half4-aligned (lane*8 elements). N=2816/4096/8192 → 44/64/128 groups, all
    // exact 4-blocks; the remainder covers any non-multiple-of-4 group count.
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Read the 4-byte weight chunk as two ushorts. The resident weight
        // tensors are 2-byte aligned but NOT 4-byte aligned (BF16 scale/bias
        // regions leave a 2-aligned weightsOffset), so a `uint*` load would be
        // misaligned (undefined → garbage); a `ushort*` load is safe (row stride
        // N/2, weightsOffset, and byte_base are all even) and halves the loads
        // vs byte-by-byte.
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kGroupSize / 2) + lane];
        const float x0 = float(x[g * kGroupSize + lane * 2u]);
        const float x1 = float(x[g * kGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[row] = half(acc);
    }
}

kernel void dequant_int4_gemv_simd(
    device const uint8_t* W      [[buffer(0)]],
    device const bfloat*  scales [[buffer(1)]],
    device const bfloat*  biases [[buffer(2)]],
    device const half*    x      [[buffer(3)]],
    device half*          y      [[buffer(4)]],
    constant uint&        M      [[buffer(5)]],
    constant uint&        N      [[buffer(6)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, MM, NN,
                                rows_per_tg, tg_idx, sg_idx, lane);
}


kernel void dequant_int4_qkv_gemv_simd(
    device const uint8_t* qW      [[buffer(0)]],
    device const bfloat*  qScales [[buffer(1)]],
    device const bfloat*  qBiases [[buffer(2)]],
    device const uint8_t* kW      [[buffer(3)]],
    device const bfloat*  kScales [[buffer(4)]],
    device const bfloat*  kBiases [[buffer(5)]],
    device const uint8_t* vW      [[buffer(6)]],
    device const bfloat*  vScales [[buffer(7)]],
    device const bfloat*  vBiases [[buffer(8)]],
    device const half*    x       [[buffer(9)]],
    device half*          qY      [[buffer(10)]],
    device half*          kY      [[buffer(11)]],
    device half*          vY      [[buffer(12)]],
    constant uint&        Mq      [[buffer(13)]],
    constant uint&        Mkv     [[buffer(14)]],
    constant uint&        N       [[buffer(15)]],
    uint                  tg_idx  [[threadgroup_position_in_grid]],
    uint                  sg_idx  [[simdgroup_index_in_threadgroup]],
    uint                  lane    [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint QQ = int4_qkv_fc_mq(Mq);
    const uint KK = int4_qkv_fc_mkv(Mkv);
    const uint NN = int4_qkv_fc_n(N);
    const uint global_row = tg_idx * rows_per_tg + sg_idx;
    const uint total_rows = QQ + 2u * KK;
    if (global_row >= total_rows) { return; }

    device const uint8_t* W;
    device const bfloat* scales;
    device const bfloat* biases;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; biases = qBiases; y = qY;
        local_row = global_row;
        M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; biases = kBiases; y = kY;
        local_row = global_row - QQ;
        M = KK;
    } else {
        W = vW; scales = vScales; biases = vBiases; y = vY;
        local_row = global_row - QQ - KK;
        M = KK;
    }
    dequant_int4_gemv_simd_body(W, scales, biases, x, y, M, NN,
                                1u, local_row, 0u, lane);
}


// y[r][o] = sum_k W[o, k] * x[r][k] for R <= 16 FP16 or BF16 input rows. The
// decode GEMV's
// lane layout and accumulation order per row, so each weight chunk is read
// once, coalesced, dequantized once and applied to every row, and row r's
// result is bitwise the GEMV's. Small-batch weight-bound projections (verify
// blocks, the shared FFN) then cost about one GEMV instead of one per row.
template <typename T, typename TY>
kernel void dequant_int4_gemv_rows(
    device const uint8_t* W        [[buffer(0)]],
    device const bfloat*  scales   [[buffer(1)]],
    device const bfloat*  biases   [[buffer(2)]],
    device const T*       x        [[buffer(3)]],
    device TY*            y        [[buffer(4)]],
    constant uint&        O        [[buffer(5)]],   // output columns (weight rows)
    constant uint&        K        [[buffer(6)]],   // input width
    constant uint&        R        [[buffer(7)]],   // input rows, 1..16
    constant uint&        xStride  [[buffer(8)]],
    constant uint&        yStride  [[buffer(9)]],
    uint                  tg_idx   [[threadgroup_position_in_grid]],
    uint                  sg_idx   [[simdgroup_index_in_threadgroup]],
    uint                  lane     [[thread_index_in_simdgroup]]
) {
    constexpr uint simds_per_tg = 8;
    // Specialized pipelines fix the row count, the width and the weight rows
    // per SIMD group, so the guarded loops below fold to exactly R x WPS
    // accumulators and the group loop unrolls. Each input chunk is loaded and
    // converted once per SIMD group and applied to all WPS weight rows: past a
    // few input rows the kernel is bound by input traffic, not weights.
    const bool fc = is_function_constant_defined(FC_INT4_ROWS_USE_FC) && FC_INT4_ROWS_USE_FC;
    const uint RR = fc ? FC_INT4_ROWS_R : min(R, 16u);
    const uint KK = fc ? FC_INT4_ROWS_K : K;
    const uint WPS = (fc && is_function_constant_defined(FC_INT4_ROWS_WPS)) ? FC_INT4_ROWS_WPS : 1u;
    const uint row0 = (tg_idx * simds_per_tg + sg_idx) * WPS;
    if (row0 >= O) return;
    const uint n_groups  = KK / kGroupSize;

    float acc[4][16];
    #pragma unroll
    for (uint w = 0; w < 4; ++w) {
        #pragma unroll
        for (uint r = 0; r < 16; ++r) acc[w][r] = 0.0f;
    }

    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint g  = blk * 4u + (lane >> 3);
        const uint elem = byte_base * 2u;
        float q[4][8], s[4], b[4];
        #pragma unroll
        for (uint w = 0; w < 4; ++w) {
            if (w < WPS) {
                const uint row = min(row0 + w, O - 1);
                device const ushort* wp = (device const ushort*)(W + row * (KK / 2) + byte_base);
                const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
                s[w] = float(scales[row * n_groups + g]);
                b[w] = float(biases[row * n_groups + g]);
                #pragma unroll
                for (uint j = 0; j < 8; ++j) q[w][j] = float((w4 >> (4u * j)) & 0xFu);
            }
        }
        #pragma unroll
        for (uint r = 0; r < 16; ++r) {
            if (r < RR) {
                device const T* xr = x + r * xStride + elem;
                const vec<T, 4> xa = *((device const vec<T, 4>*)xr);
                const vec<T, 4> xb = *((device const vec<T, 4>*)(xr + 4u));
                const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
                const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
                const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
                #pragma unroll
                for (uint w = 0; w < 4; ++w) {
                    if (w < WPS) {
                        float dot = 0.0f;
                        dot = fma(q[w][0], e0, dot); dot = fma(q[w][1], e1, dot);
                        dot = fma(q[w][2], e2, dot); dot = fma(q[w][3], e3, dot);
                        dot = fma(q[w][4], e4, dot); dot = fma(q[w][5], e5, dot);
                        dot = fma(q[w][6], e6, dot); dot = fma(q[w][7], e7, dot);
                        acc[w][r] = fma(s[w], dot, acc[w][r]);
                        acc[w][r] = fma(b[w], sum, acc[w][r]);
                    }
                }
            }
        }
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        float q0[4], q1[4], s[4], b[4];
        #pragma unroll
        for (uint w = 0; w < 4; ++w) {
            if (w < WPS) {
                const uint row = min(row0 + w, O - 1);
                const uint8_t byte = W[row * (KK / 2) + g * (kGroupSize / 2) + lane];
                q0[w] = float(uint(byte & 0x0Fu));
                q1[w] = float(uint(byte >> 4));
                s[w] = float(scales[row * n_groups + g]);
                b[w] = float(biases[row * n_groups + g]);
            }
        }
        #pragma unroll
        for (uint r = 0; r < 16; ++r) {
            if (r < RR) {
                device const T* xr = x + r * xStride + g * kGroupSize + lane * 2u;
                const float x0 = float(xr[0]), x1 = float(xr[1]);
                #pragma unroll
                for (uint w = 0; w < 4; ++w) {
                    if (w < WPS) {
                        float dot = fma(q0[w], x0, 0.0f);
                        dot = fma(q1[w], x1, dot);
                        acc[w][r] = fma(s[w], dot, acc[w][r]);
                        acc[w][r] = fma(b[w], x0 + x1, acc[w][r]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (uint w = 0; w < 4; ++w) {
        if (w < WPS && row0 + w < O) {
            #pragma unroll
            for (uint r = 0; r < 16; ++r) {
                if (r < RR) {
                    const float v = simd_sum(acc[w][r]);
                    if (lane == 0) y[r * yStride + row0 + w] = TY(v);
                }
            }
        }
    }
}

template [[host_name("dequant_int4_gemv_rows")]] kernel void dequant_int4_gemv_rows<half, half>(
    device const uint8_t*, device const bfloat*, device const bfloat*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("dequant_int4_gemv_rows_bf16")]] kernel void dequant_int4_gemv_rows<bfloat, bfloat>(
    device const uint8_t*, device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("dequant_int4_gemv_rows_f32")]] kernel void dequant_int4_gemv_rows<half, float>(
    device const uint8_t*, device const bfloat*, device const bfloat*, device const half*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint);

// y[r][o] = sum_k W[o, k] * x[r][k] for 5..16 input rows on the SIMD-group
// matrix units. dequant_int4_gemv_rows converts and multiplies each weight
// once per input row, so past four rows it is ALU-bound (8 rows cost 2x, 16
// rows 4x, measured on M5). Here each SIMD group owns 8 weight rows and
// dequantizes each weight once, as q * s + b in FP32, into an 8x8 fragment;
// one multiply covers 8 input rows. Results match the ALU kernel to FP32
// rounding but not bitwise. K % 128 may be 64: the last group runs as a
// half step.
//
// Fragment layout (Apple SIMD-group 8x8): lane l holds row
// fm = 4*((l>>4)&1) + ((l>>1)&3) and columns fn, fn+1 with
// fn = 4*((l>>3)&1) + 2*(l&1). Per 128-wide step, tile t (0..15) column c is
// k = 16c + t. The 4 lanes of a weight row read 64 contiguous bytes, lane
// j = fn/2 bytes [16j, 16j + 16): its first 8 bytes are column fn, the next 8
// column fn + 1, all in one quant group. Each lane's B values, x[r][16fm + t],
// are 16 contiguous halves per input row, so no x load is wasted; the x loads,
// not the weights, bound this kernel.
template <typename TX>
static inline void int4_mma_load_x16(device const TX* p, bool live, thread float (&v)[16]) {
    typedef vec<TX, 4> TX4;
    for (uint i = 0; i < 4; ++i) {
        const float4 a = live ? float4(*((device const TX4*)(p + 4u * i))) : float4(0.0f);
        v[4 * i] = a.x; v[4 * i + 1] = a.y; v[4 * i + 2] = a.z; v[4 * i + 3] = a.w;
    }
}

template <typename TX, typename TY>
kernel void dequant_int4_gemv_rows_mma(
    device const uint8_t* W        [[buffer(0)]],
    device const bfloat*  scales   [[buffer(1)]],
    device const bfloat*  biases   [[buffer(2)]],
    device const TX*      x        [[buffer(3)]],
    device TY*            y        [[buffer(4)]],
    constant uint&        O        [[buffer(5)]],
    constant uint&        K        [[buffer(6)]],
    constant uint&        R        [[buffer(7)]],
    constant uint&        xStride  [[buffer(8)]],
    constant uint&        yStride  [[buffer(9)]],
    uint                  tg_idx   [[threadgroup_position_in_grid]],
    uint                  sg_idx   [[simdgroup_index_in_threadgroup]],
    uint                  lane     [[thread_index_in_simdgroup]],
    device const uint*    ids      [[buffer(10), function_constant(FC_INT4_ROWS_LISTED)]]
) {
    constexpr uint simds_per_tg = 4;
    const bool fc = is_function_constant_defined(FC_INT4_ROWS_USE_FC) && FC_INT4_ROWS_USE_FC;
    const uint RR = fc ? FC_INT4_ROWS_R : min(R, 16u);
    const uint KK = fc ? FC_INT4_ROWS_K : K;
    const uint o0 = (tg_idx * simds_per_tg + sg_idx) * 8u;
    if (o0 >= O) return;
    const uint n_groups = KK / kGroupSize;
    const uint row_bytes = KK / 2u;

    const uint fm = 4u * ((lane >> 4) & 1u) + ((lane >> 1) & 3u);
    const uint fn = 4u * ((lane >> 3) & 1u) + 2u * (lane & 1u);
    const bool listed = is_function_constant_defined(FC_INT4_ROWS_LISTED) && FC_INT4_ROWS_LISTED;
    const uint row = listed ? ids[min(o0 + fm, O - 1u)] : min(o0 + fm, O - 1u);
    device const uint8_t* w_row = W + row * row_bytes + (fn >> 1) * 16u;
    device const bfloat* s_row = scales + row * n_groups + (fn >> 2);
    device const bfloat* b_row = biases + row * n_groups + (fn >> 2);
    const uint frags = RR > 8u ? 2u : 1u;
    device const TX* xr[4];
    for (uint f = 0; f < 2; ++f) {
        xr[2 * f]     = x + min(8u * f + fn,      RR - 1u) * xStride + 16u * fm;
        xr[2 * f + 1] = x + min(8u * f + fn + 1u, RR - 1u) * xStride + 16u * fm;
    }

    simdgroup_float8x8 acc[2] = { simdgroup_float8x8(0.0f), simdgroup_float8x8(0.0f) };
    const uint steps = KK / 128u;
    // The last pass is a 64-k half step when K % 128 == 64: columns c < 4
    // hold its k; lanes holding columns or B rows c >= 4 contribute zeros.
    const uint passes = steps + (KK % 128u != 0u ? 1u : 0u);
    for (uint step = 0; step < passes; ++step) {
        const bool half_step = step == steps;
        const bool live_a = !half_step || fn < 4u, live_b = !half_step || fm < 4u;
        device const ushort* wp = (device const ushort*)(w_row + step * 64u - (live_a ? 0u : 32u));
        uint wv[4];
        for (uint i = 0; i < 4; ++i) wv[i] = uint(wp[2 * i]) | (uint(wp[2 * i + 1]) << 16);
        const float s = live_a ? float(s_row[step * 2u]) : 0.0f;
        const float b = live_a ? float(b_row[step * 2u]) : 0.0f;
        float v[4][16];
        for (uint i = 0; i < 2 * frags; ++i) int4_mma_load_x16(xr[i] + step * 128u, live_b, v[i]);
        for (uint t = 0; t < 16; ++t) {
            const uint shift = 8u * ((t >> 1) & 3u) + 4u * (t & 1u);
            const uint q0 = (wv[t >> 3] >> shift) & 0x0Fu;
            const uint q1 = (wv[2u + (t >> 3)] >> shift) & 0x0Fu;
            simdgroup_float8x8 a;
            a.thread_elements()[0] = fma(float(q0), s, b);
            a.thread_elements()[1] = fma(float(q1), s, b);
            for (uint f = 0; f < frags; ++f) {
                simdgroup_float8x8 bt;
                bt.thread_elements()[0] = v[2 * f][t];
                bt.thread_elements()[1] = v[2 * f + 1][t];
                simdgroup_multiply_accumulate(acc[f], a, bt, acc[f]);
            }
        }
    }
    if (o0 + fm < O) {
        for (uint f = 0; f < frags; ++f) {
            const uint r0 = 8u * f + fn;
            if (r0 < RR)      y[r0 * yStride + o0 + fm] = TY(acc[f].thread_elements()[0]);
            if (r0 + 1u < RR) y[(r0 + 1u) * yStride + o0 + fm] = TY(acc[f].thread_elements()[1]);
        }
    }
}

template [[host_name("dequant_int4_gemv_rows_mma")]] kernel void dequant_int4_gemv_rows_mma<half, half>(
    device const uint8_t*, device const bfloat*, device const bfloat*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint,
    device const uint*);
template [[host_name("dequant_int4_gemv_rows_mma_f32")]] kernel void dequant_int4_gemv_rows_mma<half, float>(
    device const uint8_t*, device const bfloat*, device const bfloat*, device const half*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint,
    device const uint*);
template [[host_name("dequant_int4_gemv_rows_mma_bf16")]] kernel void dequant_int4_gemv_rows_mma<bfloat, bfloat>(
    device const uint8_t*, device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint,
    device const uint*);
