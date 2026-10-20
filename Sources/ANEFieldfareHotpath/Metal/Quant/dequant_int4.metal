#include <metal_stdlib>
using namespace metal;

// ============================================================================
// dequant_int4 — Q4_0 4-bit dequant.
//
// Layout (per row of length N):
//   W       : N/2 bytes. Low nibble of byte k = component 2k (unsigned 0..15),
//             high nibble = component 2k+1.
//   scales  : N/32 FP16, one per group of 32.
//   value   : w[i] = scale[i/32] * (float(nibble[i]) - 8).
//
// Written as an affine map with bias b = -8 s, the GEMV factors per group:
//   sum_k (q_k * s + b) * x_k = s * sum_k(q_k * x_k) + b * sum_k x_k
// so scale and bias each cost one FMA per group instead of per element.
// Every 8 consecutive components a lane handles lie in one group.
// ============================================================================

constant constexpr uint kGroupSize = 32;
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

kernel void embed_lookup_int4(
    device const uint8_t* table     [[buffer(0)]],   // [V, D/2] nibbles
    device const half*    scales    [[buffer(1)]],   // [V, D/32] FP16
    device half*          out       [[buffer(2)]],   // [D] FP16
    constant uint&        token_id  [[buffer(3)]],
    constant uint&        D         [[buffer(4)]],
    constant float&       out_scale [[buffer(5)]],   // pass 1.0 to disable
    uint                  gid       [[thread_position_in_grid]]
) {
    if (gid >= D) return;
    const uint groups_per_row = D / kGroupSize;
    device const uint8_t* row_q = table  + uint(token_id) * (D / 2u);
    device const half*    row_s = scales + uint(token_id) * groups_per_row;
    uint8_t byte = row_q[gid >> 1];
    uint    q    = (gid & 1u) ? uint(byte >> 4) : uint(byte & 0xFu);
    float   s    = float(row_s[gid / kGroupSize]);
    out[gid] = half((float(q) - 8.0f) * s * out_scale);
}

// y[m] = sum_{n} W[m, n] * x[n]: one SIMD group per output row, reduced with
// simd_sum, lane 0 writes. Requires N % 64 == 0 (validated by the wrapper).
// Eight consecutive rows per threadgroup share the L1-cached input reads.
static inline void dequant_int4_gemv_simd_body(
    device const uint8_t* W,
    device const half*    scales,
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
    device const half*    s_row = scales + uint(row) * n_groups;

    float acc = 0.0f;
    // Weights are read 4 bytes (8 nibbles) and x as half4 in 256-element
    // (128-byte) blocks, with a byte-per-lane remainder for a width that is
    // not a multiple of 256. Within a block the 32 lanes split 4 per group,
    // each handling 8 contiguous elements of one group, so the affine
    // factoring s·Σqx + b·Σx holds (s/b are constant within a group).
    const uint full_blocks = N / 256u;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Two ushort loads keep the kernel correct for any 2-aligned
        // weightsOffset, which is all the wrapper requires.
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 8u + (lane >> 2);
        const float s = float(s_row[g]);
        const float b = -8.0f * s;
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
    // Remainder in 64-element steps, one byte per lane: lanes 0-15 hold the
    // step's first group, lanes 16-31 its second.
    for (uint c = full_blocks * 4u; c < N / 64u; ++c) {
        const float s = float(s_row[c * 2u + (lane >> 4)]);
        const float b = -8.0f * s;
        const uint8_t byte = W_row[c * 32u + lane];
        const float x0 = float(x[c * 64u + lane * 2u]);
        const float x1 = float(x[c * 64u + lane * 2u + 1u]);
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
    device const half*    scales [[buffer(1)]],
    device const half*    x      [[buffer(2)]],
    device half*          y      [[buffer(3)]],
    constant uint&        M      [[buffer(4)]],
    constant uint&        N      [[buffer(5)]],
    uint                  tg_idx [[threadgroup_position_in_grid]],
    uint                  sg_idx [[simdgroup_index_in_threadgroup]],
    uint                  lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    const uint MM = int4_fc_m(M);
    const uint NN = int4_fc_n(N);
    dequant_int4_gemv_simd_body(W, scales, x, y, MM, NN,
                                rows_per_tg, tg_idx, sg_idx, lane);
}

kernel void dequant_int4_qkv_gemv_simd(
    device const uint8_t* qW      [[buffer(0)]],
    device const half*    qScales [[buffer(1)]],
    device const uint8_t* kW      [[buffer(2)]],
    device const half*    kScales [[buffer(3)]],
    device const uint8_t* vW      [[buffer(4)]],
    device const half*    vScales [[buffer(5)]],
    device const half*    x       [[buffer(6)]],
    device half*          qY      [[buffer(7)]],
    device half*          kY      [[buffer(8)]],
    device half*          vY      [[buffer(9)]],
    constant uint&        Mq      [[buffer(10)]],
    constant uint&        Mkv     [[buffer(11)]],
    constant uint&        N       [[buffer(12)]],
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
    device const half* scales;
    device half* y;
    uint local_row;
    uint M;
    if (global_row < QQ) {
        W = qW; scales = qScales; y = qY;
        local_row = global_row;
        M = QQ;
    } else if (global_row < QQ + KK) {
        W = kW; scales = kScales; y = kY;
        local_row = global_row - QQ;
        M = KK;
    } else {
        W = vW; scales = vScales; y = vY;
        local_row = global_row - QQ - KK;
        M = KK;
    }
    dequant_int4_gemv_simd_body(W, scales, x, y, M, NN,
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
    device const half*    scales   [[buffer(1)]],
    device const T*       x        [[buffer(2)]],
    device TY*            y        [[buffer(3)]],
    constant uint&        O        [[buffer(4)]],   // output columns (weight rows)
    constant uint&        K        [[buffer(5)]],   // input width
    constant uint&        R        [[buffer(6)]],   // input rows, 1..16
    constant uint&        xStride  [[buffer(7)]],
    constant uint&        yStride  [[buffer(8)]],
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

    const uint full_blocks = KK / 256u;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint g  = blk * 8u + (lane >> 2);
        const uint elem = byte_base * 2u;
        float q[4][8], s[4], b[4];
        #pragma unroll
        for (uint w = 0; w < 4; ++w) {
            if (w < WPS) {
                const uint row = min(row0 + w, O - 1);
                device const ushort* wp = (device const ushort*)(W + row * (KK / 2) + byte_base);
                const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
                s[w] = float(scales[row * n_groups + g]);
                b[w] = -8.0f * s[w];
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
    // Remainder in 64-element steps, one byte per lane (see the GEMV).
    for (uint c = full_blocks * 4u; c < KK / 64u; ++c) {
        float q0[4], q1[4], s[4], b[4];
        #pragma unroll
        for (uint w = 0; w < 4; ++w) {
            if (w < WPS) {
                const uint row = min(row0 + w, O - 1);
                const uint8_t byte = W[row * (KK / 2) + c * 32u + lane];
                q0[w] = float(uint(byte & 0x0Fu));
                q1[w] = float(uint(byte >> 4));
                s[w] = float(scales[row * n_groups + c * 2u + (lane >> 4)]);
                b[w] = -8.0f * s[w];
            }
        }
        #pragma unroll
        for (uint r = 0; r < 16; ++r) {
            if (r < RR) {
                device const T* xr = x + r * xStride + c * 64u + lane * 2u;
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
    device const uint8_t*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("dequant_int4_gemv_rows_bf16")]] kernel void dequant_int4_gemv_rows<bfloat, bfloat>(
    device const uint8_t*, device const half*, device const bfloat*, device bfloat*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("dequant_int4_gemv_rows_f32")]] kernel void dequant_int4_gemv_rows<half, float>(
    device const uint8_t*, device const half*, device const half*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint);

// y[r][o] = sum_k W[o, k] * x[r][k] for 5..16 input rows on the SIMD-group
// matrix units. dequant_int4_gemv_rows converts and multiplies each weight
// once per input row, so past four rows it is ALU-bound (8 rows cost 2x, 16
// rows 4x, measured on M5). Here each SIMD group owns 8 weight rows and
// dequantizes each weight once, as q * s - 8 s in FP32, into an 8x8 fragment;
// one multiply covers 8 input rows. Results match the ALU kernel to FP32
// rounding but not bitwise. K % 128 may be 64: the last group runs as a
// half step.
//
// Fragment layout (Apple SIMD-group 8x8): lane l holds row
// fm = 4*((l>>4)&1) + ((l>>1)&3) and columns fn, fn+1 with
// fn = 4*((l>>3)&1) + 2*(l&1). Per 128-wide step, tile t (0..15) column c is
// k = 16c + t. The 4 lanes of a weight row read 64 contiguous bytes, lane
// j = fn/2 bytes [16j, 16j + 16): its first 8 bytes are column fn, the next 8
// column fn + 1, all in quant group 4 * step + j. Each lane's B values, x[r][16fm + t],
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
    device const half*    scales   [[buffer(1)]],
    device const TX*      x        [[buffer(2)]],
    device TY*            y        [[buffer(3)]],
    constant uint&        O        [[buffer(4)]],
    constant uint&        K        [[buffer(5)]],
    constant uint&        R        [[buffer(6)]],
    constant uint&        xStride  [[buffer(7)]],
    constant uint&        yStride  [[buffer(8)]],
    uint                  tg_idx   [[threadgroup_position_in_grid]],
    uint                  sg_idx   [[simdgroup_index_in_threadgroup]],
    uint                  lane     [[thread_index_in_simdgroup]],
    device const uint*    ids      [[buffer(9), function_constant(FC_INT4_ROWS_LISTED)]]
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
    device const half* s_row = scales + row * n_groups + (fn >> 1);
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
        const float s = live_a ? float(s_row[step * 4u]) : 0.0f;
        const float b = -8.0f * s;
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
    device const uint8_t*, device const half*, device const half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint,
    device const uint*);
template [[host_name("dequant_int4_gemv_rows_mma_f32")]] kernel void dequant_int4_gemv_rows_mma<half, float>(
    device const uint8_t*, device const half*, device const half*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint,
    device const uint*);
template [[host_name("dequant_int4_gemv_rows_mma_bf16")]] kernel void dequant_int4_gemv_rows_mma<bfloat, bfloat>(
    device const uint8_t*, device const half*, device const bfloat*, device bfloat*,
    constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint,
    device const uint*);
