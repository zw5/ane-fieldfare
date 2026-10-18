#include <metal_stdlib>
using namespace metal;

struct AttentionParams {
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint rotary_angles;
    uint position_count;
    uint first_position;
    uint capacity;
    uint sliding_window;
    uint full_attention;
    float rope_theta;
    float epsilon;
    float attention_scale;
};

kernel void attention_prepare_f32(
    device const float *query_in [[buffer(0)]],
    device const float *key_in [[buffer(1)]],
    device const float *value_in [[buffer(2)]],
    device const float *query_norm [[buffer(3)]],
    device const float *key_norm [[buffer(4)]],
    device float *query_out [[buffer(5)]],
    device float *key_cache [[buffer(6)]],
    device float *value_cache [[buffer(7)]],
    constant AttentionParams& p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    uint heads_per_position = p.query_heads + p.kv_heads;
    uint position = gid / heads_per_position;
    uint head = gid - position * heads_per_position;
    if (position >= p.position_count) return;
    uint absolute = p.first_position + position;
    uint slot = p.full_attention ? absolute : absolute % p.capacity;
    if (head < p.query_heads) {
        uint base = (position * p.query_heads + head) * p.head_dim;
        float square_sum = 0.0f;
        for (uint d = 0; d < p.head_dim; ++d) {
            float x = query_in[base + d];
            square_sum += x * x;
        }
        float inverse = rsqrt(square_sum / float(p.head_dim) + p.epsilon);
        uint half_dim = p.head_dim / 2;
        for (uint d = 0; d < p.head_dim; ++d)
            query_out[base + d] = query_in[base + d] * inverse * query_norm[d];
        for (uint angle = 0; angle < p.rotary_angles; ++angle) {
            float frequency = powr(p.rope_theta,
                -float(2 * angle) / float(p.head_dim));
            float phase = float(absolute) * frequency;
            float c = cos(phase), s = sin(phase);
            float left = query_out[base + angle];
            float right = query_out[base + half_dim + angle];
            query_out[base + angle] = left * c - right * s;
            query_out[base + half_dim + angle] = left * s + right * c;
        }
        return;
    }
    uint kv_head = head - p.query_heads;
    uint source_base = (position * p.kv_heads + kv_head) * p.head_dim;
    uint cache_base = (slot * p.kv_heads + kv_head) * p.head_dim;
    float key_square_sum = 0.0f, value_square_sum = 0.0f;
    for (uint d = 0; d < p.head_dim; ++d) {
        float k = key_in[source_base + d];
        float v = value_in[source_base + d];
        key_square_sum += k * k;
        value_square_sum += v * v;
    }
    float key_inverse = rsqrt(key_square_sum / float(p.head_dim) + p.epsilon);
    float value_inverse = rsqrt(value_square_sum / float(p.head_dim) + p.epsilon);
    for (uint d = 0; d < p.head_dim; ++d) {
        key_cache[cache_base + d] = key_in[source_base + d] * key_inverse * key_norm[d];
        value_cache[cache_base + d] = value_in[source_base + d] * value_inverse;
    }
    uint half_dim = p.head_dim / 2;
    for (uint angle = 0; angle < p.rotary_angles; ++angle) {
        float frequency = powr(p.rope_theta,
            -float(2 * angle) / float(p.head_dim));
        float phase = float(absolute) * frequency;
        float c = cos(phase), s = sin(phase);
        float left = key_cache[cache_base + angle];
        float right = key_cache[cache_base + half_dim + angle];
        key_cache[cache_base + angle] = left * c - right * s;
        key_cache[cache_base + half_dim + angle] = left * s + right * c;
    }
}

kernel void attention_prepare_ane_tile(
    device const half *qkv [[buffer(0)]],
    device const float *query_norm [[buffer(1)]],
    device const float *key_norm [[buffer(2)]],
    device float *query_out [[buffer(3)]],
    device float *key_cache [[buffer(4)]],
    device float *value_cache [[buffer(5)]],
    constant AttentionParams& p [[buffer(6)]],
    constant uint& query_channels [[buffer(7)]],
    constant uint& key_channels [[buffer(8)]],
    constant uint& value_stacked [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    uint heads_per_position = p.query_heads + p.kv_heads;
    uint position = gid / heads_per_position;
    uint head = gid - position * heads_per_position;
    if (position >= p.position_count) return;
    uint absolute = p.first_position + position;
    uint slot = p.full_attention ? absolute : absolute % p.capacity;
    if (head < p.query_heads) {
        uint row_base = (position * p.query_heads + head) * p.head_dim;
        uint channel_base = head * p.head_dim;
        float square_sum = 0.0f;
        for (uint d = 0; d < p.head_dim; ++d) {
            float x = float(qkv[(channel_base + d) * 64 + position]);
            square_sum += x * x;
        }
        float inverse = rsqrt(square_sum / float(p.head_dim) + p.epsilon);
        uint half_dim = p.head_dim / 2;
        for (uint d = 0; d < p.head_dim; ++d)
            query_out[row_base + d] =
                float(qkv[(channel_base + d) * 64 + position]) * inverse * query_norm[d];
        for (uint angle = 0; angle < p.rotary_angles; ++angle) {
            float frequency = powr(p.rope_theta, -float(2 * angle) / float(p.head_dim));
            float phase = float(absolute) * frequency;
            float c = cos(phase), s = sin(phase);
            float left = query_out[row_base + angle];
            float right = query_out[row_base + half_dim + angle];
            query_out[row_base + angle] = left * c - right * s;
            query_out[row_base + half_dim + angle] = left * s + right * c;
        }
        return;
    }
    uint kv_head = head - p.query_heads;
    uint cache_base = (slot * p.kv_heads + kv_head) * p.head_dim;
    uint key_channel_base = query_channels + kv_head * p.head_dim;
    uint value_channel_base = value_stacked
        ? query_channels + key_channels + kv_head * p.head_dim : key_channel_base;
    float key_square_sum = 0.0f, value_square_sum = 0.0f;
    for (uint d = 0; d < p.head_dim; ++d) {
        float k = float(qkv[(key_channel_base + d) * 64 + position]);
        float v = float(qkv[(value_channel_base + d) * 64 + position]);
        key_square_sum += k * k; value_square_sum += v * v;
    }
    float key_inverse = rsqrt(key_square_sum / float(p.head_dim) + p.epsilon);
    float value_inverse = rsqrt(value_square_sum / float(p.head_dim) + p.epsilon);
    for (uint d = 0; d < p.head_dim; ++d) {
        key_cache[cache_base + d] =
            float(qkv[(key_channel_base + d) * 64 + position]) * key_inverse * key_norm[d];
        value_cache[cache_base + d] =
            float(qkv[(value_channel_base + d) * 64 + position]) * value_inverse;
    }
    uint half_dim = p.head_dim / 2;
    for (uint angle = 0; angle < p.rotary_angles; ++angle) {
        float frequency = powr(p.rope_theta, -float(2 * angle) / float(p.head_dim));
        float phase = float(absolute) * frequency;
        float c = cos(phase), s = sin(phase);
        float left = key_cache[cache_base + angle];
        float right = key_cache[cache_base + half_dim + angle];
        key_cache[cache_base + angle] = left * c - right * s;
        key_cache[cache_base + half_dim + angle] = left * s + right * c;
    }
}

kernel void attention_causal_f32(
    device const float *query [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    device const float *value_cache [[buffer(2)]],
    device float *combined [[buffer(3)]],
    constant AttentionParams& p [[buffer(4)]],
    threadgroup float *scores [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 thread_shape [[threads_per_threadgroup]]) {
    uint threads = thread_shape.x;
    uint position = group.x;
    uint query_head = group.y;
    if (position >= p.position_count || query_head >= p.query_heads) return;
    uint absolute = p.first_position + position;
    uint first = (!p.full_attention && absolute + 1 > p.sliding_window)
        ? absolute + 1 - p.sliding_window : 0;
    uint causal_count = absolute - first + 1;
    uint groups = p.query_heads / p.kv_heads;
    uint kv_head = query_head / groups;
    uint query_base = (position * p.query_heads + query_head) * p.head_dim;
    threadgroup float reduction[256];
    for (uint offset = 0; offset < causal_count; ++offset) {
        uint source_position = first + offset;
        uint slot = p.full_attention ? source_position : source_position % p.capacity;
        uint key_base = (slot * p.kv_heads + kv_head) * p.head_dim;
        float partial = 0.0f;
        for (uint d = tid; d < p.head_dim; d += threads)
            partial += query[query_base + d] * key_cache[key_base + d];
        reduction[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduction[tid] += reduction[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) scores[offset] = reduction[0] * p.attention_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float maximum = -INFINITY;
        for (uint offset = 0; offset < causal_count; ++offset)
            maximum = max(maximum, scores[offset]);
        float denominator = 0.0f;
        for (uint offset = 0; offset < causal_count; ++offset) {
            scores[offset] = exp(scores[offset] - maximum);
            denominator += scores[offset];
        }
        for (uint offset = 0; offset < causal_count; ++offset)
            scores[offset] /= denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < p.head_dim; d += threads) {
        float result = 0.0f;
        for (uint offset = 0; offset < causal_count; ++offset) {
            uint source_position = first + offset;
            uint slot = p.full_attention ? source_position : source_position % p.capacity;
            uint value_base = (slot * p.kv_heads + kv_head) * p.head_dim;
            result += scores[offset] * value_cache[value_base + d];
        }
        combined[query_base + d] = result;
    }
}

kernel void attention_pack_ane_tile(
    device const float *combined [[buffer(0)]],
    device half *carrier [[buffer(1)]],
    constant AttentionParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint query_width = p.query_heads * p.head_dim;
    uint channel = gid / 64;
    uint position = gid - channel * 64;
    if (channel >= query_width) return;
    carrier[channel * 64 + position] = position < p.position_count
        ? half(combined[position * query_width + channel]) : half(0.0f);
}
