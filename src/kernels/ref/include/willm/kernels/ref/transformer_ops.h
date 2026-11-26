#pragma once

// Reference kernels for one transformer layer, scalar f32. M1 is yours: each
// function below has its contract (shape and dtype checks) written and a
// TODO(me) where the math goes. Make tests/kernels/test_ref_transformer_ops.cpp
// pass, then the golden tests in tests/model/.
//
// Conventions, same as elementwise.h:
//   * outputs first (TensorView), inputs after (ConstTensorView)
//   * all views are contiguous f32, checked on entry
//   * no allocation, the caller passes any scratch memory
//
// Shapes use the llama2.c names: dim (model width), head_dim, kv_dim
// (n_kv_heads * head_dim), hidden (FFN width), seq_len (context length).

#include "willm/core/tensor_view.h"

#include <cstdint>

namespace willm::kernels::ref {

// RMSNorm: out[i] = weight[i] * x[i] / sqrt(mean(x^2) + eps).
// x, weight, out are [n]. out may alias x.
// Accumulate the sum of squares in float. llama2.c uses eps = 1e-5.
void rmsnorm(core::TensorView out, core::ConstTensorView x, core::ConstTensorView weight,
             float eps);

// Matrix times vector: out[i] = sum_j w[i, j] * x[j].
// w is [d, n] row major (one output per row, which is how llama2.c stores
// every weight matrix), x is [n], out is [d]. out must not alias x.
// This is the kernel that dominates runtime, M4 is all about making it fast.
void matmul(core::TensorView out, core::ConstTensorView w, core::ConstTensorView x);

// Rotary position embedding, applied in place to q and k for position `pos`.
// q is [n_heads * head_dim], k is [n_kv_heads * head_dim].
// Within each head, consecutive pairs (v[2i], v[2i+1]) are rotated by the angle
//     pos * theta^(-2i / head_dim),  theta = 10000
// i.e.  v0' = v0 cos - v1 sin,  v1' = v0 sin + v1 cos.
// (This is the llama2.c pairing. Hugging Face models pair i with i + head_dim/2
// instead, remember that when M7 loads safetensors.)
void rope(core::TensorView q, core::TensorView k, std::int64_t pos, std::int64_t head_dim);

// Numerically stable softmax in place over all elements of x ([n]).
// Subtract the max before exp so large scores cannot overflow.
void softmax(core::TensorView x);

// Causal multi head attention for the token at position `pos`, given a
// cache that already holds this token's k and v at row `pos`.
//   q        [n_heads * head_dim]
//   k_cache  [seq_len, kv_dim]   one layer of the key cache, rows 0..pos are valid
//   v_cache  [seq_len, kv_dim]
//   scores   [n_heads, seq_len]  scratch for the attention weights
//   out      [n_heads * head_dim]
// For each head h, using kv head g = h / (n_heads / n_kv_heads):
//   score[t] = dot(q_h, k_cache[t, g]) / sqrt(head_dim)   for t in 0..pos
//   weights  = softmax(score[0..pos])
//   out_h    = sum_t weights[t] * v_cache[t, g]
void attention(core::TensorView out, core::TensorView scores, core::ConstTensorView q,
               core::ConstTensorView k_cache, core::ConstTensorView v_cache, std::int64_t pos,
               std::int64_t n_heads, std::int64_t n_kv_heads);

// SwiGLU gate: out[i] = silu(gate[i]) * up[i], silu(x) = x / (1 + exp(-x)).
// All [hidden]. out may alias gate or up.
void swiglu(core::TensorView out, core::ConstTensorView gate, core::ConstTensorView up);

} // namespace willm::kernels::ref
