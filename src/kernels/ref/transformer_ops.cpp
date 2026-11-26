#include "willm/kernels/ref/transformer_ops.h"

#include "willm/core/check.h"

#include <algorithm>
#include <cmath>

namespace willm::kernels::ref {

using core::ConstTensorView;
using core::DType;
using core::TensorView;

namespace {

bool is_f32_vector(ConstTensorView v) {
    return v.dtype() == DType::F32 && v.rank() == 1 && v.is_contiguous();
}

} // namespace

void rmsnorm(TensorView out, ConstTensorView x, ConstTensorView weight, float eps) {
    WILLM_CHECK(is_f32_vector(out) && is_f32_vector(x) && is_f32_vector(weight),
                "rmsnorm expects contiguous f32 vectors");
    WILLM_CHECK(out.dim(0) == x.dim(0) && weight.dim(0) == x.dim(0), "rmsnorm size mismatch");
    WILLM_CHECK(eps > 0.0F, "eps must be positive");

    // TODO(me): sum of squares, then scale every element.
    throw core::NotImplementedError("TODO(me) ref::rmsnorm");
}

void matmul(TensorView out, ConstTensorView w, ConstTensorView x) {
    WILLM_CHECK(is_f32_vector(out) && is_f32_vector(x), "matmul expects f32 vectors");
    WILLM_CHECK(w.dtype() == DType::F32 && w.rank() == 2 && w.is_contiguous(),
                "matmul expects a contiguous f32 matrix");
    WILLM_CHECK(w.dim(0) == out.dim(0) && w.dim(1) == x.dim(0), "matmul shape mismatch");

    // TODO(me): one dot product per row of w.
    throw core::NotImplementedError("TODO(me) ref::matmul");
}

void rope(TensorView q, TensorView k, std::int64_t pos, std::int64_t head_dim) {
    WILLM_CHECK(is_f32_vector(q) && is_f32_vector(k), "rope expects f32 vectors");
    WILLM_CHECK(head_dim > 0 && head_dim % 2 == 0, "head_dim must be even");
    WILLM_CHECK(q.dim(0) % head_dim == 0 && k.dim(0) % head_dim == 0,
                "q and k must hold whole heads");
    WILLM_CHECK(pos >= 0, "pos must be >= 0");

    // TODO(me): rotate each pair in q, then in k. Compute the angle in float.
    throw core::NotImplementedError("TODO(me) ref::rope");
}

void softmax(TensorView x) {
    WILLM_CHECK(is_f32_vector(x) && x.dim(0) > 0, "softmax expects a non empty f32 vector");

    // TODO(me): max, exp and sum, normalize.
    throw core::NotImplementedError("TODO(me) ref::softmax");
}

void attention(TensorView out, TensorView scores, ConstTensorView q, ConstTensorView k_cache,
               ConstTensorView v_cache, std::int64_t pos, std::int64_t n_heads,
               std::int64_t n_kv_heads) {
    WILLM_CHECK(is_f32_vector(out) && is_f32_vector(q), "attention expects f32 q and out");
    WILLM_CHECK(n_heads > 0 && n_kv_heads > 0 && n_heads % n_kv_heads == 0,
                "n_heads must be a multiple of n_kv_heads");
    WILLM_CHECK(q.dim(0) == out.dim(0) && q.dim(0) % n_heads == 0, "q and out size mismatch");
    const std::int64_t head_dim = q.dim(0) / n_heads;
    for (const ConstTensorView& cache : {k_cache, v_cache}) {
        WILLM_CHECK(cache.dtype() == DType::F32 && cache.rank() == 2 && cache.is_contiguous(),
                    "caches must be contiguous f32 [seq_len, kv_dim]");
        WILLM_CHECK(cache.dim(1) == n_kv_heads * head_dim, "cache kv_dim mismatch");
    }
    WILLM_CHECK(pos >= 0 && pos < k_cache.dim(0), "pos outside the cache");
    WILLM_CHECK(scores.dtype() == DType::F32 && scores.rank() == 2 && scores.is_contiguous() &&
                    scores.dim(0) == n_heads && scores.dim(1) >= pos + 1,
                "scores must be f32 [n_heads, >= pos + 1]");

    // TODO(me): per head, scores over 0..pos, softmax (reuse yours on a
    // TensorView of the first pos + 1 scores), weighted sum of values.
    throw core::NotImplementedError("TODO(me) ref::attention");
}

void swiglu(TensorView out, ConstTensorView gate, ConstTensorView up) {
    WILLM_CHECK(is_f32_vector(out) && is_f32_vector(gate) && is_f32_vector(up),
                "swiglu expects f32 vectors");
    WILLM_CHECK(out.dim(0) == gate.dim(0) && up.dim(0) == gate.dim(0), "swiglu size mismatch");

    // TODO(me): silu(gate) * up, elementwise.
    throw core::NotImplementedError("TODO(me) ref::swiglu");
}

} // namespace willm::kernels::ref
