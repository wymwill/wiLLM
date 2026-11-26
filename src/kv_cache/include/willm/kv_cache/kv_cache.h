#pragma once

// KV cache: stores each layer's attention keys and values so a new token only
// computes its own K and V instead of recomputing the whole prefix.
//
// This is the contiguous version: one f32 block of [n_layers, seq_len, kv_dim]
// for K and another for V, allocated up front for the full context. M6
// replaces it with a paged cache. This module does not depend on model/, it
// takes plain sizes, so the two can evolve separately.

#include "willm/core/tensor_view.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace willm::kv_cache {

struct KVCacheShape {
    std::int64_t n_layers = 0;
    std::int64_t seq_len = 0; // maximum number of positions the cache holds
    std::int64_t kv_dim = 0;  // n_kv_heads * head_dim

    // Elements in the K cache alone (the V cache is the same size).
    std::int64_t elements_per_cache() const { return n_layers * seq_len * kv_dim; }
};

// Bytes needed for K and V together at f32.
std::size_t f32_bytes(const KVCacheShape& shape);

// Owns the K and V storage for one sequence, zero initialized.
// The forward pass writes row `pos` of keys(layer) and values(layer) for each
// new token, and attention reads rows 0..pos.
class KVCache {
public:
    explicit KVCache(const KVCacheShape& shape);

    const KVCacheShape& shape() const { return shape_; }

    // [seq_len, kv_dim] view of one layer's keys or values.
    core::TensorView keys(std::int64_t layer);
    core::TensorView values(std::int64_t layer);
    core::ConstTensorView keys(std::int64_t layer) const;
    core::ConstTensorView values(std::int64_t layer) const;

private:
    std::int64_t layer_offset(std::int64_t layer) const; // checked, in elements

    KVCacheShape shape_;
    std::vector<float> keys_;
    std::vector<float> values_;
};

} // namespace willm::kv_cache
