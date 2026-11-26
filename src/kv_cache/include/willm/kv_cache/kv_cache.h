#pragma once

// KV cache: stores each layer's attention keys and values so a new token only
// computes its own K and V instead of recomputing the whole prefix.
//
// M1 adds a contiguous cache of shape [n_layers, seq_len, kv_dim] for K and
// another for V. M6 replaces it with a paged cache. This module does not
// depend on model/, it takes plain sizes, so the two can evolve separately.

#include <cstddef>
#include <cstdint>

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

} // namespace willm::kv_cache
