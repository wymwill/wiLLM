#include "willm/core/check.h"
#include "willm/kv_cache/kv_cache.h"

#include <doctest/doctest.h>

using willm::kv_cache::KVCacheShape;

TEST_CASE("KV cache size for stories15M") {
    // stories15M: 6 layers, 256 positions, 6 heads of 48 dims (kv_dim 288).
    const KVCacheShape shape{.n_layers = 6, .seq_len = 256, .kv_dim = 288};
    CHECK(shape.elements_per_cache() == 6 * 256 * 288);
    CHECK(willm::kv_cache::f32_bytes(shape) == 2UL * 6 * 256 * 288 * 4); // about 3.5 MB

    CHECK_THROWS_AS(willm::kv_cache::f32_bytes(KVCacheShape{}), willm::core::CheckError);
}
