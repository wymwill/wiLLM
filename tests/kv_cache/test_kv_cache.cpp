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

TEST_CASE("KVCache hands out per layer views into separate storage") {
    willm::kv_cache::KVCache cache({.n_layers = 2, .seq_len = 4, .kv_dim = 3});

    const auto k1 = cache.keys(1);
    CHECK(k1.dim(0) == 4);
    CHECK(k1.dim(1) == 3);
    CHECK(k1.at<float>({3, 2}) == 0.0F); // zero initialized

    k1.at<float>({2, 1}) = 7.0F;
    CHECK(cache.keys(1).at<float>({2, 1}) == 7.0F);
    CHECK(cache.keys(0).at<float>({2, 1}) == 0.0F);
    CHECK(cache.values(1).at<float>({2, 1}) == 0.0F);

    const auto& const_cache = cache;
    CHECK(const_cache.keys(1).at<float>({2, 1}) == 7.0F);
    CHECK_THROWS_AS(cache.keys(2), willm::core::CheckError);
}
