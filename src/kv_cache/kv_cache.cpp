#include "willm/kv_cache/kv_cache.h"

#include "willm/core/check.h"

namespace willm::kv_cache {

std::size_t f32_bytes(const KVCacheShape& shape) {
    WILLM_CHECK(shape.n_layers > 0 && shape.seq_len > 0 && shape.kv_dim > 0,
                "cache dimensions must be positive");
    return 2 * static_cast<std::size_t>(shape.elements_per_cache()) * sizeof(float);
}

KVCache::KVCache(const KVCacheShape& shape)
    : shape_(shape), keys_(f32_bytes(shape) / 2 / sizeof(float), 0.0F),
      values_(keys_.size(), 0.0F) {}

std::int64_t KVCache::layer_offset(std::int64_t layer) const {
    WILLM_CHECK(layer >= 0 && layer < shape_.n_layers, "layer out of range");
    return layer * shape_.seq_len * shape_.kv_dim;
}

core::TensorView KVCache::keys(std::int64_t layer) {
    return core::TensorView::of(keys_.data() + layer_offset(layer),
                                {shape_.seq_len, shape_.kv_dim});
}
core::TensorView KVCache::values(std::int64_t layer) {
    return core::TensorView::of(values_.data() + layer_offset(layer),
                                {shape_.seq_len, shape_.kv_dim});
}
core::ConstTensorView KVCache::keys(std::int64_t layer) const {
    return core::ConstTensorView::of(keys_.data() + layer_offset(layer),
                                     {shape_.seq_len, shape_.kv_dim});
}
core::ConstTensorView KVCache::values(std::int64_t layer) const {
    return core::ConstTensorView::of(values_.data() + layer_offset(layer),
                                     {shape_.seq_len, shape_.kv_dim});
}

} // namespace willm::kv_cache
