#include "willm/kv_cache/kv_cache.h"

#include "willm/core/check.h"

namespace willm::kv_cache {

std::size_t f32_bytes(const KVCacheShape& shape) {
    WILLM_CHECK(shape.n_layers > 0 && shape.seq_len > 0 && shape.kv_dim > 0,
                "cache dimensions must be positive");
    return 2 * static_cast<std::size_t>(shape.elements_per_cache()) * sizeof(float);
}

} // namespace willm::kv_cache
