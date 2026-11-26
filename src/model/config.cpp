#include "willm/model/config.h"

#include "willm/core/check.h"

namespace willm::model {

void validate(const Config& c) {
    WILLM_CHECK(c.dim > 0 && c.hidden_dim > 0 && c.n_layers > 0 && c.n_heads > 0 &&
                    c.n_kv_heads > 0 && c.vocab_size > 0 && c.seq_len > 0,
                "every config field must be positive");
    WILLM_CHECK(c.dim % c.n_heads == 0, "dim must be divisible by n_heads");
    WILLM_CHECK(c.n_heads % c.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads");
}

} // namespace willm::model
