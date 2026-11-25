#include "willm/core/check.h"
#include "willm/model/config.h"

#include <doctest/doctest.h>

using willm::core::CheckError;
using willm::model::Config;

namespace {
// The header of Karpathy's stories15M.bin.
Config stories15m() {
    return Config{.dim = 288,
                  .hidden_dim = 768,
                  .n_layers = 6,
                  .n_heads = 6,
                  .n_kv_heads = 6,
                  .vocab_size = 32000,
                  .seq_len = 256};
}
} // namespace

TEST_CASE("stories15M config is valid and derives head sizes") {
    const Config c = stories15m();
    CHECK_NOTHROW(validate(c));
    CHECK(c.head_dim() == 48);
    CHECK(c.kv_dim() == 288);
}

TEST_CASE("invalid configs are rejected") {
    Config c = stories15m();
    c.n_heads = 5; // 288 is not divisible by 5
    CHECK_THROWS_AS(validate(c), CheckError);

    c = stories15m();
    c.n_kv_heads = 4; // 6 query heads cannot be grouped over 4 kv heads
    CHECK_THROWS_AS(validate(c), CheckError);
}
