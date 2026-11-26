#include "willm/tokenizer/tokenizer.h"

#include <doctest/doctest.h>

namespace tok = willm::tokenizer;

TEST_CASE("special ids follow the Llama 2 vocabulary") {
    CHECK(tok::kBosId == 1);
    CHECK(tok::kEosId == 2);
    CHECK(tok::is_special(tok::kBosId));
    CHECK(tok::is_special(tok::kEosId));
    CHECK_FALSE(tok::is_special(100));
}
