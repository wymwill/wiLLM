#include "willm/core/check.h"
#include "willm/engine/options.h"

#include <doctest/doctest.h>

using willm::engine::GenerateOptions;

TEST_CASE("GenerateOptions defaults are valid") {
    GenerateOptions options;
    CHECK_NOTHROW(validate(options));

    options.max_new_tokens = 0;
    CHECK_THROWS_AS(validate(options), willm::core::CheckError);
}
