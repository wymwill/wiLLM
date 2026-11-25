#include "willm/core/check.h"
#include "willm/sampler/sampler.h"

#include <doctest/doctest.h>

using willm::core::CheckError;
using willm::sampler::SamplerConfig;

TEST_CASE("SamplerConfig defaults are valid and not greedy") {
    const SamplerConfig config;
    CHECK_NOTHROW(validate(config));
    CHECK_FALSE(config.is_greedy());
}

TEST_CASE("temperature 0 means greedy, and bad values are rejected") {
    SamplerConfig config;
    config.temperature = 0.0F;
    CHECK(config.is_greedy());

    config.top_p = 0.0F;
    CHECK_THROWS_AS(validate(config), CheckError);
}
