#include "willm/core/check.h"
#include "willm/server/server_config.h"

#include <doctest/doctest.h>

using willm::server::ServerConfig;

TEST_CASE("ServerConfig defaults are valid") {
    ServerConfig config;
    CHECK_NOTHROW(validate(config));

    config.port = 0;
    CHECK_THROWS_AS(validate(config), willm::core::CheckError);
}
