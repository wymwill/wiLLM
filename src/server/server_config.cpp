#include "willm/server/server_config.h"

#include "willm/core/check.h"

namespace willm::server {

void validate(const ServerConfig& config) {
    WILLM_CHECK(!config.host.empty(), "host must not be empty");
    WILLM_CHECK(config.port != 0, "port must not be 0");
    WILLM_CHECK(config.max_batch_size > 0, "max_batch_size must be positive");
}

} // namespace willm::server
