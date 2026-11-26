#pragma once

// Server: HTTP API (OpenAI style /v1/completions), request queue and
// continuous batching scheduler. Arrives in M6. For now only its config.

#include <cstdint>
#include <string>

namespace willm::server {

struct ServerConfig {
    std::string host = "127.0.0.1";
    std::uint16_t port = 8080;
    int max_batch_size = 8; // sequences decoded together per step
};

// Throws core::CheckError if host is empty, port is 0, or max_batch_size <= 0.
void validate(const ServerConfig& config);

} // namespace willm::server
