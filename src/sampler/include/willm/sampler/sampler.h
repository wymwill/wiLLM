#pragma once

// Sampler: picks the next token from the model's logits. M2 adds greedy,
// temperature, top k and top p sampling. For now only the configuration.

#include <cstdint>

namespace willm::sampler {

struct SamplerConfig {
    float temperature = 1.0F; // 0 means greedy (always take the argmax)
    int top_k = 0;            // 0 disables top k
    float top_p = 1.0F;       // 1 disables top p (nucleus) sampling
    std::uint64_t seed = 0;   // fixed seed gives deterministic output

    bool is_greedy() const { return temperature == 0.0F; }
};

// Throws core::CheckError if a field is out of range
// (temperature < 0, top_k < 0, top_p outside (0, 1]).
void validate(const SamplerConfig& config);

} // namespace willm::sampler
