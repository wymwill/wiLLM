#pragma once

// Engine: owns a session (model weights, KV cache, sampler, tokenizer) and
// runs prefill followed by the decode loop. M1 and M2 fill it in. For now
// only the options a caller passes to generate().

#include "willm/sampler/sampler.h"

namespace willm::engine {

struct GenerateOptions {
    int max_new_tokens = 256;
    sampler::SamplerConfig sampler{};
};

// Throws core::CheckError if max_new_tokens <= 0 or the sampler config is invalid.
void validate(const GenerateOptions& options);

} // namespace willm::engine
