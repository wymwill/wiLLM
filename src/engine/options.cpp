#include "willm/engine/options.h"

#include "willm/core/check.h"

namespace willm::engine {

void validate(const GenerateOptions& options) {
    WILLM_CHECK(options.max_new_tokens > 0, "max_new_tokens must be positive");
    sampler::validate(options.sampler);
}

} // namespace willm::engine
