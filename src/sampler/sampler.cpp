#include "willm/sampler/sampler.h"

#include "willm/core/check.h"

namespace willm::sampler {

void validate(const SamplerConfig& config) {
    WILLM_CHECK(config.temperature >= 0.0F, "temperature must be >= 0");
    WILLM_CHECK(config.top_k >= 0, "top_k must be >= 0");
    WILLM_CHECK(config.top_p > 0.0F && config.top_p <= 1.0F, "top_p must be in (0, 1]");
}

} // namespace willm::sampler
