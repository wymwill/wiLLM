#include "willm/core/check.h"

namespace willm::core::detail {

void check_failed(const char* expr, const std::string& msg, const char* file, int line) {
    throw CheckError(std::string(file) + ":" + std::to_string(line) + ": check failed (" + expr +
                     "): " + msg);
}

} // namespace willm::core::detail
