#include "willm/tokenizer/tokenizer.h"

namespace willm::tokenizer {

bool is_special(TokenId id) {
    return id == kUnknownId || id == kBosId || id == kEosId;
}

} // namespace willm::tokenizer
