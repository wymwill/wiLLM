#pragma once

// Tokenizer: text <-> token ids. M2 adds the BPE tokenizer that reads
// llama2.c's tokenizer.bin. For now this module only defines the vocabulary
// conventions the rest of the engine relies on.

#include <cstdint>

namespace willm::tokenizer {

// Token ids are 32 bit, matching how llama2.c stores them.
using TokenId = std::int32_t;

// Special ids in the Llama 2 sentencepiece vocabulary (also used by stories15M).
inline constexpr TokenId kUnknownId = 0;
inline constexpr TokenId kBosId = 1; // beginning of sequence, prepended to every prompt
inline constexpr TokenId kEosId = 2; // end of sequence, generation stops here

// True for the ids above, which are never printed as text.
bool is_special(TokenId id);

} // namespace willm::tokenizer
