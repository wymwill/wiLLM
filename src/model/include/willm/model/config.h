#pragma once

// Model hyperparameters, mirroring the 7 int32 header of a llama2.c checkpoint.
//
// The loader (M1) fills this from whatever format it reads, so llama2.c,
// safetensors and GGUF all become the same Config. Nothing downstream needs to
// know which file format the weights came from.

#include <cstdint>

namespace willm::model {

struct Config {
    std::int32_t dim = 0;        // transformer width (embedding size)
    std::int32_t hidden_dim = 0; // FFN inner width
    std::int32_t n_layers = 0;
    std::int32_t n_heads = 0;    // query heads
    std::int32_t n_kv_heads = 0; // key and value heads (< n_heads means grouped query attention)
    std::int32_t vocab_size = 0;
    std::int32_t seq_len = 0; // maximum context length

    std::int32_t head_dim() const { return dim / n_heads; }
    std::int32_t kv_dim() const { return n_kv_heads * head_dim(); }
};

// Throws core::CheckError unless every field is positive, dim divides evenly
// into n_heads, and n_heads is a multiple of n_kv_heads.
void validate(const Config& config);

} // namespace willm::model
