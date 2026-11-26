#pragma once

// Model weights as read only views into the file they were loaded from.
//
// Every loader (llama2.c today, safetensors or GGUF later) produces the same
// ModelWeights, so the forward pass never knows which format it came from.

#include "willm/core/mmap_file.h"
#include "willm/core/tensor_view.h"
#include "willm/model/config.h"

#include <filesystem>

namespace willm::model {

// Shapes follow llama2.c: matrices are [out, in], stacked per layer as [L, out, in].
struct Weights {
    core::ConstTensorView token_embedding; // [vocab, dim]
    core::ConstTensorView rms_att;         // [L, dim]
    core::ConstTensorView wq;              // [L, dim, dim]
    core::ConstTensorView wk;              // [L, kv_dim, dim]
    core::ConstTensorView wv;              // [L, kv_dim, dim]
    core::ConstTensorView wo;              // [L, dim, dim]
    core::ConstTensorView rms_ffn;         // [L, dim]
    core::ConstTensorView w1;              // [L, hidden, dim]  gate projection
    core::ConstTensorView w2;              // [L, dim, hidden]  down projection
    core::ConstTensorView w3;              // [L, hidden, dim]  up projection
    core::ConstTensorView rms_final;       // [dim]
    core::ConstTensorView wcls;            // [vocab, dim]  often the same memory as token_embedding
};

// A loaded model: the config, the weight views, and the file mapping they point into.
// Move only. Moving keeps the views valid because the mapping itself does not move.
struct ModelWeights {
    Config config;
    Weights weights;
    bool shared_classifier = true; // wcls aliases token_embedding
    core::MmapFile file;
};

// Loads a legacy llama2.c checkpoint (the format of stories15M.bin).
// Layout: 7 x int32 header (Config, vocab_size negative when wcls is stored
// separately), then f32 tensors in the order of Weights, with two unused RoPE
// tables between rms_final and wcls. See tools/llama2c.py for the same layout in Python.
// Throws std::system_error if the file cannot be read, core::CheckError if the
// header is invalid or the file size does not match the header.
ModelWeights load_llama2c(const std::filesystem::path& path);

} // namespace willm::model
