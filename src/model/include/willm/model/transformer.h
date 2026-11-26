#pragma once

// The Llama 2 transformer forward pass, one token at a time, f32 reference kernels.

#include "willm/core/tensor_view.h"
#include "willm/kv_cache/kv_cache.h"
#include "willm/model/activation_sink.h"
#include "willm/model/weights.h"

#include <cstdint>
#include <vector>

namespace willm::model {

inline constexpr float kRmsNormEps = 1e-5F;

class Transformer {
public:
    // Keeps a reference to `model`, which must outlive the Transformer.
    // Allocates the activation buffers once, so forward() never allocates.
    explicit Transformer(const ModelWeights& model);

    const Config& config() const { return model_.config; }

    // A KV cache sized for this model's full context.
    kv_cache::KVCache make_cache() const;

    // Runs `token` at position `pos` through every layer and returns the
    // logits ([vocab]). The view points into this Transformer and is
    // overwritten by the next call. Writes this position's keys and values
    // into `cache` (rows 0..pos - 1 must already hold the earlier tokens).
    //
    // If `sink` is not null, record these activations with exactly these
    // names, in this order (l is the layer index). The golden test compares
    // each one against tools/llama2c.py, so the first mismatch tells you
    // which step is wrong:
    //   "embedding"             x after the embedding lookup
    //   "layer{l}.attn_norm"    rmsnorm(x) before attention
    //   "layer{l}.q"            query after RoPE           [dim]
    //   "layer{l}.k"            key after RoPE             [kv_dim]
    //   "layer{l}.v"            value                      [kv_dim]
    //   "layer{l}.attn_out"     attention output before wo [dim]
    //   "layer{l}.attn_residual" x after x += wo @ attn_out
    //   "layer{l}.ffn_norm"     rmsnorm(x) before the FFN
    //   "layer{l}.ffn_out"      w2 @ swiglu(w1 @ xb, w3 @ xb)
    //   "layer{l}.output"       x after x += ffn_out
    //   "final_norm"            rmsnorm(x) with rms_final
    //   "logits"                wcls @ final_norm          [vocab]
    core::ConstTensorView forward(std::int32_t token, std::int64_t pos, kv_cache::KVCache& cache,
                                  ActivationSink* sink = nullptr);

private:
    // Scratch activations, same names as llama2.c's RunState. Add, remove or
    // rename as you like, they are private.
    struct Buffers {
        std::vector<float> x;      // [dim]    residual stream
        std::vector<float> xb;     // [dim]    normed input / attention output
        std::vector<float> xb2;    // [dim]    output of wo and w2
        std::vector<float> q;      // [dim]
        std::vector<float> hb;     // [hidden] gate (w1) and then swiglu result
        std::vector<float> hb2;    // [hidden] up (w3)
        std::vector<float> scores; // [n_heads, seq_len] attention weights
        std::vector<float> logits; // [vocab]
    };

    const ModelWeights& model_;
    Buffers buf_;
};

} // namespace willm::model
