#include "willm/model/transformer.h"

#include "willm/core/check.h"
#include "willm/kernels/ref/elementwise.h"
#include "willm/kernels/ref/transformer_ops.h"

#include <string>

namespace willm::model {

namespace ref = kernels::ref;
using core::ConstTensorView;
using core::TensorView;

Transformer::Transformer(const ModelWeights& model) : model_(model) {
    const Config& c = model.config;
    const auto dim = static_cast<std::size_t>(c.dim);
    const auto hidden = static_cast<std::size_t>(c.hidden_dim);
    buf_.x.resize(dim);
    buf_.xb.resize(dim);
    buf_.xb2.resize(dim);
    buf_.q.resize(dim);
    buf_.hb.resize(hidden);
    buf_.hb2.resize(hidden);
    buf_.scores.resize(static_cast<std::size_t>(c.n_heads) * static_cast<std::size_t>(c.seq_len));
    buf_.logits.resize(static_cast<std::size_t>(c.vocab_size));
}

kv_cache::KVCache Transformer::make_cache() const {
    const Config& c = model_.config;
    return kv_cache::KVCache({.n_layers = c.n_layers, .seq_len = c.seq_len, .kv_dim = c.kv_dim()});
}

ConstTensorView Transformer::forward(std::int32_t token, std::int64_t pos, kv_cache::KVCache& cache,
                                     ActivationSink* sink) {
    const Config& c = model_.config;
    WILLM_CHECK(token >= 0 && token < c.vocab_size, "token id out of range");
    WILLM_CHECK(pos >= 0 && pos < c.seq_len, "position beyond the context length");
    WILLM_CHECK(cache.shape().n_layers == c.n_layers && cache.shape().kv_dim == c.kv_dim(),
                "cache does not match this model");

    // TODO(me): the forward pass. Roughly, following llama2.c's forward():
    //   1. copy row `token` of token_embedding into x
    //   2. for each layer l (Weights::wq.index(l) gives that layer's [dim, dim] view):
    //        xb = rmsnorm(x, rms_att[l])
    //        q = wq[l] @ xb,  k and v go straight into row `pos` of cache.keys(l) / values(l)
    //        rope(q, k, pos, head_dim)
    //        xb = attention(q, cache.keys(l), cache.values(l), pos)
    //        x += wo[l] @ xb
    //        xb = rmsnorm(x, rms_ffn[l])
    //        hb = swiglu(w1[l] @ xb, w3[l] @ xb)
    //        x += w2[l] @ hb
    //   3. logits = wcls @ rmsnorm(x, rms_final)
    // Record each activation listed in transformer.h when sink != nullptr.
    // TensorView::of(buf_.x.data(), {c.dim}) wraps a buffer as a view.
    (void)sink;
    throw core::NotImplementedError("TODO(me) Transformer::forward");
}

} // namespace willm::model
