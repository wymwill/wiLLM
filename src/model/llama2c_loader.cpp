#include "willm/core/check.h"
#include "willm/model/weights.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace willm::model {

namespace {

// Hands out consecutive f32 tensors from the weight section of the file.
class TensorCursor {
public:
    TensorCursor(const std::byte* data, std::size_t size) : data_(data), size_(size) {}

    core::ConstTensorView next(std::initializer_list<std::int64_t> shape) {
        const auto view =
            core::ConstTensorView::contiguous(data_ + offset_, core::DType::F32, shape);
        skip(view.nbytes());
        return view;
    }

    void skip(std::size_t bytes) {
        WILLM_CHECK(offset_ + bytes <= size_, "checkpoint is truncated");
        offset_ += bytes;
    }

    std::size_t remaining() const { return size_ - offset_; }

private:
    const std::byte* data_;
    std::size_t size_;
    std::size_t offset_ = 0;
};

} // namespace

ModelWeights load_llama2c(const std::filesystem::path& path) {
    ModelWeights m;
    m.file = core::MmapFile::open(path);

    std::int32_t header[7];
    WILLM_CHECK(m.file.size() >= sizeof(header), "file too small for a llama2.c header");
    std::memcpy(header, m.file.data(), sizeof(header)); // memcpy, the bytes may be unaligned

    Config& c = m.config;
    c.dim = header[0];
    c.hidden_dim = header[1];
    c.n_layers = header[2];
    c.n_heads = header[3];
    c.n_kv_heads = header[4];
    c.vocab_size = std::abs(header[5]);
    c.seq_len = header[6];
    m.shared_classifier = header[5] > 0;
    validate(c);

    const std::int64_t L = c.n_layers;
    const std::int64_t dim = c.dim;
    const std::int64_t kv_dim = c.kv_dim();
    const std::int64_t hidden = c.hidden_dim;
    const std::int64_t vocab = c.vocab_size;

    TensorCursor cur(m.file.data() + sizeof(header), m.file.size() - sizeof(header));
    Weights& w = m.weights;
    w.token_embedding = cur.next({vocab, dim});
    w.rms_att = cur.next({L, dim});
    w.wq = cur.next({L, dim, dim});
    w.wk = cur.next({L, kv_dim, dim});
    w.wv = cur.next({L, kv_dim, dim});
    w.wo = cur.next({L, dim, dim});
    w.rms_ffn = cur.next({L, dim});
    w.w1 = cur.next({L, hidden, dim});
    w.w2 = cur.next({L, dim, hidden});
    w.w3 = cur.next({L, hidden, dim});
    w.rms_final = cur.next({dim});
    // Precomputed RoPE tables from an old llama2.c version, real and imaginary parts.
    cur.skip(2 * static_cast<std::size_t>(c.seq_len) * static_cast<std::size_t>(c.head_dim() / 2) *
             sizeof(float));
    w.wcls = m.shared_classifier ? w.token_embedding : cur.next({vocab, dim});

    WILLM_CHECK(cur.remaining() == 0,
                "checkpoint has " + std::to_string(cur.remaining()) + " unexpected trailing bytes");
    return m;
}

} // namespace willm::model
