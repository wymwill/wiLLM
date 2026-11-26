#include "golden.h"

#include "willm/core/check.h"

#include <cstdint>
#include <cstring>

namespace willm::testing {

namespace {

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <typename T>
    T read() {
        T value;
        std::memcpy(&value, take(sizeof(T)), sizeof(T));
        return value;
    }

    const std::byte* take(std::size_t n) {
        WILLM_CHECK(offset_ + n <= bytes_.size(), "golden file is truncated");
        const std::byte* p = bytes_.data() + offset_;
        offset_ += n;
        return p;
    }

    bool done() const { return offset_ == bytes_.size(); }

private:
    std::span<const std::byte> bytes_;
    std::size_t offset_ = 0;
};

} // namespace

GoldenFile GoldenFile::open(const std::filesystem::path& path) {
    GoldenFile g;
    g.file_ = core::MmapFile::open(path);
    Reader r(g.file_.bytes());

    WILLM_CHECK(std::memcmp(r.take(4), "WGLD", 4) == 0, "not a golden file");
    WILLM_CHECK(r.read<std::uint32_t>() == 1, "unsupported golden file version");
    const auto count = r.read<std::uint32_t>();

    for (std::uint32_t i = 0; i < count; ++i) {
        const auto name_len = r.read<std::uint32_t>();
        const auto* name_bytes = reinterpret_cast<const char*>(r.take(name_len));
        std::string name(name_bytes, name_len);
        r.take((4 - name_len % 4) % 4); // padding

        const auto rank = r.read<std::uint32_t>();
        WILLM_CHECK(rank <= core::kMaxRank, "golden tensor rank too large");
        std::vector<std::int64_t> shape(rank);
        for (auto& d : shape) {
            d = r.read<std::int64_t>();
        }
        auto view = core::ConstTensorView::contiguous(nullptr, core::DType::F32, shape);
        view = core::ConstTensorView::contiguous(r.take(view.nbytes()), core::DType::F32, shape);

        g.names_.push_back(name);
        g.tensors_.emplace(std::move(name), view);
    }
    WILLM_CHECK(r.done(), "golden file has trailing bytes");
    return g;
}

core::ConstTensorView GoldenFile::get(const std::string& name) const {
    const auto it = tensors_.find(name);
    WILLM_CHECK(it != tensors_.end(), "no golden tensor named " + name);
    return it->second;
}

} // namespace willm::testing
