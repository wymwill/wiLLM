#pragma once

// TensorView: a non owning view of an N dimensional array (N <= 4).
//
// A view is a pointer plus metadata: dtype, shape and strides. It never
// allocates and never frees. The memory belongs to someone else, usually an
// MmapFile (weights), an arena (activations) or the KV cache. Views are small
// and cheap to copy, so pass them by value, like std::span.
//
// Two flavours share one implementation:
//   TensorView       points at mutable data (kernel outputs)
//   ConstTensorView  points at read only data (weights, kernel inputs)
// A TensorView converts implicitly to a ConstTensorView, never the reverse.
//
// Strides are counted in ELEMENTS, not bytes, and are row major by default:
// for shape {rows, cols} the strides are {cols, 1}. Element (i, j) lives at
// data + (i * stride[0] + j * stride[1]) elements.
// See docs/adr/0002-tensor-view.md for why the design looks like this.

#include "willm/core/check.h"
#include "willm/core/dtype.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <type_traits>

namespace willm::core {

inline constexpr int kMaxRank = 4;

template <bool IsConst>
class BasicTensorView {
public:
    using Byte = std::conditional_t<IsConst, const std::byte, std::byte>;

    // An empty view: null data, rank 0.
    BasicTensorView() = default;

    // General constructor. `shape` and `strides` must have the same length (the rank).
    BasicTensorView(Byte* data, DType dtype, std::span<const std::int64_t> shape,
                    std::span<const std::int64_t> strides)
        : data_(data), dtype_(dtype), rank_(static_cast<int>(shape.size())) {
        WILLM_CHECK(shape.size() <= kMaxRank, "rank exceeds kMaxRank");
        WILLM_CHECK(shape.size() == strides.size(), "shape and strides differ in length");
        for (int i = 0; i < rank_; ++i) {
            const auto u = static_cast<std::size_t>(i);
            WILLM_CHECK(shape[u] >= 0, "negative dimension");
            shape_[u] = shape[u];
            strides_[u] = strides[u];
        }
    }

    // A row major (C order) view over contiguous memory. The usual way to make a view.
    static BasicTensorView contiguous(Byte* data, DType dtype,
                                      std::initializer_list<std::int64_t> shape) {
        return contiguous(data, dtype, std::span<const std::int64_t>(shape.begin(), shape.size()));
    }
    static BasicTensorView contiguous(Byte* data, DType dtype,
                                      std::span<const std::int64_t> shape) {
        WILLM_CHECK(shape.size() <= kMaxRank, "rank exceeds kMaxRank");
        std::array<std::int64_t, kMaxRank> strides{};
        std::int64_t stride = 1;
        for (std::size_t i = shape.size(); i-- > 0;) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return BasicTensorView(data, dtype, shape, std::span(strides.data(), shape.size()));
    }

    // Typed convenience: the dtype is deduced from the pointer.
    //   float buf[6]; auto v = TensorView::of(buf, {2, 3});
    template <typename T>
    static BasicTensorView of(T* data, std::initializer_list<std::int64_t> shape) {
        using Elem = std::remove_const_t<T>;
        return contiguous(reinterpret_cast<Byte*>(data), dtype_of<Elem>, shape);
    }

    // A mutable view converts to a const view. (It is a template so it never
    // counts as the copy constructor of TensorView itself.)
    template <bool OtherIsConst>
        requires(IsConst && !OtherIsConst)
    BasicTensorView(const BasicTensorView<OtherIsConst>& other)
        : data_(other.data_), dtype_(other.dtype_), rank_(other.rank_), shape_(other.shape_),
          strides_(other.strides_) {}

    Byte* raw_data() const noexcept { return data_; }
    DType dtype() const noexcept { return dtype_; }
    int rank() const noexcept { return rank_; }
    std::span<const std::int64_t> shape() const noexcept {
        return {shape_.data(), static_cast<std::size_t>(rank_)};
    }
    std::span<const std::int64_t> strides() const noexcept {
        return {strides_.data(), static_cast<std::size_t>(rank_)};
    }
    std::int64_t dim(int i) const {
        WILLM_CHECK(i >= 0 && i < rank_, "dimension index out of range");
        return shape_[static_cast<std::size_t>(i)];
    }
    std::int64_t stride(int i) const {
        WILLM_CHECK(i >= 0 && i < rank_, "dimension index out of range");
        return strides_[static_cast<std::size_t>(i)];
    }

    // Number of elements (1 for a rank 0 scalar view).
    std::int64_t numel() const noexcept {
        std::int64_t n = 1;
        for (int i = 0; i < rank_; ++i) {
            n *= shape_[static_cast<std::size_t>(i)];
        }
        return n;
    }

    // numel() * element size. Equals the memory spanned only when is_contiguous().
    std::size_t nbytes() const noexcept {
        return static_cast<std::size_t>(numel()) * dtype_size(dtype_);
    }

    // True if the elements are packed in row major order with no gaps.
    // Dimensions of size 1 may have any stride, since it is never used.
    bool is_contiguous() const noexcept {
        std::int64_t expected = 1;
        for (int i = rank_ - 1; i >= 0; --i) {
            const auto u = static_cast<std::size_t>(i);
            if (shape_[u] != 1 && strides_[u] != expected) {
                return false;
            }
            expected *= shape_[u];
        }
        return true;
    }

    // Typed pointer to the first element. Throws CheckError if T does not match dtype().
    template <typename T>
    std::conditional_t<IsConst, const T, T>* data() const {
        WILLM_CHECK(dtype_of<T> == dtype_,
                    std::string("dtype mismatch, view holds ") + std::string(dtype_name(dtype_)));
        return reinterpret_cast<std::conditional_t<IsConst, const T, T>*>(data_);
    }

    // Offset in elements of the element at `index` (one entry per dimension), bounds checked.
    std::int64_t offset_of(std::initializer_list<std::int64_t> index) const {
        WILLM_CHECK(static_cast<int>(index.size()) == rank_, "index rank does not match view");
        std::int64_t offset = 0;
        std::size_t d = 0;
        for (std::int64_t i : index) {
            WILLM_CHECK(i >= 0 && i < shape_[d], "index out of bounds");
            offset += i * strides_[d];
            ++d;
        }
        return offset;
    }

    // Bounds and dtype checked element access. Fine for tests and reference
    // code, too slow for hot loops (use data<T>() and strides there).
    template <typename T>
    std::conditional_t<IsConst, const T, T>& at(std::initializer_list<std::int64_t> index) const {
        return data<T>()[offset_of(index)];
    }

    // Drops the first dimension: for shape {L, R, C}, index(l) is the {R, C} view of layer l.
    // This is how the model will pick one layer's weights or one row of a matrix.
    BasicTensorView index(std::int64_t i) const {
        WILLM_CHECK(rank_ >= 1, "cannot index a rank 0 view");
        WILLM_CHECK(i >= 0 && i < shape_[0], "index out of bounds");
        BasicTensorView out;
        out.data_ = data_ + i * strides_[0] * static_cast<std::int64_t>(dtype_size(dtype_));
        out.dtype_ = dtype_;
        out.rank_ = rank_ - 1;
        for (int d = 1; d < rank_; ++d) {
            const auto u = static_cast<std::size_t>(d);
            out.shape_[u - 1] = shape_[u];
            out.strides_[u - 1] = strides_[u];
        }
        return out;
    }

private:
    template <bool>
    friend class BasicTensorView;

    Byte* data_ = nullptr;
    DType dtype_ = DType::F32;
    int rank_ = 0;
    std::array<std::int64_t, kMaxRank> shape_{};
    std::array<std::int64_t, kMaxRank> strides_{};
};

using TensorView = BasicTensorView<false>;
using ConstTensorView = BasicTensorView<true>;

} // namespace willm::core
