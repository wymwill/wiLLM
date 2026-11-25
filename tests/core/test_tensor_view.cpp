#include "willm/core/tensor_view.h"

#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

using willm::core::CheckError;
using willm::core::ConstTensorView;
using willm::core::DType;
using willm::core::TensorView;

TEST_CASE("contiguous views get row major strides") {
    std::vector<float> buf(2 * 3 * 4);
    const auto v = TensorView::of(buf.data(), {2, 3, 4});

    CHECK(v.rank() == 3);
    CHECK(v.dtype() == DType::F32);
    CHECK(v.dim(0) == 2);
    CHECK(v.dim(2) == 4);
    CHECK(v.stride(0) == 12);
    CHECK(v.stride(1) == 4);
    CHECK(v.stride(2) == 1);
    CHECK(v.numel() == 24);
    CHECK(v.nbytes() == 24 * sizeof(float));
    CHECK(v.is_contiguous());
}

TEST_CASE("at() reads and writes the right element") {
    std::vector<float> buf(6);
    std::iota(buf.begin(), buf.end(), 0.0F); // 0 1 2 / 3 4 5
    const auto v = TensorView::of(buf.data(), {2, 3});

    CHECK(v.at<float>({0, 0}) == 0.0F);
    CHECK(v.at<float>({1, 2}) == 5.0F);
    v.at<float>({1, 0}) = 42.0F;
    CHECK(buf[3] == 42.0F);

    CHECK_THROWS_AS(v.at<float>({2, 0}), CheckError); // out of bounds
    CHECK_THROWS_AS(v.at<float>({0}), CheckError);    // wrong index rank
}

TEST_CASE("index() drops the leading dimension") {
    std::vector<float> buf(2 * 3 * 4);
    std::iota(buf.begin(), buf.end(), 0.0F);
    const auto v = TensorView::of(buf.data(), {2, 3, 4});

    const auto layer1 = v.index(1);
    CHECK(layer1.rank() == 2);
    CHECK(layer1.dim(0) == 3);
    CHECK(layer1.dim(1) == 4);
    CHECK(layer1.at<float>({0, 0}) == 12.0F);
    CHECK(layer1.index(2).at<float>({3}) == 23.0F);
}

TEST_CASE("custom strides describe a transposed view") {
    // A 2x3 matrix stored row major, viewed as its 3x2 transpose without copying.
    std::array<float, 6> buf{0, 1, 2, 3, 4, 5};
    const std::array<std::int64_t, 2> shape{3, 2};
    const std::array<std::int64_t, 2> strides{1, 3};
    const TensorView t(reinterpret_cast<std::byte*>(buf.data()), DType::F32, shape, strides);

    CHECK_FALSE(t.is_contiguous());
    CHECK(t.at<float>({2, 1}) == 5.0F); // original (1, 2)
    CHECK(t.at<float>({1, 0}) == 1.0F); // original (0, 1)
}

TEST_CASE("mutable views convert to const views, and dtype is checked") {
    std::vector<float> buf(4, 1.0F);
    const TensorView v = TensorView::of(buf.data(), {4});
    const ConstTensorView c = v; // implicit, adds const

    CHECK(c.data<float>() == buf.data());
    CHECK_THROWS_AS(c.data<std::int32_t>(), CheckError);

    static_assert(std::is_convertible_v<TensorView, ConstTensorView>);
    static_assert(!std::is_convertible_v<ConstTensorView, TensorView>);
    static_assert(std::is_same_v<decltype(c.data<float>()), const float*>);
}

TEST_CASE("views can be built over const data") {
    const std::array<std::int32_t, 3> ids{1, 2, 3};
    const auto v = ConstTensorView::of(ids.data(), {3});
    CHECK(v.dtype() == DType::I32);
    CHECK(v.at<std::int32_t>({2}) == 3);
}
