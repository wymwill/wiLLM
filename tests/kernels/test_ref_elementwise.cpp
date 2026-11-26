#include "willm/kernels/ref/elementwise.h"

#include <doctest/doctest.h>

#include <vector>

using willm::core::CheckError;
using willm::core::ConstTensorView;
using willm::core::TensorView;
namespace ref = willm::kernels::ref;

TEST_CASE("ref::add adds elementwise") {
    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b{10, 20, 30, 40};
    std::vector<float> out(4);

    ref::add(TensorView::of(out.data(), {2, 2}), TensorView::of(a.data(), {2, 2}),
             TensorView::of(b.data(), {2, 2}));
    CHECK(out == std::vector<float>{11, 22, 33, 44});
}

TEST_CASE("ref::add works in place, as the residual connection does") {
    std::vector<float> x{1, 1, 1};
    std::vector<float> delta{0.5F, 1.5F, 2.5F};
    const auto xv = TensorView::of(x.data(), {3});

    ref::add(xv, xv, TensorView::of(delta.data(), {3}));
    CHECK(x == std::vector<float>{1.5F, 2.5F, 3.5F});
}

TEST_CASE("ref::add rejects mismatched shapes") {
    std::vector<float> a(4), b(3), out(4);
    CHECK_THROWS_AS(ref::add(TensorView::of(out.data(), {4}), TensorView::of(a.data(), {4}),
                             TensorView::of(b.data(), {3})),
                    CheckError);
}
