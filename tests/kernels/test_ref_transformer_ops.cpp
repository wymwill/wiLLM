// Unit tests for the M1 reference kernels. Expected values come from the numpy
// functions in tools/llama2c.py, run on the same inputs.

#include "willm/kernels/ref/transformer_ops.h"

#include <doctest/doctest.h>

#include <cmath>
#include <vector>

using willm::core::CheckError;
using willm::core::TensorView;
namespace ref = willm::kernels::ref;

namespace {

TensorView vec(std::vector<float>& v) {
    return TensorView::of(v.data(), {static_cast<std::int64_t>(v.size())});
}

void check_close(const std::vector<float>& actual, const std::vector<float>& expected,
                 float tol = 1e-5F) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        INFO("element " << i);
        CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(tol));
    }
}

} // namespace

TEST_CASE("ref::rmsnorm") {
    std::vector<float> x{1, -2, 3, -4};
    std::vector<float> w{0.5F, 1, 1.5F, 2};
    std::vector<float> out(4);
    ref::rmsnorm(vec(out), vec(x), vec(w), 1e-5F);
    check_close(out, {0.182574064F, -0.730296256F, 1.643166577F, -2.921185026F});

    SUBCASE("in place") {
        ref::rmsnorm(vec(x), vec(x), vec(w), 1e-5F);
        check_close(x, out);
    }
}

TEST_CASE("ref::matmul") {
    std::vector<float> w{1, 2, 3, -1, 0.5F, 2}; // [2, 3]
    std::vector<float> x{2, -1, 0.5F};
    std::vector<float> out(2);
    ref::matmul(vec(out), TensorView::of(w.data(), {2, 3}), vec(x));
    check_close(out, {1.5F, -1.5F});

    std::vector<float> wrong(3);
    CHECK_THROWS_AS(ref::matmul(vec(wrong), TensorView::of(w.data(), {2, 3}), vec(x)), CheckError);
}

TEST_CASE("ref::rope") {
    std::vector<float> q{1, 2, 3, 4, 5, 6, 7, 8}; // 2 heads of 4
    std::vector<float> k{0.5F, -1, 2, 1};         // 1 kv head of 4
    ref::rope(vec(q), vec(k), 3, 4);
    check_close(q, {-1.272232513F, -1.838864985F, 2.878668100F, 4.088186636F, -5.796682531F,
                    -5.234354939F, 6.756886235F, 8.206368771F});
    check_close(k, {-0.353876240F, 1.060552501F, 1.969104567F, 1.059541034F});

    SUBCASE("position 0 is the identity") {
        std::vector<float> q0{1, 2, 3, 4};
        std::vector<float> k0{5, 6, 7, 8};
        ref::rope(vec(q0), vec(k0), 0, 4);
        check_close(q0, {1, 2, 3, 4});
        check_close(k0, {5, 6, 7, 8});
    }
}

TEST_CASE("ref::softmax") {
    std::vector<float> x{0, 1, 2};
    ref::softmax(vec(x));
    check_close(x, {0.090030573F, 0.244728471F, 0.665240956F});

    SUBCASE("large inputs do not overflow") {
        std::vector<float> big{1, 2, 3, 1000};
        ref::softmax(vec(big));
        CHECK(std::isfinite(big[0]));
        CHECK(big[3] == doctest::Approx(1.0F));
    }
}

TEST_CASE("ref::swiglu") {
    std::vector<float> gate{-2, 0, 1, 3};
    std::vector<float> up{1, 2, -1, 0.5F};
    std::vector<float> out(4);
    ref::swiglu(vec(out), vec(gate), vec(up));
    check_close(out, {-0.238405844F, 0.0F, -0.731058579F, 1.428861190F});
}

TEST_CASE("ref::attention with grouped query heads") {
    // 2 query heads share 1 kv head, head_dim 2, cache of 4 positions, pos = 2.
    // Row 3 of the cache holds junk that must be ignored (causal mask).
    std::vector<float> q{1, 0, 0, 1};
    std::vector<float> k_cache{1, 0, 0, 1, 1, 1, 9, 9};
    std::vector<float> v_cache{1, 2, 3, 4, 5, 6, 9, 9};
    std::vector<float> scores(2 * 4);
    std::vector<float> out(4);
    ref::attention(vec(out), TensorView::of(scores.data(), {2, 4}), vec(q),
                   TensorView::of(k_cache.data(), {4, 2}), TensorView::of(v_cache.data(), {4, 2}),
                   2, 2, 1);
    check_close(out, {3.0F, 4.0F, 3.406672556F, 4.406672556F});
}
