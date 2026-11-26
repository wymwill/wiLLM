#include "willm/kernels/ref/elementwise.h"

#include <algorithm>

namespace willm::kernels::ref {

using core::DType;

void add(core::TensorView out, core::ConstTensorView a, core::ConstTensorView b) {
    WILLM_CHECK(out.dtype() == DType::F32 && a.dtype() == DType::F32 && b.dtype() == DType::F32,
                "add expects f32");
    WILLM_CHECK(std::ranges::equal(out.shape(), a.shape()) &&
                    std::ranges::equal(out.shape(), b.shape()),
                "add expects equal shapes");
    WILLM_CHECK(out.is_contiguous() && a.is_contiguous() && b.is_contiguous(),
                "add expects contiguous views");

    float* o = out.data<float>();
    const float* x = a.data<float>();
    const float* y = b.data<float>();
    const std::int64_t n = out.numel();
    for (std::int64_t i = 0; i < n; ++i) {
        o[i] = x[i] + y[i];
    }
}

} // namespace willm::kernels::ref
