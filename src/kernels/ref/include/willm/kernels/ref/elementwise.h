#pragma once

// Reference elementwise kernels.
//
// This file doubles as the template for every kernel you write in M1:
//   * outputs first, as TensorView, inputs after, as ConstTensorView
//   * check shapes and dtypes once at the top, then loop over raw pointers
//   * no allocation inside a kernel, the caller owns all memory
//   * scalar, obvious code. The reference is the definition of correct.

#include "willm/core/tensor_view.h"

namespace willm::kernels::ref {

// out[i] = a[i] + b[i] for every element. Used for the residual connections.
// All three views must be contiguous f32 with the same shape. `out` may alias `a` or `b`.
void add(core::TensorView out, core::ConstTensorView a, core::ConstTensorView b);

} // namespace willm::kernels::ref
