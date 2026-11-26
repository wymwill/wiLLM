#pragma once

// ActivationSink: lets tests (and later a debugger or profiler) see the
// intermediate activations of a forward pass without the model knowing who
// is listening.

#include "willm/core/tensor_view.h"

#include <string_view>

namespace willm::model {

class ActivationSink {
public:
    virtual ~ActivationSink() = default;

    // Called by the forward pass with a name and a view that is only valid
    // during the call. Implementations copy what they need.
    virtual void record(std::string_view name, core::ConstTensorView value) = 0;
};

} // namespace willm::model
