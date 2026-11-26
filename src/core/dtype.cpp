#include "willm/core/dtype.h"

namespace willm::core {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return 4;
    case DType::F16:
        return 2;
    case DType::BF16:
        return 2;
    case DType::I8:
        return 1;
    case DType::I32:
        return 4;
    }
    return 0; // unreachable, the switch above is exhaustive
}

std::string_view dtype_name(DType dtype) {
    switch (dtype) {
    case DType::F32:
        return "f32";
    case DType::F16:
        return "f16";
    case DType::BF16:
        return "bf16";
    case DType::I8:
        return "i8";
    case DType::I32:
        return "i32";
    }
    return "unknown";
}

} // namespace willm::core
