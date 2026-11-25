#pragma once

// Element types a tensor can hold.
//
// The dtype is a runtime value, not a template parameter, because a model file
// mixes types (fp32 norms next to int8 or fp16 matrices) and the loader only
// learns them at runtime. Kernels check the dtype once on entry and then work
// with plain typed pointers.

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace willm::core {

enum class DType : std::uint8_t {
    F32,  // IEEE 754 single precision, the reference dtype
    F16,  // IEEE 754 half precision (later, Metal backend)
    BF16, // bfloat16 (later, safetensors models)
    I8,   // signed 8 bit, quantized weights (M4)
    I32,  // signed 32 bit, token ids
};

// Size of one element in bytes.
std::size_t dtype_size(DType dtype);

// Short lowercase name such as "f32", for logs and error messages.
std::string_view dtype_name(DType dtype);

// Maps a C++ element type to its DType. Only types with a native C++
// representation are listed. F16 and BF16 get one when we need them.
template <typename T>
struct DTypeOf;
template <>
struct DTypeOf<float> {
    static constexpr DType value = DType::F32;
};
template <>
struct DTypeOf<std::int8_t> {
    static constexpr DType value = DType::I8;
};
template <>
struct DTypeOf<std::int32_t> {
    static constexpr DType value = DType::I32;
};

template <typename T>
inline constexpr DType dtype_of = DTypeOf<T>::value;

} // namespace willm::core
