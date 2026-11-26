# 0002 Non owning tensor view with runtime dtype and element strides

Status: accepted
Date: 2026-11-25

## Context
Kernels, the model and the KV cache all need to describe "a block of numbers
with a shape". Weights come from an mmap'ed file with dtypes only known at
runtime. Activations come from an arena. Nobody but the owner should free memory.

## Decision
`core::TensorView` is a pointer plus dtype, rank (at most 4), shape and
strides, stored in fixed size arrays.

* **Non owning.** Ownership stays with `MmapFile`, an arena or the cache.
  Views are trivially copyable and passed by value, like `std::span`.
* **Runtime dtype** rather than `TensorView<float>`. A model file mixes dtypes
  and the loader learns them at runtime. Kernels check the dtype once at entry,
  then use raw typed pointers from `data<T>()`.
* **Strides in elements, not bytes.** Index math reads naturally
  (`i * stride[0] + j`). Byte offsets are only computed in one place (`index()`).
* **Max rank 4, no heap.** A transformer never needs more than
  [layer, position, head, dim]. Fixed arrays keep views allocation free.
* **Const and mutable flavours** (`ConstTensorView`, `TensorView`) from one
  template, so read only weights cannot be passed as a kernel output.

## Consequences
* Kernels must validate shapes themselves. Each one starts with WILLM_CHECKs.
* Views can dangle if their owner dies first. RAII owners and clear ownership
  (see docs/architecture.md) are the mitigation.
* F16 and BF16 have no C++ element type yet, so `data<T>()` cannot return them.
  We will add storage types when the Metal and safetensors work needs them.
