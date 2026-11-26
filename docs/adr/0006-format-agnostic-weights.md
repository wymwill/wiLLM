# 0006 One ModelWeights struct for every checkpoint format

Status: accepted
Date: 2026-11-26

## Context
M1 loads llama2.c checkpoints, M7 wants safetensors and maybe GGUF. The
forward pass should not change when the file format does.

## Decision
A loader is a function `path -> ModelWeights`. `ModelWeights` holds the
`Config`, a `Weights` struct of `ConstTensorView`s in llama2.c orientation
([out, in] matrices stacked as [L, out, in]), and the `MmapFile` they point into.
`load_llama2c` is the first loader. It validates the header and checks that the
file size matches exactly, so a truncated or wrong file fails at load time and
not as garbage output.

## Consequences
* Zero copy: views point straight into the mapping, so loading is instant and
  the OS pages weights in on first use.
* A future format whose layout differs (Hugging Face q and k rows are permuted
  for a different RoPE pairing, per tensor dtypes) is handled inside its loader,
  either by permuting on load into owned memory or by recording the convention.
* Mixed dtypes will need `Weights` fields that are not all f32, the views
  already carry a dtype.
