# wiLLM

An LLM inference engine and server written from scratch in C++20, as a
learning project. No ML libraries: every kernel, the forward pass, the KV cache
and the scheduler are hand written. It starts with Karpathy's tiny
[stories15M](https://github.com/karpathy/llama2.c) model on the CPU, then moves
to NEON SIMD, Apple Metal, and an OpenAI compatible server with continuous batching.

**Status:** M1 in progress. The checkpoint loader, KV cache, golden test data
and kernel interfaces are in place. The reference kernels and forward pass
are being written, so the kernel and golden tests fail until they are done.

## Build

Requirements: macOS with Xcode command line tools (Apple clang), CMake 3.25 or
newer (`brew install cmake`). Linux with clang or gcc also works.

```sh
cmake --preset debug          # Debug, with AddressSanitizer and UBSan
cmake --build --preset debug
ctest --preset debug          # run every module's tests

cmake --preset release        # -O3 -mcpu=native
cmake --build --preset release
```

Binaries land in `build/<preset>/apps/` and `build/<preset>/tests/`.
The `ci-debug` and `ci-release` presets are the same with warnings treated as errors.

Format code with `clang-format -i <files>` (CI pins clang-format 18.1.8).

## Run

```sh
python3 tools/download_stories15m.py                  # fetch stories15M.bin into models/
./build/release/apps/generate models/stories15M.bin   # loads the checkpoint and prints its config
```

Golden tests against the numpy reference are described in [tools/README.md](tools/README.md).

## Layout

| Path | What lives there |
| --- | --- |
| `src/core` | tensor view, dtypes, mmap file, timer (arena and logging later) |
| `src/kernels/ref` | naive scalar reference kernels, never deleted |
| `src/kernels/neon`, `metal` | optimized CPU and GPU kernels (M4, M5) |
| `src/model` | config, weight loader, transformer forward pass |
| `src/tokenizer` | BPE tokenizer for llama2.c `tokenizer.bin` |
| `src/sampler` | greedy, temperature, top k, top p |
| `src/kv_cache` | contiguous cache, then paged |
| `src/engine` | session with prefill and decode loop |
| `src/server` | HTTP API, scheduler, continuous batching (M6) |
| `apps/` | `generate` CLI and `bench` harness |
| `tests/` | one doctest executable per module, golden tests from M1 |
| `tools/` | Python reference tools (golden activations, weight download) |
| `docs/` | [architecture](docs/architecture.md), [ADRs](docs/adr/) |

## Results

Filled in from M3 on. Numbers come from `benchmarks/results/`.

| Milestone | Backend | Model | Prefill tok/s | Decode tok/s | TTFT | % peak BW |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

## Milestones

- [x] **M0** Skeleton: layout, CMake, doctest, CI, core (mmap, tensor view, timer)
- [ ] **M1** Reference FP32 forward pass, golden tests within 1e-4
- [ ] **M2** Tokenizer, sampler, `generate` CLI
- [ ] **M3** Benchmark harness
- [ ] **M4** CPU optimization: tiling, NEON, threads, int8
- [ ] **M5** Metal backend
- [ ] **M6** OpenAI compatible server with continuous batching and paged KV cache
- [ ] **M7+** Bigger models, weight streaming, speculative decoding
