# 0001 One CMake library per module, one way dependencies

Status: accepted
Date: 2026-11-25

## Context
The engine will grow from a single file forward pass to SIMD and Metal
backends, a paged KV cache and an HTTP server. Without boundaries, code like
the scheduler ends up calling kernels directly and nothing can be tested alone.

## Decision
Each folder under `src/` is a static library `willm_<module>` with its public
headers under `include/willm/<module>/`. Dependencies follow
`core ← kernels ← model ← engine ← server`, with tokenizer, sampler and kv_cache
depending only on core. A helper `willm_add_module` in `cmake/WillmModule.cmake`
creates every module the same way.

## Consequences
* Including a header from a module you do not link is a compile error, so the
  graph cannot rot silently.
* Each module has its own test executable.
* Slightly more CMake boilerplate, and moving code between modules means
  moving headers. Acceptable for the clarity it buys.
