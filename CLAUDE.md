# CLAUDE.md

wiLLM is a learning project: an LLM inference engine and server built from
scratch in C++20 (CMake, Apple clang, no ML libraries). It starts with the
stories15M llama2.c model, then CPU SIMD, Metal, and an OpenAI compatible
server with continuous batching. See README.md for milestones and
docs/architecture.md for the module graph.

## Roles
* Will writes the core logic: kernels, forward pass, KV cache, scheduler.
* Claude is mentor and architect: structure, build, interfaces, docs, and
  failing tests with `TODO(me)` markers for Will to make pass.
* One milestone at a time. Stop at the end of each and wait.
* Explain the why of design decisions and record important ones in docs/adr/.
* Ask before adding any dependency or making a hard to undo decision.

## Code rules
* C++20, `-Wall -Wextra -Wpedantic` (errors in CI), ASan and UBSan in Debug.
* Format with `.clang-format`. Namespaces per module, RAII for every resource,
  no global mutable state.
* Every optimized kernel keeps a naive `kernels/ref` twin and a test comparing them.
* Python only inside `tools/`, numpy plus torch where needed.

## Branches
* One branch per milestone or feature with a descriptive name such as
  `m0-skeleton`, `m1-reference-forward` or `fix-mmap-empty-file`.
* Never prefix or label branches with claude or session ids.

## Commits
* Commit at regular intervals, batching related changes.
* Each message is 1 to 3 plain sentences saying what was done, with no
  hyphens, em dashes, emojis, colons or semicolons.
* Do not list Claude as a contributor. No Co-Authored-By or other attribution lines.
