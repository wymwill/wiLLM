# 0004 doctest via FetchContent, header only

Status: accepted
Date: 2026-11-25

## Context
We want a test framework that is small, fast to compile and has no install
step. It is the only third party dependency in M0.

## Decision
Fetch doctest v2.4.12 at configure time with CMake `FetchContent`, pinned to
its commit hash. We skip doctest's own CMakeLists (`SOURCE_SUBDIR` points at a
missing folder) and wrap the single header in our `willm_doctest` INTERFACE
target, marked SYSTEM so its code never trips our `-Werror` builds.
`tests/test_main.cpp` is the one file that defines `main`.

## Consequences
* The first configure needs network access, later ones reuse `build/_deps`.
* Skipping doctest's CMake avoids compatibility warnings with new CMake
  versions and keeps install rules out of our build.
* Upgrading means changing one hash in `cmake/Doctest.cmake`.
