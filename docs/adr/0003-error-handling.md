# 0003 Exceptions at boundaries, no exceptions in hot loops

Status: accepted
Date: 2026-11-25

## Context
We need to report bad files, bad user input and programming mistakes, and
Apple clang's C++20 library has no `std::expected` yet.

## Decision
* I/O failures throw `std::system_error` with the errno and path
  (for example `MmapFile::open`).
* Broken preconditions use `WILLM_CHECK(cond, msg)`, which throws
  `core::CheckError`. Checks stay on in Release because they also guard data
  read from files (shapes, sizes).
* Checks run once per call at API boundaries, never per element inside a
  kernel loop, so they cost nothing measurable.
* Because failures throw instead of aborting, tests can assert them with
  `CHECK_THROWS_AS`.

## Consequences
* Code must be exception safe, which RAII already gives us.
* The server (M6) will catch at the request boundary and turn errors into
  HTTP responses. If exceptions ever show up in a profile we revisit this.
