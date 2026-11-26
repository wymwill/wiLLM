#pragma once

// WILLM_CHECK(cond, msg): verify a precondition and throw CheckError if it fails.
//
// Use it at API boundaries (constructing a tensor view, loading a file), not
// inside per element loops. Checks stay on in Release builds because they guard
// things like shapes read from untrusted files. See docs/adr/0003-error-handling.md.

#include <stdexcept>
#include <string>

namespace willm::core {

// Thrown when a WILLM_CHECK fails. It signals a bug or bad input, not a
// condition the caller is expected to recover from in normal operation.
class CheckError : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

// Thrown by stubs that are still waiting to be written (see TODO(me) markers).
// Tests that hit one fail with a clear "not implemented" message instead of
// comparing garbage.
class NotImplementedError : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

namespace detail {
[[noreturn]] void check_failed(const char* expr, const std::string& msg, const char* file,
                               int line);
} // namespace detail

} // namespace willm::core

#define WILLM_CHECK(cond, msg)                                                                     \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            ::willm::core::detail::check_failed(#cond, (msg), __FILE__, __LINE__);                 \
        }                                                                                          \
    } while (false)
