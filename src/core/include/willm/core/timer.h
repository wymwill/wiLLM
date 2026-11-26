#pragma once

// Timer: wall clock stopwatch for benchmarks and logs.
//
// steady_clock, not system_clock, because it never jumps backwards when the
// system time is adjusted. That matters when measuring time to first token.

#include <chrono>
#include <cstdint>

namespace willm::core {

class Timer {
public:
    using Clock = std::chrono::steady_clock;

    // Starts timing immediately.
    Timer() noexcept : start_(Clock::now()) {}

    // Restarts the stopwatch from zero.
    void reset() noexcept { start_ = Clock::now(); }

    Clock::duration elapsed() const noexcept { return Clock::now() - start_; }

    std::int64_t elapsed_ns() const noexcept {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed()).count();
    }
    double elapsed_ms() const noexcept {
        return std::chrono::duration<double, std::milli>(elapsed()).count();
    }
    double elapsed_s() const noexcept { return std::chrono::duration<double>(elapsed()).count(); }

private:
    Clock::time_point start_;
};

} // namespace willm::core
