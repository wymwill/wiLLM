#include "willm/core/timer.h"

#include <doctest/doctest.h>

#include <chrono>
#include <thread>

using willm::core::Timer;

TEST_CASE("Timer measures at least the time slept") {
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Only a lower bound: the OS may oversleep, but never undersleeps.
    CHECK(timer.elapsed_ms() >= 5.0);
    CHECK(timer.elapsed_ns() >= 5'000'000);
    CHECK(timer.elapsed_s() >= 0.005);
}

TEST_CASE("Timer::reset restarts from zero") {
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    const auto before = timer.elapsed_ns();
    timer.reset();
    CHECK(timer.elapsed_ns() < before);
}
