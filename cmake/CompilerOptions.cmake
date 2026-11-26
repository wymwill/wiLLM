# Compiler flags shared by every wiLLM target.
#
# Warnings live on an INTERFACE target (willm_warnings) so they apply to our
# code only and never to third party code such as doctest. Sanitizers and
# optimization flags are global because every object in a binary must agree
# on them (mixing ASan and non ASan objects gives confusing results).
#
# Warnings become errors when CMAKE_COMPILE_WARNING_AS_ERROR is ON, which the
# ci presets set. Locally they stay warnings so you can iterate quickly.

include(CheckCXXCompilerFlag)

add_library(willm_warnings INTERFACE)
target_compile_options(willm_warnings INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Woverloaded-virtual
    -Wnull-dereference
    -Wimplicit-fallthrough)

# Debug builds run with AddressSanitizer and UndefinedBehaviorSanitizer.
# -fno-sanitize-recover makes UB abort the test instead of printing and moving on.
set(WILLM_SANITIZE_FLAGS
    -fsanitize=address,undefined
    -fno-sanitize-recover=undefined
    -fno-omit-frame-pointer)
add_compile_options("$<$<CONFIG:Debug>:${WILLM_SANITIZE_FLAGS}>")
add_link_options("$<$<CONFIG:Debug>:-fsanitize=address,undefined>")

# Release builds target the exact CPU they are built on.
# Apple silicon (and ARM Linux) spell this -mcpu=native, x86 spells it -march=native.
check_cxx_compiler_flag(-mcpu=native WILLM_HAS_MCPU_NATIVE)
if(WILLM_HAS_MCPU_NATIVE)
    set(WILLM_NATIVE_FLAG -mcpu=native)
else()
    set(WILLM_NATIVE_FLAG -march=native)
endif()
add_compile_options("$<$<CONFIG:Release>:-O3;${WILLM_NATIVE_FLAG}>")
