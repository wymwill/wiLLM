cmake_minimum_required(VERSION 3.25)

project(wiLLM
    VERSION 0.0.1
    DESCRIPTION "An LLM inference engine and server written from scratch"
    LANGUAGES CXX)

# Default to Debug so a bare `cmake -B build` gets sanitizers and assertions.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WILLM_BUILD_TESTS "Build the unit tests" ON)
option(WILLM_BUILD_APPS "Build the command line apps" ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CompilerOptions)
include(WillmModule)

# Modules, listed in dependency order. A module may only link modules above it.
add_subdirectory(src/core)
add_subdirectory(src/kernels)
add_subdirectory(src/tokenizer)
add_subdirectory(src/sampler)
add_subdirectory(src/kv_cache)
add_subdirectory(src/model)
add_subdirectory(src/engine)
add_subdirectory(src/server)

if(WILLM_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(WILLM_BUILD_TESTS)
    enable_testing()
    include(Doctest)
    add_subdirectory(tests)
endif()
