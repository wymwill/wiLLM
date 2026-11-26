# doctest, a single header test framework, fetched at configure time.
#
# We only need the header, so we skip doctest's own CMakeLists (SOURCE_SUBDIR
# points at a folder that does not exist) and wrap the header in our own
# INTERFACE target. SYSTEM keeps its warnings out of our -Werror builds.
# See docs/adr/0004-doctest-via-fetchcontent.md.

include(FetchContent)

FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG 1da23a3e8119ec5cce4f9388e91b065e20bf06f5 # v2.4.12
    SOURCE_SUBDIR do-not-add-subdirectory)
FetchContent_MakeAvailable(doctest)

add_library(willm_doctest INTERFACE)
target_include_directories(willm_doctest SYSTEM INTERFACE "${doctest_SOURCE_DIR}")
