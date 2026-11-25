# One test executable per module, each registered with CTest.
# Run them all with `ctest --preset debug`, or one with `ctest --preset debug -R core`.

add_library(willm_test_main STATIC test_main.cpp)
target_link_libraries(willm_test_main PUBLIC willm_doctest)

# willm_add_test(<name> SOURCES <files...> DEPS <targets...>)
function(willm_add_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
    add_executable(test_${name} ${ARG_SOURCES})
    target_link_libraries(test_${name} PRIVATE willm_test_main ${ARG_DEPS} willm_warnings)
    add_test(NAME ${name} COMMAND test_${name})
endfunction()

willm_add_test(core
    SOURCES core/test_mmap_file.cpp core/test_tensor_view.cpp core/test_timer.cpp
    DEPS willm::core)
