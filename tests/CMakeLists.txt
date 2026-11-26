# One test executable per module, each registered with CTest.
# Run them all with `ctest --preset debug`, or one with `ctest --preset debug -R core`.

add_library(willm_test_main STATIC test_main.cpp)
target_link_libraries(willm_test_main PUBLIC willm_doctest)

# Shared test helpers (golden file reader).
add_library(willm_test_support STATIC support/golden.cpp)
target_include_directories(willm_test_support PUBLIC support)
target_link_libraries(willm_test_support PUBLIC willm::core PRIVATE willm_warnings)

# willm_add_test(<name> SOURCES <files...> DEPS <targets...>)
function(willm_add_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
    add_executable(test_${name} ${ARG_SOURCES})
    target_link_libraries(test_${name} PRIVATE willm_test_main ${ARG_DEPS} willm_warnings)
    target_compile_definitions(test_${name} PRIVATE
        WILLM_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data"
        WILLM_MODELS_DIR="${PROJECT_SOURCE_DIR}/models")
    add_test(NAME ${name} COMMAND test_${name})
endfunction()

willm_add_test(core
    SOURCES core/test_mmap_file.cpp core/test_tensor_view.cpp core/test_timer.cpp
    DEPS willm::core)
willm_add_test(kernels_ref
    SOURCES kernels/test_ref_elementwise.cpp kernels/test_ref_transformer_ops.cpp
    DEPS willm::kernels_ref)
willm_add_test(tokenizer SOURCES tokenizer/test_tokenizer.cpp DEPS willm::tokenizer)
willm_add_test(sampler SOURCES sampler/test_sampler.cpp DEPS willm::sampler)
willm_add_test(kv_cache SOURCES kv_cache/test_kv_cache.cpp DEPS willm::kv_cache)
willm_add_test(model
    SOURCES model/test_config.cpp model/test_llama2c_loader.cpp model/test_golden.cpp
    DEPS willm::model willm_test_support)
willm_add_test(engine SOURCES engine/test_options.cpp DEPS willm::engine)
willm_add_test(server SOURCES server/test_server_config.cpp DEPS willm::server)
