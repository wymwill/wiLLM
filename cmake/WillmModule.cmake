# willm_add_module(<name> SOURCES <files...> [DEPS <targets...>])
#
# Creates the static library willm_<name> (alias willm::<name>) for one module
# under src/. Each module exposes only its own include/ folder, so a module can
# include headers from another module only if it links it. That makes the
# dependency graph in docs/architecture.md something the compiler enforces.
function(willm_add_module name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})

    set(target willm_${name})
    add_library(${target} STATIC ${ARG_SOURCES})
    add_library(willm::${name} ALIAS ${target})

    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(${target} PUBLIC cxx_std_20)
    target_link_libraries(${target}
        PUBLIC ${ARG_DEPS}
        PRIVATE willm_warnings)
endfunction()
