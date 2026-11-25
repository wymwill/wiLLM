add_executable(willm_generate generate/main.cpp)
target_link_libraries(willm_generate PRIVATE willm::engine willm_warnings)
set_target_properties(willm_generate PROPERTIES OUTPUT_NAME generate)

add_executable(willm_bench bench/main.cpp)
target_link_libraries(willm_bench PRIVATE willm::engine willm_warnings)
set_target_properties(willm_bench PROPERTIES OUTPUT_NAME bench)
