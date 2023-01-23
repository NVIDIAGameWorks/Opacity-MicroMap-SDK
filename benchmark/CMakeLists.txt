cmake_minimum_required(VERSION 3.12)

add_executable(benchmarks bm_ommbake.cpp)

target_link_libraries(benchmarks benchmark::benchmark omm-sdk)
set_target_properties(benchmarks PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}")