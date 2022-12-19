cmake_minimum_required(VERSION 3.10)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory("thirdparty/gbenchmark")
set_target_properties("benchmark" "benchmark_main" PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")

add_executable(benchmarks bm_ommbake.cpp)

target_link_libraries(benchmarks benchmark::benchmark omm-sdk)
set_target_properties(benchmarks PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}")