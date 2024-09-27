cmake_minimum_required(VERSION 3.12)

set(omm_benchmark_src benchmark.cpp )
add_executable(omm-benchmark ${omm_benchmark_src})
target_link_libraries(omm-benchmark omm-sdk benchmark::benchmark benchmark::benchmark_main )
set_target_properties(omm-benchmark PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}")