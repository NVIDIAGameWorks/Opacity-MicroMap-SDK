add_executable(omm-cpu-benchmark main.cpp subdivision.h)
target_link_libraries(omm-cpu-benchmark PRIVATE ${OMM_LIB_TARGET_NAME})
target_compile_definitions(omm-cpu-benchmark PRIVATE
    OMM_BENCHMARK_DEFAULT_INPUT="${PROJECT_SOURCE_DIR}/assets/omm_example_data/leaves.bin")
set_target_properties(omm-cpu-benchmark PROPERTIES FOLDER "Tools/Benchmarks")

option(OMM_BUILD_GPU_BENCHMARK "Build the D3D12 GPU bake benchmark when its dependencies are available" ON)
if (OMM_BUILD_GPU_BENCHMARK)
    if (WIN32 AND TARGET omm-gpu-nvrhi AND TARGET nvrhi_d3d12 AND OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
        add_executable(omm-gpu-benchmark gpu_main.cpp gpu_runner.cpp gpu_runner.h subdivision.h)
        target_link_libraries(omm-gpu-benchmark PRIVATE omm-gpu-nvrhi nvrhi_d3d12 d3d12 dxgi)
        target_compile_definitions(omm-gpu-benchmark PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX
            OMM_GPU_REFERENCE_MODE=$<BOOL:${OMM_GPU_REFERENCE_MODE}>
            OMM_BENCHMARK_NVRHI_VALIDATION=$<BOOL:${NVRHI_WITH_VALIDATION}>
            OMM_BENCHMARK_DEFAULT_INPUT="${PROJECT_SOURCE_DIR}/assets/omm_example_data/leaves.bin")
        set_target_properties(omm-gpu-benchmark PROPERTIES FOLDER "Tools/Benchmarks")
        if (NOT OMM_STATIC_LIBRARY)
            add_custom_command(TARGET omm-gpu-benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:${OMM_LIB_TARGET_NAME}> $<TARGET_FILE_DIR:omm-gpu-benchmark>
                VERBATIM)
        endif()
    else()
        message(STATUS "Skipping omm-gpu-benchmark: requires Windows, omm-gpu-nvrhi, nvrhi_d3d12 and precompiled DXIL shaders")
    endif()
endif()
