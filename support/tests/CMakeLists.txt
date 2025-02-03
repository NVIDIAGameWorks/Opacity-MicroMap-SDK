cmake_minimum_required(VERSION 3.12)

if (WIN32)
	option(OMM_ENABLE_GPU_TESTS "Build GPU tests. Require nvrhi integration layer" ON)
else()
	option(OMM_ENABLE_GPU_TESTS "Build GPU tests. Require nvrhi integration layer" OFF)
endif()
option(OMM_TEST_ENABLE_IMAGE_DUMP "Certain tests may dump the result of the baking operation. Enable this option when tests are failing and need to be debugged." OFF )

if (OMM_TEST_ENABLE_IMAGE_DUMP)
    set(OMM_TEST_ENABLE_IMAGE_DUMP_VALUE 1)
else()
    set(OMM_TEST_ENABLE_IMAGE_DUMP_VALUE 0)
endif()

if (NOT OMM_INTEGRATION_LAYER_NVRHI AND OMM_ENABLE_GPU_TESTS)
    message(WARNING "Nvrhi integration layer is required for GPU unit tests to be functional. GPU Unit tests will be disabled (OMM_ENABLE_GPU_TESTS). To resolve make sure OMM_INTEGRATION_LAYER_NVRHI is ON.")
    set(OMM_ENABLE_GPU_TESTS OFF CACHE BOOL "" FORCE)
endif()

if (NOT WIN32)
    message(STATUS "Not running WIN32. GPU tests will be disabled.")
    set(OMM_ENABLE_GPU_TESTS OFF CACHE BOOL "" FORCE)
endif()

if (OMM_ENABLE_GPU_TESTS)

    set(omm_tests_src_gpu nvrhi/nvrhi_environment.h nvrhi/nvrhi_environment.cpp nvrhi/nvrhi_wrapper.h nvrhi/nvrhi_wrapper.cpp test_omm_bake_gpu.cpp )

    if (WIN32)
        set(OMM_GPU_LIBS dxgi.lib d3d12.lib nvrhi nvrhi_d3d12 omm-gpu-nvrhi )
    endif()
endif()

set(omm_tests_src_cpu util/stb_lib.cpp util/image.h util/omm.h util/omm_histogram.h util/omm_histogram.cpp test_basic.cpp test_texture.cpp test_raster_tri.cpp test_raster_line.cpp test_minimal_sample.cpp test_util.cpp test_tesselator.cpp test_omm_bake_cpu.cpp test_subdiv.cpp test_omm_indexing.cpp test_omm_log.cpp )
add_executable(tests main.cpp ${omm_tests_src_cpu} ${omm_tests_src_gpu})
if (OMM_ENABLE_GPU_TESTS)
    set(OMM_ENABLE_GPU_TESTS_VALUE 1)
else()
    set(OMM_ENABLE_GPU_TESTS_VALUE 0)
endif()

target_compile_definitions(tests  PRIVATE -DOMM_ENABLE_GPU_TESTS=${OMM_ENABLE_GPU_TESTS_VALUE} -DOMM_TEST_ENABLE_IMAGE_DUMP=${OMM_TEST_ENABLE_IMAGE_DUMP_VALUE} -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
include_directories(tests  ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/libraries/omm-lib/src)

target_link_libraries(tests gtest gtest_main stb_lib omm-lib ${OMM_GPU_LIBS})

set_target_properties(tests  PROPERTIES FOLDER "Support/Tests")

if (OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
    target_compile_definitions(tests  PRIVATE OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
endif()
if (OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
    target_compile_definitions(tests  PRIVATE OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
endif()

if (TARGET nvrhi)
    target_compile_definitions(tests  PRIVATE OMM_NVRHI_AVAILABLE)
endif()
