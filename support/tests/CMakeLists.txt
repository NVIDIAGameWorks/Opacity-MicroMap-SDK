option(OMM_TEST_ENABLE_IMAGE_DUMP "Certain tests may dump the result of the baking operation. Enable this option when tests are failing and need to be debugged." OFF )
option(OMM_TEST_USE_WARP "Run GPU tests on the D3D12 WARP software adapter" OFF)

if (OMM_TEST_ENABLE_IMAGE_DUMP)
    set(OMM_TEST_ENABLE_IMAGE_DUMP_VALUE 1)
else()
    set(OMM_TEST_ENABLE_IMAGE_DUMP_VALUE 0)
endif()

set(OMM_TEST_SHADER_COMPILER "")
if (OMM_ENABLE_GPU_TESTS)

    if (NOT OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
        message(FATAL_ERROR "GPU tests require embedded DXIL shaders")
    endif()

    # Query the actual binary, including custom compiler paths.
    execute_process(
        COMMAND "${SHADERMAKE_DXC_PATH}" --version
        OUTPUT_VARIABLE OMM_TEST_SHADER_COMPILER
        ERROR_VARIABLE OMM_TEST_SHADER_COMPILER
        RESULT_VARIABLE omm_test_compiler_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
        TIMEOUT 10
    )
    if (NOT omm_test_compiler_result STREQUAL "0" OR OMM_TEST_SHADER_COMPILER STREQUAL "")
        message(FATAL_ERROR "Cannot query shader compiler ${SHADERMAKE_DXC_PATH}: ${OMM_TEST_SHADER_COMPILER}")
    endif()
    string(REGEX REPLACE "[ \t\r\n]+" " " OMM_TEST_SHADER_COMPILER "${OMM_TEST_SHADER_COMPILER}")

    # Refresh the recorded version if the compiler is replaced in place.
    get_filename_component(omm_test_compiler_dir "${SHADERMAKE_DXC_PATH}" DIRECTORY)
    foreach(omm_test_compiler_file IN ITEMS "${SHADERMAKE_DXC_PATH}"
            "${omm_test_compiler_dir}/dxcompiler.dll" "${omm_test_compiler_dir}/dxil.dll")
        if (EXISTS "${omm_test_compiler_file}")
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${omm_test_compiler_file}")
        endif()
    endforeach()

    set(omm_tests_src_gpu nvrhi/nvrhi_environment.h nvrhi/nvrhi_environment.cpp nvrhi/nvrhi_wrapper.h nvrhi/nvrhi_wrapper.cpp test_omm_bake_gpu.cpp )

    if (WIN32)
        set(OMM_GPU_LIBS dxgi.lib d3d12.lib nvrhi nvrhi_d3d12 omm-gpu-nvrhi )
    endif()
endif()

configure_file(util/test_build_info.h.in "${CMAKE_CURRENT_BINARY_DIR}/test_build_info.h" @ONLY)

set(omm_tests_src_cpu util/stb_lib.cpp util/image.h util/omm.h util/omm_histogram.h util/omm_histogram.cpp test_basic.cpp test_texture.cpp test_raster_tri.cpp test_raster_line.cpp test_minimal_sample.cpp test_util.cpp test_tesselator.cpp test_omm_bake_cpu.cpp test_subdiv.cpp test_omm_indexing.cpp test_omm_log.cpp )
add_executable(tests main.cpp ${omm_tests_src_cpu} ${omm_tests_src_gpu})
# Test the shared integer math utility and software sampler on the CPU.
set(omm_tests_software_cpu test_software_math_cpu.cpp test_software_sampling_cpu.cpp)
target_sources(tests PRIVATE ${omm_tests_software_cpu} util/cpu_math_test.h util/hlsl_cpu_compat.h)
target_include_directories(tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/libraries/omm-lib/shaders")
if (MSVC)
    set_source_files_properties(${omm_tests_software_cpu} PROPERTIES COMPILE_OPTIONS /fp:strict)
else()
    set_source_files_properties(${omm_tests_software_cpu} PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-frounding-math")
endif()
if (OMM_ENABLE_GPU_TESTS)
    set(OMM_ENABLE_GPU_TESTS_VALUE 1)
else()
    set(OMM_ENABLE_GPU_TESTS_VALUE 0)
endif()

target_compile_definitions(tests  PRIVATE -DOMM_ENABLE_GPU_TESTS=${OMM_ENABLE_GPU_TESTS_VALUE} -DOMM_TEST_ENABLE_IMAGE_DUMP=${OMM_TEST_ENABLE_IMAGE_DUMP_VALUE} -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_compile_definitions(tests PRIVATE
    OMM_TEST_USE_WARP=$<BOOL:${OMM_TEST_USE_WARP}>
    OMM_GPU_REFERENCE_MODE=$<BOOL:${OMM_GPU_REFERENCE_MODE}>
    OMM_ENABLE_PRECISE_GPU_MATH=$<BOOL:${OMM_ENABLE_PRECISE_GPU_MATH}>
    OMM_DISABLE_GPU_SHADER_OPTIMIZATIONS=$<BOOL:${OMM_DISABLE_GPU_SHADER_OPTIMIZATIONS}>)
include_directories(tests  ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/libraries/omm-lib/src)

target_link_libraries(tests gtest gtest_main stb_lib ${OMM_LIB_TARGET_NAME} ${OMM_GPU_LIBS})

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
