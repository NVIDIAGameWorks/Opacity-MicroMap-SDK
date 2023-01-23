cmake_minimum_required(VERSION 3.12)

option(OMM_DISABLE_INTERPROCEDURAL_OPTIMIZATION "disable interprocedural optimization" OFF)
option(OMM_ENABLE_OPENMP "enable openmp" ON)
option(OMM_ENABLE_PRECOMPILED_SHADERS_DXIL "Embed precompiled dxil shaders. Require path to dxc.exe (normally located in WindowSDK)." ON)
option(OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV "Embed precompiled spirv shaders. Require path to VulkanSDK." ON)
option(OMM_STATIC_LIBRARY "build static lib" OFF)
option(OMM_CROSSCOMPILE_AARCH64 "cross compilation for aarch64" OFF)
option(OMM_CROSSCOMPILE_X86_64 "cross compilation for x86_64" OFF)
option(OMM_SHADER_DEBUG_INFO "enable embedded shader debug info" OFF)
option(OMM_INSTALL "Generate install rules for OMM" ON)

if (OMM_ENABLE_OPENMP)
find_package(OpenMP)
endif()

if (OMM_CROSSCOMPILE_AARCH64)
    set(CMAKE_SYSTEM_PROCESSOR "aarch64")
    message(STATUS "CROSSCOMPILE_AARCH64 enabled.")
endif()
if (OMM_CROSSCOMPILE_X86_64)
    set(CMAKE_SYSTEM_PROCESSOR "x86_64")
    message(STATUS "CROSSCOMPILE_X86_64 enabled.")
endif()

set(OMM_ENABLE_PRECOMPILED_SHADERS ${OMM_ENABLE_PRECOMPILED_SHADERS_DXIL} OR ${OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV} CACHE INTERNAL "")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POSITION_INDEPENDENT_BINARIES ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if (OMM_DISABLE_INTERPROCEDURAL_OPTIMIZATION)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

if ("${OMM_SHADER_OUTPUT_PATH}" STREQUAL "")
    set(OMM_SHADER_OUTPUT_PATH "${PROJECT_BINARY_DIR}/shaders")
endif()

# Vulkan resource offsets
set (OMM_VK_S_SHIFT 100)
set (OMM_VK_T_SHIFT 200)
set (OMM_VK_B_SHIFT 300)
set (OMM_VK_U_SHIFT 400)
add_definitions(-DOMM_VK_S_SHIFT=${OMM_VK_S_SHIFT} -DOMM_VK_T_SHIFT=${OMM_VK_T_SHIFT} -DOMM_VK_B_SHIFT=${OMM_VK_B_SHIFT} -DOMM_VK_U_SHIFT=${OMM_VK_U_SHIFT})

if (OMM_ENABLE_PRECOMPILED_SHADERS)
    # include path for DXC and FXC (used in ShaderCompilation.cmake)
    set(OMM_SHADER_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(OMM_HEADER_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/shaders")

    # list HLSL shaders and headers
    file(GLOB_RECURSE OMM_HLSL_FILES "shaders/*.hlsl")
    file(GLOB_RECURSE OMM_HLSL_INCLUDE "shaders/*.hlsli")

    # Vulkan resource offsets
    include(ShaderCompilation.cmake)

    # create "build/Shaders"
    add_custom_target(CreateFolderForShaders ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OMM_SHADER_OUTPUT_PATH}")
    set_property(TARGET CreateFolderForShaders PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/CustomScripts") 

    # define build rules for the shaders
    list_hlsl_headers("${OMM_HLSL_FILES}" OMM_HEADER_FILES)
    list_hlsl_headers("${OMM_HLSL_INCLUDE}" OMM_HEADER_FILES)
    list_hlsl_shaders("${OMM_HLSL_FILES}" "${OMM_HEADER_FILES}" OMM_SHADER_FILES)
    add_custom_target(Shaders ALL DEPENDS ${OMM_SHADER_FILES} SOURCES "${OMM_HEADER_FILES}")
    add_dependencies(Shaders CreateFolderForShaders)
    set_property(TARGET Shaders PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/Shaders") 
endif()

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(OMM_IS_SUBMODULE OFF)
else()
    set(OMM_IS_SUBMODULE ON)
endif()

# list C/C++ files and add them to OMMBake project
file(GLOB OMM_SOURCE "src/*.cpp" "src/*.h" "src/*.hpp")
source_group("Sources" FILES ${OMM_SOURCE})
file(GLOB OMM_RESOURCE "src/*.rc")
source_group("Sources" FILES ${OMM_RESOURCE})
file(GLOB OMM_HEADERS "include/*")
source_group("Include" FILES ${OMM_HEADERS})

set(TARGET_NAME omm-sdk)

if (OMM_STATIC_LIBRARY)
    add_library(${TARGET_NAME} STATIC ${OMM_SOURCE} ${OMM_RESOURCE} ${OMM_HEADERS})
else()
    add_library(${TARGET_NAME} SHARED ${OMM_SOURCE} ${OMM_RESOURCE} ${OMM_HEADERS})
endif()

if(OMM_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${TARGET_NAME} OpenMP::OpenMP_CXX)
    else()
        message(STATUS "OpenMP not found. OpenMP will be disabled")
    endif()
endif()

target_link_libraries(${TARGET_NAME} omm-shared glm stb_lib xxHash::xxhash) 

set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
target_include_directories(${TARGET_NAME} PUBLIC "include" "shaders")
set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER "${OMM_PROJECT_FOLDER}") 

if (OMM_ENABLE_PRECOMPILED_SHADERS)
    add_dependencies(${TARGET_NAME} Shaders)
endif()

find_package(Python COMPONENTS Interpreter)

if (Python_Interpreter_FOUND)

    add_custom_target(
     GenInterfaceH ALL
     COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/_interface_gen_c.py ${CMAKE_CURRENT_SOURCE_DIR}
     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/omm.json
     COMMENT "Generating Interface H"
    )

    add_custom_target(
     GenInterfaceHPP ALL
     COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/_interface_gen_cpp.py ${CMAKE_CURRENT_SOURCE_DIR}
     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/omm.json
     COMMENT "Generating Interface HPP"
    )
    
    set_property(TARGET GenInterfaceH PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/CustomScripts") 
    set_property(TARGET GenInterfaceHPP PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/CustomScripts") 

    add_dependencies(${TARGET_NAME} GenInterfaceH GenInterfaceHPP)

else()
    message(WARNING "Python not found. Interface generator will be disabled.")
endif()

if (OMM_ENABLE_PRECOMPILED_SHADERS)
    target_include_directories(${TARGET_NAME} PRIVATE "${OMM_SHADER_OUTPUT_PATH}")
endif()
if (OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
    target_compile_definitions(${TARGET_NAME} PRIVATE OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
endif()
if (OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
    target_compile_definitions(${TARGET_NAME} PRIVATE OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
endif()

if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS _UNICODE UNICODE)
else()
    # on non-Windows, FXC is not available and VulkanSDK DXC does not produce signed DXIL
    target_compile_definitions(${TARGET_NAME} PRIVATE OMM_ONLY_SPIRV_SHADERS_AVAILABLE=1)
endif()

if (OMM_STATIC_LIBRARY)
    target_compile_definitions(${TARGET_NAME} PUBLIC "OMM_API=extern \"C\"")
else()
    if(WIN32)
        target_compile_definitions(${TARGET_NAME} PRIVATE "OMM_API=extern \"C\" __declspec(dllexport)")
    else()
        target_compile_definitions(${TARGET_NAME} PRIVATE "OMM_API=extern \"C\" __attribute__((visibility(\"default\")))")
    endif()
endif()

if ((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    set(OMM_ARCHITECTURE_COMPILE_OPTIONS -msse4.1)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${TARGET_NAME} PRIVATE ${OMM_ARCHITECTURE_COMPILE_OPTIONS} -Wno-missing-braces -Wno-return-type-c-linkage)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(${TARGET_NAME} PRIVATE ${OMM_ARCHITECTURE_COMPILE_OPTIONS})
endif()

if (NOT OMM_IS_SUBMODULE)
    set(OMM_OUTPUT_BIN_PATH "${CMAKE_SOURCE_DIR}/Bin")
else()
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})
endif()

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}")

if (NOT ${OMM_OUTPUT_BIN_PATH} STREQUAL "")
    message(STATUS "OMMBake output path: '${OMM_OUTPUT_BIN_PATH}'")
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OMM_OUTPUT_BIN_PATH}/$<CONFIG>")
    set_target_properties(${TARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OMM_OUTPUT_BIN_PATH}/$<CONFIG>")
    set_target_properties(${TARGET_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${OMM_OUTPUT_BIN_PATH}/$<CONFIG>")
endif()

if (OMM_INSTALL)
    install (TARGETS ${TARGET_NAME}
             ARCHIVE DESTINATION lib
             LIBRARY DESTINATION lib
             RUNTIME DESTINATION bin)

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/omm.h" DESTINATION include)
endif()
