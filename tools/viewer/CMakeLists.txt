#
# Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


include(../../external/donut/compileshaders.cmake)
file(GLOB shaders "*.hlsl")
file(GLOB sources "*.cpp" "*.h")

set(project viewer)
set(folder "Tools/Viewer")

option(OMM_VIEWER_INSTALL "Generate install rules for OMM Viewer" ON)

set(output_base ${CMAKE_CURRENT_BINARY_DIR}/compiled_shaders)

donut_compile_shaders(
    TARGET ${project}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    SOURCES ${shaders}
    FOLDER ${folder}
    DXBC ${output_base}
    DXIL ${output_base}
    SPIRV_DXC ${output_base}
    OUTPUT_FORMAT "HEADER"
    SHADERMAKE_OPTIONS_SPIRV --spirvExt=SPV_EXT_fragment_shader_interlock
)

add_executable(${project} WIN32 ${sources})
target_link_libraries(${project} donut_app donut_engine donut_render omm-lib)
target_include_directories(${project} PRIVATE "${CMAKE_SOURCE_DIR}/external/imgui-filebrowser" "${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(${project} ${project}_shaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})
target_compile_definitions(${project} PRIVATE OMM_VIEWER_DEFAULT_BINARY_FOLDER="${PROJECT_SOURCE_DIR}/assets/omm_example_data")

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()

if (OMM_VIEWER_INSTALL)
    install(TARGETS ${project} ${INSTALL_EXPORT} RUNTIME DESTINATION bin )
endif()