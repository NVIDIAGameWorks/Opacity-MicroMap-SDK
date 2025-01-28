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

donut_compile_shaders(
    TARGET ${project}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    SOURCES ${shaders}
    FOLDER ${folder}
    DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${project}/dxil
    SPIRV_DXC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${project}/spirv
    SHADERMAKE_OPTIONS_SPIRV --spirvExt=SPV_EXT_fragment_shader_interlock
)

add_executable(${project} WIN32 ${sources})
target_link_libraries(${project} donut_app donut_engine donut_render omm-lib)
target_include_directories(${project} PRIVATE "${CMAKE_SOURCE_DIR}/external/imgui-filebrowser")
add_dependencies(${project} ${project}_shaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})
target_compile_definitions(${project} PRIVATE OMM_VIEWER_DEFAULT_BINARY_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/example_data")

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()

#install(TARGETS ${project} ${INSTALL_EXPORT} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} )
#install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${project}/dxil DESTINATION ${CMAKE_INSTALL_BINDIR}/../shaders/${project} )
#install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/framework/dxil DESTINATION ${CMAKE_INSTALL_BINDIR}/../shaders/framework )
