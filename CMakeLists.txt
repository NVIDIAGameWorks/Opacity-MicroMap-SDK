cmake_minimum_required(VERSION 3.12)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/gtest)
	option(OMM_ENABLE_TESTS "Enable unit test" ON)
endif()

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/donut AND WIN32)
	option(OMM_BUILD_VIEWER "Build omm viewer tool" ON)
else()
	option(OMM_BUILD_VIEWER "Build omm viewer tool" OFF)
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/libraries/omm-lib/src/version.h" ver_h)
string(REGEX MATCH "VERSION_MAJOR ([0-9]*)" _ ${ver_h})
set(ver_major ${CMAKE_MATCH_1})
string(REGEX MATCH "VERSION_MINOR ([0-9]*)" _ ${ver_h})
set(ver_minor ${CMAKE_MATCH_1})
string(REGEX MATCH "VERSION_BUILD ([0-9]*)" _ ${ver_h})
set(ver_patch ${CMAKE_MATCH_1})

project("Opacity Micro-Map SDK" VERSION ${ver_major}.${ver_minor}.${ver_patch} LANGUAGES CXX)
message(STATUS "${PROJECT_NAME} version: ${ver_major}.${ver_minor}.${ver_patch}")

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
message(STATUS "${CMAKE_BINARY_DIR}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # Windows: if install locations not set by user, set install prefix to "<build_dir>\install".
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
   
if (NOT DEFINED OMM_PROJECT_FOLDER)
    set(OMM_PROJECT_FOLDER "Libraries")
endif()

if (OMM_BUILD_VIEWER)
    add_subdirectory(tools/viewer)
endif()

add_subdirectory(external)
add_subdirectory(libraries/omm-util)
add_subdirectory(libraries/omm-lib)
add_subdirectory(libraries/omm-gpu-nvrhi)

if (OMM_ENABLE_TESTS)
    add_subdirectory(support/tests)
endif()
