# GLM
add_subdirectory("glm")

if(BUILD_STATIC_LIBS)
set_property(TARGET glm_static PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty") 
endif()

if(BUILD_SHARED_LIBS)
set_property(TARGET glm_shared PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty") 
endif()

# xxHash
set(XXHASH_BUILD_XXHSUM OFF)
add_subdirectory("xxHash/cmake_unofficial")
set_property(TARGET xxhash PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty") 

# stb
add_library(stb_lib STATIC "stb_lib.cpp" "stb")
target_include_directories(stb_lib INTERFACE "stb")
set_property(TARGET stb_lib PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty") 

# nvrhi
if (NOT TARGET nvrhi)
	option(NVRHI_INSTALL OFF)
	set(NVRHI_WITH_SHADER_COMPILER OFF CACHE BOOL "" FORCE)
	set(NVRHI_WITH_DX11 OFF CACHE BOOL "" FORCE)
	set(NVRHI_WITH_VULKAN OFF CACHE BOOL "" FORCE)
	add_subdirectory("nvrhi")
	set_target_properties(nvrhi PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")
	set_target_properties(nvrhi_d3d12 PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")
endif()

# gtest
set(INSTALL_GTEST OFF)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory("gtest")
set_target_properties("gmock" "gmock_main" "gtest_main" "gtest" PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")