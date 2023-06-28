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
add_library(stb_lib INTERFACE)
target_include_directories(stb_lib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/stb>)

# ShaderMake
if (NOT TARGET ShaderMake)
    add_subdirectory("ShaderMake")
    set_property(TARGET ShaderMake PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty") 
    set_property(TARGET ShaderMakeBlob PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")
endif()

# nvrhi
if(EXISTS nvrhi)
	if (NOT TARGET nvrhi)
		option(NVRHI_INSTALL OFF)
		set(NVRHI_WITH_SHADER_COMPILER OFF CACHE BOOL "" FORCE)
		set(NVRHI_WITH_DX11 OFF CACHE BOOL "" FORCE)
		set(NVRHI_WITH_VULKAN OFF CACHE BOOL "" FORCE)
		add_subdirectory("nvrhi")

		if (TARGET nvrhi)
			set_target_properties(nvrhi PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")
		endif()

		if (TARGET nvrhi_d3d12)
			set_target_properties(nvrhi_d3d12 PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")
		endif()
	endif()
endif()

# gtest
if(EXISTS gtest)
	set(INSTALL_GTEST OFF)
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	add_subdirectory("gtest")
	set_target_properties("gmock" "gmock_main" "gtest_main" "gtest" PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")
endif()

# gbenchmark
if(EXISTS gbenchmark)
	set(BENCHMARK_ENABLE_INSTALL OFF)
	set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
	add_subdirectory("gbenchmark")
	set_target_properties("benchmark" "benchmark_main" PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")
endif()
