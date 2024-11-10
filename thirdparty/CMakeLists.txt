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

# gbenchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory("benchmark")

# stb
add_library(stb_lib INTERFACE)
target_include_directories(stb_lib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/stb>)

# lz4
if (NOT TARGET lz4)
	set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
	set(LZ4_BUILD_CLI OFF)
	set(LZ4_BUILD_LEGACY_LZ4C OFF)
	add_subdirectory("lz4/build/cmake")
	set_property(TARGET lz4 PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty") 
endif()

# ShaderMake
if (NOT TARGET ShaderMake)
    add_subdirectory("ShaderMake")
    set_property(TARGET ShaderMake PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty") 
    set_property(TARGET ShaderMakeBlob PROPERTY FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")
endif()

# nvrhi
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/nvrhi)
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
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gtest AND OMM_ENABLE_TESTS)
	set(INSTALL_GTEST OFF)
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	add_subdirectory("gtest")
	set_target_properties("gmock" "gmock_main" "gtest_main" "gtest" PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Thirdparty")
endif()