
option(NVRHI_INSTALL OFF)

# donut
if (OMM_BUILD_VIEWER)
    option(DONUT_WITH_ASSIMP "" OFF)
    set(DONUT_SHADERS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/framework")
	add_subdirectory(donut)

	set_property(TARGET donut_app PROPERTY FOLDER "Third-Party Libraries/Donut") 
	set_property(TARGET donut_core PROPERTY FOLDER "Third-Party Libraries/Donut") 
	set_property(TARGET donut_engine PROPERTY FOLDER "Third-Party Libraries/Donut") 
	set_property(TARGET donut_render PROPERTY FOLDER "Third-Party Libraries/Donut") 
	set_property(TARGET donut_shaders PROPERTY FOLDER "Third-Party Libraries/Donut") 
	set_property(TARGET update_mappings PROPERTY FOLDER "Third-Party Libraries/GLFW3") 

	set_property(TARGET nvrhi PROPERTY FOLDER "Third-Party Libraries") 
	set_property(TARGET nvrhi_d3d11 PROPERTY FOLDER "Third-Party Libraries") 
	set_property(TARGET nvrhi_d3d12 PROPERTY FOLDER "Third-Party Libraries") 
	set_property(TARGET nvrhi_vk PROPERTY FOLDER "Third-Party Libraries") 

endif()

# GLM
add_subdirectory("glm")

if(BUILD_STATIC_LIBS)
set_property(TARGET glm_static PROPERTY FOLDER "Third-Party Libraries") 
endif()

if(BUILD_SHARED_LIBS)
set_property(TARGET glm_shared PROPERTY FOLDER "Third-Party Libraries") 
endif()

# xxHash
set(XXHASH_BUILD_XXHSUM OFF)
add_subdirectory("xxHash/cmake_unofficial")
set_property(TARGET xxhash PROPERTY FOLDER "Third-Party Libraries") 

# stb
add_library(stb_lib INTERFACE)
target_include_directories(stb_lib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/stb>)

# lz4
if (NOT TARGET lz4)
	set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
	set(LZ4_BUILD_CLI OFF)
	set(LZ4_BUILD_LEGACY_LZ4C OFF)
	add_subdirectory("lz4/build/cmake")
	# set_property(TARGET lz4 PROPERTY FOLDER "Third-Party Libraries") 
endif()

# ShaderMake
if (NOT TARGET ShaderMake)
    add_subdirectory("ShaderMake")
endif()

if (TARGET ShaderMake)
    set_property(TARGET ShaderMake PROPERTY FOLDER "Third-Party Libraries") 
    set_property(TARGET ShaderMakeBlob PROPERTY FOLDER "Third-Party Libraries")
endif()

# nvrhi
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/nvrhi)
	if (NOT TARGET nvrhi)
		set(NVRHI_WITH_SHADER_COMPILER OFF CACHE BOOL "" FORCE)
		set(NVRHI_WITH_DX11 OFF CACHE BOOL "" FORCE)
		set(NVRHI_WITH_VULKAN OFF CACHE BOOL "" FORCE)
		add_subdirectory("nvrhi")

		if (TARGET nvrhi)
			set_target_properties(nvrhi PROPERTIES FOLDER "Third-Party Libraries")
		endif()

		if (TARGET nvrhi_d3d12)
			set_target_properties(nvrhi_d3d12 PROPERTIES FOLDER "Third-Party Libraries")
		endif()
	endif()
endif()

# gtest
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gtest AND OMM_ENABLE_TESTS)
	set(INSTALL_GTEST OFF)
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	add_subdirectory("gtest")
	set_target_properties("gmock" "gmock_main" "gtest_main" "gtest" PROPERTIES FOLDER "Third-Party Libraries")
endif()