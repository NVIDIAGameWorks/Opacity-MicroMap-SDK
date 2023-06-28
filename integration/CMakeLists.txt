
if (WIN32)
	option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" ON)
else()
	option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" OFF)
endif()

if (OMM_INTEGRATION_LAYER_NVRHI)
    set(OMM_SDK_NVRHI_HEADERS omm-sdk-nvrhi/omm-sdk-nvrhi.h)
    set(OMM_SDK_NVRHI_SOURCE omm-sdk-nvrhi/omm-sdk-nvrhi.cpp)
    add_library(omm-sdk-nvrhi STATIC ${OMM_SDK_NVRHI_HEADERS} ${OMM_SDK_NVRHI_SOURCE})
    target_include_directories(omm-sdk-nvrhi PUBLIC omm-sdk-nvrhi)
    target_link_libraries(omm-sdk-nvrhi omm-sdk nvrhi nvrhi_d3d12 d3d12.lib dxgi.lib)
    set_target_properties(omm-sdk-nvrhi PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Integration")

    if (TARGET nvrhi)
        target_compile_definitions(omm-sdk-nvrhi PRIVATE OMM_NVRHI_AVAILABLE)
    endif()
endif()

