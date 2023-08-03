
if (WIN32 AND TARGET nvrhi)
	option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" ON)
 	set(EXTRA_LIBS nvrhi_d3d12 d3d12.lib dxgi.lib)
else()
	option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" OFF)
 	set(EXTRA_LIBS )
endif()

if (OMM_INTEGRATION_LAYER_NVRHI)
    set(OMM_SDK_NVRHI_HEADERS omm-sdk-nvrhi/omm-sdk-nvrhi.h)
    set(OMM_SDK_NVRHI_SOURCE omm-sdk-nvrhi/omm-sdk-nvrhi.cpp)
    add_library(omm-sdk-nvrhi STATIC ${OMM_SDK_NVRHI_HEADERS} ${OMM_SDK_NVRHI_SOURCE})
    target_include_directories(omm-sdk-nvrhi PUBLIC omm-sdk-nvrhi)
    target_link_libraries(omm-sdk-nvrhi omm-sdk nvrhi ${EXTRA_LIBS})
    set_target_properties(omm-sdk-nvrhi PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Integration")
endif()

