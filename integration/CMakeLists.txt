
if (WIN32 AND TARGET nvrhi)
	option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" ON)
 	set(EXTRA_LIBS nvrhi_d3d12 d3d12.lib dxgi.lib)
else()
	option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" OFF)
 	set(EXTRA_LIBS )
endif()

if (OMM_INTEGRATION_LAYER_NVRHI)
    set(OMM_LIB_NVRHI_HEADERS omm-lib-nvrhi/omm-lib-nvrhi.h)
    set(OMM_LIB_NVRHI_SOURCE omm-lib-nvrhi/omm-lib-nvrhi.cpp)
    add_library(omm-lib-nvrhi STATIC ${OMM_LIB_NVRHI_HEADERS} ${OMM_LIB_NVRHI_SOURCE})
    target_include_directories(omm-lib-nvrhi PUBLIC omm-lib-nvrhi)
    target_link_libraries(omm-lib-nvrhi omm-lib nvrhi ${EXTRA_LIBS})
    set_target_properties(omm-lib-nvrhi PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Integration")
endif()

