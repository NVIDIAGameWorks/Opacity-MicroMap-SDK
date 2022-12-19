
option(OMM_INTEGRATION_LAYER_NVRHI "Build nvrhi integration layer" ON)

if (OMM_INTEGRATION_LAYER_NVRHI AND NOT OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
    message(WARNING "Embedded DXIL shaders is required for NVRHI integration layer. Integration layer will be disabled (OMM_INTEGRATION_LAYER_NVRHI). To resolve make sure OMM_ENABLE_PRECOMPILED_SHADERS_DXIL is ON.")
    set(OMM_ENABLE_GPU_TESTS OFF CACHE BOOL "" FORCE)
endif()

if (OMM_INTEGRATION_LAYER_NVRHI)
    set(OMM_SDK_NVRHI_HEADERS omm-sdk-nvrhi/omm-sdk-nvrhi/NVRHIOmmBakeIntegration.h omm-sdk-nvrhi/omm-sdk-nvrhi/NVRHIWrapper.h)
    set(OMM_SDK_NVRHI_SOURCE omm-sdk-nvrhi/NVRHIOmmBakeIntegration.cpp omm-sdk-nvrhi/NVRHIWrapper.cpp)
    add_library(omm-sdk-nvrhi STATIC ${OMM_SDK_NVRHI_HEADERS} ${OMM_SDK_NVRHI_SOURCE})
    target_include_directories(omm-sdk-nvrhi PUBLIC omm-sdk-nvrhi)
    target_link_libraries(omm-sdk-nvrhi omm-sdk nvrhi nvrhi_d3d12 d3d12.lib dxgi.lib)
    set_target_properties(omm-sdk-nvrhi PROPERTIES FOLDER "${OMM_PROJECT_FOLDER}/Integration")
endif()

