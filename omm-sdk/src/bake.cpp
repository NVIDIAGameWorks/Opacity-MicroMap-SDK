/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "omm.h"
#include "bake_cpu_impl.h"
#include "bake_gpu_impl.h"
#include "debug_impl.h"
#include "texture_impl.h"
#include "version.h"

#include <array>

using namespace omm;

namespace
{

static const ommLibraryDesc g_vmbakeLibraryDesc =
{
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_BUILD
};

ommBakerType GetBakerType(ommBaker baker) {
    return static_cast<ommBakerType>(baker & 0x7);
}

template<class T>
T* GetBakerImpl(ommBaker baker) {
    return reinterpret_cast<T*>(baker & 0xFFFFFFFFFFFFFFF8);
}

template<class T>
ommBaker CreateHandle(ommBakerType type, T * impl) {
    return (uintptr_t)(impl) | (uintptr_t)type;
}

} // namespace

OMM_API ommLibraryDesc OMM_CALL ommGetLibraryDesc()
{
    static_assert(VERSION_MAJOR == OMM_VERSION_MAJOR);
    static_assert(VERSION_MINOR == OMM_VERSION_MINOR);
    static_assert(VERSION_BUILD == OMM_VERSION_BUILD);
    return g_vmbakeLibraryDesc;
}

OMM_API ommResult OMM_CALL ommCpuCreateTexture(ommBaker baker, const ommCpuTextureDesc* desc, ommCpuTexture* outTexture)
{
    if (baker == 0)
        return ommResult_INVALID_ARGUMENT;
    if (desc == 0)
        return ommResult_INVALID_ARGUMENT;
    if (GetBakerType(baker) != ommBakerType_CPU)
        return ommResult_INVALID_ARGUMENT;

    Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
    StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();

    TextureImpl* implementation = Allocate<TextureImpl>(memoryAllocator, memoryAllocator);
    const ommResult result = implementation->Create(*desc);

    if (result == ommResult_SUCCESS)
    {
        *outTexture = (ommCpuTexture)implementation;
        return ommResult_SUCCESS;
    }

    Deallocate(memoryAllocator, implementation);
    return result;
}

OMM_API ommResult OMM_CALL ommCpuDestroyTexture(ommBaker baker, ommCpuTexture texture)
{
    if (texture == 0)
        return ommResult_INVALID_ARGUMENT;
    if (GetBakerType(baker) != ommBakerType_CPU)
        return ommResult_INVALID_ARGUMENT;

    Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
    StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();

    Deallocate(memoryAllocator, (TextureImpl*)texture);

    return ommResult_SUCCESS;
}

OMM_API ommResult OMM_CALL ommCpuBake(ommBaker baker, const ommCpuBakeInputDesc* bakeInputDesc, ommCpuBakeResult* bakeResult)
{
    if (baker == 0)
        return ommResult_INVALID_ARGUMENT;
    if (bakeInputDesc == 0)
        return ommResult_INVALID_ARGUMENT;
    if (GetBakerType(baker) != ommBakerType_CPU)
        return ommResult_INVALID_ARGUMENT;

    Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
    return (*impl).BakeOpacityMicromap(*bakeInputDesc, bakeResult);
}

OMM_API ommResult OMM_CALL ommCpuDestroyBakeResult(ommCpuBakeResult bakeResult)
{
    if (bakeResult == 0)
        return ommResult_INVALID_ARGUMENT;

    StdAllocator<uint8_t>& memoryAllocator = (*(omm::Cpu::BakeOutputImpl*)bakeResult).GetStdAllocator();
    Deallocate(memoryAllocator, (omm::Cpu::BakeOutputImpl*)bakeResult);

    return ommResult_SUCCESS;
}

OMM_API ommResult OMM_CALL ommCpuGetBakeResultDesc(ommCpuBakeResult bakeResult, const ommCpuBakeResultDesc** desc)
{
    if (bakeResult == 0)
        return ommResult_INVALID_ARGUMENT;

    return (*(omm::Cpu::BakeOutputImpl*)bakeResult).GetBakeResultDesc(desc);
}

OMM_API ommResult OMM_CALL ommGpuGetStaticResourceData(ommGpuResourceType resource, uint8_t* data, size_t* outByteSize)
{
    return Gpu::OmmStaticBuffers::GetStaticResourceData(resource, data, outByteSize);
}

///< The Pipeline determines the 
OMM_API ommResult OMM_CALL ommGpuCreatePipeline(ommBaker baker, const ommGpuBakePipelineConfigDesc* config, ommGpuPipeline* outPipeline)
{
    if (baker == 0)
        return ommResult_INVALID_ARGUMENT;
    if (config == 0)
        return ommResult_INVALID_ARGUMENT;
    if (GetBakerType(baker) != ommBakerType_GPU)
        return ommResult_INVALID_ARGUMENT;
    Gpu::BakerImpl* bakePtr = GetBakerImpl<Gpu::BakerImpl>(baker);
    return bakePtr->CreatePipeline(*config, outPipeline);
}

OMM_API ommResult OMM_CALL ommGpuGetPipelineDesc(ommGpuPipeline pipeline, const ommGpuBakePipelineInfoDesc** outPipelineDesc)
{
    if (pipeline == 0)
        return ommResult_INVALID_ARGUMENT;
    Gpu::PipelineImpl* impl = (Gpu::PipelineImpl*)(pipeline);
    return impl->GetPipelineDesc(outPipelineDesc);
}

OMM_API ommResult OMM_CALL ommGpuDestroyPipeline(ommBaker baker, ommGpuPipeline pipeline)
{
    if (pipeline == 0)
        return ommResult_INVALID_ARGUMENT;
    if (baker == 0)
        return ommResult_INVALID_ARGUMENT;
    if (GetBakerType(baker) != ommBakerType_GPU)
        return ommResult_INVALID_ARGUMENT;
    Gpu::BakerImpl* bakePtr = GetBakerImpl<Gpu::BakerImpl>(baker);
    return bakePtr->DestroyPipeline(pipeline);
}

OMM_API ommResult OMM_CALL ommGpuGetPreBakeInfo(ommGpuPipeline pipeline, const ommGpuBakeDispatchConfigDesc* config, ommGpuPreBakeInfo* outPreBuildInfo)
{
    if (pipeline == 0)
        return ommResult_INVALID_ARGUMENT;
    if (config == 0)
        return ommResult_INVALID_ARGUMENT;
    Gpu::PipelineImpl* impl = (Gpu::PipelineImpl*)(pipeline);
    return impl->GetPreBakeInfo(*config, outPreBuildInfo);
}

OMM_API ommResult OMM_CALL ommGpuBake(ommGpuPipeline pipeline, const ommGpuBakeDispatchConfigDesc* dispatchConfig, const ommGpuBakeDispatchChain** outDispatchDesc)
{
    if (pipeline == 0)
        return ommResult_INVALID_ARGUMENT;
    if (dispatchConfig == 0)
        return ommResult_INVALID_ARGUMENT;
    Gpu::PipelineImpl* impl = (Gpu::PipelineImpl*)(pipeline);
    return impl->GetDispatcheDesc(*dispatchConfig, outDispatchDesc);
}
    
OMM_API ommResult OMM_CALL ommDebugSaveAsImages(ommBaker baker, const ommCpuBakeInputDesc* bakeInputDesc, const ommCpuBakeResultDesc* res, const ommDebugSaveImagesDesc* desc)
{
    if (baker == 0)
        return ommResult_INVALID_ARGUMENT;
    if (bakeInputDesc == 0)
        return ommResult_INVALID_ARGUMENT;
    if (desc == 0)
        return ommResult_INVALID_ARGUMENT;
    if (GetBakerType(baker) == ommBakerType_CPU)
    {
        Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
        StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();
        return SaveAsImagesImpl(memoryAllocator, *bakeInputDesc, res, *desc);
    }
    else if (GetBakerType(baker) == ommBakerType_GPU)
    {
        Gpu::BakerImpl* impl = GetBakerImpl<Gpu::BakerImpl>(baker);
        StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();
        return SaveAsImagesImpl(memoryAllocator, *bakeInputDesc, res, *desc);
    }
    else
        return ommResult_INVALID_ARGUMENT;
}

OMM_API ommResult OMM_CALL ommDebugGetStats(ommBaker baker, const ommCpuBakeResultDesc* res, ommDebugStats* out)
{
    if (baker == 0)
        return ommResult_INVALID_ARGUMENT;

    if (GetBakerType(baker) == ommBakerType_CPU)
    {
        Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
        StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();
        return GetStatsImpl(memoryAllocator, res, out);
    }
    else if (GetBakerType(baker) == ommBakerType_GPU)
    {
        Gpu::BakerImpl* impl = GetBakerImpl<Gpu::BakerImpl>(baker);
        StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();
        return GetStatsImpl(memoryAllocator, res, out);
    }
    else
        return ommResult_INVALID_ARGUMENT;
}

OMM_API ommResult OMM_CALL ommCreateBaker(const ommBakerCreationDesc* desc, ommBaker* baker)
{
    if (desc == 0)
        return ommResult_INVALID_ARGUMENT;

    StdMemoryAllocatorInterface o = 
    {
        .Allocate = desc->memoryAllocatorInterface.Allocate,
        .Reallocate = desc->memoryAllocatorInterface.Reallocate,
        .Free = desc->memoryAllocatorInterface.Free,
        .UserArg = desc->memoryAllocatorInterface.UserArg
    };

    CheckAndSetDefaultAllocator(o);
    StdAllocator<uint8_t> memoryAllocator(o);

    if (desc->type == ommBakerType_CPU)
    {
        Cpu::BakerImpl* implementation = Allocate<Cpu::BakerImpl>(memoryAllocator, memoryAllocator);
        const ommResult result = implementation->Create(*desc);

        if (result == ommResult_SUCCESS)
        {
            *baker = CreateHandle(desc->type, implementation);
            return ommResult_SUCCESS;
        }

        Deallocate(memoryAllocator, implementation);
    } 
    else if (desc->type == ommBakerType_GPU)
    {
        Gpu::BakerImpl* implementation = Allocate<Gpu::BakerImpl>(memoryAllocator, memoryAllocator);
        const ommResult result = implementation->Create(*desc);

        if (result == ommResult_SUCCESS)
        {
            *baker = CreateHandle(desc->type, implementation);
            return ommResult_SUCCESS;
        }

        Deallocate(memoryAllocator, implementation);
    }
    else {
        return ommResult_INVALID_ARGUMENT;
    }
   
    return ommResult_FAILURE;
}

OMM_API ommResult OMM_CALL ommDestroyBaker(ommBaker baker)
{
    if (baker == 0)
        return ommResult_INVALID_ARGUMENT;

    const ommBakerType type = GetBakerType(baker);
    if (type == ommBakerType_CPU)
    {
        Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
        StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();
        Deallocate(memoryAllocator, impl);
        return ommResult_SUCCESS;
    }
    else  if (type == ommBakerType_GPU)
    {
        Gpu::BakerImpl* bakePtr = GetBakerImpl<Gpu::BakerImpl>(baker);
        StdAllocator<uint8_t>& memoryAllocator = (*bakePtr).GetStdAllocator();
        Deallocate(memoryAllocator, bakePtr);
        return ommResult_SUCCESS;
    }
    return ommResult_FAILURE;
}