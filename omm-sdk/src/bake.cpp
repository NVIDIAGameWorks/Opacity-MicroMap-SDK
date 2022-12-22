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

static const LibraryDesc g_vmbakeLibraryDesc =
{
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_BUILD
};

OMM_API const LibraryDesc& OMM_CALL omm::GetLibraryDesc()
{
    static_assert(VERSION_MAJOR == OMM_VERSION_MAJOR);
    static_assert(VERSION_MINOR == OMM_VERSION_MINOR);
    static_assert(VERSION_BUILD == OMM_VERSION_BUILD);
    return g_vmbakeLibraryDesc;
}

namespace omm
{

BakerType GetBakerType(Baker baker) {
    return static_cast<BakerType>(baker & 0x7);
}

template<class T>
T* GetBakerImpl(Baker baker) {
    return reinterpret_cast<T*>(baker & 0xFFFFFFFFFFFFFFF8);
}

template<class T>
Baker CreateHandle(BakerType type, T* impl) {
    return (uintptr_t)(impl) | (uintptr_t)type;
}

namespace Cpu
{
    OMM_API Result OMM_CALL CreateTexture(Baker baker, const TextureDesc& desc, Texture* outTexture)
    {
        if (baker == 0)
            return Result::INVALID_ARGUMENT;
        if (GetBakerType(baker) != BakerType::CPU)
            return Result::INVALID_ARGUMENT;

        Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
        StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();

        TextureImpl* implementation = Allocate<TextureImpl>(memoryAllocator, memoryAllocator);
        const Result result = implementation->Create(desc);

        if (result == Result::SUCCESS)
        {
            *outTexture = (Texture)implementation;
            return Result::SUCCESS;
        }

        Deallocate(memoryAllocator, implementation);
        return result;
    }

    OMM_API Result OMM_CALL DestroyTexture(Baker baker, Texture texture)
    {
        if (texture == 0)
            return Result::INVALID_ARGUMENT;
        if (GetBakerType(baker) != BakerType::CPU)
            return Result::INVALID_ARGUMENT;

        Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
        StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();

        Deallocate(memoryAllocator, (TextureImpl*)texture);

        return Result::SUCCESS;
    }

    OMM_API Result OMM_CALL BakeOpacityMicromap(Baker baker, const BakeInputDesc& bakeInputDesc, BakeResult* bakeResult)
    {
        if (baker == 0)
            return Result::INVALID_ARGUMENT;
        if (GetBakerType(baker) != BakerType::CPU)
            return Result::INVALID_ARGUMENT;

        Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
        return (*impl).BakeOpacityMicromap(bakeInputDesc, bakeResult);
    }

    OMM_API Result OMM_CALL DestroyBakeResult(BakeResult bakeResult)
    {
        if (bakeResult == 0)
            return Result::INVALID_ARGUMENT;

        StdAllocator<uint8_t>& memoryAllocator = (*(BakeOutputImpl*)bakeResult).GetStdAllocator();
        Deallocate(memoryAllocator, (BakeOutputImpl*)bakeResult);

        return Result::SUCCESS;
    }

    OMM_API Result OMM_CALL GetBakeResultDesc(BakeResult bakeResult, const Cpu::BakeResultDesc*& desc)
    {
        if (bakeResult == 0)
            return Result::INVALID_ARGUMENT;

        return (*(BakeOutputImpl*)bakeResult).GetBakeResultDesc(desc);
    }
} // namespace Cpu

namespace Gpu
{
    OMM_API Result OMM_CALL GetStaticResourceData(ResourceType resource, uint8_t* data, size_t& byteSize)
    {
        return Gpu::OmmStaticBuffers::GetStaticResourceData(resource, data, byteSize);
    }

    ///< The Pipeline determines the 
    OMM_API Result OMM_CALL CreatePipeline(Baker baker, const BakePipelineConfigDesc& config, Pipeline* outPipeline)
    {
        if (baker == 0)
            return Result::INVALID_ARGUMENT;
        if (GetBakerType(baker) != BakerType::GPU)
            return Result::INVALID_ARGUMENT;
        Gpu::BakerImpl* bakePtr = GetBakerImpl<Gpu::BakerImpl>(baker);
        return bakePtr->CreatePipeline(config, outPipeline);
    }

    OMM_API Result OMM_CALL GetPipelineDesc(Pipeline pipeline, const BakePipelineInfoDesc*& outPipelineDesc)
    {
        if (pipeline == 0)
            return Result::INVALID_ARGUMENT;
        Gpu::PipelineImpl* impl = (Gpu::PipelineImpl*)(pipeline);
        return impl->GetPipelineDesc(outPipelineDesc);
    }

    OMM_API Result OMM_CALL DestroyPipeline(Baker baker, Pipeline pipeline)
    {
        if (pipeline == 0)
            return Result::INVALID_ARGUMENT;
        if (baker == 0)
            return Result::INVALID_ARGUMENT;
        if (GetBakerType(baker) != BakerType::GPU)
            return Result::INVALID_ARGUMENT;
        Gpu::BakerImpl* bakePtr = GetBakerImpl<Gpu::BakerImpl>(baker);
        return bakePtr->DestroyPipeline(pipeline);
    }

    OMM_API Result OMM_CALL GetPreBakeInfo(Pipeline pipeline, const BakeDispatchConfigDesc& config, PreBakeInfo* outPreBuildInfo)
    {
        if (pipeline == 0)
            return Result::INVALID_ARGUMENT;
        Gpu::PipelineImpl* impl = (Gpu::PipelineImpl*)(pipeline);
        return impl->GetPreBakeInfo(config, outPreBuildInfo);
    }

    OMM_API Result OMM_CALL BakePrepass(Pipeline pipeline, const BakeDispatchConfigDesc& config, const BakeDispatchChain*& outDispatchDesc)
    {
        return Result::NOT_IMPLEMENTED;
    }

    OMM_API Result OMM_CALL Bake(Pipeline pipeline, const BakeDispatchConfigDesc& dispatchConfig, const BakeDispatchChain*& outDispatchDesc)
    {
        if (pipeline == 0)
            return Result::INVALID_ARGUMENT;
        Gpu::PipelineImpl* impl = (Gpu::PipelineImpl*)(pipeline);
        return impl->GetDispatcheDesc(dispatchConfig, outDispatchDesc);
    }
    
} // namespace Gpu

namespace Debug
{
    OMM_API Result OMM_CALL SaveAsImages(Baker baker, const Cpu::BakeInputDesc& bakeInputDesc, const Cpu::BakeResultDesc* res, const Debug::SaveImagesDesc& desc)
    {
        if (baker == 0)
            return Result::INVALID_ARGUMENT;

        if (GetBakerType(baker) == BakerType::CPU)
        {
            Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
            StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();
            return SaveAsImagesImpl(memoryAllocator, bakeInputDesc, res, desc);
        }
        else if (GetBakerType(baker) == BakerType::GPU)
        {
            Gpu::BakerImpl* impl = GetBakerImpl<Gpu::BakerImpl>(baker);
            StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();
            return SaveAsImagesImpl(memoryAllocator, bakeInputDesc, res, desc);
        }
        else
            return Result::INVALID_ARGUMENT;
    }

    OMM_API Result OMM_CALL GetStats(Baker baker, const Cpu::BakeResultDesc* res, Stats* out)
    {
        if (baker == 0)
            return Result::INVALID_ARGUMENT;

        if (GetBakerType(baker) == BakerType::CPU)
        {
            Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
            StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();
            return GetStatsImpl(memoryAllocator, res, out);
        }
        else if (GetBakerType(baker) == BakerType::GPU)
        {
            Gpu::BakerImpl* impl = GetBakerImpl<Gpu::BakerImpl>(baker);
            StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();
            return GetStatsImpl(memoryAllocator, res, out);
        }
        else
            return Result::INVALID_ARGUMENT;
    }
} // namespace Debug

OMM_API Result OMM_CALL CreateOpacityMicromapBaker(const BakerCreationDesc& desc, Baker* baker)
{
    StdMemoryAllocatorInterface o = 
    {
        .Allocate = desc.memoryAllocatorInterface.Allocate,
        .Reallocate = desc.memoryAllocatorInterface.Reallocate,
        .Free = desc.memoryAllocatorInterface.Free,
        .userArg = desc.memoryAllocatorInterface.userArg
    };

    CheckAndSetDefaultAllocator(o);
    StdAllocator<uint8_t> memoryAllocator(o);

    if (desc.type == BakerType::CPU)
    {
        Cpu::BakerImpl* implementation = Allocate<Cpu::BakerImpl>(memoryAllocator, memoryAllocator);
        const Result result = implementation->Create(desc);

        if (result == Result::SUCCESS)
        {
            *baker = CreateHandle(desc.type, implementation);
            return Result::SUCCESS;
        }

        Deallocate(memoryAllocator, implementation);
    } 
    else if (desc.type == BakerType::GPU)
    {
        Gpu::BakerImpl* implementation = Allocate<Gpu::BakerImpl>(memoryAllocator, memoryAllocator);
        const Result result = implementation->Create(desc);

        if (result == Result::SUCCESS)
        {
            *baker = CreateHandle(desc.type, implementation);
            return Result::SUCCESS;
        }

        Deallocate(memoryAllocator, implementation);
    }
    else {
        return Result::INVALID_ARGUMENT;
    }
   
    return Result::FAILURE;
}

OMM_API Result OMM_CALL DestroyOpacityMicromapBaker(Baker baker)
{
    if (baker == 0)
        return Result::INVALID_ARGUMENT;

    const BakerType type = GetBakerType(baker);
    if (type == BakerType::CPU)
    {
        Cpu::BakerImpl* impl = GetBakerImpl<Cpu::BakerImpl>(baker);
        StdAllocator<uint8_t>& memoryAllocator = (*impl).GetStdAllocator();
        Deallocate(memoryAllocator, impl);
        return Result::SUCCESS;
    }
    else  if (type == BakerType::GPU)
    {
        Gpu::BakerImpl* bakePtr = GetBakerImpl<Gpu::BakerImpl>(baker);
        StdAllocator<uint8_t>& memoryAllocator = (*bakePtr).GetStdAllocator();
        Deallocate(memoryAllocator, bakePtr);
        return Result::SUCCESS;
    }
    return Result::FAILURE;
}

} // namespace omm