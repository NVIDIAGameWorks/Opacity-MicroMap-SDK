/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "omm.h"
#include "defines.h"
#include "std_containers.h"
#include "texture_impl.h"

#include <shared/math.h>
#include <shared/texture.h>

#include <map>
#include <set>

typedef omm::MemoryAllocatorInterface MemoryAllocatorInterface;
#include "std_allocator.h"

namespace omm
{
namespace Cpu
{
    class BakerImpl
    {
    // Internal
    public:
        inline BakerImpl(const StdAllocator<uint8_t>& stdAllocator) :
            m_stdAllocator(stdAllocator)
        {}

        ~BakerImpl();

        inline StdAllocator<uint8_t>& GetStdAllocator()
        { return m_stdAllocator; }

        Result Create(const BakerCreationDesc& bakeCreationDesc);
        Result BakeOpacityMicromap(const Cpu::BakeInputDesc& bakeInputDesc, Cpu::BakeResult* bakeOutput);

    private:
        Result Validate(const Cpu::BakeInputDesc& desc);
    private:
        StdAllocator<uint8_t> m_stdAllocator;
    };

    struct BakeResultImpl
    {
        Cpu::BakeResultDesc bakeOutputDesc;
        vector<int32_t> ommIndexBuffer;
        vector<OpacityMicromapDesc> ommDescArray;
        vector<uint8_t> ommArrayData;
        vector<OpacityMicromapUsageCount> ommArrayHistogram;
        vector<OpacityMicromapUsageCount> ommIndexHistogram;

        BakeResultImpl(const StdAllocator<uint8_t>& stdAllocator) :
            ommIndexBuffer(stdAllocator),
            ommDescArray(stdAllocator),
            ommArrayData(stdAllocator),
            ommArrayHistogram(stdAllocator),
            ommIndexHistogram(stdAllocator)
        {
        }

        void Finalize(IndexFormat ommIndexFormat)
        {
            bakeOutputDesc.ommArrayData                 = ommArrayData.data();
            bakeOutputDesc.ommArrayDataSize             = (uint32_t)ommArrayData.size();
            bakeOutputDesc.ommDescArray                 = ommDescArray.data();
            bakeOutputDesc.ommDescArrayCount            = (uint32_t)ommDescArray.size();
            bakeOutputDesc.ommDescArrayHistogram        = ommArrayHistogram.data();
            bakeOutputDesc.ommDescArrayHistogramCount   = (uint32_t)ommArrayHistogram.size();
            bakeOutputDesc.ommIndexBuffer               = ommIndexBuffer.data();
            bakeOutputDesc.ommIndexCount                = (uint32_t)ommIndexBuffer.size();
            bakeOutputDesc.ommIndexFormat               = ommIndexFormat;
            bakeOutputDesc.ommIndexHistogram            = ommIndexHistogram.data();
            bakeOutputDesc.ommIndexHistogramCount       = (uint32_t)ommIndexHistogram.size();
        }
    };

    class BakeOutputImpl
    {
    public:
        BakeOutputImpl(const StdAllocator<uint8_t>& stdAllocator);
        ~BakeOutputImpl();

        inline StdAllocator<uint8_t>& GetStdAllocator()
        {
            return m_stdAllocator;
        }

        inline const Cpu::BakeResultDesc& GetBakeOutputDesc() const
        {
            return m_bakeResult.bakeOutputDesc;
        }

        inline Result GetBakeResultDesc(const Cpu::BakeResultDesc*& desc)
        {
            desc = &m_bakeResult.bakeOutputDesc;
            return Result::SUCCESS;
        }

        Result Bake(const Cpu::BakeInputDesc& desc);

    private:
        static Result ValidateDesc(const BakeInputDesc& desc);

        template<TilingMode eTextureFormat, TextureAddressMode eTextureAddressMode, TextureFilterMode eFilterMode>
        Result BakeImpl(const Cpu::BakeInputDesc& desc);

        template<class... TArgs>
        void RegisterDispatch(TArgs... args, std::function < Result(const Cpu::BakeInputDesc& desc)> fn);
        map<std::tuple<TilingMode, TextureAddressMode, TextureFilterMode>, std::function<Result(const Cpu::BakeInputDesc& desc)>> bakeDispatchTable;
        Result InvokeDispatch(const Cpu::BakeInputDesc& desc);
    private:
        StdAllocator<uint8_t> m_stdAllocator;
        Cpu::BakeInputDesc m_bakeInputDesc;
        BakeResultImpl m_bakeResult;
    };
} // namespace Cpu
} // namespace omm
