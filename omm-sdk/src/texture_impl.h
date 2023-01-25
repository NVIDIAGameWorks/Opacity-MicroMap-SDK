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

#include "std_containers.h"

#include <shared/math.h>
#include <shared/assert.h>
#include <shared/bit_tricks.h>
#include <shared/texture.h>

namespace omm
{
    enum class TilingMode {
        Linear,
        MortonZ,
        MAX_NUM,
    };

    class TextureImpl
    {
    public:
        TextureImpl(const StdAllocator<uint8_t>& stdAllocator);
        ~TextureImpl();

        ommResult Create(const ommCpuTextureDesc& desc);

        template<TilingMode eTilingMode>
        float Load(const int2& texCoord, int32_t mip) const;

        float Load(const int2& texCoord, int32_t mip) const;

        float Bilinear(ommTextureAddressMode mode, const float2& p, int32_t mip) const;

        TilingMode GetTilingMode() const {
            return m_tilingMode;
        }

        int2 GetSize(int32_t mip) const {
            return m_mips[mip].size;
        }

        float2 GetRcpSize(int32_t mip) const {
            return m_mips[mip].rcpSize;
        }

        uint32_t GetMipCount() const {
            return (uint32_t)m_mips.size();
        }

    private:
        static ommResult Validate(const ommCpuTextureDesc& desc);
        void Deallocate();
        template<TilingMode eTilingMode>
        static uint64_t From2Dto1D(const int2& idx, const int2& size) {
            OMM_ASSERT(false && "Not implemented");
            return 0;
        }
    private:
        static constexpr uint2  kMaxDim = int2(65536);
        static constexpr size_t kAlignment = 64;

        StdAllocator<uint8_t> m_stdAllocator;

        struct Mips
        {
            int2 size;
            float2 rcpSize;
            int2 sizeMinusOne;
            uintptr_t dataOffset;
            size_t numElements;
        };

        vector<Mips> m_mips;
        TilingMode m_tilingMode;
        uint8_t* m_data;
        size_t m_dataSize;
    };

    template<TilingMode eTilingMode>
    float TextureImpl::Load(const int2& texCoord, int32_t mip) const
    {
        OMM_ASSERT(eTilingMode == m_tilingMode);
        OMM_ASSERT(texCoord.x >= 0);
        OMM_ASSERT(texCoord.y >= 0);
        OMM_ASSERT(texCoord.x < m_mips[mip].size.x);
        OMM_ASSERT(texCoord.y < m_mips[mip].size.y);
        OMM_ASSERT(glm::all(glm::notEqual(texCoord, kTexCoordBorder2)));
        OMM_ASSERT(glm::all(glm::notEqual(texCoord, kTexCoordInvalid2)));
        const uint64_t idx = From2Dto1D<eTilingMode>(texCoord, m_mips[mip].size);
        OMM_ASSERT(idx < m_mips[mip].numElements);
        return ((float*)(m_data + m_mips[mip].dataOffset))[idx];
    }

   	template<> uint64_t TextureImpl::From2Dto1D<TilingMode::Linear>(const int2& idx, const int2& size);
   	template<> uint64_t TextureImpl::From2Dto1D<TilingMode::MortonZ>(const int2& idx, const int2& size);
}
