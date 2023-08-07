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

        template<ommCpuTextureFormat eFormat, TilingMode eTilingMode>
        float Load(const int2& texCoord, int32_t mip) const;

        float Load(const int2& texCoord, int32_t mip) const;

        float Bilinear(ommTextureAddressMode mode, const float2& p, int32_t mip) const;

        TilingMode GetTilingMode() const {
            return m_tilingMode;
        }
        
        ommCpuTextureFormat GetTextureFormat() const {
            return m_textureFormat;
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

        bool HasAlphaCutoff() const {
            return m_alphaCutoff >= 0.f;
        }

        float GetAlphaCutoff() const {
            return m_alphaCutoff;
        }

        bool InTexture(int2 texCoord, int32_t mip) const
        {
            return texCoord.x >= 0 &&
                texCoord.y >= 0 &&
                texCoord.x < m_mips[mip].size.x &&
                texCoord.y < m_mips[mip].size.y;
        }

        bool HasSAT() const
        {
            return m_dataSAT != nullptr;
        }

        uint32_t SAT(int2 s, int2 e, int32_t mip) const
        {
            OMM_ASSERT(InTexture(s, mip));
            OMM_ASSERT(InTexture(e, mip));
#if 1
            uint32_t* dataSAT = (uint32_t*)(m_dataSAT + m_mips[mip].dataOffsetSAT);

            int32_t s_x_minus_one = (s.x - 1);
            int32_t s_y_minus_one = (s.y - 1);

            const uint32_t A = s_x_minus_one >= 0 && s_y_minus_one >= 0 ? dataSAT[(s_x_minus_one) + (s_y_minus_one) * m_mips[mip].size.x] : 0;
            const uint32_t B = s_y_minus_one >= 0 ? dataSAT[e.x + (s_y_minus_one) * m_mips[mip].size.x] : 0;
            const uint32_t C = s_x_minus_one >= 0 ? dataSAT[(s_x_minus_one) + (e.y) * m_mips[mip].size.x] : 0;
            const uint32_t D = dataSAT[e.x + (e.y) * m_mips[mip].size.x];
            int32_t sum = D + A - B - C;
            return sum;
#else

            uint32_t sumTrue = 0;
            for (int j = s.y; j <= e.y; ++j)
            {
                for (int i = s.x; i <= e.x; ++i)
                {
                    const float val = Load(int2(i, j), mip);
                    sumTrue += val > 0.5f;
                }
            }
            return sumTrue;
#endif
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
            uintptr_t dataOffsetSAT;
        };

        vector<Mips> m_mips;
        TilingMode m_tilingMode;
        ommCpuTextureFormat m_textureFormat;
        float m_alphaCutoff;
        uint8_t* m_data;
        uint8_t* m_dataSAT;
        size_t m_dataSize;
    };

    template<ommCpuTextureFormat eFormat, TilingMode eTilingMode>
    float TextureImpl::Load(const int2& texCoord, int32_t mip) const
    {
        OMM_ASSERT(eTilingMode == m_tilingMode);
        OMM_ASSERT(eFormat == m_textureFormat);
        OMM_ASSERT(texCoord.x >= 0);
        OMM_ASSERT(texCoord.y >= 0);
        OMM_ASSERT(texCoord.x < m_mips[mip].size.x);
        OMM_ASSERT(texCoord.y < m_mips[mip].size.y);
        OMM_ASSERT(glm::all(glm::notEqual(texCoord, kTexCoordBorder2)));
        OMM_ASSERT(glm::all(glm::notEqual(texCoord, kTexCoordInvalid2)));
        const uint64_t idx = From2Dto1D<eTilingMode>(texCoord, m_mips[mip].size);
        OMM_ASSERT(idx < m_mips[mip].numElements);

        if constexpr (eFormat == ommCpuTextureFormat_FP32)
            return ((float*)(m_data + m_mips[mip].dataOffset))[idx];
        else if constexpr (eFormat == ommCpuTextureFormat_UNORM8)
            return (float)((uint8_t*)(m_data + m_mips[mip].dataOffset))[idx] * (1.f / 255.f);
        else
        {
            assert(false);
            return 0;
        }
    }

   	template<> uint64_t TextureImpl::From2Dto1D<TilingMode::Linear>(const int2& idx, const int2& size);
   	template<> uint64_t TextureImpl::From2Dto1D<TilingMode::MortonZ>(const int2& idx, const int2& size);
}
