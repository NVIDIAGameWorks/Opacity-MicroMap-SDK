/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "texture_impl.h"

#include "defines.h"
#include "std_containers.h"

#include <shared/math.h>
#include <shared/bit_tricks.h>
#include <shared/texture.h>

#include <cstring>

namespace omm
{
    TextureImpl::TextureImpl(const StdAllocator<uint8_t>& stdAllocator) :
        m_stdAllocator(stdAllocator),
        m_mips(stdAllocator),
        m_tilingMode(TilingMode::MAX_NUM),
        m_data(nullptr)
    {
    }

    TextureImpl::~TextureImpl()
    {
        Deallocate();
    }

    Result TextureImpl::Validate(const Cpu::TextureDesc& desc) {
        if (desc.mipCount == 0)
            return Result::INVALID_ARGUMENT;
        if (desc.format == Cpu::TextureFormat::MAX_NUM)
            return Result::INVALID_ARGUMENT;

        for (uint32_t i = 0; i < desc.mipCount; ++i)
        {
            if (!desc.mips[i].textureData)
                return Result::INVALID_ARGUMENT;
            if (desc.mips[i].width == 0)
                return Result::INVALID_ARGUMENT;
            if (desc.mips[i].height == 0)
                return Result::INVALID_ARGUMENT;
            if (desc.mips[i].width > kMaxDim.x)
                return Result::INVALID_ARGUMENT;
            if (desc.mips[i].height > kMaxDim.y)
                return Result::INVALID_ARGUMENT;
        }

        return Result::SUCCESS;
    }

    Result TextureImpl::Create(const Cpu::TextureDesc& desc)
    {
        RETURN_STATUS_IF_FAILED(Validate(desc));

        Deallocate();

        m_mips.resize(desc.mipCount);
        m_tilingMode = !!((uint32_t)desc.flags & (uint32_t)Cpu::TextureFlags::DisableZOrder) ? TilingMode::Linear : TilingMode::MortonZ;

        size_t totalSize = 0;
        for (uint32_t mipIt = 0; mipIt < desc.mipCount; ++mipIt)
        {
            m_mips[mipIt].size = { desc.mips[mipIt].width, desc.mips[mipIt].height };
            m_mips[mipIt].sizeMinusOne = m_mips[mipIt].size - 1;
            m_mips[mipIt].rcpSize = 1.f / (float2)m_mips[mipIt].size;
            m_mips[mipIt].dataOffset = totalSize;

            if (desc.format == Cpu::TextureFormat::FP32)
            {
                if (m_tilingMode == TilingMode::Linear)
                {
                    m_mips[mipIt].numElements = size_t(m_mips[mipIt].size.x) * m_mips[mipIt].size.y;
                }
                else if (m_tilingMode == TilingMode::MortonZ)
                {
                    size_t maxDim = nextPow2(std::max(m_mips[mipIt].size.x, m_mips[mipIt].size.y));
                    m_mips[mipIt].numElements = maxDim * maxDim;
                }
                else
                {
                    OMM_ASSERT(false);
                    return Result::INVALID_ARGUMENT;
                }
            }
            else
            {
                OMM_ASSERT(false);
                return Result::INVALID_ARGUMENT;
            }

            totalSize += sizeof(float) * m_mips[mipIt].numElements;
            totalSize = math::Align(totalSize, kAlignment);
        }

        m_data = m_stdAllocator.allocate(totalSize, kAlignment);

        for (uint32_t mipIt = 0; mipIt < desc.mipCount; ++mipIt)
        {
            if (desc.format == Cpu::TextureFormat::FP32)
            {
                if (m_tilingMode == TilingMode::Linear)
                {
                    const size_t kDefaultRowPitch = sizeof(float) * desc.mips[mipIt].width;
                    const size_t srcRowPitch = desc.mips[mipIt].rowPitch == 0 ? kDefaultRowPitch : desc.mips[mipIt].rowPitch;

                    if (kDefaultRowPitch == srcRowPitch)
                    {
                       void* dst = m_data + m_mips[mipIt].dataOffset;
                       const float* src = (float*)(desc.mips[mipIt].textureData);
                       std::memcpy(dst, src, sizeof(float) * m_mips[mipIt].numElements);
                    }
                    else
                    {
                        uint8_t* dstBegin = m_data + m_mips[mipIt].dataOffset;
                        const uint8_t* srcBegin = (const uint8_t*)desc.mips[mipIt].textureData;

                        const size_t dstRowPitch = m_mips[mipIt].size.x * sizeof(float);
                        for (uint32_t rowIt = 0; rowIt < desc.mips[mipIt].height; rowIt++)
                        {
                            uint8_t* dst = dstBegin + rowIt * dstRowPitch;
                            const uint8_t* src = srcBegin + rowIt * srcRowPitch;
                            std::memcpy(dst, src, dstRowPitch);
                        }
                    }
                }
                else if (m_tilingMode == TilingMode::MortonZ)
                {
                    float* dst = (float*)(m_data + m_mips[mipIt].dataOffset);
                    const float* src = (float*)(desc.mips[mipIt].textureData);

                    const size_t rowPitch = desc.mips[mipIt].rowPitch == 0 ? desc.mips[mipIt].width : desc.mips[mipIt].rowPitch;

                    for (int j = 0; j < m_mips[mipIt].size.y; ++j)
                    {
                        for (int i = 0; i < m_mips[mipIt].size.x; ++i)
                        {
                            const uint64_t idx = From2Dto1D<TilingMode::MortonZ>(int2(i, j), m_mips[mipIt].size);
                            OMM_ASSERT(idx < m_mips[mipIt].numElements);
                            dst[idx] = src[i + j * rowPitch];
                        }
                    }

                    size_t maxDim = nextPow2(std::max(m_mips[mipIt].size.x, m_mips[mipIt].size.y));
                    m_mips[mipIt].numElements = maxDim * maxDim;
                }
                else
                {
                    OMM_ASSERT(false);
                    return Result::INVALID_ARGUMENT;
                }
            }
            else
            {
                OMM_ASSERT(false);
                return Result::INVALID_ARGUMENT;
            }
        }

        return Result::SUCCESS;
    }

    void TextureImpl::Deallocate()
    {
        if (m_data != nullptr)
        {
            m_stdAllocator.deallocate(m_data, 0);
            m_data = nullptr;
        }
        m_mips.clear();
    }

    float TextureImpl::Load(const int2& texCoord, int32_t mip) const 
    {
        if (m_tilingMode == TilingMode::Linear)
            return Load<TilingMode::Linear>(texCoord, mip);
        else if (m_tilingMode == TilingMode::MortonZ)
            return Load<TilingMode::MortonZ>(texCoord, mip);
        OMM_ASSERT(false);
        return 0.f;
    }

    float TextureImpl::Bilinear(omm::TextureAddressMode mode, const float2& p, int32_t mip) const 
    {
        float2 pixel = p * (float2)(m_mips[mip].size)-0.5f;
        float2 pixelFloor = glm::floor(pixel);
        int2 coords[omm::TexelOffset::MAX_NUM];
        omm::GatherTexCoord4(mode, int2(pixelFloor), m_mips[mip].size, coords);

        float a = (float)Load(coords[omm::TexelOffset::I0x0], mip);
        float b = (float)Load(coords[omm::TexelOffset::I0x1], mip);
        float c = (float)Load(coords[omm::TexelOffset::I1x0], mip);
        float d = (float)Load(coords[omm::TexelOffset::I1x1], mip);

        const float2 weight = glm::fract(pixel);
        float ac = glm::lerp<float>(a, c, weight.x);
        float bd = glm::lerp<float>(b, d, weight.x);
        float bilinearValue = glm::lerp(ac, bd, weight.y);
        return bilinearValue;
    }

    template<>
    uint64_t TextureImpl::From2Dto1D<TilingMode::Linear>(const int2& idx, const int2& size) 
    {
        return idx.x + idx.y * uint64_t(size.x);
    }

    template<>
    uint64_t TextureImpl::From2Dto1D<TilingMode::MortonZ>(const int2& idx, const int2& size) 
    {
        // Based on
        // "Optimizing Memory Access on GPUs using Morton Order Indexing"
        // https://www.nocentino.com/Nocentino10.pdf
        // return mortonNumberBinIntl(idx.x, idx.y);

        // https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
        return xy_to_morton(idx.x, idx.y);
    }
}