/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "omm_handle.h"
#include "serialize_impl.h"
#include <xxhash.h>

namespace omm
{
namespace Cpu
{
    class MemoryStreamBuf : public std::streambuf {
    public:
        MemoryStreamBuf(uint8_t* data, size_t size) {
            setg((char*)data, (char*)data, (char*)data + size);
            setp((char*)data, (char*)data + size);
        }
    };

    class PassthroughStreamBuf : public std::streambuf {
    public:
        PassthroughStreamBuf() { }

        std::streamsize GetWrittenSize() const
        {
            return written_size;
        }

        std::streamsize xsputn(const char* s, std::streamsize n) override {
            written_size += n;
            return n;
        }
    private:
        std::streamsize written_size = 0;
    };

    template<class TElem>
    void WriteArray(std::ostream& os, const TElem* data, uint32_t elementCount)
    {
        os.write(reinterpret_cast<const char*>(&elementCount), sizeof(elementCount));
        if (elementCount != 0)
            os.write(reinterpret_cast<const char*>(data), sizeof(TElem) * elementCount);
    };

    template<class TElem>
    void ReadArray(std::istream& os, StdAllocator<uint8_t>& stdAllocator, const TElem*& outData, uint32_t& outElementCount)
    {
        os.read(reinterpret_cast<char*>(&outElementCount), sizeof(outElementCount));
        if (outElementCount != 0)
        {
            size_t dataSize = sizeof(TElem) * outElementCount;
            uint8_t* data = stdAllocator.allocate(dataSize, 16);
            os.read(reinterpret_cast<char*>(data), dataSize);
            outData = (TElem*)data;
        }
        else
        {
            outData = nullptr;
        }
    };

    SerializeResultImpl::SerializeResultImpl(const StdAllocator<uint8_t>& stdAllocator, const Logger& log)
        : m_stdAllocator(stdAllocator)
        , m_log(log)
        , m_desc(ommCpuBlobDescDefault())
    {
    }

    SerializeResultImpl::~SerializeResultImpl()
    {
        if (m_desc.data != nullptr)
        {
            m_stdAllocator.deallocate((uint8_t*)m_desc.data, 0);
        }
    }

    uint32_t SerializeResultImpl::_GetMaxIndex(const ommCpuBakeInputDesc& inputDesc)
    {
        uint32_t maxIndex = 0;
        const int32_t triangleCount = inputDesc.indexCount / 3u;

        for (int32_t i = 0; i < triangleCount; ++i)
        {
            uint32_t triangleIndices[3];
            GetUInt32Indices(inputDesc.indexFormat, inputDesc.indexBuffer, 3ull * i, triangleIndices);

            maxIndex = std::max(maxIndex, triangleIndices[0]);
            maxIndex = std::max(maxIndex, triangleIndices[1]);
            maxIndex = std::max(maxIndex, triangleIndices[2]);
        }

        return maxIndex;
    }

    template<class TMemoryStreamBuf>
    ommResult SerializeResultImpl::_Serialize(const ommCpuBakeInputDesc& inputDesc, TMemoryStreamBuf& buffer)
    {
        std::ostream os(&buffer);

        static_assert(sizeof(ommCpuBakeInputDesc) == 128);

        os.write(reinterpret_cast<const char*>(&inputDesc.bakeFlags), sizeof(inputDesc.bakeFlags));

        const TextureImpl* texture = GetHandleImpl<TextureImpl>(inputDesc.texture);
        texture->Serialize(buffer);

        os.write(reinterpret_cast<const char*>(&inputDesc.runtimeSamplerDesc.addressingMode), sizeof(inputDesc.runtimeSamplerDesc.addressingMode));
        os.write(reinterpret_cast<const char*>(&inputDesc.runtimeSamplerDesc.filter), sizeof(inputDesc.runtimeSamplerDesc.filter));
        os.write(reinterpret_cast<const char*>(&inputDesc.runtimeSamplerDesc.borderAlpha), sizeof(inputDesc.runtimeSamplerDesc.borderAlpha));
        os.write(reinterpret_cast<const char*>(&inputDesc.alphaMode), sizeof(inputDesc.alphaMode));

        os.write(reinterpret_cast<const char*>(&inputDesc.texCoordFormat), sizeof(inputDesc.texCoordFormat));
        const size_t texCoordsSize = GetTexCoordFormatSize(inputDesc.texCoordFormat) * (_GetMaxIndex(inputDesc) + 1);
        os.write(reinterpret_cast<const char*>(&texCoordsSize), sizeof(texCoordsSize));
        if (texCoordsSize != 0)
        {
            os.write(reinterpret_cast<const char*>(inputDesc.texCoords), texCoordsSize);
        }
        os.write(reinterpret_cast<const char*>(&inputDesc.texCoordStrideInBytes), sizeof(inputDesc.texCoordStrideInBytes));

        os.write(reinterpret_cast<const char*>(&inputDesc.indexFormat), sizeof(inputDesc.indexFormat));
        os.write(reinterpret_cast<const char*>(&inputDesc.indexCount), sizeof(inputDesc.indexCount));

        static_assert(ommIndexFormat_MAX_NUM == 2);
        size_t indexBufferSize = inputDesc.indexCount * (inputDesc.indexFormat == ommIndexFormat_UINT_16 ? 2 : 4);
        os.write(reinterpret_cast<const char*>(inputDesc.indexBuffer), indexBufferSize);

        os.write(reinterpret_cast<const char*>(&inputDesc.dynamicSubdivisionScale), sizeof(inputDesc.dynamicSubdivisionScale));
        os.write(reinterpret_cast<const char*>(&inputDesc.rejectionThreshold), sizeof(inputDesc.rejectionThreshold));
        os.write(reinterpret_cast<const char*>(&inputDesc.alphaCutoff), sizeof(inputDesc.alphaCutoff));
        os.write(reinterpret_cast<const char*>(&inputDesc.alphaCutoffLessEqual), sizeof(inputDesc.alphaCutoffLessEqual));
        os.write(reinterpret_cast<const char*>(&inputDesc.alphaCutoffGreater), sizeof(inputDesc.alphaCutoffGreater));
        os.write(reinterpret_cast<const char*>(&inputDesc.format), sizeof(inputDesc.format));

        size_t numFormats = inputDesc.formats == nullptr ? 0 : inputDesc.indexCount;
        os.write(reinterpret_cast<const char*>(&numFormats), sizeof(numFormats));

        if (numFormats != 0)
        {
            os.write(reinterpret_cast<const char*>(inputDesc.formats), numFormats * sizeof(ommFormat));
        }

        os.write(reinterpret_cast<const char*>(&inputDesc.unknownStatePromotion), sizeof(inputDesc.unknownStatePromotion));
        os.write(reinterpret_cast<const char*>(&inputDesc.maxSubdivisionLevel), sizeof(inputDesc.maxSubdivisionLevel));

        size_t numSubdivLvls = inputDesc.subdivisionLevels == nullptr ? 0 : inputDesc.indexCount;
        os.write(reinterpret_cast<const char*>(&numSubdivLvls), sizeof(numSubdivLvls));
        if (numSubdivLvls != 0)
        {
            os.write(reinterpret_cast<const char*>(inputDesc.subdivisionLevels), numSubdivLvls * sizeof(uint8_t));
        }

        os.write(reinterpret_cast<const char*>(&inputDesc.maxWorkloadSize), sizeof(inputDesc.maxWorkloadSize));

        return ommResult_SUCCESS;
    }

    template<class TMemoryStreamBuf>
    ommResult SerializeResultImpl::_Serialize(const ommCpuBakeResultDesc& resultDesc, TMemoryStreamBuf& buffer)
    {
        std::ostream os(&buffer);

        WriteArray<uint8_t>(os, (const uint8_t*)resultDesc.arrayData, resultDesc.arrayDataSize);
        WriteArray<ommCpuOpacityMicromapDesc>(os, resultDesc.descArray, resultDesc.descArrayCount);
        WriteArray<ommCpuOpacityMicromapUsageCount>(os, resultDesc.descArrayHistogram, resultDesc.descArrayHistogramCount);
        
        os.write(reinterpret_cast<const char*>(&resultDesc.indexFormat), sizeof(resultDesc.indexFormat));
        if (resultDesc.indexFormat == ommIndexFormat_UINT_16)
        {
            WriteArray<uint16_t>(os, (const uint16_t*)resultDesc.indexBuffer, resultDesc.indexCount);
        }
        else
        {
            OMM_ASSERT(resultDesc.indexFormat == ommIndexFormat_UINT_32);
            WriteArray<uint32_t>(os, (const uint32_t*)resultDesc.indexBuffer, resultDesc.indexCount);
        }

        WriteArray<ommCpuOpacityMicromapUsageCount>(os, resultDesc.indexHistogram, resultDesc.indexHistogramCount);

        return ommResult_SUCCESS;
    }

    template<class TMemoryStreamBuf>
    ommResult SerializeResultImpl::_Serialize(const ommCpuDeserializedDesc& inputDesc, TMemoryStreamBuf& buffer)
    {
        std::ostream os(&buffer);

        // BEGIN HEADER
        // Reserve space for the digest
        XXH64_hash_t digest = { 0 };
        os.write(reinterpret_cast<const char*>(&digest), sizeof(digest));
        int major = OMM_VERSION_MAJOR;
        int minor = OMM_VERSION_MINOR;
        int patch = OMM_VERSION_BUILD;
        int inputDescVersion = SerializeResultImpl::VERSION;
        os.write(reinterpret_cast<const char*>(&major), sizeof(major));
        os.write(reinterpret_cast<const char*>(&minor), sizeof(minor));
        os.write(reinterpret_cast<const char*>(&patch), sizeof(patch));
        os.write(reinterpret_cast<const char*>(&inputDescVersion), sizeof(inputDescVersion));
        // END HEADER

        os.write(reinterpret_cast<const char*>(&inputDesc.flags), sizeof(inputDesc.flags));

        os.write(reinterpret_cast<const char*>(&inputDesc.numInputDescs), sizeof(inputDesc.numInputDescs));
        for (int i = 0; i < inputDesc.numInputDescs; ++i)
        {
            _Serialize(inputDesc.inputDescs[i], buffer);
        }

        os.write(reinterpret_cast<const char*>(&inputDesc.numResultDescs), sizeof(inputDesc.numResultDescs));
        for (int i = 0; i < inputDesc.numResultDescs; ++i)
        {
            _Serialize(inputDesc.resultDescs[i], buffer);
        }

        return ommResult_SUCCESS;
    }

    ommResult SerializeResultImpl::Serialize(const ommCpuDeserializedDesc& desc)
    {
        PassthroughStreamBuf passthrough;
        RETURN_STATUS_IF_FAILED(_Serialize(desc, passthrough));

        size_t size = passthrough.GetWrittenSize();

        m_desc.data = m_stdAllocator.allocate(size, 16);
        m_desc.size = size;

        MemoryStreamBuf buf((uint8_t*)m_desc.data, m_desc.size);
        RETURN_STATUS_IF_FAILED(_Serialize(desc, buf));

        // Compute the digest
        XXH64_hash_t hash = XXH64((uint8_t*)m_desc.data + sizeof(XXH64_hash_t), m_desc.size - sizeof(XXH64_hash_t), 42/*seed*/);
        *(XXH64_hash_t*)m_desc.data = hash;

        return ommResult_SUCCESS;
    }

    DeserializedResultImpl::DeserializedResultImpl(const StdAllocator<uint8_t>& stdAllocator, const Logger& log)
        : m_stdAllocator(stdAllocator)
        , m_log(log)
        , m_inputDesc(ommCpuDeserializedDescDefault())
    {
    }

    DeserializedResultImpl::~DeserializedResultImpl()
    {
        for (int i = 0; i < m_inputDesc.numInputDescs; ++i)
        {
            auto& inputDesc = m_inputDesc.inputDescs[i];
            OMM_ASSERT(inputDesc.texture != 0);
            if (inputDesc.texture)
            {
                const StdAllocator<uint8_t>& memoryAllocator = GetStdAllocator();
                TextureImpl* texture = GetHandleImpl<TextureImpl>(inputDesc.texture);
                Deallocate(memoryAllocator, texture);
            }

            if (inputDesc.texCoords)
            {
                m_stdAllocator.deallocate((uint8_t*)inputDesc.texCoords, 0);
            }

            if (inputDesc.indexBuffer)
            {
                m_stdAllocator.deallocate((uint8_t*)inputDesc.indexBuffer, 0);
            }

            if (inputDesc.formats)
            {
                m_stdAllocator.deallocate((uint8_t*)inputDesc.formats, 0);
            }

            if (inputDesc.subdivisionLevels)
            {
                m_stdAllocator.deallocate((uint8_t*)inputDesc.subdivisionLevels, 0);
            }
        }

        for (int i = 0; i < m_inputDesc.numResultDescs; ++i)
        {
            auto& resultDesc = m_inputDesc.resultDescs[i];

            if (resultDesc.arrayData)
            {
                m_stdAllocator.deallocate((uint8_t*)resultDesc.arrayData, 0);
            }

            if (resultDesc.descArray)
            {
                m_stdAllocator.deallocate((uint8_t*)resultDesc.descArray, 0);
            }

            if (resultDesc.descArrayHistogram)
            {
                m_stdAllocator.deallocate((uint8_t*)resultDesc.descArrayHistogram, 0);
            }

            if (resultDesc.indexBuffer)
            {
                m_stdAllocator.deallocate((uint8_t*)resultDesc.indexBuffer, 0);
            }

            if (resultDesc.indexHistogram)
            {
                m_stdAllocator.deallocate((uint8_t*)resultDesc.indexHistogram, 0);
            }
        }
    }

    template<class TMemoryStreamBuf>
    ommResult DeserializedResultImpl::_Deserialize(ommCpuBakeInputDesc& inputDesc, TMemoryStreamBuf& buffer)
    {
        std::istream os(&buffer);

        static_assert(sizeof(ommCpuBakeInputDesc) == 128);

        os.read(reinterpret_cast<char*>(&inputDesc.bakeFlags), sizeof(inputDesc.bakeFlags));

        TextureImpl* texture = Allocate<TextureImpl>(m_stdAllocator, m_stdAllocator, m_log);
        texture->Deserialize(buffer);
        inputDesc.texture = CreateHandle<omm::Cpu::Texture, TextureImpl>(texture);

        os.read(reinterpret_cast<char*>(&inputDesc.runtimeSamplerDesc.addressingMode), sizeof(inputDesc.runtimeSamplerDesc.addressingMode));
        os.read(reinterpret_cast<char*>(&inputDesc.runtimeSamplerDesc.filter), sizeof(inputDesc.runtimeSamplerDesc.filter));
        os.read(reinterpret_cast<char*>(&inputDesc.runtimeSamplerDesc.borderAlpha), sizeof(inputDesc.runtimeSamplerDesc.borderAlpha));
        os.read(reinterpret_cast<char*>(&inputDesc.alphaMode), sizeof(inputDesc.alphaMode));

        os.read(reinterpret_cast<char*>(&inputDesc.texCoordFormat), sizeof(inputDesc.texCoordFormat));

        size_t texCoordsSize = 0;
        os.read(reinterpret_cast<char*>(&texCoordsSize), sizeof(texCoordsSize));
        if (texCoordsSize != 0)
        {
            uint8_t* texCoords = m_stdAllocator.allocate(texCoordsSize, 16);
            os.read(reinterpret_cast<char*>(texCoords), texCoordsSize);
            inputDesc.texCoords = texCoords;
        }
        os.read(reinterpret_cast<char*>(&inputDesc.texCoordStrideInBytes), sizeof(inputDesc.texCoordStrideInBytes));

        os.read(reinterpret_cast<char*>(&inputDesc.indexFormat), sizeof(inputDesc.indexFormat));
        os.read(reinterpret_cast<char*>(&inputDesc.indexCount), sizeof(inputDesc.indexCount));

        static_assert(ommIndexFormat_MAX_NUM == 2);
        const size_t indexBufferSize = inputDesc.indexCount * (inputDesc.indexFormat == ommIndexFormat_UINT_16 ? 2 : 4);
        uint8_t* indexBuffer = m_stdAllocator.allocate(indexBufferSize, 16);
        os.read(reinterpret_cast<char*>(indexBuffer), indexBufferSize);
        inputDesc.indexBuffer = indexBuffer;

        os.read(reinterpret_cast<char*>(&inputDesc.dynamicSubdivisionScale), sizeof(inputDesc.dynamicSubdivisionScale));
        os.read(reinterpret_cast<char*>(&inputDesc.rejectionThreshold), sizeof(inputDesc.rejectionThreshold));
        os.read(reinterpret_cast<char*>(&inputDesc.alphaCutoff), sizeof(inputDesc.alphaCutoff));
        os.read(reinterpret_cast<char*>(&inputDesc.alphaCutoffLessEqual), sizeof(inputDesc.alphaCutoffLessEqual));
        os.read(reinterpret_cast<char*>(&inputDesc.alphaCutoffGreater), sizeof(inputDesc.alphaCutoffGreater));
        os.read(reinterpret_cast<char*>(&inputDesc.format), sizeof(inputDesc.format));

        size_t numFormats = 0;
        os.read(reinterpret_cast<char*>(&numFormats), sizeof(numFormats));
        if (numFormats != 0)
        {
            const size_t formatsSize = numFormats * sizeof(ommFormat);
            uint8_t* formats = m_stdAllocator.allocate(formatsSize, 16);
            os.read(reinterpret_cast<char*>(formats), formatsSize);
            inputDesc.formats = (ommFormat*)formats;
        }

        os.read(reinterpret_cast<char*>(&inputDesc.unknownStatePromotion), sizeof(inputDesc.unknownStatePromotion));
        os.read(reinterpret_cast<char*>(&inputDesc.maxSubdivisionLevel), sizeof(inputDesc.maxSubdivisionLevel));

        size_t numSubdivLvls = 0;
        os.read(reinterpret_cast<char*>(&numSubdivLvls), sizeof(numSubdivLvls));
        if (numSubdivLvls != 0)
        {
            const size_t subdivLvlSize = numSubdivLvls * sizeof(uint8_t);
            uint8_t* subdivisionLevels = m_stdAllocator.allocate(subdivLvlSize, 16);
            os.read(reinterpret_cast<char*>(subdivisionLevels), subdivLvlSize);
            inputDesc.subdivisionLevels = subdivisionLevels;
        }

        os.read(reinterpret_cast<char*>(&inputDesc.maxWorkloadSize), sizeof(inputDesc.maxWorkloadSize));

        return ommResult_SUCCESS;
    }

    template<class TMemoryStreamBuf>
    ommResult DeserializedResultImpl::_Deserialize(ommCpuBakeResultDesc& resultDesc, TMemoryStreamBuf& buffer)
    {
        std::istream os(&buffer);

        StdAllocator<uint8_t>& mem = m_stdAllocator;

        ReadArray<uint8_t>(os, mem, reinterpret_cast<const uint8_t*&>(resultDesc.arrayData), resultDesc.arrayDataSize);
        ReadArray<ommCpuOpacityMicromapDesc>(os, mem, resultDesc.descArray, resultDesc.descArrayCount);
        ReadArray<ommCpuOpacityMicromapUsageCount>(os, mem, resultDesc.descArrayHistogram, resultDesc.descArrayHistogramCount);

        os.read(reinterpret_cast<char*>(&resultDesc.indexFormat), sizeof(resultDesc.indexFormat));

        if (resultDesc.indexFormat == ommIndexFormat_UINT_16)
        {
            ReadArray<uint16_t>(os, mem, reinterpret_cast<const uint16_t*&>(resultDesc.indexBuffer), resultDesc.indexCount);
        }
        else
        {
            OMM_ASSERT(resultDesc.indexFormat == ommIndexFormat_UINT_32);
            ReadArray<uint32_t>(os, mem, reinterpret_cast<const uint32_t*&>(resultDesc.indexBuffer), resultDesc.indexCount);
        }

        ReadArray<ommCpuOpacityMicromapUsageCount>(os, mem, resultDesc.indexHistogram, resultDesc.indexHistogramCount);

        return ommResult_SUCCESS;
    }

    template<class TMemoryStreamBuf>
    ommResult DeserializedResultImpl::_Deserialize(XXH64_hash_t hash, ommCpuDeserializedDesc& desc, TMemoryStreamBuf& buffer)
    {
        std::istream os(&buffer);

        // BEGIN HEADER
        XXH64_hash_t storedHash;
        os.read(reinterpret_cast<char*>(&storedHash), sizeof(storedHash));

        if (hash != storedHash)
        {
            return m_log.InvalidArgf("The serialized blob appears corrupted, computed digest != header value %ull, %ull", hash, storedHash);
        }

        int major = 0;
        int minor = 0;
        int patch = 0;
        int inputDescVersion = 0;
        os.read(reinterpret_cast<char*>(&major), sizeof(major));
        os.read(reinterpret_cast<char*>(&minor), sizeof(minor));
        os.read(reinterpret_cast<char*>(&patch), sizeof(patch));
        os.read(reinterpret_cast<char*>(&inputDescVersion), sizeof(inputDescVersion));
        // END HEADER

        if (inputDescVersion != SerializeResultImpl::VERSION)
        {
            return m_log.InvalidArgf("The serialized blob appears to be generated from an incompatible version of the SDK (%d.%d.%d:%d)", major, minor, patch, inputDescVersion);
        }

        os.read(reinterpret_cast<char*>(&desc.flags), sizeof(desc.flags));

        os.read(reinterpret_cast<char*>(&desc.numInputDescs), sizeof(desc.numInputDescs));
        if (desc.numInputDescs != 0)
        {
            ommCpuBakeInputDesc* inputDescs = AllocateArray<ommCpuBakeInputDesc>(m_stdAllocator, desc.numInputDescs);
            for (int i = 0; i < desc.numInputDescs; ++i)
            {
                inputDescs[i] = ommCpuBakeInputDescDefault();
                _Deserialize(inputDescs[i], buffer);
            }
            desc.inputDescs = inputDescs;
        }

        os.read(reinterpret_cast<char*>(&desc.numResultDescs), sizeof(desc.numResultDescs));
        if (desc.numResultDescs != 0)
        {
            ommCpuBakeResultDesc* resultDescs = AllocateArray<ommCpuBakeResultDesc>(m_stdAllocator, desc.numResultDescs);
            for (int i = 0; i < desc.numResultDescs; ++i)
            {
               // resultDescs[i] = ommCpuBakeResultDescDefault();
                _Deserialize(resultDescs[i], buffer);
            }
            desc.resultDescs = resultDescs;
        }
        
        return ommResult_SUCCESS;
    }

    ommResult DeserializedResultImpl::Deserialize(const ommCpuBlobDesc& desc)
    {
        if (desc.data == nullptr)
            return m_log.InvalidArg("data must be non-null");
        if (desc.size == 0)
            return m_log.InvalidArg("size must be non-zero");

        MemoryStreamBuf buf((uint8_t*)desc.data, desc.size);

        // Compute the digest
        XXH64_hash_t hash = XXH64((const uint8_t*)desc.data + sizeof(XXH64_hash_t), desc.size - sizeof(XXH64_hash_t), 42/*seed*/);

        return _Deserialize(hash, m_inputDesc, buf);
    }

} // namespace Cpu
} // namespace omm