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
#include "log.h"

#include <shared/math.h>
#include <shared/texture.h>

#include <map>
#include <set>

#include "std_allocator.h"

typedef uint64_t XXH64_hash_t;

namespace omm
{
namespace Cpu
{
    class SerializeResultImpl
    {
    public:
        static inline constexpr HandleType kHandleType = HandleType::SerializeResult;

        enum {
            VERSION = 2
        };

        SerializeResultImpl(const StdAllocator<uint8_t>& stdAllocator, const Logger& log);
        ~SerializeResultImpl();

        inline const StdAllocator<uint8_t>& GetStdAllocator() const
        {
            return m_stdAllocator;
        }

        const ommCpuBlobDesc* GetDesc() const
        {
            return &m_desc;
        }

        ommResult Serialize(const ommCpuDeserializedDesc& desc);

    private:
        static uint32_t _GetMaxIndex(const ommCpuBakeInputDesc& inputDesc);

        template<class TMemoryStreamBuf>
        ommResult _Serialize(const ommCpuBakeInputDesc& inputDesc, TMemoryStreamBuf& buffer);
        template<class TMemoryStreamBuf>
        ommResult _Serialize(const ommCpuBakeResultDesc& resultDesc, TMemoryStreamBuf& buffer);
        template<class TMemoryStreamBuf>
        ommResult _Serialize(const ommCpuDeserializedDesc& desc, TMemoryStreamBuf& buffer);

        StdAllocator<uint8_t> m_stdAllocator;
        const Logger& m_log;
        ommCpuBlobDesc m_desc;
    };

    class DeserializedResultImpl
    {
    public:
        static inline constexpr HandleType kHandleType = HandleType::DeserializeResult;

        DeserializedResultImpl(const StdAllocator<uint8_t>& stdAllocator, const Logger& log);
        ~DeserializedResultImpl();

        inline const StdAllocator<uint8_t>& GetStdAllocator() const
        {
            return m_stdAllocator;
        }

        const ommCpuDeserializedDesc* GetDesc() const
        {
            return &m_inputDesc;
        }

        ommResult Deserialize(const ommCpuBlobDesc& desc);

    private:

        template<class TMemoryStreamBuf>
        ommResult _Deserialize(ommCpuBakeInputDesc& inputDesc, int inputDescVersion, TMemoryStreamBuf& buffer);
        template<class TMemoryStreamBuf>
        ommResult _Deserialize(ommCpuBakeResultDesc& resultDesc, TMemoryStreamBuf& buffer);
        template<class TMemoryStreamBuf>
        ommResult _Deserialize(XXH64_hash_t hash, ommCpuDeserializedDesc& desc, TMemoryStreamBuf& buffer);

        StdAllocator<uint8_t> m_stdAllocator;
        const Logger& m_log;
        ommCpuDeserializedDesc m_inputDesc;
    };
} // namespace Cpu
} // namespace omm
