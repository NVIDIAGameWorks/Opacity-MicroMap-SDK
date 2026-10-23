/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <array>
#include <algorithm>
#include <cstring>

#include "defines.h"
#include "std_containers.h"
#include "util/assert.h"

// Expose the streaming state so hashing needs no heap allocation.
#ifndef XXH_STATIC_LINKING_ONLY
#define XXH_STATIC_LINKING_ONLY
#define OMM_UNDEF_XXH_STATIC_LINKING_ONLY
#endif
#include <xxhash.h>
#ifdef OMM_UNDEF_XXH_STATIC_LINKING_ONLY
#undef XXH_STATIC_LINKING_ONLY
#undef OMM_UNDEF_XXH_STATIC_LINKING_ONLY
#endif

namespace omm
{
namespace Cpu
{
    class OmmArrayDataVector final
    {
    public:
        OmmArrayDataVector() = delete;
        OmmArrayDataVector(const StdAllocator<uint8_t>& stdAllocator, ommFormat format, uint32_t subdivisionLevel)
            : _initialStateCount(GetStateCount(subdivisionLevel))
            , _stateCount(_initialStateCount)
            , _data((_initialStateCount + 3u) / 4u, uint8_t(0xff), stdAllocator)
        {
            OMM_ASSERT(format == ommFormat_OC1_2_State || format == ommFormat_OC1_4_State);
            static_assert(ommOpacityState_Transparent == 0 && ommOpacityState_Opaque == 1 &&
                ommOpacityState_UnknownTransparent == 2 && ommOpacityState_UnknownOpaque == 3);
        }

        void SetState(uint32_t index, ommOpacityState state)
        {
            OMM_ASSERT(index < _stateCount);
            OMM_ASSERT(uint32_t(state) <= uint32_t(ommOpacityState_UnknownOpaque));
            const uint32_t shift = (index & 3u) * 2u;
            uint8_t& packed = _data[index >> 2u];
            packed = uint8_t((packed & ~(3u << shift)) | (uint32_t(state) << shift));
        }

        OMM_FORCE_INLINE ommOpacityState GetState(uint32_t index) const
        {
            OMM_ASSERT(index < _stateCount);
            return ommOpacityState((_data[index >> 2u] >> ((index & 3u) * 2u)) & 3u);
        }

        OMM_FORCE_INLINE ommOpacityState Get3State(uint32_t index) const
        {
            OMM_ASSERT(index < _stateCount);
            const ommOpacityState state = ommOpacityState((_data[index >> 2u] >> ((index & 3u) * 2u)) & 3u);
            return state == ommOpacityState_UnknownTransparent ? ommOpacityState_UnknownOpaque : state;
        }

        void ShrinkTo(uint32_t subdivisionLevel)
        {
            const uint32_t stateCount = GetStateCount(subdivisionLevel);
            OMM_ASSERT(stateCount < _stateCount);
            _stateCount = stateCount;
            // Exact deduplication historically hashes the initial extent, including
            // the retained tail after downsampling. Keep that behavior unchanged.
        }

        uint64_t Compute3StateHash(uint64_t seed = 42) const
        {
            XXH64_state_t hashState;
            XXH64_reset(&hashState, seed);
            // Distinguish state counts that occupy the same number of packed bytes.
            const std::array<uint8_t, 4> stateCountLE = {
                uint8_t(_initialStateCount), uint8_t(_initialStateCount >> 8u),
                uint8_t(_initialStateCount >> 16u), uint8_t(_initialStateCount >> 24u)
            };
            XXH64_update(&hashState, stateCountLE.data(), stateCountLE.size());

            std::array<uint8_t, 1024> canonical;
            for (size_t first = 0; first < _data.size();)
            {
                const size_t count = std::min(canonical.size(), _data.size() - first);
                size_t i = 0;
                for (; i + sizeof(uint64_t) <= count; i += sizeof(uint64_t))
                {
                    uint64_t packed;
                    std::memcpy(&packed, _data.data() + first + i, sizeof(packed));
                    packed |= (packed & 0xAAAAAAAAAAAAAAAAull) >> 1u;
                    std::memcpy(canonical.data() + i, &packed, sizeof(packed));
                }
                for (; i < count; ++i)
                {
                    const uint8_t packed = _data[first + i];
                    canonical[i] = uint8_t(packed | ((packed & 0xAAu) >> 1u));
                }
                if (first + count == _data.size() && (_initialStateCount & 3u) != 0)
                    canonical[count - 1] &= uint8_t((1u << ((_initialStateCount & 3u) * 2u)) - 1u);

                XXH64_update(&hashState, canonical.data(), count);
                first += count;
            }
            return XXH64_digest(&hashState);
        }

        uint32_t StateCount() const { return _stateCount; }
        size_t StorageSize() const { return _data.size(); }

    private:
        static uint32_t GetStateCount(uint32_t subdivisionLevel)
        {
            OMM_ASSERT(subdivisionLevel <= kMaxSubdivLevel);
            return 1u << (subdivisionLevel * 2u);
        }

        uint32_t _initialStateCount;
        uint32_t _stateCount;
        vector<uint8_t> _data;
    };
}
}
