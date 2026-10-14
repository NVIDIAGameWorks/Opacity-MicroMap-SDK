/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#if defined(__clang__) || defined(__GNUC__)
#define OMM_USE_BUILTIN_CTZ (1)
#elif defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM64EC))
#define OMM_USE_MSVC_ARM64_CTZ (1)
#elif defined(_MSC_VER)
#define OMM_USE_MSVC_CTZ (1)
#endif

#if OMM_USE_MSVC_ARM64_CTZ || OMM_USE_MSVC_CTZ
#include <intrin.h>
#endif

#if defined(OMM_ENABLE_BMI2) && !defined(_M_ARM64EC) && \
    (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)) && \
    (defined(__BMI2__) || (defined(_MSC_VER) && !defined(__clang__)))
#define OMM_USE_BMI2 (1)
#include <immintrin.h>
#endif

#include "math.h"
#include <stdint.h>

namespace omm
{
    static uint32_t nextPow2(uint32_t v) {
        v += (v == 0);
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return ++v;
    }

    static bool isPow2(int32_t x) {
        return x > 0 && !(x & (x-1));
    }

    static uint32_t _bit_interleave_sw(uint32_t in_x, uint32_t in_y) {
        // 'Interleave bits by Binary Magic Numbers'
        // https://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
        
        static constexpr unsigned int B[] = { 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF };
        static constexpr unsigned int S[] = { 1, 2, 4, 8 };

        unsigned int x = in_x;  // Interleave lower 16 bits of x and y, so the bits of x
        unsigned int y = in_y;  // are in the even positions and bits from y in the odd;
        unsigned int z;         // z gets the resulting 32-bit Morton Number.  
                                // x and y must initially be less than 65536.

        x = (x | (x << S[3])) & B[3];
        x = (x | (x << S[2])) & B[2];
        x = (x | (x << S[1])) & B[1];
        x = (x | (x << S[0])) & B[0];

        y = (y | (y << S[3])) & B[3];
        y = (y | (y << S[2])) & B[2];
        y = (y | (y << S[1])) & B[1];
        y = (y | (y << S[0])) & B[0];

        z = x | (y << 1);
        return z;
    }

    inline uint32_t ctz(uint32_t value)
    {
#if OMM_USE_BUILTIN_CTZ
        return value ? static_cast<uint32_t>(__builtin_ctz(value)) : 32u;
#elif OMM_USE_MSVC_ARM64_CTZ
        return _CountTrailingZeros(value);
#elif OMM_USE_MSVC_CTZ
        unsigned long index;
        return _BitScanForward(&index, value) ? static_cast<uint32_t>(index) : 32u;
#else
        if (value == 0) {
            return 32;
        }

        uint32_t count = 0;
        while ((value & 1) == 0) {
            count++;
            value >>= 1;
        }
        return count;
#endif
    }

    inline uint2 ctz(uint2 value)
    {
        return uint2(ctz(value.x), ctz(value.y));
    }

    inline uint32_t bit_interleave(uint32_t x, uint32_t y)
    {
#if OMM_USE_BMI2
        return _pdep_u32(x, 0x55555555) | _pdep_u32(y, 0xaaaaaaaa);
#else
        return _bit_interleave_sw(x, y);
#endif
    }

    inline uint32_t morton1(uint32_t x)
    {
        x = x & 0x55555555;
        x = (x | (x >> 1)) & 0x33333333;
        x = (x | (x >> 2)) & 0x0F0F0F0F;
        x = (x | (x >> 4)) & 0x00FF00FF;
        x = (x | (x >> 8)) & 0x0000FFFF;
        return x;
    }

    inline void bit_deinterleave_sw(uint32_t i, uint32_t& x, uint32_t& y)
    {
        x = morton1(i);
        y = morton1(i >> 1);
    }

    inline uint32_t xy_to_morton_sw(uint32_t x, uint32_t y)
    {
        return _bit_interleave_sw(x, y);
    }

    inline uint32_t xy_to_morton(uint32_t x, uint32_t y)
    {
        return bit_interleave(x, y);
    }

    inline void morton_to_xy(uint32_t i, uint32_t& x, uint32_t& y)
    {
        return bit_deinterleave_sw(i, x, y);
    }

} // namespace omm
