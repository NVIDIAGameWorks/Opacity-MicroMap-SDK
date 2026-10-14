/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#if defined(_MSC_VER)
    #define OMM_FORCE_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
    #define OMM_FORCE_INLINE inline __attribute__((always_inline))
#else
    #define OMM_FORCE_INLINE inline
#endif

// Clang's precise guard does not fully undo global fast-math assumptions.
#if defined(_MSC_VER) && !defined(__clang__)
    #define OMM_PRECISE_MATH_BEGIN __pragma(float_control(precise, on, push))
    #define OMM_PRECISE_MATH_END __pragma(float_control(pop))
#elif defined(__GNUC__) && !defined(__clang__)
    #define OMM_PRECISE_MATH_BEGIN _Pragma("GCC push_options") _Pragma("GCC optimize(\"no-fast-math\")")
    #define OMM_PRECISE_MATH_END _Pragma("GCC pop_options")
#else
    #define OMM_PRECISE_MATH_BEGIN
    #define OMM_PRECISE_MATH_END
#endif

#define RETURN_STATUS_IF_FAILED(expr)           \
do {                                            \
    ommResult sts##__COUNTER__ = expr;          \
    if (sts##__COUNTER__ != ommResult_SUCCESS)  \
        return sts##__COUNTER__;                \
} while(false)

#include<stdint.h>

namespace omm
{
    ///< Defined by DX/VK spec.
    static constexpr uint32_t kMaxSubdivLevel         = 12;
    ///< Useful for statically allocated arrays.
    static constexpr uint32_t kMaxNumSubdivLevels     = kMaxSubdivLevel + 1;
}