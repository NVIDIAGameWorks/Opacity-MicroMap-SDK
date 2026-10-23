/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "omm.h"
#include <cstring>

namespace omm
{
    inline bool IsFiniteSubdivisionValue(float value)
    {
        // The baker is compiled with fast math; inspect the exponent so validation still rejects NaN/Inf.
        uint32_t bits;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & 0x7f800000u) != 0x7f800000u;
    }

    struct ResolvedSubdivision
    {
        ommSubdivisionMode mode;
        float targetAreaInSquareTexels;
        float targetEdgeLengthInTexels;
        // Retained separately to preserve legacy arithmetic and rounding.
        float legacyScale;
        bool legacy;
    };

    // Only settings used by the selected mode are validated. Explicit modes ignore legacy inputs.
    inline const char* ValidateSubdivision(const ommSubdivisionDesc& desc, float legacyScale)
    {
        switch (desc.mode)
        {
        case ommSubdivisionMode_Legacy:
            return IsFiniteSubdivisionValue(legacyScale) ? nullptr : "dynamicSubdivisionScale must be finite in Legacy mode";
        case ommSubdivisionMode_Fixed:
            return nullptr;
        case ommSubdivisionMode_Area:
            if (!IsFiniteSubdivisionValue(desc.targetAreaInSquareTexels) || desc.targetAreaInSquareTexels <= 0.f)
                return "subdivision.targetAreaInSquareTexels must be finite and positive in Area mode";
            // Area mode needs an edge target for degenerate UV triangles.
            [[fallthrough]];
        case ommSubdivisionMode_Edge:
            if (!IsFiniteSubdivisionValue(desc.targetEdgeLengthInTexels) || desc.targetEdgeLengthInTexels <= 0.f)
                return "subdivision.targetEdgeLengthInTexels must be finite and positive in Area or Edge mode";
            return nullptr;
        default:
            return "subdivision.mode is invalid";
        }
    }

    inline ResolvedSubdivision ResolveSubdivision(const ommSubdivisionDesc& desc, float legacyScale, bool legacyEdge)
    {
        if (desc.mode != ommSubdivisionMode_Legacy)
            return { desc.mode, desc.targetAreaInSquareTexels, desc.targetEdgeLengthInTexels, 0.f, false };

        return {
            legacyScale > 0.f ? (legacyEdge ? ommSubdivisionMode_Edge : ommSubdivisionMode_Area) : ommSubdivisionMode_Fixed,
            legacyScale * legacyScale,
            legacyScale,
            legacyScale,
            true
        };
    }
}
