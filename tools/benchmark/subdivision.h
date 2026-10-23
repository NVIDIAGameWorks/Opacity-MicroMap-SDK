/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <omm.hpp>

#include <charconv>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace omm::benchmark
{
    inline SubdivisionDesc EffectiveSubdivision(const Cpu::BakeInputDesc& input, bool cpuLegacyEdge = true)
    {
        auto subdivision = input.subdivision;
        if (subdivision.mode == SubdivisionMode::Legacy)
        {
            if (input.dynamicSubdivisionScale <= 0.f)
                subdivision.mode = SubdivisionMode::Fixed;
            else
            {
                // The CPU SDK has an internal edge-selection bit; the GPU has no legacy equivalent.
                constexpr uint32_t internalCpuEdgeHeuristic = 1u << 11;
                subdivision.mode = (cpuLegacyEdge && (static_cast<uint32_t>(input.bakeFlags) & internalCpuEdgeHeuristic))
                    ? SubdivisionMode::Edge : SubdivisionMode::Area;
                subdivision.targetAreaInSquareTexels = input.dynamicSubdivisionScale * input.dynamicSubdivisionScale;
                subdivision.targetEdgeLengthInTexels = input.dynamicSubdivisionScale;
            }
        }
        return subdivision;
    }

    inline float ParseSubdivisionTarget(std::string_view text)
    {
        float value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc() || end != text.data() + text.size() || !std::isfinite(value) || value <= 0.f)
            throw std::runtime_error("Subdivision targets must be finite and positive: " + std::string(text));
        return value;
    }

    inline void PrintSubdivision(std::ostream& output, const Cpu::BakeInputDesc& input, bool cpuLegacyEdge = true)
    {
        const auto subdivision = EffectiveSubdivision(input, cpuLegacyEdge);
        switch (subdivision.mode)
        {
        case SubdivisionMode::Fixed:
            output << "fixed subdivision";
            break;
        case SubdivisionMode::Area:
            output << "area target " << subdivision.targetAreaInSquareTexels << " square texels"
                << ", degenerate edge target " << subdivision.targetEdgeLengthInTexels << " texels";
            break;
        case SubdivisionMode::Edge:
            output << "edge target " << subdivision.targetEdgeLengthInTexels << " texels";
            break;
        default:
            output << "invalid subdivision mode";
            break;
        }
        if (input.subdivision.mode == SubdivisionMode::Legacy)
            output << " (legacy settings)";
    }
}
