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

#include <cstdint>
#include <memory>
#include <string>

namespace omm::benchmark
{
    struct GpuStats
    {
        uint64_t micromapCount = 0;
        uint64_t arrayBytes = 0;
        uint64_t indexCount = 0;
        uint64_t outputBytes = 0;
    };

    class GpuContext;

    class GpuInput
    {
    public:
        ~GpuInput();

        // GPU elapsed milliseconds for setup and bake; uploads and readback are excluded.
        double Run();
        GpuStats ReadStats();

    private:
        friend class GpuContext;
        struct Impl;
        explicit GpuInput(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> m_impl;
    };

    class GpuContext
    {
    public:
        explicit GpuContext(bool debug = false);
        ~GpuContext();

        const std::string& AdapterName() const;

        // The serialized input may be released after preparation completes.
        // Release a file's prepared inputs before loading the next file.
        std::unique_ptr<GpuInput> Prepare(const Cpu::BakeInputDesc& input, bool computeOnly = true);

    private:
        friend class GpuInput;
        struct Impl;
        std::shared_ptr<Impl> m_impl;
    };
}
