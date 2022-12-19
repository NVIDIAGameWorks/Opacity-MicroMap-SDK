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

typedef omm::MemoryAllocatorInterface MemoryAllocatorInterface;
#include "std_allocator.h"

namespace omm
{
    OMM_API Result SaveAsImagesImpl(StdAllocator<uint8_t>& memoryAllocator, const Cpu::BakeInputDesc& bakeInputDesc, const Cpu::BakeResultDesc* res, const Debug::SaveImagesDesc& desc);

    OMM_API Result GetStatsImpl(StdAllocator<uint8_t>& memoryAllocator, const Cpu::BakeResultDesc* res, Debug::Stats* out);
}
