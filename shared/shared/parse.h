/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "math.h"
#include "omm.h"

namespace omm
{
namespace parse
{
	static int32_t GetOmmIndexForTriangleIndex(const omm::Cpu::BakeResultDesc& resDesc, uint32_t i) {
		OMM_ASSERT(resDesc.ommIndexFormat == omm::IndexFormat::I16_UINT || resDesc.ommIndexFormat == omm::IndexFormat::I32_UINT);
		if (resDesc.ommIndexFormat == omm::IndexFormat::I16_UINT)
			return reinterpret_cast<const int16_t*>(resDesc.ommIndexBuffer)[i];
		else
			return reinterpret_cast<const int32_t*>(resDesc.ommIndexBuffer)[i];
	}

	static int32_t GetOmmBitSize(const omm::Cpu::OpacityMicromapDesc& desc)
	{
		uint32_t bitSize = ((omm::OMMFormat)desc.format == omm::OMMFormat::OC1_2_State) ? 2 : 4;
		return bitSize * omm::bird::GetNumMicroTriangles(desc.subdivisionLevel);
	}

	static int32_t GetTriangleStates(uint32_t triangleIdx, const omm::Cpu::BakeResultDesc& resDesc, omm::OpacityState* outStates) {

		const int32_t vmIdx = GetOmmIndexForTriangleIndex(resDesc, triangleIdx);

		if (vmIdx < 0) {
			if (outStates) {
				outStates[0] = (omm::OpacityState)~vmIdx;
			}
			return 0;
		}
		else {

			const omm::Cpu::OpacityMicromapDesc& vmDesc = resDesc.ommDescArray[vmIdx];
			const uint8_t* ommArrayData = (const uint8_t*)((const char*)resDesc.ommArrayData) + vmDesc.offset;
			const uint32_t numMicroTriangles = 1u << (vmDesc.subdivisionLevel << 1u);
			const uint32_t is2State = (omm::OMMFormat)vmDesc.format == omm::OMMFormat::OC1_2_State ? 1 : 0;
			if (outStates) {
				const uint32_t vmBitCount = omm::bird::GetBitCount((omm::OMMFormat)vmDesc.format);
				for (uint32_t uTriIt = 0; uTriIt < numMicroTriangles; ++uTriIt)
				{
					int byteIndex = uTriIt >> (2 + is2State);
					uint8_t v = ((uint8_t*)ommArrayData)[byteIndex];
					omm::OpacityState state;
					if (is2State)   state = omm::OpacityState((v >> ((uTriIt << 0) & 7)) & 1); // 2-state
					else			state = omm::OpacityState((v >> ((uTriIt << 1) & 7)) & 3); // 4-state
					outStates[uTriIt] = state;
				}
			}

			return vmDesc.subdivisionLevel;
		}
	}
} // namespace parse
} // namespace omm