/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "platform.hlsli"
#include "omm.hlsli"
#include "omm_global_cb.hlsli"
#include "omm_global_samplers.hlsli"
#include "omm_copy_omm_index_buffer.cs.resources.hlsli"

OMM_DECLARE_GLOBAL_CONSTANT_BUFFER
OMM_DECLARE_GLOBAL_SAMPLERS
OMM_DECLARE_INPUT_RESOURCES
OMM_DECLARE_OUTPUT_RESOURCES
OMM_DECLARE_SUBRESOURCES

int GetOmmDescOffset(uint primitiveIndex)
{
	// TODO: support 16-bit indices.
	return t_ommIndexBuffer.Load(4 * primitiveIndex.x);
}

[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= g_GlobalConstants.PrimitiveCount)
		return;

	const uint primitiveIndex = tid.x;
	const int ommDescOffsetOrSpcecialIndex = GetOmmDescOffset(primitiveIndex);
	OMM_SUBRESOURCE_STORE(TempOmmIndexBuffer, 4 * primitiveIndex, ommDescOffsetOrSpcecialIndex);
}