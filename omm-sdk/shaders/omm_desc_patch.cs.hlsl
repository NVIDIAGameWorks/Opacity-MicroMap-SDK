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
#include "omm_desc_patch.cs.resources.hlsli"

OMM_DECLARE_GLOBAL_CONSTANT_BUFFER
OMM_DECLARE_GLOBAL_SAMPLERS
OMM_DECLARE_INPUT_RESOURCES
OMM_DECLARE_OUTPUT_RESOURCES
OMM_DECLARE_SUBRESOURCES

bool GetSpecialIndex(uint primitiveIndex, out SpecialIndex specialIndex)
{
	if (!g_GlobalConstants.EnableSpecialIndices)
		return false;

	const uint3 counts = OMM_SUBRESOURCE_LOAD3(SpecialIndicesStateBuffer, 12 * primitiveIndex);

	if (counts.x == 1 && counts.y == 0 && counts.z == 0)
	{
		specialIndex = SpecialIndex::FullyOpaque;
		return true;
	}
	else if (counts.x == 0 && counts.y == 1 && counts.z == 0)
	{
		specialIndex = SpecialIndex::FullyTransparent;
		return true;
	}
	else if (counts.x == 0 && counts.y == 0 && counts.z == 1)
	{
		specialIndex = SpecialIndex::FullyUnknownOpaque;
		return true;
	}
	return false;
}

uint GetSourcePrimitiveIndex(uint primitiveIndex)
{
	const int primitiveIndexOrHashTableEntryIndex = OMM_SUBRESOURCE_LOAD(TempOmmIndexBuffer, 4 * primitiveIndex);

	if (primitiveIndexOrHashTableEntryIndex < -4)
	{
		const uint hashTableEntryIndex = -(primitiveIndexOrHashTableEntryIndex + 4);
		const uint primitiveIndexRef =  OMM_SUBRESOURCE_LOAD(HashTableBuffer, 8 * hashTableEntryIndex + 4); // [hash|primitiveIndex]
		return primitiveIndexRef;
	}
	return primitiveIndex; // Source and dest is the same => no reuse
}

void IncrementIndexHistogram(uint ommDescOffset)
{
	const uint kOMMFormatNum = 2;

	const uint vmDescData = t_ommDescArrayBuffer.Load(ommDescOffset * 8 + 4);
	const OMMFormat vmFormat = (OMMFormat)(vmDescData >> 16u);
	const uint subdivisionLevel = (uint)(vmDescData & 0x0000FFFF);

	const uint strideInBytes = 8;	// sizeof(VisibilityMapUsageDesc), [count32, format16, level16]
	const uint index = (kOMMFormatNum * subdivisionLevel + ((uint)vmFormat - 1));
	const uint offset = strideInBytes * index;

	u_ommIndexHistogramBuffer.InterlockedAdd(offset, 1);
}

[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
	if (tid.x >= g_GlobalConstants.PrimitiveCount)
		return;

	const uint dstPrimitiveIndex = tid.x;
	const uint srcPrimitiveIndex = GetSourcePrimitiveIndex(dstPrimitiveIndex);

	SpecialIndex specialIndex;
	if (GetSpecialIndex(srcPrimitiveIndex, specialIndex))
	{
		OMM_SUBRESOURCE_STORE(TempOmmIndexBuffer, 4 * dstPrimitiveIndex, specialIndex);
	}
	else
	{
		const uint ommDescIndex = OMM_SUBRESOURCE_LOAD(TempOmmIndexBuffer, 4 * srcPrimitiveIndex);
		IncrementIndexHistogram(ommDescIndex);
		if (srcPrimitiveIndex != dstPrimitiveIndex)
		{
			OMM_SUBRESOURCE_STORE(TempOmmIndexBuffer, 4 * dstPrimitiveIndex, ommDescIndex);
		}
	}
}