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

struct PrimitiveHistogram
{
	void Init(uint3 _counts)
	{
		counts = _counts;

		specialIndex = 0;

		if (counts.x >= 1 && counts.y == 0 && counts.z == 0)
		{
			specialIndex = (int)SpecialIndex::FullyOpaque;
		}
		else if (counts.x == 0 && counts.y >= 1 && counts.z == 0)
		{
			specialIndex = (int)SpecialIndex::FullyTransparent;
		}
		else if (counts.x == 0 && counts.y == 0 && counts.z >= 1)
		{
			specialIndex = (int)SpecialIndex::FullyUnknownOpaque;
		}
	}

	bool IsValid()
	{
		return counts.x != 0 || counts.y != 0 || counts.z != 0;
	}

	bool IsSpecialIndex()
	{
		return specialIndex != 0;
	}

	SpecialIndex GetSpecialIndex()
	{
		return (SpecialIndex)specialIndex;
	}

	uint3 counts;
	int specialIndex;
};

PrimitiveHistogram LoadHistogram(uint primitiveIndex)
{
	PrimitiveHistogram histogram;
	histogram.Init(0);

	if (!g_GlobalConstants.EnablePostDispatchInfoStats && !g_GlobalConstants.EnableSpecialIndices)
		return histogram;

	const uint3 counts = OMM_SUBRESOURCE_LOAD3(SpecialIndicesStateBuffer, 12 * primitiveIndex);
	histogram.Init(counts);
	return histogram;
}

void UpdatePostBuildInfo(PrimitiveHistogram h)
{
	// UGH. must match format of PostBuildInfo...
	const uint kOffsetTotalOpaque = 2;
	const uint kOffsetTotalTransparent = 3;
	const uint kOffsetTotalUnknown = 4;
	const uint kOffsetTotalFullyOpaque = 5;
	const uint kOffsetTotalFullyTransparent = 6;
	const uint kOffsetTotalFullyUnknown = 7;

	if (g_GlobalConstants.EnableSpecialIndices && h.IsSpecialIndex())
	{
		const SpecialIndex specialIndex = h.GetSpecialIndex();

		if (specialIndex == SpecialIndex::FullyOpaque)
			u_postBuildInfo.InterlockedAdd(4 * kOffsetTotalFullyOpaque, 1);
		if (specialIndex == SpecialIndex::FullyTransparent)
			u_postBuildInfo.InterlockedAdd(4 * kOffsetTotalFullyTransparent, 1);
		if (specialIndex == SpecialIndex::FullyUnknownTransparent ||
			specialIndex == SpecialIndex::FullyUnknownOpaque)
			u_postBuildInfo.InterlockedAdd(4 * kOffsetTotalFullyUnknown, 1);
	}
	else
	{
		if (h.counts.x != 0)
			u_postBuildInfo.InterlockedAdd(4 * kOffsetTotalOpaque, h.counts.x);
		if (h.counts.y != 0)
			u_postBuildInfo.InterlockedAdd(4 * kOffsetTotalTransparent, h.counts.y);
		if (h.counts.z != 0)
			u_postBuildInfo.InterlockedAdd(4 * kOffsetTotalUnknown, h.counts.z);
	}
}

bool GetSpecialIndex(uint primitiveIndex, out SpecialIndex specialIndex)
{
	if (!g_GlobalConstants.EnableSpecialIndices)
		return false;

	const uint3 counts = OMM_SUBRESOURCE_LOAD3(SpecialIndicesStateBuffer, 12 * primitiveIndex);

	if (counts.x >= 1 && counts.y == 0 && counts.z == 0)
	{
		specialIndex = SpecialIndex::FullyOpaque;
		return true;
	}
	else if (counts.x == 0 && counts.y >= 1 && counts.z == 0)
	{
		specialIndex = SpecialIndex::FullyTransparent;
		return true;
	}
	else if (counts.x == 0 && counts.y == 0 && counts.z >= 1)
	{
		specialIndex = SpecialIndex::FullyUnknownOpaque;
		return true;
	}
	return false;
}

uint GetSourcePrimitiveIndex(uint primitiveIndex)
{
	if (g_GlobalConstants.DoSetup)
	{
		const int primitiveIndexOrHashTableEntryIndex = OMM_SUBRESOURCE_LOAD(TempOmmIndexBuffer, 4 * primitiveIndex);

		if (primitiveIndexOrHashTableEntryIndex < -4)
		{
			const uint hashTableEntryIndex = -(primitiveIndexOrHashTableEntryIndex + 5);
			const uint primitiveIndexRef = OMM_SUBRESOURCE_LOAD(HashTableBuffer, 8 * hashTableEntryIndex + 4); // [hash|primitiveIndex]
			return primitiveIndexRef;
		}

		return primitiveIndex; // Source and dest is the same => no reuse
	}
	else
	{
		const int ommDescOffsetOrPrimitiveIndex = OMM_SUBRESOURCE_LOAD(TempOmmIndexBuffer, 4 * primitiveIndex);

		if (ommDescOffsetOrPrimitiveIndex < -4)
		{
			const uint primitiveIndexRef = -(ommDescOffsetOrPrimitiveIndex + 5);
			return primitiveIndexRef;
		}

		return primitiveIndex; // Source and dest is the same => no reuse
	}
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

	const PrimitiveHistogram histogram = LoadHistogram(srcPrimitiveIndex);

	// We only update this once.
	if (srcPrimitiveIndex == dstPrimitiveIndex && g_GlobalConstants.EnablePostDispatchInfoStats)
		UpdatePostBuildInfo(histogram);

	if (g_GlobalConstants.EnableSpecialIndices && histogram.IsSpecialIndex())
	{
		SpecialIndex specialIndex = histogram.GetSpecialIndex();

		OMM_SUBRESOURCE_STORE(TempOmmIndexBuffer, 4 * dstPrimitiveIndex, specialIndex);
	}
	else
	{
		const int ommDescIndex = OMM_SUBRESOURCE_LOAD(TempOmmIndexBuffer, 4 * srcPrimitiveIndex);

		if (ommDescIndex >= 0)
		{
			IncrementIndexHistogram(ommDescIndex);
		}

		if (srcPrimitiveIndex != dstPrimitiveIndex)
		{
			OMM_SUBRESOURCE_STORE(TempOmmIndexBuffer, 4 * dstPrimitiveIndex, ommDescIndex);
		}
	}
}