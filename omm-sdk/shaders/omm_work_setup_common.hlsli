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

/// Unrolled version of murmur hash that takes N integers as input (up to 8)
uint murmur_32_scramble(uint k)
{
	k *= 0xcc9e2d51;
	k = (k << 15) | (k >> 17);
	k *= 0x1b873593;
	return k;
};

uint murmur_32_process(uint k, uint h)
{
	h ^= murmur_32_scramble(k);
	h = (h << 13) | (h >> 19);
	h = h * 5 + 0xe6546b64;
	return h;
};

uint murmurUint7(
	uint key0,
	uint key1,
	uint key2,
	uint key3,
	uint key4,
	uint key5,
	uint key6,
	uint SEED)
{
	uint h = SEED;
	h = murmur_32_process(key0, h);
	h = murmur_32_process(key1, h);
	h = murmur_32_process(key2, h);
	h = murmur_32_process(key3, h);
	h = murmur_32_process(key4, h);
	h = murmur_32_process(key5, h);
	h = murmur_32_process(key6, h);

	// A swap is *not* necessary here because the preceding loop already
	// places the low bytes in the low places according to whatever endianness
	// we use. Swaps only apply when the memory is copied in a chunk.
	h ^= murmur_32_scramble(0);
	/* Finalize. */
	h ^= 24; // len = 6 * sizeof(float)
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
	return h;
};

uint GetOMMFormatBitCount(OMMFormat vmFormat)
{
	return (uint)vmFormat;
}

struct TexCoords
{
	float2 p0;
	float2 p1;
	float2 p2;

	void Init(float2 _p0, float2 _p1, float2 _p2) {
		p0 = _p0;
		p1 = _p1;
		p2 = _p2;
	}
};

uint GetHash(TexCoords tex, uint subdivisionLevel)
{
	const uint seed = 1337;

	return murmurUint7(
		asuint(tex.p0.x),
		asuint(tex.p0.y), 
		asuint(tex.p1.x), 
		asuint(tex.p1.y),
		asuint(tex.p2.x),
		asuint(tex.p2.y),
		asuint(subdivisionLevel),
		seed);
}

namespace hashTable
{
	enum class Result
	{
		Null, // not initialized

		Found,
		Inserted,
		ReachedMaxAttemptCount, //
	};

	// returns offset in hash table for a given hash,
	// if Inserted, return value will be input value.
	// if Found, return value will be value at hash table entry location
	// if ReachedMaxAttemptCount, return value is undefined.
	Result FindOrInsertValue(uint hash, out uint hashTableEntryIndex)
	{
		hashTableEntryIndex = hash % g_GlobalConstants.TexCoordHashTableEntryCount;

		const uint kMaxNumAttempts		= 16;	// Completely arbitrary.
		const uint kInvalidEntryHash	= 0;	// Buffer must be cleared before dispatch.

		ALLOW_UAV_CONDITION
		for (uint attempts = 0; attempts < kMaxNumAttempts; ++attempts)
		{
			uint existingHash = 0;
			OMM_SUBRESOURCE_CAS(HashTableBuffer, 8 * hashTableEntryIndex, kInvalidEntryHash, hash, existingHash); // Each entry consists of [hash|primitiveId]

			// Inserted.
			if (existingHash == kInvalidEntryHash)
				return Result::Inserted;

			// Entry was already inserted.
			if (existingHash == hash)
				return Result::Found;

			// Conflict, keep searching using lienar probing 
			hashTableEntryIndex = (hashTableEntryIndex + 1) % g_GlobalConstants.TexCoordHashTableEntryCount;
		}

		return Result::ReachedMaxAttemptCount;
	}

	void Store(uint hashTableEntryIndex, uint value)
	{
		OMM_SUBRESOURCE_STORE(HashTableBuffer, 8 * hashTableEntryIndex + 4, value);
	}
}

TexCoords FetchTexCoords(uint primitiveIndex)
{
	uint3 indices;
	indices.x		= t_indexBuffer[primitiveIndex * 3 + 0];
	indices.y		= t_indexBuffer[primitiveIndex * 3 + 1];
	indices.z		= t_indexBuffer[primitiveIndex * 3 + 2];

	float2 vertexUVs[3];
	vertexUVs[0] = asfloat(t_texCoordBuffer.Load2(g_GlobalConstants.TexCoord1Offset + indices.x * g_GlobalConstants.TexCoord1Stride));											   
	vertexUVs[1] = asfloat(t_texCoordBuffer.Load2(g_GlobalConstants.TexCoord1Offset + indices.y * g_GlobalConstants.TexCoord1Stride));											   
	vertexUVs[2] = asfloat(t_texCoordBuffer.Load2(g_GlobalConstants.TexCoord1Offset + indices.z * g_GlobalConstants.TexCoord1Stride));

	TexCoords tex;
	tex.Init(vertexUVs[0], vertexUVs[1], vertexUVs[2]);
	return tex;
}

uint GetNextPow2(uint v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

float GetArea2D(float2 p0, float2 p1, float2 p2) {
	const float2 v0 = p2 - p0;
	const float2 v1 = p1 - p0;
	return 0.5f * length(cross(float3(v0, 0), float3(v1, 0)));
}

uint GetLog2(uint v) { // V must be power of 2.
	const unsigned int b[5] = { 0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0,
									 0xFF00FF00, 0xFFFF0000 };
	unsigned int r = (v & b[0]) != 0;
	for (uint i = 4; i > 0; i--) // unroll for speed...
	{
		r |= ((v & b[i]) != 0) << i;
	}
	return r;
};

uint GetDynamicSubdivisionLevel(TexCoords tex, float scale)
{
	const float pixelUvArea = GetArea2D(tex.p0 * g_GlobalConstants.TexSize, tex.p1 * g_GlobalConstants.TexSize, tex.p2 * g_GlobalConstants.TexSize);

	// Solves the following eqn:
	// targetPixelArea / (4^N) = pixelUvArea 

	const float targetPixelArea = scale * scale;
	const uint ratio			= pixelUvArea / targetPixelArea;
	const uint ratioNextPow2	= GetNextPow2(ratio);
	const uint log2_ratio		= GetLog2(ratioNextPow2);

	const uint SubdivisionLevel = log2_ratio >> 1u; // log2(ratio) / log2(4)

	return min(SubdivisionLevel, g_GlobalConstants.MaxSubdivisionLevel);
}

uint InterlockedAdd(RWByteAddressBuffer buffer, uint offset, uint increment)
{
#if 1
	uint val = 0;
	buffer.InterlockedAdd(offset, increment, val);
	return val;
#else
	const uint totalIncrement		= WaveActiveSum(increment);
	const uint localOffset			= WavePrefixSum(increment);
	
	uint globalOffset = 0;
	if (WaveIsFirstLane())
	{
		buffer.InterlockedAdd(offset, totalIncrement, globalOffset);
	}
	return WaveReadLaneFirst(globalOffset) + localOffset;
#endif
}

uint GetNumMicroTriangles(uint subdivisionLevel)
{
	return 1u << (subdivisionLevel << 1u);
}

uint GetMaxItemsPerBatch(uint subdivisionLevel)
{
	const uint numMicroTri				= GetNumMicroTriangles(subdivisionLevel);
	const uint rasterItemByteSize		= (numMicroTri) * 8; // We need 2 x uint32 for each micro-VM state.
	return g_GlobalConstants.BakeResultBufferSize / rasterItemByteSize;
}

uint GetSubdivisionLevel(TexCoords texCoords)
{
	const bool bEnableDynamicSubdivisionLevel = g_GlobalConstants.DynamicSubdivisionScale > 0.f;

	if (bEnableDynamicSubdivisionLevel)
	{
		return GetDynamicSubdivisionLevel(texCoords, g_GlobalConstants.DynamicSubdivisionScale);
	}
	else
	{
		return g_GlobalConstants.GlobalSubdivisionLevel;
	}
}

hashTable::Result FindOrInsertOMMEntry(TexCoords texCoords, uint subdivisionLevel, out uint hashTableEntryIndex)
{
	hashTableEntryIndex = 0;
	if (g_GlobalConstants.EnableTexCoordDeduplication)
	{
		const uint hash = GetHash(texCoords, subdivisionLevel);

		return hashTable::FindOrInsertValue(hash, hashTableEntryIndex);
	}
	else
	{
		return hashTable::Result::Null;
	}
}