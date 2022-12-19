/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// This function returns the texture coordinate for a given vertex and primitive index,
// i_pos is a vertex in the pre-generated triangle subdivision mesh that follows the bird curve pattern.
// each i_pos is the discrete 2D barycentric coordinate packed.
// The resulting texture coord is then mapped to a raster position in the vertex shader.

float2 GetTexCoordForVertex(
	uint		i_pos,				// : POSITION
	uint		i_primitiveIndex	// : SV_InstanceID
)
{
	const uint primitiveIndexOffset 		= (g_LocalConstants.PrimitiveIdOffset + i_primitiveIndex) + g_LocalConstants.SubdivisionLevel * g_GlobalConstants.PrimitiveCount;

	const uint vmFormatAndPrimitiveIndex	= OMM_SUBRESOURCE_LOAD(RasterItemsBuffer, primitiveIndexOffset * 8 + 4);
	const OMMFormat vmFormat				= (OMMFormat)(vmFormatAndPrimitiveIndex >> 30);
	const uint primitiveIndex				= vmFormatAndPrimitiveIndex & 0x3FFFFFFF;

	uint3 indices;
	indices.x = t_indexBuffer[primitiveIndex * 3 + 0];
	indices.y = t_indexBuffer[primitiveIndex * 3 + 1];
	indices.z = t_indexBuffer[primitiveIndex * 3 + 2];

	float2 vertexUVs[3];
	vertexUVs[0] = asfloat(t_texCoordBuffer.Load2(g_GlobalConstants.TexCoord1Offset + indices.x * g_GlobalConstants.TexCoord1Stride));
	vertexUVs[1] = asfloat(t_texCoordBuffer.Load2(g_GlobalConstants.TexCoord1Offset + indices.y * g_GlobalConstants.TexCoord1Stride));
	vertexUVs[2] = asfloat(t_texCoordBuffer.Load2(g_GlobalConstants.TexCoord1Offset + indices.z * g_GlobalConstants.TexCoord1Stride));

	const float2 p0 =   vertexUVs[0];
	const float2 v0 =  (vertexUVs[1] - p0);
	const float2 v1 =   vertexUVs[2] - p0;

	const uint N		= 1u << g_LocalConstants.SubdivisionLevel;
	const uint i		= (i_pos & 0xFFFF);
	const uint j		= ((i_pos >> 16) & 0xFFFF);
	const float2 pos	= float2((float)i, (float)j) / float(N);

	const float2 texCoord = pos.x * v0 + (1.f - pos.y) * v1 + p0;

	return texCoord;
}