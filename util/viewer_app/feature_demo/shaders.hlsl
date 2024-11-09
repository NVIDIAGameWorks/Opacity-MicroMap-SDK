/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include "shader_cb.h"


struct OmmDesc
{
    uint32_t offset;
    uint16_t subdivisionLevel;
    uint16_t format;
};

SamplerState s_Sampler : register(s0);
Texture2D t_Texture : register(t0);
Buffer<uint> t_OmmIndexBuffer : register(t1);
StructuredBuffer<OmmDesc> t_OmmDesc : register(t2);
ByteAddressBuffer t_OmmArrayData : register(t3);

cbuffer c_Constants : register(b0)
{
    Constants g_constants;
};

void main_vs(
	in float2 i_texCoord : SV_Position,
	out float2 o_texCoord : TEXCOORD0,
	out float4 o_pos : SV_Position
)
{
    o_texCoord = i_texCoord;
    float2 vert = 2 * i_texCoord - 1.0;
    
    vert += g_constants.offset;
    vert *= g_constants.zoom;

	o_pos = float4(vert, 0, 1);
}

static inline uint prefixEor2(uint x)
{
    x ^= (x >> 1) & 0x7fff7fff;
    x ^= (x >> 2) & 0x3fff3fff;
    x ^= (x >> 4) & 0x0fff0fff;
    x ^= (x >> 8) & 0x00ff00ff;
    return x;
}

// Interleave 16 even bits from x with 16 odd bits from y
static inline uint interleaveBits2(uint x, uint y)
{
    x = (x & 0xffff) | (y << 16);
    x = ((x >> 8) & 0x0000ff00) | ((x << 8) & 0x00ff0000) | (x & 0xff0000ff);
    x = ((x >> 4) & 0x00f000f0) | ((x << 4) & 0x0f000f00) | (x & 0xf00ff00f);
    x = ((x >> 2) & 0x0c0c0c0c) | ((x << 2) & 0x30303030) | (x & 0xc3c3c3c3);
    x = ((x >> 1) & 0x22222222) | ((x << 1) & 0x44444444) | (x & 0x99999999);

    return x;
}

static uint dbary2index(uint u, uint v, uint w, uint level)
{
    const uint coordMask = ((1U << level) - 1);

    uint b0 = ~(u ^ w) & coordMask;
    uint t = (u ^ v) & b0; //  (equiv: (~u & v & ~w) | (u & ~v & w))
    uint c = (((u & v & w) | (~u & ~v & ~w)) & coordMask) << 16;
    uint f = prefixEor2(t | c) ^ u;
    uint b1 = (f & ~b0) | t; // equiv: (~u & v & ~w) | (u & ~v & w) | (f0 & u & ~w) | (f0 & ~u & w))

    return interleaveBits2(b0, b1); // 13 instructions
}

static uint bary2index(float2 bc, uint level, out bool isUpright)
{
    float numSteps = float(1u << level);
    uint iu = uint(numSteps * bc.x);
    uint iv = uint(numSteps * bc.y);
    uint iw = uint(numSteps * (1.f - bc.x - bc.y));
    isUpright = (iu & 1) ^ (iv & 1) ^ (iw & 1);
    return dbary2index(iu, iv, iw, level);
}

float3 MicroStateColor(int state)
{
    if (state == 0)
        return float3(0, 0, 1.f);
    if (state == 1)
        return float3(0, 1, 0.f);
    if (state == 2)
        return float3(1.f, 0, 1.f);
    //if (state == 3)
        return float3(1.f, 1.f, 0.f);
}

void main_ps(
	in float2 i_texCoord : TEXCOORD0,
	in uint i_primitiveId : SV_PrimitiveID,
    in float3 bc : SV_Barycentrics,
    in bool isFrontFace : SV_IsFrontFace,
	out float4 o_color : SV_Target0
)
{
    if (g_constants.mode == 1) // wireframe
    {
        o_color = float4(1, 0, 0, 1.0);
        return;
    }
    int ommIndex = t_OmmIndexBuffer[i_primitiveId + g_constants.primitiveOffset];
    
    if (ommIndex < 0)
    {
        o_color = float4(MicroStateColor(-(ommIndex + 1)), 0.5);
        return;
    }
    
    OmmDesc ommDesc = t_OmmDesc[ommIndex];
    const bool is2State = ommDesc.format == 1;
    
    bool isUpright;
    const uint microIndex = bary2index(bc.yz, ommDesc.subdivisionLevel, isUpright);
    const uint statesPerDW = is2State ? 32 : 16;
    const uint startOffset = ommDesc.offset;
    const uint offsetDW = startOffset + 4 * (microIndex / statesPerDW);
    uint stateDW = t_OmmArrayData.Load(offsetDW);
    const uint bitOffset = (is2State ? 1 : 2) * (microIndex % statesPerDW);
    const uint state = (stateDW >> bitOffset) & (is2State ? 0x1u : 0x3u);
    float3 clr = MicroStateColor(state);

    clr *= 0.5;
    if (isUpright)
    {
        clr *= 0.5f;
    }
    
    float alphaLerp = t_Texture.SampleLevel(s_Sampler, i_texCoord, 0).r;
    const float e = 0.0001f;
    const float alpha00 = t_Texture.SampleLevel(s_Sampler, i_texCoord, 0).r;
    const float alpha01 = t_Texture.SampleLevel(s_Sampler, i_texCoord + float2(e, 0.f), 0).r;
    const float alpha10 = t_Texture.SampleLevel(s_Sampler, i_texCoord + float2(0.0f, e), 0).r;
    const float alpha11 = t_Texture.SampleLevel(s_Sampler, i_texCoord + float2(e, e), 0).r;
    const float4 alpha = float4(alpha00, alpha01, alpha10, alpha11);
    
    const bool isIntersection = any(alpha < 0.5f) && any(alpha >= 0.5f);
    
    float3 color = float3(0, 0, 0);
    if (isIntersection)
    {
        o_color = float4(1, 0, 0, 1.0);
        return;
    }
    else
    {
        color = 0.01 * alphaLerp.xxx;
    }

    o_color = float4(clr.xyz + 0.5 * color, 1.0);
}
