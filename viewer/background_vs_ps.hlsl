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
#include "util.hlsli"

SamplerState s_SamplerAniso : register(s0);
SamplerState s_SamplerAnisoCmp : register(s1);
RWTexture2D<float4> u_ReadbackTexture : register(u0);
Texture2D t_Texture : register(t0);
Texture2D t_TextureMin : register(t1);
Texture2D t_TextureMax : register(t2);
Buffer<uint> t_OmmIndexBuffer : register(t3);

cbuffer c_Constants : register(b0)
{
    Constants g_constants;
};

float2 GetTextureUvFromScreenPos(float2 screenPos)
{
    float2 uv = screenPos;
    uv -= float2(0.5, 0.5);
    uv /= g_constants.zoom;
    uv /= g_constants.aspectRatio;
    uv += float2(0.5, 0.5);
    uv -= 0.5 * g_constants.offset;
    return uv;
}

void main_vs(
	in uint iVertex : SV_VertexID,
	out float4 o_posClip : SV_Position,
	out float2 o_uv : UV0,
    out float2 o_uvDelta : UV1)
{
    uint u = iVertex & 1;
    uint v = (iVertex >> 1) & 1;
	
    o_posClip = float4(float(u) * 2 - 1, 1 - float(v) * 2, 0, 1);
    
    float2 uv = float2(u, 1.f -v);
    o_uv = GetTextureUvFromScreenPos(uv);
    
    float2 uvDelta = float2(u + g_constants.invScreenSize.x, 1.f - v + g_constants.invScreenSize.y);
    o_uvDelta = GetTextureUvFromScreenPos(uvDelta);
}

void main_ps(
	in float4 i_pos : SV_Position,
	in float2 i_uv : UV0,
	in float2 i_uvDelta : UV1,
	out float4 o_rgba : SV_Target)
{
    uint2 dim;
    uint mipNum;
    t_Texture.GetDimensions(0, dim.x, dim.y, mipNum);
    
    int2 texel = (int2) round(i_uv * dim);
    int2 texelDelta = (int2) round(i_uvDelta * dim);
    const float2 texelf = (float2) i_uv * dim;
    
    const float alpha = t_Texture.Sample(s_SamplerAniso, i_uv).r;
    u_ReadbackTexture[i_pos.xy] = float4(alpha, asfloat(texel), 0);
    
    if (g_constants.mouseCoordX == texel.x && g_constants.mouseCoordY == texel.y)
    {
        float2 fracTexelf = frac(texelf - 0.5f);
        
        const float e = 2 * ddx(texelf.x);
        if (fracTexelf.x < e || fracTexelf.x > (1 - e) || 
            fracTexelf.y < e || fracTexelf.y > (1 - e))
        {
            o_rgba = float4(1, 1, 0, 0.5);
            return;
        }
    }

    float3 checker = float3(0, 0, 0);
    
    const float size = 0.5f;
    
    const float maxDD = max(max(ddx(texelf.x), ddx(texelf.y)), max(ddy(texelf.x), ddy(texelf.y)));
    
    const float fade = clamp(maxDD / size, 0, 1);
    
    // if (maxDD < size)  // Micro Tiles (entire textures)
    {
        const float clr = lerp(0.3f, 0.0f, fade);
        if ((texel.x & 1) == (texel.y & 1))
        {
            checker += clr.xxx;
        }
    }

    { // Macro Tiles (entire textures)
        const float clr = lerp(0.0f, 0.1f, fade);
        if (((texel.x / dim.x) & 1) == ((texel.y / dim.y) & 1))
        {
           // checker += float3(clr, clr, clr);
        }
        else
        {
           // checker += float3(0, 0, clr);
        }
    }
    
    // Draw vertical and horizontal lines to outline macro textures
    if (texel.x / dim.x != texelDelta.x / dim.x || 
        texel.y / dim.y != texelDelta.y / dim.y)
    {
        checker = float3(0.0, 1.0, 1.0);
    }
    
    bool isIntersection = IsOverIntersectionLine(t_Texture, t_TextureMin, t_TextureMax, s_SamplerAnisoCmp, g_constants.invTexSize, g_constants.alphaCutoff, i_uv);
    
    if (g_constants.drawAlphaContour && isIntersection)
    {
        o_rgba = float4(kContourLineColor, 1.0);
    }
    else
    {
        o_rgba = float4(alpha.xxx + checker, 0.5);
    }
}
