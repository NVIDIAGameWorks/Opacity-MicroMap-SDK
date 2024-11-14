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

static const float3 kContourLineColor = float3(0.8f, 0, 0);

bool IsOverIntersectionLine(Texture2D texture, SamplerState s, float2 invTexSize, float alphaCutoff, float2 uv)
{
    float alphaLerp = texture.SampleLevel(s, uv, 0).r;
    const float2 e = 0.35f * invTexSize;
    const float alpha00 = texture.SampleLevel(s, uv, 0).r;
    const float alpha01 = texture.SampleLevel(s, uv + float2(e.x, 0.f), 0).r;
    const float alpha10 = texture.SampleLevel(s, uv + float2(0.0f, e.y), 0).r;
    const float alpha11 = texture.SampleLevel(s, uv + e, 0).r;
    const float4 alpha = float4(alpha00, alpha01, alpha10, alpha11);
    
    const bool isIntersection = any(alpha < alphaCutoff) && any(alpha >= alphaCutoff);
    return isIntersection;
}