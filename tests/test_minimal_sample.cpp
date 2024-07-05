/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#include <omm.hpp>
#include <shared/math.h>

namespace {

	TEST(MinimalSample, CPU) 
	{
		// This sample will demonstrate the use of OMMs on a triangle fan modeled on top of a donut.
		const float rMin = 0.2f; // Circle inner radius.
		const float rMax = 0.3f; // Circle outer radius.

		// The pixels in our alpha texture. 
		// Here we'll create a procedural image (circle).
		// In practice we'll load image from disk, and may have to run compression / decompression before storing it.
		const uint32_t alphaTextureWidth = 256;
		const uint32_t alphaTextureHeight = 256;
		std::vector<float> alphaTextureDataFP32;
		alphaTextureDataFP32.reserve(alphaTextureWidth * alphaTextureHeight);
		for (uint32_t j = 0; j < alphaTextureHeight; ++j)
		{
			for (uint32_t i = 0; i < alphaTextureWidth; ++i)
			{
				const int2 idx = int2(i, j);
				const float2 uv = float2(idx) / float2((float)alphaTextureWidth);
				const float alphaValue = glm::length(uv - 0.5f) > rMin && glm::length(uv - 0.5f) < rMax ? 1.f : 0.f;
				alphaTextureDataFP32.push_back(alphaValue);
			}
		}

		// Here we'll setup a triangle "diamond" of 4 triangles in total that covers our circle.
		std::vector<float2> texCoordBuffer = 
		{ 
			{0.05f, 0.50f},
			{0.50f, 0.05f},
			{0.50f, 0.50f},
			{0.95f, 0.50f},
			{0.50f, 0.95f},
		};

		std::vector<uint32_t> indexBuffer =
		{
			0, 1, 2,
			1, 3, 2,
			3, 4, 2,
			2, 4, 0,
		};

		std::vector<uint8_t> subdivisionLevels =
		{
			2,
			3,
			4,
			5,
		};

		omm::BakerCreationDesc desc;
		desc.type = omm::BakerType::CPU;
		desc.enableValidation = true;
		// desc.memoryAllocatorInterface = ...; // If we prefer to track memory allocations and / or use custom memory allocators we can override these callbacks. But it's not required.

		omm::Baker bakerHandle; // Create the baker instance. This instance can be shared among all baking tasks. Typucally one per application.

		omm::Result res = omm::CreateBaker(desc, &bakerHandle);
		ASSERT_EQ(res, omm::Result::SUCCESS);

		// Since we configured the CPU baker we are limited to the functions in the ::Cpu namespace
		// First we create our input texture data.
		// The texture object can be reused between baking passes.

		omm::Cpu::TextureMipDesc mipDesc;
		mipDesc.width = alphaTextureWidth;
		mipDesc.height = alphaTextureHeight;
		mipDesc.textureData = alphaTextureDataFP32.data();

		omm::Cpu::TextureDesc texDesc;
		texDesc.format = omm::Cpu::TextureFormat::FP32;
		texDesc.mipCount = 1;
		texDesc.mips = &mipDesc;

		omm::Cpu::Texture textureHandle;
		res = omm::Cpu::CreateTexture(bakerHandle, texDesc, &textureHandle);
		ASSERT_EQ(res, omm::Result::SUCCESS);

		// Setup the baking parameters, setting only required data.
		omm::Cpu::BakeInputDesc bakeDesc;
		bakeDesc.bakeFlags = omm::Cpu::BakeFlags::None; // Default bake flags.
		// Texture object
		bakeDesc.texture = textureHandle;
		// Alpha test parameters.
		bakeDesc.alphaCutoff = 0.5f;
		bakeDesc.alphaMode = omm::AlphaMode::Test;
		bakeDesc.runtimeSamplerDesc = { .addressingMode = omm::TextureAddressMode::Clamp, .filter = omm::TextureFilterMode::Linear };
		
		// Input geometry / texcoords
		bakeDesc.texCoordFormat = omm::TexCoordFormat::UV32_FLOAT;
		bakeDesc.texCoordStrideInBytes = sizeof(float2);
		bakeDesc.texCoords = texCoordBuffer.data();
		bakeDesc.indexBuffer = indexBuffer.data();
		bakeDesc.indexCount = (uint32_t)indexBuffer.size();
		bakeDesc.indexFormat = omm::IndexFormat::UINT_32;
		bakeDesc.subdivisionLevels = subdivisionLevels.data();
		// Desired output config
		bakeDesc.format = omm::Format::OC1_2_State;
		bakeDesc.unknownStatePromotion = omm::UnknownStatePromotion::ForceOpaque;
		// leave the rest of the parameters to default.

		// perform the baking... processing time may vary depending on triangle count, triangle size, subdivision level and texture size.
		omm::Cpu::BakeResult bakeResultHandle;
		res = omm::Cpu::Bake(bakerHandle, bakeDesc, &bakeResultHandle);
		ASSERT_EQ(res, omm::Result::SUCCESS);

		// Read back the result.
		const omm::Cpu::BakeResultDesc* bakeResultDesc = nullptr;
		res = omm::Cpu::GetBakeResultDesc(bakeResultHandle, &bakeResultDesc);
		ASSERT_EQ(res, omm::Result::SUCCESS);

		// ... 
		// Consume data
		// Copy the bakeResultDesc data to GPU buffers directly, or cache to disk for later consumption.
		// ....

		// Visualize the bake result in a .png file
#if OMM_TEST_ENABLE_IMAGE_DUMP
		const bool debug = true;
		if (debug)
		{
			omm::Debug::SaveAsImages(bakerHandle, bakeDesc, bakeResultDesc,
				{ 
					.path = "MinimalSample",
					.oneFile = true /* Will draw all triangles in the same file.*/
				});
		}
#endif
		
		// Cleanup. Result no longer needed
		res = omm::Cpu::DestroyBakeResult(bakeResultHandle);
		ASSERT_EQ(res, omm::Result::SUCCESS);
		// Cleanup. Texture no longer needed
		res = omm::Cpu::DestroyTexture(bakerHandle, textureHandle);
		ASSERT_EQ(res, omm::Result::SUCCESS);
		// Cleanup. Baker no longer needed
		res = omm::DestroyBaker(bakerHandle);
		ASSERT_EQ(res, omm::Result::SUCCESS);
	}

}  // namespace