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

#include <fstream>

#pragma optimize("", off)

class Profiler {
public:
	// Start the timer when the Profiler object is created
	Profiler(const std::string& funcName) : functionName(funcName), start(std::chrono::high_resolution_clock::now()) {}

	// Stop the timer when the Profiler object goes out of scope
	~Profiler() {
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed = end - start;
		std::cout << "Function [" << functionName << "] took " << elapsed.count() << " ms" << std::endl;
	}

private:
	std::string functionName;
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

#define PROFILE_SCOPE(name) Profiler profiler##__LINE__(name);

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
		desc.messageInterface.messageCallback = [](omm::MessageSeverity severity, const char* message, void* userArg) {
			std::cout << "[omm-sdk]: " << message << std::endl;
		};

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
		bakeDesc.bakeFlags = omm::Cpu::BakeFlags::EnableWorkloadValidation;
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

	TEST(ReadFromFile, CPU)
	{
		// This sample will demonstrate the use of OMMs on a triangle fan modeled on top of a donut.

		omm::BakerCreationDesc desc;
		desc.type = omm::BakerType::CPU;
		desc.messageInterface.messageCallback = [](omm::MessageSeverity severity, const char* message, void* userArg) {
			std::cout << "[omm-sdk]: " << message << std::endl;
			};

		// desc.memoryAllocatorInterface = ...; // If we prefer to track memory allocations and / or use custom memory allocators we can override these callbacks. But it's not required.

		omm::Baker bakerHandle; // Create the baker instance. This instance can be shared among all baking tasks. Typucally one per application.
		omm::Result res = omm::CreateBaker(desc, &bakerHandle);
		ASSERT_EQ(res, omm::Result::SUCCESS);


		auto readFile = [](const char* filename)->std::vector<char>
			{
				// open the file:
				std::ifstream file(filename, std::ios::binary);

				// read the data:
				return std::vector<char>((std::istreambuf_iterator<char>(file)),
					std::istreambuf_iterator<char>());
			};

		auto data = readFile("C:\\Users\\jdeligiannis\\Downloads\\myExpensiveBakeJob_80mb.bin");

		omm::Cpu::BlobDesc blob;
		blob.data = data.data();
		blob.size = data.size();

		ommCpuDeserializedResult deserializedResult;
		res = omm::Cpu::Deserialize(bakerHandle, blob, &deserializedResult);
		ASSERT_EQ(res, omm::Result::SUCCESS);


		const omm::Cpu::DeserializedDesc* desDesc = nullptr;
		res = omm::Cpu::GetDeserializedDesc(deserializedResult, &desDesc);
		ASSERT_EQ(res, omm::Result::SUCCESS);
		ASSERT_EQ(desDesc->numInputDescs, 1);
		ASSERT_EQ(desDesc->numResultDescs, 0);

		if (false)
		{
			omm::Cpu::DeserializedDesc desDescCopy = *desDesc;
			desDescCopy.flags = omm::Cpu::SerializeFlags::Compress;
			ommCpuSerializedResult serializedResult;
			res = omm::Cpu::Serialize(bakerHandle, desDescCopy, &serializedResult);
			ASSERT_EQ(res, omm::Result::SUCCESS);

			const omm::Cpu::BlobDesc* blob;
			res = omm::Cpu::GetSerializedResultDesc(serializedResult, &blob);
			ASSERT_EQ(res, omm::Result::SUCCESS);

			res = omm::Debug::SaveBinaryToDisk(bakerHandle, *blob, "C:\\Users\\jdeligiannis\\Downloads\\myExpensiveBakeJob_80mb_compress.bin");
			ASSERT_EQ(res, omm::Result::SUCCESS);

			res = omm::Cpu::DestroySerializedResult(serializedResult);
			ASSERT_EQ(res, omm::Result::SUCCESS);
		}

		// Setup the baking parameters, setting only required data.
		omm::Cpu::BakeInputDesc bakeDesc = desDesc->inputDescs[0];

		// Adjust the workload size

		uint32_t flags = (uint32_t)(bakeDesc.bakeFlags) |(uint32_t)omm::Cpu::BakeFlags::DisableSpecialIndices;

		int method = 3;

		if (method == 0)
		{
			// onds to roughly 2464 1024x1024 textures. This is unusually large and may result in long bake times.
			//flags |= 1u << 9u; // DisableFineClassification
			//flags |= 1u << 11u; // EnableWrapping
			//flags |= 1u << 12u; // EnableSnapping

			// totalOpaque                  7055132
			// totalTransparent             196578
			// totalUnknownTransparent      0
			// totalUnknownOpaque           1243394
			// totalFullyOpaque             0
			// totalFullyTransparent        0
			// totalFullyUnknownOpaque      0
			// totalFullyUnknownTransparent 0

			// Function [Bake] took 415.433 ms
		}
		else if (method == 1)
		{
			// onds to roughly 2464 1024x1024 textures. This is unusually large and may result in long bake times.
			flags |= 1u << 9u; // DisableFineClassification
			flags |= 1u << 11u; // EnableWrapping
			//flags |= 1u << 12u; // EnableSnapping

			// totalOpaque                  7055118
			// totalTransparent             196576
			// totalUnknownTransparent      0
			// totalUnknownOpaque           1243410
			// totalFullyOpaque             0
			// totalFullyTransparent        0
			// totalFullyUnknownOpaque      0
			// totalFullyUnknownTransparent 0

			// Function [Bake] took 410.791 ms
			// unknown ~14.6%
		}
		else if (method == 2)
		{
			// onds to roughly 2464 1024x1024 textures. This is unusually large and may result in long bake times
			// totalOpaque                  7475534
			// totalTransparent             221013
			// totalUnknownTransparent      0
			// totalUnknownOpaque           798557
			// totalFullyOpaque             0
			// totalFullyTransparent        0
			// totalFullyUnknownOpaque      0
			// totalFullyUnknownTransparent 0

			// Function [Bake] took 78628.3 ms
			// unknown ~9.4%
		}
		else if (method == 3)
		{
			flags |= 1u << 13u; // EnableBetterCoarseClassification

			// onds to roughly 2464 1024x1024 textures. This is unusually large and may result in long bake times
			// totalOpaque                  x
			// totalTransparent             x
			// totalUnknownTransparent      0
			// totalUnknownOpaque           x
			// totalFullyOpaque             0
			// totalFullyTransparent        0
			// totalFullyUnknownOpaque      0
			// totalFullyUnknownTransparent 0
			//onds to roughly 2464 1024x1024 textures.This is unusually large and may result in long bake times.
			//	Function[Bake] took 66458.6 ms
		}

		bakeDesc.bakeFlags = (omm::Cpu::BakeFlags)flags;
		bakeDesc.maxWorkloadSize = 0xFFFFFFFFFFFFFFFF;


		// perform the baking... processing time may vary depending on triangle count, triangle size, subdivision level and texture size.
		// Read back the result.
		omm::Cpu::BakeResult bakeResultHandle;
		{
			PROFILE_SCOPE("Bake")
			res = omm::Cpu::Bake(bakerHandle, bakeDesc, &bakeResultHandle);
		}
		ASSERT_EQ(res, omm::Result::SUCCESS);

		const omm::Cpu::BakeResultDesc* bakeResultDesc = nullptr;
		res = omm::Cpu::GetBakeResultDesc(bakeResultHandle, &bakeResultDesc);
		ASSERT_EQ(res, omm::Result::SUCCESS);

		omm::Debug::Stats stats;
		res = omm::Debug::GetStats(bakerHandle, bakeResultDesc, &stats);
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
			res = omm::Debug::SaveAsImages(bakerHandle, bakeDesc, bakeResultDesc,
				{
					.path = "ReadFromFile",
					.oneFile = false /* Will draw all triangles in the same file.*/
				});
			ASSERT_EQ(res, omm::Result::SUCCESS);
		}
#endif

		// Cleanup. Result no longer needed
		res = omm::Cpu::DestroyBakeResult(bakeResultHandle);
		ASSERT_EQ(res, omm::Result::SUCCESS);
		// Cleanup. Texture no longer needed
		res = omm::Cpu::DestroyDeserializedResult(deserializedResult);
		ASSERT_EQ(res, omm::Result::SUCCESS);
		// Cleanup. Baker no longer needed
		res = omm::DestroyBaker(bakerHandle);
		ASSERT_EQ(res, omm::Result::SUCCESS);
	}

}  // namespace


