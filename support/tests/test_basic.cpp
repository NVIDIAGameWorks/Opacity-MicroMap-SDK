/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#include <omm.h>
#include <algorithm>

#include "util/omm.h"

namespace {

	TEST(Lib, VersionCheck) {

		const omm::LibraryDesc desc = omm::GetLibraryDesc();

		EXPECT_EQ(desc.versionMajor, OMM_VERSION_MAJOR);
		EXPECT_EQ(desc.versionMinor, OMM_VERSION_MINOR);
		EXPECT_EQ(desc.versionBuild, OMM_VERSION_BUILD);
	}

	TEST(Baker, DestroyNull) {

		omm::Baker baker = nullptr;
		EXPECT_EQ(omm::DestroyBaker(baker), omm::Result::INVALID_ARGUMENT);
	}

	TEST(Baker, CreateDestroy) {
		omm::Baker baker = nullptr;
		EXPECT_EQ(omm::CreateBaker({ .type = omm::BakerType::CPU }, &baker), omm::Result::SUCCESS);
		EXPECT_NE(baker, nullptr);
		EXPECT_EQ(omm::DestroyBaker(baker), omm::Result::SUCCESS);
	}

	TEST(Baker, CreateInvalid) {
		omm::Baker baker = nullptr;
		EXPECT_EQ(omm::CreateBaker({.type = omm::BakerType::MAX_NUM}, &baker), omm::Result::INVALID_ARGUMENT);
	}

	TEST(Baker, CreateDestroyGPU) {
		omm::Baker baker = 0;
		EXPECT_EQ(omm::CreateBaker({ .type = omm::BakerType::GPU }, &baker), omm::Result::SUCCESS);
		EXPECT_NE(baker, nullptr);
		EXPECT_EQ(omm::DestroyBaker(baker), omm::Result::SUCCESS);
	}

	TEST(Baker, StaticDataGPU) {
		{
			size_t byteSize = 0;
			EXPECT_EQ(omm::Gpu::GetStaticResourceData(omm::Gpu::ResourceType::STATIC_VERTEX_BUFFER, nullptr, &byteSize), omm::Result::SUCCESS);
			EXPECT_NE(byteSize, 0);

			std::vector<uint8_t> data(byteSize);
			std::fill(data.begin(), data.end(), 0);
			EXPECT_EQ(omm::Gpu::GetStaticResourceData(omm::Gpu::ResourceType::STATIC_VERTEX_BUFFER, data.data(), &byteSize), omm::Result::SUCCESS);
			EXPECT_NE(std::all_of(data.begin(), data.end(), [](uint8_t i) { return i == 0; }), true);
		}

		{
			size_t byteSize = 0;
			EXPECT_EQ(omm::Gpu::GetStaticResourceData(omm::Gpu::ResourceType::STATIC_INDEX_BUFFER, nullptr, &byteSize), omm::Result::SUCCESS);
			EXPECT_NE(byteSize, 0);

			std::vector<uint8_t> data(byteSize);
			std::fill(data.begin(), data.end(), 0);
			EXPECT_EQ(omm::Gpu::GetStaticResourceData(omm::Gpu::ResourceType::STATIC_INDEX_BUFFER, data.data(), &byteSize), omm::Result::SUCCESS);
			EXPECT_NE(std::all_of(data.begin(), data.end(), [](uint8_t i) { return i == 0; }), true);
		}
	}

	class GpuTest : public ::testing::Test {
	protected:
		void SetUp() override {
			EXPECT_EQ(omm::CreateBaker({ .type = omm::BakerType::GPU }, &_baker), omm::Result::SUCCESS);
		}
		void TearDown() override {
			EXPECT_EQ(omm::DestroyBaker(_baker), omm::Result::SUCCESS);
		}
		void TestShaders(omm::Gpu::RenderAPI api, bool expectingShaders);
		omm::Baker _baker = 0;
	};

	TEST_F(GpuTest, Pipeline) {

		omm::Gpu::PipelineConfigDesc cfg;
		cfg.renderAPI				= omm::Gpu::RenderAPI::DX12;

		omm::Gpu::Pipeline pipeline = 0;
		omm::Result res = omm::Gpu::CreatePipeline(_baker, cfg, &pipeline);
		ASSERT_EQ(res, omm::Result::SUCCESS);

		res = omm::Gpu::DestroyPipeline(_baker, pipeline);
		EXPECT_EQ(res, omm::Result::SUCCESS);
	}

	void GpuTest::TestShaders(omm::Gpu::RenderAPI renderAPI, bool expectingShaders)
	{
		omm::Gpu::PipelineConfigDesc cfg;
		cfg.renderAPI = renderAPI;

		omm::Gpu::Pipeline pipeline = 0;
		omm::Result res = omm::Gpu::CreatePipeline(_baker, cfg, &pipeline);
		ASSERT_EQ(res, omm::Result::SUCCESS);

		const omm::Gpu::PipelineInfoDesc* outPipelineDescs = nullptr;
		res = omm::Gpu::GetPipelineDesc(pipeline, &outPipelineDescs);
		ASSERT_EQ(res, omm::Result::SUCCESS);

		for (uint32_t i = 0; i < outPipelineDescs->pipelineNum; ++i)
		{
			const omm::Gpu::PipelineDesc& pipeline = outPipelineDescs->pipelines[i];
			ASSERT_LT(pipeline.type, omm::Gpu::PipelineType::MAX_NUM);

			switch (pipeline.type)
			{
			case omm::Gpu::PipelineType::Compute:
			{
				if (expectingShaders)
				{
					ASSERT_NE(pipeline.compute.computeShader.data, nullptr);
					ASSERT_NE(pipeline.compute.computeShader.size, 0);
				}
				else
				{
					ASSERT_EQ(pipeline.compute.computeShader.data, nullptr);
					ASSERT_EQ(pipeline.compute.computeShader.size, 0);
				}
				break;
			}
			case omm::Gpu::PipelineType::Graphics:
			{
				if (expectingShaders)
				{
					bool anyShader = false;
					if (pipeline.graphics.pixelShader.data)
					{
						anyShader = true;
						ASSERT_NE(pipeline.graphics.pixelShader.size, 0);
					}

					if (pipeline.graphics.geometryShader.data)
					{
						anyShader = true;
						ASSERT_NE(pipeline.graphics.geometryShader.size, 0);
					}

					if (pipeline.graphics.vertexShader.data)
					{
						anyShader = true;
						ASSERT_NE(pipeline.graphics.vertexShader.size, 0);
					}

					ASSERT_EQ(anyShader, true);
				}
				else
				{
					ASSERT_EQ(pipeline.graphics.pixelShader.data, nullptr);
					ASSERT_EQ(pipeline.graphics.pixelShader.size, 0);

					ASSERT_EQ(pipeline.graphics.geometryShader.data, nullptr);
					ASSERT_EQ(pipeline.graphics.geometryShader.size, 0);

					ASSERT_EQ(pipeline.graphics.vertexShader.data, nullptr);
					ASSERT_EQ(pipeline.graphics.vertexShader.size, 0);
				}
				break;
			}
			default:
			{
				ASSERT_EQ(false, true);
				break;
			}
			}
		}

		res = omm::Gpu::DestroyPipeline(_baker, pipeline);
		EXPECT_EQ(res, omm::Result::SUCCESS);
	}

#if defined(OMM_ENABLE_PRECOMPILED_SHADERS_DXIL)
	TEST_F(GpuTest, ShadersDXIL) {
		TestShaders(omm::Gpu::RenderAPI::DX12, true /*expectingShaders*/);
	}
#else
	TEST_F(GpuTest, NoShadersDXIL) {
		TestShaders(omm::Gpu::RenderAPI::DX12, false /*expectingShaders*/);
	}
#endif

#if defined(OMM_ENABLE_PRECOMPILED_SHADERS_SPIRV)
	TEST_F(GpuTest, ShadersSPIRV) {
		TestShaders(omm::Gpu::RenderAPI::Vulkan, true /*expectingShaders*/);
	}
#else
	TEST_F(GpuTest, NoShadersSPIRV) {
		TestShaders(omm::Gpu::RenderAPI::Vulkan, false /*expectingShaders*/);
	}
#endif

	class TextureTest : public ::testing::Test {
	protected:
		void SetUp() override {
			EXPECT_EQ(omm::CreateBaker({ .type = omm::BakerType::CPU }, &_baker), omm::Result::SUCCESS);
		}
		void TearDown() override {
			EXPECT_EQ(omm::DestroyBaker(_baker), omm::Result::SUCCESS);
		}
		omm::Baker _baker = 0;
	};

	TEST_F(TextureTest, DestroyNull) {
		omm::Cpu::Texture texture = 0;
		EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, texture), omm::Result::INVALID_ARGUMENT);
	}

	TEST_F(TextureTest, Create64x100) {

		vmtest::TextureFP32 tex(64, 100, 1, true /*enableZorder*/, -1.f /*alphaCutoff*/, [](int i, int j, int w, int h, int mip)->float {return 0.f; });
		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::SUCCESS);
		EXPECT_NE(outTexture, nullptr);
		EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, outTexture), omm::Result::SUCCESS);
	}


	TEST_F(TextureTest, Create100x100) {
		vmtest::TextureFP32 tex(100, 100, 1, true /*enableZorder*/, -1.f /*alphaCutoff*/, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::SUCCESS);
		EXPECT_NE(outTexture, nullptr);
		EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, outTexture), omm::Result::SUCCESS);
	}


	TEST_F(TextureTest, Create100x64) {
		vmtest::TextureFP32 tex(100, 64, 1, true /*enableZorder*/, -1.f /*alphaCutoff*/, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::SUCCESS);
		EXPECT_NE(outTexture, nullptr);
		EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, outTexture), omm::Result::SUCCESS);
	}

	TEST_F(TextureTest, Create0x64) {
		vmtest::TextureFP32 tex(0, 64, 1, true /*enableZorder*/, -1.f /*alphaCutoff*/, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::INVALID_ARGUMENT);
	}

	TEST_F(TextureTest, Create0x0) {
		vmtest::TextureFP32 tex(0, 0, 1, true /*enableZorder*/, -1.f /*alphaCutoff*/, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::INVALID_ARGUMENT);
	}

	TEST_F(TextureTest, Create65536x0) {
		vmtest::TextureFP32 tex(65536, 1, 1, false /*enableZorder*/, -1.f /*alphaCutoff*/, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::SUCCESS);
	}

	TEST_F(TextureTest, Create65537x0) {
		vmtest::TextureFP32 tex(65537, 1, 1, false /*enableZorder*/, -1.f /*alphaCutoff*/, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::INVALID_ARGUMENT);
	}

}  // namespace
