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

		const omm::LibraryDesc& desc = omm::GetLibraryDesc();

		EXPECT_EQ(desc.versionMajor, OMM_VERSION_MAJOR);
		EXPECT_EQ(desc.versionMinor, OMM_VERSION_MINOR);
		EXPECT_EQ(desc.versionBuild, OMM_VERSION_BUILD);
	}

	TEST(Baker, DestroyNull) {

		omm::Baker baker = 0;
		EXPECT_EQ(omm::DestroyOpacityMicromapBaker(baker), omm::Result::INVALID_ARGUMENT);
	}

	TEST(Baker, CreateDestroy) {
		omm::Baker baker = 0;
		EXPECT_EQ(omm::CreateOpacityMicromapBaker({ .type = omm::BakerType::CPU }, &baker), omm::Result::SUCCESS);
		EXPECT_NE(baker, 0);
		EXPECT_EQ(omm::DestroyOpacityMicromapBaker(baker), omm::Result::SUCCESS);
	}

	TEST(Baker, CreateInvalid) {
		omm::Baker baker = 0;
		EXPECT_EQ(omm::CreateOpacityMicromapBaker({.type = omm::BakerType::MAX_NUM}, &baker), omm::Result::INVALID_ARGUMENT);
	}

	TEST(Baker, CreateDestroyGPU) {
		omm::Baker baker = 0;
		EXPECT_EQ(omm::CreateOpacityMicromapBaker({ .type = omm::BakerType::GPU }, &baker), omm::Result::SUCCESS);
		EXPECT_NE(baker, 0);
		EXPECT_EQ(omm::DestroyOpacityMicromapBaker(baker), omm::Result::SUCCESS);
	}

	TEST(Baker, StaticDataGPU) {
		{
			size_t byteSize = 0;
			EXPECT_EQ(omm::Gpu::GetStaticResourceData(omm::Gpu::ResourceType::STATIC_VERTEX_BUFFER, nullptr, byteSize), omm::Result::SUCCESS);
			EXPECT_NE(byteSize, 0);

			std::vector<uint8_t> data(byteSize);
			std::fill(data.begin(), data.end(), 0);
			EXPECT_EQ(omm::Gpu::GetStaticResourceData(omm::Gpu::ResourceType::STATIC_VERTEX_BUFFER, data.data(), byteSize), omm::Result::SUCCESS);
			EXPECT_NE(std::all_of(data.begin(), data.end(), [](uint8_t i) { return i == 0; }), true);
		}

		{
			size_t byteSize = 0;
			EXPECT_EQ(omm::Gpu::GetStaticResourceData(omm::Gpu::ResourceType::STATIC_INDEX_BUFFER, nullptr, byteSize), omm::Result::SUCCESS);
			EXPECT_NE(byteSize, 0);

			std::vector<uint8_t> data(byteSize);
			std::fill(data.begin(), data.end(), 0);
			EXPECT_EQ(omm::Gpu::GetStaticResourceData(omm::Gpu::ResourceType::STATIC_INDEX_BUFFER, data.data(), byteSize), omm::Result::SUCCESS);
			EXPECT_NE(std::all_of(data.begin(), data.end(), [](uint8_t i) { return i == 0; }), true);
		}
	}

	class GpuTest : public ::testing::Test {
	protected:
		void SetUp() override {
			EXPECT_EQ(omm::CreateOpacityMicromapBaker({ .type = omm::BakerType::GPU }, &_baker), omm::Result::SUCCESS);
		}
		void TearDown() override {
			EXPECT_EQ(omm::DestroyOpacityMicromapBaker(_baker), omm::Result::SUCCESS);
		}
		omm::Baker _baker = 0;
	};


	TEST_F(GpuTest, Pipeline) {

		omm::Gpu::BakePipelineConfigDesc cfg;
		cfg.renderAPI				= omm::Gpu::RenderAPI::DX12;

		omm::Gpu::Pipeline pipeline = 0;
		omm::Result res = omm::Gpu::CreatePipeline(_baker, cfg, &pipeline);
		ASSERT_EQ(res, omm::Result::SUCCESS);

		res = omm::Gpu::DestroyPipeline(_baker, pipeline);
		EXPECT_EQ(res, omm::Result::SUCCESS);
	}

	class TextureTest : public ::testing::Test {
	protected:
		void SetUp() override {
			EXPECT_EQ(omm::CreateOpacityMicromapBaker({ .type = omm::BakerType::CPU }, &_baker), omm::Result::SUCCESS);
		}
		void TearDown() override {
			EXPECT_EQ(omm::DestroyOpacityMicromapBaker(_baker), omm::Result::SUCCESS);
		}
		omm::Baker _baker = 0;
	};

	TEST_F(TextureTest, DestroyNull) {
		omm::Cpu::Texture texture = 0;
		EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, texture), omm::Result::INVALID_ARGUMENT);
	}

	TEST_F(TextureTest, Create64x100) {

		vmtest::Texture tex(64, 100, 1, [](int i, int j, int w, int h, int mip)->float {return 0.f; });
		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::SUCCESS);
		EXPECT_NE(outTexture, 0);
		EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, outTexture), omm::Result::SUCCESS);
	}


	TEST_F(TextureTest, Create100x100) {
		vmtest::Texture tex(100, 100, 1, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::SUCCESS);
		EXPECT_NE(outTexture, 0);
		EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, outTexture), omm::Result::SUCCESS);
	}


	TEST_F(TextureTest, Create100x64) {
		vmtest::Texture tex(100, 64, 1, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::SUCCESS);
		EXPECT_NE(outTexture, 0);
		EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, outTexture), omm::Result::SUCCESS);
	}

	TEST_F(TextureTest, Create0x64) {
		vmtest::Texture tex(0, 64, 1, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::INVALID_ARGUMENT);
	}

	TEST_F(TextureTest, Create0x0) {
		vmtest::Texture tex(0, 0, 1, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::INVALID_ARGUMENT);
	}

	TEST_F(TextureTest, Create65536x0) {
		vmtest::Texture tex(65536, 1, 1, false /*enableZorder*/, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::SUCCESS);
	}

	TEST_F(TextureTest, Create65537x0) {
		vmtest::Texture tex(65537, 1, 1, false /*enableZorder*/, [](int i, int j, int w, int h, int mip)->float {return 0.f; });

		omm::Cpu::Texture outTexture = 0;
		EXPECT_EQ(omm::Cpu::CreateTexture(_baker, tex.GetDesc(), &outTexture), omm::Result::INVALID_ARGUMENT);
	}

}  // namespace
