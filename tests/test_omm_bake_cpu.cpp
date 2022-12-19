/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#include "util/omm.h"
#include "util/image.h"
#include "util/omm_histogram.h"

#include <stb_image.h>

#include <omm.h>
#include <shared/bird.h>

namespace {

	enum TestSuiteConfig
	{
		Default,
		TextureDisableZOrder,
		Force32BitIndices,
	};

	struct Options
	{
		omm::OMMFormat format = omm::OMMFormat::OC1_4_State;
		omm::UnknownStatePromotion unknownStatePromotion = omm::UnknownStatePromotion::Nearest;
		bool mergeSimilar = false;
		uint32_t mipCount = 1;
		bool enableSpecialIndices = true;
		bool oneFile = true;
		bool detailedCutout = false;
		bool monochromeUnknowns = false;
	};

	class OMMBakeTestCPU : public ::testing::TestWithParam<TestSuiteConfig> {
	protected:
		void SetUp() override {
			EXPECT_EQ(omm::CreateOpacityMicromapBaker({.type = omm::BakerType::CPU }, &_baker), omm::Result::SUCCESS);
		}

		void TearDown() override {
			for (omm::Cpu::Texture tex : _textures) {
				EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, tex), omm::Result::SUCCESS);
			}

			EXPECT_EQ(omm::DestroyOpacityMicromapBaker(_baker), omm::Result::SUCCESS);
		}

		bool EnableZOrder() const { return !((GetParam() & TestSuiteConfig::TextureDisableZOrder) == TestSuiteConfig::TextureDisableZOrder); }
		bool Force32BitIndices() const { return (GetParam() & TestSuiteConfig::Force32BitIndices) == TestSuiteConfig::Force32BitIndices; }

		omm::Cpu::Texture CreateTexture(const omm::Cpu::TextureDesc& desc) {
			omm::Cpu::Texture tex = 0;
			EXPECT_EQ(omm::Cpu::CreateTexture(_baker, desc, &tex), omm::Result::SUCCESS);
			_textures.push_back(tex);
			return tex;
		}

		void ExpectEqual(const omm::Debug::Stats& stats, const omm::Debug::Stats& expectedStats) {
			EXPECT_EQ(stats.totalOpaque, expectedStats.totalOpaque);
			EXPECT_EQ(stats.totalTransparent, expectedStats.totalTransparent);
			EXPECT_EQ(stats.totalUnknownTransparent, expectedStats.totalUnknownTransparent);
			EXPECT_EQ(stats.totalUnknownOpaque, expectedStats.totalUnknownOpaque);
			EXPECT_EQ(stats.totalFullyOpaque, expectedStats.totalFullyOpaque);
			EXPECT_EQ(stats.totalFullyTransparent, expectedStats.totalFullyTransparent);
			EXPECT_EQ(stats.totalFullyUnknownOpaque, expectedStats.totalFullyUnknownOpaque);
			EXPECT_EQ(stats.totalFullyUnknownTransparent, expectedStats.totalFullyUnknownTransparent);
		}

		omm::Debug::Stats RunVmBake(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			uint32_t indexCount,
			uint32_t* triangleIndices,
			float* texCoords,
			std::function<float(int i, int j, int w, int h, int mip)> tex,
			const Options opt = {}) {

			omm::Cpu::Texture tex_04 = 0;
			{
				vmtest::Texture texture(texSize.x, texSize.y, opt.mipCount, EnableZOrder(), tex);

				tex_04 = CreateTexture(texture.GetDesc());
			}

			omm::Cpu::BakeInputDesc desc;
			desc.texture = tex_04;
			desc.ommFormat = opt.format;
			desc.alphaMode = omm::AlphaMode::Test;
			desc.runtimeSamplerDesc.addressingMode = omm::TextureAddressMode::Clamp;
			desc.runtimeSamplerDesc.filter = omm::TextureFilterMode::Linear;
			desc.indexFormat = omm::IndexFormat::I32_UINT;
			desc.indexBuffer = triangleIndices;
			desc.texCoords = texCoords;
			desc.texCoordFormat = omm::TexCoordFormat::UV32_FLOAT;
			desc.indexCount = indexCount;
			desc.maxSubdivisionLevel = subdivisionLevel;
			desc.alphaCutoff = alphaCutoff;
			desc.unknownStatePromotion = opt.unknownStatePromotion;
			desc.bakeFlags = (omm::Cpu::BakeFlags)((uint32_t)omm::Cpu::BakeFlags::EnableInternalThreads);
			if (opt.mergeSimilar)
				desc.bakeFlags = (omm::Cpu::BakeFlags)((uint32_t)desc.bakeFlags | (uint32_t)omm::Cpu::BakeFlags::EnableNearDuplicateDetection);
			if (Force32BitIndices())
				desc.bakeFlags = (omm::Cpu::BakeFlags)((uint32_t)desc.bakeFlags | (uint32_t)omm::Cpu::BakeFlags::Force32BitIndices);
			if (!opt.enableSpecialIndices)
				desc.bakeFlags = (omm::Cpu::BakeFlags)((uint32_t)desc.bakeFlags | (uint32_t)omm::Cpu::BakeFlags::DisableSpecialIndices);

			desc.dynamicSubdivisionScale = 0.f;

			omm::Cpu::BakeResult res = 0;

			EXPECT_EQ(omm::Cpu::BakeOpacityMicromap(_baker, desc, &res), omm::Result::SUCCESS);
			EXPECT_NE(res, 0);

			const omm::Cpu::BakeResultDesc* resDesc = nullptr;
			EXPECT_EQ(omm::Cpu::GetBakeResultDesc(res, resDesc), omm::Result::SUCCESS);

			
#if OMM_TEST_ENABLE_IMAGE_DUMP
			const bool dump = true;
			if (dump) {
				std::string name = ::testing::UnitTest::GetInstance()->current_test_suite()->name();
				std::string tname = ::testing::UnitTest::GetInstance()->current_test_info()->name();
				std::string fileName = name + "_" + tname;
				std::replace(fileName.begin(), fileName.end(), '/', '_');

				omm::Debug::SaveAsImages(_baker, desc, resDesc, 
					{	.path = "OmmBakeOutput", 
						.filePostfix = fileName.c_str(), 
						.detailedCutout = opt.detailedCutout,
						.dumpOnlyFirstOMM = false,
						.monochromeUnknowns = opt.monochromeUnknowns,
						.oneFile = opt.oneFile });
			}
#endif
			// Manually collected stats from parsing the 
			omm::Debug::Stats stats = omm::Debug::Stats{};
			if (resDesc)
			{
				EXPECT_EQ(omm::Debug::GetStats(_baker, resDesc, &stats), omm::Result::SUCCESS);
			}

			omm::Test::ValidateHistograms(resDesc);

			EXPECT_EQ(omm::Cpu::DestroyBakeResult(res), omm::Result::SUCCESS);

			return stats;
		}

		omm::Debug::Stats RunVmBake(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			std::function<float(int i, int j, int w, int h, int mip)> tex,
			const Options opt = {}) {
			uint32_t triangleIndices[6] = { 0, 1, 2, 3, 1, 2 };
			float texCoords[8] = { 0.f, 0.f,	0.f, 1.f,	1.f, 0.f,	 1.f, 1.f };
			// float texCoords[8] = { 0.6f, 0.02f,	0.02f, 0.7f,	0.98f, 0.88f,	 1.f, 1.f };
			return RunVmBake(alphaCutoff, subdivisionLevel, texSize, 6, triangleIndices, texCoords, tex, opt);
		}

		omm::Debug::Stats LeafletMipN(uint32_t mipStart, uint32_t NumMip)
		{
			uint32_t subdivisionLevel = 6;
			uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

			uint32_t indexCount = 3;
			uint32_t triangleIndices[3] = { 0, 1, 2 };
			//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
			float texCoords[6] = { 0.05f, 0.1f,  0.1f, 0.9f,  0.9f, 0.9f };

			int width, height, channels;
			unsigned char* pixelData = stbi_load(PROJECT_SOURCE_DIR "/tests/data/leaflet.png", &width, &height, &channels, 0);

			const uint32_t NumMipToGenerate = mipStart + NumMip;
			std::vector<std::vector<float>> mips;
			std::vector<std::tuple<uint32_t, uint32_t>> mipDims;
			mips.resize(NumMipToGenerate);
			mipDims.resize(NumMipToGenerate);

			mips[0].reserve(width * height);
			mipDims[0] = std::make_tuple(width, height);

			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i++)
				{
					uint8_t* pixel = pixelData + j * width * channels + channels * i + 2;
					mips[0].push_back(*pixel / 255.f);
				}
			}

			auto GenerateMipAvgFilter = [](std::vector<float> tex, int32_t& w, int32_t& h)->std::vector<float>
			{
				int32_t halfW = w / 2;
				int32_t halfH = h / 2;
				std::vector<float> mip;
				mip.reserve(halfW * halfH);

				for (int32_t j = 0; j < halfH; ++j)
				{
					for (int32_t i = 0; i < halfW; ++i)
					{
						float pixel0 = tex[2 * j * w + 2 * i];
						float pixel1 = tex[(2 * j + 1) * w + 2 * i];
						float pixel2 = tex[2 * j * w + (2 * i + 1)];
						float pixel3 = tex[(2 * j + 1) * w + (2 * i + 1)];

						const float pixel = (pixel0 + pixel1 + pixel2 + pixel3) * 0.25f;
						mip.push_back(pixel);
					}
				}

				w = halfW;
				h = halfH;
				return mip;
			};

			int mipWidth = width;
			int mipHeight = height;
			for (uint32_t i = 1; i < NumMipToGenerate; ++i)
			{
				mips[i] = GenerateMipAvgFilter(mips[i - 1], mipWidth, mipHeight);
				mipDims[i] = std::make_tuple(mipWidth, mipHeight);
			}

			int2 size = { std::get<0>(mipDims[mipStart]), std::get<1>(mipDims[mipStart]) };
			omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, size, indexCount, triangleIndices, texCoords, [mipStart, mips](int i, int j, int w, int h, int mip)->float {

				return 1.f - mips[mipStart + mip][w * j + i];

				}, { .format = omm::OMMFormat::OC1_4_State, .mipCount = NumMip, .oneFile = false, .detailedCutout = false });

			return stats;
		}

		omm::Debug::Stats LeafletLevelN(uint32_t subdivisionLevel)
		{
			uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

			uint32_t indexCount = 3;
			uint32_t triangleIndices[3] = { 0, 1, 2 };
			//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
			float texCoords[6] = { 0.35f, 0.1f,  0.1f, 0.9f,  0.9f, 0.8f };

			int width, height, channels;
			unsigned char* pixelData = stbi_load(PROJECT_SOURCE_DIR "/tests/data/leaflet.png", &width, &height, &channels, 0);

			std::vector<float> mips;

			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i++)
				{
					uint8_t* pixel = pixelData + j * width * channels + channels * i + 2;
					mips.push_back(*pixel / 255.f);
				}
			}

			int2 size = { width, height };
			omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, size, indexCount, triangleIndices, texCoords, [mips](int i, int j, int w, int h, int mip)->float {

				return 1.f - mips[w * j + i];

				}, { .format = omm::OMMFormat::OC1_4_State, .unknownStatePromotion = omm::UnknownStatePromotion::Nearest, .enableSpecialIndices = false, .oneFile = true, .detailedCutout = false });

			return stats;
		}


		std::vector< omm::Cpu::Texture> _textures;
		omm::Baker _baker = 0;
	};

	TEST_P(OMMBakeTestCPU, NullDesc) {

		omm::Cpu::BakeInputDesc nullDesc;
		omm::Cpu::BakeResult res = 0;
		EXPECT_EQ(omm::Cpu::BakeOpacityMicromap(_baker, nullDesc, &res), omm::Result::INVALID_ARGUMENT);
		EXPECT_EQ(res, 0);
	}

	TEST_P(OMMBakeTestCPU, AllOpaque4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, {1024, 1024}, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			});

		ExpectEqual(stats, { .totalFullyOpaque = 2 });
	}


	TEST_P(OMMBakeTestCPU, AllOpaque3) {

		uint32_t subdivisionLevel = 3;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			});

		ExpectEqual(stats, { .totalFullyOpaque = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllOpaque2) {

		uint32_t subdivisionLevel = 2;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			});

		ExpectEqual(stats, { .totalFullyOpaque = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllOpaque1) {

		uint32_t subdivisionLevel = 1;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			});

		ExpectEqual(stats, { .totalFullyOpaque = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllOpaque0) {

		uint32_t subdivisionLevel = 0;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.6f;
			});

		ExpectEqual(stats, { .totalFullyOpaque = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllTransparent4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.4f;
			});

		ExpectEqual(stats, { .totalFullyTransparent = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllTransparent3) {

		uint32_t subdivisionLevel = 3;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.4f;
			});

		ExpectEqual(stats, { .totalFullyTransparent = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllTransparent2) {

		uint32_t subdivisionLevel = 2;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.4f;
			});

		ExpectEqual(stats, { .totalFullyTransparent = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllTransparent1) {

		uint32_t subdivisionLevel = 1;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			return 0.4f;
			});

		ExpectEqual(stats, { .totalFullyTransparent = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllUnknownTransparent) {

		uint32_t subdivisionLevel = 1;

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if ((i) % 8 != (j) % 8)
				return 0.f;
			else
				return 1.f;
			});

		ExpectEqual(stats, { .totalFullyUnknownTransparent = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllUnknownOpaque) {

		uint32_t subdivisionLevel = 1;

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if ((i) % 8 != (j) % 8)
				return 1.f;
			else
				return 0.f;
			});

		ExpectEqual(stats, { .totalFullyUnknownOpaque = 2 });
	}

	TEST_P(OMMBakeTestCPU, AllTransparentOpaqueCorner4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;
			return 0.4f;
			});

		ExpectEqual(stats, {
			.totalTransparent = numMicroTris - 1,
			.totalUnknownTransparent = 1,
			.totalFullyTransparent = 1,
			});
	}

	TEST_P(OMMBakeTestCPU, Circle) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float r = 0.4f;

			const int2 idx = int2(i, j);
			const float2 uv = float2(idx) / float2((float)w);
			if (glm::length(uv - 0.5f) < r)
				return 0.f;
			return 1.f;
			});

		ExpectEqual(stats, {
			.totalOpaque = 204,
			.totalTransparent = 219,
			.totalUnknownTransparent = 39,
			.totalUnknownOpaque = 50,
			});
	}

	TEST_P(OMMBakeTestCPU, CircleMergeSimilar) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float r = 0.4f;

			const int2 idx = int2(i, j);
			const float2 uv = float2(idx) / float2((float)w);
			if (glm::length(uv - 0.5f) < r)
				return 0.f;
			return 1.f;
			}, { .mergeSimilar = true });

		ExpectEqual(stats, {
			.totalOpaque = 200,
			.totalTransparent = 216,
			.totalUnknownTransparent = 42,
			.totalUnknownOpaque = 54,
			});
	}

	TEST_P(OMMBakeTestCPU, CircleOC2) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float r = 0.4f;

			const int2 idx = int2(i, j);
			const float2 uv = float2(idx) / float2((float)w);
			if (glm::length(uv - 0.5f) < r)
				return 0.f;
			return 1.f;
			}, { .format = omm::OMMFormat::OC1_2_State });

		ExpectEqual(stats, {
			.totalOpaque = 254,
			.totalTransparent = 258,
			});
	}

	TEST_P(OMMBakeTestCPU, Sine) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)w;

			return 1.f - std::sinf(uv*15);
			});

		ExpectEqual(stats, {
			.totalOpaque = 224,
			.totalTransparent = 128,
			.totalUnknownTransparent = 96,
			.totalUnknownOpaque = 64,
			});
	}

	TEST_P(OMMBakeTestCPU, SineOC2) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)w;

			return 1.f - std::sinf(uv * 15);
			}, { .format = omm::OMMFormat::OC1_2_State });

		ExpectEqual(stats, {
			.totalOpaque = 288,
			.totalTransparent = 224,
			});
	}


	TEST_P(OMMBakeTestCPU, SineOC2Neg) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)w;

			return 1.f - std::sinf(uv * 15);
			}, { .format = omm::OMMFormat::OC1_2_State });

		ExpectEqual(stats, {
			.totalOpaque = 288,
			.totalTransparent = 224,
			});
	}

	TEST_P(OMMBakeTestCPU, Mandelbrot) {

		uint32_t subdivisionLevel = 5;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j, int w, int h, int mip)->float {
			
			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(w, h) - 0.1f;
			float2 coord = 2.f * uv - 1.f;
			float2 z = float2(0, 0);
			float2 c = coord - float2(0.5, 0);
			bool inMandelbrotSet = true;

			for (int i = 0; i < 20; i++) {
				z = complexMultiply(z, z) + c;
				if (length(z) > 2.) {
					inMandelbrotSet = false;
					break;
				}
			}
			if (inMandelbrotSet) {
				return 0.f;
			}
			else { 
				return 1.f;
			}

		}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 1212,
			.totalTransparent = 484,
			.totalUnknownTransparent = 124,
			.totalUnknownOpaque = 228,
			});
	}

	TEST_P(OMMBakeTestCPU, Mandelbrot2) {

		uint32_t subdivisionLevel = 5;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, texCoords, [](int i, int j, int w, int h, int mip)->float {

			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(w, h) - 0.1f;
			float2 coord = 2.f * uv - 1.f;
			float2 z = float2(0, 0);
			float2 c = coord - float2(0.5, 0);
			bool inMandelbrotSet = true;

			for (int i = 0; i < 20; i++) {
				z = complexMultiply(z, z) + c;
				if (length(z) > 2.) {
					inMandelbrotSet = false;
					break;
				}
			}
			if (inMandelbrotSet) {
				return 0.f;
			}
			else {
				return 1.f;
			}

			}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 521,
			.totalTransparent = 287,
			.totalUnknownTransparent = 81,
			.totalUnknownOpaque = 135,
			});
	}

	TEST_P(OMMBakeTestCPU, Mandelbrot3) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, texCoords, [](int i, int j, int w, int h, int mip)->float {

			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(w, h) - 0.1f;
			float2 coord = 2.f * uv - 1.f;
			float2 z = float2(0, 0);
			float2 c = coord - float2(0.5, 0);
			bool inMandelbrotSet = true;

			for (int i = 0; i < 20; i++) {
				z = complexMultiply(z, z) + c;
				if (length(z) > 2.) {
					inMandelbrotSet = false;
					break;
				}
			}
			if (inMandelbrotSet) {
				return 0.f;
			}
			else {
				return 1.f;
			}

			}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 164040,
			.totalTransparent = 91320,
			.totalUnknownTransparent = 3039,
			.totalUnknownOpaque = 3745,
			});
	}

	TEST_P(OMMBakeTestCPU, Julia) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 3;
		uint32_t triangleIndices[6] = { 0, 1, 2, };
		float texCoords[8] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, texCoords, [](int i, int j, int w, int h, int mip)->float {

			auto multiply = [](float2 x, float2 y)->float2 {
				return float2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(w, h) - 0.1f;

			float2 z0 = 5.f * (uv - float2(.5, .27));
			float2 col;
			float time = 3.1f;
			float2 c = cos(time) * float2(cos(time / 2.), sin(time / 2.));
			for (int i = 0; i < 500; i++) {
				float2 z = multiply(z0, z0) + c;
				float mq = dot(z, z);
				if (mq > 4.) {
					col = float2(float(i) / 20., 0.);
					break;
				}
				else {
					z0 = z;
				}
				col = float2(mq / 2., mq / 2.);
			}
			
			float alpha = std::clamp(col.x, 0.f, 1.f);
			return 1.f - alpha;

			}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 254735,
			.totalTransparent = 4295,
			.totalUnknownTransparent = 2054,
			.totalUnknownOpaque = 1060,
			});
	}

	TEST_P(OMMBakeTestCPU, Uniform) {

		uint32_t subdivisionLevel = 6;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 6;
		uint32_t triangleIndices[6] = { 0, 1, 2, 1, 2, 3, };
		//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
		float texCoords[8] = { 0.f, 0.f,  0.f, 1.0f,  1.f, 1.f, 1.f, 0.f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 4, 4 }, indexCount, triangleIndices, texCoords, [](int i, int j, int w, int h, int mip)->float {

			uint32_t x = (i) % 2;
			uint32_t y = (j) % 2;

			float values[4] = 
			{
				0.9f, 0.1f,
				0.1f, 0.7f
			};

			return 1.f - values[x + 2 *y];

			}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 5132,
			.totalTransparent = 2393,
			.totalUnknownTransparent = 357,
			.totalUnknownOpaque = 310,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsLvl6) {

		uint32_t subdivisionLevel = 6;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 6;
		uint32_t triangleIndices[6] = { 0, 1, 2, 1, 2, 3, };
		//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
		float texCoords[8] = { 0.f, 0.f,  0.f, 1.0f,  1.f, 1.f, 1.f, 0.f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, texCoords, [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 902,
			.totalTransparent = 0,
			.totalUnknownTransparent = 3,
			.totalUnknownOpaque = 7287,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsLvl8) {

		uint32_t subdivisionLevel = 8;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t indexCount = 6;
		uint32_t triangleIndices[6] = { 0, 1, 2, 1, 2, 3, };
		//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
		float texCoords[8] = { 0.f, 0.f,  0.f, 1.0f,  1.f, 1.f, 1.f, 0.f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, indexCount, triangleIndices, texCoords, [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 77995,
			.totalTransparent = 535,
			.totalUnknownTransparent = 23163,
			.totalUnknownOpaque = 29379,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsReuseLvl2) {

		uint32_t subdivisionLevel = 2;

		std::vector<uint32_t> indices;
		std::vector<float2> texCoords;

		const uint32_t N = 32;
		const uint32_t M = 32;
		for (uint32_t j = 0; j < M; ++j)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t indexOffset = 3 * (i + j * N);
				indices.push_back(indexOffset + 0);
				indices.push_back(indexOffset + 1);
				indices.push_back(indexOffset + 2);

				const float2 offset = float2(float(i) / float(N), float(j) / float(M));
				texCoords.push_back(offset + float2(0.f, 0.f) / float2(N, M));
				texCoords.push_back(offset + float2(0.f, 1.f) / float2(N, M));
				texCoords.push_back(offset + float2(1.f, 1.f) / float2(N, M));
			}
		}

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, (uint32_t)indices.size(), indices.data(), (float*)texCoords.data(), [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 6933,
			.totalUnknownTransparent = 1935,
			.totalUnknownOpaque = 7516,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsReuseLvl3) {

		uint32_t subdivisionLevel = 3;

		std::vector<uint32_t> indices;
		std::vector<float2> texCoords;

		const uint32_t N = 32;
		const uint32_t M = 32;
		for (uint32_t j = 0; j < M; ++j)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t indexOffset = 3 * (i + j * N);
				indices.push_back(indexOffset + 0);
				indices.push_back(indexOffset + 1);
				indices.push_back(indexOffset + 2);

				const float2 offset = float2(float(i) / float(N), float(j) / float(M));
				texCoords.push_back(offset + float2(0.f, 0.f) / float2(N, M));
				texCoords.push_back(offset + float2(0.f, 1.f) / float2(N, M));
				texCoords.push_back(offset + float2(1.f, 1.f) / float2(N, M));
			}
		}

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, (uint32_t)indices.size(), indices.data(), (float*)texCoords.data(), [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 40134,
			.totalTransparent = 250,
			.totalUnknownTransparent = 11939,
			.totalUnknownOpaque = 13213,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsReuseLvl4) {

		uint32_t subdivisionLevel = 4;

		std::vector<uint32_t> indices;
		std::vector<float2> texCoords;

		const uint32_t N = 32;
		const uint32_t M = 32;
		for (uint32_t j = 0; j < M; ++j)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t indexOffset = 3 * (i + j * N);
				indices.push_back(indexOffset + 0);
				indices.push_back(indexOffset + 1);
				indices.push_back(indexOffset + 2);

				const float2 offset = float2(float(i) / float(N), float(j) / float(M));
				texCoords.push_back(offset + float2(0.f, 0.f) / float2(N, M));
				texCoords.push_back(offset + float2(0.f, 1.f) / float2(N, M));
				texCoords.push_back(offset + float2(1.f, 1.f) / float2(N, M));
			}
		}

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, (uint32_t)indices.size(), indices.data(), (float*)texCoords.data(), [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 187129,
			.totalTransparent = 17979,
			.totalUnknownTransparent = 30309,
			.totalUnknownOpaque = 26727,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsReuseLvl5) {

		uint32_t subdivisionLevel = 5;

		std::vector<uint32_t> indices;
		std::vector<float2> texCoords;

		const uint32_t N = 32;
		const uint32_t M = 32;
		for (uint32_t j = 0; j < M; ++j)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t indexOffset = 3 * (i + j * N);
				indices.push_back(indexOffset + 0);
				indices.push_back(indexOffset + 1);
				indices.push_back(indexOffset + 2);

				const float2 offset = float2(float(i) / float(N), float(j) / float(M));
				texCoords.push_back(offset + float2(0.f, 0.f) / float2(N, M));
				texCoords.push_back(offset + float2(0.f, 1.f) / float2(N, M));
				texCoords.push_back(offset + float2(1.f, 1.f) / float2(N, M));
			}
		}

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, (uint32_t)indices.size(), indices.data(), (float*)texCoords.data(), [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::OMMFormat::OC1_4_State });

		ExpectEqual(stats, {
			.totalOpaque = 796515,
			.totalTransparent = 138195,
			.totalUnknownTransparent = 56743,
			.totalUnknownOpaque = 57123,
			});
	}

	TEST_P(OMMBakeTestCPU, HexagonsReuseLSH) {

		uint32_t subdivisionLevel = 4;

		std::vector<uint32_t> indices;
		std::vector<float2> texCoords;

		const uint32_t N = 32;
		const uint32_t M = 32;
		for (uint32_t j = 0; j < M; ++j)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t indexOffset = 3 * (i + j * N);
				indices.push_back(indexOffset + 0);
				indices.push_back(indexOffset + 1);
				indices.push_back(indexOffset + 2);

				const float2 offset = float2(float(i) / float(N), float(j) / float(M));
				texCoords.push_back(offset + float2(0.f, 0.f) / float2(N, M));
				texCoords.push_back(offset + float2(0.f, 1.f) / float2(N, M));
				texCoords.push_back(offset + float2(1.f, 1.f) / float2(N, M));
			}
		}

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, (uint32_t)indices.size(), indices.data(), (float*)texCoords.data(), [](int i, int j, int w, int h, int mip)->float {

			const float scale = 30.f;
			const float gridThickness = 0.2f;

			float2 pos = scale * float2(i, j) / float2(1024, 1024);
			pos.x *= 0.57735f * 2.0f;
			pos.y += 0.5f * ((uint32_t)floor(pos.x) % 2);
			pos = glm::abs(glm::fract(pos) - float2(0.5f));
			float d = std::abs(glm::max(pos.x * 1.5f + pos.y, pos.y * 2.0f) - 1.0f);

			return glm::smoothstep(0.0f, gridThickness, d);
			}, { .format = omm::OMMFormat::OC1_4_State, .mergeSimilar = true });

		ExpectEqual(stats, {
			.totalOpaque = 172854, 
			.totalTransparent = 11500,
			.totalUnknownTransparent = 38296,
			.totalUnknownOpaque = 39494,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_0) {

		omm::Debug::Stats stats = LeafletMipN(0, 1);

		ExpectEqual(stats, {
			.totalOpaque = 817,
			.totalTransparent = 2763,
			.totalUnknownTransparent = 232,
			.totalUnknownOpaque = 284,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_1) {

		omm::Debug::Stats stats = LeafletMipN(0, 2);

		ExpectEqual(stats, {
			.totalOpaque = 809,
			.totalTransparent = 2720,
			.totalUnknownTransparent = 275,
			.totalUnknownOpaque = 292,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_2) {

		omm::Debug::Stats stats = LeafletMipN(0, 3);

		ExpectEqual(stats, {
			.totalOpaque = 784,
			.totalTransparent = 2688,
			.totalUnknownTransparent = 307,
			.totalUnknownOpaque = 317,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_3) {

		omm::Debug::Stats stats = LeafletMipN(0, 4);

		ExpectEqual(stats, {
			.totalOpaque = 776,
			.totalTransparent = 2684,
			.totalUnknownTransparent = 311,
			.totalUnknownOpaque = 325,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_4) {

		omm::Debug::Stats stats = LeafletMipN(0, 5);

		ExpectEqual(stats, {
			.totalOpaque = 724,
			.totalTransparent = 2586,
			.totalUnknownTransparent = 409,
			.totalUnknownOpaque = 377,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_5) {

		omm::Debug::Stats stats = LeafletMipN(0, 6);

		ExpectEqual(stats, {
			.totalOpaque = 615,
			.totalTransparent = 2430,
			.totalUnknownTransparent = 565 ,
			.totalUnknownOpaque = 486,
			});
	}


	TEST_P(OMMBakeTestCPU, LeafletMip0_to_6) {

		omm::Debug::Stats stats = LeafletMipN(0, 7);

		ExpectEqual(stats, {
			.totalOpaque = 349,
			.totalTransparent = 2408,
			.totalUnknownTransparent = 587,
			.totalUnknownOpaque = 752,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0_to_7) {

		omm::Debug::Stats stats = LeafletMipN(0, 8);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 2408,
			.totalUnknownTransparent = 587,
			.totalUnknownOpaque = 1101,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip0) {

		omm::Debug::Stats stats = LeafletMipN(0, 1);

		ExpectEqual(stats, {
			.totalOpaque = 817,
			.totalTransparent = 2763,
			.totalUnknownTransparent = 232,
			.totalUnknownOpaque = 284,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip1) {

		omm::Debug::Stats stats = LeafletMipN(1, 1);

		ExpectEqual(stats, {
			.totalOpaque = 847,
			.totalTransparent = 2728,
			.totalUnknownTransparent = 248,
			.totalUnknownOpaque = 273,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip2) {

		omm::Debug::Stats stats = LeafletMipN(2, 1);

		ExpectEqual(stats, {
			.totalOpaque = 857,
			.totalTransparent = 2725,
			.totalUnknownTransparent = 268,
			.totalUnknownOpaque = 246,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip3) {

		omm::Debug::Stats stats = LeafletMipN(3, 1);

		ExpectEqual(stats, {
			.totalOpaque = 867,
			.totalTransparent = 2735,
			.totalUnknownTransparent = 239,
			.totalUnknownOpaque = 255,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip4) {

		omm::Debug::Stats stats = LeafletMipN(4, 1);

		ExpectEqual(stats, {
			.totalOpaque = 928,
			.totalTransparent = 2777,
			.totalUnknownTransparent = 199,
			.totalUnknownOpaque = 192,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip5) {

		omm::Debug::Stats stats = LeafletMipN(5, 1);

		ExpectEqual(stats, {
			.totalOpaque = 965,
			.totalTransparent = 2821,
			.totalUnknownTransparent = 156,
			.totalUnknownOpaque = 154,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletMip6) {

		omm::Debug::Stats stats = LeafletMipN(6, 1);

		ExpectEqual(stats, {
			.totalOpaque = 526,
			.totalTransparent = 3335,
			.totalUnknownTransparent = 119,
			.totalUnknownOpaque = 116,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel0) {

		omm::Debug::Stats stats = LeafletLevelN(0);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 0,
			.totalUnknownTransparent = 1,
			.totalUnknownOpaque = 0,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel1) {

		omm::Debug::Stats stats = LeafletLevelN(1);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 0,
			.totalUnknownTransparent = 4,
			.totalUnknownOpaque = 0,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel2) {

		omm::Debug::Stats stats = LeafletLevelN(2);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 1,
			.totalUnknownTransparent = 10,
			.totalUnknownOpaque = 5,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel3) {

		omm::Debug::Stats stats = LeafletLevelN(3);

		ExpectEqual(stats, {
			.totalOpaque = 0,
			.totalTransparent = 16,
			.totalUnknownTransparent = 31,
			.totalUnknownOpaque = 17,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel4) {

		omm::Debug::Stats stats = LeafletLevelN(4);

		ExpectEqual(stats, {
			.totalOpaque = 36,
			.totalTransparent = 110,
			.totalUnknownTransparent = 66,
			.totalUnknownOpaque = 44,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel5) {

		omm::Debug::Stats stats = LeafletLevelN(5);

		ExpectEqual(stats, {
			.totalOpaque = 207,
			.totalTransparent = 554,
			.totalUnknownTransparent = 139,
			.totalUnknownOpaque = 124,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel6) {

		omm::Debug::Stats stats = LeafletLevelN(6);

		ExpectEqual(stats, {
			.totalOpaque = 1021,
			.totalTransparent = 2508,
			.totalUnknownTransparent = 275,
			.totalUnknownOpaque = 292,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel7) {

		omm::Debug::Stats stats = LeafletLevelN(7);

		ExpectEqual(stats, {
			.totalOpaque = 4666,
			.totalTransparent = 10580,
			.totalUnknownTransparent = 549,
			.totalUnknownOpaque = 589,
			});
	}

	TEST_P(OMMBakeTestCPU, LeafletLevel8) {

		omm::Debug::Stats stats = LeafletLevelN(8);

		ExpectEqual(stats, {
			.totalOpaque = 19831,
			.totalTransparent = 43424,
			.totalUnknownTransparent = 1110,
			.totalUnknownOpaque = 1171,
			});
	}

	TEST_P(OMMBakeTestCPU, DestroyOpacityMicromapBaker) {
	}

	TEST(VMUtil, GetNumMicroTriangles) {
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(0), 1);
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(1), 4);
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(2), 16);
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(3), 64);
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(4), 256);
		EXPECT_EQ(omm::bird::GetNumMicroTriangles(5), 1024);
	}

	INSTANTIATE_TEST_SUITE_P(OMMTestCPU, OMMBakeTestCPU, ::testing::Values(
		TestSuiteConfig::Default
		, TestSuiteConfig::TextureDisableZOrder
		, TestSuiteConfig::Force32BitIndices
	));

}  // namespace