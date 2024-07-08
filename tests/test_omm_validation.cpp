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

#include <omm.h>
#include <shared/bird.h>
#include <shared/cpu_raster.h>

#include <algorithm>
#include <random>

namespace {

	class LogTest : public ::testing::Test {
	protected:

		void SetUp() override {
		}

		void InitBaker(bool setCallback) {

			omm::BakerCreationDesc desc = { .type = omm::BakerType::CPU };
			if (setCallback)
				desc.messageInterface = { &LogCallback, this };

			EXPECT_EQ(omm::CreateBaker(desc, &_baker), omm::Result::SUCCESS);
		}

		void TearDown() override {

			EXPECT_EQ(_logCounter, _expectedLogMsg.size());

			for (omm::Cpu::Texture tex : _textures) {
				EXPECT_EQ(omm::Cpu::DestroyTexture(_baker, tex), omm::Result::SUCCESS);
			}

			EXPECT_EQ(omm::DestroyBaker(_baker), omm::Result::SUCCESS);
		}

		omm::Cpu::Texture CreateTexture(const omm::Cpu::TextureDesc& desc) {
			omm::Cpu::Texture tex = 0;
			EXPECT_EQ(omm::Cpu::CreateTexture(_baker, desc, &tex), omm::Result::SUCCESS);
			_textures.push_back(tex);
			return tex;
		}

		omm::Cpu::BakeInputDesc CreateDefaultBakeInputDesc(uint32_t triangleCount = 256, const float alphaCutoff = 0.3f) {

			vmtest::TextureFP32 texture(1024, 1024, 1, false, alphaCutoff, [](int i, int j, int w, int h, int mip) {
				if ((i) % 2 != (j) % 2)
					return 0.f;
				else
					return 1.f;
				});

			omm::Cpu::Texture tex = CreateTexture(texture.GetDesc());

			uint32_t seed = 32;
			std::default_random_engine eng(seed);
			std::uniform_real_distribution<float> distr(0.f, 1.f);

			uint32_t numIdx = triangleCount * 3;
			_indices.resize(numIdx);
			_texCoords.resize(numIdx);
			for (uint32_t i = 0; i < numIdx / 3; ++i) {
				for (uint32_t j = 0; j < 3; ++j) {
					_indices[3 * i + j] = 3 * i + j;
					_texCoords[3 * i + j] = float2(distr(eng), distr(eng));
				}
			}

			omm::Cpu::BakeInputDesc desc;
			desc.texture = tex;
			desc.alphaMode = omm::AlphaMode::Test;
			desc.runtimeSamplerDesc.addressingMode = omm::TextureAddressMode::Clamp;
			desc.runtimeSamplerDesc.filter = omm::TextureFilterMode::Nearest;
			desc.indexFormat = omm::IndexFormat::UINT_32;
			desc.indexBuffer = _indices.data();
			desc.texCoords = _texCoords.data();
			desc.texCoordFormat = omm::TexCoordFormat::UV32_FLOAT;
			desc.indexCount = (uint32_t)_indices.size();
			desc.maxSubdivisionLevel = 4;
			desc.alphaCutoff = alphaCutoff;
			desc.bakeFlags = (omm::Cpu::BakeFlags)(
				(uint32_t)omm::Cpu::BakeFlags::EnableWorkloadValidation |
				(uint32_t)omm::Cpu::BakeFlags::DisableSpecialIndices |
				(uint32_t)omm::Cpu::BakeFlags::DisableDuplicateDetection);

			desc.dynamicSubdivisionScale = 0.f;

			return desc;
		}

		static void LogCallback(omm::MessageSeverity severity, const char* message, void* userArg)
		{
			LogTest* _this = (LogTest*)userArg;
			EXPECT_LT(_this->_logCounter, _this->_expectedLogMsg.size());

			ASSERT_STREQ(_this->_expectedLogMsg[_this->_logCounter].c_str(), message);

			_this->_logCounter++;
		}

		void Bake(omm::Cpu::BakeInputDesc desc, std::vector<std::string> expectedLogMsg, omm::Result expectedResult)
		{
			_expectedLogMsg = expectedLogMsg;
			_logCounter = 0;

			omm::Cpu::BakeResult res = 0;

			ASSERT_EQ(omm::Cpu::Bake(_baker, desc, &res), expectedResult);

			if (expectedResult != omm::Result::SUCCESS)
				return;

			EXPECT_NE(res, 0);

			const omm::Cpu::BakeResultDesc* resDesc = nullptr;
			EXPECT_EQ(omm::Cpu::GetBakeResultDesc(res, &resDesc), omm::Result::SUCCESS);
			EXPECT_NE(resDesc, nullptr);

			EXPECT_EQ(omm::Cpu::DestroyBakeResult(res), omm::Result::SUCCESS);

		}

		std::vector< omm::Cpu::Texture> _textures;
		omm::Baker _baker = 0;
		std::vector<uint32_t> _indices;
		std::vector<float2> _texCoords;
		std::vector<std::string> _expectedLogMsg;
		int _logCounter = 0;
	};

	TEST_F(LogTest, InvalidParameter_Texture)
	{
		InitBaker(true /*set callback*/);
		omm::Cpu::BakeInputDesc desc = CreateDefaultBakeInputDesc();
		desc.texture = 0;
		Bake(desc, {"[Invalid Argument] - ommCpuBakeInputDesc has no texture set"}, omm::Result::INVALID_ARGUMENT);
	}

	TEST_F(LogTest, InvalidParameter_IndexFormat)
	{
		InitBaker(true /*set callback*/);
		omm::Cpu::BakeInputDesc desc = CreateDefaultBakeInputDesc();
		desc.indexFormat = omm::IndexFormat::MAX_NUM;
		Bake(desc, { "[Invalid Argument] - indexFormat is not set" }, omm::Result::INVALID_ARGUMENT);
	}

	TEST_F(LogTest, InvalidParameter_MaxSubdivisionLevel)
	{
		InitBaker(true /*set callback*/);
		omm::Cpu::BakeInputDesc desc = CreateDefaultBakeInputDesc();
		desc.maxSubdivisionLevel = 13;
		Bake(desc, { "[Invalid Argument] - maxSubdivisionLevel (13) is greater than maximum supported (12)" }, omm::Result::INVALID_ARGUMENT);
	}

	TEST_F(LogTest, InvalidParameter_AlphaCutoff)
	{
		InitBaker(true /*set callback*/);
		omm::Cpu::BakeInputDesc desc = CreateDefaultBakeInputDesc();
		desc.alphaCutoff = 0.4f;
		Bake(desc, { "[Invalid Argument] - Texture object alpha cutoff threshold (0.300000) is different from alpha cutoff threshold in bake input (0.400000)" }, omm::Result::INVALID_ARGUMENT);
	}

	TEST_F(LogTest, PerfWarning_HugeWorkload)
	{
		InitBaker(true /*set callback*/);
		omm::Cpu::BakeInputDesc desc = CreateDefaultBakeInputDesc(511);
		Bake(desc, { "[Perf Warning] - The workload consists of 137972015 work items (number of texels to classify), which corresponds to roughly 131 1024x1024 textures."
					 " This is unusually large and may result in long bake times." }, omm::Result::SUCCESS);
	}

	TEST_F(LogTest, InvalidParameter_ValidationWithoutLog)
	{
		InitBaker(false /*set callback*/);
		omm::Cpu::BakeInputDesc desc = CreateDefaultBakeInputDesc();
		Bake(desc, { }, omm::Result::INVALID_ARGUMENT);
	}

}  // namespace
