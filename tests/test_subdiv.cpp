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
#include <shared/parse.h>

#include <algorithm>
#include <random>

namespace {

	class BakeSubDiv : public ::testing::Test {
	protected:
		void SetUp() override {
			EXPECT_EQ(omm::CreateBaker({ .type = omm::BakerType::CPU }, &_baker), omm::Result::SUCCESS);
		}

		void TearDown() override {
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

		struct SubDivDistr {
			uint32_t globalSubDivLvl = 0;
			uint32_t numSubDivLvlGlobal = 0;
			uint32_t numSubDivLvl0 = 0;
			uint32_t numSubDivLvl1 = 0;
			uint32_t numSubDivLvl2 = 0;
			uint32_t numSubDivLvl3 = 0;
			uint32_t numSubDivLvl4 = 0;
		};

		void ValidateDesc(omm::Format vmFormat, const omm::Cpu::BakeResultDesc& desc, uint32_t triangleCount) {

			static constexpr uint32_t kMaxNumSubDivLvl = 5;
			uint32_t numSubDivLvl[kMaxNumSubDivLvl] = { 0, };
			for (uint32_t i = 0; i < desc.descArrayCount; ++i) {
				int32_t subDivLvl = omm::parse::GetTriangleStates(i, desc, nullptr);
				EXPECT_GE(subDivLvl, 0);
				EXPECT_LE(subDivLvl, 4);
				numSubDivLvl[subDivLvl]++;
			}

			uint32_t numUsageSubDivLvl[kMaxNumSubDivLvl] = { 0, };
			for (uint32_t i = 0; i < desc.descArrayHistogramCount; ++i) {
				int32_t subDivLvl = desc.descArrayHistogram[i].subdivisionLevel;
				EXPECT_GE(subDivLvl, 0);
				EXPECT_LE(subDivLvl, 4);
				EXPECT_EQ(vmFormat, (omm::Format)desc.descArrayHistogram[i].format);
				numUsageSubDivLvl[subDivLvl] += desc.descArrayHistogram[i].count;
			}

			for (uint32_t i = 0; i < kMaxNumSubDivLvl; ++i) {
				EXPECT_EQ(numSubDivLvl[i], numUsageSubDivLvl[i]);
			}
		}

		void BakeMixedSubDivs(
			const SubDivDistr& p) {

			const float alphaCutoff = 0.3f;

			omm::Cpu::Texture tex_04 = 0;
			{
				// Create checkerboard pattern to make sure that no special-index case will happen.
				vmtest::TextureFP32 texture(1024, 1024, 1, true /*enableZorder*/, alphaCutoff, [](int i, int j, int w, int h, int mip){
					if ((i) % 2 != (j) % 2)
						return 0.f;
					else
						return 1.f;
					});

				tex_04 = CreateTexture(texture.GetDesc());
			}

			uint32_t seed = 32;
			std::default_random_engine eng(seed);

			uint32_t triangleCount = p.numSubDivLvlGlobal + p.numSubDivLvl0 + p.numSubDivLvl1 + p.numSubDivLvl2 + p.numSubDivLvl3 + p.numSubDivLvl4;
			EXPECT_NE(triangleCount, 0);

			std::vector<uint8_t> subDivNum;
			subDivNum.insert(subDivNum.end(), p.numSubDivLvlGlobal, 0xF);
			subDivNum.insert(subDivNum.end(), p.numSubDivLvl0, 0);
			subDivNum.insert(subDivNum.end(), p.numSubDivLvl1, 1);
			subDivNum.insert(subDivNum.end(), p.numSubDivLvl2, 2);
			subDivNum.insert(subDivNum.end(), p.numSubDivLvl3, 3);
			subDivNum.insert(subDivNum.end(), p.numSubDivLvl4, 4);

			std::shuffle(subDivNum.begin(), subDivNum.end(), eng);

			std::uniform_real_distribution<float> distr(0.f, 1.f);

			auto IsZeroArea = [](const float2& p0, const float2& p1, const float2& p2) {

				const float3 N = glm::cross(float3(p2 - p0, 0), float3(p1 - p0, 0));
				const bool bIsZeroArea = N.z * N.z < 1e-6;
				return bIsZeroArea;
			};

			uint32_t numIdx = triangleCount * 3;
			std::vector<uint32_t> indices(numIdx);
			std::vector<float2> texCoords(numIdx);
			for (uint32_t i = 0; i < numIdx / 3; ++i) {
				bool isGood = false;
				const uint32_t kMaxN = 10;
				uint32_t N = 0;
				while (!isGood && N++ < kMaxN)
				{
					for (uint32_t j = 0; j < 3; ++j) {
						indices[3 * i + j] = 3 * i + j;
						texCoords[3 * i + j] = float2(distr(eng), distr(eng));
					}

					isGood = !IsZeroArea(texCoords[3 * i + 0], texCoords[3 * i + 1], texCoords[3 * i + 2]);
				}
				EXPECT_LT(N, kMaxN);
			}

			omm::Cpu::BakeInputDesc desc;
			desc.texture = tex_04;
			desc.alphaMode = omm::AlphaMode::Test;
			desc.runtimeSamplerDesc.addressingMode = omm::TextureAddressMode::Clamp;
			desc.runtimeSamplerDesc.filter = omm::TextureFilterMode::Nearest;
			desc.indexFormat = omm::IndexFormat::I32_UINT;
			desc.indexBuffer = indices.data();
			desc.texCoords = texCoords.data();
			desc.texCoordFormat = omm::TexCoordFormat::UV32_FLOAT;
			desc.indexCount = (uint32_t)indices.size();
			desc.maxSubdivisionLevel = p.globalSubDivLvl;
			desc.subdivisionLevels = subDivNum.data();
			desc.alphaCutoff = alphaCutoff;
			desc.bakeFlags = (omm::Cpu::BakeFlags)(
				(uint32_t)omm::Cpu::BakeFlags::EnableInternalThreads |
				(uint32_t)omm::Cpu::BakeFlags::DisableSpecialIndices | 
				(uint32_t)omm::Cpu::BakeFlags::Force32BitIndices |
				(uint32_t)omm::Cpu::BakeFlags::DisableDuplicateDetection
				);

			desc.dynamicSubdivisionScale = 0.f;
			omm::Cpu::BakeResult res = 0;

			EXPECT_EQ(omm::Cpu::Bake(_baker, desc, &res), omm::Result::SUCCESS);
			EXPECT_NE(res, 0);

			const omm::Cpu::BakeResultDesc* resDesc = nullptr;
			EXPECT_EQ(omm::Cpu::GetBakeResultDesc(res, &resDesc), omm::Result::SUCCESS);
			EXPECT_NE(resDesc, nullptr);

			ValidateDesc(desc.format, *resDesc, triangleCount);

			return;

			uint32_t numSubDivLvl0 = 0;
			uint32_t numSubDivLvl1 = 0;
			uint32_t numSubDivLvl2 = 0;
			uint32_t numSubDivLvl3 = 0;
			uint32_t numSubDivLvl4 = 0;

			for (uint32_t i = 0; i < triangleCount; ++i) {
				int32_t subDivLvl = omm::parse::GetTriangleStates(i, *resDesc, nullptr);
				EXPECT_GE(subDivLvl, 0);
				EXPECT_LE(subDivLvl, 4);

				if (subDivLvl == 0)
					numSubDivLvl0++;
				else if (subDivLvl == 1)
					numSubDivLvl1++;
				else if (subDivLvl == 2)
					numSubDivLvl2++;
				else if (subDivLvl == 3)
					numSubDivLvl3++;
				else if (subDivLvl == 4)
					numSubDivLvl4++;
			}

			uint32_t expNumSubDivLvl0 = p.numSubDivLvl0;
			uint32_t expNumSubDivLvl1 = p.numSubDivLvl1;
			uint32_t expNumSubDivLvl2 = p.numSubDivLvl2;
			uint32_t expNumSubDivLvl3 = p.numSubDivLvl3;
			uint32_t expNumSubDivLvl4 = p.numSubDivLvl4;

			if (p.globalSubDivLvl == 0)
				expNumSubDivLvl0 += p.numSubDivLvlGlobal;
			else if (p.globalSubDivLvl == 1)
				expNumSubDivLvl1 += p.numSubDivLvlGlobal;
			else if (p.globalSubDivLvl == 2)
				expNumSubDivLvl2 += p.numSubDivLvlGlobal;
			else if (p.globalSubDivLvl == 3)
				expNumSubDivLvl3 += p.numSubDivLvlGlobal;
			else if (p.globalSubDivLvl == 4)
				expNumSubDivLvl4 += p.numSubDivLvlGlobal;

			EXPECT_EQ(numSubDivLvl0, expNumSubDivLvl0);
			EXPECT_EQ(numSubDivLvl1, expNumSubDivLvl1);
			EXPECT_EQ(numSubDivLvl2, expNumSubDivLvl2);
			EXPECT_EQ(numSubDivLvl3, expNumSubDivLvl3);
			EXPECT_EQ(numSubDivLvl4, expNumSubDivLvl4);

			// Here we check the following rules:
			// VMs are sorted with lower sub-div levels first
			// And Large VMs must be cache line aligned.
			// NOTE: these rules are not enforced by the DX spec,
			// so the vm blob might still be valid, but not too efficient.
			int32_t subDivLvl = -1;
			for (uint32_t i = 0; i < resDesc->descArrayCount; ++i)
			{
				const int32_t ommIdx = omm::parse::GetOmmIndexForTriangleIndex(*resDesc, i);
				EXPECT_GE(ommIdx, 0);
				if (ommIdx >= 0) {
					const omm::Cpu::OpacityMicromapDesc& ommDesc = resDesc->descArray[ommIdx];

					EXPECT_LE(subDivLvl, ommDesc.subdivisionLevel) << "OMMs are not sorted by subdiv level.";
					subDivLvl = std::max(subDivLvl, (int32_t)ommDesc.subdivisionLevel);

					static constexpr uint32_t kCacheLineSize = 128u;
					static constexpr uint32_t kCacheLineBitSize = kCacheLineSize << 3u;
					const uint32_t vmBitSize = omm::parse::GetOmmBitSize(ommDesc);
					if (vmBitSize % kCacheLineBitSize == 0)
					{
						EXPECT_EQ(ommDesc.offset % kCacheLineSize, 0) << "VM idx:" << ommIdx << " has unexpected alignment" << ommDesc.offset;
					}
				}
			}

			EXPECT_EQ(omm::Cpu::DestroyBakeResult(res), omm::Result::SUCCESS);
		}

		std::vector< omm::Cpu::Texture> _textures;
		omm::Baker _baker = 0;
	};

	TEST_F(BakeSubDiv, Mixed) {
		BakeMixedSubDivs(
			{
				.globalSubDivLvl = 2, 
				.numSubDivLvlGlobal = 8,
				.numSubDivLvl0 = 4,
				.numSubDivLvl1 = 7,
				.numSubDivLvl2 = 7,
				.numSubDivLvl3 = 7,
				.numSubDivLvl4 = 7,
			});
	}

	TEST_F(BakeSubDiv, Mixed2) {
		BakeMixedSubDivs(
			{
				.globalSubDivLvl = 4,
				.numSubDivLvlGlobal = 84,
				.numSubDivLvl0 = 234,
				.numSubDivLvl1 = 0,
				.numSubDivLvl2 = 23,
				.numSubDivLvl3 = 34,
				.numSubDivLvl4 = 57,
			});
	}

	TEST_F(BakeSubDiv, Lvl0Only) {
		BakeMixedSubDivs(
			{
				.globalSubDivLvl = 2,
				.numSubDivLvlGlobal = 0,
				.numSubDivLvl0 = 56,
				.numSubDivLvl1 = 0,
				.numSubDivLvl2 = 0,
				.numSubDivLvl3 = 0,
				.numSubDivLvl4 = 0,
			});
	}

	TEST_F(BakeSubDiv, Lvl1Only) {
		BakeMixedSubDivs(
			{
				.globalSubDivLvl = 2,
				.numSubDivLvlGlobal = 0,
				.numSubDivLvl0 = 0,
				.numSubDivLvl1 = 526,
				.numSubDivLvl2 = 0,
				.numSubDivLvl3 = 0,
				.numSubDivLvl4 = 0,
			});
	}

	TEST_F(BakeSubDiv, Lvl2Only) {
		BakeMixedSubDivs(
			{
				.globalSubDivLvl = 2,
				.numSubDivLvlGlobal = 0,
				.numSubDivLvl0 = 0,
				.numSubDivLvl1 = 0,
				.numSubDivLvl2 = 91,
				.numSubDivLvl3 = 0,
				.numSubDivLvl4 = 0,
			});
	}

	TEST_F(BakeSubDiv, Lvl3Only) {
		BakeMixedSubDivs(
			{
				.globalSubDivLvl = 2,
				.numSubDivLvlGlobal = 0,
				.numSubDivLvl0 = 0,
				.numSubDivLvl1 = 0,
				.numSubDivLvl2 = 0,
				.numSubDivLvl3 = 391,
				.numSubDivLvl4 = 0,
			});
	}

	TEST_F(BakeSubDiv, Lvl4Only) {
		BakeMixedSubDivs(
			{
				.globalSubDivLvl = 2,
				.numSubDivLvlGlobal = 0,
				.numSubDivLvl0 = 0,
				.numSubDivLvl1 = 0,
				.numSubDivLvl2 = 0,
				.numSubDivLvl3 = 0,
				.numSubDivLvl4 = 391,
			});
	}

	TEST_F(BakeSubDiv, LvlGlobalOnly) {
		BakeMixedSubDivs(
			{
				.globalSubDivLvl = 4,
				.numSubDivLvlGlobal = 430,
				.numSubDivLvl0 = 0,
				.numSubDivLvl1 = 0,
				.numSubDivLvl2 = 0,
				.numSubDivLvl3 = 0,
				.numSubDivLvl4 = 0,
			});
	}

	static void SubdivideTrianlge(const std::string& name, const omm::Triangle& t) {
		int32_t subdivLvl = 2;
		int2 size(1024, 1024);
		uint32_t numSubTri = omm::bird::GetNumMicroTriangles(subdivLvl);

		auto IndexToColor = [](uint32_t index, uint32_t subdivLvl)->float3 {
			uint32_t numSubTri = omm::bird::GetNumMicroTriangles(subdivLvl);
			return float3((float)index) / float(numSubTri);
		};

		// Raster macro triangle with colors
		ImageRGB imageA(size, { 0, 0, 0 });
		omm::RasterizeConservativeSerial(t, size, [subdivLvl, numSubTri, IndexToColor, &imageA](int2 pixel, float3* bc, void* ctc) {

			bool isUpright;
			uint32_t idx = omm::bird::bary2index(float2(bc->z, bc->x), subdivLvl, isUpright);
			float3 color = IndexToColor(idx, subdivLvl);
			imageA.Store(pixel, uchar4(color.x * 255, color.y * 255, color.z * 255, 255));

			}, nullptr);


		// Raster each micro triangle with same colors
		ImageRGB imageB(size, { 0, 0, 0 });
		for (uint32_t idx = 0; idx < numSubTri; ++idx) {

			omm::Triangle subTri = omm::bird::GetMicroTriangle(t, idx, subdivLvl);

			omm::RasterizeConservativeSerial(subTri, int2(1024, 1024), [subdivLvl, numSubTri, IndexToColor, idx, &imageB](int2 pixel, float3* bc, void*) {

				float3 color = IndexToColor(idx, subdivLvl);
				imageB.Store(pixel, uchar4(color.x * 255, color.y * 255, color.z * 255, 255));

				}, nullptr);
		}

		SaveImageToFile("SubdivideTriangle", name + "A.png", imageA);
		SaveImageToFile("SubdivideTriangle", name + "B.png", imageB);
		// Outputs should be identical.
	}

	TEST(SubdivideTriangle, Dump) {
		SubdivideTrianlge("Straight", omm::Triangle(float2(0.f, 0.f), float2(1.f, 1.f), float2(0.f, 1.f)));
		SubdivideTrianlge("Rot", omm::Triangle(float2(0.675f, 0.05f), float2(0.125f, 0.985f), float2(0.675f, 0.985f)));
	}

}  // namespace