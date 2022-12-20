/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>

#include "gtest/gtest.h"
#include "util/omm_histogram.h"

#include <nvrhi/nvrhi.h>

#include "nvrhi_environment.h"
#include <omm-sdk-nvrhi/NVRHIWrapper.h>
#include <omm-sdk-nvrhi/NVRHIOmmBakeIntegration.h>

#include <omm.h>
#include <shared/bird.h>

namespace {

	enum TestSuiteConfig
	{
		None							= 0,
		ComputeOnly						= 1u << 0,
		DisableSpecialIndices			= 1u << 1,
		Force32BitIndices				= 1u << 2,
		DisableTexCoordDeduplication	= 1u << 3,
		RedChannel						= 1u << 4,
		GreenChannel					= 1u << 5,
		BlueChannel						= 1u << 6,
	};

	class OMMBakeTestGPU : public ::testing::TestWithParam<TestSuiteConfig> {
	protected:
		void SetUp() override {
			ASSERT_NE(g_nvrhiEnvironment, nullptr);
			NVRHIContext* nvrhiContext = g_nvrhiEnvironment->GetContext();
			ASSERT_NE(nvrhiContext, nullptr);
			m_device = nvrhiContext->CreateDevice();
			ASSERT_NE(m_device.Get(), nullptr);

			m_commandList = m_device->createCommandList();
			ASSERT_NE(m_commandList.Get(), nullptr);
		}

		void TearDown() override {
			m_commandList = nullptr;
			m_device = nullptr;
		}

		bool ComputeOnly() const { return (GetParam() & TestSuiteConfig::ComputeOnly) == TestSuiteConfig::ComputeOnly; }
		bool EnableSpecialIndices() const { return (GetParam() & TestSuiteConfig::DisableSpecialIndices) != TestSuiteConfig::DisableSpecialIndices; }
		bool Force32BitIndices() const { return (GetParam() & TestSuiteConfig::Force32BitIndices) == TestSuiteConfig::Force32BitIndices; }
		bool EnableTexCoordDeduplication() const { return (GetParam() & TestSuiteConfig::DisableTexCoordDeduplication) != TestSuiteConfig::DisableTexCoordDeduplication; }
		uint GetAlphaChannelIndex() const { 
			if ((GetParam() & TestSuiteConfig::RedChannel) == TestSuiteConfig::RedChannel)
				return 0;
			if ((GetParam() & TestSuiteConfig::GreenChannel) == TestSuiteConfig::GreenChannel)
				return 1;
			if ((GetParam() & TestSuiteConfig::BlueChannel) == TestSuiteConfig::BlueChannel)
				return 2;
			return 3;
		}

		omm::Debug::Stats RunVmBake(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			uint32_t indexBufferSize,
			uint32_t* triangleIndices,
			float* texCoords,
			uint32_t texCoordBufferSize,
			std::function<float(int i, int j)> texCb,
			omm::OMMFormat format = omm::OMMFormat::OC1_4_State) {

			const uint32_t alphaTextureChannel = GetAlphaChannelIndex();

			nvrhi::TextureDesc desc;
			desc.width = texSize.x;
			desc.height = texSize.y;
			desc.format = nvrhi::Format::RGBA32_FLOAT;

			nvrhi::StagingTextureHandle staging = m_device->createStagingTexture(desc, nvrhi::CpuAccessMode::Write);

			nvrhi::TextureSlice slice;
			slice = slice.resolve(desc);
			size_t rowPitch;
			void* data = m_device->mapStagingTexture(staging.Get(), slice, nvrhi::CpuAccessMode::Write, &rowPitch);
			std::vector<float> imageData;
			for (uint32_t j = 0; j < desc.height; ++j)
			{
				for (uint32_t i = 0; i < desc.width; ++i)
				{
					float* rgba = (float*)((uint8_t*)data + j * rowPitch + (4 * i) * sizeof(float));
					float val = texCb(i, j);
					rgba[0] = alphaTextureChannel == 0 ? val : 0.f;
					rgba[1] = alphaTextureChannel == 1 ? val : 0.f;
					rgba[2] = alphaTextureChannel == 2 ? val : 0.f;
					rgba[3] = alphaTextureChannel == 3 ? val : 0.f;
					imageData.push_back(val);
				}
			}

			m_device->unmapStagingTexture(staging.Get());

			// Upload alpha texture
			m_commandList->open();
			nvrhi::TextureHandle alphaTexture;
			{
				alphaTexture = m_device->createTexture(desc);
				m_commandList->copyTexture(alphaTexture, slice, staging, slice);
			}

			// Upload index buffer
			nvrhi::BufferHandle ib;
			{
				ib = m_device->createBuffer({ .byteSize = indexBufferSize, .debugName = "ib", .format = nvrhi::Format::R32_UINT, .canHaveUAVs = true, .canHaveTypedViews = true, .canHaveRawViews = true });
				m_commandList->beginTrackingBufferState(ib, nvrhi::ResourceStates::Common);
				m_commandList->writeBuffer(ib, triangleIndices, indexBufferSize);
			}

			// Upload texcoords
			nvrhi::BufferHandle vb;
			{
				vb = m_device->createBuffer({ .byteSize = texCoordBufferSize, .debugName = "vb", .canHaveUAVs = true, .canHaveRawViews = true });
				m_commandList->beginTrackingBufferState(vb, nvrhi::ResourceStates::Common);

				m_commandList->writeBuffer(vb, texCoords, texCoordBufferSize);
			}

			// Upload index buffer
			NVRHIVmBakeIntegration bake(m_device, m_commandList, true /*enable debug*/);

			NVRHIVmBakeIntegration::Input input;
			input.alphaTexture = alphaTexture;
			input.alphaTextureChannel = alphaTextureChannel;
			input.alphaCutoff = 0.5f;
			input.texCoordBuffer = vb;
			input.texCoordStrideInBytes = sizeof(float2);
			input.indexBuffer = ib;
			input.numIndices = indexBufferSize / sizeof(uint32_t);
			input.globalSubdivisionLevel = subdivisionLevel;
			input.use2State = format == omm::OMMFormat::OC1_2_State;
			input.dynamicSubdivisionScale = 0.f;
			input.enableSpecialIndices = EnableSpecialIndices();
			input.force32BitIndices = Force32BitIndices();
			input.enableTexCoordDeuplication = EnableTexCoordDeduplication();
			input.computeOnly = ComputeOnly();

			NVRHIVmBakeIntegration::PreBakeInfo info;
			bake.GetPreBakeInfo(input, info);

			NVRHIVmBakeIntegration::Output res;
			res.ommArrayBuffer = m_device->createBuffer({.byteSize = info.ommArrayBufferSize, .debugName  = "omArrayBuffer", .canHaveUAVs = true, .canHaveRawViews = true });
			res.ommDescBuffer = m_device->createBuffer({ .byteSize = info.ommDescBufferSize, .debugName = "omDescBuffer", .canHaveUAVs = true, .canHaveRawViews = true });
			res.ommIndexBuffer = m_device->createBuffer({ .byteSize = info.ommIndexBufferSize, .debugName = "omIndexBuffer", .canHaveUAVs = true, .canHaveRawViews = true });
			res.ommDescArrayHistogramBuffer = m_device->createBuffer({ .byteSize = info.ommDescArrayHistogramSize , .debugName = "omUsageDescBuffer" , .canHaveUAVs = true, .canHaveRawViews = true });
			res.ommIndexHistogramBuffer = m_device->createBuffer({ .byteSize = info.ommIndexHistogramSize , .debugName = "ommIndexHistogramBuffer" , .canHaveUAVs = true, .canHaveRawViews = true });
			res.ommPostBuildInfoBuffer = m_device->createBuffer({ .byteSize = info.ommPostBuildInfoBufferSize , .debugName = "ommPostBuildInfoBuffer" , .canHaveUAVs = true, .canHaveRawViews = true });

			m_commandList->beginTrackingBufferState(res.ommArrayBuffer, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(res.ommDescBuffer, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(res.ommIndexBuffer, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(res.ommDescArrayHistogramBuffer, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(res.ommIndexHistogramBuffer, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(res.ommPostBuildInfoBuffer, nvrhi::ResourceStates::Common);

			bake.RunBake(
				m_commandList,
				input,
				res);

			nvrhi::BufferHandle ommArrayBufferReadback = m_device->createBuffer({ .byteSize = info.ommArrayBufferSize , .debugName = "omArrayBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			nvrhi::BufferHandle ommDescBufferReadback = m_device->createBuffer({ .byteSize = info.ommDescBufferSize , .debugName = "omDescBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			nvrhi::BufferHandle ommIndexBufferReadback = m_device->createBuffer({ .byteSize = info.ommIndexBufferSize , .debugName = "omIndexBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			nvrhi::BufferHandle ommDescArrayHistogramBufferReadback = m_device->createBuffer({ .byteSize = info.ommDescArrayHistogramSize , .debugName = "vmArrayHistogramBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			nvrhi::BufferHandle ommIndexHistogramBufferReadback = m_device->createBuffer({ .byteSize = info.ommIndexHistogramSize , .debugName = "vmArrayHistogramBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			nvrhi::BufferHandle ommPostBuildInfoBufferReadback = m_device->createBuffer({ .byteSize = info.ommPostBuildInfoBufferSize , .debugName = "ommPostBuildInfoBufferReadback" , .cpuAccess = nvrhi::CpuAccessMode::Read });
			m_commandList->beginTrackingBufferState(ommArrayBufferReadback, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(ommIndexBufferReadback, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(ommDescBufferReadback, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(ommDescArrayHistogramBufferReadback, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(ommIndexHistogramBufferReadback, nvrhi::ResourceStates::Common);
			m_commandList->beginTrackingBufferState(ommPostBuildInfoBufferReadback, nvrhi::ResourceStates::Common);

			m_commandList->copyBuffer(ommArrayBufferReadback, 0, res.ommArrayBuffer, 0, info.ommArrayBufferSize);
			m_commandList->copyBuffer(ommDescBufferReadback, 0, res.ommDescBuffer, 0, info.ommDescBufferSize);
			m_commandList->copyBuffer(ommIndexBufferReadback, 0, res.ommIndexBuffer, 0, info.ommIndexBufferSize);
			m_commandList->copyBuffer(ommDescArrayHistogramBufferReadback, 0, res.ommDescArrayHistogramBuffer, 0, info.ommDescArrayHistogramSize);
			m_commandList->copyBuffer(ommIndexHistogramBufferReadback, 0, res.ommIndexHistogramBuffer, 0, info.ommIndexHistogramSize);
			m_commandList->copyBuffer(ommPostBuildInfoBufferReadback, 0, res.ommPostBuildInfoBuffer, 0, info.ommPostBuildInfoBufferSize);

			m_commandList->close();

			// Execute & Sync.
			uint64_t fence = m_device->executeCommandList(m_commandList);

			m_device->waitForIdle();

			// Readback.
			auto ReadBuffer = [this](nvrhi::BufferHandle buffer, size_t size = 0)->std::vector<uint8_t>
			{
				std::vector<uint8_t> data;
				void* pData = m_device->mapBuffer(buffer, nvrhi::CpuAccessMode::Read);
				assert(pData);
				size_t byteSize = size == 0 ? buffer->getDesc().byteSize : size;
				assert(size <= buffer->getDesc().byteSize);
				data.resize(byteSize);
				memcpy(data.data(), pData, byteSize);
				m_device->unmapBuffer(buffer);
				return data;
			};

			std::vector<uint8_t> vmPostBuildInfoData = ReadBuffer(ommPostBuildInfoBufferReadback);
			NVRHIVmBakeIntegration::PostBuildInfo postBuildInfo;
			NVRHIVmBakeIntegration::ReadPostBuildInfo(vmPostBuildInfoData.data(), vmPostBuildInfoData.size(), postBuildInfo);

			std::vector<uint8_t> ommArrayBufferData = ReadBuffer(ommArrayBufferReadback, postBuildInfo.ommArrayBufferSize);
			std::vector<uint8_t> ommIndexBufferData = ReadBuffer(ommIndexBufferReadback);
			std::vector<uint8_t> ommDescBufferData = ReadBuffer(ommDescBufferReadback, postBuildInfo.ommDescBufferSize);
			std::vector<uint8_t> ommArrayHistogramData = ReadBuffer(ommDescArrayHistogramBufferReadback);
			std::vector<uint8_t> ommIndexHistogramData = ReadBuffer(ommIndexHistogramBufferReadback);


			std::string name = ::testing::UnitTest::GetInstance()->current_test_suite()->name();
			std::replace(name.begin(), name.end(), '/', '_');
			std::string tname = ::testing::UnitTest::GetInstance()->current_test_info()->name();
			std::replace(tname.begin(), tname.end(), '/', '_');

#if OMM_TEST_ENABLE_IMAGE_DUMP
			bool dumpDebug = true;

			if (dumpDebug)
			{
				bake.DumpDebug(
					"OmmBakeOutput_GPU",
					tname.c_str(),
					input,
					ommArrayBufferData,
					ommDescBufferData,
					ommIndexBufferData,
					info.ommIndexFormat,
					ommArrayHistogramData,
					ommIndexHistogramData,
					triangleIndices,
					indexBufferSize / sizeof(uint32_t),
					texCoords,
					imageData.data(),
					texSize.x,
					texSize.y
				);
			}
#endif
			size_t indexFormatSize = nvrhi::getFormatInfo(info.ommIndexFormat).bytesPerBlock;

			omm::Cpu::BakeResultDesc resDesc;
			resDesc.ommArrayData = ommArrayBufferData.data();
			resDesc.ommArrayDataSize = (uint32_t)ommArrayBufferData.size();
			resDesc.ommDescArray = (const omm::Cpu::OpacityMicromapDesc*)ommDescBufferData.data();
			resDesc.ommDescArrayCount = (uint32_t)(ommDescBufferData.size() / sizeof(omm::Cpu::OpacityMicromapDesc));
			resDesc.ommIndexBuffer = ommIndexBufferData.data();
			resDesc.ommIndexCount = info.ommIndexCount;
			resDesc.ommIndexFormat = info.ommIndexFormat == nvrhi::Format::R32_UINT ? omm::IndexFormat::I32_UINT : omm::IndexFormat::I16_UINT;
			resDesc.ommDescArrayHistogram = (const omm::Cpu::OpacityMicromapUsageCount*)ommArrayHistogramData.data();
			resDesc.ommDescArrayHistogramCount = (uint32_t)(ommArrayHistogramData.size() / sizeof(omm::Cpu::OpacityMicromapUsageCount));
			resDesc.ommIndexHistogram = (const omm::Cpu::OpacityMicromapUsageCount*)ommIndexHistogramData.data();
			resDesc.ommIndexHistogramCount = (uint32_t)(ommIndexHistogramData.size() / sizeof(omm::Cpu::OpacityMicromapUsageCount));

			omm::Test::ValidateHistograms(&resDesc);

			return bake.GetStats(resDesc);
		}

		omm::Debug::Stats RunVmBake(
			float alphaCutoff,
			uint32_t subdivisionLevel,
			int2 texSize,
			std::function<float(int i, int j)> tex,
			omm::OMMFormat format = omm::OMMFormat::OC1_4_State) {
			uint32_t triangleIndices[] = { 0, 1, 2, 3, 1, 2 };
			float texCoords[] = { 0.f, 0.f,	0.f, 1.f,	1.f, 0.f,	 1.f, 1.f };
			return RunVmBake(alphaCutoff, subdivisionLevel, texSize, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), tex, format);
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

	private:
		nvrhi::DeviceHandle m_device;
		nvrhi::CommandListHandle m_commandList;
	};


	TEST_P(OMMBakeTestGPU, AllOpaque4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.6f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalOpaque = 512 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllOpaque3) {

		uint32_t subdivisionLevel = 3;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.6f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalOpaque = 128 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllOpaque2) {

		uint32_t subdivisionLevel = 2;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.6f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalOpaque = 32 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllOpaque1) {

		uint32_t subdivisionLevel = 1;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.6f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalOpaque = 8 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllOpaque0) {

		uint32_t subdivisionLevel = 0;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.6f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalOpaque = 2 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparent4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 512 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparent3) {

		uint32_t subdivisionLevel = 3;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 128 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparent2) {

		uint32_t subdivisionLevel = 2;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 32 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparent1) {

		uint32_t subdivisionLevel = 1;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 8 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparent0) {

		uint32_t subdivisionLevel = 0;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyTransparent = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalTransparent = 2 });
		}
	}

	// A bit lame because GPU baker doesn't differentiate (yet) between UT and UO
	TEST_P(OMMBakeTestGPU, AllUnknownTransparent) {

		uint32_t subdivisionLevel = 1;

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if ((i) % 8 != (j) % 8)
				return 0.f;
			else
				return 1.f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyUnknownOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalUnknownOpaque = 8 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllUnknownOpaque) {

		uint32_t subdivisionLevel = 1;

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if ((i) % 8 != (j) % 8)
				return 1.f;
			else
				return 0.f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, { .totalFullyUnknownOpaque = 2 });
		}
		else
		{
			ExpectEqual(stats, { .totalUnknownOpaque = 8 });
		}
	}

	TEST_P(OMMBakeTestGPU, AllTransparentOpaqueCorner4) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;
			return 0.4f;
			});

		if (EnableSpecialIndices())
		{
			ExpectEqual(stats, {
				.totalTransparent = numMicroTris - 1,
				.totalUnknownOpaque = 1,
				.totalFullyTransparent = 1,
				});
		}
		else
		{
			ExpectEqual(stats, {
				.totalTransparent = 2 * numMicroTris - 1,
				.totalUnknownOpaque = 1,
				});
		}
	}

	TEST_P(OMMBakeTestGPU, Circle) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float r = 0.4f;

			const int2 idx = int2(i, j);
			const float2 uv = float2(idx) / float2((float)1024);
			if (glm::length(uv - 0.5f) < r)
				return 0.f;
			return 1.f;
			});

		ExpectEqual(stats, {
			.totalOpaque = 204,
			.totalTransparent = 219,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 89,
			});
	}

	TEST_P(OMMBakeTestGPU, CircleOC2) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float r = 0.4f;

			const int2 idx = int2(i, j);
			const float2 uv = float2(idx) / float2((float)1024);
			if (glm::length(uv - 0.5f) < r)
				return 0.f;
			return 1.f;
			}, omm::OMMFormat::OC1_2_State);

		ExpectEqual(stats, {
			.totalOpaque = 293,
			.totalTransparent = 219,
			});
	}

	TEST_P(OMMBakeTestGPU, Sine) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)1024;

			return 1.f - std::sinf(uv * 15);
			});

		ExpectEqual(stats, {
			.totalOpaque = 224,
			.totalTransparent = 128,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 160,
			});
	}

	TEST_P(OMMBakeTestGPU, SineOC2) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)1024;

			return 1.f - std::sinf(uv * 15);
			}, omm::OMMFormat::OC1_2_State);

		ExpectEqual(stats, {
			.totalOpaque = 384,
			.totalTransparent = 128,
			});
	}


	TEST_P(OMMBakeTestGPU, SineOC2Neg) {

		uint32_t subdivisionLevel = 4;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {
			if (i == 0 && j == 0)
				return 0.6f;

			const float uv = float(i) / (float)1024;

			return 1.f - std::sinf(uv * 15);
			}, omm::OMMFormat::OC1_2_State);

		ExpectEqual(stats, {
			.totalOpaque = 384,
			.totalTransparent = 128,
			});
	}

	TEST_P(OMMBakeTestGPU, Mandelbrot) {

		uint32_t subdivisionLevel = 5;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, [](int i, int j)->float {

			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(1024, 1024) - 0.1f;
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

			}, omm::OMMFormat::OC1_4_State);

		ExpectEqual(stats, {
			.totalOpaque = 1212,
			.totalTransparent = 484,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 352,
			});
	}

	TEST_P(OMMBakeTestGPU, Mandelbrot2) {

		uint32_t subdivisionLevel = 5;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), [](int i, int j)->float {

			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(1024, 1024) - 0.1f;
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

			}, omm::OMMFormat::OC1_4_State);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
				.totalOpaque = 522,
				.totalTransparent = 286,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 216,
							});
		}
		else
		{
			ExpectEqual(stats, {
				.totalOpaque = 524,
				.totalTransparent = 287,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 213,
							});
		}
	}

	TEST_P(OMMBakeTestGPU, Mandelbrot3) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), [](int i, int j)->float {

			auto complexMultiply = [](float2 a, float2 b)->float2 {
				return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(1024, 1024) - 0.1f;
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

			}, omm::OMMFormat::OC1_4_State);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
			.totalOpaque = 164039,
			.totalTransparent = 91321,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 6784,
							});
		}
		else
		{
			ExpectEqual(stats, {
			.totalOpaque = 164027,
			.totalTransparent = 91410,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 6707,
						});
		}
	}

	TEST_P(OMMBakeTestGPU, Julia) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), [](int i, int j)->float {

			auto multiply = [](float2 x, float2 y)->float2 {
				return float2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(1024, 1024) - 0.1f;

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

			}, omm::OMMFormat::OC1_4_State);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
				.totalOpaque = 254728,
				.totalTransparent = 4300,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 3116,
				});
		}
		else
		{
			ExpectEqual(stats, {
				.totalOpaque = 254723,
				.totalTransparent = 4300,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 3121,
				});
		}
	}

	TEST_P(OMMBakeTestGPU, Julia2x) {

		uint32_t subdivisionLevel = 9;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, 3, 4, 5, };
		float texCoords[] = { 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f, 0.2f, 0.f,  0.1f, 0.8f,  0.9f, 0.1f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 1024, 1024 }, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), [](int i, int j)->float {

			auto multiply = [](float2 x, float2 y)->float2 {
				return float2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
			};

			float2 uv = 1.2f * float2(i, j) / float2(1024, 1024) - 0.1f;

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

			}, omm::OMMFormat::OC1_4_State);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
				.totalOpaque = 2 * 254728,
				.totalTransparent = 2 * 4300,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 2 * 3116,
				});
		}
		else
		{
			ExpectEqual(stats, {
				.totalOpaque = 2 * 254723,
				.totalTransparent = 2 * 4300,
				.totalUnknownTransparent = 0,
				.totalUnknownOpaque = 2 * 3121,
				});
		}
	}

	TEST_P(OMMBakeTestGPU, Uniform) {

		uint32_t subdivisionLevel = 6;
		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(subdivisionLevel);

		uint32_t triangleIndices[] = { 0, 1, 2, 1, 2, 3, };
		//float texCoords[8] = { 0.25f, 0.25f,  0.25f, 0.75f,  0.75f, 0.25f };
		float texCoords[] = { 0.f, 0.f,  0.f, 1.0f,  1.f, 1.f, 1.f, 0.f };

		omm::Debug::Stats stats = RunVmBake(0.5f, subdivisionLevel, { 4, 4 }, sizeof(triangleIndices), triangleIndices, texCoords, sizeof(texCoords), [](int i, int j)->float {

			uint32_t x = (i) % 2;
			uint32_t y = (j) % 2;

			float values[4] =
			{
				0.9f, 0.1f,
				0.1f, 0.7f
			};

			return 1.f - values[x + 2 * y];

			}, omm::OMMFormat::OC1_4_State);

		if (ComputeOnly())
		{
			ExpectEqual(stats, {
			.totalOpaque = 5132,
			.totalTransparent = 2394,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 666,
							});
		}
		else
		{
			ExpectEqual(stats, {
			.totalOpaque = 5132,
			.totalTransparent = 2393,
			.totalUnknownTransparent = 0,
			.totalUnknownOpaque = 667,
			});
		}
	}

	INSTANTIATE_TEST_SUITE_P(OMMTestGPU, OMMBakeTestGPU, 
		::testing::Values(	
							TestSuiteConfig::None,
							TestSuiteConfig::DisableSpecialIndices,
							TestSuiteConfig::Force32BitIndices,
							TestSuiteConfig::DisableTexCoordDeduplication,
							TestSuiteConfig::RedChannel,
							TestSuiteConfig::BlueChannel,
							TestSuiteConfig::GreenChannel,
							
							TestSuiteConfig::ComputeOnly,
							TestSuiteConfig::ComputeOnly | TestSuiteConfig::DisableSpecialIndices, 
							TestSuiteConfig::ComputeOnly | TestSuiteConfig::Force32BitIndices,
							TestSuiteConfig::ComputeOnly | TestSuiteConfig::DisableTexCoordDeduplication,
							TestSuiteConfig::ComputeOnly | TestSuiteConfig::RedChannel,
							TestSuiteConfig::ComputeOnly | TestSuiteConfig::BlueChannel,
							TestSuiteConfig::ComputeOnly | TestSuiteConfig::GreenChannel

						));

}  // namespace