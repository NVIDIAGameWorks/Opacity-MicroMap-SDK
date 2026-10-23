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
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <xxhash.h>

#include "util/omm.h"

namespace {

	class ScopedEnvironmentVariable {
	public:
		explicit ScopedEnvironmentVariable(const char* name) : m_name(name) {
			if (const char* value = std::getenv(name)) {
				m_wasSet = true;
				m_previousValue = value;
			}
		}

		~ScopedEnvironmentVariable() {
			EXPECT_EQ(Set(m_wasSet ? m_previousValue.c_str() : nullptr), 0);
		}

		int Set(const char* value) {
#if defined(_WIN32)
			return _putenv_s(m_name, value ? value : "");
#else
			return value ? setenv(m_name, value, 1) : unsetenv(m_name);
#endif
		}

	private:
		const char* m_name;
		std::string m_previousValue;
		bool m_wasSet = false;
	};

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

	TEST(Baker, GPUReferenceModeEnvironment) {
		ScopedEnvironmentVariable environment("OMM_ALLOW_GPU_REFERENCE_MODE");
		const char* values[] = { nullptr, "", "0", "invalid", "1" };
		for (const char* value : values) {
			SCOPED_TRACE(value ? value : "<unset>");
			ASSERT_EQ(environment.Set(value), 0);

			std::string error;
			omm::BakerCreationDesc desc = { .type = omm::BakerType::GPU };
			desc.messageInterface = {
				[](omm::MessageSeverity severity, const char* message, void* userArg) {
					if (severity == omm::MessageSeverity::Error)
						*static_cast<std::string*>(userArg) = message;
				}, &error
			};

			omm::Baker baker = reinterpret_cast<omm::Baker>(1);
			const bool allowed = !OMM_GPU_REFERENCE_MODE || (value && std::string(value) == "1");
			EXPECT_EQ(omm::CreateBaker(desc, &baker), allowed ? omm::Result::SUCCESS : omm::Result::FAILURE);
			if (allowed) {
				EXPECT_NE(baker, nullptr);
				EXPECT_EQ(omm::DestroyBaker(baker), omm::Result::SUCCESS);
				EXPECT_TRUE(error.empty());
			} else {
				EXPECT_EQ(baker, nullptr);
				EXPECT_NE(error.find("OMM_ALLOW_GPU_REFERENCE_MODE"), std::string::npos);
			}

			error.clear();
			desc.type = omm::BakerType::CPU;
			baker = nullptr;
			EXPECT_EQ(omm::CreateBaker(desc, &baker), omm::Result::SUCCESS);
			EXPECT_NE(baker, nullptr);
			EXPECT_EQ(omm::DestroyBaker(baker), omm::Result::SUCCESS);
			EXPECT_TRUE(error.empty());
		}
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

#if OMM_ENABLE_GPU_TESTS
	TEST_F(GpuTest, ReferenceMode) {
		EXPECT_TRUE(OMM_GPU_REFERENCE_MODE)
			<< "Enable OMM_GPU_REFERENCE_MODE (-DOMM_GPU_REFERENCE_MODE=ON) for accurate GPU test references.";
	}
#endif

	TEST_F(GpuTest, Pipeline) {

		omm::Gpu::PipelineConfigDesc cfg;
		cfg.renderAPI				= omm::Gpu::RenderAPI::DX12;

		omm::Gpu::Pipeline pipeline = 0;
		omm::Result res = omm::Gpu::CreatePipeline(_baker, cfg, &pipeline);
		ASSERT_EQ(res, omm::Result::SUCCESS);

		res = omm::Gpu::DestroyPipeline(_baker, pipeline);
		EXPECT_EQ(res, omm::Result::SUCCESS);
	}

	TEST_F(GpuTest, SubdivisionValidatesActiveTargets) {
		omm::Gpu::Pipeline pipeline = nullptr;
		ASSERT_EQ(omm::Gpu::CreatePipeline(_baker, {}, &pipeline), omm::Result::SUCCESS);
		struct PipelineCleanup {
			omm::Baker baker;
			omm::Gpu::Pipeline pipeline;
			~PipelineCleanup() { EXPECT_EQ(omm::Gpu::DestroyPipeline(baker, pipeline), omm::Result::SUCCESS); }
		} cleanup = { _baker, pipeline };

		const float nan = std::numeric_limits<float>::quiet_NaN();
		const float invalidTargets[] = { 0.f, -1.f, nan, std::numeric_limits<float>::infinity() };
		omm::Gpu::DispatchConfigDesc base;
		base.runtimeSamplerDesc.addressingMode = omm::TextureAddressMode::Clamp;
		base.runtimeSamplerDesc.filter = omm::TextureFilterMode::Linear;
		base.alphaMode = omm::AlphaMode::Test;
		base.alphaTextureWidth = base.alphaTextureHeight = 64;
		base.texCoordFormat = omm::TexCoordFormat::UV32_FLOAT;
		base.indexFormat = omm::IndexFormat::UINT_32;
		base.indexCount = 6;
		base.maxSubdivisionLevel = 6;
		base.dynamicSubdivisionScale = nan;
		base.subdivision.mode = omm::SubdivisionMode::Area;

		auto validate = [&](const omm::Gpu::DispatchConfigDesc& config) {
			omm::Gpu::PreDispatchInfo info;
			return omm::Gpu::GetPreDispatchInfo(pipeline, config, &info);
		};

		EXPECT_EQ(validate(base), omm::Result::SUCCESS);
		for (float target : invalidTargets) {
			SCOPED_TRACE(target);
			auto config = base;
			config.subdivision.targetAreaInSquareTexels = target;
			EXPECT_EQ(validate(config), omm::Result::INVALID_ARGUMENT);
			config = base;
			config.subdivision.targetEdgeLengthInTexels = target;
			EXPECT_EQ(validate(config), omm::Result::INVALID_ARGUMENT);
		}

		base.subdivision.mode = omm::SubdivisionMode::Edge;
		base.subdivision.targetAreaInSquareTexels = nan;
		EXPECT_EQ(validate(base), omm::Result::SUCCESS);
		for (float target : invalidTargets) {
			SCOPED_TRACE(target);
			auto config = base;
			config.subdivision.targetEdgeLengthInTexels = target;
			EXPECT_EQ(validate(config), omm::Result::INVALID_ARGUMENT);
		}

		base.subdivision.mode = omm::SubdivisionMode::Fixed;
		base.subdivision.targetEdgeLengthInTexels = nan;
		EXPECT_EQ(validate(base), omm::Result::SUCCESS);
		base.subdivision.mode = omm::SubdivisionMode::MAX_NUM;
		EXPECT_EQ(validate(base), omm::Result::INVALID_ARGUMENT);
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

	class SubdivisionSerialization : public ::testing::Test {
	protected:
		void SetUp() override {
			ommBakerCreationDesc bakerDesc = ommBakerCreationDescDefault();
			bakerDesc.type = ommBakerType_CPU;
			ASSERT_EQ(ommCreateBaker(&bakerDesc, &_baker), ommResult_SUCCESS);

			const uint8_t alpha[] = { 0, 255, 255, 0 };
			ommCpuTextureMipDesc mip = ommCpuTextureMipDescDefault();
			mip.width = mip.height = mip.rowPitch = 2;
			mip.textureData = alpha;
			ommCpuTextureDesc textureDesc = ommCpuTextureDescDefault();
			textureDesc.format = ommCpuTextureFormat_UNORM8;
			textureDesc.mips = &mip;
			textureDesc.mipCount = 1;
			ASSERT_EQ(ommCpuCreateTexture(_baker, &textureDesc, &_texture), ommResult_SUCCESS);
		}

		void TearDown() override {
			if (_deserialized)
				EXPECT_EQ(ommCpuDestroyDeserializedResult(_deserialized), ommResult_SUCCESS);
			if (_serialized)
				EXPECT_EQ(ommCpuDestroySerializedResult(_serialized), ommResult_SUCCESS);
			if (_texture)
				EXPECT_EQ(ommCpuDestroyTexture(_baker, _texture), ommResult_SUCCESS);
			if (_baker)
				EXPECT_EQ(ommDestroyBaker(_baker), ommResult_SUCCESS);
		}

		ommCpuBakeInputDesc MakeInput() const {
			ommCpuBakeInputDesc input = ommCpuBakeInputDescDefault();
			input.texture = _texture;
			input.runtimeSamplerDesc.addressingMode = ommTextureAddressMode_Clamp;
			input.runtimeSamplerDesc.filter = ommTextureFilterMode_Nearest;
			input.alphaMode = ommAlphaMode_Test;
			input.texCoordFormat = ommTexCoordFormat_UV32_FLOAT;
			input.texCoords = _uv.data();
			input.indexFormat = ommIndexFormat_UINT_32;
			input.indexBuffer = _indices.data();
			input.indexCount = static_cast<uint32_t>(_indices.size());
			return input;
		}

		ommBaker _baker = nullptr;
		ommCpuTexture _texture = nullptr;
		ommCpuSerializedResult _serialized = nullptr;
		ommCpuDeserializedResult _deserialized = nullptr;
		const std::array<float, 6> _uv = { 0.f, 0.f, 1.f, 0.f, 0.f, 1.f };
		const std::array<uint32_t, 3> _indices = { 0, 1, 2 };
	};

	class SubdivisionSerializationRoundTrip : public SubdivisionSerialization, public ::testing::WithParamInterface<bool> {};

	TEST_P(SubdivisionSerializationRoundTrip, PreservesSettings) {
		std::array<ommCpuBakeInputDesc, 4> inputs;
		const ommSubdivisionMode modes[] = {
			ommSubdivisionMode_Legacy, ommSubdivisionMode_Fixed, ommSubdivisionMode_Area, ommSubdivisionMode_Edge
		};
		for (size_t i = 0; i < inputs.size(); ++i) {
			inputs[i] = MakeInput();
			inputs[i].bakeFlags = ommCpuBakeFlags_DisableDuplicateDetection;
			inputs[i].dynamicSubdivisionScale = i == 0 ? 3.25f : std::numeric_limits<float>::quiet_NaN();
			inputs[i].subdivision.mode = modes[i];
			inputs[i].subdivision.targetAreaInSquareTexels = 13.5f + static_cast<float>(i);
			inputs[i].subdivision.targetEdgeLengthInTexels = 5.25f + static_cast<float>(i);
		}

		ommCpuDeserializedDesc source = ommCpuDeserializedDescDefault();
		source.flags = GetParam() ? ommCpuSerializeFlags_Compress : ommCpuSerializeFlags_None;
		source.numInputDescs = static_cast<int>(inputs.size());
		source.inputDescs = inputs.data();
		ASSERT_EQ(ommCpuSerialize(_baker, source, &_serialized), ommResult_SUCCESS);
		const ommCpuBlobDesc* blob = nullptr;
		ASSERT_EQ(ommCpuGetSerializedResultDesc(_serialized, &blob), ommResult_SUCCESS);
		ASSERT_EQ(ommCpuDeserialize(_baker, *blob, &_deserialized), ommResult_SUCCESS);
		const ommCpuDeserializedDesc* restored = nullptr;
		ASSERT_EQ(ommCpuGetDeserializedDesc(_deserialized, &restored), ommResult_SUCCESS);
		ASSERT_EQ(restored->numInputDescs, source.numInputDescs);
		EXPECT_EQ(restored->numResultDescs, 0);
		for (size_t i = 0; i < inputs.size(); ++i) {
			SCOPED_TRACE(i);
			const auto& actual = restored->inputDescs[i];
			EXPECT_EQ(actual.subdivision.mode, inputs[i].subdivision.mode);
			EXPECT_FLOAT_EQ(actual.subdivision.targetAreaInSquareTexels, inputs[i].subdivision.targetAreaInSquareTexels);
			EXPECT_FLOAT_EQ(actual.subdivision.targetEdgeLengthInTexels, inputs[i].subdivision.targetEdgeLengthInTexels);
			EXPECT_EQ(actual.bakeFlags, inputs[i].bakeFlags);
			if (i == 0)
				EXPECT_FLOAT_EQ(actual.dynamicSubdivisionScale, inputs[i].dynamicSubdivisionScale);
			else
				EXPECT_TRUE(std::isnan(actual.dynamicSubdivisionScale));
		}
	}

	INSTANTIATE_TEST_SUITE_P(Compression, SubdivisionSerializationRoundTrip, ::testing::Bool(),
		[](const ::testing::TestParamInfo<bool>& info) { return info.param ? "Compressed" : "Uncompressed"; });

	TEST_F(SubdivisionSerialization, Version5PreservesLegacySettings) {
		for (ommCpuBakeFlags flags : { ommCpuBakeFlags_None, ommCpuBakeFlags_Force32BitIndices }) {
			SCOPED_TRACE(static_cast<uint32_t>(flags));
			ommCpuBakeInputDesc input = MakeInput();
			input.bakeFlags = flags;
			input.dynamicSubdivisionScale = 3.25f;
			input.subdivision = { ommSubdivisionMode_Edge, 13.5f, 5.25f };
			ommCpuDeserializedDesc source = ommCpuDeserializedDescDefault();
			source.numInputDescs = 1;
			source.inputDescs = &input;
			ASSERT_EQ(ommCpuSerialize(_baker, source, &_serialized), ommResult_SUCCESS);
			const ommCpuBlobDesc* blob = nullptr;
			ASSERT_EQ(ommCpuGetSerializedResultDesc(_serialized, &blob), ommResult_SUCCESS);

			// Version 6 only appends these three fields to each version-5 input record.
			// With one input and no results, that extension precedes numResultDescs.
			std::vector<uint8_t> legacy(static_cast<const uint8_t*>(blob->data), static_cast<const uint8_t*>(blob->data) + blob->size);
			constexpr size_t extensionSize = sizeof(ommSubdivisionMode) + 2 * sizeof(float);
			legacy.erase(legacy.end() - sizeof(int) - extensionSize, legacy.end() - sizeof(int));
			const int version = 5;
			std::memcpy(legacy.data() + sizeof(uint64_t) + 3 * sizeof(int), &version, sizeof(version));
			const uint64_t digest = XXH64(legacy.data() + sizeof(uint64_t), legacy.size() - sizeof(uint64_t), 42);
			std::memcpy(legacy.data(), &digest, sizeof(digest));
			ommCpuBlobDesc legacyBlob = { legacy.data(), legacy.size() };
			ASSERT_EQ(ommCpuDeserialize(_baker, legacyBlob, &_deserialized), ommResult_SUCCESS);
			const ommCpuDeserializedDesc* restored = nullptr;
			ASSERT_EQ(ommCpuGetDeserializedDesc(_deserialized, &restored), ommResult_SUCCESS);
			ASSERT_EQ(restored->numInputDescs, 1);
			const auto& actual = restored->inputDescs[0];
			EXPECT_EQ(actual.bakeFlags, flags);
			EXPECT_FLOAT_EQ(actual.dynamicSubdivisionScale, input.dynamicSubdivisionScale);
			EXPECT_EQ(actual.subdivision.mode, ommSubdivisionMode_Legacy);
			EXPECT_FLOAT_EQ(actual.subdivision.targetAreaInSquareTexels, 4.f);
			EXPECT_FLOAT_EQ(actual.subdivision.targetEdgeLengthInTexels, 8.f);
			ASSERT_EQ(ommCpuDestroyDeserializedResult(_deserialized), ommResult_SUCCESS);
			_deserialized = nullptr;
			ASSERT_EQ(ommCpuDestroySerializedResult(_serialized), ommResult_SUCCESS);
			_serialized = nullptr;
		}
	}

}  // namespace
