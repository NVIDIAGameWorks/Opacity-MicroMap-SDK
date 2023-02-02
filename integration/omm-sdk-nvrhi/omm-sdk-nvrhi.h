/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <omm.hpp>
#include <nvrhi/nvrhi.h>
#include <vector>
#include <stdint.h>
#include <utility>
#include <functional>

namespace omm
{
	class BindingCache;

	class GpuBakeNvrhi
	{
	public:

		// In case the shaders are compiled externally the ShaderProviderCb can be used.
		using ShaderProviderCb = std::function<nvrhi::ShaderHandle(nvrhi::ShaderType type, const char* shaderName, const char* shaderEntryName)>;

		GpuBakeNvrhi(nvrhi::DeviceHandle device, nvrhi::CommandListHandle commandList, bool enableDebug, ShaderProviderCb* shaderProviderCb = nullptr);
		~GpuBakeNvrhi();

		enum class Operation
		{
			Invalid			= 0,
			Setup			= 1u << 0,
			Bake			= 1u << 1,
			SetupAndBake	= Setup | Bake
		};

		struct Input
		{
			Operation							operation = Operation::Invalid;
			nvrhi::TextureHandle				alphaTexture;
			uint32_t							alphaTextureChannel = 3;
			float								alphaCutoff = 0.5f;
			bool								bilinearFilter = true;
			bool								enableLevelLineIntersection = true;
			nvrhi::SamplerAddressMode			sampleMode = nvrhi::SamplerAddressMode::Clamp;

			nvrhi::BufferHandle					texCoordBuffer;
			uint32_t							texCoordBufferOffsetInBytes = 0;
			uint32_t							texCoordStrideInBytes = 0;
			nvrhi::BufferHandle					indexBuffer;
			uint32_t							indexBufferOffsetInBytes = 0;
			uint32_t							numIndices = 0;

			uint32_t							maxSubdivisionLevel = 0;
			nvrhi::rt::OpacityMicromapFormat	format = nvrhi::rt::OpacityMicromapFormat::OC1_4_State;
			float								dynamicSubdivisionScale = 0.5f;
			bool								minimalMemoryMode = false;
			bool								enableSpecialIndices = true;
			bool								force32BitIndices = false;
			bool								enableTexCoordDeduplication = true;
			bool								computeOnly = false;
			bool								enableNsightDebugMode = false;
		};

		struct PreDispatchInfo
		{
			nvrhi::Format	ommIndexFormat;
			uint32_t		ommIndexCount;
			size_t			ommIndexBufferSize;
			size_t			ommIndexHistogramSize;
			size_t			ommArrayBufferSize;
			size_t			ommDescBufferSize;
			size_t			ommDescArrayHistogramSize;
			size_t			ommPostBuildInfoBufferSize;
		};

		struct Buffers
		{
			nvrhi::BufferHandle ommArrayBuffer;
			nvrhi::BufferHandle ommDescBuffer;
			nvrhi::BufferHandle ommIndexBuffer;
			nvrhi::BufferHandle ommDescArrayHistogramBuffer;
			nvrhi::BufferHandle ommIndexHistogramBuffer;
			nvrhi::BufferHandle ommPostBuildInfoBuffer;
		};

		struct OpacityMicromapUsageCount
		{
			uint32_t count = 0;
			uint16_t subdivisionLevel = 0;
			uint16_t format = 0;
		};

		struct PostBuildInfo
		{
			uint32_t ommArrayBufferSize;
			uint32_t ommDescBufferSize;
		};

		// CPU side pre-build info.
		void GetPreDispatchInfo(const Input& params, PreDispatchInfo& info);

		void Dispatch(
			nvrhi::CommandListHandle commandList,
			const Input& params,
			const Buffers& buffers);

		void Clear();

		// This assumes pData is the CPU-side pointer of the contents in vmUsageDescReadbackBufferSize.
		static void ReadPostBuildInfo(void* pData, size_t byteSize, PostBuildInfo& outPostBuildInfo);
		static void ReadUsageDescBuffer(void* pData, size_t byteSize, std::vector<OpacityMicromapUsageCount>& outVmUsages);

		// Debug dumping
		void DumpDebug(
			const char* folderName,
			const char* debugName,
			const Input& params,
			const std::vector<uint8_t>& ommArrayBuffer,
			const std::vector<uint8_t>& ommDescBuffer,
			const std::vector<uint8_t>& ommIndexBuffer,
			nvrhi::Format ommIndexBufferFormat,
			const std::vector<uint8_t>& ommDescArrayHistogramBuffer,
			const std::vector<uint8_t>& ommIndexHistogramBuffer,
			const void* indexBuffer,
			const uint32_t indexCount,
			const void* texCoords,
			const float* imageData,
			const uint32_t width,
			const uint32_t height
		);

		omm::Debug::Stats GetStats(const omm::Cpu::BakeResultDesc& desc);

	private:

		void InitStaticBuffers(nvrhi::CommandListHandle commandList);
		void InitBaker(ShaderProviderCb* shaderProviderCb);
		void DestroyBaker();

		void SetupPipelines(
			const omm::Gpu::BakePipelineInfoDesc* desc, 
			ShaderProviderCb* shaderProviderCb);

		omm::Gpu::BakeDispatchConfigDesc GetConfig(const Input& params);

		void ReserveGlobalCBuffer(size_t size, uint32_t slot);
		void ReserveScratchBuffers(const omm::Gpu::PreBakeInfo& info);
		nvrhi::TextureHandle GetTextureResource(const Input& params, const Buffers& output, const omm::Gpu::Resource& resource);
		nvrhi::BufferHandle GetBufferResource(const Input& params, const Buffers& output, const omm::Gpu::Resource& resource, uint32_t& offsetInBytes);

		void ExecuteBakeOperation(
			nvrhi::CommandListHandle commandList,
			const Input& params,
			const Buffers& output,
			const omm::Gpu::BakeDispatchChain* outDispatchDesc);

		nvrhi::DeviceHandle m_device;
		nvrhi::BufferHandle m_staticIndexBuffer;
		nvrhi::BufferHandle m_staticVertexBuffer;
		nvrhi::BufferHandle m_globalCBuffer;
		uint32_t m_globalCBufferSlot;
		uint32_t m_localCBufferSlot;
		uint32_t m_localCBufferSize;
		nvrhi::FramebufferHandle m_nullFbo;
		nvrhi::FramebufferHandle m_debugFbo;
		std::vector<nvrhi::BufferHandle> m_transientPool;
		std::vector<nvrhi::ResourceHandle> m_pipelines;
		std::vector<std::pair<nvrhi::SamplerHandle, uint32_t>> m_samplers;
		BindingCache* m_bindingCache;

		omm::Baker m_baker;
		omm::Baker m_cpuBaker;
		omm::Gpu::Pipeline m_pipeline;
		bool m_enableDebug = false;
	};
} // namespace omm
