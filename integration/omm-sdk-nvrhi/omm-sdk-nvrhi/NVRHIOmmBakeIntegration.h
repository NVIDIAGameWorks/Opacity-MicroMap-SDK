/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <omm.h>
#include <nvrhi/nvrhi.h>
#include <vector>
#include <stdint.h>
#include <utility>

class BindingCache;

class NVRHIVmBakeIntegration
{
public:

	NVRHIVmBakeIntegration(nvrhi::DeviceHandle device, nvrhi::CommandListHandle commandList, bool enableDebug);
	~NVRHIVmBakeIntegration();

	struct Input
	{
		nvrhi::TextureHandle		alphaTexture;
		uint32_t					alphaTextureChannel = 3;
		float						alphaCutoff			= 0.5f;
		bool						bilinearFilter		= true;
		nvrhi::SamplerAddressMode	sampleMode			= nvrhi::SamplerAddressMode::Clamp;

		nvrhi::BufferHandle		texCoordBuffer;
		uint32_t				texCoordBufferOffsetInBytes = 0;
		uint32_t				texCoordStrideInBytes		= 0;
		nvrhi::BufferHandle		indexBuffer;
		uint32_t				indexBufferOffsetInBytes	= 0;
		size_t					numIndices					= 0;

		uint32_t				globalSubdivisionLevel		= 0;
		bool					use2State					= false;
		float					dynamicSubdivisionScale		= 0.5f;
		bool					minimalMemoryMode			= false;
		bool					enableSpecialIndices		= true;
		bool					force32BitIndices			= false;
		bool					enableTexCoordDeuplication	= true;
		bool					computeOnly					= false;
	};

	struct PreBakeInfo
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

	struct Output
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
	void GetPreBakeInfo(const Input& params, PreBakeInfo& info);

	// Run VM bake on GPU
	void RunBake(
		nvrhi::CommandListHandle commandList,
		const Input& params,
		const Output& result);

	// This assumes pData is the CPU-side pointer of the contents in vmUsageDescReadbackBufferSize.
	static void ReadPostBuildInfo(void* pData, size_t byteSize, PostBuildInfo& outPostBuildInfo);
	static void ReadUsageDescBuffer(void* pData, size_t byteSize, std::vector<OpacityMicromapUsageCount> & outVmUsages);

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
	void InitBaker();
	void DestroyBaker();

	void SetupPipelines(
		const omm::Gpu::BakePipelineInfoDesc* desc);

	omm::Gpu::BakeDispatchConfigDesc GetConfig(const Input& params);

	void ReserveGlobalCBuffer(size_t size, uint32_t slot);
	void ReserveScratchBuffers(const omm::Gpu::PreBakeInfo& info);
	nvrhi::TextureHandle GetTextureResource(const Input& params, const Output& output, const omm::Gpu::Resource& resource);
	nvrhi::BufferHandle GetBufferResource(const Input& params, const Output& output, const omm::Gpu::Resource& resource, uint32_t& offsetInBytes);

	void ExecuteBakeOperation(
		nvrhi::CommandListHandle commandList,
		const Input& params,
		const Output& output,
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