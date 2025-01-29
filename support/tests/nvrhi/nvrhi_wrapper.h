#pragma once

#include <nvrhi/nvrhi.h>
#include <memory>

struct NVRHIContext
{
	struct InitParams
	{
		nvrhi::GraphicsAPI api = nvrhi::GraphicsAPI::D3D12;
		std::wstring adapterNameSubstring = L"";
		bool enableDebugRuntime = true;
		bool enableNvrhiValidationLayer = true;
	};

	static std::unique_ptr<NVRHIContext> Init(const InitParams& desc);

	virtual ~NVRHIContext() {  }

	[[nodiscard]] virtual nvrhi::DeviceHandle CreateDevice() const = 0;
};