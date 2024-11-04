/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "nvrhi_wrapper.h"

#include <iostream>

#include <nvrhi/nvrhi.h>
#include <nvrhi/d3d12.h>
#include <nvrhi/validation.h>

//#include <atlbase.h>
#include <wrl/client.h>
#include <dxgi.h>
#include <d3d12.h>

#include <nvrhi/validation.h>

using namespace Microsoft::WRL;

namespace {

	struct DefaultMessageCallback : public nvrhi::IMessageCallback
	{
		static DefaultMessageCallback& GetInstance()
		{
			static DefaultMessageCallback s_instance;
			return s_instance;
		}

		void message(nvrhi::MessageSeverity severity, const char* messageText) override
		{
			std::cout << messageText << std::endl;
		}
	};

	static ComPtr<IDXGIAdapter> FindAdapter(const std::wstring& targetName)
	{
		ComPtr<IDXGIAdapter> targetAdapter;
		ComPtr<IDXGIFactory1> DXGIFactory;
		HRESULT hres = CreateDXGIFactory1(IID_PPV_ARGS(&DXGIFactory));
		assert(hres == S_OK);

		unsigned int adapterNo = 0;
		while (SUCCEEDED(hres))
		{
			ComPtr<IDXGIAdapter> pAdapter;
			hres = DXGIFactory->EnumAdapters(adapterNo, &pAdapter);

			if (SUCCEEDED(hres))
			{
				DXGI_ADAPTER_DESC aDesc;
				pAdapter->GetDesc(&aDesc);

				// If no name is specified, return the first adapater.  This is the same behaviour as the
				// default specified for D3D11CreateDevice when no adapter is specified.
				if (targetName.length() == 0)
				{
					targetAdapter = pAdapter;
					break;
				}

				std::wstring aName = aDesc.Description;

				if (aName.find(targetName) != std::string::npos)
				{
					targetAdapter = pAdapter;
					break;
				}
			}

			adapterNo++;
		}

		return targetAdapter;
	}

	class NVRHIContextDX12 : public NVRHIContext {
	public:

		[[nodiscard]] virtual nvrhi::DeviceHandle CreateDevice() const
		{
			nvrhi::d3d12::DeviceDesc deviceDesc;
			deviceDesc.errorCB = &DefaultMessageCallback::GetInstance();
			deviceDesc.pDevice = device12.Get();
			deviceDesc.pGraphicsCommandQueue = graphicsQueue.Get();
			deviceDesc.pComputeCommandQueue = nullptr;
			deviceDesc.pCopyCommandQueue = nullptr;

			nvrhi::DeviceHandle nvrhiDevice = nvrhi::d3d12::createDevice(deviceDesc);

			if (desc.enableNvrhiValidationLayer)
			{
				nvrhiDevice = nvrhi::validation::createValidationLayer(nvrhiDevice);
			}

			return nvrhiDevice;
		}

		NVRHIContextDX12(const NVRHIContext::InitParams& desc) :desc(desc)
		{
			if (desc.enableDebugRuntime)
			{
				ComPtr<ID3D12Debug> debugController;
				if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
				{
					debugController->EnableDebugLayer();
				}
			}

			targetAdapter = FindAdapter(desc.adapterNameSubstring);

			HRESULT hr = D3D12CreateDevice(
				targetAdapter.Get(),
				D3D_FEATURE_LEVEL_12_0,
				IID_PPV_ARGS(&device12));
			assert(hr == S_OK);

			if (desc.enableDebugRuntime)
			{
				ComPtr<ID3D12InfoQueue> pInfoQueue;
				device12->QueryInterface(pInfoQueue.GetAddressOf());

				if (pInfoQueue)
				{
					pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_MESSAGE, true);
					pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
					pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
				}
			}

			D3D12_COMMAND_QUEUE_DESC queueDesc;
			ZeroMemory(&queueDesc, sizeof(queueDesc));
			queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
			queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
			queueDesc.NodeMask = 1;
			hr = device12->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&graphicsQueue));
			assert(hr == S_OK);
			graphicsQueue->SetName(L"Graphics Queue");
		}

		~NVRHIContextDX12()
		{
		}

	private:
		const NVRHIContext::InitParams desc;
		nvrhi::DeviceHandle nvrhiDevice;
		ComPtr<IDXGIAdapter> targetAdapter;
		ComPtr<ID3D12Device> device12;
		ComPtr<ID3D12CommandQueue> graphicsQueue;
	};

}  // namespace


std::unique_ptr<NVRHIContext> NVRHIContext::Init(const NVRHIContext::InitParams& desc)
{
	if (desc.api == nvrhi::GraphicsAPI::D3D12)
		return std::make_unique<NVRHIContextDX12>(NVRHIContextDX12(desc));
	return nullptr;
}