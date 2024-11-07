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

#include <comdef.h> // for _com_ptr_t (optional, for convenience)
#include <dxgi.h>
#include <d3d12.h>

#include <nvrhi/validation.h>

template <typename T>
class ComPtrWrapper {
public:
	ComPtrWrapper() : ptr_(nullptr) {}
	ComPtrWrapper(T* ptr) : ptr_(ptr) {
		if (ptr_) ptr_->AddRef();
	}

	~ComPtrWrapper() {
		if (ptr_) ptr_->Release();
	}

	ComPtrWrapper(const ComPtrWrapper& other) : ptr_(other.ptr_) {
		if (ptr_) ptr_->AddRef();
	}

	ComPtrWrapper(ComPtrWrapper&& other) noexcept : ptr_(other.ptr_) {
		other.ptr_ = nullptr;
	}

	ComPtrWrapper& operator=(const ComPtrWrapper& other) {
		if (this != &other) {
			if (ptr_) ptr_->Release();
			ptr_ = other.ptr_;
			if (ptr_) ptr_->AddRef();
		}
		return *this;
	}

	// Access underlying pointer
	T* Get() const { return ptr_; }
	T** GetAddressOf() { return &ptr_; }
	T* operator->() const { return ptr_; }
	operator bool() const { return ptr_ != nullptr; }

private:
	T* ptr_;
};


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

	static ComPtrWrapper<IDXGIAdapter> FindAdapter(const std::wstring& targetName)
	{
		ComPtrWrapper<IDXGIAdapter> targetAdapter;
		ComPtrWrapper<IDXGIFactory1> DXGIFactory;
		HRESULT hres = CreateDXGIFactory1(IID_PPV_ARGS(DXGIFactory.GetAddressOf()));
		assert(hres == S_OK);

		unsigned int adapterNo = 0;
		while (SUCCEEDED(hres))
		{
			ComPtrWrapper<IDXGIAdapter> pAdapter;
			hres = DXGIFactory->EnumAdapters(adapterNo, pAdapter.GetAddressOf());

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
				ComPtrWrapper<ID3D12Debug> debugController;
				if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(debugController.GetAddressOf()))))
				{
					debugController->EnableDebugLayer();
				}
			}

			targetAdapter = FindAdapter(desc.adapterNameSubstring);

			HRESULT hr = D3D12CreateDevice(
				targetAdapter.Get(),
				D3D_FEATURE_LEVEL_12_0,
				IID_PPV_ARGS(device12.GetAddressOf()));
			assert(hr == S_OK);

			if (desc.enableDebugRuntime)
			{
				ComPtrWrapper<ID3D12InfoQueue> pInfoQueue;
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
			hr = device12->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(graphicsQueue.GetAddressOf()));
			assert(hr == S_OK);
			graphicsQueue->SetName(L"Graphics Queue");
		}

		~NVRHIContextDX12()
		{
		}

	private:
		const NVRHIContext::InitParams desc;
		nvrhi::DeviceHandle nvrhiDevice;
		ComPtrWrapper<IDXGIAdapter> targetAdapter;
		ComPtrWrapper<ID3D12Device> device12;
		ComPtrWrapper<ID3D12CommandQueue> graphicsQueue;
	};

}  // namespace


std::unique_ptr<NVRHIContext> NVRHIContext::Init(const NVRHIContext::InitParams& desc)
{
	if (desc.api == nvrhi::GraphicsAPI::D3D12)
		return std::make_unique<NVRHIContextDX12>(NVRHIContextDX12(desc));
	return nullptr;
}