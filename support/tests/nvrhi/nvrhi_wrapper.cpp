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
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <nvrhi/nvrhi.h>
#include <nvrhi/d3d12.h>
#include <nvrhi/validation.h>

#include <dxgi.h>
#include <dxgi1_4.h>
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

	static void PrintAdapterInfo(IDXGIAdapter* adapter)
	{
		std::string gpuName = "unavailable";
		DXGI_ADAPTER_DESC adapterDesc = {};
		if (adapter && SUCCEEDED(adapter->GetDesc(&adapterDesc)))
		{
			const int nameSize = WideCharToMultiByte(CP_UTF8, 0, adapterDesc.Description, -1, nullptr, 0, nullptr, nullptr);
			if (nameSize > 0)
			{
				gpuName.resize(nameSize);
				if (WideCharToMultiByte(CP_UTF8, 0, adapterDesc.Description, -1, gpuName.data(), nameSize, nullptr, nullptr) > 0)
					gpuName.pop_back(); // Conversion size includes the terminating null.
				else
					gpuName = "unavailable";
			}
		}
		std::cout << "[ OMM GPU  ] GPU: " << gpuName << std::endl;

		LARGE_INTEGER driverVersion = {};
		// Query IDXGIDevice rather than ID3D12Device to retrieve the UMD version.
		if (adapter && SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driverVersion)))
		{
			std::cout << "[ OMM GPU  ] Windows driver: "
				<< HIWORD(driverVersion.HighPart) << '.' << LOWORD(driverVersion.HighPart) << '.'
				<< HIWORD(driverVersion.LowPart) << '.' << LOWORD(driverVersion.LowPart) << std::endl;
		}
		else
			std::cout << "[ OMM GPU  ] Windows driver: unavailable" << std::endl;
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
			std::cout << "[ OMM GPU  ] API: D3D12" << std::endl;
			if (desc.enableDebugRuntime)
			{
				ComPtrWrapper<ID3D12Debug> debugController;
				if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(debugController.GetAddressOf()))))
				{
					debugController->EnableDebugLayer();
				}
			}

			if (desc.useWarp)
			{
				ComPtrWrapper<IDXGIFactory4> factory;
				HRESULT warpResult = CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf()));
				if (SUCCEEDED(warpResult))
					warpResult = factory->EnumWarpAdapter(IID_PPV_ARGS(targetAdapter.GetAddressOf()));
				if (FAILED(warpResult))
				{
					std::ostringstream message;
					message << "Explicit WARP adapter selection failed (HRESULT 0x" << std::hex << std::uppercase
						<< std::setw(8) << std::setfill('0') << static_cast<unsigned long>(warpResult)
						<< "). OMM_TEST_USE_WARP=ON requires IDXGIFactory4 and an available WARP adapter; hardware fallback is disabled.";
					throw std::runtime_error(message.str());
				}
			}
			else
				targetAdapter = FindAdapter(desc.adapterNameSubstring);
			PrintAdapterInfo(targetAdapter.Get());

			HRESULT hr = D3D12CreateDevice(
				targetAdapter.Get(),
				D3D_FEATURE_LEVEL_12_0,
				IID_PPV_ARGS(device12.GetAddressOf()));
			if (FAILED(hr))
			{
				std::ostringstream message;
				message << "D3D12CreateDevice at feature level 12_0 failed (HRESULT 0x" << std::hex << std::uppercase
					<< std::setw(8) << std::setfill('0') << static_cast<unsigned long>(hr) << ").";
				throw std::runtime_error(message.str());
			}

			if (desc.disableDriverBackgroundOptimizations)
			{
				auto CheckControlResult = [](HRESULT result, const char* operation)
				{
					if (FAILED(result))
					{
						std::ostringstream message;
						message << operation << " failed (HRESULT 0x" << std::hex << std::uppercase
							<< std::setw(8) << std::setfill('0') << static_cast<unsigned long>(result)
							<< "). Disabling D3D12 driver background optimizations requires ID3D12Device6 and Windows Developer Mode. "
							<< "Enable Windows Developer Mode before running GPU tests.";
						throw std::runtime_error(message.str());
					}
				};

				// Apply before creating queues or PSOs so no workload can train dynamic optimizations.
				ComPtrWrapper<ID3D12Device6> device6;
				CheckControlResult(device12->QueryInterface(IID_PPV_ARGS(device6.GetAddressOf())), "QueryInterface(ID3D12Device6)");
				CheckControlResult(device6->SetBackgroundProcessingMode(
					D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_PROFILING_BY_SYSTEM,
					D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr), "SetBackgroundProcessingMode");
				std::cout << "[ OMM GPU  ] D3D12 driver background optimizations: DISABLED" << std::endl;
			}
			else
			{
				std::cout << "[ OMM GPU  ] D3D12 driver background optimizations: DEFAULT (control disabled)" << std::endl;
			}

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
