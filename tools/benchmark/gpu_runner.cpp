/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gpu_runner.h"
#include "subdivision.h"

#include <omm-gpu-nvrhi.h>
#include <nvrhi/d3d12.h>
#include <nvrhi/validation.h>

#include <directx/d3d12sdklayers.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace omm::benchmark
{
    namespace
    {
        using Microsoft::WRL::ComPtr;

        void Check(HRESULT result, const char* operation)
        {
            if (FAILED(result))
            {
                std::ostringstream text;
                text << operation << " failed (HRESULT 0x" << std::hex << static_cast<uint32_t>(result) << ")";
                throw std::runtime_error(text.str());
            }
        }

        void Check(omm::Result result, const char* operation)
        {
            if (result != omm::Result::SUCCESS)
                throw std::runtime_error(std::string(operation) + " failed (omm::Result "
                    + std::to_string(static_cast<int>(result)) + ")");
        }

        std::string ToUtf8(const wchar_t* text)
        {
            const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
            if (size <= 0)
                return "Unknown adapter";
            std::string result(static_cast<size_t>(size), '\0');
            WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
            result.pop_back();
            return result;
        }

        uint32_t TexCoordSize(TexCoordFormat format)
        {
            switch (format)
            {
            case TexCoordFormat::UV16_UNORM:
            case TexCoordFormat::UV16_FLOAT: return 4;
            case TexCoordFormat::UV32_FLOAT: return 8;
            default: throw std::runtime_error("Unsupported texture coordinate format");
            }
        }

        nvrhi::Format TexCoordViewFormat(TexCoordFormat format)
        {
            switch (format)
            {
            case TexCoordFormat::UV16_UNORM: return nvrhi::Format::R16_UNORM;
            case TexCoordFormat::UV16_FLOAT: return nvrhi::Format::R16_FLOAT;
            case TexCoordFormat::UV32_FLOAT: return nvrhi::Format::R32_FLOAT;
            default: throw std::runtime_error("Unsupported texture coordinate format");
            }
        }

        nvrhi::SamplerAddressMode AddressMode(TextureAddressMode mode)
        {
            switch (mode)
            {
            case TextureAddressMode::Wrap: return nvrhi::SamplerAddressMode::Wrap;
            case TextureAddressMode::Mirror: return nvrhi::SamplerAddressMode::Mirror;
            case TextureAddressMode::Clamp: return nvrhi::SamplerAddressMode::Clamp;
            case TextureAddressMode::Border: return nvrhi::SamplerAddressMode::Border;
            case TextureAddressMode::MirrorOnce:
                throw std::runtime_error("GPU benchmark does not support MirrorOnce sampler addressing");
            default: throw std::runtime_error("Unsupported sampler address mode");
            }
        }

        void Validate(const Cpu::BakeInputDesc& input)
        {
            if (!input.texture || !input.texCoords || !input.indexBuffer || !input.indexCount || input.indexCount % 3 != 0)
                throw std::runtime_error("GPU input requires a texture, UVs, and a nonempty triangle index buffer");
            if (input.alphaMode != AlphaMode::Test)
                throw std::runtime_error("GPU benchmark supports AlphaMode::Test only");
            if (input.formats)
                throw std::runtime_error("GPU benchmark does not support per-triangle formats");
            if (input.subdivisionLevels)
                throw std::runtime_error("GPU benchmark does not support per-triangle subdivision; use --subdivision to override it");
            if (input.maxSubdivisionLevel > 12)
                throw std::runtime_error("GPU subdivision level must be in the range 0..12");
            if (input.format != Format::OC1_2_State && input.format != Format::OC1_4_State)
                throw std::runtime_error("Unsupported opacity micromap format");
            const auto compatibleState = [&](OpacityState state)
            {
                const auto value = static_cast<uint32_t>(state);
                return value <= static_cast<uint32_t>(OpacityState::UnknownOpaque)
                    && (input.format != Format::OC1_2_State || value <= static_cast<uint32_t>(OpacityState::Opaque));
            };
            if (!compatibleState(input.alphaCutoffGreater) || !compatibleState(input.alphaCutoffLessEqual))
                throw std::runtime_error("Alpha cutoff states are incompatible with the opacity micromap format");
            if (input.runtimeSamplerDesc.filter != TextureFilterMode::Nearest
                && input.runtimeSamplerDesc.filter != TextureFilterMode::Linear)
                throw std::runtime_error("Unsupported sampler filter mode");
            if (input.runtimeSamplerDesc.borderAlpha != 0)
                throw std::runtime_error("GPU benchmark supports zero border alpha only");
            if (input.rejectionThreshold != 0)
                throw std::runtime_error("GPU benchmark does not support CPU rejection thresholds");
            if (input.unresolvedTriState != SpecialIndex::FullyUnknownOpaque)
                throw std::runtime_error("GPU benchmark requires FullyUnknownOpaque for unresolved triangles");
            const uint32_t packedSize = TexCoordSize(input.texCoordFormat);
            if (input.texCoordStrideInBytes && input.texCoordStrideInBytes != packedSize)
                throw std::runtime_error("GPU benchmark requires packed UVs in serialized inputs");
            if (!std::isfinite(input.alphaCutoff) || input.alphaCutoff < 0 || input.alphaCutoff > 1)
                throw std::runtime_error("Invalid GPU alpha cutoff");
            if (input.subdivision.mode == SubdivisionMode::Legacy && !std::isfinite(input.dynamicSubdivisionScale))
                throw std::runtime_error("Invalid legacy dynamic subdivision scale");
            const auto subdivision = EffectiveSubdivision(input, false);
            const auto positive = [](float value) { return std::isfinite(value) && value > 0.f; };
            if (subdivision.mode != SubdivisionMode::Fixed && subdivision.mode != SubdivisionMode::Area
                && subdivision.mode != SubdivisionMode::Edge)
                throw std::runtime_error("Invalid GPU subdivision mode");
            if ((subdivision.mode == SubdivisionMode::Area && !positive(subdivision.targetAreaInSquareTexels))
                || ((subdivision.mode == SubdivisionMode::Area || subdivision.mode == SubdivisionMode::Edge)
                    && !positive(subdivision.targetEdgeLengthInTexels)))
                throw std::runtime_error("GPU subdivision targets must be finite and positive");

            const auto flags = static_cast<uint32_t>(input.bakeFlags);
            constexpr uint32_t mappedFlags = static_cast<uint32_t>(Cpu::BakeFlags::DisableSpecialIndices)
                | static_cast<uint32_t>(Cpu::BakeFlags::Force32BitIndices)
                | static_cast<uint32_t>(Cpu::BakeFlags::DisableDuplicateDetection);
            if (flags & ~mappedFlags)
                std::cerr << "Warning: CPU-only bake flags are ignored by the GPU benchmark (0x"
                    << std::hex << (flags & ~mappedFlags) << std::dec << ").\n";
            if (input.unknownStatePromotion != UnknownStatePromotion::Nearest)
                std::cerr << "Warning: GPU baking uses nearest unknown-state promotion instead of the saved CPU setting.\n";
            if (input.maxWorkloadSize != std::numeric_limits<uint64_t>::max())
                std::cerr << "Warning: The CPU workload limit is ignored by the GPU benchmark.\n";
        }
    }

    struct GpuContext::Impl : nvrhi::IMessageCallback
    {
        // COM objects and the callback must outlive all NVRHI objects.
        ComPtr<ID3D12Device> nativeDevice;
        ComPtr<ID3D12CommandQueue> queue;
        std::string adapterName;
        std::string firstError;
        nvrhi::DeviceHandle device;
        nvrhi::CommandListHandle commandList;
        nvrhi::TimerQueryHandle timer;
        std::unique_ptr<GpuBakeNvrhi> baker;

        explicit Impl(bool debug)
        {
            if (debug)
            {
                ComPtr<ID3D12Debug> debugLayer;
                Check(D3D12GetDebugInterface(IID_PPV_ARGS(debugLayer.GetAddressOf())), "D3D12GetDebugInterface");
                debugLayer->EnableDebugLayer();
            }

            ComPtr<IDXGIFactory1> factory;
            Check(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())), "CreateDXGIFactory1");
            for (UINT i = 0; ; ++i)
            {
                ComPtr<IDXGIAdapter1> adapter;
                const HRESULT result = factory->EnumAdapters1(i, adapter.GetAddressOf());
                if (result == DXGI_ERROR_NOT_FOUND)
                    break;
                Check(result, "EnumAdapters1");
                DXGI_ADAPTER_DESC1 desc{};
                Check(adapter->GetDesc1(&desc), "GetDesc1");
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                    continue;
                if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                    IID_PPV_ARGS(nativeDevice.GetAddressOf()))))
                {
                    adapterName = ToUtf8(desc.Description);
                    break;
                }
            }
            if (!nativeDevice)
                throw std::runtime_error("No hardware adapter supports Direct3D 12 feature level 12_0");

            D3D12_COMMAND_QUEUE_DESC queueDesc{};
            queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            Check(nativeDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(queue.GetAddressOf())), "CreateCommandQueue");

            nvrhi::d3d12::DeviceDesc deviceDesc;
            deviceDesc.errorCB = this;
            deviceDesc.pDevice = nativeDevice.Get();
            deviceDesc.pGraphicsCommandQueue = queue.Get();
            device = nvrhi::d3d12::createDevice(deviceDesc);
            CheckMessages();
            if (!device)
                throw std::runtime_error("Could not create the NVRHI D3D12 device");
            if (debug)
            {
#if OMM_BENCHMARK_NVRHI_VALIDATION
                device = nvrhi::validation::createValidationLayer(device);
#else
                throw std::runtime_error("--debug requires NVRHI_WITH_VALIDATION=ON at build time");
#endif
            }
            commandList = device->createCommandList();
            if (!commandList)
                throw std::runtime_error("Could not create the GPU command list");
            timer = device->createTimerQuery();
            if (!timer)
                throw std::runtime_error("Could not create the GPU timestamp query");

            commandList->open();
            baker = GpuBakeNvrhi::Create(device, commandList, false, nullptr,
                [this](MessageSeverity severity, const char* text)
                {
                    if (severity == MessageSeverity::Fatal && firstError.empty())
                        firstError = text;
                    if (severity != MessageSeverity::Info)
                        std::cerr << "OMM GPU: " << text << '\n';
                });
            CheckMessages();
            if (!baker)
                throw std::runtime_error("Could not initialize the GPU baker");
            SubmitAndWait();
        }

        ~Impl()
        {
            if (device)
            {
                device->waitForIdle();
                device->runGarbageCollection();
            }
        }

        void message(nvrhi::MessageSeverity severity, const char* text) override
        {
            if ((severity == nvrhi::MessageSeverity::Error || severity == nvrhi::MessageSeverity::Fatal)
                && firstError.empty())
                firstError = text;
            std::cerr << "NVRHI: " << text << '\n';
        }

        void CheckMessages() const
        {
            if (!firstError.empty())
                throw std::runtime_error("GPU error: " + firstError);
        }

        void SubmitAndWait()
        {
            commandList->close();
            CheckMessages();
            device->executeCommandList(commandList);
            if (!device->waitForIdle())
                throw std::runtime_error("Waiting for GPU work failed");
            CheckMessages();
            Check(nativeDevice->GetDeviceRemovedReason(), "GPU execution");
            device->runGarbageCollection();
        }

        nvrhi::BufferHandle CreateBuffer(uint64_t size, const char* name, bool output,
            nvrhi::Format format = nvrhi::Format::UNKNOWN)
        {
            nvrhi::BufferDesc desc;
            desc.byteSize = (std::max<uint64_t>(size, 4) + 3) & ~uint64_t(3);
            desc.debugName = name;
            desc.format = format;
            desc.canHaveRawViews = true;
            desc.canHaveTypedViews = format != nvrhi::Format::UNKNOWN;
            desc.canHaveUAVs = output;
            desc.initialState = output ? nvrhi::ResourceStates::Common : nvrhi::ResourceStates::ShaderResource;
            desc.keepInitialState = true;
            auto buffer = device->createBuffer(desc);
            CheckMessages();
            if (!buffer)
                throw std::runtime_error(std::string("Could not allocate GPU buffer: ") + name);
            return buffer;
        }
    };

    struct GpuInput::Impl
    {
        std::shared_ptr<GpuContext::Impl> context;
        GpuBakeNvrhi::Input input;
        GpuBakeNvrhi::PreDispatchInfo info{};
        GpuBakeNvrhi::Buffers buffers;
        nvrhi::BufferHandle readback;
        bool hasRun = false;

        Impl(std::shared_ptr<GpuContext::Impl> context, const Cpu::BakeInputDesc& source, bool computeOnly)
            : context(std::move(context))
        {
            Validate(source);
            auto& gpu = *this->context;
            // The SDK's graphics topology buffers contain levels 0 through 9.
            if (!computeOnly && source.maxSubdivisionLevel > 9)
                throw std::runtime_error("Graphics GPU baking supports subdivision levels 0..9; omit --graphics for 10..12");
            if (!computeOnly && !gpu.device->queryFeatureSupport(nvrhi::Feature::ConservativeRasterization))
                throw std::runtime_error("The GPU lacks conservative rasterization; omit --graphics");

            Cpu::TextureDesc textureDesc;
            Check(Cpu::GetTextureDesc(source.texture, &textureDesc), "GetTextureDesc");
            if (textureDesc.mipCount != 1)
                throw std::runtime_error("GPU benchmark requires a single-mip alpha texture");
            Cpu::TextureMipDesc mip;
            textureDesc.mips = &mip;
            Check(Cpu::GetTextureDesc(source.texture, &textureDesc), "GetTextureDesc dimensions");
            if (!mip.width || !mip.height || mip.width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
                || mip.height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
                throw std::runtime_error("Alpha texture dimensions exceed the D3D12 texture limits");
            uint32_t texelBytes;
            nvrhi::Format textureFormat;
            if (textureDesc.format == Cpu::TextureFormat::UNORM8)
            {
                texelBytes = 1;
                textureFormat = nvrhi::Format::R8_UNORM;
            }
            else if (textureDesc.format == Cpu::TextureFormat::FP32)
            {
                texelBytes = 4;
                textureFormat = nvrhi::Format::R32_FLOAT;
            }
            else
                throw std::runtime_error("Unsupported CPU alpha texture format");
            std::vector<uint8_t> pixels(size_t(mip.width) * mip.height * texelBytes);
            mip.textureData = pixels.data();
            Check(Cpu::GetTextureDesc(source.texture, &textureDesc), "GetTextureDesc pixels");

            uint32_t maxIndex = 0;
            std::vector<uint32_t> expandedIndices;
            if (source.indexFormat == IndexFormat::UINT_8)
                expandedIndices.resize(source.indexCount);
            for (uint32_t i = 0; i < source.indexCount; ++i)
            {
                uint32_t index = 0;
                switch (source.indexFormat)
                {
                case IndexFormat::UINT_8:
                    index = static_cast<const uint8_t*>(source.indexBuffer)[i];
                    expandedIndices[i] = index;
                    break;
                case IndexFormat::UINT_16:
                {
                    uint16_t value;
                    std::memcpy(&value, static_cast<const uint8_t*>(source.indexBuffer) + size_t(i) * 2, 2);
                    index = value;
                    break;
                }
                case IndexFormat::UINT_32:
                    std::memcpy(&index, static_cast<const uint8_t*>(source.indexBuffer) + size_t(i) * 4, 4);
                    break;
                default: throw std::runtime_error("Unsupported input index format");
                }
                maxIndex = std::max(maxIndex, index);
            }
            const uint64_t uvBytes = (uint64_t(maxIndex) + 1) * TexCoordSize(source.texCoordFormat);
            if (uvBytes > std::numeric_limits<uint32_t>::max())
                throw std::runtime_error("GPU texture coordinate buffer exceeds 4 GiB");
            const bool shortIndices = source.indexFormat == IndexFormat::UINT_16;
            const uint64_t indexBytes = uint64_t(source.indexCount) * (shortIndices ? 2 : 4);

            input.operation = GpuBakeNvrhi::Operation::SetupAndBake;
            input.alphaTextureChannel = 0;
            input.alphaCutoff = source.alphaCutoff;
            input.alphaCutoffGreater = source.alphaCutoffGreater;
            input.alphaCutoffLessEqual = source.alphaCutoffLessEqual;
            input.bilinearFilter = source.runtimeSamplerDesc.filter == TextureFilterMode::Linear;
            input.sampleMode = AddressMode(source.runtimeSamplerDesc.addressingMode);
            input.texCoordFormat = TexCoordViewFormat(source.texCoordFormat);
            input.texCoordStrideInBytes = source.texCoordStrideInBytes;
            input.numIndices = source.indexCount;
            input.maxSubdivisionLevel = source.maxSubdivisionLevel;
            input.maxOutOmmArraySize = source.maxArrayDataSize;
            input.format = source.format == Format::OC1_2_State
                ? nvrhi::rt::OpacityMicromapFormat::OC1_2_State : nvrhi::rt::OpacityMicromapFormat::OC1_4_State;
            // NVRHI accepts explicit settings; translate serialized legacy inputs at this boundary.
            input.subdivision = EffectiveSubdivision(source, false);
            input.computeOnly = computeOnly;
            const auto sourceFlags = static_cast<uint32_t>(source.bakeFlags);
            input.enableSpecialIndices = (sourceFlags & static_cast<uint32_t>(Cpu::BakeFlags::DisableSpecialIndices)) == 0;
            input.force32BitIndices = (sourceFlags & static_cast<uint32_t>(Cpu::BakeFlags::Force32BitIndices)) != 0;
            input.enableTexCoordDeduplication = (sourceFlags & static_cast<uint32_t>(Cpu::BakeFlags::DisableDuplicateDetection)) == 0;

            nvrhi::TextureDesc desc;
            desc.width = mip.width;
            desc.height = mip.height;
            desc.format = textureFormat;
            desc.debugName = "Benchmark alpha texture";
            desc.initialState = nvrhi::ResourceStates::ShaderResource;
            desc.keepInitialState = true;
            input.alphaTexture = gpu.device->createTexture(desc);
            gpu.CheckMessages();
            if (!input.alphaTexture)
                throw std::runtime_error("Could not allocate the GPU alpha texture");
            input.texCoordBuffer = gpu.CreateBuffer(uvBytes, "Benchmark UVs", false);
            input.indexBuffer = gpu.CreateBuffer(indexBytes, "Benchmark indices", false,
                shortIndices ? nvrhi::Format::R16_UINT : nvrhi::Format::R32_UINT);
            gpu.baker->GetPreDispatchInfo(input, info);
            gpu.CheckMessages();
            buffers.ommArrayBuffer = gpu.CreateBuffer(info.ommArrayBufferSize, "Benchmark OMM array", true);
            buffers.ommDescBuffer = gpu.CreateBuffer(info.ommDescBufferSize, "Benchmark OMM descriptors", true);
            buffers.ommIndexBuffer = gpu.CreateBuffer(info.ommIndexBufferSize, "Benchmark OMM indices", true);
            buffers.ommDescArrayHistogramBuffer = gpu.CreateBuffer(info.ommDescArrayHistogramSize, "Benchmark descriptor histogram", true);
            buffers.ommIndexHistogramBuffer = gpu.CreateBuffer(info.ommIndexHistogramSize, "Benchmark index histogram", true);
            buffers.ommPostDispatchInfoBuffer = gpu.CreateBuffer(info.ommPostDispatchInfoBufferSize, "Benchmark post-dispatch info", true);

            nvrhi::BufferDesc readbackDesc;
            readbackDesc.byteSize = info.ommPostDispatchInfoBufferSize;
            readbackDesc.debugName = "Benchmark post-dispatch readback";
            readbackDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
            readbackDesc.initialState = nvrhi::ResourceStates::CopyDest;
            readbackDesc.keepInitialState = true;
            readback = gpu.device->createBuffer(readbackDesc);
            gpu.CheckMessages();
            if (!readback)
                throw std::runtime_error("Could not create the GPU readback buffer");

            gpu.commandList->open();
            // GetTextureDesc rowPitch is in texels; NVRHI expects bytes.
            gpu.commandList->writeTexture(input.alphaTexture, 0, 0, pixels.data(), size_t(mip.rowPitch) * texelBytes);
            gpu.commandList->writeBuffer(input.texCoordBuffer, source.texCoords, uvBytes);
            gpu.commandList->writeBuffer(input.indexBuffer,
                expandedIndices.empty() ? source.indexBuffer : expandedIndices.data(), indexBytes);
            gpu.SubmitAndWait();
        }

        double Run()
        {
            auto& gpu = *context;
            gpu.device->resetTimerQuery(gpu.timer);
            gpu.commandList->open();
            gpu.commandList->beginTimerQuery(gpu.timer);
            gpu.baker->Dispatch(gpu.commandList, input, buffers);
            gpu.commandList->endTimerQuery(gpu.timer);
            gpu.SubmitAndWait();
            if (!gpu.device->pollTimerQuery(gpu.timer))
                throw std::runtime_error("GPU timestamp results are unavailable after synchronization");
            const double milliseconds = double(gpu.device->getTimerQueryTime(gpu.timer)) * 1000;
            gpu.CheckMessages();
            if (!std::isfinite(milliseconds) || milliseconds < 0)
                throw std::runtime_error("GPU returned an invalid timestamp duration");
            hasRun = true;
            return milliseconds;
        }

        GpuStats ReadStats()
        {
            if (!hasRun)
                throw std::runtime_error("GPU bake output is unavailable before the first run");
            auto& gpu = *context;
            gpu.commandList->open();
            gpu.commandList->copyBuffer(readback, 0, buffers.ommPostDispatchInfoBuffer, 0, info.ommPostDispatchInfoBufferSize);
            gpu.SubmitAndWait();
            void* data = gpu.device->mapBuffer(readback, nvrhi::CpuAccessMode::Read);
            if (!data)
                throw std::runtime_error("Could not map the GPU output statistics");
            GpuBakeNvrhi::PostDispatchInfo post{};
            GpuBakeNvrhi::ReadPostDispatchInfo(data, info.ommPostDispatchInfoBufferSize, post);
            gpu.device->unmapBuffer(readback);
            gpu.CheckMessages();
            if (post.ommArrayBufferSize > info.ommArrayBufferSize || post.ommDescBufferSize > info.ommDescBufferSize
                || post.ommDescBufferSize % sizeof(Cpu::OpacityMicromapDesc) != 0)
                throw std::runtime_error("GPU bake returned invalid output sizes");
            GpuStats stats;
            stats.arrayBytes = post.ommArrayBufferSize;
            stats.micromapCount = post.ommDescBufferSize / sizeof(Cpu::OpacityMicromapDesc);
            stats.indexCount = info.ommIndexCount;
            stats.outputBytes = uint64_t(post.ommArrayBufferSize) + post.ommDescBufferSize
                + info.ommIndexBufferSize + info.ommDescArrayHistogramSize + info.ommIndexHistogramSize;
            return stats;
        }
    };

    GpuInput::GpuInput(std::unique_ptr<Impl> impl) : m_impl(std::move(impl)) {}
    GpuInput::~GpuInput()
    {
        // Binding sets retain each input's resources; release them between batch inputs.
        // Run and ReadStats synchronize before returning, so no work is still using them.
        auto context = m_impl->context;
        context->baker->Clear();
        m_impl.reset();
        context->device->runGarbageCollection();
    }
    double GpuInput::Run() { return m_impl->Run(); }
    GpuStats GpuInput::ReadStats() { return m_impl->ReadStats(); }

    GpuContext::GpuContext(bool debug) : m_impl(std::make_shared<Impl>(debug)) {}
    GpuContext::~GpuContext() = default;
    const std::string& GpuContext::AdapterName() const { return m_impl->adapterName; }
    std::unique_ptr<GpuInput> GpuContext::Prepare(const Cpu::BakeInputDesc& input, bool computeOnly)
    {
        return std::unique_ptr<GpuInput>(new GpuInput(std::make_unique<GpuInput::Impl>(m_impl, input,
            computeOnly)));
    }
}
