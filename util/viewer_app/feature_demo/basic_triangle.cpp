/*
* Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>
#include <donut/app/imgui_console.h>
#include <donut/app/imgui_renderer.h>

using namespace donut;
using namespace donut::math;

#include "shader_cb.h"

#include <fstream>
#include <vector>
#include <cstdint>

#include <omm.hpp>

using namespace donut;

static const char* g_WindowTitle = "Opacity Micro-Map Viewer Tool";

static const char* ToString(omm::Result result)
{
    switch (result)
    {
    case omm::Result::SUCCESS: return "SUCCESS";
    case omm::Result::FAILURE: return "FAILURE";
    case omm::Result::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case omm::Result::INSUFFICIENT_SCRATCH_MEMORY: return "INSUFFICIENT_SCRATCH_MEMORY";
    case omm::Result::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
    case omm::Result::WORKLOAD_TOO_BIG: return "WORKLOAD_TOO_BIG";
    case omm::Result::MAX_NUM: return "MAX_NUM";
    default:
        return "unknown error code";
    }
}

static void AbortOnFailure(const char* funName, omm::Result result)
{
    donut::log::fatal("%s returned %s", funName, ToString(result));
}

static void PopupOnFailure(const char* funName, omm::Result result)
{
    donut::log::error("%s returned %s", funName, ToString(result));
}

static void Log(omm::MessageSeverity severity, const char* message, void* userArg)
{
    donut::log::message(donut::log::Severity::Info, "[omm-sdk]: %s", message);
}

#define _OMM_ON_ERROR(fun, onError)do { \
    omm::Result res_##__LINE__ = fun; \
    if (res_##__LINE__ != omm::Result::SUCCESS) \
    { \
        onError(#fun, res_##__LINE__); \
    } \
    } while (false)

#define OMM_ABORT_ON_ERROR(fun) _OMM_ON_ERROR(fun, AbortOnFailure)

#define OMM_POPUP_ON_ERROR(fun) _OMM_ON_ERROR(fun, PopupOnFailure)

class OmmLibrary
{
public:
    OmmLibrary():_baker(0)
    {
        omm::BakerCreationDesc desc;
        desc.type = omm::BakerType::CPU;
        desc.messageInterface.messageCallback = &Log;

        OMM_ABORT_ON_ERROR(omm::CreateBaker(desc, &_baker));
    }

    ~OmmLibrary()
    {
        if (_baker != 0)
        {
            OMM_ABORT_ON_ERROR(omm::DestroyBaker(_baker));
        }
    }

    omm::Baker GetBaker() const { return _baker; }

private:
    omm::Baker _baker;
};

class SimpleProfiler {
public:
    SimpleProfiler(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {
       // std::cout << "Profiling started: " << name_ << std::endl;
    }

    ~SimpleProfiler() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
       // std::cout << "Profiling ended: " << name_ << " - Duration: " << duration << " microseconds" << std::endl;
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

enum class Bool3
{
    Default,
    Enable,
    Disable
};

struct UIData
{
    bool ShowUI = true;

    int selectedFile = 0;
    int primitiveStart = 0;
    int primitiveEnd = -1;
    float zoom = 1.f;
    float2 offset = 0.f;
    float2 prevOffset = 0.f;

    std::optional<omm::Cpu::TextureDesc> textureDesc;
    std::optional<omm::Cpu::BakeInputDesc> input;

    bool load = false;
    bool rebake = false;
    bool recompile = true;
};

class OmmGpuData
{
    OmmLibrary _lib;
    UIData& m_ui;
    nvrhi::IDevice* m_device = nullptr;
    nvrhi::CommandListHandle m_commandList = nullptr;

    std::string _fileName;
    std::vector<uint8_t> _data;

    nvrhi::SamplerHandle _sampler;
    nvrhi::TextureHandle _alphaTexture = nullptr;
    nvrhi::BufferHandle _texCoordBuffer = nullptr;
    nvrhi::BufferHandle _indexBuffer = nullptr;
    nvrhi::BufferHandle _ommIndexBuffer = nullptr;
    nvrhi::BufferHandle _ommDesc = nullptr;
    nvrhi::BufferHandle _ommArrayData = nullptr;

    omm::Cpu::TextureDesc _textureDesc;
    omm::Cpu::BakeInputDesc _input;
    omm::Cpu::BakeResult _result = 0;
    const omm::Cpu::BakeResultDesc* _resultDesc = nullptr;
    omm::Debug::Stats _stats;

    uint64_t _bakeTimeInMs = 0;
    uint64_t _bakeTimeInSeconds = 0;
    uint32_t _indexCount = 0;

    void ClearAll()
    {
        if (_result != 0)
        {
            OMM_ABORT_ON_ERROR(omm::Cpu::DestroyBakeResult(_result));
            _result = 0;
        }
        _resultDesc = nullptr;

        _stats = {};

       // _sampler = nullptr;
        _alphaTexture = nullptr;
        _texCoordBuffer = nullptr;
        _indexBuffer = nullptr;
        _ommIndexBuffer = nullptr;
        _ommDesc = nullptr;
        _ommArrayData = nullptr;
    }

public:
    OmmGpuData(UIData& ui):m_ui(ui)
    {
    }

    void Init(nvrhi::IDevice* device)
    {
        m_device = device;
        m_commandList = device->createCommandList();
    }

    ~OmmGpuData()
    {
        ClearAll();
    }

    void Load(const std::string& fileName)
    {
        ClearAll();
        _LoadOmmData(fileName);
        _RebuildOmmData(true /*loadOnly*/);
    }

    void Bake()
    {
        ClearAll();
        _RebuildOmmData(false /*loadOnly*/);
    }

    std::string GetFileName() const { return _fileName; }

    nvrhi::SamplerHandle GetSampler() const { return _sampler; }
    nvrhi::TextureHandle GetAlphaTexture() const { return _alphaTexture; }
    nvrhi::BufferHandle GetIndexBuffer() const { return _indexBuffer; }
    nvrhi::BufferHandle GetTexCoordBuffer() const { return _texCoordBuffer; }
    nvrhi::BufferHandle GetOmmIndexBuffer() const { return _ommIndexBuffer; }
    nvrhi::BufferHandle GetOmmDesc() const { return _ommDesc; }
    nvrhi::BufferHandle GetOmmArrayData() const { return _ommArrayData; }

    uint32_t GetIndexCount() const { return _indexCount; }
    const omm::Cpu::TextureDesc& GetDefaultTextureDesc() const { return _textureDesc; }
    const omm::Cpu::BakeInputDesc& GetDefaultInput() const { return _input; }
    const omm::Cpu::BakeResultDesc* GetResult() const { return _resultDesc; }
    const omm::Debug::Stats& GetStats() const { return _stats; }
    const uint64_t GetBakeTimeInMs() const { return _bakeTimeInMs; }
    const uint64_t GetBakeTimeInSeconds() const { return _bakeTimeInSeconds; }

private:
    std::vector<uint8_t> _LoadDataFile(const std::string& fileName)
    {
        _fileName = fileName;
        // Open the file in binary mode and position the file pointer at the end
        std::ifstream file(fileName, std::ios::binary | std::ios::ate);

        if (!file) {
            throw std::runtime_error("Failed to open file: " + fileName);
        }

        // Determine the file size
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);  // Go back to the beginning of the file

        // Create a vector with enough space and read the file contents into it
        std::vector<uint8_t> buffer(size);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read file: " + fileName);
        }
        return buffer;
    }

    static nvrhi::SamplerAddressMode GetSampler(omm::TextureAddressMode addressMode)
    {
        switch (addressMode)
        {
        case omm::TextureAddressMode::Wrap: return nvrhi::SamplerAddressMode::Wrap;
        case omm::TextureAddressMode::Mirror: return nvrhi::SamplerAddressMode::Mirror;
        case omm::TextureAddressMode::Clamp: return nvrhi::SamplerAddressMode::Clamp;
        case omm::TextureAddressMode::Border: return nvrhi::SamplerAddressMode::Border;
        case omm::TextureAddressMode::MirrorOnce: return nvrhi::SamplerAddressMode::MirrorOnce;
        default:
        {
            assert(false);
            return nvrhi::SamplerAddressMode::Wrap;
        }
        }
    }

    void _InitSampler(const omm::Cpu::BakeInputDesc& input)
    {
        nvrhi::SamplerAddressMode addressMode = GetSampler(input.runtimeSamplerDesc.addressingMode);
        
        auto samplerDesc = nvrhi::SamplerDesc()
            .setAllFilters(false)
            .setAllAddressModes(addressMode);
        samplerDesc.setAllFilters(true);
        _sampler = m_device->createSampler(samplerDesc);
    }

    void _InitTexture(const omm::Cpu::TextureDesc& ommTex)
    {
        nvrhi::TextureDesc d;
        d.height = ommTex.mips[0].height;
        d.width = ommTex.mips[0].width;
        d.format = ommTex.format == omm::Cpu::TextureFormat::FP32 ? nvrhi::Format::R32_FLOAT : nvrhi::Format::R8_UNORM;
        d.initialState = nvrhi::ResourceStates::ShaderResource;
        d.keepInitialState = true;
        d.debugName = "AlphaTexture";
        _alphaTexture = m_device->createTexture(d);

        size_t texelSize = ommTex.format == omm::Cpu::TextureFormat::FP32 ? sizeof(float) : sizeof(uint8_t);

        m_commandList->open();
        m_commandList->setEnableAutomaticBarriers(true);
        m_commandList->writeTexture(_alphaTexture, 0, 0, ommTex.mips[0].textureData, texelSize * ommTex.mips[0].rowPitch);
        m_commandList->close();
        m_device->executeCommandList(m_commandList);
        m_device->waitForIdle();
    }

    void _InitBuffers(const omm::Cpu::BakeInputDesc& input)
    {
        if (input.indexCount != 0)
        {
            nvrhi::BufferDesc ib;
            ib.debugName = "IndexBuffer";
            ib.byteSize = input.indexCount * (input.indexFormat == omm::IndexFormat::UINT_32 ? 4 : 2);
            ib.format = input.indexFormat == omm::IndexFormat::UINT_32 ? nvrhi::Format::R32_UINT : nvrhi::Format::R16_UINT;
            ib.initialState = nvrhi::ResourceStates::ShaderResource;
            ib.keepInitialState = true;
            ib.isIndexBuffer = true;
            _indexBuffer = m_device->createBuffer(ib);
        }
        else
        {
            _indexBuffer = nullptr;
        }

        {
            _indexCount = input.indexCount;

            uint32_t maxTexCoordIndex = 0;
            for (uint32_t i = 0; i < input.indexCount; ++i)
            {
                if (input.indexFormat == omm::IndexFormat::UINT_16)
                {
                    uint16_t val = ((uint16_t*)input.indexBuffer)[i];
                    maxTexCoordIndex = std::max<uint32_t>(maxTexCoordIndex, val);
                }
                else
                {
                    assert(input.indexFormat == omm::IndexFormat::UINT_32);
                    uint32_t val = ((uint32_t*)input.indexBuffer)[i];
                    maxTexCoordIndex = std::max(maxTexCoordIndex, val);
                }
            }

            const size_t texCoordBufferSize = (maxTexCoordIndex + 1) * (input.texCoordFormat == omm::TexCoordFormat::UV32_FLOAT ? 4 : 2);

            if (texCoordBufferSize != 0)
            {
                assert(input.texCoordFormat == omm::TexCoordFormat::UV32_FLOAT);

                nvrhi::BufferDesc texCoord;
                texCoord.debugName = "TexCoordBuffer";
                texCoord.byteSize = texCoordBufferSize * 2;
                texCoord.format = input.texCoordFormat == omm::TexCoordFormat::UV32_FLOAT ? nvrhi::Format::RG32_FLOAT : nvrhi::Format::RG16_FLOAT;
                texCoord.initialState = nvrhi::ResourceStates::ShaderResource;
                texCoord.keepInitialState = true;
                texCoord.isVertexBuffer = true;
                _texCoordBuffer = m_device->createBuffer(texCoord);
            }
            else
            {
                _texCoordBuffer = nullptr;
            }
        }

        m_commandList->open();
        m_commandList->setEnableAutomaticBarriers(true);
        if (_indexBuffer)
            m_commandList->writeBuffer(_indexBuffer, input.indexBuffer, _indexBuffer->getDesc().byteSize);
        if (_texCoordBuffer)
            m_commandList->writeBuffer(_texCoordBuffer, input.texCoords, _texCoordBuffer->getDesc().byteSize);
        m_commandList->close();
        m_device->executeCommandList(m_commandList);
        m_device->waitForIdle();
    }

    void _InitBakeResults(const omm::Cpu::BakeInputDesc& input)
    {
        {
            nvrhi::BufferDesc ommIB;
            ommIB.debugName = "OmmIndexBuffer";
            if (_resultDesc && _resultDesc->indexCount != 0)
            {
                ommIB.format = _resultDesc->indexFormat == omm::IndexFormat::UINT_32 ? nvrhi::Format::R32_SINT : nvrhi::Format::R16_SINT;
                ommIB.byteSize = _resultDesc->indexCount * (_resultDesc->indexFormat == omm::IndexFormat::UINT_32 ? 4 : 2);
            }
            else
            {
                ommIB.format = nvrhi::Format::R32_SINT;
                ommIB.byteSize = 8;
            }

            ommIB.initialState = nvrhi::ResourceStates::ShaderResource;
            ommIB.keepInitialState = true;
            ommIB.canHaveTypedViews = true;
            _ommIndexBuffer = m_device->createBuffer(ommIB);
        }

        {
            nvrhi::BufferDesc ommDesc;
            ommDesc.debugName = "OmmDescBuffer";
            if (_resultDesc && _resultDesc->descArrayCount != 0)
                ommDesc.byteSize = _resultDesc->descArrayCount * sizeof(omm::Cpu::OpacityMicromapDesc);
            else
                ommDesc.byteSize = 8;
            ommDesc.format = nvrhi::Format::UNKNOWN;
            ommDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            ommDesc.structStride = sizeof(omm::Cpu::OpacityMicromapDesc);
            ommDesc.keepInitialState = true;
            _ommDesc = m_device->createBuffer(ommDesc);
        }

        {
            nvrhi::BufferDesc ommArray;
            ommArray.debugName = "OmmArrayBuffer";
            if (_resultDesc && _resultDesc->arrayDataSize != 0)
                ommArray.byteSize = _resultDesc->arrayDataSize;
            else
                ommArray.byteSize = 8;
            //ommArray.format = nvrhi::Format::R32_UINT;
            ommArray.initialState = nvrhi::ResourceStates::ShaderResource;
            ommArray.keepInitialState = true;
            ommArray.canHaveRawViews = true;
            _ommArrayData = m_device->createBuffer(ommArray);
        }

        m_commandList->open();
        m_commandList->setEnableAutomaticBarriers(true);
        if (_resultDesc && _resultDesc->indexCount != 0)
            m_commandList->writeBuffer(_ommIndexBuffer, _resultDesc->indexBuffer, _ommIndexBuffer->getDesc().byteSize);
        if (_resultDesc && _resultDesc->descArrayCount != 0)
            m_commandList->writeBuffer(_ommDesc, _resultDesc->descArray, _ommDesc->getDesc().byteSize);
        if (_resultDesc && _resultDesc->arrayDataSize != 0)
            m_commandList->writeBuffer(_ommArrayData, _resultDesc->arrayData, _ommArrayData->getDesc().byteSize);
        m_commandList->close();
        m_device->executeCommandList(m_commandList);
        m_device->waitForIdle();
    }

    void _LoadOmmData(const std::string& fileName)
    {
        if (_fileName != fileName)
        {
            m_ui.primitiveEnd = -1;

            m_ui.input.reset();
            m_ui.textureDesc.reset();
            _data = _LoadDataFile(fileName);
        }
    }

    void _RebuildOmmData(bool loadOnly)
    {
        omm::Cpu::BlobDesc blobDesc;
        blobDesc.data = _data.data();
        blobDesc.size = _data.size();

        omm::Baker baker = _lib.GetBaker();

        omm::Cpu::DeserializedResult res;
        OMM_ABORT_ON_ERROR(omm::Cpu::Deserialize(baker, blobDesc, &res));

        const omm::Cpu::DeserializedDesc* deserializeDesc = nullptr;
        OMM_ABORT_ON_ERROR(omm::Cpu::GetDeserializedDesc(res, &deserializeDesc));

        assert(deserializeDesc->numInputDescs > 0);

        nvrhi::TextureHandle tex;
        
        omm::Cpu::BakeInputDesc input = deserializeDesc->inputDescs[0];
        _input = input;

        if (!m_ui.input.has_value())
        {
            m_ui.input = input;
            reinterpret_cast<uint32_t&>(m_ui.input->bakeFlags) |= uint32_t(omm::Cpu::BakeFlags::EnableInternalThreads);
            m_ui.input->maxWorkloadSize = 0xFFFFFFFFFFFFFFFF;
        }

        input = m_ui.input.value();
            
        input.texture = _input.texture;

        input.texCoords = _input.texCoords;
        input.texCoordStrideInBytes = _input.texCoordStrideInBytes;
        input.texCoordFormat = _input.texCoordFormat;
            
        input.indexFormat = _input.indexFormat;
        input.indexBuffer = _input.indexBuffer;
        input.indexCount = _input.indexCount;

        input.subdivisionLevels = _input.subdivisionLevels;
        
        omm::Cpu::TextureMipDesc mips[16];
        omm::Cpu::TextureDesc texDesc;
        texDesc.mips = mips;

        OMM_ABORT_ON_ERROR(omm::Cpu::FillTextureDesc(input.texture, &texDesc));
        std::vector<uint8_t> textureData(texDesc.mips[0].width * texDesc.mips[0].height);

        mips[0].textureData = (const void*)textureData.data();

        OMM_ABORT_ON_ERROR(omm::Cpu::FillTextureDesc(input.texture, &texDesc));

        if (!m_ui.textureDesc.has_value())
        {
            m_ui.textureDesc = texDesc;
        }

        _InitBuffers(input);
        _InitSampler(input);
        _InitTexture(texDesc);

        if (loadOnly)
        {
            _InitBakeResults(input);
            OMM_ABORT_ON_ERROR(omm::Cpu::DestroyDeserializedResult(res));
            return;
        }

        {
            texDesc = m_ui.textureDesc.value();
            texDesc.mips = mips;

            omm::Cpu::Texture textureClone;
            OMM_ABORT_ON_ERROR(omm::Cpu::CreateTexture(baker, texDesc, &textureClone));

            input.texture = textureClone;

            auto start = std::chrono::high_resolution_clock::now();
            omm::Result res = omm::Cpu::Bake(baker, input, &_result);
            OMM_POPUP_ON_ERROR(res);
            auto end = std::chrono::high_resolution_clock::now();
            _bakeTimeInMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            _bakeTimeInSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

            if (res == omm::Result::SUCCESS)
            {
                OMM_ABORT_ON_ERROR(omm::Cpu::GetBakeResultDesc(_result, &_resultDesc));

                OMM_ABORT_ON_ERROR(omm::Debug::GetStats(baker, _resultDesc, &_stats));
            }

            OMM_ABORT_ON_ERROR(omm::Cpu::DestroyTexture(baker, textureClone));
        }

        _InitBakeResults(input);

        OMM_ABORT_ON_ERROR(omm::Cpu::DestroyDeserializedResult(res));
    }
};

class BasicTriangle : public app::IRenderPass
{
private:
    nvrhi::BufferHandle m_ConstantBuffer;
    nvrhi::ShaderHandle m_VertexShader;
    nvrhi::ShaderHandle m_PixelShader;

    nvrhi::ShaderHandle m_BackgroundVS;
    nvrhi::ShaderHandle m_BackgroundPS;
    nvrhi::GraphicsPipelineHandle m_BackgroundPSO;

    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSets;
    nvrhi::GraphicsPipelineHandle m_Pipeline;
    nvrhi::GraphicsPipelineHandle m_PipelineWireFrame;

    nvrhi::SamplerHandle m_LinearSampler;
    nvrhi::InputLayoutHandle m_InputLayout;
    nvrhi::CommandListHandle m_CommandList;
    std::vector<std::filesystem::path> m_ommFiles;
    OmmGpuData m_ommData;
    UIData& m_ui;
    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;

public:
    using IRenderPass::IRenderPass;

    BasicTriangle(app::DeviceManager* deviceManager, UIData& ui):IRenderPass(deviceManager),m_ui(ui), m_ommData(ui){}

    std::shared_ptr<engine::ShaderFactory> GetShaderFactory()
    {
        return m_ShaderFactory;
    }

    const std::vector<std::filesystem::path>& GetOmmFiles() const
    {
        return m_ommFiles;
    }

    const OmmGpuData& GetOmmGpuData() const
    {
        return m_ommData;
    }

    bool Init()
    {
        std::string path = "E:\\git\\Opacity-MicroMap-SDK\\util\\viewer_app\\feature_demo\\data\\";
        for (const auto& entry : std::filesystem::directory_iterator(path))
        {
            m_ommFiles.push_back(entry.path());
        }

        std::sort(m_ommFiles.begin(), m_ommFiles.end(), [](const std::filesystem::path& a, const std::filesystem::path& b) {
            return a.filename().string() < b.filename().string();
        });

        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/basic_triangle" /  app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        //auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        auto rootFs = std::make_shared<vfs::RootFileSystem>();
        rootFs->mount("basic_triangle", appShaderPath);
        rootFs->mount("donut", frameworkShaderPath);

        m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFs, "");

        m_CommandList = GetDevice()->createCommandList();

        m_ommData.Init(GetDevice());
        m_ommData.Load(m_ommFiles[0].string());
        m_ommData.Bake();
    
        return true;
    }

    void ClearAllResource()
    {
        m_Pipeline = nullptr;
        m_BindingLayout = nullptr;
        m_BindingSets = nullptr;
        m_BackgroundPSO = nullptr;
        m_VertexShader = nullptr;
        m_PixelShader = nullptr;
        m_BackgroundVS = nullptr;
        m_BackgroundPS = nullptr;

        m_ShaderFactory->ClearCache();
    }

    void BackBufferResizing() override
    { 
        ClearAllResource();
        m_ui.recompile = false;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }
    
    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        if (m_ui.rebake || m_ui.load || m_ui.recompile)
        {
            GetDevice()->waitForIdle();

            if (m_ui.load)
            {
                m_ommData.Load(m_ommFiles[m_ui.selectedFile].string());
            }

            if (m_ui.rebake)
            {
                m_ommData.Bake();
            }
            
            ClearAllResource();
            m_ui.load = false;
            m_ui.rebake = false;
            m_ui.recompile = false;
        }

        if (!m_Pipeline)
        {
            m_VertexShader = m_ShaderFactory->CreateShader("basic_triangle/shaders.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_PixelShader = m_ShaderFactory->CreateShader("basic_triangle/shaders.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

            m_BackgroundVS = m_ShaderFactory->CreateShader("basic_triangle/background_vs_ps.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
            m_BackgroundPS = m_ShaderFactory->CreateShader("basic_triangle/background_vs_ps.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

            m_ConstantBuffer = GetDevice()->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(sizeof(Constants), "Constants", 16));

            nvrhi::VertexAttributeDesc vertexAttr;
            vertexAttr.name = "SV_POSITION";
            vertexAttr.format = nvrhi::Format::RG32_FLOAT;
            vertexAttr.elementStride = sizeof(float) * 2;
            m_InputLayout = GetDevice()->createInputLayout(&vertexAttr, 1, m_VertexShader);

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer),
                nvrhi::BindingSetItem::Sampler(0, m_ommData.GetSampler()),
                nvrhi::BindingSetItem::Texture_SRV(0, m_ommData.GetAlphaTexture()),
                nvrhi::BindingSetItem::TypedBuffer_SRV(1, m_ommData.GetOmmIndexBuffer()),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_ommData.GetOmmDesc()),
                nvrhi::BindingSetItem::RawBuffer_SRV(3, m_ommData.GetOmmArrayData()),
            };

            // Create the binding layout (if it's empty -- so, on the first iteration) and the binding set.
            if (!nvrhi::utils::CreateBindingSetAndLayout(GetDevice(), nvrhi::ShaderType::All, 0, bindingSetDesc, m_BindingLayout, m_BindingSets))
            {
                log::error("Couldn't create the binding set or layout");
            }

            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_VertexShader;
            psoDesc.PS = m_PixelShader;
            psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
            psoDesc.renderState.depthStencilState.depthTestEnable = false;
            psoDesc.bindingLayouts = { m_BindingLayout };
            psoDesc.inputLayout = m_InputLayout;
            psoDesc.renderState.rasterState.setFrontCounterClockwise(false);
            psoDesc.renderState.rasterState.fillMode = nvrhi::RasterFillMode::Wireframe;
            psoDesc.renderState.rasterState.setCullNone();

            // nvrhi::BlendState::RenderTarget blendState;
            // blendState.setBlendEnable(true);
            // blendState.setSrcBlend(nvrhi::BlendFactor::SrcAlpha);
            // blendState.setDestBlend(nvrhi::BlendFactor::InvSrcAlpha);

           // psoDesc.renderState.blendState.setRenderTarget(0, blendState);
            
            m_PipelineWireFrame = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
            
            psoDesc.renderState.rasterState.fillMode = nvrhi::RasterFillMode::Fill;
            m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
        }

        if (!m_BackgroundPSO)
        {
            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_BackgroundVS;
            psoDesc.PS = m_BackgroundPS;
            psoDesc.primType = nvrhi::PrimitiveType::TriangleStrip;
            psoDesc.renderState.depthStencilState.depthTestEnable = false;
            psoDesc.bindingLayouts = { m_BindingLayout };
           // psoDesc.inputLayout = m_InputLayout;
           // psoDesc.renderState.rasterState.setFrontCounterClockwise(true);
            psoDesc.renderState.rasterState.setCullNone();

            m_BackgroundPSO = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
        }

        m_CommandList->open();

        Constants constants;
        constants.texSize = donut::math::uint2(m_ommData.GetAlphaTexture()->getDesc().width, m_ommData.GetAlphaTexture()->getDesc().height);
        constants.invTexSize = float2(1.f / constants.texSize.x, 1.f / constants.texSize.y);
        constants.zoom = m_ui.zoom;
        constants.offset = m_ui.offset + m_ui.prevOffset;
        constants.primitiveOffset = m_ui.primitiveStart;
        constants.mode = 0;

        m_CommandList->writeBuffer(m_ConstantBuffer, &constants, sizeof(constants));

        nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));

        int sizePerIndex = m_ommData.GetIndexBuffer()->getDesc().format == nvrhi::Format::R32_UINT ? 4 : 2;

        nvrhi::IndexBufferBinding indexBinding;
        indexBinding.buffer = m_ommData.GetIndexBuffer();
        indexBinding.format = m_ommData.GetIndexBuffer()->getDesc().format;
        indexBinding.offset = 3 * m_ui.primitiveStart * sizePerIndex;

        nvrhi::VertexBufferBinding vertexBinding;
        vertexBinding.buffer = m_ommData.GetTexCoordBuffer();
        vertexBinding.slot = 0;
        vertexBinding.offset = 0;

        {
            nvrhi::GraphicsState state;
            state.pipeline = m_BackgroundPSO;
            state.framebuffer = framebuffer;
            state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
            state.bindings = { m_BindingSets };
            m_CommandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.vertexCount = 4;
            m_CommandList->draw(args);
        }

        {
            nvrhi::GraphicsState state;
            state.pipeline = m_Pipeline;
            state.framebuffer = framebuffer;
            state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
            state.bindings = { m_BindingSets };
            state.setIndexBuffer(indexBinding);
            state.addVertexBuffer(vertexBinding);
            m_CommandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.vertexCount = 3 * std::max(0, (m_ui.primitiveEnd - m_ui.primitiveStart));
            m_CommandList->drawIndexed(args);
        }

        constants.mode = 1;
        m_CommandList->writeBuffer(m_ConstantBuffer, &constants, sizeof(constants));

        {
            nvrhi::GraphicsState state;
            state.pipeline = m_PipelineWireFrame;
            state.framebuffer = framebuffer;
            state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
            state.bindings = { m_BindingSets };
            state.setIndexBuffer(indexBinding);
            state.addVertexBuffer(vertexBinding);
            m_CommandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.vertexCount = 3 * std::max(0, (m_ui.primitiveEnd - m_ui.primitiveStart));
            m_CommandList->drawIndexed(args);
        }

        

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
    }
};

template<class T>
void ImGui_CheckBoxFlag(const char* name, uint32_t ID, T& flags, T origFlags, T mask)
{
    bool value = (mask & flags) == mask;
    bool origValue = (mask & origFlags) == mask;
    
    ImGui::BeginDisabled(origValue == value);

    ImGui::PushID(ID);
    if (ImGui::Button("Reset"))
    {
        flags &= ~mask;
        flags |= origFlags & mask;
    }
    ImGui::PopID();

    ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Checkbox(name, &value))
    {
        if (value)
        {
            flags |= mask;
        }
        else
        {
            flags &= ~mask;
        }
    }
}

template<class T>
void ImGui_SliderInt(const char* name, uint32_t ID, T& value, T origValue, int min, int max)
{
    ImGui::BeginDisabled(origValue == value);

    ImGui::PushID(ID);
    if (ImGui::Button("Reset"))
    {
        value = origValue;
    }
    ImGui::PopID();

    ImGui::EndDisabled();

    ImGui::SameLine();

    int intVal = value;
    ImGui::SliderInt(name, &intVal, min, max);
    value = intVal;
}

void ImGui_SliderFloat(const char* name, uint32_t ID, float& value, float origValue, float min, float max)
{
    ImGui::BeginDisabled(origValue == value);

    ImGui::PushID(ID);
    if (ImGui::Button("Reset"))
    {
        value = origValue;
    }
    ImGui::PopID();

    ImGui::EndDisabled();

    ImGui::SameLine();

    ImGui::SliderFloat(name, &value, min, max);
}

void ImGui_ValueUInt64(const char* name, uint32_t ID, uint64_t& value, uint64_t origValue)
{
    ImGui::BeginDisabled(origValue == value);

    ImGui::PushID(ID);
    if (ImGui::Button("Reset"))
    {
        value = origValue;
    }
    ImGui::PopID();

    ImGui::EndDisabled();

    ImGui::SameLine();

    ImGuiInputTextFlags flags = ImGuiInputTextFlags_CharsHexadecimal;
    const char* format = (flags & ImGuiInputTextFlags_CharsHexadecimal) ? "%llX" : "%llu";
    int step = 1;
    int step_fast = 100;
    ImGui::InputScalar(name, ImGuiDataType_U64, (void*)&value, (void*)(step > 0 ? &step : NULL), (void*)(step_fast > 0 ? &step_fast : NULL), format, flags);
}

template<int N>
void ImGui_Combo(const char* name, uint32_t ID, std::array<const char*, N> items, omm::TextureAddressMode& value, omm::TextureAddressMode origValue)
{
    ImGui::BeginDisabled(origValue == value);

    ImGui::PushID(ID);
    if (ImGui::Button("Reset"))
    {
        value = origValue;
    }
    ImGui::PopID();

    ImGui::EndDisabled();

    ImGui::SameLine();

    ImGui::Combo(name, reinterpret_cast<int*>(&value), items.data(), (int)items.size());
}

class UIRenderer : public donut::app::ImGui_Renderer
{
private:
    std::unique_ptr<donut::app::ImGui_Console> m_console;
    UIData& m_ui;
    std::shared_ptr<engine::ShaderFactory> m_ShaderFactory;
    std::shared_ptr<BasicTriangle> m_app;
    bool m_mouseDown = false;
    float2 m_mousePos = float2(0, 0);
    float2 m_referencePos = float2(0, 0);
    int2 m_windowSize = 0;
public:
    UIRenderer(donut::app::DeviceManager* deviceManager, std::shared_ptr<BasicTriangle> app, UIData& ui)
        : ImGui_Renderer(deviceManager)
        , m_app(app)
        , m_ui(ui)
    {
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        auto rootFs = std::make_shared<vfs::RootFileSystem>();
        rootFs->mount("/shaders/donut", frameworkShaderPath);

        m_ShaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFs, "/shaders");
        ImGui::GetIO().IniFilename = nullptr;

        UpdateWindowSize();
    }

    void Init()
    {
        donut::app::ImGui_Renderer::Init(m_ShaderFactory);
    }

protected:
    virtual bool MouseButtonUpdate(int button, int action, int mods)
    {
        if (donut::app::ImGui_Renderer::MouseButtonUpdate(button, action, mods))
            return true;

        if (button == 0)
        {
            if (action == 1)
            {
                m_mouseDown = true;
                m_referencePos = m_mousePos;
            }
            else
            {
                m_ui.prevOffset += m_ui.offset;
                m_ui.offset = float2(0,0);
                m_mouseDown = false;
            }
        }
        return false;
    }
    virtual bool MousePosUpdate(double xpos, double ypos)
    {
        if (donut::app::ImGui_Renderer::MousePosUpdate(xpos, ypos))
            return true;

        float scaleX, scaleY;
        GetDeviceManager()->GetDPIScaleInfo(scaleX, scaleY);

        xpos *= scaleX;
        ypos *= scaleY;

        m_mousePos = float2((float)xpos, (float)ypos);
        if (m_mouseDown)
        {
            m_ui.offset = (2.f / (float2)m_windowSize) * (m_referencePos - m_mousePos) / m_ui.zoom;
            m_ui.offset.x = -m_ui.offset.x;
        }
        return false;
    }

    virtual bool MouseScrollUpdate(double xoffset, double yoffset)
    {
        if (donut::app::ImGui_Renderer::MouseScrollUpdate(xoffset, yoffset))
            return true;
        m_ui.zoom += 0.1f * m_ui.zoom * (float)yoffset;

        return false;
    }

protected:

    void UpdateWindowSize()
    {
        GetDeviceManager()->GetWindowDimensions(m_windowSize.x, m_windowSize.y);
    }

    virtual void buildUI(void) override
    {
        if (!m_ui.ShowUI)
            return;

        const auto& io = ImGui::GetIO();

        UpdateWindowSize();

        ImGui::SetNextWindowPos(ImVec2(10.f, 10.f), 0);
        ImGui::Begin("Settings", 0, ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Renderer: %s", GetDeviceManager()->GetRendererString());
        double frameTime = GetDeviceManager()->GetAverageFrameTimeSeconds();
        if (frameTime > 0.0)
            ImGui::Text("%.3f ms/frame (%.1f FPS)", frameTime * 1e3, 1.0 / frameTime);

        int maxPrimitiveCount = m_app->GetOmmGpuData().GetIndexCount() / 3;


        auto& files = m_app->GetOmmFiles();
        auto selected = files[m_ui.selectedFile].filename().string();

        if (ImGui::BeginCombo("Data", selected.c_str())) // Pass in the label and the current item
        {
            for (int i = 0; i < files.size(); i++)
            {
                auto file = files[i].filename().string();

                bool is_selected = (m_ui.selectedFile == i);
                if (ImGui::Selectable(file.c_str(), is_selected))
                {
                    m_ui.selectedFile = i;
                    m_ui.load = true;
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (m_ui.primitiveEnd == -1)
        {
            m_ui.primitiveEnd = maxPrimitiveCount;// std::min(128, maxPrimitiveCount);
        }

        if (ImGui::SliderInt("Primitive Start", &m_ui.primitiveStart, 0, maxPrimitiveCount - 1, "%d"))
        {
            if (m_ui.primitiveStart >= m_ui.primitiveEnd)
                m_ui.primitiveEnd = m_ui.primitiveStart + 1;
        }

        if (ImGui::SliderInt("Primitive End", &m_ui.primitiveEnd, 1, maxPrimitiveCount, "%d"))
        {
            if (m_ui.primitiveStart >= m_ui.primitiveEnd)
                m_ui.primitiveStart = m_ui.primitiveEnd - 1;
        }

        const omm::Cpu::TextureDesc& texDesc = m_app->GetOmmGpuData().GetDefaultTextureDesc();
        const omm::Cpu::BakeInputDesc& input = m_app->GetOmmGpuData().GetDefaultInput();

        uint32_t id = 0;

        ImGui::SeparatorText("Texture Settings");

        uint width = m_app->GetOmmGpuData().GetAlphaTexture()->getDesc().width;
        uint height = m_app->GetOmmGpuData().GetAlphaTexture()->getDesc().height;
        nvrhi::Format format = m_app->GetOmmGpuData().GetAlphaTexture()->getDesc().format;

        const char* formatstr = "Format unknown";
        if (format == nvrhi::Format::R32_FLOAT)
            formatstr = "Float 32";
        else if (format == nvrhi::Format::R8_UNORM)
            formatstr = "Unorm 8";

        ImGui::Text("Alpha Texture %dx%d,%s", width, height, formatstr);

        ImGui_Combo<5>("Addressing Mode", id++, { "Wrap", "Mirror", "Clamp", "Border", "MirrorOnce" }, m_ui.input->runtimeSamplerDesc.addressingMode, input.runtimeSamplerDesc.addressingMode);

        ImGui_CheckBoxFlag<omm::Cpu::TextureFlags>("Disable Z Order", id++, m_ui.textureDesc->flags, texDesc.flags, omm::Cpu::TextureFlags::DisableZOrder);

        {
            ImGui::BeginDisabled(texDesc.alphaCutoff == m_ui.textureDesc->alphaCutoff);

            ImGui::PushID(id++);
            if (ImGui::Button("Reset"))
            {
                m_ui.textureDesc->alphaCutoff = texDesc.alphaCutoff;
            }
            ImGui::PopID();

            ImGui::EndDisabled();

            ImGui::SameLine();

            bool value = m_ui.textureDesc->alphaCutoff >= 0.f;
            if (ImGui::Checkbox("Enable SAT acceleration", &value))
            {
                if (value)
                {
                    m_ui.textureDesc->alphaCutoff = input.alphaCutoff;
                }
                else
                {
                    m_ui.textureDesc->alphaCutoff = -1.f;
                }
            }
        }

        ImGui::SeparatorText("Bake Settings");

        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Enable Internal Threads", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::EnableInternalThreads);
        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Disable Special Indices", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::DisableSpecialIndices);
        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Force 32 Bit Indices", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::Force32BitIndices);
        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Disable Duplicate Detection", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::DisableDuplicateDetection);
        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Enable Near-Duplicate Detection", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::EnableNearDuplicateDetection);
        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Enable Validation", id++, m_ui.input->bakeFlags, input.bakeFlags, omm::Cpu::BakeFlags::EnableValidation);

        ImGui_SliderInt<uint8_t>("Max Subdivision Level", id++, m_ui.input->maxSubdivisionLevel, input.maxSubdivisionLevel, 0, 12);
        ImGui_SliderFloat("Dynamic Subdivision Scale", id++, m_ui.input->dynamicSubdivisionScale, input.dynamicSubdivisionScale, 0.f, 10.f);
        ImGui_SliderFloat("Rejection Threshold", id++, m_ui.input->rejectionThreshold, input.rejectionThreshold, 0.f, 1.f);
        
        ImGui_ValueUInt64("Max Workload Size", id++, m_ui.input->maxWorkloadSize, input.maxWorkloadSize);
        ImGui::SeparatorText("Unofficial Bake Settings");

        constexpr uint32_t kEnableAABBTesting = 1u << 6u;
        constexpr uint32_t kDisableLevelLineIntersection = 1u << 7u;
        constexpr uint32_t kDisableFineClassification = 1u << 8u;
        constexpr uint32_t kEnableNearDuplicateDetectionBruteForce = 1u << 9u;
        constexpr uint32_t kEdgeHeuristic = 1u << 10u;
        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Enable AABB Testing", id++, m_ui.input->bakeFlags, input.bakeFlags, (omm::Cpu::BakeFlags)kEnableAABBTesting);
        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Disable Level Line Intersection", id++, m_ui.input->bakeFlags, input.bakeFlags, (omm::Cpu::BakeFlags)kDisableLevelLineIntersection);
        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Disable Fine Classification", id++, m_ui.input->bakeFlags, input.bakeFlags, (omm::Cpu::BakeFlags)kDisableFineClassification);
        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Enable Near-Duplicate Detection Brute-Force", id++, m_ui.input->bakeFlags, input.bakeFlags, (omm::Cpu::BakeFlags)kEnableNearDuplicateDetectionBruteForce);
        ImGui_CheckBoxFlag<omm::Cpu::BakeFlags>("Edge Heuristic", id++, m_ui.input->bakeFlags, input.bakeFlags, (omm::Cpu::BakeFlags)kEdgeHeuristic);

        ImGui::Separator();

        if (ImGui::Button("Rebake"))
        {
            m_ui.rebake = true;
        }

        ImGui::SameLine();
        ImGui::Text("Last bake time %llus, (%llu ms)", m_app->GetOmmGpuData().GetBakeTimeInSeconds(), m_app->GetOmmGpuData().GetBakeTimeInMs());

        ImGui::SeparatorText("Memory");

        if (const omm::Cpu::BakeResultDesc* result = m_app->GetOmmGpuData().GetResult())
        {
            size_t arrayDataSize = result->arrayDataSize;
            size_t indexSize = result->indexCount * (result->indexFormat == omm::IndexFormat::UINT_16 ? 2 : 4);
            size_t descArraySize = result->descArrayCount * sizeof(omm::Cpu::OpacityMicromapDesc);
            size_t totalSize = arrayDataSize + indexSize + descArraySize;
            ImGui::Text("Array Data Size %.4f mb", arrayDataSize / (1024.f * 1024.f));
            ImGui::Text("Index Data Size %.4f mb", indexSize / (1024.f * 1024.f));
            ImGui::Text("Desc Array Size %.4f mb", descArraySize / (1024.f * 1024.f));
            ImGui::Text("Total Size %.4f mb", totalSize / (1024.f * 1024.f));
        }

        ImGui::SeparatorText("Stats");

        omm::Debug::Stats stats = m_app->GetOmmGpuData().GetStats();
        const float known = (float)stats.totalOpaque + stats.totalTransparent;
        const float unknown = (float)stats.totalUnknownTransparent + stats.totalUnknownOpaque;

        ImGui::Text("Known %.2f%%", 100.f * known / (known + unknown));
        ImGui::Text("Total Opaque %llu", stats.totalOpaque);
        ImGui::Text("Total Transparent %llu", stats.totalTransparent);
        ImGui::Text("Total Unknown Transparent %llu", stats.totalUnknownTransparent);
        ImGui::Text("Total Unknown Opaque %llu", stats.totalUnknownOpaque);

        ImGui::Text("Total Fully Opaque %llu", stats.totalFullyOpaque);
        ImGui::Text("Total Fully Transparent %llu", stats.totalFullyTransparent);
        ImGui::Text("Total Fully Unknown Transparent %llu", stats.totalFullyUnknownTransparent);
        ImGui::Text("Total Fully Unknown Opaque %llu", stats.totalFullyUnknownOpaque);

        if (ImGui::CollapsingHeader("Development")) {
            if (ImGui::Button("Recompile Shaders"))
            {
                m_ui.recompile = true;
            }
        }

        ImGui::End();
    }
};

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true; 
    deviceParams.enableNvrhiValidationLayer = true;
#endif
    deviceParams.enablePerMonitorDPI = true;
    deviceParams.backBufferWidth = 2 * 1280;
    deviceParams.backBufferHeight = 2 * 720;

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    
    {
        UIData ui;
        std::shared_ptr<BasicTriangle> example = std::make_shared<BasicTriangle>(deviceManager, ui);
        example->Init();
        std::shared_ptr<UIRenderer> gui = std::make_shared<UIRenderer>(deviceManager, example, ui);
        gui->Init();
        //if (example->Init())
        {
            deviceManager->AddRenderPassToBack(example.get());
            deviceManager->AddRenderPassToBack(gui.get());
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(example.get());
            deviceManager->RemoveRenderPass(gui.get());
        }
    }
    
    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
