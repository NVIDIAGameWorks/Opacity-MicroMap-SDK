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
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>

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

static void Log(omm::MessageSeverity severity, const char* message, void* userArg)
{
    donut::log::message(donut::log::Severity::Info, "[omm-sdk]: %s", message);
}

#define OMM_ABORT_ON_ERROR(fun) do { \
    omm::Result res_##__LINE__ = fun; \
    if (res_##__LINE__ != omm::Result::SUCCESS) \
    { \
        AbortOnFailure(#fun, res_##__LINE__); \
    } \
    } while (false)

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

class BasicTriangle : public app::IRenderPass
{
private:
    nvrhi::ShaderHandle m_VertexShader;
    nvrhi::ShaderHandle m_PixelShader;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSets;
    nvrhi::GraphicsPipelineHandle m_Pipeline;
    nvrhi::SamplerHandle m_LinearSampler;
    nvrhi::CommandListHandle m_CommandList;
    nvrhi::TextureHandle m_Texture;
    OmmLibrary m_ommLib;

public:
    using IRenderPass::IRenderPass;

    std::vector<uint8_t> LoadDataFile(const std::string& fileName)
    {
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

    nvrhi::TextureHandle CreateTexture(const omm::Cpu::TextureDesc& ommTex)
    {
        nvrhi::TextureDesc d;
        d.height = ommTex.mips[0].height;
        d.width = ommTex.mips[0].width;
        d.format = ommTex.format == omm::Cpu::TextureFormat::FP32 ? nvrhi::Format::R32_FLOAT : nvrhi::Format::R8_UNORM;
        d.initialState = nvrhi::ResourceStates::ShaderResource;
        d.keepInitialState = true;
        d.debugName = "AlphaTexture";
        nvrhi::TextureHandle tex = GetDevice()->createTexture(d);

        size_t texelSize = ommTex.format == omm::Cpu::TextureFormat::FP32 ? sizeof(float) : sizeof(uint8_t);

        m_CommandList->open();
        m_CommandList->setEnableAutomaticBarriers(true);
       // memset((void*)ommTex.mips[0].textureData, 0xFF, texelSize * ommTex.mips[0].rowPitch * ommTex.mips[0].height);
        m_CommandList->writeTexture(tex, 0, 0, ommTex.mips[0].textureData, texelSize * ommTex.mips[0].rowPitch);

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
        GetDevice()->waitForIdle();

        return tex;
    }

    nvrhi::TextureHandle LoadOmmData(const std::string& fileName)
    {
        std::vector<uint8_t> data = LoadDataFile(fileName);

        omm::Cpu::BlobDesc blobDesc;
        blobDesc.data = data.data();
        blobDesc.size = data.size();

        omm::Baker baker = m_ommLib.GetBaker();

        omm::Cpu::DeserializedResult res;
        OMM_ABORT_ON_ERROR(omm::Cpu::Deserialize(baker, blobDesc, &res));

        const omm::Cpu::DeserializedDesc* deserializeDesc = nullptr;
        OMM_ABORT_ON_ERROR(omm::Cpu::GetDeserializedDesc(res, &deserializeDesc));

        nvrhi::TextureHandle tex;
        if (deserializeDesc->numInputDescs > 0)
        {
            omm::Cpu::BakeInputDesc input = deserializeDesc->inputDescs[0];
            reinterpret_cast<uint32_t&>(input.bakeFlags) |= uint32_t(omm::Cpu::BakeFlags::EnableInternalThreads);
            input.maxWorkloadSize = 0xFFFFFFFFFFFFFFFF;
            omm::Cpu::TextureMipDesc mips[16];
            omm::Cpu::TextureDesc texDesc;
            texDesc.mips = mips;

            OMM_ABORT_ON_ERROR(omm::Cpu::FillTextureDesc(input.texture, &texDesc));
            std::vector<uint8_t> textureData(texDesc.mips[0].width * texDesc.mips[0].height);

            mips[0].textureData = (const void*)textureData.data();

            OMM_ABORT_ON_ERROR(omm::Cpu::FillTextureDesc(input.texture, &texDesc));

#if 0
            omm::Cpu::Texture textureClone;
            OMM_ABORT_ON_ERROR(omm::Cpu::CreateTexture(baker, texDesc, &textureClone));

            input.texture = textureClone;

            omm::Cpu::BakeResult bakeResult;
            OMM_ABORT_ON_ERROR(omm::Cpu::Bake(baker, input, &bakeResult));

            const omm::Cpu::BakeResultDesc* resultDesc;
            OMM_ABORT_ON_ERROR(omm::Cpu::GetBakeResultDesc(bakeResult, &resultDesc));

            omm::Debug::SaveImagesDesc debugDesc;
            debugDesc.path = "OmmBakeOutput";
            debugDesc.filePostfix = "_SearchForMe";
            OMM_ABORT_ON_ERROR(omm::Debug::SaveAsImages(baker, input, resultDesc, debugDesc));

            OMM_ABORT_ON_ERROR(omm::Cpu::DestroyTexture(baker, textureClone));
#endif

            tex = CreateTexture(texDesc);
        }

        OMM_ABORT_ON_ERROR(omm::Cpu::DestroyDeserializedResult(res));

        return tex;
    }

    bool Init()
    {
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/basic_triangle" /  app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        engine::ShaderFactory shaderFactory(GetDevice(), nativeFS, appShaderPath);

        m_VertexShader = shaderFactory.CreateShader("shaders.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
        m_PixelShader = shaderFactory.CreateShader("shaders.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

        if (!m_VertexShader || !m_PixelShader)
        {
            return false;
        }
        
        m_CommandList = GetDevice()->createCommandList();

        m_Texture = LoadOmmData("E:\\git\\Opacity-MicroMap-SDK\\bin\\myExpensiveBakeJob.bin");

        return true;
    }

    void BackBufferResizing() override
    { 
        m_Pipeline = nullptr;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }
    
    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        if (!m_Pipeline)
        {
            auto samplerDesc = nvrhi::SamplerDesc()
                .setAllFilters(false)
                .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
            samplerDesc.setAllFilters(true);
            m_LinearSampler = GetDevice()->createSampler(samplerDesc);

            struct ConstantBufferEntry
            {
                uint32_t pad[4];
            };

            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                // nvrhi::BindingSetItem::ConstantBuffer(0, m_ConstantBuffer, nvrhi::BufferRange(sizeof(ConstantBufferEntry), sizeof(ConstantBufferEntry))),
                 nvrhi::BindingSetItem::Texture_SRV(0, m_Texture),
                 nvrhi::BindingSetItem::Sampler(0, m_LinearSampler)
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

            m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer);
        }

        m_CommandList->open();

        nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));

        nvrhi::GraphicsState state;
        state.pipeline = m_Pipeline;
        state.framebuffer = framebuffer;
        state.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
        state.bindings = { m_BindingSets };
        m_CommandList->setGraphicsState(state);

        nvrhi::DrawArguments args;
        args.vertexCount = 3;
        m_CommandList->draw(args);

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
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

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    
    {
        BasicTriangle example(deviceManager);
        if (example.Init())
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&example);
        }
    }
    
    deviceManager->Shutdown();

    delete deviceManager;

    return 0;
}
