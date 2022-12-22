/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <cstdint>
#include <cstddef>

#define OMM_VERSION_MAJOR 0
#define OMM_VERSION_MINOR 9
#define OMM_VERSION_BUILD 0
#define OMM_VERSION_DATE "12 December 2022"

#if defined(_MSC_VER)
    #define OMM_CALL __fastcall
#elif !defined(__aarch64__) && !defined(__x86_64) && (defined(__GNUC__)  || defined (__clang__))
    #define OMM_CALL __attribute__((fastcall))
#else
    #define OMM_CALL 
#endif

#ifndef OMM_API
    #define OMM_API extern "C"
#endif

#ifdef DEFINE_ENUM_FLAG_OPERATORS
#define OMM_DEFINE_ENUM_FLAG_OPERATORS(x) DEFINE_ENUM_FLAG_OPERATORS(x)
#else
#define OMM_DEFINE_ENUM_FLAG_OPERATORS(x)
#endif

#ifndef OMM_SUPPORTS_CPP17
    #if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
        #define OMM_SUPPORTS_CPP17 (1)
    #else
        #define OMM_SUPPORTS_CPP17 (0)
    #endif
#endif

namespace omm
{
    enum class Result : uint32_t
    {
        SUCCESS,
        FAILURE,
        INVALID_ARGUMENT,
        INSUFFICIENT_SCRATCH_MEMORY,
        NOT_IMPLEMENTED,
        WORKLOAD_TOO_BIG,

        MAX_NUM
    };

    enum class OpacityState : int32_t 
    {
        Transparent,
        Opaque,
        UnknownTransparent,
        UnknownOpaque
    };

    enum class SpecialIndex : int32_t 
    {
        FullyTransparent        = -1, // Value maps to DX/VK spec.
        FullyOpaque             = -2, // Value maps to DX/VK spec.
        FullyUnknownTransparent = -3, // Value maps to DX/VK spec.
        FullyUnknownOpaque      = -4, // Value maps to DX/VK spec.
    };

    enum class OMMFormat : uint16_t 
    {
        INVALID     = 0,
        OC1_2_State = 1, // Value maps to DX/VK spec.
        OC1_4_State = 2, // Value maps to DX/VK spec.

        MAX_NUM     = 3,
    };

    // Determines how to promote mixed states to either UT or UO
    enum class UnknownStatePromotion : uint8_t
    {
        // Will either be UO or UT depending on the coverage:
        // if the micro-triangle is "mostly" opaque it will be UO (4-state) or O (2-state)
        // if the micro-triangle is "mostly" transparent it will be UT (4-state) or T (2-state)
        Nearest,

        // All unknown states get promoted to O in 2-state mode, or UO in 4-state mode
        ForceOpaque,
        // All unknown states get promoted to T in 2-state mode, or UT in 4-state mode
        ForceTransparent,

        MAX_NUM,
    };

    enum class BakerType : uint8_t 
    {
        GPU,
        CPU,

        MAX_NUM
    };

    enum class TexCoordFormat : uint32_t 
    {
        UV16_UNORM,
        UV16_FLOAT,
        UV32_FLOAT,

        MAX_NUM
    };

    enum class IndexFormat : uint32_t 
    {
        I16_UINT,
        I32_UINT,

        MAX_NUM
    };

    enum class TextureAddressMode : uint32_t 
    {
        Wrap,
        Mirror,
        Clamp,
        Border,
        MirrorOnce,

        MAX_NUM
    };

    enum class TextureFilterMode : uint32_t 
    {
        Nearest,
        Linear,

        MAX_NUM
    };

    enum class AlphaMode : uint32_t 
    {
        Test,
        Blend,

        MAX_NUM
    };

    struct SamplerDesc
    {
        TextureAddressMode  addressingMode  = TextureAddressMode::MAX_NUM;
        TextureFilterMode   filter          = TextureFilterMode::MAX_NUM;
        float               borderAlpha     = 0;
    };

    struct MemoryAllocatorInterface
    {
        void* (*Allocate)(void* userArg, size_t size, size_t alignment) = nullptr;
        void* (*Reallocate)(void* userArg, void* memory, size_t size, size_t alignment) = nullptr;
        void (*Free)(void* userArg, void* memory) = nullptr;
        void* userArg = nullptr;
    };

    struct LibraryDesc
    {
        uint8_t versionMajor;
        uint8_t versionMinor;
        uint8_t versionBuild;
    };

    struct BakerCreationDesc
    {
        BakerType                   type                        = BakerType::MAX_NUM;
        bool                        enableValidation            = false;
        MemoryAllocatorInterface    memoryAllocatorInterface;
    };

    using Handle = uintptr_t;
    using Baker  = Handle;
    enum : Handle
    {
        kInvalidHandle = 0
    };

    OMM_API const LibraryDesc& OMM_CALL GetLibraryDesc();
    // Generally only a single baker context is needed, it's safe to use a single context for concurent baking tasks
    OMM_API Result OMM_CALL CreateOpacityMicromapBaker(const BakerCreationDesc& bakeCreationDesc, Baker* outBaker);
    OMM_API Result OMM_CALL DestroyOpacityMicromapBaker(Baker baker);

    namespace Cpu
    {
        using BakeResult    = Handle;
        using Texture       = Handle;

        enum class TextureFormat 
        {
            FP32,
            MAX_NUM,
        };

        enum class BakeFlags : uint32_t 
        {
            None                                    = 0,

            // Baker will use internal threads to run the baking process in parallel.
            EnableInternalThreads                   = 1u << 0,

            // Will disable the use of special indices in case the OMM-state is uniform,
            // Only set this flag for debug purposes.
            // Note: This is prevents promotion of fully known OMMs to use special indices,
            // however for invalid & degenerate UV triangles special indices may still be set.
            DisableSpecialIndices                   = 1u << 1,

            // Force 32-bit index format in ommIndexFormat
            Force32BitIndices                       = 1u << 2,

            // Will disable reuse of OMMs and instead produce duplicates omm-array data
            // Generally only needed for debug purposes.
            DisableDuplicateDetection               = 1u << 3,

            // This enables merging of "similar" OMMs where similarity is measured using hamming distance.
            // UT and UO are considered identical.
            // Pros: 
            //   - normally reduces resulting OMM size drastically, especially when there's overlapping UVs.
            // Cons: 
            //   - The merging comes at the cost of coverage. The resulting OMM Arrays will have lower fraction of known states.
            //   - For large working sets it can be quite CPU heavy to compute.
            EnableNearDuplicateDetection            = 1u << 4,

            // Workload validation is a safety mechanism that will let the SDK reject workloads that become unreasonably large, which may lead to long baking times
            // When this flag is set the bake operation may return error WORKLOAD_TOO_BIG
            EnableWorkloadValidation                = 1u << 5
        };
        OMM_DEFINE_ENUM_FLAG_OPERATORS(BakeFlags);

        enum class TextureFlags
        {
            None                    = 0,

            // Controls the internal memory layout of the texture.
            // does not change the expected input format, it does affect the baking 
            // performance and memory footprint of the texture object.
            DisableZOrder        = 1,
        };
        OMM_DEFINE_ENUM_FLAG_OPERATORS(TextureFlags);

        // The baker supports conservativle baking from a MIP array when the runtime wants to pick freely between 
        // texture levels at runtime without the need to update the OMM data.
        // _However_ baking from mip level 0 only is recommended in the general case for best performance
        // the integration guide contains more in depth discussion on the topic
        struct TextureMipDesc
        {
            uint32_t        width           = 0;
            uint32_t        height          = 0;
            uint32_t        rowPitch        = 0; // If 0 => assumed to be equal to width * format byte size
            const void*     textureData     = nullptr;
        };

        struct TextureDesc
        {
            TextureFormat           format      = TextureFormat::MAX_NUM;
            TextureFlags            flags       = TextureFlags::None;
            const TextureMipDesc*   mips        = nullptr;
            uint32_t                mipCount    = 0;
        };

        struct BakeInputDesc
        {
            BakeFlags               bakeFlags                   = BakeFlags::None;

            Texture                 texture                     = kInvalidHandle;
            // RuntimeSamplerDesc should match the sampler type used at runtime
            SamplerDesc             runtimeSamplerDesc;         
            AlphaMode               alphaMode                   = AlphaMode::MAX_NUM;
            TexCoordFormat          texCoordFormat              = TexCoordFormat::MAX_NUM;
            const void*             texCoords                   = nullptr;
            // texCoordStrideInBytes: If zero, packed aligment is assumed
            uint32_t                texCoordStrideInBytes       = 0;
            IndexFormat             indexFormat                 = IndexFormat::MAX_NUM;
            const void*             indexBuffer                 = nullptr;
            uint32_t                indexCount                  = 0;

            // Configure the target resolution when running dynamic subdivision level.
            // <= 0: disabled.
            // > 0: The subdivision level be chosen such that a single micro-triangle covers approximatley a 
            // dynamicSubdivisionScale * dynamicSubdivisionScale texel area.
            float                   dynamicSubdivisionScale      = 2;

            // Rejection threshold [0,1]. Unless OMMs achive a rate of at least rejectionThreshold known
            // states OMMs will be discarded for the primitive. Use this to weed out "poor" OMMs.
            float                   rejectionThreshold          = 0.0f;

            // The alpha cutoff value. texture > alphaCutoff ? Opaque : Transparent 
            float                   alphaCutoff                 = 0.5f;

            // Determines how to promote mixed states
            UnknownStatePromotion   unknownStatePromotion       = UnknownStatePromotion::ForceOpaque;

            // The global OMMFormat. May be overriden by the per-triangle subdivision level setting.
            OMMFormat               ommFormat                    = OMMFormat::OC1_4_State;

            // Use ommFormats to control format on a per triangle granularity.
            // If ommFormat is set to OMMFormat::INVALID the global setting will be used instead.
            OMMFormat*              ommFormats                   = nullptr;

            // Micro triangle count is 4^N, where N is the subdivision level.
            // maxSubdivisionLevel level must be in range [0, 12]
            // When dynamicSubdivisionScale is enabled maxSubdivisionLevel is the max subdivision level allowed.
            // When dynamicSubdivisionScale is disabled maxSubdivisionLevel is the subdivision level applied uniformly to all triangles.
            uint8_t                 maxSubdivisionLevel          = 8;

            // [optional] Use subdivisionLevels to control subdivision on a per triangle granularity.
            // val:+14     - reserved for future use
            // val:13      - use global value specified in 'subdivisionLevel'
            // val:0-12    - per triangle subdivision level
            uint8_t*                subdivisionLevels           = nullptr; 
        };

        struct OpacityMicromapDesc
        {
            // Byte offset into the opacity micromap map array
            uint32_t offset             = 0;
            // Micro triangle count is 4^N, where N is the subdivision level.
            uint16_t subdivisionLevel   = 0;
            // OMM input format.
            uint16_t format             = 0;
        };

        struct OpacityMicromapUsageCount
        {
            // Number of OMMs with the specified subdivision level and format.
            uint32_t count              = 0;
            // Micro triangle count is 4^N, where N is the subdivision level.
            uint16_t subdivisionLevel   = 0;
            // OMM input sub format.
            uint16_t format             = 0;
        };

        struct BakeResultDesc
        {
            // Below is used as OMM array build input DX/VK.
            const void*                         ommArrayData                    = nullptr;
            uint32_t                            ommArrayDataSize                = 0;
            const OpacityMicromapDesc*          ommDescArray                    = nullptr;
            uint32_t                            ommDescArrayCount               = 0;
            // The histogram of all omm data referenced by 'ommDescArray', can be used as 'pOMMUsageCounts' for the OMM build in D3D12
            const OpacityMicromapUsageCount*    ommDescArrayHistogram           = 0;
            uint32_t                            ommDescArrayHistogramCount      = 0;

            // Below is used for BLAS build input in DX/VK
            const void*                         ommIndexBuffer                  = nullptr;
            uint32_t                            ommIndexCount                   = 0;
            IndexFormat                         ommIndexFormat                  = IndexFormat::MAX_NUM;
            // Same as ommDescArrayHistogram but usage count equals the number of references by ommIndexBuffer. Can be used as 'pOMMUsageCounts' for the BLAS OMM attachment in D3D12
            const OpacityMicromapUsageCount*    ommIndexHistogram               = 0;
            uint32_t                            ommIndexHistogramCount          = 0;
        };

        OMM_API Result OMM_CALL CreateTexture(Baker baker, const TextureDesc& desc, Texture* outTexture);
        OMM_API Result OMM_CALL DestroyTexture(Baker baker, Texture texture);
        OMM_API Result OMM_CALL BakeOpacityMicromap(Baker baker, const BakeInputDesc& bakeInputDesc, BakeResult* outBakeResult);
        OMM_API Result OMM_CALL DestroyBakeResult(BakeResult bakeResult);
        OMM_API Result OMM_CALL GetBakeResultDesc(BakeResult bakeResult, const BakeResultDesc*& desc);
    }

    namespace Gpu 
    {
        using Baker         = Handle;
        using Pipeline      = Handle;   
        using Dispatch      = Handle;

        enum class DescriptorType : uint32_t
        {
            TextureRead,

            BufferRead,

            RawBufferRead,
            RawBufferWrite,

            MAX_NUM
        };

        enum class ResourceType : uint32_t
        {
            // INPUTS ===========================================================================================================================
            // 1-4 channels, any format.
            IN_ALPHA_TEXTURE,
            IN_TEXCOORD_BUFFER,
            IN_INDEX_BUFFER,

            // (Optional) R8
            // Values must be in range [0, 14]. 
            // Positive values to enforce specific subdibision level for the primtive
			// 0-12 per triangle subdivision level
            // 13 use global subdivision level
            // 14 use dynamic subdivision level heuristic
            IN_SUBDIVISION_LEVEL_BUFFER,
            
            // OUTPUTS - BakeOpacityMicromaps ==================================================================================================
            OUT_OMM_ARRAY_DATA,          // Used directly as argument for OMM build in DX/VK
            OUT_OMM_DESC_ARRAY,          // Used directly as argument for OMM build in DX/VK
            OUT_OMM_DESC_ARRAY_HISTOGRAM,// Used directly as argument for OMM build in DX/VK. (Read back to CPU to query memory requirements during OMM Array build)
            OUT_OMM_INDEX_BUFFER,        // Used directly as argument for OMM BLAS attachement in DX/VK
            OUT_OMM_INDEX_HISTOGRAM,     // Used directly as argument for OMM BLAS attachement in DX/VK. (Read back to CPU to query memory requirements during OMM Blas build)
            // (Optional, enabled if EnablePostBuildInfo is set)
            // Read back the PostBakeInfo struct containing the actual sizes of ARRAY_DATA and DESC_ARRAY. 
            OUT_POST_BAKE_INFO,

            // SCRATCH =========================================================================================================================
            // Can be reused after baking
            TRANSIENT_POOL_BUFFER,

            // GLOBAL STATIC ===================================================================================================================
            // Initialize on startup. Read-only.
            STATIC_VERTEX_BUFFER,
            STATIC_INDEX_BUFFER,

            MAX_NUM,
        };

        enum class PrimitiveTopology : uint32_t 
        {
            TriangleList,

            MAX_NUM
        };

        enum class PipelineType : uint32_t 
        {
            Compute,
            Graphics,

            MAX_NUM
        };

        enum class DispatchType : uint32_t 
        {
            Compute,
            ComputeIndirect,
            DrawIndexedIndirect,

            BeginLabel,
            EndLabel,

            MAX_NUM,
        };

        enum class BufferFormat : uint32_t 
        {
            R32_UINT,

            MAX_NUM
        };

        enum class RasterCullMode : uint32_t
        {
            None,
            MAX_NUM
        };

        enum class RenderAPI : uint32_t
        {
            DX12,
            Vulkan,

            MAX_NUM
        };

        enum class ScratchMemoryBudget : size_t
        {
            Undefined,

            MB_4        = 32ull << 20ull,
            MB_32       = 32ull << 20ull,
            MB_64       = 64ull << 20ull,
            MB_128      = 128ull << 20ull,
            MB_256      = 256ull << 20ull,
            MB_512      = 512ull << 20ull,
            MB_1024     = 1024ull << 20ull,
            MB_2048     = 2048ull << 20ull,
            MB_4096     = 4096ull << 20ull,

            LowMemory   = MB_128,
            HighMemory  = MB_2048,

            Default     = MB_256,
        };

        enum class BakeFlags : uint32_t 
        {
            None                            = 0,

            // Baking will only be done using compute shaders and no gfx involvement (drawIndirect or graphics PSOs)
			// (Beta) Will become default mode in the future.
			// + Useful for async workloads
			// + Less memory hungry
			// + Faster baking on low texel ratio to micro-triangle ratio (=rasterizing small triangles)
			// - May looses efficency when resampling large triangles (tail-effect). Potential mitigation is to batch multiple bake jobs. However this is generally not a big problem.
            ComputeOnly                     = 1u << 0,

            // Baking will also output post build info. (OUT_POST_BUILD_INFO)
            EnablePostBuildInfo             = 1u << 1,

            // Will disable the use of special indices in case the OMM-state is uniform,
            // Only set this flag for debug purposes.
            DisableSpecialIndices           = 1u << 2,

            // If texture coordinates are known to be unique tex cooord deduplication can be disabled to save processing time and free up scratch memory.
            DisableTexCoordDeduplication    = 1u << 3,

            // Force 32-bit indices in OUT_OMM_INDEX_BUFFER
            Force32BitIndices               = 1u << 4,

            // Slightly modifies the dispatch to aid frame capture debugging.
            EnableNsightDebugMode           = 1u << 5,
        };
        OMM_DEFINE_ENUM_FLAG_OPERATORS(BakeFlags);

        struct Resource
        {
            DescriptorType stateNeeded;
            ResourceType type;
            uint16_t indexInPool;
            uint16_t mipOffset;
            uint16_t mipNum;
        };

        struct DescriptorRangeDesc
        {
            DescriptorType descriptorType;
            uint32_t baseRegisterIndex;
            uint32_t descriptorNum;
        };

        struct BufferDesc 
        {
            size_t bufferSize;
        };

        struct ShaderBytecode
        {
            const void* data;
            size_t size;
        };

        struct ComputePipelineDesc
        {
            ShaderBytecode              computeShader;
            const char*                 shaderFileName;
            const char*                 shaderEntryPointName;
            const DescriptorRangeDesc*  descriptorRanges;
            uint32_t                    descriptorRangeNum;

            // if "true" all constant buffers share same "ConstantBufferDesc" description
            // if "false" this pipeline doesn't have a constant buffer
            bool                        hasConstantData;
        };

        struct GraphicsPipelineDesc
        {
            ShaderBytecode  vertexShader;
            const char*     vertexShaderFileName;
            const char*     vertexShaderEntryPointName;

            ShaderBytecode  geometryShader;
            const char*     geometryShaderFileName;
            const char*     geometryShaderEntryPointName;

            ShaderBytecode  pixelShader;
            const char*     pixelShaderFileName;
            const char*     pixelShaderEntryPointName;

            struct RasterState 
            {
                #if OMM_SUPPORTS_CPP17
                static inline constexpr RasterCullMode cullMode = RasterCullMode::None;
				#endif
                bool conservativeRasterization;
            };

            RasterState                 rasterState;

            const DescriptorRangeDesc*  descriptorRanges;
            uint32_t                    descriptorRangeNum;

            // if NumRenderTargets = 0 a null RTV is implied.
            uint32_t numRenderTargets; // RTV is assumed to match viewvport size.

            // if "true" all constant buffers share same "ConstantBufferDesc" description
            // if "false" this pipeline doesn't have a constant buffer
            bool hasConstantData;

            ////////////
            /// ~ Below is implied state, that may only be become dynamic in future SDK releases ~
            ////////////

            // The graphics pipeline desc structs defines dynamically only a subset of the available raster states,
            // what is not defined dynamically is defined in this header via documentation (or constexpr variables).
            // Keep in mind that the constexpr fields may change to become non-constexpr in future releases,
            // for this reason it's recommended to add static asserts in integration code to catch it if it changes.
            // Statically asserting on the GraphicsPipelineDesc::VERSION or individual options is recommended.
            // The purpose of doing this is to keep the integration code as minimal as possible, while still keeping the door open 
            // for future extensions.
            // For instance,
            // static_assert(GraphicsPipelineDesc::VERSION == 1, "Graphics pipeline state version changed, update integration code");
            // or just specific settings;
            // static_assert(GraphicsPipelineDesc::DepthTestEnable == false, "Graphics pipeline state version changed, update integration code");

            enum
            {
                VERSION = 1,
            };

            #if OMM_SUPPORTS_CPP17
            struct InputElementDesc
            {
                static inline constexpr const char*    semanticName = "POSITION";
                static inline constexpr BufferFormat   format = BufferFormat::R32_UINT;
                static inline constexpr uint32_t       inputSlot = 0;
                static inline constexpr uint32_t       semanticIndex = 0;
                static inline constexpr bool           isPerInstanced = false;
            };

            static inline constexpr uint32_t inputElementDescCount = 1;
            static inline constexpr InputElementDesc inputElementDescs[inputElementDescCount] = { InputElementDesc{} };

            struct DepthState 
            {
                static inline constexpr bool depthTestEnable = false;
                static inline constexpr bool depthWriteEnable = false;
                static inline constexpr bool stencilEnable = false;
            };

            struct StencilState 
            {
                static inline constexpr bool enable = false;
            };

            struct BlendState 
            {
                static inline constexpr bool enable = false;
            };

            static inline constexpr PrimitiveTopology  topology = PrimitiveTopology::TriangleList;
			#endif
        };

        struct PipelineDesc
        {
            PipelineType type;

            union
            {
                ComputePipelineDesc     compute;
                GraphicsPipelineDesc    graphics;
            };
        };

        struct DescriptorSetDesc
        {
            uint32_t constantBufferMaxNum;
            uint32_t storageBufferMaxNum;
            uint32_t descriptorRangeMaxNumPerPipeline;
        };

        struct ConstantBufferDesc
        {
            uint32_t registerIndex;
            uint32_t maxDataSize;
        };

        struct ComputeDesc
        {
            const char*         name;
            const Resource*     resources; // concatenated resources for all "DescriptorRangeDesc" descriptions in DenoiserDesc::pipelines[ pipelineIndex ]
            uint32_t            resourceNum;
            const uint8_t*      localConstantBufferData; // "root constants" in DX12
            uint32_t            localConstantBufferDataSize;
            uint16_t            pipelineIndex;
            uint16_t            gridWidth;
            uint16_t            gridHeight;
        };

        struct ComputeIndirectDesc
        {
            const char*         name;
            const Resource*     resources; // concatenated resources for all "DescriptorRangeDesc" descriptions in DenoiserDesc::pipelines[ pipelineIndex ]
            uint32_t            resourceNum;
            const uint8_t*      localConstantBufferData; // "root constants" in DX12
            uint32_t            localConstantBufferDataSize;
            uint16_t            pipelineIndex;
            Resource            indirectArg;
            size_t              indirectArgByteOffset;
        };

        struct DrawIndexedIndirectDesc
        {
            const char*         name;
            const Resource*     resources; // concatenated resources for all "DescriptorRangeDesc" descriptions in DenoiserDesc::pipelines[ pipelineIndex ]
            uint32_t            resourceNum;
            const uint8_t*      localConstantBufferData; // "root constants" in DX12
            uint32_t            localConstantBufferDataSize;
            uint16_t            pipelineIndex;
            Resource            indirectArg;
            size_t              indirectArgByteOffset;

            struct Viewport {
                float minWidth;
                float minHeight;
                float maxWidth;
                float maxHeight;
            };

            Viewport        viewport;

            Resource        indexBuffer;
            uint32_t        indexBufferOffset;

            Resource        vertexBuffer;
            uint32_t        vertexBufferOffset;
        };

        struct BeginLabelDesc
        {
            const char* debugName;
        };

        struct EndLabelDesc
        {
            // empty.
        };

        struct DispatchDesc 
        {
            DispatchType    type;

            union
            {
                ComputeDesc             compute;
                ComputeIndirectDesc     computeIndirect;
                DrawIndexedIndirectDesc drawIndexedIndirect;
                BeginLabelDesc          beginLabel;
                EndLabelDesc            endLabel;
            };
        };

        struct StaticSamplerDesc
        {
            SamplerDesc desc;
            uint32_t registerIndex;
        };

        struct SPIRVBindingOffsets
        {
            uint32_t samplerOffset;
            uint32_t textureOffset;
            uint32_t constantBufferOffset;
            uint32_t storageTextureAndBufferOffset;
        };

        struct BakePipelineConfigDesc
        {
            // API is required to make sure indirect buffers are written to in suitable format
            RenderAPI renderAPI = RenderAPI::DX12;
        };

        // The BakeDispatchConfigDesc sets up the runtime configurable parameters
        struct BakeDispatchConfigDesc
        {
            BakeFlags           bakeFlags                       = BakeFlags::None;
            // RuntimeSamplerDesc describes the texture sampler that will be used in the runtime alpha test shader code.
            SamplerDesc         runtimeSamplerDesc;     
            AlphaMode           alphaMode                       = AlphaMode::MAX_NUM;
            // The texture dimensions of IN_ALPHA_TEXTURE
            uint32_t            alphaTextureWidth               = 0;
            uint32_t            alphaTextureHeight              = 0;
            // Channel in IN_ALPHA_TEXTURE where alpha is found
            uint32_t            alphaTextureChannel             = 3;
            TexCoordFormat      texCoordFormat                  = TexCoordFormat::MAX_NUM;
            uint32_t            texCoordOffsetInBytes           = 0;
            // If zero packed aligment is assumed
            uint32_t            texCoordStrideInBytes           = 0;
            IndexFormat         indexFormat                     = IndexFormat::MAX_NUM;
            // The actual number of indices can be lower.
            uint32_t            indexCount                      = 0;
            // If zero packed aligment is assumed
            uint32_t            indexStrideInBytes              = 0;
            // The alpha cutoff value. texture > alphaCutoff ? Opaque : Transparent 
            float               alphaCutoff                     = 0.5f;

            // ---------- Format ------------
            // The global OMMFormat. May be overriden by the per-triangle config.
            OMMFormat           globalOMMFormat                 = OMMFormat::OC1_4_State;
            OMMFormat           supportedOMMFormats[2]          = { OMMFormat::MAX_NUM,OMMFormat::MAX_NUM };
            uint32_t            numSupportedOMMFormats          = 0;
            
            // ---------- Subdivision Level ------------
            // Micro triangle count is 4^N, where N is the subdivision level.
            // Subdivision level must be in range [0, maxSubdivisionLevel]
            // The global subdivisionLevel. May be overriden by the per-triangle subdivision level setting.
            // The subdivision level to allow in dynamic mode and value is used to allocate appropriate scratch memory
            uint8_t             globalSubdivisionLevel          = 4;
            uint8_t             maxSubdivisionLevel             = 8;

            // Configure the target resolution when running dynamic subdivision level.
            // <= 0: disabled.
            // > 0: The subdivision level be chosen such that a single micro-triangle covers approximatley a 
            // dynamicSubdivisionScale * dynamicSubdivisionScale texel area.
            float               dynamicSubdivisionScale         = 2;

            // Control the subdivision level settings per triangle using the IN_SUBDIVISION_LEVEL_BUFFER
            bool                enableSubdivisionLevelBuffer    = false;

            // ---------- Memory ------------
            // Target scratch memory budget, The SDK will try adjust the sum of the transient pool buffers to match this value.
            // Higher budget more efficiently executes the baking operation.
            // May return INSUFFICIENT_SCRATCH_MEMORY if set too low.
            ScratchMemoryBudget maxScratchMemorySize            = ScratchMemoryBudget::Default;

            // Limit the amout of omm array memory the baking may use. Set this to the max value for the OmmArraySize.
            // This may need to be configured to avoid overly conservative memory allocation. Refer to the integration guide for an in depth discussion.
            uint32_t            maxOutOmmArraySizeInBytes       = 0xFFFFFFFF;
        };

        struct BakePipelineInfoDesc
        {
            SPIRVBindingOffsets         spirvBindingOffsets;
            const PipelineDesc*         pipelines;
            uint32_t                    pipelineNum;
            ConstantBufferDesc          globalConstantBufferDesc;
            ConstantBufferDesc          localConstantBufferDesc;
            DescriptorSetDesc           descriptorSetDesc;
            const StaticSamplerDesc*    staticSamplers;
            uint32_t                    staticSamplersNum;
        };

        struct PreBakeInfo
        {
            enum { MAX_TRANSIENT_POOL_BUFFERS = 8 };

            // Format of outOmmIndexBuffer
            IndexFormat   outOmmIndexBufferFormat;
            // triangleCount
            uint32_t      outOmmIndexCount = 0;

            // Note: may return size zero, this means the buffer will not be used in the dispatch.

            // Min required size of OUT_OMM_ARRAY_DATA
            // GetPreBakeInfo returns the most conservative estimation
            uint32_t      outOmmArraySizeInBytes;
			// Min required size of OUT_OMM_DESC_ARRAY
            // GetPreBakeInfo returns the most conservative estimation
            uint32_t      outOmmDescSizeInBytes;
            // Min required size of OUT_OMM_INDEX_BUFFER
            uint32_t      outOmmIndexBufferSizeInBytes;
            // Min required size of OUT_OMM_ARRAY_HISTOGRAM
            uint32_t      outOmmArrayHistogramSizeInBytes;
            // Min required size of OUT_OMM_INDEX_HISTOGRAM
            uint32_t      outOmmIndexHistogramSizeInBytes;
            // Min required size of OUT_POST_BUILD_INFO
            uint32_t      outOmmPostBuildInfoSizeInBytes;
            // Min required sizes of TRANSIENT_POOL_BUFFERs
            uint32_t      transientPoolBufferSizeInBytes[MAX_TRANSIENT_POOL_BUFFERS];
            uint32_t      numTransientPoolBuffers;
        };

        // Format of OUT_POST_BAKE_INFO
        struct PostBakeInfo
        {
            uint32_t    outOmmArraySizeInBytes;
            uint32_t    outOmmDescSizeInBytes;
        };

        struct BakeDispatchChain
        {
            const DispatchDesc* dispatches;
            uint32_t            numDispatches = 0;
            const uint8_t*      globalCBufferData;
            uint32_t            globalCBufferDataSize;
        };

        // Global immutable resources. These contain the static immutable resources being shared acroess all bake calls.
        // Currently it's the specific IB and VB that represents a tesselated triangle arranged in bird curve order, for different subdivision levels.
        OMM_API Result OMM_CALL GetStaticResourceData(ResourceType resource, uint8_t* data, size_t& byteSize);

        OMM_API Result OMM_CALL CreatePipeline(Baker baker, const BakePipelineConfigDesc& pipelineCfg, Pipeline* outPipeline);
        OMM_API Result OMM_CALL DestroyPipeline(Baker baker, Pipeline pipeline);

        // Return the required pipelines. Does not depend on per-dispatch settings.
        OMM_API Result OMM_CALL GetPipelineDesc(Pipeline pipeline, const BakePipelineInfoDesc*& outPipelineDesc);

        // Returns the scratch and output memory requirements of the baking operation. 
        OMM_API Result OMM_CALL GetPreBakeInfo(Pipeline pipeline, const BakeDispatchConfigDesc& config, PreBakeInfo* outPreBuildInfo);

        // Returns the dispatch order to perform the baking operation. 
        // Once complete the OUT_OMM_* resources will be written to and can be consumed by the application.
        OMM_API Result OMM_CALL Bake(Pipeline pipeline, const BakeDispatchConfigDesc& config, const BakeDispatchChain*& outDispatchDesc);
    }

    namespace Debug
    {
        struct SaveImagesDesc
		{
            const char* path            = "";
            const char* filePostfix     = "";
            // The default behaviour is to dump the entire alpha texture with the OMM-triangle in it. 
            // Enabling detailedCutout will generate cropped version zoomed in on the OMM, and supersampled for detailed analysis
            bool detailedCutout         = false;
            // Only dump index 0.
            bool dumpOnlyFirstOMM       = false;
            // Will draw unknown transparent and unknown opaque in the same color.
            bool monochromeUnknowns     = false;
            // true:Will draw all primitives to the same file. false: will draw each primitive separatley
            bool oneFile                = false;
        };

        // Walk each primitive and dumps the corresponding OMM overlay to the alpha textures.
        OMM_API Result OMM_CALL SaveAsImages(Baker baker, const Cpu::BakeInputDesc& bakeInputDesc, const Cpu::BakeResultDesc* res, const SaveImagesDesc& desc);

        struct Stats 
        {
            uint64_t totalOpaque = 0;
            uint64_t totalTransparent = 0;
            uint64_t totalUnknownTransparent = 0;
            uint64_t totalUnknownOpaque = 0;

            uint64_t totalFullyOpaque = 0;
            uint64_t totalFullyTransparent = 0;
            uint64_t totalFullyUnknownOpaque = 0;
            uint64_t totalFullyUnknownTransparent = 0;
        };

        OMM_API Result OMM_CALL GetStats(Baker baker, const Cpu::BakeResultDesc* res, Stats* out);
    }
}
