/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef INCLUDE_OMM_SDK_CPP
#define INCLUDE_OMM_SDK_CPP

#include "omm.h"

namespace omm
{

   enum class Result
   {
      SUCCESS,
      FAILURE,
      INVALID_ARGUMENT,
      INSUFFICIENT_SCRATCH_MEMORY,
      NOT_IMPLEMENTED,
      WORKLOAD_TOO_BIG,
      MAX_NUM,
   };

   struct LibraryDesc
   {
      uint8_t versionMajor;
      uint8_t versionMinor;
      uint8_t versionBuild;
   };

   enum class OpacityState
   {
      Transparent,
      Opaque,
      UnknownTransparent,
      UnknownOpaque,
   };

   enum class SpecialIndex
   {
      FullyTransparent        = -1,
      FullyOpaque             = -2,
      FullyUnknownTransparent = -3,
      FullyUnknownOpaque      = -4,
   };

   enum class Format
   {
      INVALID,
      // Value maps to DX/VK spec.
      OC1_2_State = 1,
      // Value maps to DX/VK spec.
      OC1_4_State = 2,
      MAX_NUM     = 3,
   };

   enum class UnknownStatePromotion
   {
      // Will either be UO or UT depending on the coverage. If the micro-triangle is "mostly" opaque it will be UO (4-state) or O
      // (2-state). If the micro-triangle is "mostly" transparent it will be UT (4-state) or T (2-state)
      Nearest,
      // All unknown states get promoted to O in 2-state mode, or UO in 4-state mode
      ForceOpaque,
      // All unknown states get promoted to T in 2-state mode, or UT in 4-state mode
      ForceTransparent,
      MAX_NUM,
   };

   enum class BakerType
   {
      GPU,
      CPU,
      MAX_NUM,
   };

   enum class TexCoordFormat
   {
      UV16_UNORM,
      UV16_FLOAT,
      UV32_FLOAT,
      MAX_NUM,
   };

   enum class IndexFormat
   {
      I16_UINT,
      I32_UINT,
      MAX_NUM,
   };

   enum class TextureAddressMode
   {
      Wrap,
      Mirror,
      Clamp,
      Border,
      MirrorOnce,
      MAX_NUM,
   };

   enum class TextureFilterMode
   {
      Nearest,
      Linear,
      MAX_NUM,
   };

   enum class AlphaMode
   {
      Test,
      Blend,
      MAX_NUM,
   };

   struct SamplerDesc
   {
      TextureAddressMode addressingMode  = TextureAddressMode::MAX_NUM;
      TextureFilterMode  filter          = TextureFilterMode::MAX_NUM;
      float              borderAlpha     = 0;
   };

   struct MemoryAllocatorInterface
   {
      ommAllocate   Allocate    = nullptr;
      ommReallocate Reallocate  = nullptr;
      ommFree       Free        = nullptr;
      void*         UserArg     = nullptr;
   };

   struct BakerCreationDesc
   {
      BakerType                type                      = BakerType::MAX_NUM;
      bool                     enableValidation          = false;
      MemoryAllocatorInterface memoryAllocatorInterface  = {};
   };

   using Handle = uintptr_t;

   using Baker = Handle;

   static inline LibraryDesc GetLibraryDesc();

   static inline Result CreateBaker(const BakerCreationDesc& bakeCreationDesc, Baker* outBaker);

   static inline Result DestroyBaker(Baker baker);

   namespace Cpu
   {

      using BakeResult = Handle;

      using Texture = Handle;

      enum class TextureFormat
      {
         FP32,
         MAX_NUM,
      };

      enum class TextureFlags
      {
         None,
         // Controls the internal memory layout of the texture. does not change the expected input format, it does affect the baking
         // performance and memory footprint of the texture object.
         DisableZOrder = 1u << 0,
      };
      OMM_DEFINE_ENUM_FLAG_OPERATORS(TextureFlags);

      enum class BakeFlags
      {
         None,

         // Baker will use internal threads to run the baking process in parallel.
         EnableInternalThreads        = 1u << 0,

         // Will disable the use of special indices in case the OMM-state is uniform, Only set this flag for debug purposes.
         // Note: This prevents promotion of fully known OMMs to use special indices, however for invalid & degenerate UV triangles
         // special indices may still be set.
         DisableSpecialIndices        = 1u << 1,

         // Force 32-bit index format in ommIndexFormat
         Force32BitIndices            = 1u << 2,

         // Will disable reuse of OMMs and instead produce duplicates omm-array data. Generally only needed for debug purposes.
         DisableDuplicateDetection    = 1u << 3,

         // This enables merging of "similar" OMMs where similarity is measured using hamming distance.
         // UT and UO are considered identical.
         // Pros: normally reduces resulting OMM size drastically, especially when there's overlapping UVs.
         // Cons: The merging comes at the cost of coverage.
         // The resulting OMM Arrays will have lower fraction of known states. For large working sets it can be quite CPU heavy to
         // compute.
         EnableNearDuplicateDetection = 1u << 4,

         // Workload validation is a safety mechanism that will let the SDK reject workloads that become unreasonably large, which
         // may lead to long baking times. When this flag is set the bake operation may return error WORKLOAD_TOO_BIG
         EnableWorkloadValidation     = 1u << 5,
      };
      OMM_DEFINE_ENUM_FLAG_OPERATORS(BakeFlags);

      // The baker supports conservativle baking from a MIP array when the runtime wants to pick freely between texture levels at
      // runtime without the need to update the OMM data. _However_ baking from mip level 0 only is recommended in the general
      // case for best performance the integration guide contains more in depth discussion on the topic
      struct TextureMipDesc
      {
         uint32_t    width        = 0;
         uint32_t    height       = 0;
         uint32_t    rowPitch     = 0;
         const void* textureData  = nullptr;
      };

      struct TextureDesc
      {
         TextureFormat         format    = TextureFormat::MAX_NUM;
         TextureFlags          flags     = TextureFlags::None;
         const TextureMipDesc* mips      = nullptr;
         uint32_t              mipCount  = 0;
      };

      struct BakeInputDesc
      {
         BakeFlags             bakeFlags                     = BakeFlags::None;
         Texture               texture                       = 0;
         // RuntimeSamplerDesc should match the sampler type used at runtime
         SamplerDesc           runtimeSamplerDesc            = {};
         AlphaMode             alphaMode                     = AlphaMode::MAX_NUM;
         TexCoordFormat        texCoordFormat                = TexCoordFormat::MAX_NUM;
         const void*           texCoords                     = nullptr;
         // texCoordStrideInBytes: If zero, packed aligment is assumed
         uint32_t              texCoordStrideInBytes         = 0;
         IndexFormat           indexFormat                   = IndexFormat::MAX_NUM;
         const void*           indexBuffer                   = nullptr;
         uint32_t              indexCount                    = 0;
         // Configure the target resolution when running dynamic subdivision level.
         // <= 0: disabled.
         // > 0: The subdivision level be chosen such that a single micro-triangle covers approximatley a dynamicSubdivisionScale *
         // dynamicSubdivisionScale texel area.
         float                 dynamicSubdivisionScale       = 2;
         // Rejection threshold [0,1]. Unless OMMs achive a rate of at least rejectionThreshold known states OMMs will be discarded
         // for the primitive. Use this to weed out "poor" OMMs.
         float                 rejectionThreshold            = 0;
         // The alpha cutoff value. texture > alphaCutoff ? Opaque : Transparent
         float                 alphaCutoff                   = 0.5f;
         // The global Format. May be overriden by the per-triangle subdivision level setting.
         Format                format                        = Format::OC1_4_State;
         // Use Formats to control format on a per triangle granularity. If Format is set to Format::INVALID the global setting will
         // be used instead.
         const Format*         formats                       = nullptr;
         // Determines how to promote mixed states
         UnknownStatePromotion unknownStatePromotion         = UnknownStatePromotion::ForceOpaque;
         // Micro triangle count is 4^N, where N is the subdivision level.
         // maxSubdivisionLevel level must be in range [0, 12].
         // When dynamicSubdivisionScale is enabled maxSubdivisionLevel is the max subdivision level allowed.
         // When dynamicSubdivisionScale is disabled maxSubdivisionLevel is the subdivision level applied uniformly to all
         // triangles.
         uint8_t               maxSubdivisionLevel           = 8;
         bool                  enableSubdivisionLevelBuffer  = false;
         // [optional] Use subdivisionLevels to control subdivision on a per triangle granularity.
         // +14 - reserved for future use.
         // 13 - use global value specified in 'subdivisionLevel.
         // [0,12] - per triangle subdivision level'
         const uint8_t*        subdivisionLevels             = nullptr;
      };

      struct OpacityMicromapDesc
      {
         // Byte offset into the opacity micromap map array.
         uint32_t offset;
         // Micro triangle count is 4^N, where N is the subdivision level.
         uint16_t subdivisionLevel;
         // OMM input format.
         uint16_t format;
      };

      struct OpacityMicromapUsageCount
      {
         // Number of OMMs with the specified subdivision level and format.
         uint32_t count;
         // Micro triangle count is 4^N, where N is the subdivision level.
         uint16_t subdivisionLevel;
         // OMM input format.
         uint16_t format;
      };

      struct BakeResultDesc
      {
         // Below is used as OMM array build input DX/VK.
         const void*                      arrayData;
         uint32_t                         arrayDataSize;
         const OpacityMicromapDesc*       descArray;
         uint32_t                         descArrayCount;
         // The histogram of all omm data referenced by 'ommDescArray', can be used as 'pOMMUsageCounts' for the OMM build in D3D12
         const OpacityMicromapUsageCount* descArrayHistogram;
         uint32_t                         descArrayHistogramCount;
         // Below is used for BLAS build input in DX/VK
         const void*                      indexBuffer;
         uint32_t                         indexCount;
         IndexFormat                      indexFormat;
         // Same as ommDescArrayHistogram but usage count equals the number of references by ommIndexBuffer. Can be used as
         // 'pOMMUsageCounts' for the BLAS OMM attachment in D3D12
         const OpacityMicromapUsageCount* indexHistogram;
         uint32_t                         indexHistogramCount;
      };

      static inline Result CreateTexture(Baker baker, const TextureDesc& desc, Texture* outTexture);

      static inline Result DestroyTexture(Baker baker, Texture texture);

      static inline Result Bake(Baker baker, const BakeInputDesc& bakeInputDesc, BakeResult* outBakeResult);

      static inline Result DestroyBakeResult(BakeResult bakeResult);

      static inline Result GetBakeResultDesc(BakeResult bakeResult, const BakeResultDesc** desc);

   } // namespace Cpu

   namespace Gpu
   {

      using Pipeline = Handle;

      enum class DescriptorType
      {
         TextureRead,
         BufferRead,
         RawBufferRead,
         RawBufferWrite,
         MAX_NUM,
      };

      enum class ResourceType
      {
         // 1-4 channels, any format.
         IN_ALPHA_TEXTURE,
         IN_TEXCOORD_BUFFER,
         IN_INDEX_BUFFER,
         // (Optional) R8, Values must be in range [-2, 12].
         // Positive values to enforce specific subdibision level for the primtive.
         // -1 to use global subdivision level.
         // -2 to use automatic subduvision level based on tunable texel-area heuristic
         IN_SUBDIVISION_LEVEL_BUFFER,
         // Used directly as argument for OMM build in DX/VK
         OUT_OMM_ARRAY_DATA,
         // Used directly as argument for OMM build in DX/VK
         OUT_OMM_DESC_ARRAY,
         // Used directly as argument for OMM build in DX/VK. (Read back to CPU to query memory requirements during OMM Array build)
         OUT_OMM_DESC_ARRAY_HISTOGRAM,
         // Used directly as argument for OMM build in DX/VK
         OUT_OMM_INDEX_BUFFER,
         // Used directly as argument for OMM build in DX/VK. (Read back to CPU to query memory requirements during OMM Blas build)
         OUT_OMM_INDEX_HISTOGRAM,
         // (Optional, enabled if EnablePostBuildInfo is set). Read back the PostBakeInfo struct containing the actual sizes of
         // ARRAY_DATA and DESC_ARRAY.
         OUT_POST_BAKE_INFO,
         // Can be reused after baking
         TRANSIENT_POOL_BUFFER,
         // Initialize on startup. Read-only.
         STATIC_VERTEX_BUFFER,
         // Initialize on startup. Read-only.
         STATIC_INDEX_BUFFER,
         MAX_NUM,
      };

      enum class PrimitiveTopology
      {
         TriangleList,
         MAX_NUM,
      };

      enum class PipelineType
      {
         Compute,
         Graphics,
         MAX_NUM,
      };

      enum class DispatchType
      {
         Compute,
         ComputeIndirect,
         DrawIndexedIndirect,
         BeginLabel,
         EndLabel,
         MAX_NUM,
      };

      enum class BufferFormat
      {
         R32_UINT,
         MAX_NUM,
      };

      enum class RasterCullMode
      {
         None,
      };

      enum class RenderAPI
      {
         DX12,
         Vulkan,
         MAX_NUM,
      };

      enum class ScratchMemoryBudget
      {
         Undefined,
         MB_4      = 4ull << 20ull,
         MB_32     = 32ull << 20ull,
         MB_64     = 64ull << 20ull,
         MB_128    = 128ull << 20ull,
         MB_256    = 256ull << 20ull,
         MB_512    = 512ull << 20ull,
         MB_1024   = 1024ull << 20ull,
         Default   = 256ull << 20ull,
      };

      enum class BakeFlags
      {
         // Either PerformSetup, PerformBake (or both simultaneously) must be set.
         Invalid                      = 0,

         // (Default) OUT_OMM_DESC_ARRAY_HISTOGRAM, OUT_OMM_INDEX_HISTOGRAM, OUT_OMM_INDEX_BUFFER, OUT_OMM_DESC_ARRAY and
         // (optionally) OUT_POST_BAKE_INFO will be updated.
         PerformSetup                 = 1u << 0,

         // (Default) OUT_OMM_INDEX_HISTOGRAM, OUT_OMM_INDEX_BUFFER, OUT_OMM_ARRAY_DATA will be written to. If special indices are
         // detected OUT_OMM_INDEX_BUFFER may also be modified.
         // If PerformBuild is not used with this flag, OUT_OMM_DESC_ARRAY_HISTOGRAM, OUT_OMM_INDEX_HISTOGRAM, OUT_OMM_INDEX_BUFFER,
         // OUT_OMM_DESC_ARRAY must contain valid data from a prior PerformSetup pass.
         PerformBake                  = 1u << 1,

         // Baking will only be done using compute shaders and no gfx involvement (drawIndirect or graphics PSOs). (Beta)
         // Will become default mode in the future.
         // + Useful for async workloads
         // + Less memory hungry
         // + Faster baking on low texel ratio to micro-triangle ratio (=rasterizing small triangles)
         // - May looses efficency when resampling large triangles (tail-effect). Potential mitigation is to batch multiple bake
         // jobs. However this is generally not a big problem.
         ComputeOnly                  = 1u << 2,

         // Baking will also output post build info. (OUT_POST_BUILD_INFO).
         EnablePostBuildInfo          = 1u << 3,

         // Will disable the use of special indices in case the OMM-state is uniform. Only set this flag for debug purposes.
         DisableSpecialIndices        = 1u << 4,

         // If texture coordinates are known to be unique tex cooord deduplication can be disabled to save processing time and free
         // up scratch memory.
         DisableTexCoordDeduplication = 1u << 5,

         // Force 32-bit indices in OUT_OMM_INDEX_BUFFER
         Force32BitIndices            = 1u << 6,

         // Use only for debug purposes. Level Line Intersection method is vastly superior in 4-state mode.
         DisableLevelLineIntersection = 1u << 7,

         // Slightly modifies the dispatch to aid frame capture debugging.
         EnableNsightDebugMode        = 1u << 8,
      };
      OMM_DEFINE_ENUM_FLAG_OPERATORS(BakeFlags);

      struct Resource
      {
         DescriptorType stateNeeded;
         ResourceType   type;
         uint16_t       indexInPool;
         uint16_t       mipOffset;
         uint16_t       mipNum;
      };

      struct DescriptorRangeDesc
      {
         DescriptorType descriptorType;
         uint32_t       baseRegisterIndex;
         uint32_t       descriptorNum;
      };

      struct BufferDesc
      {
         size_t bufferSize;
      };

      struct ShaderBytecode
      {
         const void* data;
         size_t      size;
      };

      struct ComputePipelineDesc
      {
         ShaderBytecode             computeShader;
         const char*                shaderFileName;
         const char*                shaderEntryPointName;
         const DescriptorRangeDesc* descriptorRanges;
         uint32_t                   descriptorRangeNum;
         // if "true" all constant buffers share same "ConstantBufferDesc" description. if "false" this pipeline doesn't have a
         // constant buffer
         bool                       hasConstantData;
      };

      struct GraphicsPipelineInputElementDesc
      {
         static constexpr const char*  semanticName    = "POSITION";
         static constexpr BufferFormat format          = BufferFormat::R32_UINT;
         static constexpr uint32_t     inputSlot       = 0;
         static constexpr uint32_t     semanticIndex   = 0;
         static constexpr bool         isPerInstanced  = false;
      };

      // The graphics pipeline desc structs defines dynamically only a subset of the available raster states, what is not defined
      // dynamically is defined in this header via documentation (or constexpr variables). Keep in mind that the constexpr fields
      // may change to become non-constexpr in future releases, for this reason it's recommended to add static asserts in
      // integration code to catch it if it changes.
      // Statically asserting on the GraphicsPipelineVersion::VERSION. The purpose of doing this is to keep the integration code
      // as minimal as possible, while still keeping the door open for future extensions. For instance,
      // static_assert(GraphicsPipelineVersion::VERSION == 1, "Graphics pipeline state version changed, update integration
      // code");
      enum class GraphicsPipelineDescVersion
      {
         VERSION = 2,
      };

      // Config specification not declared in the GraphicsPipelineDesc is meant to be hard-coded and may only change in future
      // SDK versions.
      // When SDK updates the spec of GraphicsPipelineDesc GraphicsPipelineVersion::VERSION will be updated.
      // It's recommended to keep a static_assert(GraphicsPipelineVersion::VERSION == X) in the client integration layer to be
      // notified of changes.
      // Stenci state = disabled
      // BlendState = disabled
      // Primitive topology = triangle list
      // Input element = count 1, see GraphicsPipelineInputElementDesc
      // Fill mode = solid
      struct GraphicsPipelineDesc
      {
         ShaderBytecode             vertexShader;
         const char*                vertexShaderFileName;
         const char*                vertexShaderEntryPointName;
         ShaderBytecode             geometryShader;
         const char*                geometryShaderFileName;
         const char*                geometryShaderEntryPointName;
         ShaderBytecode             pixelShader;
         const char*                pixelShaderFileName;
         const char*                pixelShaderEntryPointName;
         bool                       conservativeRasterization;
         const DescriptorRangeDesc* descriptorRanges;
         uint32_t                   descriptorRangeNum;
         // if NumRenderTargets = 0 a null RTV is implied.
         uint32_t                   numRenderTargets;
         // if "true" all constant buffers share same "ConstantBufferDesc" description. if "false" this pipeline doesn't have a
         // constant buffer
         bool                       hasConstantData;
      };

      struct PipelineDesc
      {
         PipelineType type;

         union
         {
            ComputePipelineDesc  compute;
            GraphicsPipelineDesc graphics;
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

      struct Viewport
      {
         float minWidth;
         float minHeight;
         float maxWidth;
         float maxHeight;
      };

      struct ComputeDesc
      {
         const char*     name;
         // concatenated resources for all "DescriptorRangeDesc" descriptions in DenoiserDesc::pipelines[ pipelineIndex ]
         const Resource* resources;
         uint32_t        resourceNum;
         // "root constants" in DX12
         const uint8_t*  localConstantBufferData;
         uint32_t        localConstantBufferDataSize;
         uint16_t        pipelineIndex;
         uint32_t        gridWidth;
         uint32_t        gridHeight;
      };

      struct ComputeIndirectDesc
      {
         const char*     name;
         // concatenated resources for all "DescriptorRangeDesc" descriptions in DenoiserDesc::pipelines[ pipelineIndex ]
         const Resource* resources;
         uint32_t        resourceNum;
         // "root constants" in DX12
         const uint8_t*  localConstantBufferData;
         uint32_t        localConstantBufferDataSize;
         uint16_t        pipelineIndex;
         Resource        indirectArg;
         size_t          indirectArgByteOffset;
      };

      struct DrawIndexedIndirectDesc
      {
         const char*     name;
         // concatenated resources for all "DescriptorRangeDesc" descriptions in DenoiserDesc::pipelines[ pipelineIndex ]
         const Resource* resources;
         uint32_t        resourceNum;
         // "root constants" in DX12
         const uint8_t*  localConstantBufferData;
         uint32_t        localConstantBufferDataSize;
         uint16_t        pipelineIndex;
         Resource        indirectArg;
         size_t          indirectArgByteOffset;
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

      struct DispatchDesc
      {
         DispatchType type;

         union
         {
            ComputeDesc             compute;
            ComputeIndirectDesc     computeIndirect;
            DrawIndexedIndirectDesc drawIndexedIndirect;
            BeginLabelDesc          beginLabel;
         };
      };

      struct StaticSamplerDesc
      {
         SamplerDesc desc;
         uint32_t    registerIndex;
      };

      struct SPIRVBindingOffsets
      {
         uint32_t samplerOffset;
         uint32_t textureOffset;
         uint32_t constantBufferOffset;
         uint32_t storageTextureAndBufferOffset;
      };

      struct PipelineConfigDesc
      {
         // API is required to make sure indirect buffers are written to in suitable format
         RenderAPI renderAPI  = RenderAPI::DX12;
      };

      // Note: sizes may return size zero, this means the buffer will not be used in the dispatch.
      struct PreDispatchInfo
      {
         // Format of outOmmIndexBuffer
         IndexFormat outOmmIndexBufferFormat            = IndexFormat::MAX_NUM;
         uint32_t    outOmmIndexCount                   = 0xFFFFFFFF;
         // Min required size of OUT_OMM_ARRAY_DATA. GetBakeInfo returns most conservative estimation while less conservative number
         // can be obtained via BakePrepass
         size_t      outOmmArraySizeInBytes             = 0xFFFFFFFF;
         // Min required size of OUT_OMM_DESC_ARRAY. GetBakeInfo returns most conservative estimation while less conservative number
         // can be obtained via BakePrepass
         uint32_t    outOmmDescSizeInBytes              = 0xFFFFFFFF;
         // Min required size of OUT_OMM_INDEX_BUFFER
         uint32_t    outOmmIndexBufferSizeInBytes       = 0xFFFFFFFF;
         // Min required size of OUT_OMM_ARRAY_HISTOGRAM
         uint32_t    outOmmArrayHistogramSizeInBytes    = 0xFFFFFFFF;
         // Min required size of OUT_OMM_INDEX_HISTOGRAM
         uint32_t    outOmmIndexHistogramSizeInBytes    = 0xFFFFFFFF;
         // Min required size of OUT_POST_BUILD_INFO
         uint32_t    outOmmPostBuildInfoSizeInBytes     = 0xFFFFFFFF;
         // Min required sizes of TRANSIENT_POOL_BUFFERs
         uint32_t    transientPoolBufferSizeInBytes[8];
         uint32_t    numTransientPoolBuffers            = 0;
      };

      struct DispatchConfigDesc
      {
         BakeFlags           bakeFlags                     = BakeFlags::Invalid;
         // RuntimeSamplerDesc describes the texture sampler that will be used in the runtime alpha test shader code.
         SamplerDesc         runtimeSamplerDesc            = {};
         AlphaMode           alphaMode                     = AlphaMode::MAX_NUM;
         //  The texture dimensions of IN_ALPHA_TEXTURE
         uint32_t            alphaTextureWidth             = 0;
         //  The texture dimensions of IN_ALPHA_TEXTURE
         uint32_t            alphaTextureHeight            = 0;
         // The channel in IN_ALPHA_TEXTURE containing opacity values
         uint32_t            alphaTextureChannel           = 3;
         TexCoordFormat      texCoordFormat                = TexCoordFormat::MAX_NUM;
         uint32_t            texCoordOffsetInBytes         = 0;
         uint32_t            texCoordStrideInBytes         = 0;
         IndexFormat         indexFormat                   = IndexFormat::MAX_NUM;
         // The actual number of indices can be lower.
         uint32_t            indexCount                    = 0;
         // If zero packed aligment is assumed.
         uint32_t            indexStrideInBytes            = 0;
         // The alpha cutoff value. texture > alphaCutoff ? Opaque : Transparent.
         float               alphaCutoff                   = 0.5f;
         // Configure the target resolution when running dynamic subdivision level. <= 0: disabled. > 0: The subdivision level be
         // chosen such that a single micro-triangle covers approximatley a dynamicSubdivisionScale * dynamicSubdivisionScale texel
         // area.
         float               dynamicSubdivisionScale       = 2;
         // The global Format. May be overriden by the per-triangle config.
         Format              globalFormat                  = Format::OC1_4_State;
         // Micro triangle count is 4^N, where N is the subdivision level. Subdivision level must be in range [0,
         // MaxSubdivisionLevel]. The global subdivisionLevel. May be overriden by the per-triangle subdivision level setting. The
         // subdivision level to allow in dynamic mode and value is used to allocate appropriate scratch memory.
         uint8_t             globalSubdivisionLevel        = 4;
         uint8_t             maxSubdivisionLevel           = 8;
         uint8_t             enableSubdivisionLevelBuffer  = 0;
         // Target scratch memory budget, The SDK will try adjust the sum of the transient pool buffers to match this value. Higher
         // budget more efficiently executes the baking operation. May return INSUFFICIENT_SCRATCH_MEMORY if set too low.
         ScratchMemoryBudget maxScratchMemorySize          = ScratchMemoryBudget::Default;
      };

      struct PipelineInfoDesc
      {
         SPIRVBindingOffsets      spirvBindingOffsets;
         const PipelineDesc*      pipelines;
         uint32_t                 pipelineNum;
         ConstantBufferDesc       globalConstantBufferDesc;
         ConstantBufferDesc       localConstantBufferDesc;
         DescriptorSetDesc        descriptorSetDesc;
         const StaticSamplerDesc* staticSamplers;
         uint32_t                 staticSamplersNum;
      };

      // Format of OUT_POST_BAKE_INFO
      struct PostBakeInfo
      {
         uint32_t outOmmArraySizeInBytes;
         uint32_t outOmmDescSizeInBytes;
      };

      struct DispatchChain
      {
         const DispatchDesc* dispatches;
         uint32_t            numDispatches;
         const uint8_t*      globalCBufferData;
         uint32_t            globalCBufferDataSize;
      };

      // Global immutable resources. These contain the static immutable resources being shared acroess all bake calls.  Currently
      // it's the specific IB and VB that represents a tesselated triangle arranged in bird curve order, for different
      // subdivision levels.
      static inline Result GetStaticResourceData(ResourceType resource, uint8_t* data, size_t* outByteSize);

      static inline Result CreatePipeline(Baker baker, const PipelineConfigDesc& pipelineCfg, Pipeline* outPipeline);

      static inline Result DestroyPipeline(Baker baker, Pipeline pipeline);

      // Return the required pipelines. Does not depend on per-dispatch settings.
      static inline Result GetPipelineDesc(Pipeline pipeline, const PipelineInfoDesc** outPipelineDesc);

      // Returns the scratch and output memory requirements of the baking operation.
      static inline Result GetPreDispatchInfo(Pipeline pipeline, const DispatchConfigDesc& config, PreDispatchInfo* outPreDispatchInfo);

      // Returns the dispatch order to perform the baking operation. Once complete the OUT_OMM_* resources will be written to and
      // can be consumed by the application.
      static inline Result Dispatch(Pipeline pipeline, const DispatchConfigDesc& config, const DispatchChain** outDispatchDesc);

   } // namespace Gpu

   namespace Debug
   {

      struct SaveImagesDesc
      {
         const char* path                = "";
         const char* filePostfix         = "";
         // The default behaviour is to dump the entire alpha texture with the OMM-triangle in it. Enabling detailedCutout will
         // generate cropped version zoomed in on the OMM, and supersampled for detailed analysis
         bool        detailedCutout      = false;
         // Only dump index 0.
         bool        dumpOnlyFirstOMM    = false;
         // Will draw unknown transparent and unknown opaque in the same color.
         bool        monochromeUnknowns  = false;
         // true:Will draw all primitives to the same file. false: will draw each primitive separatley.
         bool        oneFile             = false;
      };

      // Walk each primitive and dumps the corresponding OMM overlay to the alpha textures.
      static inline Result SaveAsImages(Baker baker, const Cpu::BakeInputDesc& bakeInputDesc, const Cpu::BakeResultDesc* res, const SaveImagesDesc& desc);

      struct Stats
      {
         uint64_t totalOpaque                   = 0;
         uint64_t totalTransparent              = 0;
         uint64_t totalUnknownTransparent       = 0;
         uint64_t totalUnknownOpaque            = 0;
         uint32_t totalFullyOpaque              = 0;
         uint32_t totalFullyTransparent         = 0;
         uint32_t totalFullyUnknownOpaque       = 0;
         uint32_t totalFullyUnknownTransparent  = 0;
      };

      static inline Result GetStats(Baker baker, const Cpu::BakeResultDesc* res, Stats* out);

   } // namespace Debug

} // namespace omm

namespace omm
{
	static inline LibraryDesc GetLibraryDesc()
	{
        ommLibraryDesc res = ommGetLibraryDesc();
        return reinterpret_cast<LibraryDesc&>(res);
	}
	static inline Result CreateBaker(const BakerCreationDesc& bakeCreationDesc, Baker* outBaker)
	{
		static_assert(sizeof(BakerCreationDesc) == sizeof(ommBakerCreationDesc));
		return (Result)ommCreateBaker(reinterpret_cast<const ommBakerCreationDesc*>(&bakeCreationDesc), (ommBaker*)outBaker);
	}
	static inline Result DestroyBaker(Baker baker)
	{
		return (Result)ommDestroyBaker((ommBaker)baker);
	}
	namespace Cpu
	{
		static inline Result CreateTexture(Baker baker, const TextureDesc& desc, Texture* outTexture)
		{
			return (Result)ommCpuCreateTexture((ommBaker)baker, reinterpret_cast<const ommCpuTextureDesc*>(&desc), (ommCpuTexture*)outTexture);
		}
		static inline Result DestroyTexture(Baker baker, Texture texture)
		{
			return (Result)ommCpuDestroyTexture((ommBaker)baker, (ommCpuTexture)texture);
		}
		static inline Result Bake(Baker baker, const BakeInputDesc& bakeInputDesc, BakeResult* outBakeResult)
		{
			return (Result)ommCpuBake((ommBaker)baker, reinterpret_cast<const ommCpuBakeInputDesc*>(&bakeInputDesc), (ommCpuBakeResult*)outBakeResult);
		}
		static inline Result DestroyBakeResult(BakeResult bakeResult)
		{
			return (Result)ommCpuDestroyBakeResult((ommCpuBakeResult)bakeResult);
		}
		static inline Result GetBakeResultDesc(BakeResult bakeResult, const BakeResultDesc** desc)
		{
			return (Result)ommCpuGetBakeResultDesc((ommCpuBakeResult)bakeResult, reinterpret_cast<const ommCpuBakeResultDesc**>(desc));
		}
	}
	namespace Gpu
	{
		static inline Result GetStaticResourceData(ResourceType resource, uint8_t* data, size_t* outByteSize)
		{
			return (Result)ommGpuGetStaticResourceData((ommGpuResourceType)resource, data, outByteSize);
		}
		static inline Result CreatePipeline(Baker baker, const PipelineConfigDesc& pipelineCfg, Pipeline* outPipeline)
		{
			return (Result)ommGpuCreatePipeline((ommBaker)baker, reinterpret_cast<const ommGpuPipelineConfigDesc*>(&pipelineCfg), (ommGpuPipeline*)outPipeline);
		}
		static inline Result DestroyPipeline(Baker baker, Pipeline pipeline)
		{
			return (Result)ommGpuDestroyPipeline((ommBaker)baker, (ommGpuPipeline)pipeline);
		}
		static inline Result GetPipelineDesc(Pipeline pipeline, const PipelineInfoDesc** outPipelineDesc)
		{
			return (Result)ommGpuGetPipelineDesc((ommGpuPipeline)pipeline, reinterpret_cast<const ommGpuPipelineInfoDesc**>(outPipelineDesc));
		}
		static inline Result GetPreDispatchInfo(Pipeline pipeline, const DispatchConfigDesc& config, PreDispatchInfo* outPreBuildInfo)
		{
			return (Result)ommGpuGetPreDispatchInfo((ommGpuPipeline)pipeline, reinterpret_cast<const ommGpuDispatchConfigDesc*>(&config), reinterpret_cast<ommGpuPreDispatchInfo*>(outPreBuildInfo));
		}
		static inline Result Dispatch(Pipeline pipeline, const DispatchConfigDesc& config, const DispatchChain** outDispatchDesc)
		{
			return (Result)ommGpuDispatch((ommGpuPipeline)pipeline, reinterpret_cast<const ommGpuDispatchConfigDesc*>(&config), reinterpret_cast<const ommGpuDispatchChain**>(outDispatchDesc));
		}
	}
	namespace Debug
	{
		static inline Result SaveAsImages(Baker baker, const Cpu::BakeInputDesc& bakeInputDesc, const Cpu::BakeResultDesc* res, const SaveImagesDesc& desc)
		{
			return (Result)ommDebugSaveAsImages((ommBaker)baker, reinterpret_cast<const ommCpuBakeInputDesc*>(&bakeInputDesc), reinterpret_cast<const ommCpuBakeResultDesc*>(res), reinterpret_cast<const ommDebugSaveImagesDesc*>(&desc));
		}
		static inline Result GetStats(Baker baker, const Cpu::BakeResultDesc* res, Stats* out)
		{
			return (Result)ommDebugGetStats((ommBaker)baker, reinterpret_cast<const ommCpuBakeResultDesc*>(res), reinterpret_cast<ommDebugStats*>(out));
		}
	}
}

#endif // #ifndef INCLUDE_OMM_SDK_CPP
