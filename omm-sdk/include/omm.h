/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef INCLUDE_OMM_SDK_C
#define INCLUDE_OMM_SDK_C

#include <stdint.h>
#include <stddef.h>

#define OMM_VERSION_MAJOR 1
#define OMM_VERSION_MINOR 0
#define OMM_VERSION_BUILD 0

#if defined(_MSC_VER)
    #define OMM_CALL __fastcall
#elif !defined(__aarch64__) && !defined(__x86_64) && (defined(__GNUC__)  || defined (__clang__))
    #define OMM_CALL __attribute__((fastcall))
#else
    #define OMM_CALL 
#endif

#ifndef OMM_API 
    #ifdef __cplusplus
        #define OMM_API extern "C"
    #else
        #define OMM_API
    #endif
#endif

#ifdef DEFINE_ENUM_FLAG_OPERATORS
#define OMM_DEFINE_ENUM_FLAG_OPERATORS(x) DEFINE_ENUM_FLAG_OPERATORS(x)
#else
#define OMM_DEFINE_ENUM_FLAG_OPERATORS(x)
#endif

typedef void* (*ommAllocate)(void* userArg, size_t size, size_t alignment);
typedef void* (*ommReallocate)(void* userArg, void* memory, size_t size, size_t alignment);
typedef void (*ommFree)(void* userArg, void* memory);

typedef uint8_t ommBool;

typedef enum ommResult
{
   ommResult_SUCCESS,
   ommResult_FAILURE,
   ommResult_INVALID_ARGUMENT,
   ommResult_INSUFFICIENT_SCRATCH_MEMORY,
   ommResult_NOT_IMPLEMENTED,
   ommResult_WORKLOAD_TOO_BIG,
   ommResult_MAX_NUM,
} ommResult;

typedef struct ommLibraryDesc
{
   uint8_t versionMajor;
   uint8_t versionMinor;
   uint8_t versionBuild;
} ommLibraryDesc;

typedef enum ommOpacityState
{
   ommOpacityState_Transparent,
   ommOpacityState_Opaque,
   ommOpacityState_UnknownTransparent,
   ommOpacityState_UnknownOpaque,
} ommOpacityState;

typedef enum ommSpecialIndex
{
   ommSpecialIndex_FullyTransparent        = -1,
   ommSpecialIndex_FullyOpaque             = -2,
   ommSpecialIndex_FullyUnknownTransparent = -3,
   ommSpecialIndex_FullyUnknownOpaque      = -4,
} ommSpecialIndex;

typedef enum ommFormat
{
   ommFormat_INVALID,
   // Value maps to DX/VK spec.
   ommFormat_OC1_2_State = 1,
   // Value maps to DX/VK spec.
   ommFormat_OC1_4_State = 2,
   ommFormat_MAX_NUM     = 3,
} ommFormat;

typedef enum ommUnknownStatePromotion
{
   // Will either be UO or UT depending on the coverage. If the micro-triangle is "mostly" opaque it will be UO (4-state) or O
   // (2-state). If the micro-triangle is "mostly" transparent it will be UT (4-state) or T (2-state)
   ommUnknownStatePromotion_Nearest,
   // All unknown states get promoted to O in 2-state mode, or UO in 4-state mode
   ommUnknownStatePromotion_ForceOpaque,
   // All unknown states get promoted to T in 2-state mode, or UT in 4-state mode
   ommUnknownStatePromotion_ForceTransparent,
   ommUnknownStatePromotion_MAX_NUM,
} ommUnknownStatePromotion;

typedef enum ommBakerType
{
   ommBakerType_GPU,
   ommBakerType_CPU,
   ommBakerType_MAX_NUM,
} ommBakerType;

typedef enum ommTexCoordFormat
{
   ommTexCoordFormat_UV16_UNORM,
   ommTexCoordFormat_UV16_FLOAT,
   ommTexCoordFormat_UV32_FLOAT,
   ommTexCoordFormat_MAX_NUM,
} ommTexCoordFormat;

typedef enum ommIndexFormat
{
   ommIndexFormat_I16_UINT,
   ommIndexFormat_I32_UINT,
   ommIndexFormat_MAX_NUM,
} ommIndexFormat;

typedef enum ommTextureAddressMode
{
   ommTextureAddressMode_Wrap,
   ommTextureAddressMode_Mirror,
   ommTextureAddressMode_Clamp,
   ommTextureAddressMode_Border,
   ommTextureAddressMode_MirrorOnce,
   ommTextureAddressMode_MAX_NUM,
} ommTextureAddressMode;

typedef enum ommTextureFilterMode
{
   ommTextureFilterMode_Nearest,
   ommTextureFilterMode_Linear,
   ommTextureFilterMode_MAX_NUM,
} ommTextureFilterMode;

typedef enum ommAlphaMode
{
   ommAlphaMode_Test,
   ommAlphaMode_Blend,
   ommAlphaMode_MAX_NUM,
} ommAlphaMode;

typedef struct ommSamplerDesc
{
   ommTextureAddressMode addressingMode;
   ommTextureFilterMode  filter;
   float                 borderAlpha;
} ommSamplerDesc;

inline ommSamplerDesc ommSamplerDescDefault()
{
   ommSamplerDesc v;
   v.addressingMode  = ommTextureAddressMode_MAX_NUM;
   v.filter          = ommTextureFilterMode_MAX_NUM;
   v.borderAlpha     = 0;
   return v;
}

typedef struct ommMemoryAllocatorInterface
{
   ommAllocate   Allocate;
   ommReallocate Reallocate;
   ommFree       Free;
   void*         UserArg;
} ommMemoryAllocatorInterface;

inline ommMemoryAllocatorInterface ommMemoryAllocatorInterfaceDefault()
{
   ommMemoryAllocatorInterface v;
   v.Allocate    = NULL;
   v.Reallocate  = NULL;
   v.Free        = NULL;
   v.UserArg     = NULL;
   return v;
}

typedef struct ommBakerCreationDesc
{
   ommBakerType                type;
   ommBool                     enableValidation;
   ommMemoryAllocatorInterface memoryAllocatorInterface;
} ommBakerCreationDesc;

inline ommBakerCreationDesc ommBakerCreationDescDefault()
{
   ommBakerCreationDesc v;
   v.type                      = ommBakerType_MAX_NUM;
   v.enableValidation          = 0;
   v.memoryAllocatorInterface  = ommMemoryAllocatorInterfaceDefault();
   return v;
}

typedef uintptr_t ommHandle;

typedef ommHandle ommBaker;

OMM_API ommLibraryDesc ommGetLibraryDesc();

OMM_API ommResult ommCreateBaker(const ommBakerCreationDesc* bakeCreationDesc, ommBaker* outBaker);

OMM_API ommResult ommDestroyBaker(ommBaker baker);

typedef ommHandle ommCpuBakeResult;

typedef ommHandle ommCpuTexture;

typedef enum ommCpuTextureFormat
{
   ommCpuTextureFormat_UNORM8,
   ommCpuTextureFormat_FP32,
   ommCpuTextureFormat_MAX_NUM,
} ommCpuTextureFormat;

typedef enum ommCpuTextureFlags
{
   ommCpuTextureFlags_None,
   // Controls the internal memory layout of the texture. does not change the expected input format, it does affect the baking
   // performance and memory footprint of the texture object.
   ommCpuTextureFlags_DisableZOrder = 1u << 0,
} ommCpuTextureFlags;
OMM_DEFINE_ENUM_FLAG_OPERATORS(ommCpuTextureFlags);

typedef enum ommCpuBakeFlags
{
   ommCpuBakeFlags_None,

   // Baker will use internal threads to run the baking process in parallel.
   ommCpuBakeFlags_EnableInternalThreads        = 1u << 0,

   // Will disable the use of special indices in case the OMM-state is uniform, Only set this flag for debug purposes.
   // Note: This prevents promotion of fully known OMMs to use special indices, however for invalid & degenerate UV triangles
   // special indices may still be set.
   ommCpuBakeFlags_DisableSpecialIndices        = 1u << 1,

   // Force 32-bit index format in ommIndexFormat
   ommCpuBakeFlags_Force32BitIndices            = 1u << 2,

   // Will disable reuse of OMMs and instead produce duplicates omm-array data. Generally only needed for debug purposes.
   ommCpuBakeFlags_DisableDuplicateDetection    = 1u << 3,

   // This enables merging of "similar" OMMs where similarity is measured using hamming distance.
   // UT and UO are considered identical.
   // Pros: normally reduces resulting OMM size drastically, especially when there's overlapping UVs.
   // Cons: The merging comes at the cost of coverage.
   // The resulting OMM Arrays will have lower fraction of known states. For large working sets it can be quite CPU heavy to
   // compute.
   ommCpuBakeFlags_EnableNearDuplicateDetection = 1u << 4,

   // Workload validation is a safety mechanism that will let the SDK reject workloads that become unreasonably large, which
   // may lead to long baking times. When this flag is set the bake operation may return error WORKLOAD_TOO_BIG
   ommCpuBakeFlags_EnableWorkloadValidation     = 1u << 5,
} ommCpuBakeFlags;
OMM_DEFINE_ENUM_FLAG_OPERATORS(ommCpuBakeFlags);

// The baker supports conservativle baking from a MIP array when the runtime wants to pick freely between texture levels at
// runtime without the need to update the OMM data. _However_ baking from mip level 0 only is recommended in the general
// case for best performance the integration guide contains more in depth discussion on the topic
typedef struct ommCpuTextureMipDesc
{
   uint32_t    width;
   uint32_t    height;
   uint32_t    rowPitch;
   const void* textureData;
} ommCpuTextureMipDesc;

inline ommCpuTextureMipDesc ommCpuTextureMipDescDefault()
{
   ommCpuTextureMipDesc v;
   v.width        = 0;
   v.height       = 0;
   v.rowPitch     = 0;
   v.textureData  = NULL;
   return v;
}

typedef struct ommCpuTextureDesc
{
   ommCpuTextureFormat         format;
   ommCpuTextureFlags          flags;
   const ommCpuTextureMipDesc* mips;
   uint32_t                    mipCount;
} ommCpuTextureDesc;

inline ommCpuTextureDesc ommCpuTextureDescDefault()
{
   ommCpuTextureDesc v;
   v.format    = ommCpuTextureFormat_MAX_NUM;
   v.flags     = ommCpuTextureFlags_None;
   v.mips      = NULL;
   v.mipCount  = 0;
   return v;
}

typedef struct ommCpuBakeInputDesc
{
   ommCpuBakeFlags          bakeFlags;
   ommCpuTexture            texture;
   // RuntimeSamplerDesc should match the sampler type used at runtime
   ommSamplerDesc           runtimeSamplerDesc;
   ommAlphaMode             alphaMode;
   ommTexCoordFormat        texCoordFormat;
   const void*              texCoords;
   // texCoordStrideInBytes: If zero, packed aligment is assumed
   uint32_t                 texCoordStrideInBytes;
   ommIndexFormat           indexFormat;
   const void*              indexBuffer;
   uint32_t                 indexCount;
   // Configure the target resolution when running dynamic subdivision level.
   // <= 0: disabled.
   // > 0: The subdivision level be chosen such that a single micro-triangle covers approximatley a dynamicSubdivisionScale *
   // dynamicSubdivisionScale texel area.
   float                    dynamicSubdivisionScale;
   // Rejection threshold [0,1]. Unless OMMs achive a rate of at least rejectionThreshold known states OMMs will be discarded
   // for the primitive. Use this to weed out "poor" OMMs.
   float                    rejectionThreshold;
   // The alpha cutoff value. texture > alphaCutoff ? Opaque : Transparent
   float                    alphaCutoff;
   // The global Format. May be overriden by the per-triangle subdivision level setting.
   ommFormat                format;
   // Use Formats to control format on a per triangle granularity. If Format is set to Format::INVALID the global setting will
   // be used instead.
   const ommFormat*         formats;
   // Determines how to promote mixed states
   ommUnknownStatePromotion unknownStatePromotion;
   // Micro triangle count is 4^N, where N is the subdivision level.
   // maxSubdivisionLevel level must be in range [0, 12].
   // When dynamicSubdivisionScale is enabled maxSubdivisionLevel is the max subdivision level allowed.
   // When dynamicSubdivisionScale is disabled maxSubdivisionLevel is the subdivision level applied uniformly to all
   // triangles.
   uint8_t                  maxSubdivisionLevel;
   ommBool                  enableSubdivisionLevelBuffer;
   // [optional] Use subdivisionLevels to control subdivision on a per triangle granularity.
   // +14 - reserved for future use.
   // 13 - use global value specified in 'subdivisionLevel.
   // [0,12] - per triangle subdivision level'
   const uint8_t*           subdivisionLevels;
} ommCpuBakeInputDesc;

inline ommCpuBakeInputDesc ommCpuBakeInputDescDefault()
{
   ommCpuBakeInputDesc v;
   v.bakeFlags                     = ommCpuBakeFlags_None;
   v.texture                       = 0;
   v.runtimeSamplerDesc            = ommSamplerDescDefault();
   v.alphaMode                     = ommAlphaMode_MAX_NUM;
   v.texCoordFormat                = ommTexCoordFormat_MAX_NUM;
   v.texCoords                     = NULL;
   v.texCoordStrideInBytes         = 0;
   v.indexFormat                   = ommIndexFormat_MAX_NUM;
   v.indexBuffer                   = NULL;
   v.indexCount                    = 0;
   v.dynamicSubdivisionScale       = 2;
   v.rejectionThreshold            = 0;
   v.alphaCutoff                   = 0.5f;
   v.format                        = ommFormat_OC1_4_State;
   v.formats                       = NULL;
   v.unknownStatePromotion         = ommUnknownStatePromotion_ForceOpaque;
   v.maxSubdivisionLevel           = 8;
   v.enableSubdivisionLevelBuffer  = 0;
   v.subdivisionLevels             = NULL;
   return v;
}

typedef struct ommCpuOpacityMicromapDesc
{
   // Byte offset into the opacity micromap map array.
   uint32_t offset;
   // Micro triangle count is 4^N, where N is the subdivision level.
   uint16_t subdivisionLevel;
   // OMM input format.
   uint16_t format;
} ommCpuOpacityMicromapDesc;

typedef struct ommCpuOpacityMicromapUsageCount
{
   // Number of OMMs with the specified subdivision level and format.
   uint32_t count;
   // Micro triangle count is 4^N, where N is the subdivision level.
   uint16_t subdivisionLevel;
   // OMM input format.
   uint16_t format;
} ommCpuOpacityMicromapUsageCount;

typedef struct ommCpuBakeResultDesc
{
   // Below is used as OMM array build input DX/VK.
   const void*                            arrayData;
   uint32_t                               arrayDataSize;
   const ommCpuOpacityMicromapDesc*       descArray;
   uint32_t                               descArrayCount;
   // The histogram of all omm data referenced by 'ommDescArray', can be used as 'pOMMUsageCounts' for the OMM build in D3D12
   const ommCpuOpacityMicromapUsageCount* descArrayHistogram;
   uint32_t                               descArrayHistogramCount;
   // Below is used for BLAS build input in DX/VK
   const void*                            indexBuffer;
   uint32_t                               indexCount;
   ommIndexFormat                         indexFormat;
   // Same as ommDescArrayHistogram but usage count equals the number of references by ommIndexBuffer. Can be used as
   // 'pOMMUsageCounts' for the BLAS OMM attachment in D3D12
   const ommCpuOpacityMicromapUsageCount* indexHistogram;
   uint32_t                               indexHistogramCount;
} ommCpuBakeResultDesc;

OMM_API ommResult ommCpuCreateTexture(ommBaker baker, const ommCpuTextureDesc* desc, ommCpuTexture* outTexture);

OMM_API ommResult ommCpuDestroyTexture(ommBaker baker, ommCpuTexture texture);

OMM_API ommResult ommCpuBake(ommBaker baker, const ommCpuBakeInputDesc* bakeInputDesc, ommCpuBakeResult* outBakeResult);

OMM_API ommResult ommCpuDestroyBakeResult(ommCpuBakeResult bakeResult);

OMM_API ommResult ommCpuGetBakeResultDesc(ommCpuBakeResult bakeResult, const ommCpuBakeResultDesc** desc);

typedef ommHandle ommGpuPipeline;

typedef enum ommGpuDescriptorType
{
   ommGpuDescriptorType_TextureRead,
   ommGpuDescriptorType_BufferRead,
   ommGpuDescriptorType_RawBufferRead,
   ommGpuDescriptorType_RawBufferWrite,
   ommGpuDescriptorType_MAX_NUM,
} ommGpuDescriptorType;

typedef enum ommGpuResourceType
{
   // 1-4 channels, any format.
   ommGpuResourceType_IN_ALPHA_TEXTURE,
   ommGpuResourceType_IN_TEXCOORD_BUFFER,
   ommGpuResourceType_IN_INDEX_BUFFER,
   // (Optional) R8, Values must be in range [-2, 12].
   // Positive values to enforce specific subdibision level for the primtive.
   // -1 to use global subdivision level.
   // -2 to use automatic subduvision level based on tunable texel-area heuristic
   ommGpuResourceType_IN_SUBDIVISION_LEVEL_BUFFER,
   // Used directly as argument for OMM build in DX/VK
   ommGpuResourceType_OUT_OMM_ARRAY_DATA,
   // Used directly as argument for OMM build in DX/VK
   ommGpuResourceType_OUT_OMM_DESC_ARRAY,
   // Used directly as argument for OMM build in DX/VK. (Read back to CPU to query memory requirements during OMM Array build)
   ommGpuResourceType_OUT_OMM_DESC_ARRAY_HISTOGRAM,
   // Used directly as argument for OMM build in DX/VK
   ommGpuResourceType_OUT_OMM_INDEX_BUFFER,
   // Used directly as argument for OMM build in DX/VK. (Read back to CPU to query memory requirements during OMM Blas build)
   ommGpuResourceType_OUT_OMM_INDEX_HISTOGRAM,
   // Read back the PostDispatchInfo struct containing the actual sizes of ARRAY_DATA and DESC_ARRAY.
   ommGpuResourceType_OUT_POST_DISPATCH_INFO,
   // Can be reused after baking
   ommGpuResourceType_TRANSIENT_POOL_BUFFER,
   // Initialize on startup. Read-only.
   ommGpuResourceType_STATIC_VERTEX_BUFFER,
   // Initialize on startup. Read-only.
   ommGpuResourceType_STATIC_INDEX_BUFFER,
   ommGpuResourceType_MAX_NUM,
} ommGpuResourceType;

typedef enum ommGpuPrimitiveTopology
{
   ommGpuPrimitiveTopology_TriangleList,
   ommGpuPrimitiveTopology_MAX_NUM,
} ommGpuPrimitiveTopology;

typedef enum ommGpuPipelineType
{
   ommGpuPipelineType_Compute,
   ommGpuPipelineType_Graphics,
   ommGpuPipelineType_MAX_NUM,
} ommGpuPipelineType;

typedef enum ommGpuDispatchType
{
   ommGpuDispatchType_Compute,
   ommGpuDispatchType_ComputeIndirect,
   ommGpuDispatchType_DrawIndexedIndirect,
   ommGpuDispatchType_BeginLabel,
   ommGpuDispatchType_EndLabel,
   ommGpuDispatchType_MAX_NUM,
} ommGpuDispatchType;

typedef enum ommGpuBufferFormat
{
   ommGpuBufferFormat_R32_UINT,
   ommGpuBufferFormat_MAX_NUM,
} ommGpuBufferFormat;

typedef enum ommGpuRasterCullMode
{
   ommGpuRasterCullMode_None,
} ommGpuRasterCullMode;

typedef enum ommGpuRenderAPI
{
   ommGpuRenderAPI_DX12,
   ommGpuRenderAPI_Vulkan,
   ommGpuRenderAPI_MAX_NUM,
} ommGpuRenderAPI;

typedef enum ommGpuScratchMemoryBudget
{
   ommGpuScratchMemoryBudget_Undefined,
   ommGpuScratchMemoryBudget_MB_4      = 4ull << 20ull,
   ommGpuScratchMemoryBudget_MB_32     = 32ull << 20ull,
   ommGpuScratchMemoryBudget_MB_64     = 64ull << 20ull,
   ommGpuScratchMemoryBudget_MB_128    = 128ull << 20ull,
   ommGpuScratchMemoryBudget_MB_256    = 256ull << 20ull,
   ommGpuScratchMemoryBudget_MB_512    = 512ull << 20ull,
   ommGpuScratchMemoryBudget_MB_1024   = 1024ull << 20ull,
   ommGpuScratchMemoryBudget_Default   = 256ull << 20ull,
} ommGpuScratchMemoryBudget;

typedef enum ommGpuBakeFlags
{
   // Either PerformSetup, PerformBake (or both simultaneously) must be set.
   ommGpuBakeFlags_Invalid                      = 0,

   // (Default) OUT_OMM_DESC_ARRAY_HISTOGRAM, OUT_OMM_INDEX_HISTOGRAM, OUT_OMM_INDEX_BUFFER, OUT_OMM_DESC_ARRAY and
   // OUT_POST_DISPATCH_INFO will be updated.
   ommGpuBakeFlags_PerformSetup                 = 1u << 0,

   // (Default) OUT_OMM_INDEX_HISTOGRAM, OUT_OMM_INDEX_BUFFER, OUT_OMM_ARRAY_DATA and OUT_POST_DISPATCH_INFO (if stats
   // enabled) will be updated. will be written to. If special indices are detected OUT_OMM_INDEX_BUFFER may also be modified.
   // If PerformBuild is not used with this flag, OUT_OMM_DESC_ARRAY_HISTOGRAM, OUT_OMM_INDEX_HISTOGRAM, OUT_OMM_INDEX_BUFFER,
   // OUT_OMM_DESC_ARRAY must contain valid data from a prior PerformSetup pass.
   ommGpuBakeFlags_PerformBake                  = 1u << 1,

   // Alias for (PerformSetup | PerformBake)
   ommGpuBakeFlags_PerformSetupAndBake          = 3u,

   // Baking will only be done using compute shaders and no gfx involvement (drawIndirect or graphics PSOs). (Beta)
   // Will become default mode in the future.
   // + Useful for async workloads
   // + Less memory hungry
   // + Faster baking on low texel ratio to micro-triangle ratio (=rasterizing small triangles)
   // - May looses efficency when resampling large triangles (tail-effect). Potential mitigation is to batch multiple bake
   // jobs. However this is generally not a big problem.
   ommGpuBakeFlags_ComputeOnly                  = 1u << 2,

   // Must be used together with EnablePostDispatchInfo. If set baking (PerformBake) will fill the stats data of
   // OUT_POST_DISPATCH_INFO.
   ommGpuBakeFlags_EnablePostDispatchInfoStats  = 1u << 3,

   // Will disable the use of special indices in case the OMM-state is uniform. Only set this flag for debug purposes.
   ommGpuBakeFlags_DisableSpecialIndices        = 1u << 4,

   // If texture coordinates are known to be unique tex cooord deduplication can be disabled to save processing time and free
   // up scratch memory.
   ommGpuBakeFlags_DisableTexCoordDeduplication = 1u << 5,

   // Force 32-bit indices in OUT_OMM_INDEX_BUFFER
   ommGpuBakeFlags_Force32BitIndices            = 1u << 6,

   // Use only for debug purposes. Level Line Intersection method is vastly superior in 4-state mode.
   ommGpuBakeFlags_DisableLevelLineIntersection = 1u << 7,

   // Slightly modifies the dispatch to aid frame capture debugging.
   ommGpuBakeFlags_EnableNsightDebugMode        = 1u << 8,
} ommGpuBakeFlags;
OMM_DEFINE_ENUM_FLAG_OPERATORS(ommGpuBakeFlags);

typedef struct ommGpuResource
{
   ommGpuDescriptorType stateNeeded;
   ommGpuResourceType   type;
   uint16_t             indexInPool;
   uint16_t             mipOffset;
   uint16_t             mipNum;
} ommGpuResource;

typedef struct ommGpuDescriptorRangeDesc
{
   ommGpuDescriptorType descriptorType;
   uint32_t             baseRegisterIndex;
   uint32_t             descriptorNum;
} ommGpuDescriptorRangeDesc;

typedef struct ommGpuBufferDesc
{
   size_t bufferSize;
} ommGpuBufferDesc;

typedef struct ommGpuShaderBytecode
{
   const void* data;
   size_t      size;
} ommGpuShaderBytecode;

typedef struct ommGpuComputePipelineDesc
{
   ommGpuShaderBytecode             computeShader;
   const char*                      shaderFileName;
   const char*                      shaderEntryPointName;
   const ommGpuDescriptorRangeDesc* descriptorRanges;
   uint32_t                         descriptorRangeNum;
   // if "true" all constant buffers share same "ConstantBufferDesc" description. if "false" this pipeline doesn't have a
   // constant buffer
   ommBool                          hasConstantData;
} ommGpuComputePipelineDesc;

typedef struct ommGpuGraphicsPipelineInputElementDesc
{
   const char*        semanticName;
   ommGpuBufferFormat format;
   uint32_t           inputSlot;
   uint32_t           semanticIndex;
   ommBool            isPerInstanced;
} ommGpuGraphicsPipelineInputElementDesc;

inline ommGpuGraphicsPipelineInputElementDesc ommGpuGraphicsPipelineInputElementDescDefault()
{
   ommGpuGraphicsPipelineInputElementDesc v;
   v.semanticName    = "POSITION";
   v.format          = ommGpuBufferFormat_R32_UINT;
   v.inputSlot       = 0;
   v.semanticIndex   = 0;
   v.isPerInstanced  = 0;
   return v;
}

typedef enum ommGpuGraphicsPipelineDescVersion
{
   ommGpuGraphicsPipelineDescVersion_VERSION = 2,
} ommGpuGraphicsPipelineDescVersion;

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
typedef struct ommGpuGraphicsPipelineDesc
{
   ommGpuShaderBytecode             vertexShader;
   const char*                      vertexShaderFileName;
   const char*                      vertexShaderEntryPointName;
   ommGpuShaderBytecode             geometryShader;
   const char*                      geometryShaderFileName;
   const char*                      geometryShaderEntryPointName;
   ommGpuShaderBytecode             pixelShader;
   const char*                      pixelShaderFileName;
   const char*                      pixelShaderEntryPointName;
   ommBool                          conservativeRasterization;
   const ommGpuDescriptorRangeDesc* descriptorRanges;
   uint32_t                         descriptorRangeNum;
   // if NumRenderTargets = 0 a null RTV is implied.
   uint32_t                         numRenderTargets;
   // if "true" all constant buffers share same "ConstantBufferDesc" description. if "false" this pipeline doesn't have a
   // constant buffer
   ommBool                          hasConstantData;
} ommGpuGraphicsPipelineDesc;

typedef struct ommGpuPipelineDesc
{
   ommGpuPipelineType type;

   union
   {
      ommGpuComputePipelineDesc  compute;
      ommGpuGraphicsPipelineDesc graphics;
   };
} ommGpuPipelineDesc;

typedef struct ommGpuDescriptorSetDesc
{
   uint32_t constantBufferMaxNum;
   uint32_t storageBufferMaxNum;
   uint32_t descriptorRangeMaxNumPerPipeline;
} ommGpuDescriptorSetDesc;

typedef struct ommGpuConstantBufferDesc
{
   uint32_t registerIndex;
   uint32_t maxDataSize;
} ommGpuConstantBufferDesc;

typedef struct ommGpuViewport
{
   float minWidth;
   float minHeight;
   float maxWidth;
   float maxHeight;
} ommGpuViewport;

typedef struct ommGpuComputeDesc
{
   const char*           name;
   // concatenated resources for all "DescriptorRangeDesc" descriptions in DenoiserDesc::pipelines[ pipelineIndex ]
   const ommGpuResource* resources;
   uint32_t              resourceNum;
   // "root constants" in DX12
   const uint8_t*        localConstantBufferData;
   uint32_t              localConstantBufferDataSize;
   uint16_t              pipelineIndex;
   uint32_t              gridWidth;
   uint32_t              gridHeight;
} ommGpuComputeDesc;

typedef struct ommGpuComputeIndirectDesc
{
   const char*           name;
   // concatenated resources for all "DescriptorRangeDesc" descriptions in DenoiserDesc::pipelines[ pipelineIndex ]
   const ommGpuResource* resources;
   uint32_t              resourceNum;
   // "root constants" in DX12
   const uint8_t*        localConstantBufferData;
   uint32_t              localConstantBufferDataSize;
   uint16_t              pipelineIndex;
   ommGpuResource        indirectArg;
   size_t                indirectArgByteOffset;
} ommGpuComputeIndirectDesc;

typedef struct ommGpuDrawIndexedIndirectDesc
{
   const char*           name;
   // concatenated resources for all "DescriptorRangeDesc" descriptions in DenoiserDesc::pipelines[ pipelineIndex ]
   const ommGpuResource* resources;
   uint32_t              resourceNum;
   // "root constants" in DX12
   const uint8_t*        localConstantBufferData;
   uint32_t              localConstantBufferDataSize;
   uint16_t              pipelineIndex;
   ommGpuResource        indirectArg;
   size_t                indirectArgByteOffset;
   ommGpuViewport        viewport;
   ommGpuResource        indexBuffer;
   uint32_t              indexBufferOffset;
   ommGpuResource        vertexBuffer;
   uint32_t              vertexBufferOffset;
} ommGpuDrawIndexedIndirectDesc;

typedef struct ommGpuBeginLabelDesc
{
   const char* debugName;
} ommGpuBeginLabelDesc;

typedef struct ommGpuDispatchDesc
{
   ommGpuDispatchType type;

   union
   {
      ommGpuComputeDesc             compute;
      ommGpuComputeIndirectDesc     computeIndirect;
      ommGpuDrawIndexedIndirectDesc drawIndexedIndirect;
      ommGpuBeginLabelDesc          beginLabel;
   };
} ommGpuDispatchDesc;

typedef struct ommGpuStaticSamplerDesc
{
   ommSamplerDesc desc;
   uint32_t       registerIndex;
} ommGpuStaticSamplerDesc;

typedef struct ommGpuSPIRVBindingOffsets
{
   uint32_t samplerOffset;
   uint32_t textureOffset;
   uint32_t constantBufferOffset;
   uint32_t storageTextureAndBufferOffset;
} ommGpuSPIRVBindingOffsets;

typedef struct ommGpuPipelineConfigDesc
{
   // API is required to make sure indirect buffers are written to in suitable format
   ommGpuRenderAPI renderAPI;
} ommGpuPipelineConfigDesc;

inline ommGpuPipelineConfigDesc ommGpuPipelineConfigDescDefault()
{
   ommGpuPipelineConfigDesc v;
   v.renderAPI  = ommGpuRenderAPI_DX12;
   return v;
}

// Note: sizes may return size zero, this means the buffer will not be used in the dispatch.
typedef struct ommGpuPreDispatchInfo
{
   // Format of outOmmIndexBuffer
   ommIndexFormat outOmmIndexBufferFormat;
   uint32_t       outOmmIndexCount;
   // Min required size of OUT_OMM_ARRAY_DATA. GetBakeInfo returns most conservative estimation while less conservative number
   // can be obtained via BakePrepass
   uint32_t       outOmmArraySizeInBytes;
   // Min required size of OUT_OMM_DESC_ARRAY. GetBakeInfo returns most conservative estimation while less conservative number
   // can be obtained via BakePrepass
   uint32_t       outOmmDescSizeInBytes;
   // Min required size of OUT_OMM_INDEX_BUFFER
   uint32_t       outOmmIndexBufferSizeInBytes;
   // Min required size of OUT_OMM_ARRAY_HISTOGRAM
   uint32_t       outOmmArrayHistogramSizeInBytes;
   // Min required size of OUT_OMM_INDEX_HISTOGRAM
   uint32_t       outOmmIndexHistogramSizeInBytes;
   // Min required size of OUT_POST_DISPATCH_INFO
   uint32_t       outOmmPostDispatchInfoSizeInBytes;
   // Min required sizes of TRANSIENT_POOL_BUFFERs
   uint32_t       transientPoolBufferSizeInBytes[8];
   uint32_t       numTransientPoolBuffers;
} ommGpuPreDispatchInfo;

inline ommGpuPreDispatchInfo ommGpuPreDispatchInfoDefault()
{
   ommGpuPreDispatchInfo v;
   v.outOmmIndexBufferFormat            = ommIndexFormat_MAX_NUM;
   v.outOmmIndexCount                   = 0xFFFFFFFF;
   v.outOmmArraySizeInBytes             = 0xFFFFFFFF;
   v.outOmmDescSizeInBytes              = 0xFFFFFFFF;
   v.outOmmIndexBufferSizeInBytes       = 0xFFFFFFFF;
   v.outOmmArrayHistogramSizeInBytes    = 0xFFFFFFFF;
   v.outOmmIndexHistogramSizeInBytes    = 0xFFFFFFFF;
   v.outOmmPostDispatchInfoSizeInBytes  = 0xFFFFFFFF;
   v.numTransientPoolBuffers            = 0;
   return v;
}

typedef struct ommGpuDispatchConfigDesc
{
   ommGpuBakeFlags           bakeFlags;
   // RuntimeSamplerDesc describes the texture sampler that will be used in the runtime alpha test shader code.
   ommSamplerDesc            runtimeSamplerDesc;
   ommAlphaMode              alphaMode;
   //  The texture dimensions of IN_ALPHA_TEXTURE
   uint32_t                  alphaTextureWidth;
   //  The texture dimensions of IN_ALPHA_TEXTURE
   uint32_t                  alphaTextureHeight;
   // The channel in IN_ALPHA_TEXTURE containing opacity values
   uint32_t                  alphaTextureChannel;
   ommTexCoordFormat         texCoordFormat;
   uint32_t                  texCoordOffsetInBytes;
   uint32_t                  texCoordStrideInBytes;
   ommIndexFormat            indexFormat;
   // The actual number of indices can be lower.
   uint32_t                  indexCount;
   // If zero packed aligment is assumed.
   uint32_t                  indexStrideInBytes;
   // The alpha cutoff value. texture > alphaCutoff ? Opaque : Transparent.
   float                     alphaCutoff;
   // Configure the target resolution when running dynamic subdivision level. <= 0: disabled. > 0: The subdivision level be
   // chosen such that a single micro-triangle covers approximatley a dynamicSubdivisionScale * dynamicSubdivisionScale texel
   // area.
   float                     dynamicSubdivisionScale;
   // The global Format. May be overriden by the per-triangle config.
   ommFormat                 globalFormat;
   uint8_t                   maxSubdivisionLevel;
   ommBool                   enableSubdivisionLevelBuffer;
   // The SDK will try to limit the omm array size of PreDispatchInfo::outOmmArraySizeInBytes and
   // PostDispatchInfo::outOmmArraySizeInBytes.
   // Currently a greedy algorithm is implemented with a first come-first serve order.
   // The SDK may (or may not) apply more sophisticated heuristics in the future.
   // If no memory is available to allocate an OMM Array Block the state will default to Unknown Opaque (ignoring any bake
   // flags do disable special indices).
   uint32_t                  maxOutOmmArraySize;
   // Target scratch memory budget, The SDK will try adjust the sum of the transient pool buffers to match this value. Higher
   // budget more efficiently executes the baking operation. May return INSUFFICIENT_SCRATCH_MEMORY if set too low.
   ommGpuScratchMemoryBudget maxScratchMemorySize;
} ommGpuDispatchConfigDesc;

inline ommGpuDispatchConfigDesc ommGpuDispatchConfigDescDefault()
{
   ommGpuDispatchConfigDesc v;
   v.bakeFlags                     = ommGpuBakeFlags_PerformSetupAndBake;
   v.runtimeSamplerDesc            = ommSamplerDescDefault();
   v.alphaMode                     = ommAlphaMode_MAX_NUM;
   v.alphaTextureWidth             = 0;
   v.alphaTextureHeight            = 0;
   v.alphaTextureChannel           = 3;
   v.texCoordFormat                = ommTexCoordFormat_MAX_NUM;
   v.texCoordOffsetInBytes         = 0;
   v.texCoordStrideInBytes         = 0;
   v.indexFormat                   = ommIndexFormat_MAX_NUM;
   v.indexCount                    = 0;
   v.indexStrideInBytes            = 0;
   v.alphaCutoff                   = 0.5f;
   v.dynamicSubdivisionScale       = 2;
   v.globalFormat                  = ommFormat_OC1_4_State;
   v.maxSubdivisionLevel           = 8;
   v.enableSubdivisionLevelBuffer  = 0;
   v.maxOutOmmArraySize            = 0xFFFFFFFF;
   v.maxScratchMemorySize          = ommGpuScratchMemoryBudget_Default;
   return v;
}

typedef struct ommGpuPipelineInfoDesc
{
   ommGpuSPIRVBindingOffsets      spirvBindingOffsets;
   const ommGpuPipelineDesc*      pipelines;
   uint32_t                       pipelineNum;
   ommGpuConstantBufferDesc       globalConstantBufferDesc;
   ommGpuConstantBufferDesc       localConstantBufferDesc;
   ommGpuDescriptorSetDesc        descriptorSetDesc;
   const ommGpuStaticSamplerDesc* staticSamplers;
   uint32_t                       staticSamplersNum;
} ommGpuPipelineInfoDesc;

// Format of OUT_POST_DISPATCH_INFO
typedef struct ommGpuPostDispatchInfo
{
   uint32_t outOmmArraySizeInBytes;
   uint32_t outOmmDescSizeInBytes;
   // Will be set if EnablePostDispatchInfoStats is set.
   uint32_t outStatsTotalOpaqueCount;
   // Will be set if EnablePostDispatchInfoStats is set.
   uint32_t outStatsTotalTransparentCount;
   // Will be set if EnablePostDispatchInfoStats is set.
   uint32_t outStatsTotalUnknownCount;
   // Will be set if EnablePostDispatchInfoStats is set.
   uint32_t outStatsTotalFullyOpaqueCount;
   // Will be set if EnablePostDispatchInfoStats is set.
   uint32_t outStatsTotalFullyTransparentCount;
   // Will be set if EnablePostDispatchInfoStats is set.
   uint32_t outStatsTotalFullyStatsUnknownCount;
} ommGpuPostDispatchInfo;

typedef struct ommGpuDispatchChain
{
   const ommGpuDispatchDesc* dispatches;
   uint32_t                  numDispatches;
   const uint8_t*            globalCBufferData;
   uint32_t                  globalCBufferDataSize;
} ommGpuDispatchChain;

// Global immutable resources. These contain the static immutable resources being shared acroess all bake calls.  Currently
// it's the specific IB and VB that represents a tesselated triangle arranged in bird curve order, for different
// subdivision levels.
OMM_API ommResult ommGpuGetStaticResourceData(ommGpuResourceType resource, uint8_t* data, size_t* outByteSize);

OMM_API ommResult ommGpuCreatePipeline(ommBaker baker, const ommGpuPipelineConfigDesc* pipelineCfg, ommGpuPipeline* outPipeline);

OMM_API ommResult ommGpuDestroyPipeline(ommBaker baker, ommGpuPipeline pipeline);

// Return the required pipelines. Does not depend on per-dispatch settings.
OMM_API ommResult ommGpuGetPipelineDesc(ommGpuPipeline pipeline, const ommGpuPipelineInfoDesc** outPipelineDesc);

// Returns the scratch and output memory requirements of the baking operation.
OMM_API ommResult ommGpuGetPreDispatchInfo(ommGpuPipeline pipeline, const ommGpuDispatchConfigDesc* config, ommGpuPreDispatchInfo* outPreDispatchInfo);

// Returns the dispatch order to perform the baking operation. Once complete the OUT_OMM_* resources will be written to and
// can be consumed by the application.
OMM_API ommResult ommGpuDispatch(ommGpuPipeline pipeline, const ommGpuDispatchConfigDesc* config, const ommGpuDispatchChain** outDispatchDesc);

typedef struct ommDebugSaveImagesDesc
{
   const char* path;
   const char* filePostfix;
   // The default behaviour is to dump the entire alpha texture with the OMM-triangle in it. Enabling detailedCutout will
   // generate cropped version zoomed in on the OMM, and supersampled for detailed analysis
   ommBool     detailedCutout;
   // Only dump index 0.
   ommBool     dumpOnlyFirstOMM;
   // Will draw unknown transparent and unknown opaque in the same color.
   ommBool     monochromeUnknowns;
   // true:Will draw all primitives to the same file. false: will draw each primitive separatley.
   ommBool     oneFile;
} ommDebugSaveImagesDesc;

inline ommDebugSaveImagesDesc ommDebugSaveImagesDescDefault()
{
   ommDebugSaveImagesDesc v;
   v.path                = "";
   v.filePostfix         = "";
   v.detailedCutout      = 0;
   v.dumpOnlyFirstOMM    = 0;
   v.monochromeUnknowns  = 0;
   v.oneFile             = 0;
   return v;
}

// Walk each primitive and dumps the corresponding OMM overlay to the alpha textures.
OMM_API ommResult ommDebugSaveAsImages(ommBaker baker, const ommCpuBakeInputDesc* bakeInputDesc, const ommCpuBakeResultDesc* res, const ommDebugSaveImagesDesc* desc);

typedef struct ommDebugStats
{
   uint64_t totalOpaque;
   uint64_t totalTransparent;
   uint64_t totalUnknownTransparent;
   uint64_t totalUnknownOpaque;
   uint32_t totalFullyOpaque;
   uint32_t totalFullyTransparent;
   uint32_t totalFullyUnknownOpaque;
   uint32_t totalFullyUnknownTransparent;
} ommDebugStats;

inline ommDebugStats ommDebugStatsDefault()
{
   ommDebugStats v;
   v.totalOpaque                   = 0;
   v.totalTransparent              = 0;
   v.totalUnknownTransparent       = 0;
   v.totalUnknownOpaque            = 0;
   v.totalFullyOpaque              = 0;
   v.totalFullyTransparent         = 0;
   v.totalFullyUnknownOpaque       = 0;
   v.totalFullyUnknownTransparent  = 0;
   return v;
}

OMM_API ommResult ommDebugGetStats(ommBaker baker, const ommCpuBakeResultDesc* res, ommDebugStats* out);
#endif // #ifndef INCLUDE_OMM_SDK_C
