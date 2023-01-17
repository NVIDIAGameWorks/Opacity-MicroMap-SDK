/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "nvrhi_environment.h"

nvrhi_environment* g_nvrhiEnvironment = nullptr;

void nvrhi_environment::SetUp()
{
	NVRHIContext::InitParams params;
	params.api = nvrhi::GraphicsAPI::D3D12;
	params.enableDebugRuntime = true;
	params.enableNvrhiValidationLayer = true;
	m_context = NVRHIContext::Init(params);
	EXPECT_NE(m_context.get(), nullptr);
}

void nvrhi_environment::TearDown()
{
	m_context = nullptr;
}