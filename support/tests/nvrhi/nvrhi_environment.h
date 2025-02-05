/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once
#include <gtest/gtest.h>
#include "nvrhi_wrapper.h"

class nvrhi_environment : public ::testing::Environment
{
public:
	virtual ~nvrhi_environment() = default;

	virtual void SetUp();

	virtual void TearDown();

	NVRHIContext* GetContext() { return m_context.get(); }

private:
	std::unique_ptr<NVRHIContext> m_context;
};

extern nvrhi_environment* g_nvrhiEnvironment;
