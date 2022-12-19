/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#if OMM_ENABLE_GPU_TESTS
#include "nvrhi_environment.h"
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);

#if OMM_ENABLE_GPU_TESTS
	g_nvrhiEnvironment = new nvrhi_environment();
	::testing::AddGlobalTestEnvironment(g_nvrhiEnvironment);
#endif

	return RUN_ALL_TESTS();
}