/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#include <cstdio>
#if OMM_ENABLE_GPU_TESTS
#include "nvrhi/nvrhi_environment.h"
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);

#if OMM_ENABLE_GPU_TESTS
	std::printf("[ OMM GPU  ] Precise math: %s\n",
		OMM_ENABLE_PRECISE_GPU_MATH ? "ON (-Gis)" : "OFF");
	std::printf("[ OMM GPU  ] Shader optimizations: %s\n",
		OMM_DISABLE_GPU_SHADER_OPTIMIZATIONS ? "OFF (-Od)" : "ON");
	std::printf("[ OMM GPU  ] Reference mode: %s\n",
		OMM_GPU_REFERENCE_MODE ? "ON (software sqrt, division, and sampling)" : "OFF (native math and sampling)");
	std::printf("[ OMM GPU  ] Bake tests: %s\n",
		OMM_ENABLE_PRECISE_GPU_MATH ? "COMPUTE ONLY" : "COMPUTE AND GRAPHICS");
	std::printf("[ OMM GPU  ] Compute references: SOFTWARE\n");
	g_nvrhiEnvironment = new nvrhi_environment();
	::testing::AddGlobalTestEnvironment(g_nvrhiEnvironment);
#endif

	return RUN_ALL_TESTS();
}
