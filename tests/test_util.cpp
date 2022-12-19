/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#include <shared/bit_tricks.h>

namespace {

	TEST(BitFunc, MortonOrder) {

		int N = 1024;
		int M = 1024;

		for (int j = 0; j < M; ++j) {
			for (int i = 0; i < N; ++i) {
				EXPECT_EQ(omm::xy_to_morton(i, j), omm::xy_to_morton_sw(i,j));
			}
		}
	}

}  // namespace