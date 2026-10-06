/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#include "util/bit_tricks.h"

namespace {

	TEST(BitFunc, CountTrailingZeros) {
		EXPECT_EQ(omm::ctz(0u), 32u);
		for (uint32_t bit = 0; bit < 32; ++bit) {
			EXPECT_EQ(omm::ctz(1u << bit), bit);
			EXPECT_EQ(omm::ctz(0xffffffffu << bit), bit);
			const uint2 result = omm::ctz(uint2(1u << bit, 0u));
			EXPECT_EQ(result.x, bit);
			EXPECT_EQ(result.y, 32u);
		}
		const uint2 result = omm::ctz(uint2(0u, 0x80000000u));
		EXPECT_EQ(result.x, 32u);
		EXPECT_EQ(result.y, 31u);
	}

	TEST(BitFunc, MortonOrder) {
		EXPECT_EQ(omm::xy_to_morton(0xffffu, 0xffffu), 0xffffffffu);
		for (uint32_t bit = 0; bit < 16; ++bit) {
			EXPECT_EQ(omm::xy_to_morton(1u << bit, 0u), 1u << (2 * bit));
			EXPECT_EQ(omm::xy_to_morton(0u, 1u << bit), 1u << (2 * bit + 1));
		}

		int N = 1024;
		int M = 1024;

		for (int j = 0; j < M; ++j) {
			for (int i = 0; i < N; ++i) {
				EXPECT_EQ(omm::xy_to_morton(i, j), omm::xy_to_morton_sw(i,j));
			}
		}
	}

}  // namespace