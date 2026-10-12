/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#include <limits>
#include "util/bit_tricks.h"
#include "util/geometry.h"

namespace {

	void CheckUvValidity(const float& value, bool expectedInvalid) {
		volatile float runtimeValue = value;
		for (uint32_t vertex = 0; vertex < 3; ++vertex) {
			for (uint32_t component = 0; component < 2; ++component) {
				float2 uv[] = { { 0.f, 0.f }, { 1.f, 0.f }, { 0.f, 1.f } };
				uv[vertex][component] = runtimeValue;
				EXPECT_EQ(omm::IsInvalid(uv[0], uv[1], uv[2]), expectedInvalid)
					<< "value=" << value << ", vertex=" << vertex << ", component=" << component;
				const uint32_t indices[] = { 0, 1, 2 };
				const omm::Triangle triangle = omm::FetchUVTriangle(uv, sizeof(float2), ommTexCoordFormat_UV32_FLOAT, indices);
				EXPECT_EQ(triangle.GetIsInvalid(), expectedInvalid)
					<< "value=" << value << ", vertex=" << vertex << ", component=" << component;
			}
		}
	}

	TEST(GeometryValidity, RejectsNonFiniteUvs) {
		constexpr float values[] = {
			std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
			std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN(),
			std::numeric_limits<float>::signaling_NaN(), -std::numeric_limits<float>::signaling_NaN(),
		};
		for (const float& value : values)
			CheckUvValidity(value, true);
	}

	TEST(GeometryValidity, AcceptsFiniteUvs) {
		constexpr float largestSubnormal = std::numeric_limits<float>::min() - std::numeric_limits<float>::denorm_min();
		constexpr float values[] = {
			0.f, -0.f,
			std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(),
			largestSubnormal, -largestSubnormal,
			std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(),
			1.f, -1.f,
			std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest(),
		};
		for (const float& value : values)
			CheckUvValidity(value, false);
	}

	TEST(GeometryValidity, ClassifiesDegenerateUvs) {
		struct Case {
			const char* name;
			float2 uv[3];
			bool expectedDegenerate;
		};
		const Case cases[] = {
			{ "vertical", { {0.2f, 0.f}, {0.2f, 0.437582970f}, {0.2f, 0.218791485f} }, true },
			{ "horizontal", { {0.f, 0.2f}, {0.437582970f, 0.2f}, {0.218791485f, 0.2f} }, true },
			{ "diagonal", { {0.f, 0.f}, {0.2f, 0.2f}, {0.4f, 0.4f} }, true },
			{ "point", { {0.2f, 0.2f}, {0.2f, 0.2f}, {0.2f, 0.2f} }, true },
			{ "below threshold", { {0.f, 0.f}, {1.f, 0.f}, {0.f, 1e-9f} }, true },
			{ "above threshold", { {0.f, 0.f}, {1.f, 0.f}, {0.f, 1e-8f} }, false },
			{ "triangle", { {0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f} }, false },
		};
		const uint32_t permutations[][3] = { {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0} };
		for (const Case& testCase : cases) {
			SCOPED_TRACE(testCase.name);
			float2 uv[3];
			for (uint32_t vertex = 0; vertex < 3; ++vertex) {
				for (uint32_t component = 0; component < 2; ++component) {
					volatile float value = testCase.uv[vertex][component];
					uv[vertex][component] = value;
				}
			}
			for (const auto& indices : permutations) {
				const omm::Triangle triangle(uv[indices[0]], uv[indices[1]], uv[indices[2]]);
				EXPECT_EQ(omm::IsDegenerate(uv[indices[0]], uv[indices[1]], uv[indices[2]]), testCase.expectedDegenerate);
				EXPECT_EQ(triangle.GetIsDegenerate(), testCase.expectedDegenerate);
			}
		}
	}

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