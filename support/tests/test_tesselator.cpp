/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>

#include "util/image.h"
#include <shared/cpu_raster.h>
#include <omp.h>
#include <algorithm>
#include <filesystem>

namespace {

class TesselatorTest : public ::testing::TestWithParam<std::tuple<omm::Triangle, int2>> {
protected:
	omm::Triangle _triangle;
	int2 _size;

	void SetUp() override {
		_triangle = std::get<0>(GetParam());
		_size = std::get<1>(GetParam());
	}

	void TearDown() override {
	}

	void Run(int2 size, int numSubdivisionLevels, bool conservative) {


		ImageRGB image(size, { 1, 128, 5 });

		int superSampleScale = 16;
		FillWithCheckerboardRGB(image, superSampleScale);

		uint32_t numMicroTris = omm::bird::GetNumMicroTriangles(numSubdivisionLevels);
		for (uint32_t i = 0; i < numMicroTris; ++i) {
			omm::Triangle t = omm::bird::GetMicroTriangle(_triangle, i, numSubdivisionLevels);

			uint8_t color = uint8_t(255 * float(i) / numMicroTris);
			Rasterize(image, t, conservative, { color, color, color });
		}

		std::string suite = ::testing::UnitTest::GetInstance()->current_test_suite()->name();
		std::replace(suite.begin(), suite.end(), '/', '_');

		std::string test = ::testing::UnitTest::GetInstance()->current_test_info()->name();
		std::replace(test.begin(), test.end(), '/', '_');

		std::string name = suite + test;

		std::string fileName = name + std::to_string(size.x) + "x" + std::to_string(size.y) + (conservative ? "_cons_" : "") +".png";
		SaveImageToFile("TesselatorTestOutput", fileName.c_str(), image);
	}
};

TEST_P(TesselatorTest, TesselateCons0) {
	Run(_size, 0, true);
}

TEST_P(TesselatorTest, Tesselate0) {
	Run(_size, 0, false);
}

TEST_P(TesselatorTest, TesselateCons1) {
	Run(_size, 1, true);
}

TEST_P(TesselatorTest, Tesselate1) {
	Run(_size, 1, false);
}

TEST_P(TesselatorTest, TesselateCons2) {
	Run(_size, 2, true);
}

TEST_P(TesselatorTest, Tesselate2) {
	Run(_size, 2, false);
}

TEST_P(TesselatorTest, TesselateCons3) {
	Run(_size, 3, true);
}

TEST_P(TesselatorTest, Tesselate3) {
	Run(_size, 3, false);
}

TEST_P(TesselatorTest, TesselateCons4) {
	Run(_size, 4, true);
}

TEST_P(TesselatorTest, Tesselate4) {
	Run(_size, 4, false);
}

INSTANTIATE_TEST_SUITE_P(
	Tesselator0,
	TesselatorTest,
	::testing::Values(
		std::make_tuple<omm::Triangle, int2>(omm::Triangle({ 0.2f, 0.1f }, { 0.9f, 0.9f }, { 0.1f, 0.9f }), { 1024, 1024 })
	));

}  // namespace