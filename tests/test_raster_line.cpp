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

class RasterLineTest : public ::testing::TestWithParam<std::tuple<omm::Line, int2>> {
protected:
	omm::Line _line;
	int2 _size;
	void SetUp() override {
		_line = std::get<0>(GetParam());
		_size = std::get<1>(GetParam());

		// EXPECT_EQ(omm::GetWinding(_line), omm::WindingOrder::CCW);
	}

	void TearDown() override {
	}

	const char* GetRasterModeName(omm::RasterMode mode) {
		if (omm::RasterMode::UnderConservative == mode)
		{
			return "UnderConservative";
		}
		if (omm::RasterMode::OverConservative == mode)
		{
			return "OverConservative";
		}
		else // if (omm::RasterMode::Default == mode)
		{
			return "Default";
		}
	}

	void Run(int2 size, omm::RasterMode mode) {

		ImageRGB image(size, { 1, 128, 5 });

		int one = 1;
		int checkerSize = 64;
		
		auto KernelCBFill = [&image, checkerSize](int2 idx, float3* bc, void* ) {
			if (idx.x >= image.GetSize().x)
				return;
			if (idx.y >= image.GetSize().y)
				return;
			if (idx.x < 0)
				return;
			if (idx.y < 0)
				return;

			float2 p = { checkerSize * float(idx.x / checkerSize + 0.5) / image.GetWidth() , checkerSize * float(idx.y / checkerSize + 0.5) / image.GetHeight() };
			uchar3 val;
			if ((idx.x / checkerSize) % 2 != (idx.y / checkerSize) % 2)
				val = { 0, 0, 0 };
			else
				val = { 64, 64, 64 };
			image.Store(idx, val);
		};

		struct Params {
			int checkerSize = 1;
			uchar3 fillColor = { 0,0 ,0 };
		};

		auto LineFill = [&image, size](int2 idx, void* ctx) {
			Params* p = (Params*)ctx;

			auto IsInRange = [&image](int2 idx)->bool {
				if (idx.x >= image.GetSize().x)
					return false;
				if (idx.y >= image.GetSize().y)
					return false;
				if (idx.x < 0)
					return false;
				if (idx.y < 0)
					return false;

				return true;
				};

			for (int y = 0; y < p->checkerSize; ++y) {
				for (int x = 0; x < p->checkerSize; ++x) {

					int2 dst = p->checkerSize * idx + int2{ x, y };
					if (!IsInRange(dst))
						continue;

					uchar3 val = image.Load(dst);
					val += p->fillColor;
					image.Store(dst, p->fillColor);
				}
			}
		};

		// "Fullscreen" pass.
		omm::RasterizeParallel(omm::Triangle({ 0.f, -1.f }, { 0.f, 1.f }, { 2.f, 1.f }), size, KernelCBFill);

		// "Triangle pass.
		if (omm::RasterMode::OverConservative == mode)
		{
			{
				Params p;
				p.checkerSize = checkerSize;
				p.fillColor = uchar3(128, 0, 0);
				omm::RasterizeLineConservativeImpl(_line, size / checkerSize, LineFill, &p);
			}

			{
				Params p;
				p.checkerSize = 1;
				p.fillColor = uchar3(0, 128, 0);
				omm::RasterizeLine(_line, size, LineFill, &p);
			}
		}
		else if (omm::RasterMode::Default == mode)
		{
			{
				Params p;
				p.checkerSize = checkerSize;
				p.fillColor = uchar3(128, 0, 0);
				omm::RasterizeLine(_line, size / checkerSize, LineFill, &p);
			}
			{
				Params p;
				p.checkerSize = 1;
				p.fillColor = uchar3(0, 128, 0);
				omm::RasterizeLine(_line, size, LineFill, &p);
			}
		}

		std::string name = ::testing::UnitTest::GetInstance()->current_test_suite()->name();
		std::replace(name.begin(), name.end(), '/', '_');

		std::string fileName = name + GetRasterModeName(mode) + std::to_string(size.x) + "x" + std::to_string(size.y) + ".png";
		SaveImageToFile("RasterTestOutput", fileName.c_str(), image);
	}
};

TEST_P(RasterLineTest, Rasterize) {
	Run(_size, omm::RasterMode::Default);
}

TEST_P(RasterLineTest, RasterizeConservative) {
	Run(_size, omm::RasterMode::OverConservative);
}

TEST_P(RasterLineTest, RasterizeSmall) {
	Run({ _size.x / 2, _size.y / 2, }, omm::RasterMode::Default);
}

TEST_P(RasterLineTest, RasterizeConservativeSmall) {
	Run({ _size.x / 2, _size.y / 2, }, omm::RasterMode::OverConservative);
}

TEST_P(RasterLineTest, RasterizeLarge) {
	Run({ _size.x * 2, _size.y * 2, }, omm::RasterMode::Default);
}

TEST_P(RasterLineTest, RasterizeConservativeLarge) {
	Run({ _size.x * 2, _size.y * 2, }, omm::RasterMode::OverConservative);
}

TEST_P(RasterLineTest, RasterizeConservativeSuperLarge) {
	Run({ _size.x * 4, _size.y * 4, }, omm::RasterMode::OverConservative);
}

INSTANTIATE_TEST_SUITE_P(
	RasterLine_Low,
	RasterLineTest,
	::testing::Values(
		std::make_tuple<omm::Line, int2>(omm::Line({0.2f, 0.2f}, {0.7f, 0.5f}), {1024, 1024})
	));

INSTANTIATE_TEST_SUITE_P(
	RasterLine_Diagonal,
	RasterLineTest,
	::testing::Values(
		std::make_tuple<omm::Line, int2>(omm::Line({ 0.01f, 0.01f }, { 0.99f, 0.9f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterLine_LowCenter,
	RasterLineTest,
	::testing::Values(
		std::make_tuple<omm::Line, int2>(omm::Line({ 0.2f, 0.2f }, { 0.5f, 0.5f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterLine_High,
	RasterLineTest,
	::testing::Values(
		std::make_tuple<omm::Line, int2>(omm::Line({ 0.1f, 0.9f }, { 0.7f, 0.2f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterLine_HighCenter,
	RasterLineTest,
	::testing::Values(
		std::make_tuple<omm::Line, int2>(omm::Line({ 0.1f, 0.9f }, { 0.5f, 0.2f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterLine_Horizontal,
	RasterLineTest,
	::testing::Values(
		std::make_tuple<omm::Line, int2>(omm::Line({ 0.2f, 0.5f }, { 0.5f, 0.5f }), { 1024, 1024 })
	));

INSTANTIATE_TEST_SUITE_P(
	RasterLine_Vertical,
	RasterLineTest,
	::testing::Values(
		std::make_tuple<omm::Line, int2>(omm::Line({ 0.5f, 0.5f }, { 0.5f, 0.1f }), { 1024, 1024 })
	));

}  // namespace