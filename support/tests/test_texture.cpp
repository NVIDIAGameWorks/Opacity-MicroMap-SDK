/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <gtest/gtest.h>
#include "util/texture.h"
#include <omm.h>

namespace {

	const char* GetName(omm::TextureAddressMode mode) {
		switch (mode) {
		case omm::TextureAddressMode::Wrap:
			return "Wrap";
		case omm::TextureAddressMode::Mirror:
			return "Mirror";
		case omm::TextureAddressMode::Clamp:
			return "Clamp";
		case omm::TextureAddressMode::Border:
			return "Border";
		case omm::TextureAddressMode::MirrorOnce:
			return "MirrorOnce";
		default:
			return "Unknown";
		}
	}

	void TexCoordTest(omm::TextureAddressMode mode, int2 texCoord, int2 size, int2 expectedValue) {
		int2 modifiedTex = omm::GetTexCoord(mode, omm::isPow2(size.x) && omm::isPow2(size.y), texCoord, size, omm::ctz(size));
		EXPECT_EQ(modifiedTex, expectedValue) << "Input:[" << texCoord.x << "," << texCoord.y << "],Expected:[" << expectedValue.x << "," << expectedValue.y
			<< "],Was:[" << modifiedTex.x << "," << modifiedTex.y << "] Mode:" << GetName(mode);
	}

	TEST(GetTexCoord, Wrap) {
		//													[Mode, TexCoord, Size, Expected]
		TexCoordTest(omm::TextureAddressMode::Wrap, { 512, 512 },	{ 1024, 1024 }, { 512, 512 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 0, 512 },		{ 1024, 1024 }, { 0, 512 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 0, 0 },		{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -1, -1 },		{ 1024, 1024 }, { 1023, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -1024, -1 },	{ 1024, 1024 }, { 0, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -2048, -1 },	{ 1024, 1024 }, { 0, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 1024, 1024 },	{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 2048, 1024 },	{ 1024, 1024 }, { 0, 0 });

		TexCoordTest(omm::TextureAddressMode::Wrap, { 512, 512 },    { 512, 1024 },  { 0, 512 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 0, 512 },		{ 512, 1024 },  { 0, 512 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 0, 0 },		{ 512, 1024 },  { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -1, -1 },		{ 512, 1024 },  { 511, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -1024, -1 },   { 512, 1024 },  { 0, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { -2048, -1 },   { 512, 1024 },  { 0, 1023 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 1024, 1024 },  { 512, 1024 },  { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Wrap, { 2048, 1024 },  { 512, 1024 },  { 0, 0 });
	}

	TEST(GetTexCoord, Mirror) {
		//												 [Mode, TexCoord, Size, Expected]
		// Positive X
		TexCoordTest(omm::TextureAddressMode::Mirror, { 0, 4 },  { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 1, 4 },  { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 2, 4 },  { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 3, 4 },  { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 4 },  { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 5, 4 },  { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 6, 4 },  { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 7, 4 },  { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 8, 4 },  { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 9, 4 },  { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 10, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 11, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 12, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 13, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 14, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 15, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 16, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 17, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 18, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 19, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 20, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 21, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 22, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 23, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 24, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 25, 4 }, { 8, 8 }, { 6, 4 });

		// Negative X
		TexCoordTest(omm::TextureAddressMode::Mirror, { -0, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, {  0, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -1, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -2, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -3, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -4, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -5, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -6, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -7, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -8, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -9, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -10, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -11, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -12, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -13, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -14, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -15, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -16, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -17, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -18, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -19, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -20, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -21, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -22, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -23, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -24, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { -25, 4 }, { 8, 8 }, { 7, 4 });

		// Positive Y
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  0 }, { 8, 8 }, { 4, 0, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  1 }, { 8, 8 }, { 4, 1, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  2 }, { 8, 8 }, { 4, 2, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  3 }, { 8, 8 }, { 4, 3, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  4 }, { 8, 8 }, { 4, 4, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  5 }, { 8, 8 }, { 4, 5, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  6 }, { 8, 8 }, { 4, 6, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  7 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  8 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4,  9 }, { 8, 8 }, { 4, 6, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 10 }, { 8, 8 }, { 4, 5, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 11 }, { 8, 8 }, { 4, 4, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 12 }, { 8, 8 }, { 4, 3, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 13 }, { 8, 8 }, { 4, 2, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 14 }, { 8, 8 }, { 4, 1, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 15 }, { 8, 8 }, { 4, 0, });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 4, 16 }, { 8, 8 }, { 4, 0, });

		// Positive X,Y
		TexCoordTest(omm::TextureAddressMode::Mirror, { 8, 8 },   { 8, 8 }, { 7, 7 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 16, 16 }, { 8, 8 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Mirror, { 32, 32 }, { 8, 8 }, { 0, 0 });
	}

	TEST(GetTexCoord, Clamp) {
		//												[Mode, TexCoord, Size, Expected]
		TexCoordTest(omm::TextureAddressMode::Clamp, { 512, 512 },	{ 1024, 1024 }, { 512, 512 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { 0, 512 },		{ 1024, 1024 }, { 0, 512 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { 0, 0 },		{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { -1, -1 },		{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { -1024, -1 },	{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { -2048, -1 },	{ 1024, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { 1024, 1024 }, { 1024, 1024 }, { 1023, 1023 });
		TexCoordTest(omm::TextureAddressMode::Clamp, { 2048, 1024 }, { 1024, 1024 }, { 1023, 1023 });
	}

	TEST(GetTexCoord, Border) {
		//												[Mode, TexCoord, Size, Expected]
		TexCoordTest(omm::TextureAddressMode::Border, { 512, 512 }, { 512, 1024 }, int2{ omm::kTexCoordBorder, 512 });
		TexCoordTest(omm::TextureAddressMode::Border, { 0, 512 }, { 512, 1024 }, { 0, 512 });
		TexCoordTest(omm::TextureAddressMode::Border, { 0, 0 }, { 512, 1024 }, { 0, 0 });
		TexCoordTest(omm::TextureAddressMode::Border, { -1, -1 }, { 512, 1024 }, omm::kTexCoordBorder2);
		TexCoordTest(omm::TextureAddressMode::Border, { 0, -1 }, { 512, 1024 }, int2{ 0, omm::kTexCoordBorder });
		TexCoordTest(omm::TextureAddressMode::Border, { -1024, -1 }, { 512, 1024 }, omm::kTexCoordBorder2);
		TexCoordTest(omm::TextureAddressMode::Border, { -2048, -1 }, { 512, 1024 }, omm::kTexCoordBorder2);
		TexCoordTest(omm::TextureAddressMode::Border, { 1024, 1024 }, { 512, 1024 }, omm::kTexCoordBorder2);
		TexCoordTest(omm::TextureAddressMode::Border, { 2048, 1024 }, { 512, 1024 }, omm::kTexCoordBorder2);
	}

	TEST(GetTexCoord, MirrorOnce) {

		//													[Mode, TexCoord, Size, Expected]
		// Positive X
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 0, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 1, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 2, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 3, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 5, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 6, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 7, 4 }, { 8, 8 }, { 7, 4 });
		// no more mirroring -> clamp
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 8, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 9, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 10, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 11, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 12, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 13, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 14, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 15, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 16, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 17, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 18, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 19, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 20, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 21, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 22, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 23, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 24, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 25, 4 }, { 8, 8 }, { 7, 4 });

		// Negative X
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -0, 4 }, { 8, 8 }, { 0, 4 });
		// First mirror...
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 0, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -1, 4 }, { 8, 8 }, { 0, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -2, 4 }, { 8, 8 }, { 1, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -3, 4 }, { 8, 8 }, { 2, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -4, 4 }, { 8, 8 }, { 3, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -5, 4 }, { 8, 8 }, { 4, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -6, 4 }, { 8, 8 }, { 5, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -7, 4 }, { 8, 8 }, { 6, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -8, 4 }, { 8, 8 }, { 7, 4 });
		// no more mirroring -> clamp
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -9, 4 }, { 8, 8 },  { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -10, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -11, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -12, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -13, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -14, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -15, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -16, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -17, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -18, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -19, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -20, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -21, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -22, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -23, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -24, 4 }, { 8, 8 }, { 7, 4 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { -25, 4 }, { 8, 8 }, { 7, 4 });

		// Positive Y
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  0 }, { 8, 8 }, { 4, 0, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  1 }, { 8, 8 }, { 4, 1, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  2 }, { 8, 8 }, { 4, 2, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  3 }, { 8, 8 }, { 4, 3, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  4 }, { 8, 8 }, { 4, 4, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  5 }, { 8, 8 }, { 4, 5, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  6 }, { 8, 8 }, { 4, 6, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  7 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  8 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4,  9 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 10 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 11 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 12 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 13 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 14 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 15 }, { 8, 8 }, { 4, 7, });
		// no more mirroring -> clamp
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 16 }, { 8, 8 }, { 4, 7, });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 17 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 18 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 19 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 20 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 21 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 22 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 23 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 24 }, { 8, 8 }, { 4, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 4, 25 }, { 8, 8 }, { 4, 7 });

		// Positive X,Y
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 8, 8 },   { 8, 8 },  { 7, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 16, 16 }, { 8, 8 },  { 7, 7 });
		TexCoordTest(omm::TextureAddressMode::MirrorOnce, { 32, 32 }, { 8, 8 },  { 7, 7 });
	}
}