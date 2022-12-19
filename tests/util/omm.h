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
#include <glm/glm.hpp>
#include <omm.h>

namespace vmtest {

	struct Texture {

		Texture(int w, int h, int mipCount, std::function<float(int x, int y, int w, int h, int mip)> cb) :Texture(w, h, mipCount, true /*enableZorder*/, cb) {
		}
		Texture(int w, int h, int mipCount, bool enableZorder, std::function<float(int x, int y, int w, int h, int mip)> cb) {

			_mipDescs.resize(mipCount);
			_mipData.resize(mipCount);

			_desc.mipCount = mipCount;
			_desc.mips = _mipDescs.data();
			_desc.format = omm::Cpu::TextureFormat::FP32;

			if (!enableZorder)
				_desc.flags = omm::Cpu::TextureFlags::DisableZOrder;

			for (int mipIt = 0; mipIt < mipCount; ++mipIt)
			{
				uint32_t mipW = w / (1u << mipIt);
				uint32_t mipH = h / (1u << mipIt);

				_mipData[mipIt].resize(size_t(w) * h);

				_mipDescs[mipIt].width = mipW;
				_mipDescs[mipIt].height = mipH;
				_mipDescs[mipIt].textureData = _mipData[mipIt].data();

				for (uint32_t j = 0; j < mipH; ++j) {
					for (uint32_t i = 0; i < mipW; ++i) {
						_mipData[mipIt][i + j * mipW] = cb(i, j, mipW, mipH, mipIt);
					}
				}
			}
		}

		omm::Cpu::TextureDesc& GetDesc() { return _desc; }
	private:
		std::vector<omm::Cpu::TextureMipDesc> _mipDescs;
		std::vector<std::vector<float>> _mipData;
		omm::Cpu::TextureDesc _desc;
	};
}