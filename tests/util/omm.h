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
#include <omm.hpp>

namespace vmtest 
{
	template<class T, omm::Cpu::TextureFormat Format>
	struct TextureImpl
	{
		TextureImpl(int w, int h, int mipCount, std::function<T(int x, int y, int w, int h, int mip)> cb)
			:TextureImpl(w, h, mipCount, true /*enableZorder*/, cb)
		{ }

		TextureImpl(int w, int h, int mipCount, bool enableZorder, std::function<T(int x, int y, int w, int h, int mip)> cb)
		{
			_mipDescs.resize(mipCount);
			_mipData.resize(mipCount);

			_desc.mipCount = mipCount;
			_desc.mips = _mipDescs.data();
			_desc.format = Format;

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

		const omm::Cpu::TextureDesc& GetDesc() { return _desc; }

	private:
		std::vector<omm::Cpu::TextureMipDesc> _mipDescs;
		std::vector<std::vector<T>> _mipData;
		omm::Cpu::TextureDesc _desc;
	};

	using TextureFP32 = TextureImpl<float, omm::Cpu::TextureFormat::FP32>;
	using TextureUNORM8 = TextureImpl<uint8_t, omm::Cpu::TextureFormat::UNORM8>;
}