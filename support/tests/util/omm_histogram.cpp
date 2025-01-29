/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "omm_histogram.h"
#include <gtest/gtest.h>
#include <shared/bird.h>
#include <shared/parse.h>

namespace omm
{
	namespace Test
	{
		void ValidateArrayHistogram(const omm::Cpu::BakeResultDesc& resDesc)
		{
			if (resDesc.indexCount == 0)
				return;

			std::map<std::pair<uint32_t, omm::Format>, uint32_t> histogram;
			for (uint32_t idx = 0; idx < resDesc.descArrayCount; ++idx)
			{
				const omm::Cpu::OpacityMicromapDesc* desc = &resDesc.descArray[idx];

				auto key = std::make_pair((uint32_t)desc->subdivisionLevel, (omm::Format)desc->format);

				auto it = histogram.find(key);
				if (it == histogram.end())
				{
					histogram.insert(std::make_pair(key, 1));
				}
				else
				{
					histogram[key]++;
				}
			}

			for (uint32_t i = 0; i < resDesc.descArrayHistogramCount; ++i)
			{
				const omm::Cpu::OpacityMicromapUsageCount& usageCount = resDesc.descArrayHistogram[i];

				auto key = std::make_pair((uint32_t)usageCount.subdivisionLevel, (omm::Format)usageCount.format);

				auto it = histogram.find(key);

				if (usageCount.count == 0)
				{
					ASSERT_EQ(it, histogram.end());
					continue;
				}

				ASSERT_NE(it, histogram.end());

				EXPECT_EQ(it->second, usageCount.count);

				histogram.erase(it);
			}

			EXPECT_EQ(histogram.size(), 0);
		}

		void ValidateIndexHistogram(const omm::Cpu::BakeResultDesc& resDesc)
		{
			if (resDesc.indexCount == 0)
				return;

			std::map<std::pair<uint32_t, omm::Format>, uint32_t> histogram;
			for (uint32_t ommIndex = 0; ommIndex < resDesc.indexCount; ++ommIndex)
			{
				int32_t idx = omm::parse::GetOmmIndexForTriangleIndex(resDesc, ommIndex);

				if (idx < 0)
					continue;

				ASSERT_LT((uint32_t)idx, resDesc.descArrayCount);

				const omm::Cpu::OpacityMicromapDesc* desc = &resDesc.descArray[idx];

				auto key = std::make_pair((uint32_t)desc->subdivisionLevel, (omm::Format)desc->format);

				auto it = histogram.find(key);

				if (it == histogram.end())
				{
					histogram.insert(std::make_pair(key, 1));
				}
				else
				{
					histogram[key]++;
				}
			}

			for (uint32_t i = 0; i < resDesc.indexHistogramCount; ++i)
			{
				const omm::Cpu::OpacityMicromapUsageCount& usageCount = resDesc.indexHistogram[i];
				
				auto key = std::make_pair((uint32_t)usageCount.subdivisionLevel, (omm::Format)usageCount.format);

				auto it = histogram.find(key);

				if (usageCount.count == 0)
				{
					ASSERT_EQ(it, histogram.end());
					continue;
				}

				ASSERT_NE(it, histogram.end());

				EXPECT_EQ(it->second, usageCount.count);

				histogram.erase(it);
			}

			EXPECT_EQ(histogram.size(), 0);
		}

		void ValidateHistograms(const omm::Cpu::BakeResultDesc* resDesc)
		{
			ASSERT_NE(resDesc, (const omm::Cpu::BakeResultDesc*)nullptr);
			ValidateArrayHistogram(*resDesc);
			ValidateIndexHistogram(*resDesc);
		}	
	}
}