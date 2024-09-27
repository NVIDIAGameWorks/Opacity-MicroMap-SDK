/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <iostream>
#include <fstream>

#include <omm.hpp>
#include <benchmark/benchmark.h>

class OMMBake : public benchmark::Fixture {
protected:

	void SetUp(const ::benchmark::State& state) override {

		omm::BakerCreationDesc desc;
		desc.type = omm::BakerType::CPU;
		desc.messageInterface.messageCallback = [](omm::MessageSeverity severity, const char* message, void* userArg) {
			if (severity > omm::MessageSeverity::PerfWarning)
				std::cout << "[omm-sdk]: " << message << std::endl;
		};

		omm::Result res = omm::CreateBaker(desc, &_baker);
		assert(res == omm::Result::SUCCESS);

		auto readFile = [](const char* filename)->std::vector<char>
		{
			std::ifstream file(filename, std::ios::binary);
			return std::vector<char>((std::istreambuf_iterator<char>(file)),
				std::istreambuf_iterator<char>());
		};

		auto data = readFile("C:\\Users\\jdeligiannis\\Downloads\\myExpensiveBakeJob_80mb.bin");

		omm::Cpu::BlobDesc blob;
		blob.data = data.data();
		blob.size = data.size();

		ommCpuDeserializedResult deserializedResult;
		res = omm::Cpu::Deserialize(_baker, blob, &deserializedResult);
		assert(res == omm::Result::SUCCESS);

		res = omm::Cpu::GetDeserializedDesc(deserializedResult, &_desDesc);
		assert(res == omm::Result::SUCCESS);
		assert(desDesc->numInputDescs == 1);
		assert(desDesc->numResultDescs == 0);
	}

	void TearDown(const ::benchmark::State& state) override {

		omm::Result res = omm::Cpu::DestroyDeserializedResult(_deserializedResult);
		assert(res == omm::Result::SUCCESS);

		res = omm::DestroyBaker(_baker);
		assert(res == omm::Result::SUCCESS);
	}

	void Run(benchmark::State& st, omm::Cpu::BakeFlags extraFlags) {
		st.PauseTiming();
		// Setup the baking parameters, setting only required data.
		omm::Cpu::BakeInputDesc bakeDesc = _desDesc->inputDescs[0];
		bakeDesc.maxWorkloadSize = 0xFFFFFFFFFFFFFFFF;

		// Adjust the workload size
		uint32_t flags = (uint32_t)(bakeDesc.bakeFlags) | (uint32_t)omm::Cpu::BakeFlags::DisableSpecialIndices | (uint32_t)omm::Cpu::BakeFlags::EnableInternalThreads | (uint32_t)extraFlags;
		bakeDesc.bakeFlags = (omm::Cpu::BakeFlags)flags;
		bakeDesc.maxWorkloadSize = 0xFFFFFFFFFFFFFFFF;

		st.ResumeTiming();
		omm::Cpu::BakeResult bakeResultHandle;
		omm::Result res = omm::Cpu::Bake(_baker, bakeDesc, &bakeResultHandle);
		assert(res == omm::Result::SUCCESS);
		st.PauseTiming();

		// Cleanup. Result no longer needed
		res = omm::Cpu::DestroyBakeResult(bakeResultHandle);
		assert(res == omm::Result::SUCCESS);
		st.ResumeTiming();
	}

	omm::Baker _baker = 0;
	omm::Cpu::DeserializedResult _deserializedResult = 0;
	const omm::Cpu::DeserializedDesc* _desDesc = nullptr;
};

BENCHMARK_DEFINE_F(OMMBake, Default)(benchmark::State& st) {
	for (auto s : st)
	{
		Run(st, omm::Cpu::BakeFlags::None);
	}
}

BENCHMARK_DEFINE_F(OMMBake, DisableFineClassification)(benchmark::State& st) {

	omm::Cpu::BakeFlags disableFineClassification = (omm::Cpu::BakeFlags)(1u << 9u);
	for (auto s : st)
	{
		Run(st, disableFineClassification);
	}
}

BENCHMARK_DEFINE_F(OMMBake, EnableWrapping)(benchmark::State& st) {

	omm::Cpu::BakeFlags enableWrapping = (omm::Cpu::BakeFlags)(1u << 11u);
	for (auto s : st)
	{
		Run(st, enableWrapping);
	}
}

BENCHMARK_DEFINE_F(OMMBake, StochasticClassification)(benchmark::State& st) {

	omm::Cpu::BakeFlags StochasticClassification = (omm::Cpu::BakeFlags)(1u << 13u);
	for (auto s : st)
	{
		Run(st, StochasticClassification);
	}
}

BENCHMARK_DEFINE_F(OMMBake, StochasticClassification_EnableWrapping)(benchmark::State& st) {

	omm::Cpu::BakeFlags StochasticClassification_EnableWrapping = (omm::Cpu::BakeFlags)((1u << 11u) | (1u << 13u));
	for (auto s : st)
	{
		Run(st, StochasticClassification_EnableWrapping);
	}
}

BENCHMARK_REGISTER_F(OMMBake, Default)->Unit(benchmark::kSecond)->Name("Default");

#if 0
BENCHMARK_REGISTER_F(OMMBake, DisableFineClassification)->Unit(benchmark::kSecond)->Name("DisableFineClassification");

BENCHMARK_REGISTER_F(OMMBake, EnableWrapping)->Unit(benchmark::kSecond)->Name("EnableWrapping");

BENCHMARK_REGISTER_F(OMMBake, StochasticClassification)->Unit(benchmark::kSecond)->Name("StochasticClassification");

BENCHMARK_REGISTER_F(OMMBake, StochasticClassification_EnableWrapping)->Unit(benchmark::kSecond)->Name("StochasticClassification+EnableWrapping");
#endif