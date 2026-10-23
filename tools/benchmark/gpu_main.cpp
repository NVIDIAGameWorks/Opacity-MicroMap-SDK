/*
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gpu_runner.h"
#include "subdivision.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

static_assert(!OMM_GPU_REFERENCE_MODE, "OMM_GPU_REFERENCE_MODE must be OFF for the GPU benchmark.");

namespace
{
    struct Options
    {
        std::filesystem::path input = OMM_BENCHMARK_DEFAULT_INPUT;
        uint32_t iterations = 5;
        uint32_t warmup = 1;
        uint32_t limit = 0; // Zero means all files unless an explicit limit is supplied.
        std::string selection = "first";
        uint32_t seed = 0;
        int subdivision = -1;
        float targetArea = -1.f;
        float targetEdge = -1.f;
        bool computeOnly = true;
        bool debug = false;
        bool help = false;
    };

    void PrintUsage()
    {
        std::cout << "Usage: omm-gpu-benchmark [options]\n"
            "  --input PATH     Serialized input file or directory (default: bundled leaves.bin)\n"
            "  --limit N        Maximum selected files (default: all)\n"
            "  --selection MODE first, strided, or random (default: first)\n"
            "  --seed N         Random selection seed (default: 0; requires --selection random)\n"
            "  --iterations N   Measured iterations (default: 5)\n"
            "  --warmup N       Untimed warmup iterations (default: 1)\n"
            "  --subdivision N Force uniform level 0..12, or 0..9 with --graphics\n"
            "  --target-area N  Select area mode targeting N square texels per microtriangle\n"
            "  --target-edge N  Select edge mode targeting N texels per microtriangle edge\n"
            "  --graphics       Use graphics and compute shaders for baking\n"
            "  --debug          Enable D3D12/NVRHI validation (affects performance)\n"
            "  --help           Show this help\n"
            "Directories are searched recursively for .bin files, then sorted before selection.\n"
            "Strided selection spreads --limit files evenly across the full sorted list.\n"
            "Each file gets its own warmup and measured iterations; batch errors are reported and skipped.\n"
            "Compute shaders are used throughout unless --graphics is specified.\n"
            "Subdivision overrides are mutually exclusive.\n"
            "Measures GPU timestamps around SetupAndBake; uploads and readback are excluded.\n";
    }

    uint32_t ParseCount(std::string_view text, bool allowZero)
    {
        uint32_t value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc() || end != text.data() + text.size() || (!allowZero && value == 0))
            throw std::runtime_error("Invalid count: " + std::string(text));
        return value;
    }

    Options ParseOptions(int argc, char** argv)
    {
        Options options;
        bool seedSpecified = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view argument = argv[i];
            if (argument == "--help" || argument == "-h")
            {
                options.help = true;
                return options;
            }
            if (argument == "--graphics")
            {
                options.computeOnly = false;
                continue;
            }
            if (argument == "--debug")
            {
                options.debug = true;
                continue;
            }
            if (argument != "--input" && argument != "--iterations" && argument != "--warmup"
                && argument != "--subdivision" && argument != "--limit"
                && argument != "--selection" && argument != "--seed"
                && argument != "--target-area" && argument != "--target-edge")
                throw std::runtime_error("Unknown option: " + std::string(argument));
            if (++i == argc)
                throw std::runtime_error("Missing value for " + std::string(argument));
            const std::string_view value = argv[i];
            if (argument == "--input")
                options.input = argv[i];
            else if (argument == "--iterations")
                options.iterations = ParseCount(value, false);
            else if (argument == "--warmup")
                options.warmup = ParseCount(value, true);
            else if (argument == "--target-area")
                options.targetArea = omm::benchmark::ParseSubdivisionTarget(value);
            else if (argument == "--target-edge")
                options.targetEdge = omm::benchmark::ParseSubdivisionTarget(value);
            else if (argument == "--limit")
                options.limit = ParseCount(value, false);
            else if (argument == "--selection")
            {
                if (value != "first" && value != "strided" && value != "random")
                    throw std::runtime_error("--selection expects first, strided, or random");
                options.selection = value;
            }
            else if (argument == "--seed")
            {
                options.seed = ParseCount(value, true);
                seedSpecified = true;
            }
            else
            {
                const uint32_t level = ParseCount(value, true);
                if (level > 12)
                    throw std::runtime_error("--subdivision expects a level from 0 to 12");
                options.subdivision = static_cast<int>(level);
            }
        }
        if (seedSpecified && options.selection != "random")
            throw std::runtime_error("--seed requires --selection random");
        if ((options.targetArea > 0.f && options.targetEdge > 0.f)
            || (options.subdivision >= 0 && (options.targetArea > 0.f || options.targetEdge > 0.f)))
            throw std::runtime_error("Choose one of --subdivision, --target-area, or --target-edge");
        return options;
    }

    void Check(omm::Result result, const char* operation)
    {
        if (result != omm::Result::SUCCESS)
            throw std::runtime_error(std::string(operation) + " failed (omm::Result "
                + std::to_string(static_cast<int>(result)) + ")");
    }

    struct Resources
    {
        omm::Baker baker = nullptr;
        omm::Cpu::DeserializedResult deserialized = nullptr;

        Resources() = default;
        Resources(const Resources&) = delete;
        Resources& operator=(const Resources&) = delete;

        ~Resources()
        {
            if (deserialized)
                omm::Cpu::DestroyDeserializedResult(deserialized);
            if (baker)
                omm::DestroyBaker(baker);
        }
    };

    std::vector<std::filesystem::path> CollectInputs(const std::filesystem::path& input, bool directory)
    {
        if (!directory)
            return { input };

        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(input))
        {
            if (!entry.is_regular_file())
                continue;
            auto extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension == ".bin")
                paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
        if (paths.empty())
            throw std::runtime_error("No .bin inputs found under: " + input.string());
        return paths;
    }

    void SelectInputs(std::vector<std::filesystem::path>& paths, const Options& options)
    {
        const size_t count = options.limit ? std::min<size_t>(options.limit, paths.size()) : paths.size();
        if (options.selection == "strided" && count < paths.size())
        {
            std::vector<std::filesystem::path> selected;
            selected.reserve(count);
            for (size_t i = 0; i < count; ++i)
            {
                const size_t index = count == 1 ? (paths.size() - 1) / 2
                    : i * (paths.size() - 1) / (count - 1);
                selected.push_back(std::move(paths[index]));
            }
            paths = std::move(selected);
        }
        else
        {
            if (options.selection == "random")
            {
                // Explicit Fisher-Yates and bounded draws keep seeds reproducible across standard libraries.
                std::mt19937_64 random(options.seed);
                for (size_t i = 0; i < count && i + 1 < paths.size(); ++i)
                {
                    const uint64_t bound = paths.size() - i;
                    const uint64_t threshold = (uint64_t(0) - bound) % bound;
                    uint64_t value;
                    do { value = random(); } while (value < threshold);
                    std::swap(paths[i], paths[i + static_cast<size_t>(value % bound)]);
                }
            }
            paths.resize(count);
        }
    }

    struct FileResult
    {
        double medianMilliseconds = 0;
        uint64_t triangleCount = 0;
        uint64_t micromapCount = 0;
        uint64_t arrayBytes = 0;
    };

    FileResult BenchmarkFile(const std::filesystem::path& path, const Options& options,
        std::unique_ptr<omm::benchmark::GpuContext>& context)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("Cannot open input: " + path.string());
        const std::streamoff size = file.tellg();
        if (size < 8)
            throw std::runtime_error("Input is too short or unreadable: " + path.string());
        std::vector<uint8_t> data(static_cast<size_t>(size));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(data.data()), size))
            throw std::runtime_error("Cannot read input: " + path.string());

        Resources resources;
        omm::BakerCreationDesc creation;
        creation.type = omm::BakerType::CPU;
        creation.messageInterface.messageCallback = [](omm::MessageSeverity severity, const char* message, void*)
        {
            if (severity != omm::MessageSeverity::Info)
                std::cerr << "[omm-sdk] " << message << '\n';
        };
        Check(omm::CreateBaker(creation, &resources.baker), "CreateBaker");
        omm::Cpu::BlobDesc blob;
        blob.data = data.data();
        blob.size = data.size();
        Check(omm::Cpu::Deserialize(resources.baker, blob, &resources.deserialized), "Deserialize");
        const omm::Cpu::DeserializedDesc* deserialized = nullptr;
        Check(omm::Cpu::GetDeserializedDesc(resources.deserialized, &deserialized), "GetDeserializedDesc");
        if (!deserialized || deserialized->numInputDescs <= 0 || !deserialized->inputDescs)
            throw std::runtime_error("The blob contains no bake inputs");

        if (!context)
        {
            context = std::make_unique<omm::benchmark::GpuContext>(options.debug);
            std::cout << "GPU: " << context->AdapterName() << " (Direct3D 12)\n";
        }
        std::vector<std::unique_ptr<omm::benchmark::GpuInput>> inputs;
        uint64_t triangleCount = 0;
        for (int i = 0; i < deserialized->numInputDescs; ++i)
        {
            auto input = deserialized->inputDescs[i];
            if (options.subdivision >= 0)
            {
                input.maxSubdivisionLevel = static_cast<uint8_t>(options.subdivision);
                input.subdivision.mode = omm::SubdivisionMode::Fixed;
                input.subdivisionLevels = nullptr;
            }
            else if (options.targetArea > 0.f || options.targetEdge > 0.f)
            {
                input.subdivision = omm::benchmark::EffectiveSubdivision(input, false);
                if (options.targetArea > 0.f)
                {
                    input.subdivision.mode = omm::SubdivisionMode::Area;
                    input.subdivision.targetAreaInSquareTexels = options.targetArea;
                }
                else
                {
                    input.subdivision.mode = omm::SubdivisionMode::Edge;
                    if (options.targetEdge > 0.f)
                        input.subdivision.targetEdgeLengthInTexels = options.targetEdge;
                }
            }
            triangleCount += input.indexCount / 3;
            std::cout << "  Input " << i << ": " << input.indexCount / 3 << " triangles, max subdivision "
                << static_cast<unsigned>(input.maxSubdivisionLevel) << ", ";
            omm::benchmark::PrintSubdivision(std::cout, input, false);
            std::cout << '\n' << std::flush;
            inputs.push_back(context->Prepare(input, options.computeOnly));
        }
        std::cout << "Warmup: " << options.warmup << ", measured iterations: " << options.iterations
            << " (GPU SetupAndBake only; loading, uploads, CPU work and readback excluded)\n" << std::flush;

        const auto bake = [&]()
        {
            double milliseconds = 0;
            for (const auto& input : inputs)
                milliseconds += input->Run();
            return milliseconds;
        };
        for (uint32_t i = 0; i < options.warmup; ++i)
        {
            bake();
            std::cout << "Warmup " << i + 1 << '/' << options.warmup << " complete\n" << std::flush;
        }
        std::vector<double> times;
        times.reserve(options.iterations);
        std::cout << std::fixed << std::setprecision(6);
        for (uint32_t i = 0; i < options.iterations; ++i)
        {
            times.push_back(bake());
            std::cout << "Iteration " << i + 1 << '/' << options.iterations << ": " << times.back()
                << " ms\n" << std::flush;
        }

        uint64_t micromapCount = 0;
        uint64_t arrayBytes = 0;
        for (const auto& input : inputs)
        {
            const auto stats = input->ReadStats();
            micromapCount += stats.micromapCount;
            arrayBytes += stats.arrayBytes;
        }
        const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        std::sort(times.begin(), times.end());
        const size_t middle = times.size() / 2;
        const double median = times.size() % 2 ? times[middle] : (times[middle - 1] + times[middle]) * 0.5;
        std::cout << "GPU bake time: min " << times.front() << " ms, median " << median << " ms, mean " << mean
            << " ms, max " << times.back() << " ms\n"
            << "Output: " << micromapCount << " micromaps, " << arrayBytes << " array bytes for "
            << triangleCount << " triangles\n";
        return { median, triangleCount, micromapCount, arrayBytes };
    }

    int Run(const Options& options)
    {
        const bool directory = std::filesystem::is_directory(options.input);
        auto paths = CollectInputs(options.input, directory);
        const size_t discoveredCount = paths.size();
        SelectInputs(paths, options);
        std::cout << "Mode: " << (options.computeOnly ? "compute only" : "graphics + compute")
            << (options.debug ? " (validation enabled)" : "") << '\n';
        if (directory)
        {
            std::cout << "Directory: " << options.input.string() << '\n'
                << "Found " << discoveredCount << " .bin files; selected " << paths.size() << '\n';
            std::cout << "Selection: " << options.selection;
            if (options.selection == "random")
                std::cout << ", seed " << options.seed;
            std::cout << '\n';
        }

        std::unique_ptr<omm::benchmark::GpuContext> context;
        FileResult totals;
        size_t succeeded = 0;
        size_t failed = 0;
        for (size_t i = 0; i < paths.size(); ++i)
        {
            if (directory)
                std::cout << "\n[" << i + 1 << '/' << paths.size() << "] ";
            std::cout << "Input: " << paths[i].string() << '\n' << std::flush;
            try
            {
                const auto result = BenchmarkFile(paths[i], options, context);
                totals.medianMilliseconds += result.medianMilliseconds;
                totals.triangleCount += result.triangleCount;
                totals.micromapCount += result.micromapCount;
                totals.arrayBytes += result.arrayBytes;
                ++succeeded;
            }
            catch (const std::exception& error)
            {
                if (!directory)
                    throw;
                ++failed;
                std::cerr << "Failed: " << paths[i].string() << ": " << error.what() << '\n';
                // A failed GPU operation can leave an open command list or a latched error.
                context.reset();
            }
        }

        if (directory)
        {
            std::cout << std::fixed << std::setprecision(6)
                << "\nBatch summary: " << succeeded << " succeeded, " << failed << " failed\n"
                << "Sum of per-file median GPU bake times (successful files): "
                << totals.medianMilliseconds << " ms\n"
                << "Output totals (successful files): " << totals.micromapCount << " micromaps, "
                << totals.arrayBytes << " array bytes for " << totals.triangleCount << " triangles\n";
        }
        return failed ? 1 : 0;
    }
}

int main(int argc, char** argv)
{
    try
    {
        const auto options = ParseOptions(argc, argv);
        if (options.help)
        {
            PrintUsage();
            return 0;
        }
        return Run(options);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
