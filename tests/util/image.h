#pragma once
#include <omm.h>
#include "omm.h"
#include <shared/geometry.h>
#include <shared/texture.h>
#include <shared/bird.h>
#include <shared/cpu_raster.h>
#include <filesystem>
#include <stb_image_write.h>

using uchar1 = glm::u8vec1;
using uchar2 = glm::u8vec2;
using uchar3 = glm::u8vec3;
using uchar4 = glm::u8vec4;

enum class SampleMode {
	Point,
	MAX_NUM,
};

template<class T>
struct Image {
	
	Image(int2 size) : _size(size), _data(_size.x* _size.y) {
	}

	Image(int2 size, T initalValue) : _size(size), _data(_size.x * _size.y){
		std::fill(_data.begin(), _data.end(), initalValue);
	}

	bool IsInsideImage(int2 pixel) const {
		return pixel.x >= 0 && pixel.y >= 0 && pixel.x < _size.x && pixel.y < _size.y;
	}

	void Store(int2 idx, T val) {
		assert(idx.x < _size.x);
		assert(idx.y < _size.y);
		_data[idx.x + idx.y * _size.x] = val;
	}
	T Load(int2 idx) const {
		assert(idx.x < _size.x);
		assert(idx.y < _size.y);
		return _data[idx.x + idx.y * _size.x];
	}

	void parallel_for_each(const std::function<void(int2, T&)>& cb) {
		#pragma omp parallel for
		for (int j = 0; j < _size.y; ++j) {
			for (int i = 0; i < _size.x; ++i) {
				int2 dst = { i, j };
				cb(dst, _Load(dst));
			}
		}
	}

	void for_each(const std::function<void(int2, T&)>& cb) {
		for (int j = 0; j < _size.y; ++j) {
			for (int i = 0; i < _size.x; ++i) {
				int2 dst = { i, j };
				cb(dst, _Load(dst));
			}
		}
	}

	int2 GetSize() const { return _size; }
	int GetWidth() const { return _size.x; }
	int GetHeight() const { return _size.y; }
	const char* GetData() const { return (const char*)_data.data(); }
	size_t GetDataSize() const { return _size.x * _size.y * sizeof(T); }

private:
	T& _Load(int2 idx) {
		assert(idx.x < _size.x);
		assert(idx.y < _size.y);
		return _data[idx.x + idx.y * _size.x];
	}
	T _Load(int2 idx) const {
		assert(idx.x < _size.x);
		assert(idx.y < _size.y);
		return _data[idx.x + idx.y * _size.x];
	}
	int2 _size;
	std::vector<T> _data;
};

using ImageRGB = Image<uchar3>;
using ImageRGBA = Image<uchar4>;
using ImageAlpha = Image<uint8_t>;

static inline bool SaveImageToFile(const std::string& folder, const std::string& fileName, const ImageRGB& image) {

#if OMM_TEST_ENABLE_IMAGE_DUMP
	constexpr bool kDumpDebug = true;
#else
	constexpr bool kDumpDebug = false;
#endif

	if (kDumpDebug)
	{
		if (!folder.empty())
			std::filesystem::create_directory(folder);

		const uint CHANNEL_NUM = 3;
		std::string dst = folder + "/" + fileName;
		int res = stbi_write_png(dst.c_str(), image.GetWidth(), image.GetHeight(), CHANNEL_NUM, (unsigned char*)image.GetData(), 0 /*stride in bytes*/);
		return res == 1;
	}
	return false;
}

static inline void FillWithCheckerboardRGB(ImageRGB& image, int checkerSize) {

	image.parallel_for_each([&image, checkerSize](int2 idx, uchar3& val) {

		float2 p = { checkerSize * float(idx.x / checkerSize + 0.5) / image.GetWidth() , checkerSize * float(idx.y / checkerSize + 0.5) / image.GetHeight() };

		if ((idx.x / checkerSize) % 2 != (idx.y / checkerSize) % 2)
			val = { 0, 0, 0 };
		else
			val = { 64, 64, 64 };
		});
}

static inline void FillWithCheckerboardRGBA(ImageRGBA& image, int checkerSize) {

	image.parallel_for_each([&image, checkerSize](int2 idx, uchar4& val) {

		float2 p = { checkerSize * float(idx.x / checkerSize + 0.5) / image.GetWidth() , checkerSize * float(idx.y / checkerSize + 0.5) / image.GetHeight() };

		if ((idx.x / checkerSize) % 2 != (idx.y / checkerSize) % 2)
			val = { 0, 0, 0, 255 };
		else
			val = { 64, 64, 64, 255 };
		});
}

template<class T>
static inline void Rasterize(Image<T>& image, const omm::Triangle& t, bool conservative, T color) {

	auto kernel = [&image, color](int2 idx, void*) {

		if (!image.IsInsideImage(idx))
			return;

		auto val = image.Load(idx);
		val.x += color.x;
		val.y += color.y;
		val.z += color.z;

		image.Store(idx, val);
	};

	if (conservative) {
		omm::RasterizeConservativeParallel(t, image.GetSize(), kernel);
	}
	else {
		omm::RasterizeParallel(t, image.GetSize(), kernel);
	}
}

template<class T, class T2>
static inline void RasterizeImg(Image<T>& image, const omm::Triangle& t, bool conservative, T color, const Image<T2>& src) {

	auto kernel = [&image, &src](int2 idx, float3* bc, void*) {
		if (image.IsInsideImage(idx))
			return;

		auto r = src.Load(omm::TextureAddressMode::Clamp, idx);
		auto val = image.Load(idx);
		val.x = r;
		image.Store(idx, val);
	};

	if (conservative) {
		omm::RasterizeConservativeParallel(t, image.GetSize(), kernel);
	}
	else {
		omm::RasterizeParallel(t, image.GetSize(), kernel);
	}
}

static void SaveToDisk(const std::string& folder, ImageRGB image) {

	std::string name = ::testing::UnitTest::GetInstance()->current_test_suite()->name();
	std::replace(name.begin(), name.end(), '/', '_');
	std::string tname = ::testing::UnitTest::GetInstance()->current_test_info()->name();

	std::string fileName = name + "_" + tname + ".png";
	SaveImageToFile("OmmBakeOutput", fileName.c_str(), image);
}