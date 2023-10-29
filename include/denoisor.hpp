#pragma once
#define FMT_HEADER_ONLY

#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <filesystem>
#include <fmt/core.h>

#include <dv-processing/core/core.hpp>
#include <dv-processing/core/filters.hpp>

namespace dv::noise::common {

template <typename T>
class Matrix2d : public std::vector<T> {
private:
    int16_t height_;
    int16_t width_;

public:
    explicit Matrix2d() = default;

    explicit Matrix2d(const int16_t width, const int16_t height) :
        width_(width),
        height_(height) {
        this->resize(width * height);
    }

    T &at(const int16_t x, const int16_t y) {
        return std::vector<T>::at(x + y * width_);
    }

    T &operator()(const int16_t x, const int16_t y) {
        return std::vector<T>::at(x + y * width_);
    }
};

template <typename T>
class Matrix3d : public std::vector<T> {
private:
    int16_t height_;
    int16_t width_;
    int16_t depth_;

public:
    explicit Matrix3d() = default;

    explicit Matrix3d(const int16_t width, const int16_t height, const int16_t depth) :
        width_(width),
        height_(height),
        depth_(depth) {
        this->resize(width * height * depth);
    }

    T &at(const int16_t x, const int16_t y, const int16_t d) {
        return std::vector<T>::at(x + y * width_ + d * width_ * height_);
    }

    T &operator()(const int16_t x, const int16_t y, const int16_t d) {
        return std::vector<T>::at(x + y * width_ + d * width_ * height_);
    }
};

struct Coordinate {
    friend class Offsets;

private:
    int16_t x_;
    int16_t y_;

public:
    int16_t x() const {
        return x_;
    };

    int16_t y() const {
        return y_;
    }
};

class Offsets : public std::vector<Coordinate> {
public:
    explicit Offsets() = default;

    explicit Offsets(const int16_t halfWidth, const int16_t halfHeight) {
        this->resize(halfWidth, halfHeight);
    }

    void resize(const int16_t halfWidth, const int16_t halfHeight) {
        int16_t width = halfWidth * 2 + 1;
        int16_t height = halfHeight * 2 + 1;
        std::vector<Coordinate>::resize(width * height);
		for (size_t i = 0; i < this->size(); i++) {
			this->at(i).x_ = - halfWidth + i / width;
			this->at(i).y_ = - halfHeight + i % height;
		}
    }
};


} // namespace dv::noise::common
