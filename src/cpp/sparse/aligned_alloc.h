#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <immintrin.h>

namespace seismic {

constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t SIMD_ALIGNMENT = 64;

template <typename T>
T* align_alloc(size_t count) {
    size_t bytes = count * sizeof(T);
    bytes = ((bytes + SIMD_ALIGNMENT - 1) / SIMD_ALIGNMENT) * SIMD_ALIGNMENT;
    void* ptr = _mm_malloc(bytes, SIMD_ALIGNMENT);
    return static_cast<T*>(ptr);
}

inline void align_free(void* ptr) {
    _mm_free(ptr);
}

template <typename T>
class AlignedVector {
public:
    AlignedVector() : data_(nullptr), size_(0), capacity_(0) {}

    explicit AlignedVector(size_t count, T value = T()) : data_(nullptr), size_(0), capacity_(0) {
        resize(count, value);
    }

    ~AlignedVector() {
        if (data_) align_free(data_);
    }

    AlignedVector(const AlignedVector&) = delete;
    AlignedVector& operator=(const AlignedVector&) = delete;

    AlignedVector(AlignedVector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    AlignedVector& operator=(AlignedVector&& other) noexcept {
        if (this != &other) {
            if (data_) align_free(data_);
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    void resize(size_t count, T value = T()) {
        if (count > capacity_) {
            T* new_data = align_alloc<T>(count);
            if (data_) {
                std::copy(data_, data_ + std::min(size_, count), new_data);
                align_free(data_);
            }
            data_ = new_data;
            capacity_ = count;
        }
        if (value != T()) {
            for (size_t i = size_; i < count; ++i) {
                data_[i] = value;
            }
        }
        size_ = count;
    }

    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

private:
    T* data_;
    size_t size_;
    size_t capacity_;
};

}
