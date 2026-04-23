#pragma once

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <vector>

namespace executorch::backends::xnnpack::core {

template <typename T>
class Span {
public:
    constexpr Span() : data_(nullptr), size_(0) {}
    constexpr Span(T* data, size_t size) : data_(data), size_(size) {}

    template <typename U>
    Span(std::vector<U>& v) : data_(v.data()), size_(v.size()) {}

    template <typename U>
    Span(const std::vector<U>& v) : data_(v.data()), size_(v.size()) {}

    template <typename Dummy = T,
              typename = std::enable_if_t<std::is_const_v<Dummy>>>
    Span(std::initializer_list<std::remove_const_t<T>> il) : data_(il.begin()), size_(il.size()) {}

    constexpr T* data() const { return data_; }
    constexpr size_t size() const { return size_; }
    constexpr bool empty() const { return size_ == 0; }

    constexpr T& operator[](size_t i) const { return data_[i]; }

    constexpr T* begin() const { return data_; }
    constexpr T* end() const { return data_ + size_; }

private:
    T* data_;
    size_t size_;
};

}
