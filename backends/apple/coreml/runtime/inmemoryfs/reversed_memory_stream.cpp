//
// reversed_memory_stream.cpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include "reversed_memory_stream.hpp"

#include <limits>

namespace inmemoryfs {

ReversedIMemoryStreamBuf::ReversedIMemoryStreamBuf(std::shared_ptr<MemoryBuffer> buffer) noexcept
    : buffer_(buffer), start_(static_cast<char*>(buffer->data())), current_(start_), end_(start_ + buffer->size()) {
    // we are intentionally setting `gptr` to the buffer end, this makes sure
    // that `underflow` and `uflow` methods are always called.
    setg(start_, start_, start_);
}

std::streambuf::pos_type ReversedIMemoryStreamBuf::iseekoff(std::streambuf::off_type offset,
                                                            std::ios_base::seekdir dir) {
    std::streambuf::off_type next = std::streambuf::off_type(-1);
    const std::ptrdiff_t size = std::ptrdiff_t(buffer_->size());
    switch (dir) {
        case std::ios_base::beg: {
            next = offset;
            break;
        }
        case std::ios_base::cur: {
            next = std::streambuf::off_type(std::ptrdiff_t(current_ - start_)) + offset;
            break;
        }
        case std::ios_base::end: {
            next = std::streambuf::off_type(size) + offset;
            break;
        }
        default:
            break;
    }
    if (next < 0 || next >= std::streambuf::off_type(size)) {
        return std::streambuf::pos_type(std::streambuf::off_type(-1));
    }
    current_ = start_ + offset;
    setg(start_, current_, current_);
    return std::streambuf::pos_type(current_ - start_);
}

std::streambuf::pos_type ReversedIMemoryStreamBuf::seekpos(std::streambuf::pos_type pos,
                                                           std::ios_base::openmode which) {
    if ((which & std::ios_base::in) == 0) {
        return std::streambuf::pos_type(std::streambuf::off_type(-1));
    }
    return iseekoff(pos - std::streambuf::pos_type(std::streambuf::off_type(0)), std::ios::beg);
}

std::streamsize ReversedIMemoryStreamBuf::showmanyc() {
    std::ptrdiff_t n = end_ - current_;
    std::streamsize max = std::numeric_limits<std::streamsize>::max();
    return (n <= max ? std::streamsize(n) : max);
}

std::streambuf::int_type ReversedIMemoryStreamBuf::read(char* pos) {
    std::ptrdiff_t offset = pos - start_;
    // offset from the end
    std::ptrdiff_t offsetFromEnd = buffer_->size() - offset - 1;
    return traits_type::to_int_type(start_[offsetFromEnd]);
}

std::streambuf::int_type ReversedIMemoryStreamBuf::pbackfail(std::streambuf::int_type ch) {
    if (current_ == end_ || (ch != traits_type::eof() && ch != read(current_ - 1))) {
        return traits_type::eof();
    }
    --current_;
    setg(start_, current_, current_);
    return ch;
}

std::streambuf::int_type ReversedIMemoryStreamBuf::underflow() {
    if (current_ == end_) {
        return traits_type::eof();
    }
    return traits_type::to_int_type(read(current_));
}

std::streambuf::int_type ReversedIMemoryStreamBuf::uflow() {
    if (current_ == end_) {
        return traits_type::eof();
    }
    auto ch = read(current_);
    ++current_;
    setg(start_, current_, current_);
    return ch;
}

std::streamsize ReversedIMemoryStreamBuf::xsgetn(char* s, std::streamsize n) {
    const ptrdiff_t remaining = (static_cast<char*>(buffer_->data()) + buffer_->size()) - current_;
    n = std::min(n, remaining);
    if (n <= 0) {
        return std::streamsize(0);
    }
    std::ptrdiff_t offset = current_ - start_;
    std::ptrdiff_t pos = buffer_->size() - 1 - offset;
    for (std::streamsize i = 0; i < n; i++) {
        s[i] = start_[pos - i];
    }
    current_ += n;
    setg(start_, current_, current_);
    return n;
}

ReversedIMemoryStream::ReversedIMemoryStream(const std::shared_ptr<MemoryBuffer>& buffer) noexcept
    : std::istream(nullptr), streambuf(buffer) {
    rdbuf(&streambuf);
}
} // namespace inmemoryfs
