//
// memory_stream.cpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#include "memory_stream.hpp"

#include <limits>

namespace inmemoryfs {

MemoryStreamBuf::MemoryStreamBuf(const std::shared_ptr<MemoryBuffer>& buffer) noexcept : buffer_(buffer) {
    auto start = static_cast<char*>(buffer->data());
    auto end = start + buffer->size();
    setg(start, start, end);
    setp(start, end);
}

std::streambuf::pos_type MemoryStreamBuf::iseekoff(std::streambuf::off_type offset, std::ios_base::seekdir dir) {
    std::streambuf::off_type next = -1;
    const std::ptrdiff_t size = std::ptrdiff_t(egptr() - eback());
    switch (dir) {
        case std::ios_base::beg: {
            next = offset;
            break;
        }
        case std::ios_base::cur: {
            next = std::streambuf::off_type(std::ptrdiff_t(gptr() - eback())) + offset;
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

    setg(eback(), eback() + next, egptr());
    return gptr() - eback();
}

std::streambuf::pos_type MemoryStreamBuf::oseekoff(std::streambuf::off_type offset, std::ios_base::seekdir dir) {
    std::streambuf::off_type next = -1;
    const std::ptrdiff_t size = std::ptrdiff_t(epptr() - pbase());
    switch (dir) {
        case std::ios_base::beg: {
            next = offset;
            break;
        }
        case std::ios_base::cur: {
            next = std::streambuf::off_type(std::ptrdiff_t(pptr() - pbase())) + offset;
            break;
        }
        case std::ios_base::end: {
            next = std::streambuf::off_type(size) + offset;
            break;
        }
        default: {
            break;
        }
    }

    if (next < 0 || next > std::streambuf::off_type(size)) {
        return std::streambuf::pos_type(std::streambuf::off_type(-1));
    }

    setp(pbase(), epptr());
    pbump(static_cast<int>(next));
    return pptr() - pbase();
}

std::streambuf::pos_type
MemoryStreamBuf::seekoff(std::streambuf::off_type offset, std::ios_base::seekdir dir, std::ios_base::openmode which) {
    if (which & std::ios_base::out) {
        return oseekoff(offset, dir);
    } else if (which & std::ios_base::in) {
        return iseekoff(offset, dir);
    } else {
        return std::streambuf::pos_type(std::streambuf::off_type(-1));
    }
}

std::streambuf::pos_type MemoryStreamBuf::seekpos(std::streambuf::pos_type pos, std::ios_base::openmode which) {
    return seekoff(pos - std::streambuf::pos_type(std::streambuf::off_type(0)), std::ios::beg, which);
}

std::streamsize MemoryStreamBuf::showmanyc() {
    std::ptrdiff_t n = egptr() - gptr();
    std::streamsize max = std::numeric_limits<std::streamsize>::max();
    return (n <= max ? std::streamsize(n) : max);
}

std::streambuf::int_type MemoryStreamBuf::pbackfail(std::streambuf::int_type ch) {
    if (eback() == gptr() || (ch != traits_type::eof() && ch != gptr()[-1])) {
        return traits_type::eof();
    }
    gbump(-1);
    return ch;
}

std::streambuf::int_type MemoryStreamBuf::underflow() {
    if (gptr() == egptr()) {
        return traits_type::eof();
    }
    return traits_type::to_int_type(*gptr());
}

std::streambuf::int_type MemoryStreamBuf::uflow() {
    if (gptr() == egptr()) {
        return traits_type::eof();
    }
    const std::streambuf::char_type ch = *gptr();
    gbump(1);
    return traits_type::to_int_type(ch);
}

std::streambuf::int_type MemoryStreamBuf::overflow(std::streambuf::int_type ch) {
    if (ch == traits_type::eof()) {
        return ch;
    }
    const std::streambuf::char_type c = traits_type::to_char_type(ch);
    xsputn(&c, 1);
    return ch;
}

std::streamsize MemoryStreamBuf::xsgetn(char* s, std::streamsize n) {
    const ptrdiff_t remaining = egptr() - gptr();
    n = std::min(n, remaining);
    if (n <= 0) {
        return std::streamsize(0);
    }
    auto current = epptr();
    for (std::streamsize i = 0; i < n; i++) {
        *(s + i) = *(current + i);
    }
    setg(eback(), current + n, egptr());
    return n;
}

std::streamsize MemoryStreamBuf::xsputn(const char* s, std::streamsize n) {
    const ptrdiff_t remaining = epptr() - pptr();
    n = std::min(n, remaining);
    if (n <= 0) {
        return std::streamsize(0);
    }
    auto current = pptr();
    for (std::streamsize i = 0; i < n; i++) {
        *(current + i) = *(s + i);
    }

    pbump(static_cast<int>(n));
    return n;
}

MemoryIStream::MemoryIStream(const std::shared_ptr<MemoryBuffer>& buffer) noexcept
    : std::istream(nullptr), streambuf(buffer) {
    rdbuf(&streambuf);
}

MemoryOStream::MemoryOStream(const std::shared_ptr<MemoryBuffer>& buffer) noexcept
    : std::ostream(nullptr), streambuf(buffer) {
    rdbuf(&streambuf);
}

} // namespace inmemoryfs
