//
// reversed_memory_stream.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <istream>
#include <ostream>

#include "memory_buffer.hpp"

namespace inmemoryfs {

/// A class for reading an in-memory stream buffer in reverse.
class ReversedIMemoryStreamBuf: public std::streambuf {
public:
    ~ReversedIMemoryStreamBuf() = default;

    /// Constructs a `ReversedIMemoryStreamBuf` from a `MemoryBuffer`.
    ///
    /// @param buffer  The memory buffer.
    explicit ReversedIMemoryStreamBuf(std::shared_ptr<MemoryBuffer> buffer) noexcept;

protected:
    /// Called by seekof if the openmode is input.
    pos_type iseekoff(off_type off, std::ios_base::seekdir dir);

    /// Called by other member functions to alter the stream position of the controlled input sequence.
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;

    /// Called by other member functions to get an estimate on the number of characters available in controlled input sequence.
    ///
    /// Returns number of characters available in controlled input sequence.
    std::streamsize showmanyc() override;

    /// Called by other member functions to put a character back into the controlled input sequence and decrease the position indicator.
    ///
    /// Returns the value of the character put back, converted to a value of type int.
    int_type pbackfail(int_type ch) override;

    /// Called by other member functions to get the current character in the controlled input sequence without changing the current position.
    ///
    /// Returns the value of the current character, converted to a value of type int.
    std::streambuf::int_type underflow() override;

    /// Called by other member functions to get the current character in the controlled input sequence and advances the current position.
    ///
    /// Returns the value of the current character, converted to a value of type int.
    std::streambuf::int_type uflow() override;

    /// Retrieves characters from the controlled input sequence and stores them in the array pointed by s,
    /// until either n characters have been extracted or the end of the sequence is reached.
    ///
    /// Returns the number of characters copied.
    std::streamsize xsgetn(char *s, std::streamsize n) override;

private:
    /// Reads the character at the specified position.
    std::streambuf::int_type read(char *pos);

    const std::shared_ptr<MemoryBuffer> buffer_;
    char *start_;
    char *current_;
    char *end_;
};

/// A class for reading an in-memory buffer in reverse.
class ReversedIMemoryStream final : public std::istream  {
public:

    /// Constructs a `ReversedIMemoryStream` from a `MemoryBuffer`.
    ///
    /// @param buffer  The memory buffer.
    ReversedIMemoryStream(const std::shared_ptr<MemoryBuffer>& buffer) noexcept;

    ~ReversedIMemoryStream() = default;

private:
    ReversedIMemoryStreamBuf streambuf;
};

}
