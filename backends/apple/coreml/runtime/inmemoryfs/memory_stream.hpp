//
// memory_stream.hpp
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#include <istream>
#include <ostream>

#include "memory_buffer.hpp"

namespace inmemoryfs {

/// A class representing an in-memory stream buffer.
class MemoryStreamBuf: public std::streambuf {
public:
    ~MemoryStreamBuf() = default;

    /// Constructs a `MemoryStreamBuf` from a `MemoryBuffer`.
    ///
    /// @param buffer  The memory buffer.
    explicit MemoryStreamBuf(const std::shared_ptr<MemoryBuffer>& buffer) noexcept;

protected:
    /// Called by `seekof` if the `openmode` is input.
    ///
    /// @param offset  The offset  value relative to the `dir`.
    /// @param dir  The seek direction.
    /// @retval The stream position.
    pos_type iseekoff(off_type offset, std::ios_base::seekdir dir);

    /// Called by `seekof` if the `openmode` is output.
    ///
    /// @param offset  The offset  value relative to the `dir`.
    /// @param dir  The seek direction.
    /// @retval The stream position.
    pos_type oseekoff(off_type offset, std::ios_base::seekdir dir);

    /// Called by other member functions to alter the stream position of the controlled input sequence.
    ///
    /// @param which  The open mode.
    /// @retval The stream position.
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;

    /// Called by the public member function `pubseekoff` to alter the stream position.
    ///
    /// @param offset  The offset  value relative to the `dir`.
    /// @param dir  The seek direction.
    /// @param which  The open mode.
    /// @retval The stream position.
    std::streambuf::pos_type seekoff(std::streambuf::off_type offset,
                                     std::ios_base::seekdir dir,
                                     std::ios_base::openmode which) override;

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

    /// Called by other member functions to put a character into the controlled output sequence.
    ///
    /// Returns the value of the character that's put into the stream, converted to a value of type int.
    int_type overflow(int_type ch) override;

    /// Retrieves characters from the controlled input sequence and stores them in the array pointed by s,
    /// until either n characters have been extracted or the end of the sequence is reached.
    ///
    /// Returns the number of characters copied.
    std::streamsize xsgetn(char *s, std::streamsize n) override;

    /// Writes characters from the array pointed to by s into the controlled output sequence,
    /// until either n characters have been written or the end of the output sequence is reached.
    ///
    /// Returns the number of characters that's written.
    std::streamsize xsputn(const char *s, std::streamsize n) override;

private:
    /// Reads the character at the specified position.
    std::streambuf::int_type read(char *pos);

    const std::shared_ptr<MemoryBuffer> buffer_;
};

/// A class representing an in-memory input stream.
class MemoryIStream final : public std::istream  {
public:
    MemoryIStream(const std::shared_ptr<MemoryBuffer>& buffer) noexcept;

    ~MemoryIStream() = default;

private:
    MemoryStreamBuf streambuf;
};

/// A class representing an in-memory output stream.
class MemoryOStream final : public std::ostream  {
public:
    MemoryOStream(const std::shared_ptr<MemoryBuffer>& buffer) noexcept;

    ~MemoryOStream() = default;

private:
    MemoryStreamBuf streambuf;
};

}
