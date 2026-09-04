/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/text_stream.h>

#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using executorch::extension::llm::TextStream;
using executorch::runtime::Error;

namespace {

// Maps each token to the bytes it contributes. `pair_pieces` overrides the
// piece for a (previous, token) pair, so a test can observe that the preceding
// token reaches the tokenizer at all.
class FakeTokenizer : public tokenizers::Tokenizer {
 public:
  std::map<uint64_t, std::string> pieces;
  std::map<std::pair<uint64_t, uint64_t>, std::string> pair_pieces;
  // Tokens the tokenizer refuses to decode.
  std::vector<uint64_t> rejected;

  tokenizers::Error load(const std::string&) override {
    initialized_ = true;
    return tokenizers::Error::Ok;
  }

  tokenizers::Result<std::string> decode(
      uint64_t previous,
      uint64_t token,
      bool /*skip_special_tokens*/ = false) const override {
    for (uint64_t bad : rejected) {
      if (token == bad) {
        return tokenizers::Error::Internal;
      }
    }
    auto pair = pair_pieces.find({previous, token});
    if (pair != pair_pieces.end()) {
      return pair->second;
    }
    auto single = pieces.find(token);
    if (single != pieces.end()) {
      return single->second;
    }
    return std::string();
  }

  tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string&, int8_t, int8_t) const override {
    return std::vector<uint64_t>{};
  }
  tokenizers::Result<std::string> id_to_piece(uint64_t) const override {
    return std::string();
  }
  tokenizers::Result<uint64_t> piece_to_id(const std::string&) const override {
    return uint64_t{0};
  }
};

// Collects what the stream emitted, both as separate pieces and joined.
struct Sink {
  std::vector<std::string> pieces;
  std::string joined;

  void operator()(const std::string& piece) {
    pieces.push_back(piece);
    joined += piece;
  }
};

// A stream writing into `sink`.
TextStream
stream_into(const FakeTokenizer& tokenizer, Sink& sink, uint64_t previous = 0) {
  return TextStream(
      tokenizer, [&sink](const std::string& piece) { sink(piece); }, previous);
}

// The three bytes of U+4E16 (CJK), which a byte-level tokenizer can split.
constexpr const char kCjkByte0[] = "\xE4";
constexpr const char kCjkByte1[] = "\xB8";
constexpr const char kCjkByte2[] = "\x96";
constexpr const char kCjk[] = "\xE4\xB8\x96";

} // namespace

TEST(TextStreamTest, EmitsWholeCharactersImmediately) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, "Hello"}, {2, " world"}};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  EXPECT_EQ(stream.append(std::vector<uint64_t>{1, 2}), Error::Ok);
  EXPECT_EQ(sink.joined, "Hello world");
  EXPECT_EQ(sink.pieces.size(), 2u);
  EXPECT_FALSE(stream.has_pending());
}

TEST(TextStreamTest, NeverEmitsAnEmptyPiece) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, ""}, {2, "x"}};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  ASSERT_EQ(stream.append(std::vector<uint64_t>{1, 2}), Error::Ok);
  EXPECT_EQ(sink.pieces, (std::vector<std::string>{"x"}));
}

// The reason this class exists: a character split across tokens must not reach
// the sink in pieces, or a consumer treating each piece as a string breaks.
TEST(TextStreamTest, HoldsBackAPartialCharacterUntilItCompletes) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, kCjkByte0}, {2, kCjkByte1}, {3, kCjkByte2}};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  ASSERT_EQ(stream.append(1u), Error::Ok);
  EXPECT_TRUE(sink.pieces.empty()) << "one third of a character is not text";
  EXPECT_TRUE(stream.has_pending());

  ASSERT_EQ(stream.append(2u), Error::Ok);
  EXPECT_TRUE(sink.pieces.empty());

  ASSERT_EQ(stream.append(3u), Error::Ok);
  EXPECT_EQ(sink.joined, kCjk);
  EXPECT_EQ(sink.pieces.size(), 1u) << "the character arrives whole, once";
  EXPECT_FALSE(stream.has_pending());
}

TEST(TextStreamTest, EmitsTheCompletePrefixAndKeepsTheRest) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, std::string("ab") + kCjkByte0}};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  ASSERT_EQ(stream.append(1u), Error::Ok);
  EXPECT_EQ(sink.joined, "ab") << "the finished characters go now";
  EXPECT_TRUE(stream.has_pending()) << "the split character waits";
}

TEST(TextStreamTest, FlushReleasesAnUnfinishedCharacter) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, kCjkByte0}};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  ASSERT_EQ(stream.append(1u), Error::Ok);
  ASSERT_TRUE(sink.pieces.empty());

  stream.flush();
  EXPECT_EQ(sink.joined, kCjkByte0)
      << "a generation that ends mid-character must not swallow the bytes";
  EXPECT_FALSE(stream.has_pending());
}

TEST(TextStreamTest, FlushIsIdempotentAndSilentWhenEmpty) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, "done"}};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  ASSERT_EQ(stream.append(1u), Error::Ok);
  stream.flush();
  stream.flush();
  EXPECT_EQ(sink.pieces, (std::vector<std::string>{"done"}));
}

// Only SentencePiece reads the preceding token, but the stream must still
// forward it and advance it, or that tokenizer would strip the wrong space.
TEST(TextStreamTest, ForwardsThePrecedingTokenAndAdvancesIt) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{7, "?"}, {8, "!"}};
  tokenizer.pair_pieces = {{{5, 7}, " seeded"}, {{7, 8}, " advanced"}};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink, /*previous=*/5);

  ASSERT_EQ(stream.append(7u), Error::Ok);
  EXPECT_EQ(sink.joined, " seeded") << "the constructor seed reaches decode";

  ASSERT_EQ(stream.append(8u), Error::Ok);
  EXPECT_EQ(sink.joined, " seeded advanced")
      << "the previous token becomes the one just decoded";
}

TEST(TextStreamTest, ATokenizerErrorFailsTheStreamForGood) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, "a"}, {3, "c"}};
  tokenizer.rejected = {2};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  EXPECT_EQ(stream.append(1u), Error::Ok);
  EXPECT_EQ(stream.append(2u), Error::InvalidArgument);
  EXPECT_TRUE(stream.failed());
  EXPECT_EQ(stream.append(3u), Error::InvalidState)
      << "a failed stream must not resume and emit text out of order";
  EXPECT_EQ(sink.joined, "a");
}

// The batch stops where it broke rather than skipping past the bad token.
TEST(TextStreamTest, ABatchStopsAtTheTokenThatFailed) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, "a"}, {3, "c"}};
  tokenizer.rejected = {2};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  EXPECT_EQ(
      stream.append(std::vector<uint64_t>{1, 2, 3}), Error::InvalidArgument);
  EXPECT_EQ(sink.joined, "a") << "nothing after the failure is emitted";
  EXPECT_TRUE(stream.failed());
}

// A speculative executor hands back several tokens at once, so a batch must
// read the same as the tokens arriving one by one.
TEST(TextStreamTest, ABatchMatchesTokenByTokenDelivery) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {
      {1, "He"}, {2, kCjkByte0}, {3, kCjkByte1}, {4, kCjkByte2}};
  Sink batched;
  Sink one_at_a_time;

  TextStream a = stream_into(tokenizer, batched);
  ASSERT_EQ(a.append(std::vector<uint64_t>{1, 2, 3, 4}), Error::Ok);

  TextStream b = stream_into(tokenizer, one_at_a_time);
  for (uint64_t token : {1u, 2u, 3u, 4u}) {
    ASSERT_EQ(b.append(token), Error::Ok);
  }

  EXPECT_EQ(batched.joined, one_at_a_time.joined);
  EXPECT_EQ(batched.joined, std::string("He") + kCjk);
}

TEST(TextStreamTest, AnInvalidLeadByteIsEmittedRatherThanStallingOutput) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, "\xFF"}, {2, "ok"}};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  ASSERT_EQ(stream.append(std::vector<uint64_t>{1, 2}), Error::Ok);
  EXPECT_EQ(
      sink.joined,
      "\xFF"
      "ok")
      << "a byte that can never start a character must not hold up the stream";
  EXPECT_FALSE(stream.has_pending());
}

TEST(TextStreamTest, ToleratesAnAbsentSink) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, "x"}};
  TextStream stream(tokenizer, nullptr);

  EXPECT_EQ(stream.append(1u), Error::Ok);
  stream.flush();
}

// A four-byte codepoint takes the len == 4 branch, which the three-byte cases
// above leave untested, and is the widest split a byte-level tokenizer can
// make.
TEST(TextStreamTest, HoldsBackAFourByteCharacterUntilItCompletes) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {
      {1, "\xF0"}, {2, "\x9F"}, {3, "\x98"}, {4, "\x80"}}; // U+1F600
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  for (uint64_t id : {1u, 2u, 3u}) {
    ASSERT_EQ(stream.append(id), Error::Ok);
    EXPECT_TRUE(sink.joined.empty()) << "emitted before the character finished";
    EXPECT_TRUE(stream.has_pending());
  }
  ASSERT_EQ(stream.append(4u), Error::Ok);
  EXPECT_EQ(sink.joined, "\xF0\x9F\x98\x80");
  EXPECT_FALSE(stream.has_pending());
}

// A lead byte promises continuation bytes that never arrive. Holding them would
// stall the stream for good, so they go out as-is: the sink can see invalid
// UTF-8 here, which is the documented trade for never blocking.
TEST(TextStreamTest, AMalformedSequenceIsEmittedRatherThanHeldForever) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, "\xE4\x41"}}; // 3-byte lead, then ASCII 'A'
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  ASSERT_EQ(stream.append(1u), Error::Ok);
  EXPECT_EQ(sink.joined, "\xE4\x41");
  EXPECT_FALSE(stream.has_pending()) << "a malformed tail must not be held";
}

// The bytes held when a decode fails are not lost: the stream is sticky-failed,
// but flush() still surrenders what it was holding.
TEST(TextStreamTest, FlushAfterAFailureStillReleasesTheHeldBytes) {
  FakeTokenizer tokenizer;
  tokenizer.pieces = {{1, kCjkByte0}};
  tokenizer.rejected = {2};
  Sink sink;
  TextStream stream = stream_into(tokenizer, sink);

  ASSERT_EQ(stream.append(1u), Error::Ok);
  ASSERT_TRUE(stream.has_pending());
  EXPECT_NE(stream.append(2u), Error::Ok);
  EXPECT_TRUE(stream.has_pending()) << "a failure keeps what was held";

  stream.flush();
  EXPECT_EQ(sink.joined, kCjkByte0);
  EXPECT_FALSE(stream.has_pending());
}
