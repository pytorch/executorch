// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pytorch/tokenizers/base64.h>
#include <pytorch/tokenizers/string_integer_map.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#if defined(__APPLE__) || defined(WIN32) || defined(__linux__)
#define TEST_MEMORY_COMPARISON 1

#if defined(__APPLE__)
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif
#endif

namespace {

using namespace ::testing;
using ::base64::decode;
using ::tokenizers::Error;
using ::tokenizers::Result;
using ::tokenizers::detail::StringIntegerMap;
using ::tokenizers::detail::StringIntegerMapTypeBuilder;
using TokenizerMap = std::unordered_map<std::string, std::uint64_t>;

static inline std::string _get_resource_path(const std::string& name) {
  return std::getenv("RESOURCES_PATH") + std::string("/") + name;
}

} // namespace

class StringIntegerMapTest : public Test {
 public:
  void SetUp() override {
    modelPath_ = std::getenv("RESOURCES_PATH") +
        std::string("/test_tiktoken_tokenizer.model");
  }

  Result<TokenizerMap> loadModel() {
    std::ifstream file(modelPath_);
    TK_CHECK_OR_RETURN_ERROR(
        file,
        ParseFailure,
        "failed to open encoder file: %s",
        modelPath_.c_str());

    TokenizerMap model;
    for (std::string line; std::getline(file, line);) {
      if (line.empty()) {
        continue;
      }

      auto pos = line.find(' ');
      auto token = TK_UNWRAP(decode({line.data(), pos}));
      uint64_t rank = 0;
      try {
        rank = std::stoul(line.substr(pos + 1));
      } catch (const std::exception&) {
        TK_CHECK_OR_RETURN_ERROR(
            false, ParseFailure, "invalid encoder rank: %s", line.c_str());
      }
      model[token] = rank;
    }

    return model;
  }

  std::string modelPath_;
};

#if defined(TEST_MEMORY_COMPARISON) && TEST_MEMORY_COMPARISON

class TrackingAllocatorBase {
 public:
  static void reset();
  static std::size_t getSize();

 protected:
  static void* allocate(std::size_t size);
  static void deallocate(void* ptr);

  static std::size_t size_;
};

void TrackingAllocatorBase::reset() {
  size_ = 0;
}

std::size_t TrackingAllocatorBase::getSize() {
  return size_;
}

void* TrackingAllocatorBase::allocate(std::size_t size) {
  void* ptr = malloc(size);
  if (!ptr) {
    return nullptr;
  }

#if defined(WIN32)
  size_ += _msize(ptr);
#elif defined(__APPLE__)
  size_ += malloc_size(const_cast<const void*>(ptr));
#else
  size_ += malloc_usable_size(ptr);
#endif

  return ptr;
}

void TrackingAllocatorBase::deallocate(void* ptr) {
  if (!ptr) {
    return;
  }

#if defined(WIN32)
  size_ -= _msize(ptr);
#elif defined(__APPLE__)
  size_ -= malloc_size(const_cast<const void*>(ptr));
#else
  size_ -= malloc_usable_size(ptr);
#endif

  free(ptr);
}

std::size_t TrackingAllocatorBase::size_ = 0;

template <typename T>
class TrackingAllocator : public TrackingAllocatorBase {
 public:
  using value_type = T;
  TrackingAllocator() noexcept = default;
  template <class U>
  explicit TrackingAllocator(TrackingAllocator<U> const&) noexcept {}

  value_type* allocate(std::size_t count) {
    return static_cast<value_type*>(
        TrackingAllocatorBase::allocate(count * sizeof(value_type))); // NOLINT
  }

  void deallocate(value_type* ptr, std::size_t /*count*/) noexcept {
    TrackingAllocatorBase::deallocate(ptr);
  }
};

template <class T, class U>
bool operator==(
    TrackingAllocator<T> const&,
    TrackingAllocator<U> const&) noexcept {
  return true;
}

template <class T, class U>
bool operator!=(
    TrackingAllocator<T> const& lhs,
    TrackingAllocator<U> const& rhs) noexcept {
  return !(lhs == rhs);
}

#endif

TEST_F(StringIntegerMapTest, CreateFromModel) {
  const auto res = loadModel();
  ASSERT_EQ(res.ok(), true);
  const auto& model = res.get();
  StringIntegerMap map(model);

  for (const auto& [model_key, model_value] : model) {
    EXPECT_THAT(map.tryGetInteger(model_key), testing::Optional(model_value))
        << model_key;
    EXPECT_THAT(map.tryGetString(model_value), testing::Optional(model_key))
        << model_value;
  }

  EXPECT_FALSE(map.tryGetInteger("Ich weiß nicht"));
  EXPECT_FALSE(map.tryGetString(999999999));

  EXPECT_EQ(map.size(), model.size());
  std::unordered_set<std::string_view> walked_strings;
  std::unordered_set<std::uint64_t> walked_integers;

  for (std::size_t index = 0; index < map.size(); ++index) {
    const auto [str, integer] = map.getElement(index);
    EXPECT_TRUE(walked_strings.insert(str).second) << "str: " << str;
    EXPECT_TRUE(walked_integers.insert(integer).second)
        << "integer: " << integer;
  }
}

#if defined(TEST_MEMORY_COMPARISON) && TEST_MEMORY_COMPARISON

TEST_F(StringIntegerMapTest, MemoryConsumptionComparison) {
  TrackingAllocatorBase::reset();
  EXPECT_EQ(TrackingAllocatorBase::getSize(), 0);

  const auto res = loadModel();
  ASSERT_EQ(res.ok(), true);
  const auto& model = res.get();

  std::size_t string_integer_map_size = 0;
  {
    typename StringIntegerMapTypeBuilder<>::WithAllocator<
        TrackingAllocator<std::uint8_t>>::Map map(model);
    string_integer_map_size = TrackingAllocatorBase::getSize();
  }

  EXPECT_EQ(TrackingAllocatorBase::getSize(), 0);

  std::size_t unordered_map_size = 0;
  {
    std::unordered_map<
        std::string,
        std::uint64_t,
        std::hash<std::string>,
        std::equal_to<std::string>,
        TrackingAllocator<std::pair<const std::string, std::uint64_t>>>
        strings_to_ints;
    std::unordered_map<
        std::uint64_t,
        std::string,
        std::hash<std::uint64_t>,
        std::equal_to<std::uint64_t>,
        TrackingAllocator<std::pair<const std::uint64_t, std::string>>>
        ints_to_strings;
    for (const auto& [k, v] : model) {
      strings_to_ints.emplace(k, v);
      ints_to_strings.emplace(v, k);
    }

    unordered_map_size = TrackingAllocatorBase::getSize();
  }

  EXPECT_LT(string_integer_map_size, unordered_map_size);

#if 1
  std::cout << "string integer map size = " << string_integer_map_size
            << std::endl;
  std::cout << "unordered map size = " << unordered_map_size << std::endl;
#endif
}

#endif

template <std::size_t hash_offset>
struct FixedHash {
  std::size_t operator()(const std::string_view& str) const {
    if (str.empty()) {
      return hash_offset;
    }

    return str.size() - 1 + hash_offset;
  }

  std::size_t operator()(std::uint64_t value) const {
    if (value == 0) {
      return hash_offset;
    }

    return static_cast<std::size_t>(std::log10(value)) + hash_offset;
  }
};

template <typename THash>
class StringIntegerMapHashTest : public Test {
 public:
  using Container = typename StringIntegerMapTypeBuilder<>::WithIntegerHash<
      THash>::template WithStringHash<THash>::Map;
};

using StringIntegerMapHashTestTypes =
    ::testing::Types<FixedHash<0>, FixedHash<1>, FixedHash<2>, FixedHash<3>>;
TYPED_TEST_SUITE(StringIntegerMapHashTest, StringIntegerMapHashTestTypes);

TYPED_TEST(StringIntegerMapHashTest, HashCollisions) {
  std::unordered_map<std::string, std::uint64_t> source = {
      {"a", 0},
      {"b", 1},
      {"c", 2},
      {"d", 3},
  };

  typename TestFixture::Container map(source);

  //
  // Check that the strings exist in the map.
  //

  EXPECT_THAT(map.tryGetInteger("a"), Optional(0ull));
  EXPECT_THAT(map.tryGetInteger("b"), Optional(1ull));
  EXPECT_THAT(map.tryGetInteger("c"), Optional(2ull));
  EXPECT_THAT(map.tryGetInteger("d"), Optional(3ull));

  EXPECT_FALSE(map.tryGetInteger("e"));

  //
  // Check that the integers exist in the map.
  //

  EXPECT_THAT(map.tryGetString(0), Optional(std::string_view("a")));
  EXPECT_THAT(map.tryGetString(1), Optional(std::string_view("b")));
  EXPECT_THAT(map.tryGetString(2), Optional(std::string_view("c")));
  EXPECT_THAT(map.tryGetString(3), Optional(std::string_view("d")));

  EXPECT_FALSE(map.tryGetString(4));

  //
  // Test a lookup into the next bucket (which should be empty).
  //

  EXPECT_FALSE(map.tryGetInteger("aa"));
  EXPECT_FALSE(map.tryGetInteger("aaa"));
  EXPECT_FALSE(map.tryGetInteger("aaaa"));

  EXPECT_FALSE(map.tryGetString(10));
  EXPECT_FALSE(map.tryGetString(100));
  EXPECT_FALSE(map.tryGetString(1000));
}
