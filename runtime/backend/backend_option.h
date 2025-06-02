#include <executorch/runtime/core/error.h>
#include <cstddef>
#include <cstring>

namespace executorch {
namespace ET_RUNTIME_NAMESPACE {

// Strongly-typed option key template
// Wraps a string key with type information for type-safe option access
template <typename T>
struct OptionKey {
  const char* key; // String identifier for the option
  constexpr explicit OptionKey(const char* k) : key(k) {}
};

// Supported option data types
enum class OptionType { BOOL, INT, STRING };

// Union-like container for option values (only one member is valid per option)
struct OptionValue {
  bool bool_value; // Storage for boolean values
  int int_value; // Storage for integer values
  const char* string_value; // Storage for string values
};

// Represents a single backend configuration option
struct BackendOption {
  const char* key; // Name of the option
  OptionType type; // Data type of the option
  OptionValue value; // Current value of the option
};

// Fixed-capacity container for backend options with type-safe access
// MaxCapacity: Maximum number of options this container can hold
template <size_t MaxCapacity>
class BackendOptions {
 public:
  // Initialize with zero options
  BackendOptions() : size(0) {}

  // Type-safe setters ---------------------------------------------------

  /// Sets or updates a boolean option
  /// @param key: Typed option key
  /// @param value: Boolean value to set
  void set_option(OptionKey<bool> key, bool value) {
    set_option_internal(key.key, OptionType::BOOL, {.bool_value = value});
  }

  /// Sets or updates an integer option
  /// @param key: Typed option key
  /// @param value: Integer value to set
  void set_option(OptionKey<int> key, int value) {
    set_option_internal(key.key, OptionType::INT, {.int_value = value});
  }

  /// Sets or updates a string option
  /// @param key: Typed option key
  /// @param value: Null-terminated string value to set
  void set_option(OptionKey<const char*> key, const char* value) {
    set_option_internal(key.key, OptionType::STRING, {.string_value = value});
  }

  // Type-safe getters ---------------------------------------------------

  /// Retrieves a boolean option value
  /// @param key: Typed option key
  /// @param out_value: Reference to store retrieved value
  /// @return: Error code (Ok on success)
  executorch::runtime::Error get_option(OptionKey<bool> key, bool& out_value)
      const {
    OptionValue val{};
    executorch::runtime::Error err =
        get_option_internal(key.key, OptionType::BOOL, val);
    if (err == executorch::runtime::Error::Ok) {
      out_value = val.bool_value;
    }
    return err;
  }

  /// Retrieves an integer option value
  /// @param key: Typed option key
  /// @param out_value: Reference to store retrieved value
  /// @return: Error code (Ok on success)
  executorch::runtime::Error get_option(OptionKey<int> key, int& out_value)
      const {
    OptionValue val{};
    executorch::runtime::Error err =
        get_option_internal(key.key, OptionType::INT, val);
    if (err == executorch::runtime::Error::Ok) {
      out_value = val.int_value;
    }
    return err;
  }

  /// Retrieves a string option value
  /// @param key: Typed option key
  /// @param out_value: Reference to store retrieved pointer
  /// @return: Error code (Ok on success)
  executorch::runtime::Error get_option(
      OptionKey<const char*> key,
      const char*& out_value) const {
    OptionValue val{};
    executorch::runtime::Error err =
        get_option_internal(key.key, OptionType::STRING, val);
    if (err == executorch::runtime::Error::Ok) {
      out_value = val.string_value;
    }
    return err;
  }

 private:
  BackendOption options[MaxCapacity]{}; // Storage for options
  size_t size; // Current number of options

  // Internal helper to set/update an option
  void
  set_option_internal(const char* key, OptionType type, OptionValue value) {
    // Update existing key if found
    for (size_t i = 0; i < size; ++i) {
      if (strcmp(options[i].key, key) == 0) {
        options[i].type = type;
        options[i].value = value;
        return;
      }
    }
    // Add new option if capacity allows
    if (size < MaxCapacity) {
      options[size++] = {key, type, value};
    }
  }

  // Internal helper to get an option value with type checking
  executorch::runtime::Error get_option_internal(
      const char* key,
      OptionType expected_type,
      OptionValue& out) const {
    for (size_t i = 0; i < size; ++i) {
      if (strcmp(options[i].key, key) == 0) {
        // Verify type matches expectation
        if (options[i].type != expected_type) {
          return executorch::runtime::Error::InvalidArgument;
        }
        out = options[i].value;
        return executorch::runtime::Error::Ok;
      }
    }
    return executorch::runtime::Error::NotFound; // Key not found
  }
};

// Helper functions for creating typed option keys --------------------------

/// Creates a boolean option key
constexpr OptionKey<bool> BoolKey(const char* k) {
  return OptionKey<bool>(k);
}

/// Creates an integer option key
constexpr OptionKey<int> IntKey(const char* k) {
  return OptionKey<int>(k);
}

/// Creates a string option key
constexpr OptionKey<const char*> StrKey(const char* k) {
  return OptionKey<const char*>(k);
}
} // namespace ET_RUNTIME_NAMESPACE
} // namespace executorch
