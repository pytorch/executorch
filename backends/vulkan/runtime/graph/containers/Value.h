/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Constant.h>
#include <executorch/backends/vulkan/runtime/graph/containers/SymInt.h>
#include <executorch/backends/vulkan/runtime/graph/containers/Types.h>

namespace vkcompute {

using ValueRef = int32_t;

constexpr ValueRef kDummyValueRef = -1;

inline bool is_valid(ValueRef value_ref) {
  return value_ref >= 0;
}

struct IOValueRef {
  ValueRef value;
  ValueRef staging;

  // Custom cast to ValueRef
  operator ValueRef() const {
    return value;
  };
};

/*
 * This class is modelled after c10::IValue; however, it is simplified and does
 * not support as many types. However, the core design is the same; it is a
 * tagged union over the types supported by the Vulkan Graph type.
 */
struct Value final {
 private:
  /*
   * The union type which is used to store the value of the Value.
   */
  union Payload {
    /*
     * Similar to IValue::Payload, trivially copyable types are nested in their
     * own union.
     */
    union TriviallyCopyablePayload {
      TriviallyCopyablePayload() : as_int(0) {}
      int64_t as_int;
      double as_double;
      bool as_bool;
    } u;

    api::vTensor as_tensor;
    api::StagingBuffer as_staging;
    TensorRef as_tensorref;

    std::vector<int64_t> as_int_list;
    std::vector<double> as_double_list;
    std::vector<bool> as_bool_list;

    // The below is a special type that is used to represent a list of other
    // values stored in the graph. One application of the type is to represent
    // a list of tensors or a list of optional tensors.
    std::vector<ValueRef> as_value_list;

    std::string as_string;

    SymInt as_symint;

    Payload() : u() {}
    // NOLINTNEXTLINE
    ~Payload(){};
  };

 public:
  //
  // Copy constructor and assignment (disabled)
  //

  Value(const Value& rhs) = delete;
  Value& operator=(const Value&) = delete;

  //
  // Move constructor and assignment; Move assignment is disabled but
  // construction is implemented to allow for use in container types.
  //

  Value& operator=(Value&&) = delete;

#define CASE_MOVE_TRIVIALLY_COPYABLE_TYPE(type_tag, member_name) \
  case type_tag:                                                 \
    payload.u.member_name = rhs.payload.u.member_name;           \
    break;

#define CASE_MOVE_MOVEABLE_TYPE(type_tag, type, member_name, dtor_name)  \
  case type_tag:                                                         \
    new (&payload.member_name) type(std::move(rhs.payload.member_name)); \
    rhs.payload.member_name.~dtor_name();                                \
    break;

  Value(Value&& rhs) noexcept : tag(rhs.tag) {
    switch (tag) {
      // Scalar types
      CASE_MOVE_TRIVIALLY_COPYABLE_TYPE(TypeTag::INT, as_int);
      CASE_MOVE_TRIVIALLY_COPYABLE_TYPE(TypeTag::DOUBLE, as_double);
      CASE_MOVE_TRIVIALLY_COPYABLE_TYPE(TypeTag::BOOL, as_bool);
      // Tensor and tensor adjacent types
      CASE_MOVE_MOVEABLE_TYPE(
          TypeTag::TENSOR, api::vTensor, as_tensor, vTensor);
      CASE_MOVE_MOVEABLE_TYPE(
          TypeTag::STAGING, api::StagingBuffer, as_staging, StagingBuffer);
      CASE_MOVE_MOVEABLE_TYPE(
          TypeTag::TENSORREF, TensorRef, as_tensorref, TensorRef);
      // Scalar lists
      CASE_MOVE_MOVEABLE_TYPE(
          TypeTag::INTLIST, std::vector<int64_t>, as_int_list, vector);
      CASE_MOVE_MOVEABLE_TYPE(
          TypeTag::DOUBLELIST, std::vector<double>, as_double_list, vector);
      CASE_MOVE_MOVEABLE_TYPE(
          TypeTag::BOOLLIST, std::vector<bool>, as_bool_list, vector);
      // Special types
      CASE_MOVE_MOVEABLE_TYPE(
          TypeTag::VALUELIST, std::vector<ValueRef>, as_value_list, vector);
      CASE_MOVE_MOVEABLE_TYPE(
          TypeTag::STRING, std::string, as_string, basic_string);
      CASE_MOVE_MOVEABLE_TYPE(TypeTag::SYMINT, SymInt, as_symint, SymInt);

      case TypeTag::NONE:
        clearToNone();
        break;
    }
    rhs.clearToNone();
  }

#undef CASE_MOVE_TRIVIALLY_COPYABLE_TYPE
#undef CASE_MOVE_MOVEABLE_TYPE

  //
  // Accessors
  //

  inline TypeTag type() const {
    return tag;
  }

  //
  // Destructor
  //

  ~Value() {
    switch (tag) {
      case TypeTag::TENSOR:
        payload.as_tensor.~vTensor();
        break;
      case TypeTag::STAGING:
        payload.as_staging.~StagingBuffer();
        break;
      case TypeTag::TENSORREF:
        payload.as_tensorref.~TensorRef();
        break;
      case TypeTag::INTLIST:
        payload.as_int_list.~vector();
        break;
      case TypeTag::DOUBLELIST:
        payload.as_double_list.~vector();
        break;
      case TypeTag::BOOLLIST:
        payload.as_bool_list.~vector();
        break;
      case TypeTag::VALUELIST:
        payload.as_value_list.~vector();
        break;
      case TypeTag::STRING:
        payload.as_string.~basic_string();
        break;
      case TypeTag::SYMINT:
        payload.as_symint.~SymInt();
        break;
      // Manually list out the types so that if a type here is added later and
      // not handled the compiler can catch it.
      case TypeTag::NONE:
      case TypeTag::INT:
      case TypeTag::DOUBLE:
      case TypeTag::BOOL:
        break;
    }
  }

  //
  // Constructors, isType(), toType()
  //

  Value() : tag(TypeTag::NONE) {}

  inline bool isNone() const {
    return tag == TypeTag::NONE;
  }

#define SUPPORT_TRIVIALLY_COPYABLE_TYPE(                    \
    type, type_name, type_tag, member_name)                 \
  explicit Value(type t) : tag(type_tag) {                  \
    payload.u.member_name = t;                              \
  }                                                         \
  inline bool is##type_name() const {                       \
    return tag == type_tag;                                 \
  }                                                         \
  inline const type& to##type_name() const {                \
    VK_CHECK_COND(                                          \
        is##type_name(),                                    \
        "Expected value to have type " #type_name ", got ", \
        tag,                                                \
        " instead.");                                       \
    return payload.u.member_name;                           \
  }

  SUPPORT_TRIVIALLY_COPYABLE_TYPE(int64_t, Int, TypeTag::INT, as_int);
  SUPPORT_TRIVIALLY_COPYABLE_TYPE(double, Double, TypeTag::DOUBLE, as_double);
  SUPPORT_TRIVIALLY_COPYABLE_TYPE(bool, Bool, TypeTag::BOOL, as_bool);

#undef SUPPORT_TRIVIALLY_COPYABLE_TYPE

#define SUPPORT_TRIVIALLY_MOVEABLE_TYPE(                    \
    type, type_name, type_tag, member_name)                 \
  explicit Value(type&& t) : tag(type_tag) {                \
    new (&payload.member_name) type(std::move(t));          \
  }                                                         \
  inline bool is##type_name() const {                       \
    return tag == type_tag;                                 \
  }                                                         \
  inline type& to##type_name() {                            \
    VK_CHECK_COND(                                          \
        is##type_name(),                                    \
        "Expected value to have type " #type_name ", got ", \
        tag,                                                \
        " instead.");                                       \
    return payload.member_name;                             \
  }                                                         \
  inline const type& toConst##type_name() const {           \
    VK_CHECK_COND(                                          \
        is##type_name(),                                    \
        "Expected value to have type " #type_name ", got ", \
        tag,                                                \
        " instead.");                                       \
    return payload.member_name;                             \
  }

  SUPPORT_TRIVIALLY_MOVEABLE_TYPE(
      api::vTensor,
      Tensor,
      TypeTag::TENSOR,
      as_tensor);

  SUPPORT_TRIVIALLY_MOVEABLE_TYPE(
      api::StagingBuffer,
      Staging,
      TypeTag::STAGING,
      as_staging);

  SUPPORT_TRIVIALLY_MOVEABLE_TYPE(
      TensorRef,
      TensorRef,
      TypeTag::TENSORREF,
      as_tensorref);

  SUPPORT_TRIVIALLY_MOVEABLE_TYPE(
      std::vector<int64_t>,
      IntList,
      TypeTag::INTLIST,
      as_int_list);

  SUPPORT_TRIVIALLY_MOVEABLE_TYPE(
      std::vector<double>,
      DoubleList,
      TypeTag::DOUBLELIST,
      as_double_list);

  SUPPORT_TRIVIALLY_MOVEABLE_TYPE(
      std::vector<bool>,
      BoolList,
      TypeTag::BOOLLIST,
      as_bool_list);

  SUPPORT_TRIVIALLY_MOVEABLE_TYPE(
      std::vector<ValueRef>,
      ValueList,
      TypeTag::VALUELIST,
      as_value_list);

  SUPPORT_TRIVIALLY_MOVEABLE_TYPE(
      std::string,
      String,
      TypeTag::STRING,
      as_string);

  SUPPORT_TRIVIALLY_MOVEABLE_TYPE(SymInt, SymInt, TypeTag::SYMINT, as_symint);

#undef SUPPORT_TRIVIALLY_COPYABLE_TYPE
#undef SUPPORT_TRIVIALLY_MOVEABLE_TYPE

 private:
  Payload payload;
  TypeTag tag;

  //
  // Utility Functions
  //

  inline void clearToNone() noexcept {
    payload.u.as_int = -1;
    tag = TypeTag::NONE;
  }
};

} // namespace vkcompute
