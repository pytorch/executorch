/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/platform.h>
#include <stdlib.h>
#include <cstdint>

#pragma once

namespace executorch {
namespace runtime {

/// Represents an allocator id returned by track_allocator.
typedef uint32_t AllocatorID;
/// Represents the chain id that will be passed in by the user during
/// event logging.
typedef int32_t ChainID;
/// Represents the debug handle that is generally associated with each
/// op executed in the runtime.
typedef uint32_t DebugHandle;

/// Default id's for chain id and debug handle.
constexpr ChainID kUnsetChainId = -1;
constexpr DebugHandle kUnsetDebugHandle = 0;
// Default bundled input index to indicate that it hasn't been set yet.
constexpr int kUnsetBundledInputIndex = -1;

/// Different types of delegate debug identifiers that are supported currently.
enum class DelegateDebugIdType {
  /// Default value, indicates that it's not a delegate event.
  kNone,
  /// Indicates a delegate event logged using an integer delegate debug
  /// identifier.
  kInt,
  /// Indicates a delegate event logged using a string delegate debug
  /// identifier i.e. the delegate debug id is a pointer to a string table
  /// managed by the class implementing EventTracer functionality.
  kStr
};

/// Indicates the type of the EValue that was logged. These values could be
/// serialized and should not be changed.
enum class LoggedEValueType {
  /// Intermediate output from an operator.
  kIntermediateOutput = 0,
  /// Output at the program level. This is essentially the output
  /// of the model.
  kProgramOutput = 1,
};

/// Indicates the level of event tracer debug logging. Verbosity of the logging
/// increases as we go down the enum list.
enum class EventTracerDebugLogLevel {
  /// No logging.
  kNoLogging,
  /// When set to this only the program level outputs will be logged.
  kProgramOutputs,
  /// When set to this all intermediate outputs and program level outputs
  /// will be logged.
  kIntermediateOutputs,
};

/**
 * This is the struct which should be returned when a profiling event is
 * started. This is used to uniquely identify that profiling event and will be
 * required to be passed into the end_profiling call to signal that the event
 * identified by this struct has completed.
 **/
struct EventTracerEntry {
  /// An event id to uniquely identify this event that was generated during a
  /// call to start the tracking of an event.
  int64_t event_id;
  /// The chain to which this event belongs to.
  ChainID chain_id;
  /// The debug handle corresponding to this event.
  DebugHandle debug_handle;
  /// The time at which this event was started to be tracked.
  et_timestamp_t start_time;
  /// When delegate_event_id_type != DelegateDebugIdType::kNone it indicates
  /// that event_id represents a delegate event. If delegate_event_id_type is:
  /// 1) kInt then event_id contains an integer delegate debug id.
  /// 2) kStr then event_id contains a string table index into a string table
  /// maintained by the class implementing EventTracer functionality that will
  /// give us the string identifier of this delegate event. For more details
  /// refer to the DelegateMappingBuilder library present in
  /// executorch/exir/backend/utils.py.
  DelegateDebugIdType delegate_event_id_type;
};
/**
 * EventTracer is a class that users can inherit and implement to
 * log/serialize/stream etc. the profiling and debugging events that are
 * generated at runtime for a model. An example of this is the ETDump
 * implementation in the SDK codebase that serializes these events to a
 * flatbuffer.
 */
class EventTracer {
 public:
  /**
   * Start a new event block (can consist of profiling and/or debugging events.)
   * identified by this name. A block is conceptually a set of events that we
   * want to group together. e.g. all the events that occur during the call to
   * execute() (i.e. model inference) could be categorized as a block.
   *
   * @param[in] name A human readable identifier for the event block. Users
   * calling this interface do not need to keep the memory pointed to by this
   * pointer around. The string must be copied over into internal memory during
   * this call.
   */
  virtual void create_event_block(const char* name) = 0;

  /**
   * Start the profiling of the event identified by name and debug_handle.
   * The user can pass in a chain_id and debug_handle to this call, or leave
   * them empty (default values) which would then result in the chain_id and
   * debug handle stored within (set by set_chain_debug_handle) this class to be
   * used.
   * @param[in] name Human readable name for the profiling event. Users calling
   * this interface do not need to keep the memory pointed to by this pointer
   * around. The string must be copied over into internal memory during this
   * call.
   * @param[in] chain_id The id of the chain to which this event belongs to. If
   * kUnsetChainId is passed in the chain_id and kUnsetDebugHandle for
   * debug_handle then the values stored in the class internally for these
   * properties will be used.
   * @param[in] debug_handle Debug handle generated ahead-of-time during model
   * compilation.
   *
   * @return Returns an instance of EventTracerEntry which should be passed back
   * into the end_profiling() call.
   */
  virtual EventTracerEntry start_profiling(
      const char* name,
      ChainID chain_id = kUnsetChainId,
      DebugHandle debug_handle = kUnsetDebugHandle) = 0;

  /**
   * Start the profiling of a delegate event. Similar to start_profiling it will
   * return an instance of EventTracerEntry that contains the details of this
   * event.
   *
   * @param[in] name Human readable name for the delegate event. This name has
   * to be the same name that was passed in during the Debug delegate mapping
   * generation in the export/ahead-of-time process. If indices and not names
   * are used by this delegate to identify ops executed in the backend then
   * nullptr can be passed in. Users calling this interface do not need to keep
   * the memory pointed to by this pointer around. The string must be copied
   * over into internal memory during this call.
   * @param[in] delegate_debug_index The id of the delegate event. If string
   * based names are used by this delegate to identify ops executed in the
   * backend then kUnsetDebugHandle should be passed in here.
   */
  virtual EventTracerEntry start_profiling_delegate(
      const char* name,
      DebugHandle delegate_debug_index) = 0;

  /**
   * Signal the end of the delegate profiling event contained in
   * event_tracer_entry. Users also have the option to log some some free-from
   * string based metadata along with this.
   *
   * @param[in] event_tracer_entry The EventTracerEntry returned by a call to
   * start_profiling_delegate().
   * @param[in] metadata Optional data relevant to the execution that the user
   * wants to log along with this event. Pointer to metadata doesn't need to be
   * valid after the call to this function. The contents and format of the data
   * are transparent to the event tracer. It will just pipe along the data and
   * make it available for the user again in the post-processing stage.
   * @param[in] metadata_len Length of the metadata buffer.
   */
  virtual void end_profiling_delegate(
      EventTracerEntry event_tracer_entry,
      const void* metadata = nullptr,
      size_t metadata_len = 0) = 0;

  /**
   * Some delegates get access to the profiling details only after the complete
   * graph has been executed. This interface is to support such use cases. It
   * can be called in a loop etc. to log any number of profiling events that are
   * part of this delegate.
   *
   * @param[in] name Human readable name for the delegate event. This name has
   * to be the same name that was passed in during the Debug delegate mapping
   * generation in the export/ahead-of-time process. If indices and not names
   * are used by this delegate to identify ops executed in the backend then
   * nullptr can be passed in. Users calling this interface do not need to keep
   * the memory pointed to by this pointer around. The string must be copied
   * over into internal memory during this call.
   * @param[in] delegate_debug_index The id of the delegate event. If string
   * based names are used by this delegate to identify ops executed in the
   * backend then kUnsetDebugHandle should be passed in here.
   * @param[in] start_time The timestamp when the delegate event started.
   * @param[in] end_time The timestamp when the delegate event finished.
   * @param[in] metadata Optional data relevant to the execution that the user
   * wants to log along with this event. Pointer to metadata doesn't need to be
   * valid after the call to this function. The contents and format of the data
   * are transparent to the event tracer. It will just pipe along the data and
   * make it available for the user again in the post-processing stage.
   * @param[in] metadata_len Length of the metadata buffer.
   */
  virtual void log_profiling_delegate(
      const char* name,
      DebugHandle delegate_debug_index,
      et_timestamp_t start_time,
      et_timestamp_t end_time,
      const void* metadata = nullptr,
      size_t metadata_len = 0) = 0;

  /**
   * End the profiling of the event identified by prof_entry
   *
   * @param[in] prof_entry Value returned by a call to start_profiling
   */
  virtual void end_profiling(EventTracerEntry prof_entry) = 0;

  /**
   * Track this allocation done via a MemoryAllocator which had profiling
   * enabled on it.
   *
   * @param[in] id Allocator id generated by a call to track_allocator.
   * @param[in] size The size of the allocation done, in bytes.
   */
  virtual void track_allocation(AllocatorID id, size_t size) = 0;

  /**
   * Generate an allocator id for this memory allocator that will be used in the
   * future to identify all the allocations done by this allocator.
   *
   * @param[in] name Human readable name for the allocator. Users calling
   * this interface do not need to keep the memory pointed to by this pointer
   * around. The string should be copied over into internal memory during this
   * call.
   *
   * @return Identifier to uniquely identify this allocator.
   */
  virtual AllocatorID track_allocator(const char* name) = 0;

  /**
   * Log an evalue during the execution of the model. This is useful for
   * debugging purposes. Model outputs are a special case of this and will
   * be logged with the output bool enabled.
   *
   * Users of this should refer to the chain_id and debug_handle to get the
   * context for these evalues and their corresponding op.
   *
   * @param[in] evalue The value to be logged.
   * @param[in] evalue_type Indicates what type of output this is logging e.g.
   * an intermediate output, program output etc.
   */
  virtual void log_evalue(
      const EValue& evalue,
      LoggedEValueType evalue_type) = 0;

  /**
   * Log an intermediate tensor output from a delegate.
   *
   * @param[in] name Human readable name for the delegate event. This name has
   * to be the same name that was passed in during the Debug delegate mapping
   * generation in the export/ahead-of-time process. If indices and not names
   * are used by this delegate to identify ops executed in the backend then
   * nullptr can be passed in. Users calling this interface do not need to keep
   * the memory pointed to by this pointer around. The string must be copied
   * over into internal memory during this call.
   * @param[in] delegate_debug_index The id of the delegate event. If string
   * based names are used by this delegate to identify ops executed in the
   * backend then kUnsetDebugHandle should be passed in here.
   * @param[in] output The tensor type output to be logged.
   */
  virtual void log_intermediate_output_delegate(
      const char* name,
      DebugHandle delegate_debug_index,
      const exec_aten::Tensor& output) = 0;

  /**
   * Log an intermediate tensor array output from a delegate.
   *
   * @param[in] name Human readable name for the delegate event. This name has
   * to be the same name that was passed in during the Debug delegate mapping
   * generation in the export/ahead-of-time process. If indices and not names
   * are used by this delegate to identify ops executed in the backend then
   * nullptr can be passed in. Users calling this interface do not need to keep
   * the memory pointed to by this pointer around. The string must be copied
   * over into internal memory during this call.
   * @param[in] delegate_debug_index The id of the delegate event. If string
   * based names are used by this delegate to identify ops executed in the
   * backend then kUnsetDebugHandle should be passed in here.
   * @param[in] output The tensor array type output to be logged.
   */
  virtual void log_intermediate_output_delegate(
      const char* name,
      DebugHandle delegate_debug_index,
      const ArrayRef<exec_aten::Tensor> output) = 0;

  /**
   * Log an intermediate int output from a delegate.
   *
   * @param[in] name Human readable name for the delegate event. This name has
   * to be the same name that was passed in during the Debug delegate mapping
   * generation in the export/ahead-of-time process. If indices and not names
   * are used by this delegate to identify ops executed in the backend then
   * nullptr can be passed in. Users calling this interface do not need to keep
   * the memory pointed to by this pointer around. The string must be copied
   * over into internal memory during this call.
   * @param[in] delegate_debug_index The id of the delegate event. If string
   * based names are used by this delegate to identify ops executed in the
   * backend then kUnsetDebugHandle should be passed in here.
   * @param[in] output The int type output to be logged.
   */
  virtual void log_intermediate_output_delegate(
      const char* name,
      DebugHandle delegate_debug_index,
      const int& output) = 0;

  /**
   * Log an intermediate bool output from a delegate.
   *
   * @param[in] name Human readable name for the delegate event. This name has
   * to be the same name that was passed in during the Debug delegate mapping
   * generation in the export/ahead-of-time process. If indices and not names
   * are used by this delegate to identify ops executed in the backend then
   * nullptr can be passed in. Users calling this interface do not need to keep
   * the memory pointed to by this pointer around. The string must be copied
   * over into internal memory during this call.
   * @param[in] delegate_debug_index The id of the delegate event. If string
   * based names are used by this delegate to identify ops executed in the
   * backend then kUnsetDebugHandle should be passed in here.
   * @param[in] output The bool type output to be logged.
   */
  virtual void log_intermediate_output_delegate(
      const char* name,
      DebugHandle delegate_debug_index,
      const bool& output) = 0;

  /**
   * Log an intermediate double output from a delegate.
   *
   * @param[in] name Human readable name for the delegate event. This name has
   * to be the same name that was passed in during the Debug delegate mapping
   * generation in the export/ahead-of-time process. If indices and not names
   * are used by this delegate to identify ops executed in the backend then
   * nullptr can be passed in. Users calling this interface do not need to keep
   * the memory pointed to by this pointer around. The string must be copied
   * over into internal memory during this call.
   * @param[in] delegate_debug_index The id of the delegate event. If string
   * based names are used by this delegate to identify ops executed in the
   * backend then kUnsetDebugHandle should be passed in here.
   * @param[in] output The double type output to be logged.
   */
  virtual void log_intermediate_output_delegate(
      const char* name,
      DebugHandle delegate_debug_index,
      const double& output) = 0;

  /**
   * Helper function to set the chain id ands debug handle. Users have two
   * options, the first is that they can directly pass in the chain id and debug
   * handle to start_profiling or they can explicitly set them through this
   * helper before calling start_profiling.
   *
   * The reason this helper exists is to
   * solve a specific problem. We want to do profiling logging inside the
   * codegen layer which calls the kernels. The problem though is that the
   * codegen layer doesn't have access to these ids when calling
   * start_profiling.
   *
   * Users should ideally use these within a RAII scope interface to make sure
   * that these values are unset after the end_profiling call. If non-default
   * values are passed into the start_profiling call they will always be given
   * precedence over the values set by this interface.
   *
   * So what we do is call this helper in method.cpp before
   * we hit the codegen layer and in the codegen layer we do a start_profiling
   * call without passing in a chain_id or debug_handle. This ensures that the
   * values set via this helper are the ones associated with that call.
   *
   * @param[in] chain_id Chain id of the current instruction being exectuted.
   * @param[in] debug_handle Debug handle of the current instruction being
   * executed. In this context debug handle and instruction id are the same
   * thing.
   */
  void set_chain_debug_handle(ChainID chain_id, DebugHandle debug_handle) {
    chain_id_ = chain_id;
    debug_handle_ = debug_handle;
  }

  /**
   * When running a program wrapped in a bundled program, log the bundled input
   * index of the current bundled input being tested out on this method.
   * If users want to unset the index back to the default value, they can call
   * this method with kUnsetBundledInputIndex.
   *
   * @param[in] bundled_input_index Index of the current input being tested
   */
  void set_bundled_input_index(int bundled_input_index) {
    bundled_input_index_ = bundled_input_index;
  }

  /**
   * Return the current bundled input index.
   */
  int bundled_input_index() {
    return bundled_input_index_;
  }

  /**
   * Set the level of event tracer debug logging that is desired.
   *
   */
  void set_event_tracer_debug_level(EventTracerDebugLogLevel log_level) {
    event_tracer_debug_level_ = log_level;
  }

  /**
   * Return the current level of event tracer debug logging.
   */
  EventTracerDebugLogLevel event_tracer_debug_level() {
    return event_tracer_debug_level_;
  }

  /**
   * Return the current status of intermediate outputs logging mode.
   */
  bool intermediate_outputs_logging_status() {
    return log_intermediate_tensors_;
  }

  /**
   * Get the current chain id.
   *
   * @return Current chain id.
   */
  ChainID current_chain_id() {
    return chain_id_;
  }

  /**
   * Get the current debug handle.
   *
   * @return Current debug handle.
   */
  DebugHandle current_debug_handle() {
    return debug_handle_;
  }

  virtual ~EventTracer() {}

 protected:
  ChainID chain_id_ = kUnsetChainId;
  DebugHandle debug_handle_ = kUnsetDebugHandle;
  bool event_tracer_enable_debugging_ = false;
  bool log_intermediate_tensors_ = false;
  int bundled_input_index_ = kUnsetBundledInputIndex;
  EventTracerDebugLogLevel event_tracer_debug_level_ =
      EventTracerDebugLogLevel::kNoLogging;
};

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::AllocatorID;
using ::executorch::runtime::ChainID;
using ::executorch::runtime::DebugHandle;
using ::executorch::runtime::DelegateDebugIdType;
using ::executorch::runtime::EventTracer;
using ::executorch::runtime::EventTracerDebugLogLevel;
using ::executorch::runtime::EventTracerEntry;
using ::executorch::runtime::kUnsetBundledInputIndex;
using ::executorch::runtime::kUnsetChainId;
using ::executorch::runtime::kUnsetDebugHandle;
using ::executorch::runtime::LoggedEValueType;
} // namespace executor
} // namespace torch
