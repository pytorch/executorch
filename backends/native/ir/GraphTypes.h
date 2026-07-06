/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * Graph: backend-facing IR adapter.
 *
 * These types provide our backends' view of the program IR. The current
 * implementation adapts ExecuTorch's flatbuffer
 * (executorch_flatbuffer::ExecutionPlan / KernelCall) but the API is
 * intentionally backing-agnostic — a different serialization could
 * replace the underlying storage without changing this header or any
 * backend that uses it.
 *
 * Some methods are documented IR concepts that the current adapter does
 * not yet back (e.g., mutable_buffer_ids, version): they are placeholders
 * pending serialization support.
 *
 * ----------------------------------------------------------------------
 * Fictional IR schema (what Graph effectively presents)
 * ----------------------------------------------------------------------
 * If the IR were serialized in its own format (independent of the
 * underlying ExecuTorch flatbuffer), it would look like this:
 *
 *   // Top-level
 *   table Graph {
 *     version:         string;        // schema/program version
 *     values:          [Value];       // dense pool indexed by value_id (uint)
 *     inputs:          [uint];        // graph input value_ids
 *     outputs:         [uint];        // graph output value_ids
 *     mutable_buffers: [uint];        // values that persist across executes
 *     operators:       [OperatorDef]; // op-name registry (deduped)
 *     chains:          [Chain];       // chains[0] is main
 *   }
 *
 *   table OperatorDef {
 *     name:            string;        // e.g. "aten.add.Tensor"
 *   }
 *
 *   // Op chains
 *   table Chain {
 *     instructions:    [Instruction];
 *   }
 *
 *   table Instruction {
 *     body: KernelCall;  // Only Kernel instructions are supported.
 *   }
 *
 *   table KernelCall {
 *     op_index:        uint;          // → Graph.operators[op_index].name
 *     args:            [uint];        // value_ids: args[0..n-2] = inputs,
 *                                     // args[n-1] = output. Single-output
 *                                     // assumed today; multi-output is a
 *                                     // future extension.
 *   }
 *
 *   // Values
 *   union Value {
 *     None,
 *     Int     { v: int64; },
 *     Double  { v: float64; },
 *     Bool    { v: bool; },
 *     String  { v: string; },
 *     Tensor,
 *     IntList,
 *     DoubleList,
 *     BoolList,
 *     OptionalTensor,
 *     // ...
 *   }
 *
 *   table Tensor {
 *     scalar_type:     ScalarType;     // dtype
 *     sizes:           [int32];        // for DYNAMIC_BOUND, this is max-shape
 *     dim_order:       [uint8];        // permutation defining memory layout
 *     shape_dynamism:  ShapeDynamism;  // STATIC | DYNAMIC_BOUND |
 * DYNAMIC_UNBOUND allocation:      AllocationInfo?;// null = no AOT plan data:
 * TensorData;
 *   }
 *
 *   union TensorData {
 *     None,
 *     Inline   { buffer_idx: uint; },  // bytes embedded in program
 *     External { ndm_key:    string;}, // bytes in NamedDataMap, FQN-keyed
 *   }
 *
 *   table AllocationInfo {
 *     pool_id: int32;
 *     offset:  uint64;                 // raw byte offset within pool_id
 *   }
 *
 *   table IntList {
 *     member_ids: [int64];             // value_ids of list elements
 *   }
 *
 * ----------------------------------------------------------------------
 * Derived views computed by the adapter (not in the schema itself)
 * ----------------------------------------------------------------------
 *   mem_obj_id(vid)               sort-and-index over (pool_id, offset)
 *                                 → dense small int identifying shared
 *                                   storage slots. Two values with the
 *                                   same id were memory-planned to share
 *                                   storage (used by router for
 *                                   AllocRequest grouping; backends MAY
 *                                   honor it as actual storage aliasing).
 *   value_kind(vid)               from membership in inputs/outputs +
 *                                 the data field
 *                                 → INPUT / OUTPUT / CONSTANT /
 *                                   INTERMEDIATE / MUTABLE_BUFFER.
 *   tensor_constant_data_key(vid) convenience accessor for
 *                                 TensorData.External.ndm_key.
 *   tensor_nbytes_max(vid)        dtype_size × prod(sizes); upper bound
 *                                 for DYNAMIC_BOUND tensors.
 *   producer(vid)                 which instruction produces this value
 *                                 (nullptr for inputs/constants).
 *   users(vid)                    all instructions that consume this value.
 *   num_users(vid)                shortcut for users(vid).size().
 *   find_ops(name)                all KernelCall instructions matching
 *                                 the given operator base name.
 *
 * ----------------------------------------------------------------------
 * Adapter cost
 * ----------------------------------------------------------------------
 * All accessors are inline and compile down to essentially the same
 * machine code as direct flatbuffer access. The Graph constructor pays
 * a one-time O(N log N) precompute (over tensor values) for mem_obj_id,
 * O(num_inputs + num_outputs) for the value_kind sets, and O(total_args)
 * for the use-def indices (producer, users, op name index). Per-call
 * overhead in the runtime hot path is a few inline indirections plus
 * predictable ET_CHECK branches; release builds compile away these
 * checks entirely under -DNDEBUG.
 */

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/core/tensor_shape_dynamism.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/schema/program_generated.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cstdint>

namespace executorch {
namespace backends {
namespace portable {

// Forward declare
class Graph;

/**
 * Value kind — matches design doc's TensorKind
 */
enum class ValueKind : uint8_t {
  INPUT = 0, // Graph input (user provides)
  OUTPUT, // Graph output (user reads)
  CONSTANT, // Immutable weight
  MUTABLE_BUFFER, // Mutable state (e.g., KV cache)
  INTERMEDIATE, // Temporary (produced/consumed internally)
};

/**
 * Type of an EValue stored at a value_id. Mirrors the runtime EValue
 * sum-type but in adapter-level form (no flatbuffer types in the API).
 */
enum class ValueType : uint8_t {
  None = 0,
  Int,
  Double,
  Bool,
  Tensor,
  IntList,
  TensorList,
  OptionalTensorList,
  Other, // String, OptionalTensor, BoolList, DoubleList, ...
         // Adapter doesn't surface these yet; executor falls back to
         // default-constructed EValue.
};

/**
 * Kind of an Instruction in a Chain.
 *
 * Only Kernel instructions are supported; the native backend
 * partitioner guarantees that no control-flow, move, free, or
 * delegate instructions appear in the delegated subgraph.
 * Encountering any other kind during deserialization is a fatal
 * error.
 */
enum class InstructionKind : uint8_t {
  Kernel = 0,
};

/**
 * Lightweight reference to an instruction's location in the graph.
 * Returned by the use-def analysis helpers (producer, users, find_ops).
 */
struct InstructionRef {
  uint32_t chain_idx;
  uint32_t instr_idx;
};

static constexpr InstructionRef kNoProducer = {UINT32_MAX, UINT32_MAX};

/**
 * Thin wrapper around ExecuTorch's flatbuffer KernelCall providing
 * convenient access to op name and input/output value_ids.
 *
 * ExecuTorch packs all op args into a single `args` array per the
 * convention "the last arg is the (single) output." This wrapper
 * exposes inputs/outputs accordingly without copying — `inputs()` /
 * `output()` return Spans / values that reference the underlying
 * flatbuffer storage directly.
 *
 * Per-call cost: just stores two pointers + an int. No allocations.
 * Safe to construct repeatedly in hot dispatch loops.
 *
 * NOTE: Multi-output ops (e.g., `aten.split`, `aten.max.dim`) are not
 * supported by this wrapper — `num_outputs()` always returns 0 or 1.
 * Adding multi-output support requires per-op schema knowledge that
 * isn't in the flatbuffer. ET_CHECK guards the single-output access
 * path.
 */
class OperatorCall {
 public:
  explicit OperatorCall(
      const executorch_flatbuffer::KernelCall* call,
      const Graph* graph)
      : call_(call), graph_(graph) {}

  // node_id for error messages/profiling
  uint32_t node_id() const {
    return node_id_;
  }
  void set_node_id(uint32_t id) {
    node_id_ = id;
  }

  // Op base name (e.g., "aten::add"). Does NOT include the overload
  // suffix; for that use full_name().
  const char* name() const;

  // Op overload (e.g., "Tensor", "Scalar", "out"). May be empty string
  // for ops with no overload disambiguation. Use full_name() to get
  // the combined "name.overload" form most callers want.
  const char* overload() const;

  // Combined unique key "<name>.<overload>" (e.g., "aten::add.Tensor"),
  // or just "<name>" if the overload is empty. This is the canonical
  // identity used by the CPU op registry to dispatch — registering
  // by base name alone collapses every overload of an op into one
  // handler, which is wrong for ops like aten::pow_ that have multiple
  // overloads with different schemas.
  std::string full_name() const;

  // All op args (flat). Last entry is the output; the rest are inputs.
  // Optional args are NOT indicated by -1 — instead the index points to
  // a value with isNone() == true. (ExecuTorch convention.)
  runtime::Span<const int32_t> args() const {
    auto* a = call_->args();
    return a ? runtime::Span<const int32_t>(a->data(), a->size())
             : runtime::Span<const int32_t>{};
  }

  // Inputs = args[0..n-2]. Returns empty span if op has no args.
  runtime::Span<const int32_t> inputs() const {
    auto a = args();
    return a.empty() ? a : runtime::Span<const int32_t>(a.data(), a.size() - 1);
  }

  size_t num_inputs() const {
    auto a = args();
    return a.empty() ? 0 : a.size() - 1;
  }
  uint32_t input(size_t i) const {
    ET_CHECK_MSG(
        i < num_inputs(),
        "OperatorCall::input: index %zu >= num_inputs()=%zu",
        i,
        num_inputs());
    return static_cast<uint32_t>(args()[i]);
  }

  // Single-output assumption (see class doc). Multi-output ops will
  // break here.
  size_t num_outputs() const {
    return args().empty() ? 0 : 1;
  }
  uint32_t output(size_t i) const {
    ET_CHECK_MSG(
        i < num_outputs(),
        "OperatorCall::output: index %zu >= num_outputs()=%zu",
        i,
        num_outputs());
    auto a = args();
    return static_cast<uint32_t>(a[a.size() - 1]);
  }

 private:
  const executorch_flatbuffer::KernelCall* call_;
  const Graph* graph_;
  uint32_t node_id_ = 0;
};

/**
 * IR view of a program. Adapts executorch_flatbuffer::ExecutionPlan;
 * exposes value metadata, input/output IDs, the operator table, and
 * chains of operator calls. See the file-header comment for the
 * adapter-pattern rationale.
 */
class Graph {
 public:
  explicit Graph(
      const executorch_flatbuffer::ExecutionPlan* plan,
      const executorch_flatbuffer::Program* program = nullptr)
      : plan_(plan), program_(program) {
    // Precompute input/output value_id sets for O(1) value_kind lookup.
    if (auto* in = plan_->inputs()) {
      input_ids_.reserve(in->size());
      for (size_t i = 0; i < in->size(); ++i) {
        input_ids_.insert(static_cast<uint32_t>(in->Get(i)));
      }
    }
    if (auto* out = plan_->outputs()) {
      output_ids_.reserve(out->size());
      for (size_t i = 0; i < out->size(); ++i) {
        output_ids_.insert(static_cast<uint32_t>(out->Get(i)));
      }
    }

    // Precompute mem_obj_id for every tensor value.
    //
    // Algorithm: collect (pool_id, offset) keys for all aliasable tensor
    // values; sort the unique keys; assign mem_obj_id = sort rank. Two
    // values with the same (pool_id, offset) get the same id (they share
    // storage). Sort-and-index is deterministic across runs and depends
    // only on the AOT memory plan.
    size_t n_vals = num_values();
    mem_obj_ids_.assign(n_vals, -1);
    if (n_vals == 0)
      return;

    // 1. Collect (key, value_id) entries for tensor values with
    // allocation_info.
    struct Entry {
      uint64_t key; // (pool_id << 32) | offset
      uint32_t value_id;
    };
    std::vector<Entry> entries;
    entries.reserve(n_vals);
    for (uint32_t i = 0; i < n_vals; ++i) {
      auto* val = value_meta(i);
      if (!val ||
          val->val_type() != executorch_flatbuffer::KernelTypes::Tensor) {
        continue;
      }
      auto* t = val->val_as_Tensor();
      if (!t)
        continue;
      auto* alloc = t->allocation_info();
      if (!alloc)
        continue;
      uint64_t pool = static_cast<uint32_t>(alloc->memory_id());
      uint64_t off = alloc->memory_offset_low();
      entries.push_back({(pool << 32) | off, i});
    }
    if (entries.empty())
      return;

    // 2. Sort by key (lex order on (pool_id, offset)).
    std::sort(
        entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
          return a.key < b.key;
        });

    // 3. Assign mem_obj_id = sort rank (same key → same id).
    int32_t next_id = -1;
    uint64_t prev_key = ~0ULL;
    for (const auto& e : entries) {
      if (e.key != prev_key) {
        ++next_id;
        prev_key = e.key;
      }
      mem_obj_ids_[e.value_id] = next_id;
    }

    // Precompute mutable_buffer_ids_: tensor values with allocation_info,
    // not graph IO, not constants, and NOT produced by any op (i.e.
    // placeholders that aren't graph inputs). These are mutable buffer
    // placeholders pulled into the delegate by tag_mutated_buffer; their
    // state persists across execute() calls.
    //
    // Used by the router to distinguish semantic alias groups (buffer
    // mutation: AOT spec-shared the buffer placeholder with its mutation
    // source) from lifetime-reuse aliasing (the planner happened to put
    // two values at the same offset because their lifetimes don't
    // overlap). Only semantic groups need the "all touching ops on same
    // runtime else home=host" coordination.
    {
      // Collect all op-output value_ids across all chains.
      std::unordered_set<uint32_t> produced_vids;
      for (size_t ci = 0; ci < num_chains(); ++ci) {
        auto chains = plan_->chains();
        auto instrs = chains->Get(ci)->instructions();
        size_t n_instr = instrs ? instrs->size() : 0;
        for (size_t oi = 0; oi < n_instr; ++oi) {
          OperatorCall op = get_op(ci, oi);
          for (size_t k = 0; k < op.num_outputs(); ++k) {
            produced_vids.insert(op.output(k));
          }
        }
      }
      for (uint32_t i = 0; i < n_vals; ++i) {
        if (mem_obj_ids_[i] < 0)
          continue; // not an allocated tensor
        if (input_ids_.count(i) > 0)
          continue; // graph input
        if (output_ids_.count(i) > 0)
          continue; // graph output
        if (tensor_constant_data_key(i) != nullptr)
          continue; // constant
        if (produced_vids.count(i) > 0)
          continue; // produced by an op
        // Tensor with alloc, not IO, not constant, not produced → it's a
        // mutable buffer placeholder.
        mutable_buffer_ids_.push_back(i);
      }
    }

    // Precompute use-def analysis indices (producer, users, op name index).
    // Single pass over all instructions; CSR build for users.
    producers_.assign(n_vals, kNoProducer);
    std::vector<uint32_t> user_counts(n_vals, 0);

    auto record_user = [&](uint32_t vid) {
      if (vid < n_vals)
        ++user_counts[vid];
    };
    auto record_producer = [&](uint32_t vid, InstructionRef ref) {
      if (vid < n_vals)
        producers_[vid] = ref;
    };

    // Pass 1: count users per value and record producers.
    for (size_t ci = 0; ci < num_chains(); ++ci) {
      auto instrs = plan_->chains()->Get(ci)->instructions();
      size_t n_instr = instrs ? instrs->size() : 0;
      for (size_t ii = 0; ii < n_instr; ++ii) {
        InstructionRef ref{
            static_cast<uint32_t>(ci), static_cast<uint32_t>(ii)};
        OperatorCall op = get_op(ci, ii);
        for (size_t j = 0; j < op.num_inputs(); ++j)
          record_user(op.input(j));
        for (size_t j = 0; j < op.num_outputs(); ++j)
          record_producer(op.output(j), ref);
        const char* op_name = op.name();
        if (op_name)
          op_name_index_[op_name].push_back(ref);
      }
    }

    // Pass 2: build CSR from counts.
    user_starts_.resize(n_vals + 1);
    user_starts_[0] = 0;
    for (uint32_t i = 0; i < n_vals; ++i)
      user_starts_[i + 1] = user_starts_[i] + user_counts[i];
    user_entries_.resize(user_starts_[n_vals]);

    // Reuse user_counts as write cursors.
    std::fill(user_counts.begin(), user_counts.end(), 0);

    auto emit_user = [&](uint32_t vid, InstructionRef ref) {
      if (vid < n_vals) {
        user_entries_[user_starts_[vid] + user_counts[vid]] = ref;
        ++user_counts[vid];
      }
    };

    // Pass 3: fill user entries (same traversal as pass 1).
    for (size_t ci = 0; ci < num_chains(); ++ci) {
      auto instrs = plan_->chains()->Get(ci)->instructions();
      size_t n_instr = instrs ? instrs->size() : 0;
      for (size_t ii = 0; ii < n_instr; ++ii) {
        InstructionRef ref{
            static_cast<uint32_t>(ci), static_cast<uint32_t>(ii)};
        OperatorCall op = get_op(ci, ii);
        for (size_t j = 0; j < op.num_inputs(); ++j)
          emit_user(op.input(j), ref);
      }
    }
  }

  //===------------------------------------------------------------------===//
  // Version
  //===------------------------------------------------------------------===//

  // cppcheck-suppress functionStatic
  const char* version() const {
    // TODO: Return actual version when available
    return "1.0";
  }

  //===------------------------------------------------------------------===//
  // Values
  //===------------------------------------------------------------------===//

  size_t num_values() const {
    auto v = plan_->values();
    return v ? v->size() : 0;
  }

  // Access serialized value metadata.
  // NOTE: returns the raw flatbuffer EValue. This is the construction-
  // seam escape hatch — backends and routers should prefer the typed
  // accessors below (value_type, int_value, tensor_*, etc) so they
  // don't couple to the underlying serialization.
  const executorch_flatbuffer::EValue* value_meta(uint32_t value_id) const {
    auto values = plan_->values();
    if (!values || value_id >= values->size())
      return nullptr;
    return values->Get(value_id);
  }

  // Value metadata helpers
  ValueKind value_kind(uint32_t value_id) const;
  int32_t mem_obj_id(uint32_t value_id) const;

  //===------------------------------------------------------------------===//
  // Typed value accessors (adapter-level — no flatbuffer types leak)
  //===------------------------------------------------------------------===//

  // Returns the kind of the EValue stored at value_id.
  ValueType value_type(uint32_t value_id) const;

  // Scalar accessors — ET_CHECK if the value isn't of the expected kind.
  int64_t int_value(uint32_t value_id) const;
  double double_value(uint32_t value_id) const;
  bool bool_value(uint32_t value_id) const;

  // Tensor accessors — ET_CHECK if the value isn't a tensor.
  ::executorch::aten::ScalarType tensor_dtype(uint32_t value_id) const;
  ::executorch::runtime::Span<const int32_t> tensor_sizes(
      uint32_t value_id) const;
  ::executorch::runtime::Span<const uint8_t> tensor_dim_order(
      uint32_t value_id) const;
  ::executorch::aten::TensorShapeDynamism tensor_shape_dynamism(
      uint32_t value_id) const;
  // Returns NDM key (FQN) for an external constant tensor, or nullptr
  // if the tensor isn't an NDM-stored constant.
  const char* tensor_constant_data_key(uint32_t value_id) const;

  // Returns the raw bytes of an inline constant (stored in the
  // program's constant_buffer field), or an empty span if the tensor
  // isn't an inline constant. Inline constants are constants the AOT
  // didn't promote to NDM (e.g., literals lifted into _lifted_tensor_*
  // placeholders). Mutually exclusive with tensor_constant_data_key:
  // a constant is either NDM-stored (key != nullptr) or inline
  // (this returns non-empty), never both.
  ::executorch::runtime::Span<const uint8_t> tensor_inline_data(
      uint32_t value_id) const;

  // True if the tensor is a constant of either flavor (NDM-stored or
  // inline). Use this for "is this an immutable constant?" filtering
  // checks in the router; the source matters only at upload time.
  bool is_constant(uint32_t value_id) const {
    if (tensor_constant_data_key(value_id) != nullptr)
      return true;
    return !tensor_inline_data(value_id).empty();
  }

  // dtype-size × prod(sizes); 0 if not a tensor or sizes empty.
  size_t tensor_nbytes_max(uint32_t value_id) const;

  // IntList accessors — ET_CHECK if the value isn't an IntList.
  // Returns the EValue indices that the list elements reference (stored
  // as int64 in the serialization); the caller resolves them through
  // the values array.
  ::executorch::runtime::Span<const int64_t> int_list_member_ids(
      uint32_t value_id) const;

  // For a TensorList or OptionalTensorList value, returns the EValue
  // indices that the list contains. For OptionalTensorList, indices may
  // point at None values (representing nullopt).
  ::executorch::runtime::Span<const int32_t> tensor_list_member_ids(
      uint32_t value_id) const;

  //===------------------------------------------------------------------===//
  // Input/Output IDs
  //===------------------------------------------------------------------===//

  size_t num_input_ids() const {
    auto in = plan_->inputs();
    return in ? in->size() : 0;
  }

  uint32_t input_id(size_t i) const {
    auto in = plan_->inputs();
    ET_CHECK_MSG(
        in && i < in->size(),
        "Graph::input_id(%zu) out of range (have %zu inputs)",
        i,
        in ? in->size() : 0);
    return static_cast<uint32_t>(in->Get(i));
  }

  size_t num_output_ids() const {
    auto out = plan_->outputs();
    return out ? out->size() : 0;
  }

  uint32_t output_id(size_t i) const {
    auto out = plan_->outputs();
    ET_CHECK_MSG(
        out && i < out->size(),
        "Graph::output_id(%zu) out of range (have %zu outputs)",
        i,
        out ? out->size() : 0);
    return static_cast<uint32_t>(out->Get(i));
  }

  //===------------------------------------------------------------------===//
  // Mutable Buffer IDs (values that persist across execute() calls)
  //===------------------------------------------------------------------===//

  size_t num_mutable_buffer_ids() const {
    return mutable_buffer_ids_.size();
  }

  uint32_t mutable_buffer_id(size_t i) const {
    ET_CHECK_MSG(
        i < mutable_buffer_ids_.size(),
        "Graph::mutable_buffer_id: index %zu out of range "
        "(have %zu mutable buffers)",
        i,
        mutable_buffer_ids_.size());
    return mutable_buffer_ids_[i];
  }

  //===------------------------------------------------------------------===//
  // Use-def analysis (precomputed at construction)
  //===------------------------------------------------------------------===//

  // Which instruction produces this value? Returns nullptr if the value
  // has no producer (graph input, constant, or mutable buffer placeholder).
  const InstructionRef* producer(uint32_t value_id) const {
    if (value_id >= producers_.size())
      return nullptr;
    const auto& ref = producers_[value_id];
    if (ref.chain_idx == kNoProducer.chain_idx &&
        ref.instr_idx == kNoProducer.instr_idx)
      return nullptr;
    return &ref;
  }

  // All instructions that consume this value (as an input, move source,
  // jump condition, or free target).
  ::executorch::runtime::Span<const InstructionRef> users(
      uint32_t value_id) const {
    if (value_id >= producers_.size())
      return {};
    uint32_t start = user_starts_[value_id];
    uint32_t end = user_starts_[value_id + 1];
    return ::executorch::runtime::Span<const InstructionRef>(
        user_entries_.data() + start, end - start);
  }

  size_t num_users(uint32_t value_id) const {
    if (value_id >= producers_.size())
      return 0;
    return user_starts_[value_id + 1] - user_starts_[value_id];
  }

  // All KernelCall instructions whose operator base name matches.
  // Returns empty span if no matches.
  ::executorch::runtime::Span<const InstructionRef> find_ops(
      const char* name) const {
    auto it = op_name_index_.find(name);
    if (it == op_name_index_.end())
      return {};
    return ::executorch::runtime::Span<const InstructionRef>(
        it->second.data(), it->second.size());
  }

  //===------------------------------------------------------------------===//
  // Operators (for op name lookup)
  //===------------------------------------------------------------------===//

  size_t num_operators() const {
    auto ops = plan_->operators();
    return ops ? ops->size() : 0;
  }

  const char* operator_name(size_t idx) const {
    auto ops = plan_->operators();
    if (!ops || idx >= ops->size())
      return nullptr;
    auto op = ops->Get(idx);
    return op && op->name() ? op->name()->c_str() : nullptr;
  }

  const char* operator_overload(size_t idx) const {
    auto ops = plan_->operators();
    if (!ops || idx >= ops->size())
      return nullptr;
    auto op = ops->Get(idx);
    return op && op->overload() ? op->overload()->c_str() : nullptr;
  }

  //===------------------------------------------------------------------===//
  // Chains
  //===------------------------------------------------------------------===//

  size_t num_chains() const {
    auto chains = plan_->chains();
    return chains ? chains->size() : 0;
  }

  // cppcheck-suppress functionStatic
  int32_t main_chain_idx() const {
    return 0; // Default: first chain is main
  }

  // Get number of ops in a chain
  size_t num_ops_in_chain(size_t chain_idx) const {
    auto chains = plan_->chains();
    ET_CHECK_MSG(
        chains && chain_idx < chains->size(),
        "Graph::num_ops_in_chain(%zu) out of range (have %zu chains)",
        chain_idx,
        chains ? chains->size() : 0);
    auto instrs = chains->Get(chain_idx)->instructions();
    return instrs ? instrs->size() : 0;
  }

  // Get OperatorCall for op in chain
  // PRECONDITION: instruction_kind(chain_idx, op_idx) ==
  // InstructionKind::Kernel. Use instruction_kind() first to dispatch on kind.
  OperatorCall get_op(size_t chain_idx, size_t op_idx) const {
    auto chains = plan_->chains();
    ET_CHECK_MSG(
        chains && chain_idx < chains->size(),
        "Graph::get_op: chain_idx=%zu out of range (have %zu chains)",
        chain_idx,
        chains ? chains->size() : 0);
    auto instrs = chains->Get(chain_idx)->instructions();
    ET_CHECK_MSG(
        instrs && op_idx < instrs->size(),
        "Graph::get_op: op_idx=%zu out of range in chain %zu "
        "(have %zu ops)",
        op_idx,
        chain_idx,
        instrs ? instrs->size() : 0);
    auto instr = instrs->Get(op_idx);
    ET_CHECK_MSG(
        instr->instr_args_type() ==
            executorch_flatbuffer::InstructionArguments::KernelCall,
        "Graph::get_op: instruction at chain=%zu op_idx=%zu is not a KernelCall "
        "(type=%u). Use instruction_kind() to dispatch.",
        chain_idx,
        op_idx,
        static_cast<unsigned>(instr->instr_args_type()));
    auto kernel = static_cast<const executorch_flatbuffer::KernelCall*>(
        instr->instr_args());
    return OperatorCall(kernel, this);
  }

  //===------------------------------------------------------------------===//
  // Typed instruction accessors
  //===------------------------------------------------------------------===//

  InstructionKind instruction_kind(size_t chain_idx, size_t op_idx) const {
    auto chains = plan_->chains();
    ET_CHECK_MSG(
        chains && chain_idx < chains->size(),
        "Graph::instruction_kind: chain_idx=%zu out of range (have %zu chains)",
        chain_idx,
        chains ? chains->size() : 0);
    auto instrs = chains->Get(chain_idx)->instructions();
    ET_CHECK_MSG(
        instrs && op_idx < instrs->size(),
        "Graph::instruction_kind: op_idx=%zu out of range in chain %zu",
        op_idx,
        chain_idx);
    auto instr = instrs->Get(op_idx);
    using IA = executorch_flatbuffer::InstructionArguments;
    ET_CHECK_MSG(
        instr->instr_args_type() == IA::KernelCall,
        "Graph: non-Kernel instruction at chain=%zu op_idx=%zu (type=%u). "
        "Control flow is not supported in the native backend.",
        chain_idx,
        op_idx,
        static_cast<unsigned>(instr->instr_args_type()));
    return InstructionKind::Kernel;
  }

  InstructionKind instruction_kind(size_t op_idx) const {
    return instruction_kind(main_chain_idx(), op_idx);
  }

  OperatorCall get_kernel_call(size_t chain_idx, size_t op_idx) const {
    return get_op(chain_idx, op_idx);
  }

  //===------------------------------------------------------------------===//
  // Convenience: main chain accessors
  //===------------------------------------------------------------------===//

  size_t num_instructions() const {
    return num_ops_in_chain(main_chain_idx());
  }

  OperatorCall get_instruction(size_t idx) const {
    return get_op(main_chain_idx(), idx);
  }

 private:
  const executorch_flatbuffer::ExecutionPlan* plan_;
  // Optional reference to the parent Program, needed only for
  // tensor_inline_data() (which dereferences program_->constant_buffer).
  // Pre-existing constructions that pass only the plan get nullptr and
  // tensor_inline_data() returns empty for them.
  const executorch_flatbuffer::Program* program_;
  // Precomputed at construction for O(1) value_kind lookup.
  std::unordered_set<uint32_t> input_ids_;
  std::unordered_set<uint32_t> output_ids_;
  // mem_obj_ids_[value_id] = dense small int identifying the storage slot
  // (sort rank of (pool_id, offset) pairs across all aliasable tensor
  // values). -1 for non-tensor / non-allocated values. Same id ⇒ same
  // storage. Computed once at construction; O(1) lookup at use sites.
  std::vector<int32_t> mem_obj_ids_;

  // Mutable buffer placeholder value_ids: tensor values with allocation
  // info that aren't graph IO, aren't constants, and aren't produced by
  // any op. These persist across execute() calls (their storage is
  // preserved between invocations). Identified by tag_mutated_buffer at
  // AOT time.
  std::vector<uint32_t> mutable_buffer_ids_;

  // Use-def analysis indices (precomputed at construction).
  std::vector<InstructionRef> producers_; // indexed by value_id
  std::vector<uint32_t> user_starts_; // CSR offsets, size = num_values + 1
  std::vector<InstructionRef> user_entries_; // CSR entries
  std::unordered_map<std::string, std::vector<InstructionRef>> op_name_index_;
};

// Implement OperatorCall::name() after Graph is defined
inline const char* OperatorCall::name() const {
  // In ExecuTorch, op names are in the operators table, indexed by op_index
  return graph_->operator_name(call_->op_index());
}

inline const char* OperatorCall::overload() const {
  return graph_->operator_overload(call_->op_index());
}

inline std::string OperatorCall::full_name() const {
  const char* base = name();
  const char* ovl = overload();
  if (!base)
    return {};
  if (!ovl || *ovl == '\0')
    return std::string(base);
  std::string s;
  s.reserve(std::strlen(base) + 1 + std::strlen(ovl));
  s.append(base);
  s.push_back('.');
  s.append(ovl);
  return s;
}

// Implement value metadata accessors
inline ValueKind Graph::value_kind(uint32_t value_id) const {
  if (input_ids_.count(value_id))
    return ValueKind::INPUT;
  if (output_ids_.count(value_id))
    return ValueKind::OUTPUT;

  // Constant if the tensor has a baked data buffer.
  auto val = value_meta(value_id);
  if (val && val->val_type() == executorch_flatbuffer::KernelTypes::Tensor) {
    auto* tensor = val->val_as_Tensor();
    if (tensor && tensor->data_buffer_idx() > 0) {
      return ValueKind::CONSTANT;
    }
  }
  return ValueKind::INTERMEDIATE;
}

inline int32_t Graph::mem_obj_id(uint32_t value_id) const {
  return value_id < mem_obj_ids_.size() ? mem_obj_ids_[value_id] : -1;
}

//===----------------------------------------------------------------------===//
// Typed value accessors
//===----------------------------------------------------------------------===//

inline ValueType Graph::value_type(uint32_t value_id) const {
  auto* val = value_meta(value_id);
  if (!val)
    return ValueType::None;
  using KT = executorch_flatbuffer::KernelTypes;
  switch (val->val_type()) {
    case KT::Null:
      return ValueType::None;
    case KT::Int:
      return ValueType::Int;
    case KT::Double:
      return ValueType::Double;
    case KT::Bool:
      return ValueType::Bool;
    case KT::Tensor:
      return ValueType::Tensor;
    case KT::IntList:
      return ValueType::IntList;
    case KT::TensorList:
      return ValueType::TensorList;
    case KT::OptionalTensorList:
      return ValueType::OptionalTensorList;
    default:
      return ValueType::Other;
  }
}

inline int64_t Graph::int_value(uint32_t value_id) const {
  auto* val = value_meta(value_id);
  ET_CHECK_MSG(
      val && val->val_type() == executorch_flatbuffer::KernelTypes::Int,
      "Graph::int_value(%u): value is not an Int",
      value_id);
  return static_cast<const executorch_flatbuffer::Int*>(val->val())->int_val();
}

inline double Graph::double_value(uint32_t value_id) const {
  auto* val = value_meta(value_id);
  ET_CHECK_MSG(
      val && val->val_type() == executorch_flatbuffer::KernelTypes::Double,
      "Graph::double_value(%u): value is not a Double",
      value_id);
  return static_cast<const executorch_flatbuffer::Double*>(val->val())
      ->double_val();
}

inline bool Graph::bool_value(uint32_t value_id) const {
  auto* val = value_meta(value_id);
  ET_CHECK_MSG(
      val && val->val_type() == executorch_flatbuffer::KernelTypes::Bool,
      "Graph::bool_value(%u): value is not a Bool",
      value_id);
  return static_cast<const executorch_flatbuffer::Bool*>(val->val())
      ->bool_val();
}

namespace detail {
inline const executorch_flatbuffer::Tensor* tensor_or_null(
    const executorch_flatbuffer::EValue* val) {
  if (!val || val->val_type() != executorch_flatbuffer::KernelTypes::Tensor) {
    return nullptr;
  }
  return val->val_as_Tensor();
}
} // namespace detail

inline ::executorch::aten::ScalarType Graph::tensor_dtype(
    uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  ET_CHECK_MSG(t, "Graph::tensor_dtype(%u): value is not a Tensor", value_id);
  return static_cast<::executorch::aten::ScalarType>(t->scalar_type());
}

inline ::executorch::runtime::Span<const int32_t> Graph::tensor_sizes(
    uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  ET_CHECK_MSG(t, "Graph::tensor_sizes(%u): value is not a Tensor", value_id);
  auto* s = t->sizes();
  return s ? ::executorch::runtime::Span<const int32_t>(s->data(), s->size())
           : ::executorch::runtime::Span<const int32_t>{};
}

inline ::executorch::runtime::Span<const uint8_t> Graph::tensor_dim_order(
    uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  ET_CHECK_MSG(
      t, "Graph::tensor_dim_order(%u): value is not a Tensor", value_id);
  auto* d = t->dim_order();
  return d ? ::executorch::runtime::Span<const uint8_t>(d->data(), d->size())
           : ::executorch::runtime::Span<const uint8_t>{};
}

inline ::executorch::aten::TensorShapeDynamism Graph::tensor_shape_dynamism(
    uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  ET_CHECK_MSG(
      t, "Graph::tensor_shape_dynamism(%u): value is not a Tensor", value_id);
  return static_cast<::executorch::aten::TensorShapeDynamism>(
      t->shape_dynamism());
}

inline const char* Graph::tensor_constant_data_key(uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  if (!t)
    return nullptr;
  auto* eti = t->extra_tensor_info();
  if (!eti)
    return nullptr;
  if (eti->location() != executorch_flatbuffer::TensorDataLocation::EXTERNAL) {
    return nullptr;
  }
  auto* fqn = eti->fully_qualified_name();
  return (fqn && fqn->size() > 0) ? fqn->c_str() : nullptr;
}

inline ::executorch::runtime::Span<const uint8_t> Graph::tensor_inline_data(
    uint32_t value_id) const {
  if (!program_)
    return {};
  auto* t = detail::tensor_or_null(value_meta(value_id));
  if (!t)
    return {};
  uint32_t idx = static_cast<uint32_t>(t->data_buffer_idx());
  // Index 0 is reserved (placeholder for "no inline data"). External
  // constants also have idx == 0; they're handled by
  // tensor_constant_data_key.
  if (idx == 0)
    return {};
  auto* buffers = program_->constant_buffer();
  if (!buffers || idx >= buffers->size())
    return {};
  auto* buf = buffers->Get(idx);
  if (!buf)
    return {};
  auto* storage = buf->storage();
  if (!storage || storage->size() == 0)
    return {};
  return ::executorch::runtime::Span<const uint8_t>(
      storage->data(), storage->size());
}

inline size_t Graph::tensor_nbytes_max(uint32_t value_id) const {
  auto* t = detail::tensor_or_null(value_meta(value_id));
  if (!t || !t->sizes())
    return 0;
  size_t numel = 1;
  for (size_t i = 0; i < t->sizes()->size(); ++i) {
    int dim = t->sizes()->Get(i);
    if (dim < 0)
      return 0;
    numel *= static_cast<size_t>(dim);
  }
  auto stype = static_cast<::executorch::aten::ScalarType>(t->scalar_type());
  return numel * ::executorch::runtime::elementSize(stype);
}

inline ::executorch::runtime::Span<const int64_t> Graph::int_list_member_ids(
    uint32_t value_id) const {
  auto* val = value_meta(value_id);
  ET_CHECK_MSG(
      val && val->val_type() == executorch_flatbuffer::KernelTypes::IntList,
      "Graph::int_list_member_ids(%u): value is not an IntList",
      value_id);
  auto* items =
      static_cast<const executorch_flatbuffer::IntList*>(val->val())->items();
  return items
      ? ::executorch::runtime::Span<const int64_t>(items->data(), items->size())
      : ::executorch::runtime::Span<const int64_t>{};
}

inline ::executorch::runtime::Span<const int32_t> Graph::tensor_list_member_ids(
    uint32_t value_id) const {
  auto* val = value_meta(value_id);
  ET_CHECK_MSG(
      val &&
          (val->val_type() == executorch_flatbuffer::KernelTypes::TensorList ||
           val->val_type() ==
               executorch_flatbuffer::KernelTypes::OptionalTensorList),
      "Graph::tensor_list_member_ids(%u): value is not a TensorList "
      "or OptionalTensorList",
      value_id);
  // Both TensorList and OptionalTensorList have the same shape:
  // table { items: [int]; }. Cast to either to access items().
  const flatbuffers::Vector<int32_t>* items = nullptr;
  if (val->val_type() == executorch_flatbuffer::KernelTypes::TensorList) {
    items = static_cast<const executorch_flatbuffer::TensorList*>(val->val())
                ->items();
  } else {
    items = static_cast<const executorch_flatbuffer::OptionalTensorList*>(
                val->val())
                ->items();
  }
  return items
      ? ::executorch::runtime::Span<const int32_t>(items->data(), items->size())
      : ::executorch::runtime::Span<const int32_t>{};
}

} // namespace portable
} // namespace backends
} // namespace executorch
