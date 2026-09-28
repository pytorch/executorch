/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/backend_data/initialize_and_save.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include <executorch/extension/flat_tensor/flat_tensor_data_map.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_generated.h>
#include <executorch/extension/flat_tensor/serialize/flat_tensor_header.h>
#include <executorch/runtime/backend/backend_data.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/schema/extended_header.h>
#include <executorch/schema/program_generated.h>

            namespace executorch::extension {
  namespace {

  using runtime::BackendData;
  using runtime::BackendDataInitContext;
  using runtime::BackendDataInput;
  using runtime::BackendDataWriter;
  using runtime::CompileSpec;
  using runtime::DataLoader;
  using runtime::Error;
  using runtime::FreeableBuffer;
  using runtime::NamedBackendData;
  using runtime::NamedDataMap;
  using runtime::Result;
  using runtime::Span;

  enum class SourceKind { Pte, Ptd };

  struct SegmentState {
    size_t original_offset;
    size_t original_size;
    size_t alignment;
    size_t final_offset;
    size_t final_size;
    bool replaced;
    bool written;
  };

  struct Source {
    std::unique_ptr<DataLoader> loader;
    std::unique_ptr<DataWriter> writer;
  };

  struct SourceState {
    SourceKind kind;
    Source source;
    std::vector<uint8_t> metadata;
    size_t segment_base_offset;
    size_t original_segment_data_size;
    std::vector<SegmentState> segments;
    bool has_segment_data_size;
    size_t next_output_offset;
    bool modified{false};
  };

  struct NamedLookup {
    std::string key;
    size_t source_index;
    size_t named_index;
    size_t segment_index;
  };

  struct InputRecord {
    std::string method_name;
    std::string backend_id;
    size_t segment_index;
    std::vector<CompileSpec> compile_specs;
    std::string identity;
    bool yielded{false};
    bool replaced{false};
  };

  struct CurrentInputState {
    InputRecord* record{nullptr};
    std::optional<FreeableBuffer> original;
  };

  class CurrentInputFinalizer {
   public:
    virtual ~CurrentInputFinalizer() = default;
    virtual Error finalize_current_input() = 0;
  };

  bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
  }

  Result<size_t> checked_add(size_t lhs, size_t rhs) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
      return Error::InvalidArgument;
    }
    return lhs + rhs;
  }

  Result<size_t> align_up(size_t value, size_t alignment) {
    if (!is_power_of_two(alignment)) {
      return Error::InvalidArgument;
    }
    const size_t mask = alignment - 1;
    if (value > std::numeric_limits<size_t>::max() - mask) {
      return Error::InvalidArgument;
    }
    return (value + mask) & ~mask;
  }

  uint32_t read_u32_le(const uint8_t* data) {
    uint32_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i) {
      value |= static_cast<uint32_t>(data[i]) << (8 * i);
    }
    return value;
  }

  void write_u64_le(uint8_t* data, uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
      data[i] = static_cast<uint8_t>(value >> (8 * i));
    }
  }

  bool span_is_in_allocator(
      const runtime::MemoryAllocator& allocator,
      const void* data,
      size_t size) {
    if (size == 0 && data == nullptr) {
      return true;
    }
    const auto* base = allocator.base_address();
    if (base == nullptr || data == nullptr) {
      return false;
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(base);
    const size_t used = allocator.used_size();
    if (used > std::numeric_limits<uintptr_t>::max() - begin) {
      return false;
    }
    const uintptr_t end = begin + used;
    const uintptr_t value = reinterpret_cast<uintptr_t>(data);
    return value >= begin && value <= end && size <= end - value;
  }

  size_t infer_segment_alignment(size_t absolute_offset) {
    return absolute_offset & (~absolute_offset + 1);
  }

  Error validate_source(SourceState& source) {
    if (source.source.loader == nullptr) {
      return Error::InvalidArgument;
    }
    auto loader_size = source.source.loader->size();
    if (!loader_size.ok() || source.segment_base_offset > *loader_size ||
        source.original_segment_data_size >
            *loader_size - source.segment_base_offset ||
        source.metadata.size() > source.segment_base_offset) {
      return source.kind == SourceKind::Pte ? Error::InvalidProgram
                                            : Error::InvalidExternalData;
    }
    size_t previous_end = 0;
    for (size_t i = 0; i < source.segments.size(); ++i) {
      auto& segment = source.segments[i];
      auto absolute =
          checked_add(source.segment_base_offset, segment.original_offset);
      if (!absolute.ok() || *absolute == 0 ||
          segment.original_offset > source.original_segment_data_size ||
          segment.original_size >
              source.original_segment_data_size - segment.original_offset ||
          segment.original_offset < previous_end) {
        return Error::InvalidArgument;
      }
      segment.alignment = infer_segment_alignment(*absolute);
      previous_end = segment.original_offset + segment.original_size;
    }
    if (!source.segments.empty() &&
        source.segments.front().original_offset != 0) {
      return Error::NotSupported;
    }
    return Error::Ok;
  }

  Result<SourceState> parse_pte(Source source) {
    if (source.loader == nullptr) {
      return Error::InvalidArgument;
    }
    auto head = source.loader->load(
        0,
        runtime::ExtendedHeader::kNumHeadBytes,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    if (!head.ok()) {
      return head.error();
    }
    auto header = runtime::ExtendedHeader::Parse(head->data(), head->size());
    if (!header.ok() ||
        header->program_size > std::numeric_limits<size_t>::max() ||
        header->segment_base_offset > std::numeric_limits<size_t>::max() ||
        header->segment_data_size > std::numeric_limits<size_t>::max()) {
      return header.ok() ? Error::InvalidProgram : header.error();
    }
    auto loaded = source.loader->load(
        0,
        static_cast<size_t>(header->program_size),
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    if (!loaded.ok()) {
      return loaded.error();
    }
    SourceState state{
        SourceKind::Pte,
        std::move(source),
        std::vector<uint8_t>(
            static_cast<const uint8_t*>(loaded->data()),
            static_cast<const uint8_t*>(loaded->data()) + loaded->size()),
        static_cast<size_t>(header->segment_base_offset),
        static_cast<size_t>(header->segment_data_size)};
    if (state.metadata.size() < 40 ||
        !executorch_flatbuffer::ProgramBufferHasIdentifier(
            state.metadata.data())) {
      return Error::InvalidProgram;
    }
    flatbuffers::Verifier verifier(
        state.metadata.data(), state.metadata.size());
    if (!executorch_flatbuffer::VerifyProgramBuffer(verifier)) {
      return Error::InvalidProgram;
    }
    auto* program =
        executorch_flatbuffer::GetMutableProgram(state.metadata.data());
    if (program->version() > runtime::Program::kMaxSupportedSchemaVersion ||
        program->segments() == nullptr) {
      return Error::InvalidProgram;
    }
    state.segments.reserve(program->segments()->size());
    for (const auto* segment : *program->segments()) {
      if (segment == nullptr ||
          segment->offset() > std::numeric_limits<size_t>::max() ||
          segment->size() > std::numeric_limits<size_t>::max()) {
        return Error::InvalidProgram;
      }
      state.segments.push_back(SegmentState{
          static_cast<size_t>(segment->offset()),
          static_cast<size_t>(segment->size()),
          0,
          static_cast<size_t>(segment->offset()),
          static_cast<size_t>(segment->size()),
          false,
          false});
    }
    state.has_segment_data_size = read_u32_le(state.metadata.data() + 12) >= 32;
    if (!state.has_segment_data_size) {
      state.original_segment_data_size = 0;
      for (const auto& segment : state.segments) {
        auto end = checked_add(segment.original_offset, segment.original_size);
        if (!end.ok()) {
          return end.error();
        }
        state.original_segment_data_size =
            std::max(state.original_segment_data_size, *end);
      }
    }
    state.next_output_offset = state.segment_base_offset;
    Error error = validate_source(state);
    if (error != Error::Ok) {
      return error;
    }
    return std::move(state);
  }

  Result<SourceState> parse_ptd(Source source) {
    if (source.loader == nullptr) {
      return Error::InvalidArgument;
    }
    auto head = source.loader->load(
        0,
        FlatTensorHeader::kNumHeadBytes,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    if (!head.ok()) {
      return head.error();
    }
    auto header = FlatTensorHeader::Parse(head->data(), head->size());
    if (!header.ok() ||
        header->flatbuffer_offset > std::numeric_limits<size_t>::max() ||
        header->flatbuffer_size > std::numeric_limits<size_t>::max() ||
        header->segment_base_offset > std::numeric_limits<size_t>::max() ||
        header->segment_data_size > std::numeric_limits<size_t>::max()) {
      return header.ok() ? Error::InvalidExternalData : header.error();
    }
    auto metadata_size = checked_add(
        static_cast<size_t>(header->flatbuffer_offset),
        static_cast<size_t>(header->flatbuffer_size));
    if (!metadata_size.ok()) {
      return Error::InvalidExternalData;
    }
    auto loaded = source.loader->load(
        0,
        *metadata_size,
        DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
    if (!loaded.ok()) {
      return loaded.error();
    }
    SourceState state{
        SourceKind::Ptd,
        std::move(source),
        std::vector<uint8_t>(
            static_cast<const uint8_t*>(loaded->data()),
            static_cast<const uint8_t*>(loaded->data()) + loaded->size()),
        static_cast<size_t>(header->segment_base_offset),
        static_cast<size_t>(header->segment_data_size)};
    if (state.metadata.size() < 48 ||
        !flat_tensor_flatbuffer::FlatTensorBufferHasIdentifier(
            state.metadata.data())) {
      return Error::InvalidExternalData;
    }
    flatbuffers::Verifier verifier(
        state.metadata.data(), state.metadata.size());
    if (!flat_tensor_flatbuffer::VerifyFlatTensorBuffer(verifier)) {
      return Error::InvalidExternalData;
    }
    auto* flat_tensor =
        flat_tensor_flatbuffer::GetMutableFlatTensor(state.metadata.data());
    if (flat_tensor->version() >
            FlatTensorDataMap::kMaxSupportedSchemaVersion ||
        flat_tensor->segments() == nullptr ||
        flat_tensor->named_data() == nullptr) {
      return Error::InvalidExternalData;
    }
    state.segments.reserve(flat_tensor->segments()->size());
    for (const auto* segment : *flat_tensor->segments()) {
      if (segment == nullptr ||
          segment->offset() > std::numeric_limits<size_t>::max() ||
          segment->size() > std::numeric_limits<size_t>::max()) {
        return Error::InvalidExternalData;
      }
      state.segments.push_back(SegmentState{
          static_cast<size_t>(segment->offset()),
          static_cast<size_t>(segment->size()),
          0,
          static_cast<size_t>(segment->offset()),
          static_cast<size_t>(segment->size()),
          false,
          false});
    }
    state.has_segment_data_size = true;
    state.next_output_offset = state.segment_base_offset;
    Error error = validate_source(state);
    if (error != Error::Ok) {
      return error;
    }
    return std::move(state);
  }

  class PreparationNamedDataMap final : public NamedDataMap {
   public:
    PreparationNamedDataMap(
        const std::vector<SourceState>* sources,
        const std::vector<NamedLookup>* entries)
        : sources_(sources), entries_(entries) {}

    Result<const runtime::TensorLayout> get_tensor_layout(
        std::string_view key) const override {
      const NamedLookup* entry = find(key);
      if (entry == nullptr) {
        return Error::NotFound;
      }
      const SourceState& source = (*sources_)[entry->source_index];
      if (source.kind == SourceKind::Pte) {
        return Error::NotImplemented;
      }
      const auto* item =
          flat_tensor_flatbuffer::GetFlatTensor(source.metadata.data())
              ->named_data()
              ->Get(entry->named_index);
      const auto* layout = item->tensor_layout();
      if (layout == nullptr || layout->sizes() == nullptr ||
          layout->dim_order() == nullptr) {
        return Error::NotFound;
      }
      return runtime::TensorLayout::create(
          Span<const int32_t>(layout->sizes()->data(), layout->sizes()->size()),
          Span<const uint8_t>(
              layout->dim_order()->data(), layout->dim_order()->size()),
          static_cast<executorch::aten::ScalarType>(layout->scalar_type()));
    }

    Result<FreeableBuffer> get_data(std::string_view key) const override {
      const NamedLookup* entry = find(key);
      if (entry == nullptr) {
        return Error::NotFound;
      }
      const SourceState& source = (*sources_)[entry->source_index];
      const SegmentState& segment = source.segments[entry->segment_index];
      auto offset =
          checked_add(source.segment_base_offset, segment.original_offset);
      if (!offset.ok()) {
        return offset.error();
      }
      return source.source.loader->load(
          *offset,
          segment.original_size,
          DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Constant));
    }

    Error load_data_into(std::string_view key, void* buffer, size_t size)
        const override {
      const NamedLookup* entry = find(key);
      if (entry == nullptr) {
        return Error::NotFound;
      }
      const SourceState& source = (*sources_)[entry->source_index];
      const SegmentState& segment = source.segments[entry->segment_index];
      if (size > segment.original_size) {
        return Error::InvalidArgument;
      }
      auto offset =
          checked_add(source.segment_base_offset, segment.original_offset);
      if (!offset.ok()) {
        return offset.error();
      }
      return source.source.loader->load_into(
          *offset,
          size,
          DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Constant),
          buffer);
    }

    Result<uint32_t> get_num_keys() const override {
      if (entries_->size() > std::numeric_limits<uint32_t>::max()) {
        return Error::InvalidArgument;
      }
      return static_cast<uint32_t>(entries_->size());
    }

    Result<const char*> get_key(uint32_t index) const override {
      if (index >= entries_->size()) {
        return Error::InvalidArgument;
      }
      return (*entries_)[index].key.c_str();
    }

   private:
    const NamedLookup* find(std::string_view key) const {
      for (const auto& entry : *entries_) {
        if (entry.key == key) {
          return &entry;
        }
      }
      return nullptr;
    }

    const std::vector<SourceState>* sources_;
    const std::vector<NamedLookup>* entries_;
  };

  Error collect_named_entries(
      const std::vector<SourceState>& sources,
      std::vector<NamedLookup>* entries) {
    std::set<std::string> keys;
    for (size_t source_index = 0; source_index < sources.size();
         ++source_index) {
      const auto& source = sources[source_index];
      if (source.kind == SourceKind::Pte) {
        const auto* named =
            executorch_flatbuffer::GetProgram(source.metadata.data())
                ->named_data();
        if (named == nullptr) {
          continue;
        }
        for (size_t i = 0; i < named->size(); ++i) {
          const auto* item = named->Get(i);
          if (item == nullptr || item->key() == nullptr ||
              item->segment_index() >= source.segments.size() ||
              !keys.insert(item->key()->str()).second) {
            return Error::InvalidProgram;
          }
          entries->push_back(
              {item->key()->str(), source_index, i, item->segment_index()});
        }
      } else {
        const auto* named =
            flat_tensor_flatbuffer::GetFlatTensor(source.metadata.data())
                ->named_data();
        for (size_t i = 0; i < named->size(); ++i) {
          const auto* item = named->Get(i);
          if (item == nullptr || item->key() == nullptr ||
              item->segment_index() >= source.segments.size() ||
              !keys.insert(item->key()->str()).second) {
            return Error::InvalidExternalData;
          }
          entries->push_back(
              {item->key()->str(), source_index, i, item->segment_index()});
        }
      }
    }
    return Error::Ok;
  }

  std::set<std::string> collect_external_tensor_keys(const SourceState& pte) {
    std::set<std::string> keys;
    const auto* plans = executorch_flatbuffer::GetProgram(pte.metadata.data())
                            ->execution_plan();
    if (plans == nullptr) {
      return keys;
    }
    for (const auto* plan : *plans) {
      if (plan == nullptr || plan->values() == nullptr) {
        continue;
      }
      for (const auto* value : *plan->values()) {
        const auto* tensor =
            value == nullptr ? nullptr : value->val_as_Tensor();
        const auto* info =
            tensor == nullptr ? nullptr : tensor->extra_tensor_info();
        if (info != nullptr &&
            info->location() ==
                executorch_flatbuffer::TensorDataLocation::EXTERNAL &&
            info->fully_qualified_name() != nullptr) {
          keys.insert(info->fully_qualified_name()->str());
        }
      }
    }
    return keys;
  }

  Error collect_segment_uses(
      const std::vector<SourceState>& sources,
      const std::vector<NamedLookup>& named_entries,
      const std::vector<std::unique_ptr<InputRecord>>& inputs,
      std::set<std::pair<size_t, size_t>>* named_segments,
      std::set<std::pair<size_t, size_t>>* delegate_segments,
      std::set<std::pair<size_t, size_t>>* nonreplaceable_segments) {
    std::map<std::pair<size_t, size_t>, size_t> named_counts;
    for (const auto& entry : named_entries) {
      const auto segment =
          std::make_pair(entry.source_index, entry.segment_index);
      named_segments->insert(segment);
      ++named_counts[segment];
    }
    for (const auto& entry : named_counts) {
      if (entry.second > 1) {
        nonreplaceable_segments->insert(entry.first);
      }
    }
    for (const auto& input : inputs) {
      delegate_segments->insert({0, input->segment_index});
    }

    const auto* program =
        executorch_flatbuffer::GetProgram(sources[0].metadata.data());
    const auto add_pte_segment = [&](uint32_t segment_index) {
      if (segment_index >= sources[0].segments.size()) {
        return false;
      }
      nonreplaceable_segments->insert({0, segment_index});
      return true;
    };
    if (program->constant_segment() != nullptr &&
        !add_pte_segment(program->constant_segment()->segment_index())) {
      return Error::InvalidProgram;
    }
    if (program->mutable_data_segments() != nullptr) {
      for (const auto* mutable_segment : *program->mutable_data_segments()) {
        if (mutable_segment == nullptr ||
            !add_pte_segment(mutable_segment->segment_index())) {
          return Error::InvalidProgram;
        }
      }
    }
    return Error::Ok;
  }

  std::string input_identity(
      size_t segment_index,
      const executorch_flatbuffer::BackendDelegate& delegate) {
    std::string result;
    const auto append_size = [&result](size_t value) {
      for (size_t i = 0; i < sizeof(value); ++i) {
        result.push_back(static_cast<char>(value >> (8 * i)));
      }
    };
    append_size(segment_index);
    const auto* specs = delegate.compile_specs();
    append_size(specs == nullptr ? 0 : specs->size());
    if (specs != nullptr) {
      for (const auto* spec : *specs) {
        append_size(spec->key()->size());
        result.append(spec->key()->data(), spec->key()->size());
        append_size(spec->value()->size());
        result.append(
            reinterpret_cast<const char*>(spec->value()->data()),
            spec->value()->size());
      }
    }
    return result;
  }

  Error collect_inputs(
      SourceState& pte,
      std::vector<std::unique_ptr<InputRecord>>* inputs,
      std::map<std::string, std::vector<InputRecord*>>* by_backend) {
    std::map<size_t, InputRecord*> deduplicated;
    const auto* plans = executorch_flatbuffer::GetProgram(pte.metadata.data())
                            ->execution_plan();
    if (plans == nullptr) {
      return Error::Ok;
    }
    for (const auto* plan : *plans) {
      if (plan == nullptr || plan->name() == nullptr ||
          plan->delegates() == nullptr) {
        continue;
      }
      for (const auto* delegate : *plan->delegates()) {
        if (delegate == nullptr || delegate->id() == nullptr ||
            delegate->processed() == nullptr ||
            delegate->processed()->location() !=
                executorch_flatbuffer::DataLocation::SEGMENT ||
            delegate->processed()->index() >= pte.segments.size()) {
          return Error::NotSupported;
        }
        const auto* specs = delegate->compile_specs();
        if (specs != nullptr) {
          for (const auto* spec : *specs) {
            if (spec == nullptr || spec->key() == nullptr ||
                spec->value() == nullptr) {
              return Error::InvalidProgram;
            }
          }
        }
        const size_t segment_index = delegate->processed()->index();
        std::string identity = input_identity(segment_index, *delegate);
        auto existing = deduplicated.find(segment_index);
        if (existing != deduplicated.end()) {
          if (existing->second->backend_id != delegate->id()->str() ||
              existing->second->identity != identity) {
            return Error::NotSupported;
          }
          continue;
        }
        auto input = std::make_unique<InputRecord>();
        input->method_name = plan->name()->str();
        input->backend_id = delegate->id()->str();
        input->segment_index = segment_index;
        input->identity = std::move(identity);
        if (specs != nullptr) {
          input->compile_specs.reserve(specs->size());
          for (const auto* spec : *specs) {
            input->compile_specs.push_back(CompileSpec{
                spec->key()->c_str(),
                runtime::SizedBuffer{
                    const_cast<uint8_t*>(spec->value()->data()),
                    spec->value()->size()}});
          }
        }
        InputRecord* ptr = input.get();
        deduplicated.emplace(segment_index, ptr);
        (*by_backend)[input->backend_id].push_back(ptr);
        inputs->push_back(std::move(input));
      }
    }
    return Error::Ok;
  }

  class PreparationDataInitContext final : public BackendDataInitContext {
   public:
    PreparationDataInitContext(
        SourceState* pte,
        std::vector<InputRecord*> inputs,
        runtime::MemoryAllocator* temp_allocator,
        runtime::EventTracer* event_tracer,
        const NamedDataMap* named_data_map,
        CurrentInputState* current,
        CurrentInputFinalizer* finalizer)
        : BackendDataInitContext(temp_allocator, event_tracer, named_data_map),
          pte_(pte),
          inputs_(std::move(inputs)),
          current_(current),
          finalizer_(finalizer) {}

    Result<std::optional<BackendDataInput>> next_backend_data() override {
      if (error_ != Error::Ok) {
        return error_;
      }
      error_ = finalizer_->finalize_current_input();
      if (error_ != Error::Ok) {
        return error_;
      }
      current_->original.reset();
      current_->record = nullptr;
      get_temp_allocator()->reset();
      if (next_ == inputs_.size()) {
        return std::optional<BackendDataInput>{};
      }
      InputRecord& input = *inputs_[next_++];
      const SegmentState& segment = pte_->segments[input.segment_index];
      auto offset =
          checked_add(pte_->segment_base_offset, segment.original_offset);
      if (!offset.ok()) {
        error_ = offset.error();
        return error_;
      }
      auto processed = pte_->source.loader->load(
          *offset,
          segment.original_size,
          DataLoader::SegmentInfo(
              DataLoader::SegmentInfo::Type::Backend,
              input.segment_index,
              input.backend_id.c_str()));
      if (!processed.ok()) {
        error_ = processed.error();
        return error_;
      }
      input.yielded = true;
      current_->record = &input;
      current_->original.emplace(std::move(*processed));
      return std::optional<BackendDataInput>(BackendDataInput{
          input.method_name.c_str(),
          Span<const uint8_t>(
              static_cast<const uint8_t*>(current_->original->data()),
              current_->original->size()),
          runtime::ArrayRef<CompileSpec>(
              input.compile_specs.data(), input.compile_specs.size())});
    }

    Error drain() {
      while (true) {
        auto next = next_backend_data();
        if (!next.ok()) {
          return next.error();
        }
        if (!next->has_value()) {
          return Error::Ok;
        }
      }
    }

    Error error() const {
      return error_;
    }

   private:
    SourceState* pte_;
    std::vector<InputRecord*> inputs_;
    CurrentInputState* current_;
    CurrentInputFinalizer* finalizer_;
    size_t next_{0};
    Error error_{Error::Ok};
  };

  class PlanningBackendDataWriter final : public BackendDataWriter,
                                          public CurrentInputFinalizer {
   public:
    PlanningBackendDataWriter(
        std::vector<SourceState>* sources,
        const std::vector<NamedLookup>* named_entries,
        const std::set<std::string>* external_tensor_keys,
        const std::set<std::pair<size_t, size_t>>* named_segments,
        const std::set<std::pair<size_t, size_t>>* delegate_segments,
        const std::set<std::pair<size_t, size_t>>* nonreplaceable_segments,
        std::set<std::pair<size_t, size_t>>* claimed_segments,
        runtime::MemoryAllocator* temp_allocator,
        CurrentInputState* current)
        : sources_(sources),
          named_entries_(named_entries),
          external_tensor_keys_(external_tensor_keys),
          named_segments_(named_segments),
          delegate_segments_(delegate_segments),
          nonreplaceable_segments_(nonreplaceable_segments),
          claimed_segments_(claimed_segments),
          temp_allocator_(temp_allocator),
          current_(current) {}

    Error write_processed_data(const BackendData& data) override {
      if (error_ != Error::Ok) {
        return error_;
      }
      if (current_->record == nullptr || !current_->record->yielded ||
          current_->record->replaced || !valid_data(data) ||
          named_segments_->count({0, current_->record->segment_index}) != 0 ||
          nonreplaceable_segments_->count(
              {0, current_->record->segment_index}) != 0) {
        return fail(Error::InvalidArgument);
      }
      Error error = replace_segment(0, current_->record->segment_index, data);
      if (error == Error::Ok) {
        current_->record->replaced = true;
        wrote_output_ = true;
      }
      return error == Error::Ok ? Error::Ok : fail(error);
    }

    Error write_named_data(Span<const NamedBackendData> values) override {
      if (error_ != Error::Ok) {
        return error_;
      }
      if (values.size() >
              std::numeric_limits<size_t>::max() / sizeof(NamedBackendData) ||
          !span_is_in_allocator(
              *temp_allocator_,
              values.data(),
              values.size() * sizeof(NamedBackendData))) {
        return fail(Error::InvalidArgument);
      }
      std::vector<const NamedLookup*> targets;
      std::set<std::string> keys;
      std::set<std::pair<size_t, size_t>> target_segments;
      for (const auto& value : values) {
        if (!valid_temp_string(value.key) || !valid_data(value.data) ||
            !keys.insert(value.key).second ||
            external_tensor_keys_->count(value.key) != 0) {
          return fail(Error::InvalidArgument);
        }
        auto found = std::find_if(
            named_entries_->begin(),
            named_entries_->end(),
            [&value](const NamedLookup& entry) {
              return entry.key == value.key;
            });
        if (found == named_entries_->end() ||
            claimed_segments_->count(
                {found->source_index, found->segment_index}) != 0 ||
            delegate_segments_->count(
                {found->source_index, found->segment_index}) != 0 ||
            nonreplaceable_segments_->count(
                {found->source_index, found->segment_index}) != 0 ||
            !target_segments.insert({found->source_index, found->segment_index})
                 .second) {
          return fail(Error::InvalidArgument);
        }
        targets.push_back(&*found);
      }
      std::vector<size_t> write_order(values.size());
      for (size_t i = 0; i < values.size(); ++i) {
        write_order[i] = i;
      }
      std::stable_sort(
          write_order.begin(),
          write_order.end(),
          [&targets](size_t lhs, size_t rhs) {
            if (targets[lhs]->source_index != targets[rhs]->source_index) {
              return targets[lhs]->source_index < targets[rhs]->source_index;
            }
            if (targets[lhs]->source_index == 0) {
              return false;
            }
            return targets[lhs]->segment_index < targets[rhs]->segment_index;
          });
      for (const size_t i : write_order) {
        Error error = replace_segment(
            targets[i]->source_index,
            targets[i]->segment_index,
            values[i].data);
        if (error != Error::Ok) {
          return fail(error);
        }
      }
      wrote_output_ = wrote_output_ || !values.empty();
      return Error::Ok;
    }

    bool wrote_output() const {
      return wrote_output_;
    }

    Error error() const {
      return error_;
    }

    Error finalize_current_input() override {
      if (error_ != Error::Ok || current_->record == nullptr ||
          current_->record->replaced) {
        return error_;
      }
      if (!current_->original.has_value()) {
        return fail(Error::InvalidState);
      }
      SegmentState& segment =
          (*sources_)[0].segments[current_->record->segment_index];
      const auto bytes = Span<const uint8_t>(
          static_cast<const uint8_t*>(current_->original->data()),
          current_->original->size());
      return write_segment((*sources_)[0], segment, bytes, segment.alignment);
    }

   private:
    Error fail(Error error) {
      if (error_ == Error::Ok) {
        error_ = error;
      }
      return error_;
    }

    bool valid_data(const BackendData& data) const {
      return span_is_in_allocator(
                 *temp_allocator_, data.bytes.data(), data.bytes.size()) &&
          (!data.alignment.has_value() || is_power_of_two(*data.alignment));
    }

    bool valid_temp_string(const char* value) const {
      if (value == nullptr ||
          !span_is_in_allocator(*temp_allocator_, value, 1)) {
        return false;
      }
      const auto* base = temp_allocator_->base_address();
      const size_t remaining = temp_allocator_->used_size() -
          (reinterpret_cast<const uint8_t*>(value) - base);
      return std::memchr(value, '\0', remaining) != nullptr;
    }

    Error write_segment(
        SourceState& source,
        SegmentState& segment,
        Span<const uint8_t> bytes,
        size_t alignment) {
      auto aligned = align_up(source.next_output_offset, alignment);
      if (!aligned.ok()) {
        return aligned.error();
      }
      auto end = checked_add(*aligned, bytes.size());
      if (!end.ok()) {
        return end.error();
      }
      Error error = source.source.writer->write(bytes, *aligned);
      if (error != Error::Ok) {
        return error;
      }
      segment.alignment = alignment;
      segment.final_offset = *aligned - source.segment_base_offset;
      segment.final_size = bytes.size();
      segment.written = true;
      source.next_output_offset = *end;
      source.modified = true;
      return Error::Ok;
    }

    Error replace_segment(
        size_t source_index,
        size_t segment_index,
        const BackendData& data) {
      if (source_index >= sources_->size() ||
          segment_index >= (*sources_)[source_index].segments.size() ||
          claimed_segments_->count({source_index, segment_index}) != 0) {
        return Error::InvalidArgument;
      }
      SourceState& source = (*sources_)[source_index];
      SegmentState& segment = source.segments[segment_index];
      const size_t alignment = data.alignment.value_or(segment.alignment);
      if (alignment < segment.alignment) {
        return Error::InvalidArgument;
      }
      Error error = write_segment(source, segment, data.bytes, alignment);
      if (error != Error::Ok) {
        return error;
      }
      segment.replaced = true;
      claimed_segments_->insert({source_index, segment_index});
      return Error::Ok;
    }

    std::vector<SourceState>* sources_;
    const std::vector<NamedLookup>* named_entries_;
    const std::set<std::string>* external_tensor_keys_;
    const std::set<std::pair<size_t, size_t>>* named_segments_;
    const std::set<std::pair<size_t, size_t>>* delegate_segments_;
    const std::set<std::pair<size_t, size_t>>* nonreplaceable_segments_;
    std::set<std::pair<size_t, size_t>>* claimed_segments_;
    runtime::MemoryAllocator* temp_allocator_;
    CurrentInputState* current_;
    bool wrote_output_{false};
    Error error_{Error::Ok};
  };

  void discard_unpublished(std::vector<SourceState>& sources) {
    for (auto& source : sources) {
      if (source.modified) {
        source.source.writer.reset();
        source.modified = false;
      }
    }
  }

  Error update_metadata_and_layout(SourceState& source, size_t* output_size) {
    const size_t segment_data_size =
        source.next_output_offset - source.segment_base_offset;

    if (source.kind == SourceKind::Pte) {
      auto* segments =
          executorch_flatbuffer::GetMutableProgram(source.metadata.data())
              ->mutable_segments();
      for (size_t i = 0; i < source.segments.size(); ++i) {
        auto* segment = segments->GetMutableObject(i);
        if ((segment->offset() != source.segments[i].final_offset &&
             !segment->mutate_offset(source.segments[i].final_offset)) ||
            (segment->size() != source.segments[i].final_size &&
             !segment->mutate_size(source.segments[i].final_size))) {
          return Error::NotSupported;
        }
      }
      if (source.has_segment_data_size) {
        write_u64_le(source.metadata.data() + 32, segment_data_size);
      } else if (segment_data_size != source.original_segment_data_size) {
        return Error::NotSupported;
      }
    } else {
      auto* segments =
          flat_tensor_flatbuffer::GetMutableFlatTensor(source.metadata.data())
              ->mutable_segments();
      for (size_t i = 0; i < source.segments.size(); ++i) {
        auto* segment = segments->GetMutableObject(i);
        if ((segment->offset() != source.segments[i].final_offset &&
             !segment->mutate_offset(source.segments[i].final_offset)) ||
            (segment->size() != source.segments[i].final_size &&
             !segment->mutate_size(source.segments[i].final_size))) {
          return Error::NotSupported;
        }
      }
    }
    if (source.kind == SourceKind::Ptd) {
      write_u64_le(source.metadata.data() + 40, segment_data_size);
    }
    *output_size = source.next_output_offset;
    return Error::Ok;
  }

  constexpr size_t kCopyChunkSize = 1024 * 1024;

  Error write_zeros(DataWriter& writer, size_t offset, size_t size) {
    std::vector<uint8_t> zeros(std::min(size, kCopyChunkSize), 0);
    size_t written = 0;
    while (written < size) {
      const size_t chunk = std::min(zeros.size(), size - written);
      ET_CHECK_OK_OR_RETURN_ERROR(writer.write(
          Span<const uint8_t>(zeros.data(), chunk), offset + written));
      written += chunk;
    }
    return Error::Ok;
  }

  Error copy_original_segment(SourceState& source, SegmentState& segment) {
    auto aligned = align_up(source.next_output_offset, segment.alignment);
    if (!aligned.ok()) {
      return aligned.error();
    }
    auto end = checked_add(*aligned, segment.original_size);
    if (!end.ok()) {
      return end.error();
    }
    std::vector<uint8_t> buffer(
        std::min(segment.original_size, kCopyChunkSize));
    size_t copied = 0;
    while (copied < segment.original_size) {
      const size_t chunk =
          std::min(buffer.size(), segment.original_size - copied);
      const size_t input_offset =
          source.segment_base_offset + segment.original_offset + copied;
      Error error = source.source.loader->load_into(
          input_offset,
          chunk,
          DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program),
          buffer.data());
      if (error == Error::NotImplemented) {
        auto loaded = source.source.loader->load(
            input_offset,
            chunk,
            DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
        if (!loaded.ok()) {
          return loaded.error();
        }
        error = source.source.writer->write(
            Span<const uint8_t>(
                static_cast<const uint8_t*>(loaded->data()), chunk),
            *aligned + copied);
      } else if (error == Error::Ok) {
        error = source.source.writer->write(
            Span<const uint8_t>(buffer.data(), chunk), *aligned + copied);
      }
      if (error != Error::Ok) {
        return error;
      }
      copied += chunk;
    }
    segment.final_offset = *aligned - source.segment_base_offset;
    segment.final_size = segment.original_size;
    segment.written = true;
    source.next_output_offset = *end;
    source.modified = true;
    return Error::Ok;
  }

  Error materialize_source(SourceState& source) {
    for (auto& segment : source.segments) {
      if (!segment.written) {
        ET_CHECK_OK_OR_RETURN_ERROR(copy_original_segment(source, segment));
      }
    }

    size_t output_size = 0;
    Error error = update_metadata_and_layout(source, &output_size);
    if (error != Error::Ok) {
      return error;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(source.source.writer->write(
        Span<const uint8_t>(source.metadata.data(), source.metadata.size()),
        0));
    size_t covered = source.metadata.size();
    if (source.segment_base_offset > covered) {
      ET_CHECK_OK_OR_RETURN_ERROR(write_zeros(
          *source.source.writer,
          covered,
          source.segment_base_offset - covered));
      covered = source.segment_base_offset;
    }
    std::vector<const SegmentState*> physical_segments;
    physical_segments.reserve(source.segments.size());
    for (const auto& segment : source.segments) {
      physical_segments.push_back(&segment);
    }
    std::sort(
        physical_segments.begin(),
        physical_segments.end(),
        [](const SegmentState* lhs, const SegmentState* rhs) {
          if (lhs->final_offset != rhs->final_offset) {
            return lhs->final_offset < rhs->final_offset;
          }
          return lhs->final_size < rhs->final_size;
        });
    for (const auto* segment : physical_segments) {
      const size_t absolute =
          source.segment_base_offset + segment->final_offset;
      if (absolute > covered) {
        ET_CHECK_OK_OR_RETURN_ERROR(
            write_zeros(*source.source.writer, covered, absolute - covered));
      }
      covered = absolute + segment->final_size;
    }
    return covered == output_size ? Error::Ok : Error::InvalidState;
  }

  } // namespace

  Error initialize_and_save_backend_data(
      std::unique_ptr<DataLoader> pte_loader,
      std::unique_ptr<DataWriter> pte_writer,
      runtime::MemoryAllocator * delegate_temp_allocator,
      runtime::EventTracer * event_tracer) {
    return initialize_and_save_backend_data(
        std::move(pte_loader),
        std::move(pte_writer),
        nullptr,
        nullptr,
        delegate_temp_allocator,
        event_tracer);
  }

  Error initialize_and_save_backend_data(
      std::unique_ptr<DataLoader> pte_loader,
      std::unique_ptr<DataWriter> pte_writer,
      std::unique_ptr<DataLoader> ptd_loader,
      std::unique_ptr<DataWriter> ptd_writer,
      runtime::MemoryAllocator * delegate_temp_allocator,
      runtime::EventTracer * event_tracer) {
    if (pte_loader == nullptr || pte_writer == nullptr ||
        delegate_temp_allocator == nullptr ||
        ((ptd_loader == nullptr) != (ptd_writer == nullptr))) {
      return Error::InvalidArgument;
    }
    std::vector<SourceState> sources;
    sources.reserve(ptd_loader == nullptr ? 1 : 2);
    auto pte = parse_pte(Source{std::move(pte_loader), std::move(pte_writer)});
    if (!pte.ok()) {
      return pte.error();
    }
    sources.push_back(std::move(*pte));
    if (ptd_loader != nullptr) {
      auto ptd =
          parse_ptd(Source{std::move(ptd_loader), std::move(ptd_writer)});
      if (!ptd.ok()) {
        discard_unpublished(sources);
        return ptd.error();
      }
      sources.push_back(std::move(*ptd));
    }

    std::vector<NamedLookup> named_entries;
    Error error = collect_named_entries(sources, &named_entries);
    if (error != Error::Ok) {
      return error;
    }
    const auto external_tensor_keys = collect_external_tensor_keys(sources[0]);
    std::vector<std::unique_ptr<InputRecord>> inputs;
    std::map<std::string, std::vector<InputRecord*>> by_backend;
    error = collect_inputs(sources[0], &inputs, &by_backend);
    if (error != Error::Ok) {
      return error;
    }
    for (auto& entry : by_backend) {
      std::sort(
          entry.second.begin(),
          entry.second.end(),
          [](const InputRecord* lhs, const InputRecord* rhs) {
            return lhs->segment_index < rhs->segment_index;
          });
    }
    std::set<std::pair<size_t, size_t>> named_segments;
    std::set<std::pair<size_t, size_t>> delegate_segments;
    std::set<std::pair<size_t, size_t>> nonreplaceable_segments;
    error = collect_segment_uses(
        sources,
        named_entries,
        inputs,
        &named_segments,
        &delegate_segments,
        &nonreplaceable_segments);
    if (error != Error::Ok) {
      return error;
    }

    // A zero offset may be absent from the FlatBuffer. If segment zero is not a
    // delegate input, preserve it first so its final offset remains zero.
    if (!sources[0].segments.empty() && delegate_segments.count({0, 0}) == 0) {
      error = copy_original_segment(sources[0], sources[0].segments[0]);
      if (error != Error::Ok) {
        discard_unpublished(sources);
        return error;
      }
    }

    std::set<std::pair<size_t, size_t>> claimed_segments;
    bool any_backend_output = false;
    {
      PreparationNamedDataMap named_data_map(&sources, &named_entries);
      std::vector<const decltype(by_backend)::value_type*> backend_order;
      backend_order.reserve(by_backend.size());
      for (const auto& entry : by_backend) {
        backend_order.push_back(&entry);
      }
      std::stable_sort(
          backend_order.begin(),
          backend_order.end(),
          [](const auto* lhs, const auto* rhs) {
            const bool lhs_has_first =
                !lhs->second.empty() && lhs->second.front()->segment_index == 0;
            const bool rhs_has_first =
                !rhs->second.empty() && rhs->second.front()->segment_index == 0;
            return lhs_has_first && !rhs_has_first;
          });
      for (const auto* entry : backend_order) {
        auto* backend = runtime::get_backend_class(entry->first.c_str());
        if (backend == nullptr) {
          discard_unpublished(sources);
          return Error::NotFound;
        }
        if (!backend->is_available()) {
          continue;
        }
        delegate_temp_allocator->reset();
        CurrentInputState current;
        PlanningBackendDataWriter output(
            &sources,
            &named_entries,
            &external_tensor_keys,
            &named_segments,
            &delegate_segments,
            &nonreplaceable_segments,
            &claimed_segments,
            delegate_temp_allocator,
            &current);
        PreparationDataInitContext context(
            &sources[0],
            entry->second,
            delegate_temp_allocator,
            event_tracer,
            named_entries.empty() ? nullptr : &named_data_map,
            &current,
            &output);
        error = backend->initialize_backend_data(context, output);
        if (context.error() != Error::Ok) {
          discard_unpublished(sources);
          return context.error();
        }
        if (output.error() != Error::Ok) {
          discard_unpublished(sources);
          return output.error();
        }
        const bool skip =
            error == Error::NotSupported && !output.wrote_output();
        if (error != Error::Ok && !skip) {
          discard_unpublished(sources);
          return error;
        }
        if (skip) {
          continue;
        }
        any_backend_output = any_backend_output || output.wrote_output();
        error = context.drain();
        if (error != Error::Ok || output.error() != Error::Ok) {
          discard_unpublished(sources);
          return error != Error::Ok ? error : output.error();
        }
      }
    }

    if (!any_backend_output) {
      discard_unpublished(sources);
      return Error::Ok;
    }

    for (auto& source : sources) {
      if (!source.modified) {
        continue;
      }
      error = materialize_source(source);
      if (error != Error::Ok) {
        discard_unpublished(sources);
        return error;
      }
    }
    for (auto& source : sources) {
      source.source.loader.reset();
    }
    for (auto& source : sources) {
      if (!source.modified) {
        continue;
      }
      error = source.source.writer->publish();
      if (error != Error::Ok) {
        discard_unpublished(sources);
        return error;
      }
    }
    return Error::Ok;
  }

} // namespace executorch::extension
