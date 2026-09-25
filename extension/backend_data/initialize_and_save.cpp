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
    return Error::InvalidProgram;
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
  flatbuffers::Verifier verifier(state.metadata.data(), state.metadata.size());
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

class PreparationNamedDataMap final : public NamedDataMap {
 public:
  PreparationNamedDataMap(
      const SourceState* source,
      const std::vector<NamedLookup>* entries)
      : source_(source), entries_(entries) {}

  Result<const runtime::TensorLayout> get_tensor_layout(
      std::string_view) const override {
    return Error::NotImplemented;
  }

  Result<FreeableBuffer> get_data(std::string_view key) const override {
    const NamedLookup* entry = find(key);
    if (entry == nullptr) {
      return Error::NotFound;
    }
    const SegmentState& segment = source_->segments[entry->segment_index];
    auto offset =
        checked_add(source_->segment_base_offset, segment.original_offset);
    if (!offset.ok()) {
      return offset.error();
    }
    return source_->source.loader->load(
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
    const SegmentState& segment = source_->segments[entry->segment_index];
    if (size > segment.original_size) {
      return Error::InvalidArgument;
    }
    auto offset =
        checked_add(source_->segment_base_offset, segment.original_offset);
    if (!offset.ok()) {
      return offset.error();
    }
    return source_->source.loader->load_into(
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

  const SourceState* source_;
  const std::vector<NamedLookup>* entries_;
};

Error collect_named_entries(
    const SourceState& source,
    std::vector<NamedLookup>* entries) {
  const auto* named =
      executorch_flatbuffer::GetProgram(source.metadata.data())->named_data();
  if (named == nullptr) {
    return Error::Ok;
  }
  std::set<std::string> keys;
  for (size_t i = 0; i < named->size(); ++i) {
    const auto* item = named->Get(i);
    if (item == nullptr || item->key() == nullptr ||
        item->segment_index() >= source.segments.size() ||
        !keys.insert(item->key()->str()).second) {
      return Error::InvalidProgram;
    }
    entries->push_back({item->key()->str(), i, item->segment_index()});
  }
  return Error::Ok;
}

std::set<std::string> collect_external_tensor_keys(const SourceState& pte) {
  std::set<std::string> keys;
  const auto* plans =
      executorch_flatbuffer::GetProgram(pte.metadata.data())->execution_plan();
  if (plans == nullptr) {
    return keys;
  }
  for (const auto* plan : *plans) {
    if (plan == nullptr || plan->values() == nullptr) {
      continue;
    }
    for (const auto* value : *plan->values()) {
      const auto* tensor = value == nullptr ? nullptr : value->val_as_Tensor();
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

Error collect_nonreplaceable_segments(
    const SourceState& source,
    const std::vector<NamedLookup>& named_entries,
    const std::vector<std::unique_ptr<InputRecord>>& inputs,
    std::set<size_t>* nonreplaceable_segments) {
  std::map<size_t, size_t> named_counts;
  for (const auto& entry : named_entries) {
    ++named_counts[entry.segment_index];
  }
  for (const auto& entry : named_counts) {
    if (entry.second > 1) {
      nonreplaceable_segments->insert(entry.first);
    }
  }
  for (const auto& input : inputs) {
    nonreplaceable_segments->insert(input->segment_index);
  }
  const auto* program =
      executorch_flatbuffer::GetProgram(source.metadata.data());
  const auto add_segment = [&](uint32_t segment_index) {
    if (segment_index >= source.segments.size()) {
      return false;
    }
    nonreplaceable_segments->insert(segment_index);
    return true;
  };
  if (program->constant_segment() != nullptr &&
      !add_segment(program->constant_segment()->segment_index())) {
    return Error::InvalidProgram;
  }
  if (program->mutable_data_segments() != nullptr) {
    for (const auto* mutable_segment : *program->mutable_data_segments()) {
      if (mutable_segment == nullptr ||
          !add_segment(mutable_segment->segment_index())) {
        return Error::InvalidProgram;
      }
    }
  }
  // A zero offset can be omitted from the FlatBuffer. Keep the first segment
  // fixed so a later named replacement cannot force an unrepresentable offset.
  if (!source.segments.empty()) {
    nonreplaceable_segments->insert(0);
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
  const auto* plans =
      executorch_flatbuffer::GetProgram(pte.metadata.data())->execution_plan();
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
      const NamedDataMap* named_data_map)
      : BackendDataInitContext(temp_allocator, event_tracer, named_data_map),
        pte_(pte),
        inputs_(std::move(inputs)) {}

  Result<std::optional<BackendDataInput>> next_backend_data() override {
    current_.reset();
    get_temp_allocator()->reset();
    if (next_ == inputs_.size()) {
      return std::optional<BackendDataInput>{};
    }
    InputRecord& input = *inputs_[next_++];
    const SegmentState& segment = pte_->segments[input.segment_index];
    auto offset =
        checked_add(pte_->segment_base_offset, segment.original_offset);
    if (!offset.ok()) {
      return offset.error();
    }
    auto loaded = pte_->source.loader->load(
        *offset,
        segment.original_size,
        DataLoader::SegmentInfo(
            DataLoader::SegmentInfo::Type::Backend,
            input.segment_index,
            input.backend_id.c_str()));
    if (!loaded.ok()) {
      return loaded.error();
    }
    current_.emplace(std::move(*loaded));
    return std::optional<BackendDataInput>(BackendDataInput{
        input.method_name.c_str(),
        Span<const uint8_t>(
            static_cast<const uint8_t*>(current_->data()), current_->size()),
        runtime::ArrayRef<CompileSpec>(
            input.compile_specs.data(), input.compile_specs.size())});
  }

 private:
  SourceState* pte_;
  std::vector<InputRecord*> inputs_;
  std::optional<FreeableBuffer> current_;
  size_t next_{0};
};

class PlanningBackendDataWriter final : public BackendDataWriter {
 public:
  PlanningBackendDataWriter(
      SourceState* source,
      const std::vector<NamedLookup>* named_entries,
      const std::set<std::string>* external_tensor_keys,
      const std::set<size_t>* nonreplaceable_segments,
      std::set<size_t>* claimed_segments,
      runtime::MemoryAllocator* temp_allocator)
      : source_(source),
        named_entries_(named_entries),
        external_tensor_keys_(external_tensor_keys),
        nonreplaceable_segments_(nonreplaceable_segments),
        claimed_segments_(claimed_segments),
        temp_allocator_(temp_allocator) {}

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
    std::set<size_t> target_segments;
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
          claimed_segments_->count(found->segment_index) != 0 ||
          nonreplaceable_segments_->count(found->segment_index) != 0 ||
          !target_segments.insert(found->segment_index).second) {
        return fail(Error::InvalidArgument);
      }
      targets.push_back(&*found);
    }
    for (size_t i = 0; i < values.size(); ++i) {
      const Error error =
          replace_segment(targets[i]->segment_index, values[i].data);
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
    if (value == nullptr || !span_is_in_allocator(*temp_allocator_, value, 1)) {
      return false;
    }
    const auto* base = temp_allocator_->base_address();
    const size_t remaining = temp_allocator_->used_size() -
        (reinterpret_cast<const uint8_t*>(value) - base);
    return std::memchr(value, '\0', remaining) != nullptr;
  }

  Error replace_segment(size_t segment_index, const BackendData& data) {
    if (segment_index >= source_->segments.size() ||
        claimed_segments_->count(segment_index) != 0) {
      return Error::InvalidArgument;
    }
    SegmentState& segment = source_->segments[segment_index];
    const size_t alignment = data.alignment.value_or(segment.alignment);
    if (alignment < segment.alignment) {
      return Error::InvalidArgument;
    }
    auto aligned = align_up(source_->next_output_offset, alignment);
    if (!aligned.ok()) {
      return aligned.error();
    }
    auto end = checked_add(*aligned, data.bytes.size());
    if (!end.ok()) {
      return end.error();
    }
    Error error = source_->source.writer->write(data.bytes, *aligned);
    if (error != Error::Ok) {
      return error;
    }
    segment.alignment = alignment;
    segment.final_offset = *aligned - source_->segment_base_offset;
    segment.final_size = data.bytes.size();
    segment.replaced = true;
    segment.written = true;
    source_->next_output_offset = *end;
    source_->modified = true;
    claimed_segments_->insert(segment_index);
    return Error::Ok;
  }

  SourceState* source_;
  const std::vector<NamedLookup>* named_entries_;
  const std::set<std::string>* external_tensor_keys_;
  const std::set<size_t>* nonreplaceable_segments_;
  std::set<size_t>* claimed_segments_;
  runtime::MemoryAllocator* temp_allocator_;
  bool wrote_output_{false};
  Error error_{Error::Ok};
};

void discard_unpublished(SourceState& source) {
  if (source.modified) {
    source.source.writer.reset();
    source.modified = false;
  }
}

Error update_metadata_and_layout(SourceState& source, size_t* output_size) {
  const size_t segment_data_size =
      source.next_output_offset - source.segment_base_offset;

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
  std::vector<uint8_t> buffer(std::min(segment.original_size, kCopyChunkSize));
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
      Span<const uint8_t>(source.metadata.data(), source.metadata.size()), 0));
  size_t covered = source.metadata.size();
  if (source.segment_base_offset > covered) {
    ET_CHECK_OK_OR_RETURN_ERROR(write_zeros(
        *source.source.writer, covered, source.segment_base_offset - covered));
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
    const size_t absolute = source.segment_base_offset + segment->final_offset;
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
    runtime::MemoryAllocator* delegate_temp_allocator,
    runtime::EventTracer* event_tracer) {
  if (pte_loader == nullptr || pte_writer == nullptr ||
      delegate_temp_allocator == nullptr) {
    return Error::InvalidArgument;
  }
  auto parsed = parse_pte(Source{std::move(pte_loader), std::move(pte_writer)});
  if (!parsed.ok()) {
    return parsed.error();
  }
  SourceState source = std::move(*parsed);

  std::vector<NamedLookup> named_entries;
  Error error = collect_named_entries(source, &named_entries);
  if (error != Error::Ok) {
    return error;
  }
  const auto external_tensor_keys = collect_external_tensor_keys(source);
  std::vector<std::unique_ptr<InputRecord>> inputs;
  std::map<std::string, std::vector<InputRecord*>> by_backend;
  error = collect_inputs(source, &inputs, &by_backend);
  if (error != Error::Ok) {
    return error;
  }
  std::set<size_t> nonreplaceable_segments;
  error = collect_nonreplaceable_segments(
      source, named_entries, inputs, &nonreplaceable_segments);
  if (error != Error::Ok) {
    return error;
  }

  // Preserve the first segment at relative offset zero. FlatBuffers may omit
  // that default-valued scalar, so moving it would not be representable.
  if (!source.segments.empty()) {
    error = copy_original_segment(source, source.segments[0]);
    if (error != Error::Ok) {
      discard_unpublished(source);
      return error;
    }
  }

  std::set<size_t> claimed_segments;
  bool any_backend_output = false;
  {
    PreparationNamedDataMap named_data_map(&source, &named_entries);
    for (const auto& entry : by_backend) {
      auto* backend = runtime::get_backend_class(entry.first.c_str());
      if (backend == nullptr) {
        discard_unpublished(source);
        return Error::NotFound;
      }
      if (!backend->is_available()) {
        continue;
      }
      delegate_temp_allocator->reset();
      PlanningBackendDataWriter output(
          &source,
          &named_entries,
          &external_tensor_keys,
          &nonreplaceable_segments,
          &claimed_segments,
          delegate_temp_allocator);
      PreparationDataInitContext context(
          &source,
          entry.second,
          delegate_temp_allocator,
          event_tracer,
          named_entries.empty() ? nullptr : &named_data_map);
      error = backend->initialize_backend_data(context, output);
      if (output.error() != Error::Ok) {
        discard_unpublished(source);
        return output.error();
      }
      const bool skip = error == Error::NotSupported && !output.wrote_output();
      if (error != Error::Ok && !skip) {
        discard_unpublished(source);
        return error;
      }
      any_backend_output = any_backend_output || output.wrote_output();
    }
  }

  if (!any_backend_output) {
    discard_unpublished(source);
    return Error::Ok;
  }
  error = materialize_source(source);
  if (error != Error::Ok) {
    discard_unpublished(source);
    return error;
  }
  source.source.loader.reset();
  error = source.source.writer->publish();
  if (error != Error::Ok) {
    discard_unpublished(source);
  }
  return error;
}

} // namespace executorch::extension
