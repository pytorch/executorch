/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/NativeBackendInternal.h>

#include <algorithm>
#include <type_traits>
#include <variant>

namespace executorch {
namespace backends {
namespace native {

// Per-execute IO binding: fan out to every engine. Each engine
// self-filters via its internal io_*_bindings_ table built at
// init-time by set_io_bindings.
Error bind_inputs(DelegateInstance* d, Span<EValue*> args) {
  size_t n_in = d->plan.inputs.size();
  size_t in_count = std::min(n_in, args.size());
  Span<EValue* const> input_args(args.data(), in_count);
  for (Engine* inst : d->plan.instances) {
    if (!inst)
      continue;
    auto err = inst->bind_inputs(
        Span<EValue>(d->values.data(), d->values.size()), input_args);
    if (err != Error::Ok)
      return err;
  }
  return Error::Ok;
}

Error bind_outputs(DelegateInstance* d, Span<EValue*> args) {
  size_t n_in = d->plan.inputs.size();
  size_t n_out = d->plan.outputs.size();
  if (args.size() < n_in)
    return Error::InvalidArgument;
  size_t out_count = std::min(n_out, args.size() - n_in);
  Span<EValue* const> output_args(args.data() + n_in, out_count);
  for (Engine* inst : d->plan.instances) {
    if (!inst)
      continue;
    auto err = inst->bind_outputs(
        Span<EValue>(d->values.data(), d->values.size()), output_args);
    if (err != Error::Ok)
      return err;
  }
  return Error::Ok;
}

Event* resolve_event(DelegateInstance* d, EventId id) {
  if (id == kNoEvent || id >= d->plan.events.size())
    return nullptr;
  return d->plan.events[id].event.get();
}

Error execute_step(DelegateInstance* d, const Step& step) {
  return std::visit(
      [&](auto&& s) -> Error {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, JumpFalseStep>) {
          ET_LOG(
              Error,
              "execute_step called with JumpFalseStep — this is a routing "
              "bug; jumps must be handled in the PC walker");
          return Error::Internal;
        } else if constexpr (std::is_same_v<T, MoveStep>) {
          if (s.src_value_id == s.dst_value_id)
            return Error::Ok;
          d->values[s.dst_value_id] = d->values[s.src_value_id];
          ET_LOG(
              Debug,
              "[cf] MoveStep src=%u -> dst=%u (EValue assign)",
              s.src_value_id,
              s.dst_value_id);
          return Error::Ok;
        } else {
          std::vector<Event*> waits_storage;
          waits_storage.reserve(s.wait_for.size());
          for (EventId id : s.wait_for) {
            if (Event* e = resolve_event(d, id))
              waits_storage.push_back(e);
          }
          Span<Event* const> waits(waits_storage.data(), waits_storage.size());
          Event* signal = resolve_event(d, s.signal);

          if constexpr (std::is_same_v<T, ComputeStep>) {
            Engine* inst = d->plan.instances[s.runtime_idx];
            return inst->execute(
                s.segment,
                Span<EValue>(d->values.data(), d->values.size()),
                waits,
                signal);
          } else if constexpr (std::is_same_v<T, TransferStep>) {
            if (s.src_value_id >= d->values.size() ||
                s.dst_value_id >= d->values.size()) {
              return Error::InvalidState;
            }
            EValue& src_ev = d->values[s.src_value_id];
            EValue& dst_ev = d->values[s.dst_value_id];
            Engine* src_inst = d->plan.instances[s.src_idx];
            Engine* dst_inst = d->plan.instances[s.dst_idx];
            std::string src_pname = "?";
            std::string dst_pname = "?";
            if (s.src_idx < d->plan.providers.size() &&
                d->plan.providers[s.src_idx]) {
              src_pname = std::string(d->plan.providers[s.src_idx]->name());
            }
            if (s.dst_idx < d->plan.providers.size() &&
                d->plan.providers[s.dst_idx]) {
              dst_pname = std::string(d->plan.providers[s.dst_idx]->name());
            }
            size_t xfer_bytes =
                src_ev.isTensor() ? src_ev.toTensor().nbytes() : 0;
            ET_LOG(
                Debug,
                "[mem] step: TransferStep src=%u (%s) -> dst=%u (%s) bytes=%zu",
                s.src_value_id,
                src_pname.c_str(),
                s.dst_value_id,
                dst_pname.c_str(),
                xfer_bytes);
            // Direction-specific dispatch: the device (non-host) Engine
            // owns the cross-runtime move. Engine resolves its own
            // Buffer internally from the value_id. By host-canonical
            // invariant, the non-host side is always a DeviceEngine.
            if (s.src_idx == kHostIdx && s.dst_idx != kHostIdx) {
              return static_cast<DeviceEngine*>(dst_inst)->upload_from_host(
                  src_ev, dst_ev, s.dst_value_id, waits, signal);
            } else if (s.dst_idx == kHostIdx && s.src_idx != kHostIdx) {
              return static_cast<DeviceEngine*>(src_inst)->download_to_host(
                  src_ev, s.src_value_id, dst_ev, waits, signal);
            } else {
              ET_LOG(
                  Error,
                  "TransferStep with neither side on host (src_idx=%u dst_idx=%u) is unsupported",
                  s.src_idx,
                  s.dst_idx);
              return Error::NotSupported;
            }
          }
        }
        return Error::Internal;
      },
      step);
}

} // namespace native
} // namespace backends
} // namespace executorch
