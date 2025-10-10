#pragma once

namespace executorch::backends::xnnpack {
/// The key for the backend. This is used to register the backend, check
/// availability, and get/set options.
const char xnnpack_backend_key[] = "XnnpackBackend";

/// The key for the workspace sharing option. See the WorkspaceSharingMode enum
/// for a description of the associated functionality.
const char workspace_sharing_mode_option_key[] = "workspace_sharing_mode";

/// Workspace sharing mode. This is a backend option that can be set via the
/// set_option API to control memory sharing between CALL_DELEGATE instances.
/// This is useful for reducing memory consumption.
enum class WorkspaceSharingMode {
  /// No workspace sharing. Each CALL_DELEGATE instance will have its own
  /// workspace (memory arena).
  Disabled = 0,

  /// All CALL_DELEGATE instances in a given program will share a workspace.
  /// This reduces memory consumption
  /// for methods with multiple delegate calls, at the cost of only allowing one
  /// method to execute at a time.
  PerModel = 1,

  /// All CALL_DELEGATE instances accross all loaded methods will share a
  /// workspace. This reduces memory
  /// consumption by overlapping activation memory between methods but enforces
  /// synchronization between
  /// methods. If multiple methods are run concurrently, it may block as only
  /// one delegate call occur
  /// at a time. Additionally, the workspace does not shrink when a method is
  /// unloaded, so memory will
  /// only be reclaimed when all XNNPACK-delegated methods are unloaded.
  Global = 2,

  /// The number of workspace sharing modes. This is not a valid mode and is
  /// only used for tracking the
  // maximum enum value.
  Count,
};
} // namespace executorch::backends::xnnpack
