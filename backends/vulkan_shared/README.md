# Shared Vulkan runtime

This component is not a standalone ExecuTorch backend. It does not 
partition graphs, register a backend, or execute delegated operators.

This component is the backend-neutral runtime bridge used by the VGF and
ExecuTorch Vulkan delegates. It deliberately does not introduce a unified
partitioner or a wrapper delegate.

The standard AOT flow remains explicit partitioner composition, with VGF
claiming supported regions first and Vulkan filling the remaining regions:

```python
lowered = to_edge_transform_and_lower(
    exported,
    partitioner=[
        VgfPartitioner(vgf_compile_spec),
        VulkanPartitioner(vulkan_compile_spec),
    ],
)
```

## Runtime options

Context selection is configured at model load time through `RuntimeSpec` /
`BackendOptions`, not through serialized `CompileSpec` values:

| Key | Type | Default | Accepted values |
| --- | --- | --- | --- |
| `vulkan_shared_device_name` | string | `default` | Any non-empty device_name |
| `vulkan_shared_device_mode` | string | `lookup_or_create` | `disabled`, `lookup_only`, `lookup_or_create`, `create_only` |
| `vulkan_shared_group_id` | int | `0` | Any `int` value |

Both delegates must receive the same option values to resolve the same registry
key.

## Ownership and validation

`SharedVulkanDevice` carries Vulkan handles but never calls Vulkan entry points
itself. Every registered context must provide a non-null `lifetime_anchor` whose
lifetime guarantees that the Vulkan instance, physical device, device, and queue
remain valid until the final `SharedVulkanDevicePtr` is released. For a
backend-created context, the anchor can own the backend runtime and perform
teardown through that backend's Vulkan dispatch mechanism. For externally
created Vulkan objects, the application must provide an anchor whose ownership
keeps those objects alive for the same period.

`unregister_device()` removes the context from the registry and prevents new
lookups; it does not revoke `SharedVulkanDevicePtr` instances already held by
delegates. Actual Vulkan teardown is therefore safe only after the final
outstanding context reference releases its `lifetime_anchor`. The registry
itself is intentionally process-lifetime; call `unregister_device()` to remove
registry discoverability before deterministic teardown.

The `VkQueue` is shared process state and Vulkan queue operations require
external host synchronization. Consumers must issue queue operations through
`SharedVulkanDevice::with_locked_queue()` so independently initialized delegates
serialize access using the mutex stored in the shared device rather than
backend-local locks.

The registrant also declares the device extensions enabled at `VkDevice`
creation. A consuming backend must check its required extensions with
`has_device_extension()` before using the context; Vulkan does not expose a
post-creation query for the list of extensions that were enabled.

Canonical option names are `vulkan_shared_device_name` and
`vulkan_shared_device_mode`. The Phase 2 names
`vulkan_shared_context_name` and `vulkan_shared_context_mode` remain accepted
as compatibility aliases; when both variants are supplied, the new name wins.

The shared device also carries a manifest of **features enabled at VkDevice
creation**, which may be a subset of the physical GPU's supported features.
When registering an external device, fill `enabled_device_features` accurately;
unspecified optional features default to disabled. Do not report a feature as
enabled merely because the physical device supports it.

`create_only` permits multiple Vulkan delegate regions to reuse a device that
the Vulkan backend created for the key, but rejects an externally registered
device using that key. Applications should unregister their device keys during
controlled teardown after all delegate borrowers finish.
