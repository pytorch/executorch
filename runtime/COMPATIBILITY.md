# Runtime Compatibility Policy

This document describes the compatibility guarantees between the [PTE file
format](https://pytorch.org/executorch/stable/pte-file-format.html) and the
ExecuTorch runtime.

> [!IMPORTANT]
> The [canonical version of this document](https://github.com/pytorch/executorch/tree/main/runtime/COMPATIBILITY.md)
> is in the `main` branch of the `pytorch/executorch` GitHub repo.

## Backward compatibility (BC)

Definition: Whether an older version of a PTE file can run on a newer version of
the runtime.

A PTE file created at a particular ExecuTorch version using stable
(non-deprecated and non-experimental) APIs can be loaded and executed by a
runtime built from a minimum of one following non-patch (major or minor)
release. For example:

* A PTE file created with version `1.2.0` will be compatible with version
  `1.3.0`, and possibly more.
* If `1.2.0` is followed by a major release, then a PTE file created with
  version `1.2.0` will be compatible with version `2.0.0`, and possibly more.

ExecuTorch minor releases happen every three months, following the PyTorch
release cadence. This means that a PTE file will be supported for at least six
months: the release in which it was built (three months) and the following
release (another three months).

These are minimum guarantees. We will strive to maintain compatibility for as
long as possible, but it may depend on the features used by a particular PTE
file: see "Feature-specific limitations" below.

### Feature-specific limitations

A PTE file may symbolically refer to functionality outside the file itself:
operators or backends. These are linked directly into the runtime, and have
their own compatibility guarantees, but must satisfy the minimum BC guarantees
above.

**Operators**: Core ATen operators must comply with the [Core ATen opset
backward & forward compatibility
policy](https://dev-discuss.pytorch.org/t/core-aten-opset-backward-forward-compatibility-policy/1772).
Custom operators (that are not part of Core ATen) have no explicit guarantees,
but the owners of those custom operators should try to provide similar
guarantees, and clearly document them.

**Backends**: A delegated payload in a PTE file must remain loadable and
executable by its associated backend for at least the version range described
above. Each backend may provide longer guarantees, which may be described in a
file named `//executorch/backends/{name}/COMPATIBILITY.md`, or in a similar
location for backends whose code lives outside the `executorch` repo.

## Forward compatibility (FC)

Definition: Whether a newer version of a PTE file can run on an older version of
the runtime.

ExecuTorch does not make guarantees about forward compatibility. It is possible
for a newer PTE file to be loaded and executed by an older version of the
runtime, but it is not guaranteed. The parameters used to generate the PTE file
may affect this compatibility.
