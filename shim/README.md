# Open Source Shim

These files are a shim that allow us to build Buck2 with Buck2 outside Meta in
the open source world.

Note:
- `targets.bzl` files: these should remain in the tree. Targets inside a `targets.bzl` file can be referred to via absolute path, eg. `//executorch/kernels/portable:operators`, or local path, eg. `:operators`, if in the same file.
- `<util>.bzl` files: these should be added to the shim, if they're being used in OSS. For example, `op_registration_util.bzl`, `codegen.bzl`, etc. These files are loaded via `load(@<cell>//package/name:filename.bzl)`. Usually, this ends up being`load(@fbsource//xplat/...)`, which requires a redirect from fbsource to open source.
