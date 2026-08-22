# Why Do We Have These Symlinks

The `src/executorch/*` files exist primarily due to the limitations of `pip install` in editable mode. Specifically, `pip install -e .` does not recognize `<executorch root>/exir` (or any root level directory with a `__init__.py`) as a valid package module because of the presence of `<executorch root>/exir/__init__.py`. See the following GitHub issue for details: [Issue #9558](https://github.com/pytorch/executorch/issues/9558).

## The Symlink Solution

To work around this limitation, a symlink is used. With this symlink and this package entry in `pyproject.toml`:

```toml
[tool.setuptools.package-dir]
# ...
"executorch" = "src/executorch"
```
We are telling `pip install -e .` to treat `src/executorch` as the root of the `executorch` package and hence mapping `executorch.*.*` to `src/executorch/*/*`. This effectively gets modules like `exir` out from the root level package.

This allows us to perform `pip install -e .` successfully and enables the execution of the following command:

```bash
python -c "from executorch.exir import CaptureConfig"
```

## Long Term Solution

We should start to move directories from <executorch root>/ to <executorch root>/src/ and remove the symlinks. Issue [#8699](https://github.com/pytorch/executorch/issues/8699) to track this effort. This will require a lot of work internally.

TODO(mnachin T180504136): Do not put examples/models into core pip packages. Refactor out the necessary utils or core models files into a separate package.
