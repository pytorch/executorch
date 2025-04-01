# Why We Have a src/executorch/exir File

The `src/executorch/exir` file exists primarily due to the limitations of `pip install` in editable mode. Specifically, `pip install -e .` does not recognize `<executorch root>/exir` (or any root level directory with a `__init__.py`) as a valid package module because of the presence of `<executorch root>/exir/__init__.py`. See the following GitHub issue for details: [Issue #9558](https://github.com/pytorch/executorch/issues/9558).

## The Symlink Solution

To work around this limitation, a symlink is used. With this symlink and this package entry in `pyproject.toml`:

```toml
[tool.setuptools.package-dir]
...
"executorch" = "src/executorch"
```
We are telling `pip install -e .` to treat `src/executorch` as the root of the `executorch` package and hence mapping `executorch.exir` to `src/executorch/exir`.

This allows us to perform `pip install -e .` successfully and enables the execution of the following command:

```bash
python -c "from executorch.exir import CaptureConfig"
```
