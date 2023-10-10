# ExecuTorch Documentation

Welcome to the ExecuTorch documentation! This README.md will provide an overview
of the ExecuTorch docs and its features, as well as instructions on how to
contribute and build locally.

All current documentation is located in the `docs/source` directory.

<!-- toc -->

- [Toolchain Overview](#toolchain-overview)
- [Building Locally](#building-locally)
- [Using Custom Variables](#using-custom-variables)
- [Including READMEs to the Documentation Build](#including-readmes-to-the-documentation-build)
- [Contributing](#contributing)
- [Adding Tutorials](#adding-tutorials)
- [Auto-generated API documentation](#auto-generated-api-documentation)
  - [Python APIs](#python-apis)
  - [C++ APIs](#c-apis)
  <!-- tocstop -->

## Toolchain Overview

We are using [sphinx](https://www.sphinx-doc.org/en/master/) with
[myst_parser](https://myst-parser.readthedocs.io/en/latest/),
[sphinx-gallery](https://sphinx-gallery.github.io/stable/index.html), and
[sphinx_design](https://sphinx-design.readthedocs.io/en/latest/) in this
documentation set.

We support both `.rst` and `.md` files but prefer the content to be authored in
`.md` as much as possible.

## Building Locally

Documentation dependencies are stored in
[.ci/docker/requirements-ci.txt](https://github.com/pytorch/executorch/blob/main/.ci/docker/requirements-ci.txt).

To build the documentation locally:

1. Clone the ExecuTorch repo to your machine.

1. If you don't have it already, start a conda environment:

   ```{note}
   The below command generates a completely new environment and resets
   any existing dependencies. If you have an environment already, skip
   the `conda create` command.
   ```

   ```bash
   conda create -yn executorch python=3.10.0
   conda activate executorch
   ```

1. Install dependencies:

   ```bash
   pip3 install -r ./.ci/docker/requirements-ci.txt
   ```

1. Run:

   ```bash
   bash install_requirements.sh
   ```

1. Go to the `docs/` directory.

1. Build the documentation set:

   ```
   make html
   ```

   This should build both documentation and tutorials. The build will be placed
   in the `_build` directory.

1. You can preview locally by using
   [sphinx-serve](https://pypi.org/project/sphinx-serve/). To install
   sphinx-serve, run: `pip3 install sphinx-serve`. To serve your documentation:

   ```
   sphinx-serve -b _build
   ```

   Open http://0.0.0.0:8081/ in your browser to preview your updated
   documentation.

## Using Custom Variables

You can use custom variables in your `.md` and `.rst` files. The variables take
their values from the files listed in the `./.ci/docker/ci_commit_pins/`
directory. For example, to insert a variable that specifies the latest Buck2
version, use the following syntax:

```
The current version of Buck2 is ${executorch_version:buck2}.
```

This will result in the following output:

<img src="./source/_static/img/s_custom_variables_extension.png" width="300">

We have the following custom variables defined in this docset:

- `${executorch_version:buck2}`
- `${executorch_version:nightly}`
- `${executorch_version:pytorch}`
- `${executorch_version:vision}`
- `${executorch_version:audio}`

You can use the variables in both regular text and code blocks.

## Including READMEs to the Documentation Build

You might want to include some of the `README.md` files from various directories
in this repositories in your documentation build. To do that, create an `.md`
file and use the `{include}` directive to insert your `.md` files. Example:

````
```{include} ../README.md
````

**NOTE:** Many `README.md` files are written as placeholders with limited
information provided. Some of that content you might want to keep in the
repository rather than on the website. If you still want to add it, make sure to
check the content for accuracy, structure, and overall quality.

## Contributing

Use the
[PyTorch contributing guidelines](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#writing-documentation)
to contribute to the documentation.

In addition to that, see
[Markdown in Sphinx Tips and Tricks](https://pytorch.org/executorch/markdown-sphinx-tips-tricks.html)
for tips on how to author high-quality markdown pages with Myst Parser.

## Adding Tutorials

You can add both interactive (`.py`) and non-interactive tutorials (`.md`) to
this documentation. All tutorials should go to the `tutorials_source/`
directory. Use one of the following templates:

- [Python Template](https://github.com/pytorch/executorch/blob/main/docs/source/tutorials_source/template_tutorial.py)
- [Markdown template](https://github.com/pytorch/executorch/blob/main/docs/source/tutorial-template.md)

After creating a tutorial, make sure to add the corresponding path in the
[index.rst](./source/index.rst) file in the following places:

- In the
  [tutorials torctree](https://github.com/pytorch/executorch/blob/main/docs/source/index.rst?plain=1#L183)
- In the
  [customcard section](https://github.com/pytorch/executorch/blob/main/docs/source/index.rst?plain=1#L201)

If you want to include a Markdown tutorial that is stored in another directory
outside of the `docs/source` directory, complete the following steps:

1. Create an `.md` file under `source/tutorials_source`. Name that file after
   your tutorial.
2. Include the following in that file:

   ````
   ---
   orphan: true
   ---
   ```{include} ../path-to-your-file/outside-of-the-docs-dir.md```
   ````

   **NOTE:** Your tutorial source file needs to follow the tutorial template.

3. Add the file that you have created in **Step 1** to the `index.rst` toctree
   and add a `customcarditem` with the link to that file.

For example, if I wanted to include the `README.md` file from
`examples/selective_build` as a tutorial under
`pytorch.org/executorch/tutorials`, I could create a file called
`tutorials_source/selective-build-tutorial.md` and add the following to that
file:

````
---
orphan: true
---

```{include} ../../../examples/selective_build/README.md
````

In the `index.rst` file, I would add `tutorials/selective-build-tutorial` in
both the `toctree` and the `cusotmcarditem` sections.

# Auto-generated API documentation

On a high level (will go into detail later), we use Sphinx to generate both
Python and C++ documentation in the form of HTML pages.

### Python APIs

We generate Python API documentation through Sphinx, bootstrapping
[PyTorch's Sphinx theme](https://github.com/pytorch/pytorch_sphinx_theme) for a
cohesive look with the existing PyTorch API documentation.

The setup for Python documentation lies within `source_py/`. To set up Sphinx, a
`conf.py` configuration file is required. We can specify ways to generate
documentation here. Specifically, the most important/relevant parts are:

- Make sure to add a path to the directory above the directory you're trying to
  generate documentation for. For example, since we want to generate
  documentation for the `executorch/` directory. This tells Sphinx where to find
  the code to generate docs for.

- `extensions` contains extension modules. For auto-generating APIs, make sure
  to include `sphinx.ext.autodoc`.
- `autodoc_mock_imports` is where you put imports that Sphinx is unable to
  access. Sphinx runs your code in order to autogenerate the docs, so for any
  libraries that it unable to access due to it being outside of the directory,
  or containing c++ bindings, we need to specify it in the
  `autodoc_mock_imports` list. You can see what modules Sphinx is confused by
  when you run into importing errors when generating docs.

Additionally, RST files are needed in order to specify the structure of the
auto-generated pages and to tell Sphinx what modules to generate documentation
for. To auto-generate APIs for a specific module, the `automodule` tag is needed
to tell Sphinx what specific module to document. For example, if we wanted a
page to display auto-generated documentation for everything in
`executorch/exir/__init__.py`, the RST file would look something like the
following:

```
executorch.exir
=======================

.. automodule:: executorch.exir
   :members:
   :undoc-members:
   :show-inheritance:
```

These separate RST files should all be linked together, with the initial landing
page under `index.rst`. A sample of this structure can be found in `source_py/`.

A diagram of how the files work together: ![image](python_docs.png)

To view your changes on the ExecuTorch website, you can follow the same steps
listed in the "General Documentation" section.

To view just the auto-generated pages:

1. `cd executorch/docs/`
2. `sphinx-build -M html source_py sphinxbuild_py` to build Sphinx and generate
   APIs any packages.
3. `python3 -m http.server 8000 --directory sphinxbuild_py/html` to view your
   HTML files at `localhost:8000`.

### C++ APIs

Following Pytorch's way of generating C++ documentation, we generate C++ API
documentation through Doxygen, which is then converted into
[Sphinx](http://www.sphinx-doc.org/) using
[Breathe](https://github.com/michaeljones/breathe) and
[Exhale](https://github.com/svenevs/exhale).

Specifically, we use Doxygen to generate C++ documentation in the form of XML
files, and through configs set in Sphinx's `conf.py` file, we use Breathe and
Exhale to use the XML files and generate RST files which are then used to
generate HTML files.

To configure Doxygen, we can run `doxygen -g` in the root of our repository (ex.
`source_cpp`) which will generate a `Doxyfile` containing configurations for
generating c++ documentation. Specifically, the most important/relevant parts
are:

- `OUTPUT_DIRECTORY` specifies where to output the auto-generated XML files
- `INPUT` specifies which files to generate documenation for
- `GENERATE_XML = YES`

Following PyTorch's `conf.py`
[file](https://github.com/pytorch/pytorch/blob/master/docs/cpp/source/conf.py),
we can set up our own `conf.py` file to take in the directory in which we
generated the XML files, and output HTML to another directory. A sample of this
structure can be found in `source_cpp/`.

A diagram of how the files work together: ![image](cpp_docs.png)

To view the C++ API documentation locally, run:

1. `cd executorch/docs/`
2. `sphinx-build -M html source_cpp sphinxbuild_cpp`
3. `python3 -m http.server 8000 --directory sphinxbuild_py/html` to view your
   HTML files at `localhost:8000`.
