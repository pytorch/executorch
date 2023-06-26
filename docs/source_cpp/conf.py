# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Configuration file for the Sphinx documentation builder.
# Created by isphinx-quickstart on Tue Jul 19 14:58:12 2022.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os

import pytorch_sphinx_theme

FBCODE = "fbcode" in os.getcwd()

# -- Project information -----------------------------------------------------

project = "executorch"
copyright = "2022, executorch"
author = "executorch"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "exhale",
]

# Setup absolute paths for communicating with breathe / exhale where
# items are expected / should be trimmed by.
# This file is {repo_root}/docs/cpp/source/conf.py
this_file_dir = os.path.abspath(os.path.dirname(__file__))
doxygen_xml_dir = os.path.abspath("../website/sphinxbuild_cpp/xml")
repo_root = os.path.dirname(  # {repo_root}
    os.path.dirname(  # {repo_root}/docs
        os.path.dirname(  # {repo_root}/docs/cpp
            this_file_dir  # {repo_root}/docs/cpp/source
        )
    )
)

breathe_projects = {"PyTorch": doxygen_xml_dir}
breathe_default_project = "PyTorch"

# Setup the exhale extension
exhale_args = {
    ############################################################################
    # These arguments are required.                                            #
    ############################################################################
    "containmentFolder": "./cpp_api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Library API",
    "doxygenStripFromPath": repo_root,
    ############################################################################
    # Suggested optional arguments.                                            #
    ############################################################################
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleUseDoxyfile": True,
    "verboseBuild": True,
    ############################################################################
    # HTML Theme specific configurations.                                      #
    ############################################################################
    # Fix broken Sphinx RTD Theme 'Edit on GitHub' links
    # Search for 'Edit on GitHub' on the FAQ:
    #     http://exhale.readthedocs.io/en/latest/faq.html
    "pageLevelConfigMeta": ":github_url: https://github.com/pytorch/pytorch",
    ############################################################################
    # Individual page layout example configuration.                            #
    ############################################################################
    # Example of adding contents directives on custom kinds with custom title
    "contentsTitle": "Page Contents",
    "kindsWithContentsDirectives": ["class", "file", "namespace", "struct"],
    # Exclude PIMPL files from class hierarchy tree and namespace pages.
    "listingExclude": [r".*Impl$"],
    ############################################################################
    # Main library page layout example configuration.                          #
    ############################################################################
    "afterTitleDescription": "Welcome to the Executorch's documentation.",
}

# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "cpp"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "pytorch_project": "executorch",
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
