# Copyright (c) Meta Platforms, Inc. and affiliates.
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
import glob
import os
import shutil
import sys
from typing import Any

import pytorch_sphinx_theme2  # type: ignore[import-not-found]

# To let us import ./custom_directives.py
sys.path.insert(0, os.path.abspath("."))
# -- Project information -----------------------------------------------------

project = "ExecuTorch"
copyright = "2024, ExecuTorch"
author = "ExecuTorch Contributors"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

import os
import sys

FBCODE = "fbcode" in os.getcwd()

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "myst_parser",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_reredirects",
    "sphinx_sitemap",
    "sphinxcontrib.mermaid",
]

this_file_dir = os.path.abspath(os.path.dirname(__file__))
doxygen_xml_dir = os.path.join(
    os.path.dirname(this_file_dir),  # {repo_root}/docs/
    "build",  # {repo_root}/docs/build
    "xml",  # {repo_root}/docs/cpp/build/xml
)

html_favicon = "_static/img/ExecuTorch-Logo-cropped.svg"

# Get ET_VERSION_DOCS during the build.
et_version_docs = os.environ.get("ET_VERSION_DOCS", None)
print(f"et_version_docs: {et_version_docs}")

# The code below will cut version displayed in the dropdown like this:
# By default, set to "main".
# If it's a tag like refs/tags/v1.2.3-rc4 or refs/tags/v1.2.3, then
# cut to 1.2
# the version varible is used in layout.html: https://github.com/pytorch/executorch/blob/main/docs/source/_templates/layout.html#L29
version = release = "main"
if et_version_docs:
    if et_version_docs.startswith("refs/tags/v"):
        version = ".".join(
            et_version_docs.split("/")[-1].split("-")[0].lstrip("v").split(".")[:2]
        )
    elif et_version_docs.startswith("refs/heads/release/"):
        version = et_version_docs.split("/")[-1]
print(f"Version: {version}")
html_title = " ".join((project, version, "documentation"))

html_baseurl = "https://docs.pytorch.org/executorch/"  # needed for sphinx-sitemap
sitemap_locales = [None]
sitemap_excludes = [
    "search.html",
    "genindex.html",
]
sitemap_url_scheme = "{link}"

breathe_projects = {"ExecuTorch": "../build/xml/"}
breathe_default_project = "ExecuTorch"

autodoc_typehints = "description"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

myst_heading_anchors = 4

sphinx_gallery_conf: dict[str, Any] = {
    "examples_dirs": ["tutorials_source"],
    "ignore_pattern": "template_tutorial.py",
    "gallery_dirs": ["tutorials"],
    "filename_pattern": "/tutorials_source/",
    "promote_jupyter_magic": True,
    "backreferences_dir": None,
    "first_notebook_cell": ("%matplotlib inline"),
}

assert len(sphinx_gallery_conf["examples_dirs"]) == len(
    sphinx_gallery_conf["gallery_dirs"]
), "Lengths of galery_dirs and examples_dir must be same."

for i in range(len(sphinx_gallery_conf["examples_dirs"])):
    gallery_dir = sphinx_gallery_conf["gallery_dirs"][i]
    source_dir = sphinx_gallery_conf["examples_dirs"][i]

    # Create gallery dirs if it doesn't exist
    os.makedirs(gallery_dir, exist_ok=True)

    # Copy .md files from source dir to gallery dir
    for f in glob.glob(os.path.join(source_dir, "*.md")):

        shutil.copyfile(f, gallery_dir)

source_suffix = [".rst", ".md"]


autodoc_typehints = "none"

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["../_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "tutorial-template.md"]
exclude_patterns += sphinx_gallery_conf["examples_dirs"]
exclude_patterns += ["*/index.rst"]

# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme2"
html_theme_path = [pytorch_sphinx_theme2.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#

switcher_version = version

html_theme_options = {
    "logo": {
        "image_light": "_static/img/et-logo.png",
        "image_dark": "_static/img/et-logo.png",
    },
    "navigation_with_keys": False,
    "canonical_url": "https://docs.pytorch.org/executorch/stable/",
    "switcher": {
        "json_url": "https://docs.pytorch.org/executorch/executorch-versions.json",  # for testing only, will need to replace to the correct json file on the executorch website when it's added in the repo.
        "version_match": switcher_version,
    },
    "show_toc_level": 2,
    "analytics_id": "GTM-T8XT4PS",
    "icon_links": [
        {
            "name": "X",
            "url": "https://x.com/PyTorch",
            "icon": "fa-brands fa-x-twitter",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/pytorch/executorch",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discourse",
            "url": "https://discuss.pytorch.org/",
            "icon": "fa-brands fa-discourse",
        },
        {
            "name": "PyPi",
            "url": "https://pypi.org/project/executorch",
            "icon": "fa-brands fa-python",
        },
    ],
    "show_version_warning_banner": True,
    "use_edit_page_button": True,
    "header_links_before_dropdown": 8,
    "navbar_align": "left",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["search-field-custom", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
}

theme_variables = pytorch_sphinx_theme2.get_theme_variables()
templates_path = [
    "_templates",
    os.path.join(os.path.dirname(pytorch_sphinx_theme2.__file__), "templates"),
]

html_context = {
    "theme_variables": theme_variables,
    "display_github": True,
    "github_url": "https://github.com",
    "github_user": "pytorch",
    "github_repo": "executorch",
    "feedback_url": "https://github.com/pytorch/executorch",
    "github_version": "main",
    "doc_path": "docs/source",
    "pytorch_project": "executorch",
    "display_version": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Add custom 404 page for GitHub Pages
html_additional_pages = {"404": "404.html"}


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# Redirects for moved pages
redirects = {
    "getting-started-setup": "getting-started.html",
    "export-overview": "using-executorch-export.html",
    "runtime-build-and-cross-compilation": "using-executorch-building-from-source.html",
    "tutorials/export-to-executorch-tutorial": "../using-executorch-export.html",
    "build-run-vulkan": "backends-vulkan.html",
    "executorch-arm-delegate-tutorial": "backends-arm-ethos-u.html",
    "build-run-coreml": "backends-coreml.html",
    "build-run-mediatek-backend": "backends-mediatek.html",
    "build-run-mps": "backends-mps.html",
    "build-run-qualcomm-ai-engine-direct-backend": "backends-qualcomm.html",
    "build-run-xtensa": "backends-cadence.html",
    "apple-runtime": "using-executorch-ios.html",
}

# Custom directives defintions to create cards on main landing page

from custom_directives import (  # type: ignore[import-not-found]
    CustomCardEnd,
    CustomCardItem,
    CustomCardStart,
    SupportedDevices,
    SupportedProperties,
)
from docutils.parsers import rst  # type: ignore[import-untyped]

# Register custom directives


rst.directives.register_directive("devices", SupportedDevices)
rst.directives.register_directive("properties", SupportedProperties)
rst.directives.register_directive("customcardstart", CustomCardStart)
rst.directives.register_directive("customcarditem", CustomCardItem)
rst.directives.register_directive("customcardend", CustomCardEnd)
