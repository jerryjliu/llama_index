"""Configuration for sphinx."""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../"))

with open("../llama_index/VERSION") as f:
    version = f.read()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "LlamaIndex 🦙"
copyright = "2022, Jerry Liu"
author = "Jerry Liu"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "m2r2",
    "myst_nb",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_reredirects",
]

myst_heading_anchors = 4
# TODO: Fix the non-consecutive header level in our docs, until then
# disable the sphinx/myst warnings
suppress_warnings = ["myst.header"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store","DOCS_README.md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = project + " " + version
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
    "css/algolia.css",
    "https://cdn.jsdelivr.net/npm/@docsearch/css@3",
]
html_js_files = [
    "js/mendablesearch.js",
    (
        "https://cdn.jsdelivr.net/npm/@docsearch/js@3.3.3/dist/umd/index.js",
        {"defer": "defer"},
    ),
    ("js/algolia.js", {"defer": "defer"}),
]

nb_execution_mode = "off"
autodoc_pydantic_model_show_json_error_strategy = "coerce"
nitpicky = True

## Redirects

redirects = {
    "end_to_end_tutorials/usage_pattern": "/understanding/understanding.html"
}
