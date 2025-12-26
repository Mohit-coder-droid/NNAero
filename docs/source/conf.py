# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NNAero'
copyright = '2025, Mohit Sahu'
author = 'Mohit Sahu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
import os
import sys
# Point Sphinx to your source code
sys.path.insert(0, os.path.abspath('../../../nnaero')) # Adjust path to where your .py files are

extensions = [
    'sphinx.ext.autodoc',    # Pulls docstrings from code
    'sphinx.ext.napoleon',   # Supports Google/NumPy style docstrings
    'sphinx.ext.viewcode',   # Adds links to highlighted source code
]


html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
