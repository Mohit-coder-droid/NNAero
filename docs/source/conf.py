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
    'nbsphinx',
    'nbsphinx_link',
    'sphinx.ext.autodoc',    # Pulls docstrings from code
    'sphinx.ext.napoleon',   # Supports Google/NumPy style docstrings
    'sphinx.ext.viewcode',   # Adds links to highlighted source code
    'sphinx.ext.mathjax',
]

# Optional: Do not execute notebooks when building docs
# (Useful if your notebooks take a long time to run)
nbsphinx_execute = 'never' 

# Optional: Add a timeout for cell execution (in seconds)
nbsphinx_timeout = 60

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
