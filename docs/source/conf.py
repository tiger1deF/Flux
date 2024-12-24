import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Flux'
copyright = '2024, Christian de Frondeville'
author = 'Christian de Frondeville'
release = '0.2'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# AutoDoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description' 