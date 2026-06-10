import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

project = 'AdsNet'
copyright = '2025, Changzhi Ai'
author = 'Changzhi Ai'
release = '0.1.0'

templates_path = ['_templates']
exclude_patterns = []

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',
]

autodoc_mock_imports = ['torch', 'ase', 'numpy']

autodoc_class_signature = 'mixed'
