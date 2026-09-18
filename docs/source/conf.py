import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

project = 'AdsNet'
from datetime import datetime
copyright = f'2025-{datetime.now().year}, Changzhi Ai'
author = 'Changzhi Ai'
release = '0.1.1'

templates_path = ['_templates']
html_static_path = ['_static']
exclude_patterns = []

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',
]

autodoc_mock_imports = ['torch', 'ase', 'numpy', 'asap3', 'scipy']

autodoc_class_signature = 'mixed'
