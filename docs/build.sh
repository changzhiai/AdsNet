#!/bin/bash

cd "$(dirname "$0")"

pip install -q sphinx sphinx-rtd-theme sphinx-autobuild

sphinx-autobuild source build/html --open-browser
