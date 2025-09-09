#!/bin/bash
set -e

pip install tifffile matplotlib numpy pillow opencv-python

python -c "from opencv_fixer import AutoFix; AutoFix()"

