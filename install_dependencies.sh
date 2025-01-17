#!/bin/bash


# Install Keras and other necessary libraries not included in the conda create command
pip install keras

# Install other required libraries
pip install os
pip install datetime
pip install shutil
pip install matplotlib
pip install opencv-python
pip install imageio
pip install scipy
pip install warnings

# Additional dependencies (for scikit-image if needed for commented-out imports)
# pip install scikit-image

# Note: pdb and warnings are part of Python's standard library and do not require installation
