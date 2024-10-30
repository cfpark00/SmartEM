import os
import sys
from setuptools import setup, find_packages
from sys import platform

PACKAGE_NAME = "smartem"
REQUIRED_PACKAGES = [
    "numpy",#==1.24.1",
    "matplotlib",#==3.7.2",
    "scipy",#==1.11.1",
    "opencv-python",#==4.8.0.76",
    "pillow",#==9.3.0",
    "scikit-image",#==0.21.0",
    "h5py",#==3.9.0",
    "tqdm",#==4.66.1",
    #"connectomics",
    #"torch"
]

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = "0.1.0"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True
)