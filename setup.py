from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "gym",
    "box2d-py",
    "tensorflow==2.0.0",
    "tensorflow-probability==0.8.0",
    "attrs",
    "numpy",
]

setup(
    name="awr",
    version="0.0.0",
    url="https://github.com/fomorians/awr",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
)
