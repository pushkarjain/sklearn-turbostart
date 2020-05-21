# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="model_turbostart",
    version="0.0.1",
    description="A package to quick start sklearn-based models",
    author="Pushkar Kumar Jain",
    author_email="pushkarjain1991@utexas.edu",
    license="MIT",
    packages=find_packages(exclude=("notebooks")),
    python_requires=">=3.8.2",
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn",
        "pandas",
        "sklearn-pandas",
    ],
)
