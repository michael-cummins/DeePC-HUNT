from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='deepc_hunt',
    version='0.0.6',
    author='Michael Cummins',
    description='PyTorch module for DeePC',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url='https://github.com/michael-cummins/DeePC-HUNT',
    install_requires=[
        'numpy>=1.25.2',
        'cvxpylayers>=1.0',
        'torch>=1.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)
