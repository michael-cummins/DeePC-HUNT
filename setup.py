from setuptools import setup, find_packages

setup(
    name='deepc_hunt',
    version='0.0.5',
    author='Michael Cummins',
    description='PyTorch module for DeePC',
    packages=['deepc_hunt'],
    install_requires=[
        'numpy==1.25.2',
        'cvxpylayers',
        'torch',
        'matplotlib',
        'tqdm'
    ]
)