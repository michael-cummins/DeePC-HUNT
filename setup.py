from setuptools import setup, find_packages

setup(
    name='deepc_hunt',
    version='0.0.1',
    author='Michael Cummins',
    description='PyTorch module for DeePC',
    packages=['deepc_hunt'],
    install_requires=[
        'cvxpylayers==0.1.6',
        'torch==2.0.1',
        'mpc @ git+https://github.com/locuslab/mpc.pytorch.git@63732fa85ab2a151045493c4e67653210ca3d7ff',
        'matplotlib',
        'numpy',
        'cvxpy',
        'tqdm'
    ]
)