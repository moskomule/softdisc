from setuptools import find_packages, setup

install_requires = [
    'torch>=1.8.1',
]

setup(
    name='softdisc',
    version='0.0.1',
    description='Differentiable Discrete Algorithms for PyTorch',
    author='Ryuichiro Hataya',
    author_email='hataya@nlab-mpg.jp',
    install_requires=install_requires,
    packages=find_packages(exclude=["tests"])
)
