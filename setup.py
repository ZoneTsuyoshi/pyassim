# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyassim',
    version='0.0.1',
    description='Package for data assimilation',
    long_description=readme,
    author='Tsuyoshi Ishizone',
    author_email='tsuyoshi.ishizone@gmail.com',
    install_requires=["numpy", "scipy", "scikit-learn", "pandas"],
    python_requires=">=3",
    url='https://github.com/ZoneTsuyoshi/pyassim',
    license=license,
    packages=find_packages(include=('pyassim')),
    py_modules=["math", "logging", "os", "time", "multiprocessing", "itertools", "inspect"],
    test_suite='tests'
)

