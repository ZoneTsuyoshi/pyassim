# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyassim',
    version='0.1.0',
    description='Package for data assimilation',
    long_description=readme,
    author='Tsuyoshi Ishizone',
    author_email='',
    install_requires=['numpy', 'pandas', 'scipy']
    url='https://github.com/ZoneTsuyoshi/pyassim',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

