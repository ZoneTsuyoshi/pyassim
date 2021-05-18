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
    long_description_content_type='text/plain',
    author='Tsuyoshi Ishizone',
    author_email='tsuyoshi.ishizone@gmail.com',
    install_requires=["numpy>=1.13.3"],
    python_requires=">=3",
    url='https://github.com/ZoneTsuyoshi/pyassim',
    license=license,
    packages=find_packages(include=('pyassim')),
    package_dir={"pyassim":"pyassim"},
    py_modules=["math", "logging", "os", "time", "multiprocessing", "itertools", "inspect"],
    test_suite='tests',
    classifier=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)

