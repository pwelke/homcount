#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='ghc',
    version='0.3',
    license='MIT',
    description='Code to compute Expectation-Complete Graph Embeddings',
    author='Pascal Welke, Maximilian Thiessen, NT Hoang',
    author_email='pascal.welke@tuwien.ac.at',
    url='https://github.com/pwelke/homcount',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob(join('src', '*.py'))],
    data_files=[('ghc/utils', ['src/ghc/utils/logprimes1.npy'])],
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'graph homomorphism', 'graph neural networks', 'expressive graph representation'
    ],
    python_requires='>=3.7',
)
