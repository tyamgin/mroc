#!/usr/bin/python3

from distutils.core import setup, Extension
import numpy.distutils.misc_util

print('Numpy include dir: %s' % numpy.distutils.misc_util.get_numpy_include_dirs())

setup(
    ext_modules=[Extension("_mroc", ["py_module.c", "roc.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)