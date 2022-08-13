import os
import sys
import numpy

from Cython.Build import cythonize
from distutils.extension import Extension


def build(setup_kwargs):
    print("Building...")
    print(setup_kwargs)

    sourcefiles = [
        'sent2vec.pyx', 
        'fasttext.cc',
        'args.cc',
        'dictionary.cc',
        'matrix.cc',
        'shmem_matrix.cc',
        'qmatrix.cc',
        'model.cc',
        'real.cc',
        'utils.cc',
        'vector.cc',
        'real.cc',
        'productquantizer.cc'
    ]
    compile_opts = ['-std=c++0x', '-Wno-cpp', '-pthread', '-Wno-sign-compare']
    libraries = ['rt']
    if sys.platform == 'darwin':
        libraries = []

    ext = [Extension(
        '*',
        [os.path.join("sent2vec", "src", f) for f in sourcefiles],
        extra_compile_args=compile_opts,
        language='c++',
        include_dirs=[numpy.get_include()],
        libraries=libraries
    )]

    setup_kwargs.update({"ext_modules": cythonize(ext)})
