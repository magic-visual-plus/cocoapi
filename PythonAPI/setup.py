from setuptools import setup, Extension
import numpy as np
import distutils.ccompiler


#
# If running on Windows, compiler_name will be "msvc" so don't use gcc options
#
compiler_name = distutils.ccompiler.get_default_compiler()
if compiler_name == "msvc":
    extra_compile_args=[]
else:
    extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99']

print(f'extra_compile_args {extra_compile_args}')

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=['../common/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs = [np.get_include(), '../common'],
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir = {'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0'
    ],
    version='2.0',
    ext_modules= ext_modules
)
