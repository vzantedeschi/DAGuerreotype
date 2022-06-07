import os

import lpsmap.config
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

# os.environ["CC"] = "/usr/bin/gcc-11";
# os.environ["CXX"] = "/usr/bin/g++-11"
os.environ["CC"] = "/usr/bin/gcc"
os.environ["CXX"] = "/usr/bin/g++"

extensions = [
    Extension(
        "ranksp._permutahedron",
        [
            # "ranksp/_permutahedron.pxd",
            "ranksp/_permutahedron.pyx",
        ],
        libraries=["ad3"],
        language="c++",
        library_dirs=[lpsmap.config.get_libdir()],
        include_dirs=[lpsmap.config.get_include()],
        extra_compile_args=["-std=c++14"],
    )
]

setup(
    name="ranking-sparsemap",
    version="0.1.dev0",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
)
