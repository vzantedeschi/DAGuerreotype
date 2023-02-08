from setuptools import find_packages, setup
from setuptools.extension import Extension


import lpsmap.config
from Cython.Build import cythonize

from torch.utils import cpp_extension


__version__ = '0.1dev0'

# Do not override here. If you need to change compiler,
# set the environment variable when running setup.
# os.environ["CC"] = "/usr/bin/gcc"
# os.environ["CXX"] = "/usr/bin/g++"

extensions = [

    # Cython extension built on top of the lpsmap package
    Extension(
        "daguerreo.permutahedron._sparsemap",
        [
            "daguerreo/permutahedron/_sparsemap.pyx",
        ],
        libraries=["ad3"],
        language="c++",
        library_dirs=[lpsmap.config.get_libdir()],
        include_dirs=[lpsmap.config.get_include()],
        extra_compile_args=["-std=c++14"],
    ),

    # pytorch pybind11 extension
    cpp_extension.CppExtension('daguerreo.permutahedron._kbest',
        [
            "daguerreo/permutahedron/kbest.cpp"
        ],
        language='c++',
        extra_compile_args=['-std=c++14'],
    ),
]

setup(
    name="daguerreo",
    version=__version__,
    author="Zantedeschi,Franceschi,Kaddour,Niculae,Kusner",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    setup_requires=['setuptools', 'pybind11>=2.5.0'],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    zip_safe=False
)
