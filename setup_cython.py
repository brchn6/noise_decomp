from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="noise_decomp._cnoise",
        sources=["src/noise_decomp/_cnoise.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="noise_decomp_c_ext",
    ext_modules=cythonize(extensions),
    zip_safe=False,
)
