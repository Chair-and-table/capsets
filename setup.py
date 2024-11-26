# setup.py

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "solution.pyx", compiler_directives={"language_level" : "3"}
    )

)