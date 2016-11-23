"""map ocean model forecasts to observation space"""
import os
import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy


NAME = "obsoper"


# Check if user wants to use Cython to generate C code
USE_CYTHON = False
if "--use-cython" in sys.argv:
    USE_CYTHON = True
    sys.argv.remove("--use-cython")


# Capture __version__
exec(open(os.path.join(NAME, "version.py")).read())


if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize([Extension("*", [os.path.join(NAME, "*.pyx")])])
else:
    # Manually detect C code
    extensions = []
    for path in os.listdir(NAME):
        stem, ext = os.path.splitext(path)
        if ext == ".c":
            module = ".".join([NAME, stem])
            source = os.path.join(NAME, path)
            extension = Extension(module, [source])
            extensions.append(extension)


setup(name=NAME,
      version=__version__,
      description="observation operator",
      long_description=__doc__,
      author="Andrew Ryan",
      author_email="andrew.ryan@metoffice.gov.uk",
      url="https://github.com/met-office-ocean/obsoper",
      packages=find_packages(),
      package_data={
          "obsoper.test": [
              "data/*.nc"
          ]
      },
      ext_modules=extensions,
      include_dirs=[numpy.get_include()])
