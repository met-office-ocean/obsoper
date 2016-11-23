"""map ocean model forecasts to observation space"""
import os
from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy

NAME = "obsoper"

# Capture __version__
exec(open(os.path.join(NAME, "version.py")).read())

extensions = [Extension("*", [os.path.join(NAME, "*.pyx")])]

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
      ext_modules=cythonize(extensions),
      include_dirs=[numpy.get_include()])
