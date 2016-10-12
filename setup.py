"""obsoper - observation operator"""
import os
from Cython.Build import cythonize
from setuptools import setup, find_packages
from setuptools.extension import Extension

NAME = "obsoper"

# Capture __version__
exec(open(os.path.join(NAME, "version.py")).read())

extensions = [Extension("*", [os.path.join(NAME, "*.pyx")])]

setup(name=NAME,
      version=__version__,
      description=__doc__,
      author="Andrew Ryan",
      author_email="andrew.ryan@metoffice.gov.uk",
      packages=find_packages(),
      ext_modules=cythonize(extensions))
