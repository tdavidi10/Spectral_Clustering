from setuptools import setup, Extension

setup(name='myspkmeanssp',
      version='1.0',
      description='myspkmeanssp setup.py',
      ext_modules=[Extension("myspkmeanssp", sources=['spkmeansmodule.c','spkmeans.c'])])
