#!/usr/bin/env python
import ez_setup
ez_setup.use_setuptools()


from setuptools import setup
import src
version = src.__version__

setup(name='DMseg',
      version=version,
      description='DMseg',
      author='Kevin Wang',
      author_email='xwang234@fredhutch.org',
      license='MIT',
      url='https://github.com/',
      packages=['src'],
      install_requires=['pandas', 'numpy'],
      scripts=['src/dmseg'],
      long_description=open('README.rst').read(),
      classifiers=["Topic :: Scientific/Engineering :: Bio-Informatics"],
 )
