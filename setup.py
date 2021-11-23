#!/usr/bin/env python
import ez_setup
ez_setup.use_setuptools()


from setuptools import setup
import src
version = src.__version__

setup(name='DMseg',
      version=version,
      description='DMseg',
      author='aaa bbb',
      author_email='abc@fredhutch.org',
      license='MIT',
      url='https://github.com/xwang234/DMseg',
      packages=['src'],
      install_requires=['pandas', 'numpy'],
      scripts=['src/dmseg'],
      long_description=open('README.rst').read(),
      classifiers=["Topic :: Scientific/Engineering :: Bio-Informatics"],
 )
