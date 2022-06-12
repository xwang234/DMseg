#!/usr/bin/env python
#import ez_setup
#ez_setup.use_setuptools()

Version = "0.1.1"
DESCRIPTION = 'Python package for DMseg: detecting differential methylation regions in DNA methylome data'
from setuptools import setup, find_packages

setup(name='DMseg',
      version=Version,
      author='Kevin Wang',
      author_email='xwang234@fredhutch.org',
      license='MIT',
            description=DESCRIPTION,
      url='https://github.com/xwang234/DMseg',
      install_requires=['pandas', 'numpy'],
      packages=find_packages(),
      scripts=['src/dmseg'],
      long_description=open('README.md').read(),
      classifiers=["Topic :: Scientific/Engineering :: Bio-Informatics"],
      include_package_data=True,
      package_data={'': ['data/*.csv']},
 )
