#!/usr/bin/env python

Version = "0.1.2"
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
      entry_points={ 
            "console_scripts": [
            "dmseg=DMseg.dmseg:main",
            ],
      },
      long_description=open('README.md').read(),
      classifiers=["Topic :: Scientific/Engineering :: Bio-Informatics"],
      include_package_data=True,
      package_data={'DMseg': ['data/*.csv', 'img/*.png']},
 )
