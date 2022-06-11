#!/usr/bin/env python
#import ez_setup
#ez_setup.use_setuptools()


from setuptools import setup, find_packages
import src
version = src.__version__

setup(name='DMseg',
      version=version,
      description='DMseg',
      author='Kevin Wang',
      author_email='xwang234@fredhutch.org',
      license='MIT',
      url='https://github.com/',
      #packages=['src'],
      install_requires=['pandas', 'numpy'],
      packages=find_packages(),
            scripts=['src/dmseg'],
      long_description=open('README.md').read(),
      classifiers=["Topic :: Scientific/Engineering :: Bio-Informatics"],
      include_package_data=True,
      package_data={'': ['data/*.csv']},
 )
