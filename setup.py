# -*- coding: utf-8 -*-
"""
Created on 30 Aug 2019

@author: metriczulu
"""

from setuptools import setup, find_packages

setup(

    name='ensemblizer',
    url='https://github.com/metriczulu/ensemblizer',
    author='Shane Stephenson / metriczulu',
    author_email='stephenson.shane.a@gmail.com', 
    packages=find_packages(),
    install_requires = ['numpy', 'tqdm', 'sklearn'],
    version='v0.03',
    license="None",
    description='Package for ensembling models together',
    long_description_content_type='text/markdown',
    long_description=open('README.md', 'r').read(),
    download_url = 'https://github.com/metriczulu/ensemblizer/archive/v0.03.tar.gz'
)
