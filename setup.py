#!/usr/bin/env python3
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: DeepDialog
# Mail: thebotbot@sina.com
# Created Time: 2019-11-16 14:40:00
#############################################

import os
from setuptools import setup, find_packages

current_dir = os.path.realpath(os.path.dirname(__file__))

ON_RTD = os.environ.get('READTHEDOCS') == 'True'
if not ON_RTD:
    INSTALL_REQUIRES = open(os.path.join(
        current_dir,
        'requirements.txt'
    )).read().split('\n')
else:
    INSTALL_REQUIRES = []

VERSION = os.path.join(
    current_dir,
    'bert_chinese_bin',
    'version.txt'
)

setup(
    name='bert-chinese-bin',
    version=open(VERSION, 'r').read().strip(),
    keywords=('pip', 'text', 'NLP'),
    description='NLP tool',
    long_description='NLP tool',
    license='Private',
    url='https://github.com/deepdialog/bert-chinese-bin',
    author='deepdialog',
    author_email='thebotbot@sina.com',
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    install_requires=INSTALL_REQUIRES
)
