#!/usr/bin/env python2

from distutils.core import setup

setup(name='qubricks',
      version='0.93',
      description='A library to assist in the rapid prototyping and simulation of quantum systems.',
      author='Matthew Wardrop',
      author_email='mister.wardrop@gmail.com',
      url='http://www.matthewwardrop.info/',
      #package_dir={'parameters':'.'},
      download_url='https://github.com/matthewwardrop/python-qubricks',
      packages=['qubricks','qubricks.utility','qubricks.wall','qubricks.analysis'],
      requires=['numpy','sympy(>0.7.2)','scipy','parameters(>=1.2)'],
      license='''The MIT License (MIT)

Copyright (c) 2013 Matthew Wardrop

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'''

     )
