python-qubricks
===============

QuBricks is a toolkit for the analysis and simulation of quantum systems using Python. The primary goal of QuBricks is to facilitate insight into quantum systems; rather than to be the fastest or most efficient simulator. As such, the design of QuBricks is not especially geared toward very large or complicated quantum systems. It distinguishes itself from toolkits like QuTip (http://qutip.org/) in that before simulations everything can be expressed symbolically; allowing for analytic observations and computations. Simulations are nonetheless performed numerically, with various optimisations performed to make them more efficient.

For more information regarding the use of QuBricks, refer to `documentation.pdf`.

If you are interested in using QuBricks and are having trouble, please contact me at: mister.wardrop@gmail.com . I will only be too happy to help you.

Installation
------------

If you use `pip`, you can run:

	$ pip install qubricks

Otherwise, installing this module is as easy as:

	$ python2 setup.py install

If you run Arch Linux, you can instead run:

	$ makepkg -i
