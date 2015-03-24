Introduction
------------

QuBricks is a toolkit for the analysis and simulation of quantum systems in Python.
The primary goal of QuBricks
is to facilitate insight into quantum systems; rather than to be the fastest or most
efficient simulator. As such, the design of QuBricks is not especially geared toward
very large or complicated quantum systems.
It distinguishes itself from toolkits like QuTip (http://qutip.org) in that before simulations,
everything can be expressed symbolically; allowing for analytic observations
and computations. Simulations are nonetheless performed numerically, with various
optimisations performed to make them more efficient.

Basic operations are unit-tested with reference to a simple two-level system.
