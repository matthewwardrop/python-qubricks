Quick Start
===========

In this chapter, the basic behaviour of QuBricks is demonstrated in the
context of some simple problems. For specific documentation on methods and
their usage, please refer to the API documentation is subsequent chapters.

In the following, we assume that the following has been run in your 
Python environment.

>>> from qubricks import *
>>> from qubricks.wall import *
>>> import numpy as np

.. note:: QuBricks current only works in Python 2, since it depends on
	python-parameters which in turn is compatible only with Python 2.

Getting Started
---------------

In this section, we motivate the use of QuBricks in simple quantum
systems. We will make use of the submodule qubricks.wall, which brings
together many of the “bricks” that make up QuBricks into usable
standalone tools.

Consider a single isolated electron in a magnetic field. Suppose that this 
magnetic field was composed of stable magnetic field along the Z axis :math:`B_z`, and a 
noisy magnetic field along the X axis :math:`B_x(t)`. The Hamiltonian
describing the mechanics is then:

.. math::

   \begin{aligned}
   H & = & \mu_B \left(\begin{array}{cc}
   B_z & B_x(t)\\
   B_x(t) & -B_z
   \end{array}\right),\end{aligned}

where :math:`B_z` is the magnetic field along :math:`z` and
:math:`B_x(t) = \bar{B}_{x} + \tilde{B}_x(t)` is a noise field along :math:`x`
centred on some nominal value :math:`\bar{B}_{x}` with a time-dependent noisy
component :math:`\tilde{B_x}(t)`.

Let us assume that the noise in :math:`B_x(t)` is white, so that:

.. math:: \left< \tilde{B}_x(t) \tilde{B}_x(t^\prime) \right> = D\delta(t-t^\prime),

We can model such white noise using a Lindblad superoperator.

This is a simple system which can be analytically solved. 
Evolution under this Hamiltonian will lead to the
electron's spin gyrating around an axis between Z and X (i.e. at an
angle :math:`\theta = \tan^{-1}(B_z/J_x)` from the x-axis) at a
frequency of :math:`2\sqrt{B_z^2 + B_x^2}`. The effect of high frequency
noise in :math:`B_x` is to progressively increase the mixedness in the Z
quadrature until such time as measurements of Z are unbiased. 
For example, when :math:`B_z=0`, the return probability for an
initially up state is given by:
:math:`p = \frac{1}{2}(1+\cos{2B_{x}t})`. Since
:math:`\left< \tilde{B_{x}}^2 \right>_{\textrm{avg}} = D/t`, we find by taylor
expanding that: :math:`\left<p\right> = 1 - Dt`. A more careful
analysis would have found that:

.. math:: \left<p\right> = \frac{1}{2}(1+exp(-2Dt)) .

It is possible to find approximations for a general :math:`\theta`, but
we leave that as an exercise. Alternatively, you can take advantage of
QuBricks to simulate these results for us.

.. _two-level:
.. figure:: _static/twolevel.pdf

   Dynamics of a single electron spin under a magnetic field aligned 45
   degrees in the XZ plane.

For example, suppose we wanted to examine the case where :math:`B_z = B_x`, as
shown in figure :num:`two-level`. This can be simulated using the following 
QuBricks code:

.. literalinclude:: _static/twolevel.py
	:linenos:
	:lines: 4-24

We could then plot the results using `matplotlib`:

.. literalinclude:: _static/twolevel.py
	:linenos:
	:lines: 26-48

This would result in the following plot:

.. image:: _static/results.pdf

The above code takes advantage of several attributes and methods of 
`QuantumSystem` instances which may not be entirely clear. At this point, you can
look them up in the API reference in subsequent chapters.

Advanced Usage
--------------

For more fine-grained control, one can subclass `QuantumSystem`,
`Measurement`, `StateOperator` and `Basis` as necessary. For more information about
which methods are available on these objects, refer to the API
documentation below. The templates for subclassing are shown below.

QuantumSystem
~~~~~~~~~~~~~
.. literalinclude:: ../templates/quantum_system.py
	:linenos:

Measurement
~~~~~~~~~~~
.. literalinclude:: ../templates/measurement.py
	:linenos:

StateOperator
~~~~~~~~~~~~~
.. literalinclude:: ../templates/state_operator.py
	:linenos:

Basis
~~~~~
.. literalinclude:: ../templates/basis.py
	:linenos:

Integrator
~~~~~~~~~~
In rare circumstances, you may find it necessary to subclass the `Integrator` 
class. Refer to the API documentation for more details on how to do this.

.. literalinclude:: ../templates/integrator.py
	:linenos:


Operator Basics
---------------

One class that is worth discussing in more detail is `Operator`, which is among 
the most important "bricks" in the QuBricks library. It represents
all of the two-dimensional linear operators used in QuBricks. The
Operator object is neither directly a symbolic or numeric representation
of an operator; but can be used to generate both.

Consider a simple example:

>>> op = Operator([[1,2],[3,4]])
>>> op
<Operator with shape (2,2)>

To generate a matrix representation of this object for inspection, we
have two options depending upon whether we want a symbolic or numeric
representation.

>>> op() # Numeric representation as a NumPy array 
array([[ 1.+0.j, 2.+0.j], 
       [ 3.+0.j, 4.+0.j]])
>>> op.symbolic() # Symbolic representation as a SymPy matrix 
Matrix([
[1, 2],
[3, 4]])

In this case, there is not much difference, of course, since there
are no symbolic parameters used.

Creating an Operator object with named parameters can be done in two
ways. Either you must create a dictionary relating parameter names to
matrix forms, or you can create a SymPy symbolic matrix. In both cases,
one then passes this to the Operator constructor. For example:

>>> op = Operator('B':[[1,0],[0,-1]], 'J':[[0,1],[1,0]])
>>> op.symbolic()
Matrix([
[B, J],
[J, -B]])
>>> op.symbolic(J=10)
Matrix([
[ B, 10],
[10, -B]])
>>> op()
ValueError: Operator requires use of Parameters object; but none specified.

When representing Operator objects symbolically, we can override some
parameters and perform parameter substitution. We see that attempting to
generate a numeric representation of the Operator object failed, because
it did not know how to assign a value to :math:`B` and :math:`J`.
Normally, Operator objects will have a reference to a `Parameters`
instance (from `python-parameters`) passed to it in the constructor phase,
for which these parameters can be extracted. This will in most cases be
handled for you by QuantumSystem (see `QuantumSystem` in the API chapters), but
for completeness there are two keyword arguments you can pass to
Operator instances: `parameters`, which shold be a reference to an
existing Parameters instance, and `basis`, which should be a reference to
an existing Basis object or None (see `Basis` in the API chapters). For now, let us
manually add it for demonstration purposes.

>>> from parameters import Parameters
>>> p = Parameters()
>>> p(B=2,J=1)
< Parameters with 2 definitions >
>>> op = Operator('B':[[1,0],[0,-1]], 'J':[[0,1],[1,0]],parameters=p)
>>> op()
array([[ 2.+0.j,  1.+0.j],
       [ 1.+0.j, -2.+0.j]])
>>> op(J=10,B=1)
array([[  1.+0.j, 10.+0.j],
       [ 10.+0.j, -1.+0.j]])

We see in the above that we can take advantage of temporary parameter
overrides for numeric representations too [note that a parameters
instance is still necessary for this].

The `Parameters` instance allows one to have parameters which are functions
of one another, which allows for time and/or context dependent
operators.

Operator objects support basic arithmetic: addition, subtraction, and
multiplication using the standard python syntax. The inverse operation
can be performed using the inverse method:

>>> op.inverse()

The Kronecker tensor product can be applied using the tensor method:

>>> op.tensor(other_op)

To apply an Operator object to a vector, you can either use the standard
inbuilt multiplication operations, or use the slightly more optimised
apply method.

If you are only interested in how certain parameters affect the
operator, then to improve performance you can "collapse" the Operator
down to only include variables which depend upon those variables.

>>> op.collapse('t',J=1)

The result of the above command would substitute all variables (with a
parameter override of :math:`J=1`) that do not depend upon :math:`t` with
their numerical value, and then perform various optimisations to make
further substitutions more efficient. This is used, for example, by the
integrator.

The last set of key methods of the Operator object are the connected and
restrict methods. Operator.connected will return the set of all indicies
(of the basis vectors in which the Operator is represented) that are
connected by non-zero matrix elements, subject to the provided parameter
substitution. Note that this comparison is done with the numerical
values of the parameters.

>>> op = Operator('B':[[1,0],[0,-1]], 'J':[[0,1],[1,0]],parameters=p)
>>> op.connected(0)
0,1
>>> op.connected(0, J=0)
0

The restrict method returns a new Operator object which keeps only the
entries in the old Operator object which correspond to the basis
elements indicated by the indicies.

>>> op = Operator('B':[[1,0],[0,-1]], 'J':[[0,1],[1,0]],parameters=p)
>>> op.restrict(0)
<Operator with shape (1, 1)>
>>> op.symbolic()
Matrix([[B]])

For more detail, please refer to the API documentation for `Operator`.
