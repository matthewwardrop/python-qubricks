Quick Start
===========

The main purpose of this documentation is to demonstrate how to use
QuBricks, rather than to explain the mechanisms of how it works; though
there is obviously some overlap. In section [sec:basic:sub:`u`\ sage] we
demonstrate some extremely basic use cases, before exploring more
interesting cases in section [sec:advanced:sub:`u`\ sage].

In the following, we assume that the following has been run:

>>> from qubricks import Operator, QuantumSystem... >>> import numpy as
np

Basic Usage[sec:basic\ :sub:`u`\ sage]
--------------------------------------

In this section, we motivate the use of QuBricks in simple quantum
systems. We will make use of the submodule qubricks.wall, which brings
together many of the “bricks” that make up QuBricks into usable
standalone tools.

Consider a single isolated electron in a magnetic field aligned with Z
axis, and a noisy magnetic field along the X axis. The Hamiltonian
describing the mechanics is:

.. math::

   \begin{aligned}
   H & = & \mu_B \left(\begin{array}{cc}
   B_z & B_x\\
   B_x & -B_z
   \end{array}\right),\end{aligned}

where :math:`B_z` is the magnetic field along :math:`z` and
:math:`B_x= B_{x0} + \tilde{B_x}` is a noise field along :math:`x`
centred on some nominal value :math:`B_{x0}` with a high frequency noise
component :math:`\tilde{B_x}`. We assume the noise is white:

.. math:: \left< (B_x-B_{x0})_{t} (B_x-B_{x0})_{t^\prime} \right> = D\delta(t-t^\prime),

 and model it using a Lindblad superoperator.

This is a simple system which can be analytically solved easily in the
absence of noise. Evolution under this Hamiltonian will lead to the
electron’s spin gyrating around an axis between Z and X (i.e. at an
angle :math:`\theta = \tan^{-1}(B_z/J_x)` from the x-axis) at a
frequency of :math:`2\sqrt{B_z^2 + B_x^2}`. The effect of high frequency
noise in :math:`B_x` will continually increase the mixedness in the Z
quadrature until such time as measurements of Z are unbiased. The effect
of noise can also be reasonably simply computed, at least to first
order. For example, when :math:`B_z=0`, the return probability for an
initially up state is given by:
:math:`p = \frac{1}{2}(1+\cos{2B_{x}t})`. Given that we know that:
:math:`\left< \tilde{B_{x}}^2 \right>_{\textrm{avg}} = D/t`, by taylor
expanding we find: :math:`\left<p\right> = 1 - Dt`. A more careful
analysis finds that:

.. math:: \left<p\right> = \frac{1}{2}(1+exp(-2Dt)) .

It is possible to find approximations for a general :math:`\theta`, but
we leave that as an exercise.

The dynamics of the system can be simulated using the code below, where
we have chosen :math:`B_z = B_x`, and so dynamics should evolve as shown
in figure :ref:`two-level` [fig:twolevel].

.. image:: _static/results.pdf

.. _two-level:
.. figure:: _static/twolevel.pdf

   Dynamics of a single electron spin under a magnetic field aligned 45
   degrees in the XZ plane.



Advanced Usage[sec:advanced\ :sub:`u`\ sage]
--------------------------------------------

For more fine-grained control, one can subclass QuantumSystem,
Measurement and StateOperator as necessary. For more information about
which methods are available on these objects, please use the inline
python help: help(python-obj). The templates are as follows:


Operator Basics
---------------

Perhaps the most important brick is the Operator object. It represents
all of the two-dimensional linear operators used in QuBricks. The
Operator object is neither directly a symbolic or numeric representation
of an operator; but can be used to generate both.

Consider a simple example:

>>> op = Operator(np.array([[1,2],[3,4]])) >>> op <Operator with shape
(2,2)>

To generate a matrix representation of this object for inspection, we
have two options depending upon whether we want a symbolic or numeric
representation.

>>> op() # Numeric representation as a NumPy array array([[ 1.+0.j,
2.+0.j], [ 3.+0.j, 4.+0.j]]) >>> op.symbolic() # Symbolic representation
as a SymPy matrix Matrix([ [1, 2], [3, 4]])

In this case, there is not much difference.

Creating an Operator object with named parameters can be done in two
ways. Either you must create a dictionary relating parameter names to
matrix forms, or you can create a SymPy symbolic matrix. In both cases,
one then passes this to the Operator constructor instead of the numpy
array above. For example:

>>> op = Operator(’B’:[[1,0],[0,-1]], ’J’:[[0,1],[1,0]]) >>>
op.symbolic() Matrix([ [B, J], [J, -B]]) >>> op.symbolic(J=10) Matrix([
[ B, 10], [10, -B]]) >>> op() ValueError: Operator requires use of
Parameters object; but none specified.

When representing Operator objects symbolically, we can override some
parameters and perform parameter substitution. We see that attempting to
generate a numeric representation of the Operator object failed, because
it did not know how to assign a value to :math:`B` and :math:`J`.
Normally, Operator objects will have a reference to a Parameters
instance (from python-parameters) passed to it in the constructor phase,
for which these parameters can be extracted. This will in most cases be
handled for you by QuantumSystem (see section [sec:QuantumSystem]), but
for completeness there are two keyword arguments you can pass to
Operator instances: parameters, which shold be a reference to an
existing Parameters instance, and basis, which should be a reference to
an existing Basis object or None (see [sec:Basis]). For now, let us
manually add it for demonstration purposes.

>>> from parameters import Parameters >>> p = Parameters() >>>
p(B=2,J=1) < Parameters with 2 definitions > >>> op =
Operator(’B’:[[1,0],[0,-1]], ’J’:[[0,1],[1,0]],parameters=p) >>> op()
array([[ 2.+0.j, 1.+0.j], [ 1.+0.j, -2.+0.j]]) >>> op(J=10,B=1) array([[
1.+0.j, 10.+0.j], [ 10.+0.j, -1.+0.j]])

We see in the above that we can take advantage of temporary parameter
overrides for numeric representations too [note that a parameters
instance is still necessary for this].

In conjunction with functional dependence inherited from
python-parameters this allows for time and/or context dependent
operators.

Operator objects support basic arithmetic: addition, subtraction, and
multiplication using the standard python syntax. The inverse operation
can be performed using the inverse method:

>>> op.inverse()

There is a subclass of the Operator class, OrthogonalOperator, which is
for operators that have orthogonal eigenvectors; in which case the
inverse operation can be greatly simplified.

The Kronecker tensor product can be applied using the tensor method:

>>> op.tensor(other\ :sub:`o`\ p)

To apply an Operator object to a vector, you can either use the standard
inbuilt multiplication operations, or use the slightly more optimised
apply method.

If you are only interested in how a certain variables affect the
operator, then to improve performance you can “collapse” the Operator
down to only include variables which depend upon those variables.

>>> op.collapse(’t’,J=1)

The result of the above command would substitute all variables (with a
parameter override for :math:`J`) that do not depend upon :math:`t` with
their numerical value, and then perform various optimisations to make
further substitutions more efficient. This is used, for example, in the
integrator.

The last set of key methods of the Operator object are the connected and
restrict methods. Operator.connected will return the set of all indicies
(of the basis vectors in which the Operator is represented) that are
connected by non-zero matrix elements, subject to the provided parameter
substitution. Note that this comparison is done with the numerical
values of the parameters.

>>> op = Operator(’B’:[[1,0],[0,-1]], ’J’:[[0,1],[1,0]],parameters=p)
>>> op.connected(0) 0,1 >>> op.connected(0,J=0) 0

The restrict method returns a new Operator object which keeps only the
entries in the old Operator object which correspond to the basis
elements indicated by the indicies.

>>> op = Operator(’B’:[[1,0],[0,-1]], ’J’:[[0,1],[1,0]],parameters=p)
>>> op.restrict(0) <Operator with shape (1, 1)> >>> op.symbolic()
Matrix([[B]])

