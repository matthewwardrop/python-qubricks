from abc import ABCMeta, abstractmethod, abstractproperty

import sympy

import numpy as np
import scipy as sp
import scipy.linalg as spla

from .utility import dot
from .utility import getLinearlyIndependentCoeffs


# TODO: Properly remember basis information when doing arithmetic
class Operator(object):
	'''
	Operator(components, parameters=None, basis=None)

	Operator is the base class used by Qubricks to enable dynamic generation
	of vectors and matrices (with partial support for n-dimensional operations).
	Operator objects wrap around a dictionary of "components" which are indexed
	by a function of parameters. When evaluated, Operator objects evaluate the
	parameters using a Parameters instance, and then add the various components
	together.

	To make themselves more useful, Operator objects support basic arithmetic,
	such as addition, subtraction and multiplication; support for basis
	transformations, restriction to a subspace of the basis, inversion and the
	tensor (Kronecker) product.

	Parameters
	----------
	components : Either a sympy matrix or numpy array; or a dictionary of multiple
		such objects with strings representing symbolic operations of parameters in
		the Parameters instance specified with the `parameters` argument. Constant
		components should have key `None`.
	parameters : A Parameters instance, which can be shared between multiple objects.
	basis : A Basis instance that represents the Basis that the Operator is acting
		within.

	For documentation on the methods of the Operator object, please see inline.
	'''

	def __init__(self, components, parameters=None, basis=None, exact=False):
		self.__p = parameters
		self.__shape = None
		self.components = {}
		self.__exact = exact

		self.__basis = None
		self.set_basis(basis)
		self.__process_components(components)
		self.__optimised = {}

	@property
	def exact(self):
		return self.__exact
	@exact.setter
	def exact(self, value):
		self.__exact = value

	def on_attach(self,parameters):
		self.__p = parameters
		return self

	def __add_component(self,pam,component):  # Assume components of type numpy.ndarray or sympy.Matrix
		if not isinstance(component,(np.ndarray,sympy.MatrixBase)):
			component = np.array(component) if self.exact else sympy.Matrix(component)
		if self.__shape is None:
			self.__shape = np.array(component).shape
		elif component.shape != self.__shape:
			raise ValueError, "Invalid shape."
		if pam in self.components:
			self.components[pam] += component
		else:
			self.components[pam] = component

	def __process_components(self, components):
		'''
		Import the specified components, verifying that each component has the same shape.
		'''
		# TODO: Add support for second quantised forms
		if type(components) == dict:
			pass
		elif isinstance(components,(sympy.MatrixBase,sympy.Expr)):
			components = self.__symbolic_components(components)
		elif isinstance(components,(list,np.ndarray)):
			components = self.__array_components(components)
		else:
			raise ValueError("Components of type `%s` could not be understand by qubricks.Operators." % type(components))

		for pam, component in components.items():
			self.__add_component(pam, component)

	def __array_components(self,array):
		# TODO: Check for symbolic nested components?
		components = {}
		components[None] = np.array(array)
		return components

	def __symbolic_components(self,m):
		components = {}
		if isinstance(m,sympy.MatrixBase):
			for i in xrange(m.shape[0]):
				for j in xrange(m.shape[1]):
					e = m[i, j]

					if e.is_Number:
						if None not in components:
							components[None] = sympy.zeros(m.shape) if self.exact else np.zeros(m.shape,dtype=complex)
						components[None][i, j] += e
					else:
						for coefficient, symbol in getLinearlyIndependentCoeffs(e):
							key = str(symbol)

							if key not in components:
								components[key] = sympy.zeros(m.shape) if self.exact else np.zeros(m.shape,dtype=complex)

							components[key][i, j] += coefficient
		elif isinstance(m,sympy.Expr):
			components[m] = np.array([1])
		else:
			raise ValueError("Components of type `%s` could not be understand by qubricks.Operators." % type(m))
		return components

	def __call__(self, **params):
		'''
		Calling an Operator instance returns the numpy.array numeric representation
		of the operator in the current basis evaluated with the parameters specified.
		If a particular parameter is not specified in `params`, the current value in
		in the Parameters instance is used in its stead. Parameter values can be
		specified in any format that Parameters supports.

		e.g. operator(pam1=2,pam3=('2','mV')), etc.
		'''
		return self.__assemble(params=params)  # Hamiltonian evaluated with parameters as above

	def symbolic(self, **params):
		'''
		Operator.symbolic returns a sympy.Matrix symbolic representation of the Operator
		in the current basis, with parameters substituted according to `params` if provided.
		Parameters can be provided in any format that Parameters objects support.
		'''
		return self.__assemble(symbolic=True, params=params)

	def __assemble(self, symbolic=False, params=None):
		'''
		This utility function does the grunt work of compiling the various components into
		the required form.
		'''
		if symbolic:
			R = sympy.zeros(*self.shape)  # np.zeros(self.shape,dtype=object)
			for pam, component in self.components.items():
				if pam is None:
					R += component
				else:
					R += sympy.S(pam) * component
			return R.subs(params)
		else:
			R = np.zeros(self.shape, dtype=complex)
			for pam, component in self.components.items():
				if pam is None:
					R += self.__np(component)
				else:
					R += self.p(self.__optimise(pam), **params) * self.__np(component)
			return R

	def apply(self, state, symbolic=False, left=True, params=None):
		if symbolic:
			return self.__assemble(symbolic=symbolic, params=params) * state
		else:
			R = np.zeros(state.shape, dtype=complex)
			for pam, component in self.components.items():
				if pam is None:
					R += self.__np(component).dot(self.__np(state)) if left else self.__np(state).dot(self.__np(component))
				else:
					R += self.p(self.__optimise(pam), **params) * ( self.__np(component).dot(self.__np(state)) if left else self.__np(state).dot(self.__np(component)) )
			return R

	def __repr__(self):
		return "<Operator with shape %s>" % (str(self.shape))

	@property
	def shape(self):
		'''
		Operator.shape returns the shape of the operator.
		'''
		if self.__shape is None:
			return ()
		return self.__shape

	############# Parameters helper methods ################################################
	@property
	def p(self):
		'''
		Returns a reference to the internal Parameter object.
		'''
		if self.__p is None:
			raise ValueError("Operator requires use of Parameters object; but none specified.")
		return self.__p

	def __optimise(self, pam):
		'''
		This helper generates a local cache of the optimised function returned by the Parameters
		instance.
		'''
		if pam not in self.__optimised:
			self.__optimised[pam] = self.p.optimise(pam)
		return self.__optimised[pam]


	############# Basis helper methods #######################################################

	@property
	def basis(self):
		'''
		Returns a reference to the current Basis of the Operator; or None if it has no specified
		Basis.
		'''
		return self.__basis

	def set_basis(self, basis):
		'''
		Sets the current basis of the Operator object. Note that this does NOT transform the
		Operator into this basis; but simply identifies the current form the Operator as being
		written in the specified basis. To transform basis, use Operator.change_basis .
		'''
		self.__basis = basis

	def change_basis(self, basis=None, threshold=False):
		'''
		Returns a copy of this Operator object expressed in the new basis. The behaviour of the
		threshold parameter is described in the Basis.transform documentation; but this allows
		elements which differ from zero only by numerical noise to be set to zero.
		'''
		if basis == self.__basis:
			return self
		elif basis is None and self.__basis is not None:
			O = basis.transform_to(self, basis=None, threshold=threshold)
		elif basis is not None and self.__basis is None:
			O = basis.transform_from(self, basis=None, threshold=threshold)
		else:
			O = basis.transform_from(self, basis=self.__basis, threshold=threshold)
		O.set_basis(basis)
		return O

	def transform(self, transform_op=None):
		'''
		Returns a copy of this Operator instance with its components transformed according
		to the supplied transform_op function. This can effect a basis transformation without
		providing any information as to the basis of the new Operator.
		'''
		O = {}
		if transform_op is not None:
			for pam, component in self.components.items():
				O[pam] = transform_op(component)
		return self._new(O)


	############## Subspace methods ###################################################

	def connected(self, *indicies, **params):
		'''
		Operator.connected returns a list of indicies that represents the rows/columns that mix
		with the specified indicies if this operator were to be multiplied by itself. This method
		requires that the Operator object be square. If Operator.exact is True, then this will
		return connectedness based upon symbolic forms; otherwise, connectedness will be
		reported based upon the numerical values provided.
		'''
		if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
			raise ValueError("Operator not square. Connectedness only works when Operators are square. %s"%self.components)

		new = set(indicies)

		for key, component in self.components.items():
			component = np.array(component)
			if key is None or not (not self.exact and self.p.is_resolvable(key, **params) and self.p(key, **params) == 0):
				for index in indicies:
					new.update(np.where(np.logical_or(component[:, index] != 0 , component[index, :] != 0))[0].tolist())

		if len(new.difference(indicies)) != 0:
			new.update(self.connected(*new, **params))

		return new

	def restrict(self, *indicies):
		'''
		Operator.connected returns a copy of the Operator restricted to the basis elements specified
		as row/column indicies. This method requires that the shape of the Operator is square.
		'''
		if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
			raise ValueError("Operator not square. Restriction only works when Operators are square.")

		components = {}


		for pam, component in self.components.items():

			if type(component) != np.ndarray:
				new = type(component)(np.zeros( (len(indicies),len(indicies)) ))
			else:
				new = np.zeros( (len(indicies),len(indicies)) , dtype=component.dtype)

			# Do basis index sweeping to allow for duck-typing
			for i,I in enumerate(indicies):
				for j,J in enumerate(indicies):
					new[i,j] = component[I,J]

			components[pam] = new

			#if type(component) != np.ndarray:
			#	components[pam] = np.reshape(component[np.kron([1]*len(indicies),indicies),np.kron(indicies,[1]*len(indicies))],(len(indicies),len(indicies)))
			#else:
			#	components[pam] = component[tuple(indicies),tuple(indicies)]

		return self._new(components)

	############### Other utility methods #############################################

	def __np(self, sympy_matrix):
		'''
		A utility function to convert any component to a numpy array.
		'''
		if type(sympy_matrix) == np.ndarray:
			return sympy_matrix
		return np.array(sympy_matrix.tolist(), dtype=complex)

	def clean(self, threshold):
		'''
		Operator.clean allows one to set to zero all elements of the components which
		are different from zero only by a magnitude less than `threshold`. One must
		use this function with caution.
		'''
		for key in self.components:
			ind = np.where(np.abs(self.__np(self.components[key])) < threshold)
			for k in xrange(len(ind[0])):
				self.components[key][(ind[0][k], ind[1][k])] = 0

		return self

	########## Define Basic Arithmetic ################################################

	def _new(self, components={}):
		return Operator(components, parameters=self.__p,basis=self.__basis,exact=self.__exact)

	def _copy(self):
		return Operator(self.components, parameters=self.__p,basis=self.__basis,exact=self.__exact)

	def __zero(self,shape=None):
		if shape is None:
			shape = self.shape
		return sympy.zeros(shape) if self.exact else np.zeros(shape)

	def __add__(self, other):
		O = self._copy()
		for pam, component in other.components.items():
			O.components[pam] = O.components.get(pam, self.__zero()) + component
		return O

	def __sub__(self, other):
		O = self._copy()
		for pam, component in other.components.items():
			O.components[pam] = O.components.get(pam, self.__zero()) - component
		return O

	def __dot(self, one, two):
		if type(one) == type(two) and type(one) == np.ndarray:
			return np.dot(one, two)
		return one * two

	def __mul__(self, other):
		components = {}
		shape = None
		if isinstance(other, Operator):
			for pam, component in self.components.items():
				for pam_other, component_other in other.components.items():
					mpam = pam if pam_other is None else (pam_other if pam is None else '*'.join((pam, pam_other)))
					mpam = str(sympy.S(mpam)) if mpam is not None else None
					r = self.__dot(component, component_other)
					if shape is None:
						shape = r.shape
					if mpam not in components:
						if type(r) != np.ndarray or self.exact or other.exact:
							components[mpam] = sympy.zeros(shape)
					components[mpam] = components.get(mpam,self.__zero(shape)) + r
		elif isinstance(other, (np.ndarray, sympy.MatrixBase)):
			for pam, component in self.components.items(): # TODO: convert symbolic matrix to Operator and do normal multiplication
				components[pam] = self.__dot(component, other)

		else:
			raise ValueError("Operator cannot be multiplied by type: %s" % type(other))
		return self._new(components)

	def __rmul__(self, other):
		components = {}
		if isinstance(other, (np.ndarray, sympy.MatrixBase)):
			for pam, component in self.components.items():
				components[pam] = self.__dot(other, component)
		else:
			raise ValueError("Operator cannot be multiplied from left by type: %s" % type(other))
		return self._new(components)

	def tensor(self, other):
		'''
		Operator.tensor returns a new Operator object, with component-wise tensor (or Kronecker) product;
		and with parameter coefficients updated as required. `other` must also be an Operator instance.
		'''
		components = {}

		for pam, component in self.components.items():
			components[pam] = spla.block_diag(self.__np(component), sp.zeros(other.shape))

		for pam, component in other.components.items():
			components[pam] = components.get(pam, self.__zero(self.shape + component.shape)) + spla.block_diag(sp.zeros(self.shape), self.__np(component))

		return self._new(components)

	def inverse(self):
		'''
		Computes the inverse of the Operator object. This may be very slow. If you do not need a
		symbolic inversion, then simply call the Operator object and take a numerical inverse using
		numpy.
		'''
		return self._new(self.symbolic().pinv())


	def collapse(self,*wrt,**params):
		'''
		Collapses and simplifies an Operator object on the basis that certain parameters are going
		to be fixed and non-varying. As many parameters as possible are collapsed into the constant component
		of the operator. All other entries are simplified as much as possible, and then returned in a new Operator
		object. If no parameters are specified, then only the simplification is performed. Note that to
		collapse all components into a numerical form, simply call this operator.
		'''
		components = {}

		def add_comp(key,contrib):
			if key not in components:
				components[key] = np.zeros(self.shape,dtype='complex')
			components[key] += contrib

		for component,form in self.components.items():
			if component is None:
				add_comp(None,form)
			else:
				for (coeff,indet) in getLinearlyIndependentCoeffs(sympy.S(component)):
					if len(wrt) > 0 and self.p.is_constant(str(indet),*wrt,**params):
						add_comp(None,coeff*self.p(indet,**params)*form)
					else:
						subs = {}
						if len(wrt) > 0:
							for s in indet.free_symbols:
								if self.p.is_constant(str(s),*wrt,**params):
									subs[s] = self.p(str(s),**params)

						coeff2,indet2 = getLinearlyIndependentCoeffs(indet.subs(subs))[0]
						add_comp(str(indet2),coeff*coeff2*form)

		return self._new(components)

	# Support Indexing
	def __getitem__(self,index):
		components = {}
		for arg,form in self.components.items():
			components[arg] = form[index]
		return self._new(components)

class OrthogonalOperator(Operator):

	def inverse(self): # Orthogonal Operators are orthogonal matrices. Thus Q^-1 = Q^T
		components = {}

		for key, c in self.components.items():
			components[key] = c.transpose()

		return self._new(components)


class OperatorSet(object):
	'''
	OperatorSet(components, defaults=None)

	OperatorSet objects a container for multiple Operator objects, such that one
	can construct different combinations of the elements of the OperatorSet at
	runtime. This is useful, for example, if you have a Hamiltonian with various
	different couplings, and you want to consider each in turn.

	Parameters
	----------
	components : A dictionary of Operator objects, indexed by a string representing
		the name of the corresponding Operator object.
	defaults: A list of component names which should be compiled into an operator when
		a custom list is not supplied. If defaults is None, then all components are
		used.

	Usually, one wants to call OperatorSet objects, with a list of keys to be compiled
	into a single Operator object. e.g. operatorset('name1','name2',...) . Individual
	components can also be accessed using: operatorset['name'].
	'''

	def __init__(self, components, defaults=None):
		self.components = components
		self.defaults = defaults

	def __call__(self, *components):
		'''
		Calling the OperatorSet object returns an Operator object which is the sum of
		all Operator objects referred to in the `components` list. If no components are
		supplied, the OperatorSet.defaults property is used instead.

		e.g. operatorset('object_name_1','object_name_2')
		'''
		return self.__assemble(components)

	def __assemble(self, components):
		'''
		This internal method does sums the relevant components into an Operator object,
		which it then returns.
		'''
		if len(components) == 0:
			if self.defaults is not None:
				components = self.defaults
			else:
				components = self.components.keys()

		cs = []

		if len(components) == 0:
			raise ValueError("Attempted to construct an empty Operator.")
		for component in components:
			if component not in self.components:
				raise ValueError("Invalid operator component: '%s'" % component)
			cs.append(self.components[component])
		return self.__sum(cs)

	def apply(self, state, symbolic=False, left=True, params=None, components=None):

		if components is None or len(components) == 0:
			if self.defaults is not None:
				components = self.defaults
			else:
				components = self.components.keys()

		rs = []
		if len(components) == 0:
			raise ValueError("Attempted to apply an empty Operator.")
		for component in components:
			if component not in self.components:
				raise ValueError("Invalid operator component: '%s'" % component)
			rs.append(self.components[component].apply(state, symbolic=symbolic, left=left, params=params))
		return self.__sum(rs)


	def __sum(self, operators):
		'''
		A utility function to sum operators.
		'''
		s = None
		for operator in operators:
			if s is None:
				s = operator
			else:
				s += operator
		return s

	def __getitem__(self, key):
		return self.components[key]

	def __setitem__(self, key, value):
		if isinstance(value, Operator):
			self.components[key] = value
		else:
			raise ValueError("Attempted to add non-Operator object to OperatorSet.")


################## State Operators #################################################

class StateOperator(object):
	'''
	StateOperator(parameters,*args,**kwargs)

	StateOperator objects are not to be confused with Operator objects. StateOperator
	objects can encode arbitrary operations on states; including those which cannot
	be described by a linear map. Often, but not always, the internal mechanisms of
	a StateOperator will *use* an Operator object. The StateOperator object itself
	is an abstract class, and must be subclassed before it is used. In order to
	conform to the expectations of the QuBricks framework, StateOperators should
	implement basis transformations and subspace restrictions.

	Parameters
	----------
	parameters : A referenced to a Parameters instance, which can be shared among
		multiple objects. This is overridden when adding to a QuantumSystem instance,
		but is helpful for testing purposes.
	args,kwargs : Are passed onto the StateOperator.process_args method, which must
		be implemented by subclasses.

	StateOperator objects are used by calling them directly on states, and passing
	whichever addition parameters are desired.

	e.g. stateoperator(state,t=0,param1=2,param2=1,...)

	As time usually plays time is treated separately from other parameters, though it
	most usually passed directly to the Parameters instance anyway. Parameters may
	be supplied in any format that is supported by the Parameters module.
	'''
	__metaclass__ = ABCMeta

	def __init__(self,  parameters=None, basis=None, **kwargs):
		self.__p = parameters
		self.__basis = basis
		self.init(**kwargs)

	def _on_attach_to_system(self,system):
		self.p = system.p
		self.on_attach_to_system(system)
		return self

	@abstractmethod
	def on_attach_to_system(self, system):
		'''
		StateOperator.on_attach_to_system should ensure that all sub elements
		of the State Operator use the parent parameters object; and whatever
		else needs to be shared.
		'''
		pass

	@abstractmethod
	def init(self, **kwargs):
		'''
		StateOperator.init is called when StateOperator subclasses are
		initialised, which allows subclasses to set themselves up as appropriate.
		'''
		raise NotImplementedError("StateOperator.init is not implemented.")

	@property
	def p(self):
		'''
		Returns a reference to the internal Parameter object.
		'''
		if self.__p is None:
			raise ValueError("Operator requires use of Parameters object; but none specified.")
		return self.__p
	@p.setter
	def p(self,value):
		self.__p = value

	@abstractmethod
	def __call__(self, state, t=0, params={}):
		'''
		StateOperator objects are called on states to effect some desired operation.
		States may be 1-dimensional (as state vectors) or 2-dimensional (as quantum ensembles),
		and each subclass should specify in StateOperator.for_state and StateOperator.for_ensemble
		which kinds of states are supported.
		'''
		raise NotImplementedError("StateOperator.__call__ is not implemented.")

	@abstractmethod
	def restrict(self, *indicies):
		'''
		StateOperator.restrict should return a new StateOperator restricted to the basis states
		with indicies `indicies`.
		'''
		raise NotImplementedError("StateOperator.restrict is not implemented.")

	@abstractmethod
	def connected(self, *indicies, **params):
		'''
		StateOperator.connected should return the list of basis state indicies which would mix
		with the specified basis state indicies `indicies` under repeated operation of the
		StateOperator.
		'''
		raise NotImplementedError("StateOperator.connected is not implemented.")

	def collapse(self,*wrt,**params):
		'''
		Collapses and simplifies a StateOperator object on the basis that certain parameters are going
		to be fixed and non-varying. As many parameters as possible are collapsed into the constant component
		of the operator. All other entries are simplified as much as possible, and then returned in a new Operator
		object. This method is not required, as it is used only for optimisation purposes.
		'''
		return self

	@abstractproperty
	def for_state(self):
		'''
		StateOperator.for_state should return True if the StateOperator supports 1D vector
		operations; and False otherwise.
		'''
		raise NotImplementedError("StateOperator.for_state is not implemented.")

	@abstractproperty
	def for_ensemble(self):
		'''
		StateOperator.for_ensemble should return True if the StateOperator supports 2D ensemble
		operations; and False otherwise.
		'''
		raise NotImplementedError("StateOperator.process_args is not implemented.")

	############### Basis Transformations ###########################################

	@abstractmethod
	def transform(self, transform_op):
		'''
		StateOperator.transform should transform all operations on the state
		according to the basis transformation `transform_op`.
		'''
		raise NotImplementedError("StateOperator.transform is not implemented.")

	@property
	def basis(self):
		'''
		Returns a reference to the current Basis of the Operator; or None if it has no specified
		Basis.
		'''
		return self.__basis

	def set_basis(self, basis):
		'''
		Sets the current basis of the Operator object. Note that this does NOT transform the
		Operator into this basis; but simply identifies the current form the Operator as being
		written in the specified basis. To transform basis, use Operator.change_basis .
		'''
		self.__basis = basis

	def change_basis(self, basis=None, threshold=False):
		'''
		Returns a copy of this StateOperator object expressed in the new basis. The behaviour of the
		threshold parameter is described in the Basis.transform documentation; but this allows
		elements which differ from zero only by numerical noise to be set to zero.
		'''
		if basis == self.__basis:
			return self
		elif basis is None:
			O = self.transform(self.__basis.transform_op(threshold=threshold, invert=True))
		else:
			O = self.transform(basis.transform_op(basis=self.__basis, threshold=threshold))
		O.set_basis(basis)

		return O


###################### Example and useful StateOperators ###########################################

class DummyOperator(StateOperator):
	'''
	An StateOperator instance that does nothing to the state, but which
	forces the integrator to work as if it were necessary to evolve ensemble
	states.
	'''

	def __call__(self, state, t=0, params={}):
		return state

	def init(self, **kwargs):
		pass

	def on_attach_to_system(self, system):
		pass

	def transform(self, transform_op):
		return self

	def restrict(self, *indicies):
		return self

	def connected(self, *indicies, **params):
		return set(indicies)

	def collapse(self, *wrt, **params):
		return self

	@property
	def for_state(self):
		return False

	@property
	def for_ensemble(self):
		return True

class SchrodingerOperator(StateOperator):
	'''
	A StateOperator instance that effects Schroedinger evolution of the
	(quantum) state.
	'''

	def init(self, H):
		self.H = H

	def on_attach_to_system(self, system):
		self.H.on_attach(parameters=system.p)

	def __call__(self, state, t=0, params={}):
		pams = {'t':t}
		pams.update(params)
		if len(state.shape) > 1:
			H = self.H(t=t, **params)
			return 1j / self.p.c_hbar * (np.dot(state, H) - np.dot(H, state))
		return -1j / self.p.c_hbar * self.H.apply(state,params=pams,left=True,symbolic=False) # This may provide a speedup over np.dot(H, state)

	def transform(self, transform_op):
		return SchrodingerOperator(self.p, H=transform_op(self.H))

	def restrict(self, *indicies):
		return SchrodingerOperator(self.p, H=self.H.restrict(*indicies))

	def connected(self, *indicies, **params):
		return self.H.connected(*indicies, **params)

	def collapse(self, *wrt, **params):
		return SchrodingerOperator(self.p, H=self.H.collapse(*wrt,**params))

	@property
	def for_state(self):
		return True

	@property
	def for_ensemble(self):
		return True

class LindbladOperator(StateOperator):
	'''
	A StateOperator instance that effects a single-termed Lindblad master equation. This will cause decay in a simple
	two level system proportional to: exp(-8*coefficient*t)
	'''

	def init(self, coefficient, operator):
		self.coefficient = coefficient
		self.operator = operator

	def on_attach_to_system(self, system):
		if not isinstance(self.operator, Operator):
			self.operator = Operator(self.operator,parameters=system.p)
		else:
			self.operator.on_attach(parameters=system.p)

	def __call__(self, state, t=0, params={}):
		O = self.operator(t=t, **params)
		Od = O.transpose().conjugate()

		return self.p(self.coefficient, t=t, **params)/self.p.c_hbar**2 * (dot(O, state, Od) - 0.5 * (dot(Od, O, state) + dot(state, Od, O)))

	def transform(self, transform_op):
		return LindbladOperator(self.p, coefficient=self.coefficient, operator=transform_op(self.operator))

	def restrict(self, *indicies):
		return LindbladOperator(self.p, coefficient=self.coefficient, operator=self.operator.restrict(*indicies))

	def connected(self, *indicies, **params):
		return self.operator.connected(*indicies, **params)

	def collapse(self, *wrt, **params):
		return LindbladOperator(self.p, coefficient=self.coefficient, operator=self.operator.collapse(*wrt,**params))

	@property
	def for_state(self):
		return False

	@property
	def for_ensemble(self):
		return True
