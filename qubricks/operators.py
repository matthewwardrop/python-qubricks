from abc import ABCMeta, abstractmethod, abstractproperty

import sympy

import numpy as np
import scipy as sp
import scipy.linalg as spla

from .utility import dot


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
	
	def __init__(self, components, parameters=None, basis=None):
		self.__p = parameters
		self.__shape = None
		self.components = {}
		
		self.__basis = None
		self.set_basis(basis)
		self.__process_components(components)
		self.__optimised = {}
	
	def __process_components(self, components):
		'''
		Import the specified components, verifying that each component has the same shape.
		'''
		# TODO: Add support for second quantised forms
		if type(components) == dict:
			for pam, component in components.items():  # Assume components of type numpy.ndarray or sympy.Matrix
				if self.__shape is None:
					self.__shape = component.shape
				elif component.shape != self.__shape:
					raise ValueError, "Invalid shape."
				self.components[pam] = component
		else:
			if self.__shape is None:
				self.__shape = components.shape
			self.components[None] = components
	
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
	
	def __repr__(self):
		return "<Operator with shape %s>" % (self.shape)
	
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
		written in the specified basis. To trasform basis, use Operator.change_basis .
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
		return self.__new(O)
	
	
	############## Subspace methods ###################################################
	
	def connected(self, *indicies, **params):
		'''
		Operator.connected returns a list of indicies that represents the rows/columns that mix
		with the specified indicies if this operator were to be multiplied by itself. This method
		requires that the Operator object be square.
		'''
		if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
			raise ValueError("Operator not square. Connectedness only works when Operators are square. %s"%self.components)
		
		new = set(indicies)
		
		for key, component in self.components.items():
			component = np.array(component)
			if key is None or not (self.p.is_constant(key, **params) and self.p(key, **params) == 0):
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
			if type(component) == np.ndarray:
				components[pam] = np.reshape(component[np.kron([1]*len(indicies),indicies),np.kron(indicies,[1]*len(indicies))],(len(indicies),len(indicies)))
			else:
				components[pam] = component[tuple(indicies),tuple(indicies)]
			
		return self.__new(components)
	
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
	
	def __new(self, components={}):
		return Operator(components, parameters=self.__p)
	
	def __copy(self):
		return Operator(self.components, parameters=self.__p)
	
	def __add__(self, other):
		O = self.__copy()
		for pam, component in other.components.items():
			O.components[pam] = O.components.get(pam, 0) + component
		return O
		
	def __sub__(self, other):
		O = self.__copy()
		for pam, component in other.components.items():
			O.components[pam] = O.components.get(pam, 0) - component
		return O
	
	def __dot(self, one, two):
		if type(one) == type(two) and type(one) == np.ndarray:
			return np.dot(one, two)
		return one * two
	
	def __mul__(self, other):
		components = {}
		if isinstance(other, Operator):
			for pam, component in self.components.items():
				for pam_other, component_other in other.components.items():
					mpam = pam if pam_other is None else (pam_other if pam is None else '*'.join((pam, pam_other))) 
					# print mpam
					components[mpam] = components.get(mpam, sympy.zeros(self.shape)) + self.__dot(component, component_other)
		elif isinstance(other, (np.ndarray, sympy.MatrixBase)):
			for pam, component in self.components.items():
				components[pam] = self.__dot(component, other)
				
		else:
			raise ValueError("Operator cannot be multiplied by type: %s" % type(other))
		return self.__new(components)

	def __rmul__(self, other):
		components = {}
		if isinstance(other, (np.ndarray, sympy.MatrixBase)):
			for pam, component in self.components.items():
				components[pam] = self.__dot(other, component)
		else:
			raise ValueError("Operator cannot be multiplied from left by type: %s" % type(other))
		return self.__new(components)
	
	def tensor(self, other):
		'''
		Operator.tensor returns a new Operator object, with component-wise tensor (or Kronecker) product;
		and with parameter coefficients updated as required. `other` must also be an Operator instance.
		'''
		components = {}
		
		for pam, component in self.components.items():
			components[pam] = spla.block_diag(self.__np(component), sp.zeros(other.shape))
		
		for pam, component in other.components.items():
			components[pam] = components.get(pam, 0) + spla.block_diag(sp.zeros(self.shape), self.__np(component))
		
		return self.__new(components)
	
	def inverse(self):
		'''
		Computes the inverse of the Operator object. This may be very slow. If you do not need a 
		symbolic inversion, then simply call the Operator object and take a numerical inverse using
		numpy.
		'''
		M = self.symbolic().pinv()
		
		components = {}
		for i in xrange(M.rows):
			for j in xrange(M.cols):
				e = M[i, j]

				if e.is_Number:
					if None not in components:
						components[None] = sympy.zeros(M.shape)
					components[None][i, j] += e
				else:
					from .utility import getFreeSymbolCoefficients
					for coefficient, symbol in getFreeSymbolCoefficients(e):
						key = str(symbol)

						if key not in components:
							components[key] = sympy.zeros(M.shape)

						components[key][i, j] += coefficient
		
		return self.__new(components)


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
		multiple objects.
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
	
	def __init__(self, parameters=None, basis=None, **kwargs):
		self.__p = parameters
		self.__basis = basis
		self.process_args(**kwargs)
	
	@abstractmethod
	def process_args(self, **kwargs):
		'''
		StateOperator.process_args is called when StateOperator subclasses are 
		initialised, which allows subclasses to set themselves up as appropriate.
		'''
		raise NotImplementedError("StateOperator.process_args is not implemented.")
	
	@property
	def p(self):
		'''
		Returns a reference to the internal Parameter object.
		'''
		if self.__p is None:
			raise ValueError("Operator requires use of Parameters object; but none specified.")
		return self.__p

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
		written in the specified basis. To trasform basis, use Operator.change_basis .
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
	
	def process_args(self, **kwargs):
		pass
	
	def transform(self, transform_op):
		return self
	
	def restrict(self, *indicies):
		return self
	
	def connected(self, *indicies, **params):
		return set(indicies)
	
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
	
	def process_args(self, H):
		self.H = H
	
	def __call__(self, state, t=0, params={}):
		H = self.H(t=t, **params)
		if len(state.shape) > 1:
			return 1j / self.p.c_hbar * (np.dot(state, H) - np.dot(H, state))
		return -1j / self.p.c_hbar * np.dot(H, state)
	
	def transform(self, transform_op):
		return SchrodingerOperator(self.p, H=transform_op(self.H))
	
	def restrict(self, *indicies):
		return SchrodingerOperator(self.p, H=self.H.restrict(*indicies))
	
	def connected(self, *indicies, **params):
		return self.H.connected(*indicies, **params)
	
	@property
	def for_state(self):
		return True
	
	@property
	def for_ensemble(self):
		return True

class LindbladOperator(StateOperator):
	'''
	A StateOperator instance that effects a single-termed Lindblad master equation.
	'''
	
	def process_args(self, coefficient, operator):
		self.coefficient = coefficient
		self.operator = operator
	
	def __call__(self, state, t=0, params={}):
		if len(state.shape) == 1:
			raise ValueError, "Lindblad operator can only be used when evolving the density operator."
		
		O = self.operator(t=t, **params)
		Od = O.transpose().conjugate()

		return self.p(self.coefficient, t=t, **params) / self.p.c_hbar ** 2 * (dot(O, state, Od) - 0.5 * (dot(Od, O, state) + dot(state, Od, O)))
	
	def transform(self, transform_op):
		return LindbladOperator(self.p, coefficient=self.coefficient, operator=transform_op(self.operator))
	
	def restrict(self, *indicies):
		return LindbladOperator(self.p, coefficient=self.coefficient, operator=self.operator.restrict(*indicies))
	
	def connected(self, *indicies, **params):
		return self.operator.connected(*indicies, **params)
	
	@property
	def for_state(self):
		return False
	
	@property
	def for_ensemble(self):
		return True	
