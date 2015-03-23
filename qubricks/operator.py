from .utility import getLinearlyIndependentCoeffs
from parameters import Parameters
import numpy as np
import scipy as sp
import scipy.linalg as spla
import sympy
import types


class Operator(object):
	'''
	Operator is the base class used by QuBricks to facilitate dynamic generation
	of matrices (with partial support for n-dimensional operations when `exact` is False).
	Operator objects wrap around a dictionary of "components" which are indexed
	by a function of parameters. When evaluated, Operator objects evaluate the
	parameters using a Parameters instance, and then add the various components
	together. Operator objects support arithmetic (addition, subtraction
	and multiplication); basis transformations; restriction to a subspace of the basis;
	inversion (where appropriate); and the tensor (Kronecker) product.

	:param components: Specification of the operator form
	:type components: dict or numpy.ndarray or sympy.Matrix
	:param parameters: Parameters instance
	:type parameters: parameters.Parameters
	:param basis: The basis in which Operator is represented
	:type basis: Basis or None
	:param exact: :code:`True` if Operator is to maintain an exact representation of numbers.
	:type exact: bool

	Operator Specifications:
		The first and simplest way to construct an Operator object is to wrap an
		Operator around a pre-existing numpy array or sympy Matrix.

		>>> p = Parameters()
		>>> a = numpy.array([[1,2,3],[4,5,6]])
		>>> op = Operator(a, parameters=p)
		>>> x,y,z = sympy.var('x,y,z')
		>>> b = sympy.Matrix([[x,y**2,z+x],[y+z,x**2,x*y*z]])
		>>> op2 = Operator(b, parameters=p)

		The first example above demonstrates wrapping a static numeric matrix
		into an Operator object; while the second demonstrates a conversion of an
		already symbolic operator into an Operator object.

		The other way to define specify the form of an Operator object is to
		create a dictionary with keys of (functions of) parameters and values corresponding to
		the representation of those parameters in the matrix/array. For example:

		>>> d = {'x':[[1,0,0],[0,0,0]], 'sin(y)':[[0,0,0],[0,0,3]], None: [[1,1,1],[2,2,2]]}
		>>> op = Operator(d, parameters=p)

		The above code snippet represents the below matrix:

		::

			[x+1, 1 ,      1     ]
			[ 2 , 2 , 2 + sin(y) ]

	Evaluating an Operator:
		Operator objects can be evaluated to numeric matricies (whereby parameters)
		are substituted in from the Parameters instance, or symbolic sympy.Matrix
		objects.

		Numeric evaluation looks like calling the operator:

		>>> d = {'x':[[1,0,0],[0,0,0]], 'sin(y)':[[0,0,0],[0,0,3]], None: [[1,1,1],[2,2,2]]}
		>>> op = Operator(d, parameters=p)
		>>> op(x=2, y=0)
		array([[ 3.+0.j,  1.+0.j,  1.+0.j],
		       [ 2.+0.j,  2.+0.j,  2.+0.j]])

		.. note:: Providing parameters as shown above makes use of functionality
				  in the Parameters instance, where they are called parameter overrides.
				  Consequently, you can also supply functions of other parameters here.
				  You can also supply united quantities. See the Parameters documentation
				  for more.

		Symbolic evaluation looks like:

		>>> op.symbolic()
		Matrix([
		[x + 1, 1,            1],
		[    2, 2, 3*sin(y) + 2]])

		Parameters can also be specified during symbolic evaluation:

		>>> op.symbolic(y=0)
		Matrix([
		[x + 1, 1, 1],
		[    2, 2, 2]])

		.. note:: Parameter substitution during symbolic evaluation makes use of
				  the `subs` function of sympy objects. It too supports substitution
				  with functions of parameters. See the sympy documentation for more.

		It is also possible to *apply* an Operator to a vector without ever
		explicitly evaluating the Operator. This may lead to faster runtimes.
		See the documentation for the *apply* method.

	Operator arithmetic:
		Operator objects can be added, subtracted and multiplied like any
		other Pythonic numeric object.

		Supported operations are:
		-	Addition: :code:`op1+op2`
		-	Subtraction: :code:`op1-op2`
		-	Multiplication: :code:`op1*op2`
		-	Scaling: :code:`k*op1`, with *k* a scalar constant.

	The rest of the functionality of the Operator object is described in the
	method documentation below.
	'''

	def __init__(self, components, parameters=None, basis=None, exact=False):
		self.__p = parameters
		self.__optimised = {}

		self.__shape = None
		self.components = {}
		self.__exact = exact

		self.basis = basis

		self.__process_components(components)

	@property
	def exact(self):
		"A boolean value indicating whether the Operator should maintain exact representations of numbers."
		return self.__exact
	@exact.setter
	def exact(self, value):
		self.__exact = value

	def __add_component(self, pam, component):  # Assume components of type numpy.ndarray or sympy.Matrix
		if not isinstance(component, (np.ndarray, sympy.MatrixBase)):
			component = np.array(component) if not self.exact else sympy.Matrix(component)
		if self.__shape is None:
			self.__shape = np.array(component).shape
		elif component.shape != self.__shape:
			raise ValueError("Invalid shape.")
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
		elif isinstance(components, (sympy.MatrixBase, sympy.Expr)):
			components = self.__symbolic_components(components)
		elif isinstance(components, (list, np.ndarray)):
			components = self.__array_components(components)
		else:
			raise ValueError("Components of type `%s` could not be understand by qubricks.Operators." % type(components))

		for pam, component in components.items():
			self.__add_component(pam, component)

	def __array_components(self, array):
		# TODO: Check for symbolic nested components?
		components = {}
		components[None] = np.array(array)
		return components

	def __symbolic_components(self, m):
		components = {}
		if isinstance(m, sympy.MatrixBase):
			for i in xrange(m.shape[0]):
				for j in xrange(m.shape[1]):
					e = m[i, j]

					if e.is_Number:
						if None not in components:
							components[None] = sympy.zeros(*m.shape) if self.exact else np.zeros(m.shape, dtype=complex)
						components[None][i, j] += e
					else:
						for coefficient, symbol in getLinearlyIndependentCoeffs(e):
							key = str(symbol)

							if key not in components:
								components[key] = sympy.zeros(*m.shape) if self.exact else np.zeros(m.shape, dtype=complex)

							components[key][i, j] += coefficient
		elif isinstance(m, sympy.Expr):
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
		This method returns a symbolic representation of the Operator object,
		as documented in the general class documentation.

		:param params: A dictionary of parameter overrides.
		:type params: dict

		For example:

		>>> op.symbolic(x=2,y='sin(x)')
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

	def apply(self, state, symbolic=False, left=True, params={}):
		'''
		This method returns the object resulting from multiplication by this Operator
		without ever fully constructing the Operator; making it potentially a little
		faster.

		:param state: An array with suitable dimensions for being pre- (or post-, if left is False) multipled by the Operator represented by this object.
		:type state: numpy.array or object
		:param symbolic: :code:`True` if multiplication should be done symbolically using sympy, and :code:`False` otherwise (uses numpy).
		:type symbolic: bool
		:param left: :code:`True` if the state should be multiplied from the left; and :code:`False` otherwise.
		:type left: bool
		:param params: A dictionary of parameter overrides.
		:type params: dict

		For example:

		>>> op = Operator([[1,2],[3,4]])
		>>> op.apply([1,2])
		array([  5.+0.j,  11.+0.j])
		'''

		if symbolic:
			return self.__assemble(symbolic=symbolic, params=params) * state
		else:
			R = np.zeros(self.__np(state).shape, dtype=complex)
			for pam, component in self.components.items():
				if pam is None:
					R += self.__np(component).dot(self.__np(state)) if left else self.__np(state).dot(self.__np(component))
				else:
					R += self.p(self.__optimise(pam), **params) * (self.__np(component).dot(self.__np(state)) if left else self.__np(state).dot(self.__np(component)))
			return R

	def __repr__(self):
		return "<Operator with shape %s>" % (str(self.shape))

	@property
	def shape(self):
		"A tuple representing the dimensions of the Operator."
		if self.__shape is None:
			return ()
		return self.__shape

	############# Parameters helper methods ################################################
	@property
	def p(self):
		'''
		A reference to the internal Parameter object.
		'''
		if self.__p is None:
			raise ValueError("Operator requires use of Parameters object; but none specified. You can add one using: Operator.p = parametersInstance.")
		if type(self.__p) in (types.FunctionType, types.MethodType):
			return self.__p()
		return self.__p
	@p.setter
	def p(self, parameters):
		if not (isinstance(parameters, Parameters) or isinstance(parameters, (types.FunctionType, types.MethodType)) and (isinstance(parameters(), Parameters) or parameters() is None)):
			raise ValueError("You must provide a Parameters instance or a function with no arguments that returns a Parameters instance.")
		self.__p = parameters
		return self

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
		"A reference to the current Basis of the Operator; or None if it has no specified Basis."
		return self.__basis
	@basis.setter
	def basis(self, basis):
		'''
		Sets the current basis of the Operator object. Note that this does NOT transform the
		Operator into this basis; but simply identifies the current form the Operator as being
		written in the specified basis. To transform basis, use Operator.change_basis .
		'''
		self.__basis = basis

	def change_basis(self, basis=None, threshold=False, params={}):
		'''
		This method returns a copy of this Operator object expressed in the new basis. The threshold
		parameter allows elements in the resulting Operator which differ from zero
		only by numerical noise to be set to zero. For more information, refer
		to the documentation for Basis.transform.

		:param basis: The basis in which to represent this Operator.
		:type basis: Basis or None
		:param threshold: A boolean indicating whether to threshold to minimise numerical error, or a float indicating the threshold level.
		:type threshold: bool or float
		:param params: Parameter overrides to use during the basis transformation.
		:type params: dict

		:returns: A reference to the transformed Operator.
		'''
		if basis == self.__basis:
			return self
		elif basis is None and self.__basis is not None:
			O = self.__basis.transform_to(self, basis=None, threshold=threshold, params=params)
		elif basis is not None and self.__basis is None:
			O = basis.transform_from(self, basis=None, threshold=threshold, params=params)
		else:
			O = basis.transform_from(self, basis=self.__basis, threshold=threshold, params=params)
		O.basis = basis
		return O

	def transform(self, transform_op=None):
		'''
		This method returns a copy of this Operator instance with its components transformed according
		to the supplied transform_op function. This can effect a basis transformation without
		providing any information as to the basis of the new Operator. If you want to transform
		basis, it is probably better you use: Operator.change_basis.

		:param transform_op: A function which maps numpy.ndarray and sympy.MatrixBase instances to themselves, potentially with some transformation.
		:type transform_op: callable
		'''
		O = {}
		if transform_op is not None:
			for pam, component in self.components.items():
				O[pam] = transform_op(component)
		return self._new(O)

	############## Subspace methods ###################################################

	def connected(self, *indices, **params):
		'''
		This method returns a set of indices that represents the rows/columns that mix
		with the specified indices if this operator were to be multiplied by itself. This method
		requires that the Operator object be square.

		:param indices: A sequence of zero-indexed indices to test for connectedness.
		:type indices: tuple
		:param params: A parameter override context.
		:type params: dict

		For example:

		>>> op.connected(1,2,3,x=12,y=(3,'ms'))
		{1,2,3}

		The above output would suggest that in the context of x and y set as indicated,
		the specified subspace is closed under repeated self-multiplication of the Operator.
		'''
		if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
			raise ValueError("Operator not square. Connectedness only works when Operators are square. %s" % self.components)

		new = set(indices)

		for key, component in self.components.items():
			component = np.array(component)
			if key is None or not (not self.exact and self.p.is_resolvable(key, **params) and self.p(key, **params) == 0):
				for index in indices:
					new.update(np.where(np.logical_or(component[:, index] != 0, component[index, :] != 0))[0].tolist())

		if len(new.difference(indices)) != 0:
			new.update(self.connected(*new, **params))

		return new

	def restrict(self, *indices):
		'''
		This method returns a copy of the Operator restricted to the basis elements specified
		as row/column indices. This method requires that the shape of the Operator is square.

		:param indices: A sequence of zero-indexed indices which correspond to the rows/columns to keep.
		:type indices: tuple

		For example:

		>>> op = Operator([[1,2,3],[4,5,6],[7,8,9]]).restrict(1,2)
		>>> op()
		array([[ 5.+0.j,  6.+0.j],
		       [ 8.+0.j,  9.+0.j]])
		'''
		if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
			raise ValueError("Operator not square. Restriction only works when Operators are square.")

		components = {}

		for pam, component in self.components.items():

			if type(component) != np.ndarray:
				new = type(component)(np.zeros( (len(indices), len(indices)) ))
			else:
				new = np.zeros( (len(indices),len(indices)), dtype=component.dtype )

			# Do basis index sweeping to allow for duck-typing
			for i, I in enumerate(indices):
				for j, J in enumerate(indices):
					new[i, j] = component[I, J]

			components[pam] = new

			#if type(component) != np.ndarray:
			#	components[pam] = np.reshape(component[np.kron([1]*len(indices),indices),np.kron(indices,[1]*len(indices))],(len(indices),len(indices)))
			#else:
			#	components[pam] = component[tuple(indices),tuple(indices)]

		return self._new(components)

	############### Other utility methods #############################################

	def __np(self, sympy_matrix):
		'''
		A utility function to convert any component to a numpy array.
		'''
		if type(sympy_matrix) == np.ndarray:
			return sympy_matrix
		if isinstance(sympy_matrix,sympy.MatrixBase):
			return np.array(sympy_matrix.tolist(), dtype=complex)
		if isinstance(sympy_matrix, (tuple,list)):
			return np.array(sympy_matrix, dtype=complex)
		raise ValueError("Unknown conversion from %s to numpy array." % type(sympy_matrix))

	def clean(self, threshold):
		'''
		This method zeroes out all elements of the components which
		are different from zero only by a magnitude less than `threshold`. One must
		use this function with caution, as it does not take into account the value
		of the parameter multiplying the matrix form.

		:param threshold: A threshold value.
		:param type: float
		'''
		for key in self.components:
			ind = np.where(np.abs(self.__np(self.components[key])) < threshold)
			for k in xrange(len(ind[0])):
				self.components[key][(ind[0][k], ind[1][k])] = 0

		return self

	########## Define Basic Arithmetic ################################################

	def _new(self, components={}):
		return Operator(components, parameters=self.__p, basis=self.__basis, exact=self.__exact)

	def _copy(self):
		return Operator(self.components, parameters=self.__p, basis=self.__basis, exact=self.__exact)

	def __get_other_operator(self, other):
		if self.basis == other.basis or other.basis == "*" or self.basis == '*':
			return other
		else:
			return other.change_basis(self.basis)

	def __zero(self, shape=None):
		if shape is None:
			shape = self.shape
		return sympy.zeros(*shape) if self.exact else np.zeros(shape)

	def __add__(self, other):
		O = self._copy()
		if not isinstance(other, Operator):
			other = Operator(other)
		for pam, component in self.__get_other_operator(other).components.items():
			O.components[pam] = O.components.get(pam, self.__zero()) + component
		return O

	def __sub__(self, other):
		O = self._copy()
		if not isinstance(other, Operator):
			other = Operator(other)
		for pam, component in self.__get_other_operator(other).components.items():
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
				for pam_other, component_other in self.__get_other_operator(other).components.items():
					mpam = pam if pam_other is None else (pam_other if pam is None else '*'.join((pam, pam_other)))
					mpam = str(sympy.S(mpam)) if mpam is not None else None
					r = self.__dot(component, component_other)
					if shape is None:
						shape = r.shape
					if mpam not in components:
						if type(r) != np.ndarray or self.exact or other.exact:
							components[mpam] = sympy.zeros(*shape)
					components[mpam] = components.get(mpam, self.__zero(shape)) + r
		elif isinstance(other, (np.ndarray, sympy.MatrixBase)):
			for pam, component in self.components.items():  # TODO: convert symbolic matrix to Operator and do normal multiplication
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
		This method returns a new Operator object that is the component-wise tensor
		(or Kronecker) product of this Operator with *other*.

		:param other: Another Operator with which to perform the tensor product
		:type other: Operator
		'''
		components = {}

		if not isinstance(other, Operator):
			other = Operator(other)

		for pam, component in self.components.items():
			components[pam] = spla.block_diag(self.__np(component), sp.zeros(other.shape))

		for pam, component in self.__get_other_operator(other).components.items():
			components[pam] = components.get(pam, self.__zero(self.shape + component.shape)) + spla.block_diag(sp.zeros(self.shape), self.__np(component))

		return self._new(components)

	def inverse(self):
		'''
		This method computes and returns the inverse of the Operator object. This may be very slow. If you do not need a
		symbolic inversion, then simply numerically evaluate the Operator object and take a numerical inverse using
		numpy.
		'''
		return self._new(self.symbolic().pinv())

	def collapse(self, *wrt, **params):
		'''
		This method returns a new Operator object that is a simplification of this one.
		It collapses and simplifies this Operator object by assuming certain parameters are going
		to be fixed and non-varying. As many parameters as possible are collapsed into the constant component
		of the operator. All other entries are analytically simplified as much as possible.
		If no parameters are specified, then only the simplification is performed. A full collapse
		to a numerical matrix should be achieved by evaluating it numerically, as described in the class description.

		:param wrt: A sequence of parameter names.
		:type wrt: tuple
		:param params: Parameter overrides.
		:type params: dict

		For example:

		>>> op.collapse('x',y=1)

		This will lead to a new Operator being formed which is numeric except for terms involving
		`x`, when `y` is set to `1`.
		'''
		components = {}

		def add_comp(key, contrib):
			if key not in components:
				components[key] = np.zeros(self.shape, dtype='complex')
			components[key] += contrib

		for component, form in self.components.items():
			if component is None:
				add_comp(None, form)
			else:
				for (coeff, indet) in getLinearlyIndependentCoeffs(sympy.S(component)):
					if len(wrt) > 0 and self.p.is_constant(str(indet), *wrt, **params):
						add_comp(None, coeff * self.p(indet, **params) * form)
					else:
						subs = {}
						if len(wrt) > 0:
							for s in indet.free_symbols:
								if self.p.is_constant(str(s), *wrt, **params):
									subs[s] = self.p(str(s), **params)

						coeff2, indet2 = getLinearlyIndependentCoeffs(indet.subs(subs))[0]
						add_comp(str(indet2), coeff * coeff2 * form)

		return self._new(components)

	# Support Indexing
	def __getitem__(self, index):
		components = {}
		for arg, form in self.components.items():
			components[arg] = form[index]
		return self._new(components)


class OrthogonalOperator(Operator):
	'''
	OrthogonalOperator is a subclass of Operator for representing Orthogonal matrices.
	The only difference is that the inversion process is simplified,
	using the result that the inverse of Q is simply its transpose.

	Apart from the inversion operation, all other operations will result in a
	normal Operator object being returned.
	'''

	def inverse(self):  # Orthogonal Operators are orthogonal matrices. Thus Q^-1 = Q^T
		'''
		:returns: An OrthogonalOperator object which is the inverse of this one.

		This method computes the inverse of the Operator object. This may be very slow. If you do not need a
		symbolic inversion, then simply call the Operator object and take a numerical inverse using
		numpy.
		'''
		components = {}

		for key, c in self.components.items():
			components[key] = c.transpose()

		return OrthogonalOperator(components, parameters=self.p, basis=self.basis, exact=self.exact)


class OperatorSet(object):
	'''
	OperatorSet objects a container for multiple Operator objects, such that one
	can construct different combinations of the elements of the OperatorSet at
	runtime. This is useful, for example, if you have a Hamiltonian with various
	different couplings, and you want to consider various combinations of them.

	:param components: A dictionary of Operator objects, indexed by a string representing
		the name of the corresponding Operator object.
	:type components: dict of Operator instances
	:param defaults: A list of component names which should be compiled into an operator when
		a custom list is not supplied. If defaults is None, then all components are
		used.
	:type: list of str or None

	Creating an OperatorSet instance:
		>>> ops = OperatorSet ( components = { 'op1': Operator(...),
		                                       'op2': Operator(...) } )

	Extracting Operator instances:
		Usually, one wants to call OperatorSet objects, with a list of keys to be compiled
		into a single Operator object. e.g. operatorset('name1','name2',...) . For
		example:

		>>> ops('op1') # This is just op1 on its own
		>>> ops('op1','op2') # This is the sum of op1 and op2
		>>> ops() # Same as above, since no defaults provided.

		Individual components can also be accessed using: :code:`operatorset['name']`.
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
		'''
		This method applies the Operator.apply method from each of the stored Operators, and
		sums up their results. The only difference from the Operator.apply method
		is the `components` keyword, which allows you to specify which operators are used. If
		`components` is not specified, then the default components are used.

		:param state: An array with suitable dimensions for being pre- (or post-, if left is False) multipled by the Operator represented by this object.
		:type state: numpy.array or object
		:param symbolic: :code:`True` if multiplication should be done symbolically using sympy, and :code:`False` otherwise (uses numpy).
		:type symbolic: bool
		:param left: :code:`True` if the state should be multiplied from the left; and :code:`False` otherwise.
		:type left: bool
		:param params: A dictionary of parameter overrides.
		:type params: dict
		:param components: A list of components to use.
		:type components: iterable
		'''
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
