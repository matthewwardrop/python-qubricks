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

	Operator indexing:
		Operator objects use `numpy.ndarray` objects to internally represent the
		array; and thus inherits sophisticated indexing. You can index an `Operator`
		object using any indexing method supported by numpy. For example:

		>>> op[1:3,2] # Will return a new `Operator` object sliced with the 2nd and third rows, with the third column

	The rest of the functionality of the Operator object is described in the
	method documentation below.
	'''

	__array_priority__ = 1000  # Cause pre-multiplication by numpy array to call __rmul__ appropriately.

	def __init__(self, components, parameters=None, basis=None, exact=False):
		self.__p = parameters
		self.__optimised = {}

		self.__shape = None
		self.components = {}
		self.__exact = exact

		self.basis = basis

		self.__add_components(components)

	@property
	def exact(self):
		"A boolean value indicating whether the Operator should maintain exact representations of numbers."
		return self.__exact
	@exact.setter
	def exact(self, value):
		self.__exact = value

	def __add_components(self, components):
		'''
		Import the specified components, verifying that each component has the same shape.
		'''
		if type(components) == dict:
			for pam, component in components.items():
				self.__add_component(pam, component)
		else:
			self.__add_component(None, components)

	def __add_component(self, pam, component):  # Assume components of type numpy.ndarray or sympy.Matrix

		if not isinstance(component, np.ndarray):
			component = np.array(component)

		if component.dtype == np.dtype(object):
			try:
				S = np.vectorize(lambda x: complex(x))
				S(component)
			except:
				return self.__add_components(self.__extract_components(component, pam=pam))

		if self.exact:
			S = np.vectorize(lambda x: sympy.S(x))
		else:
			S = np.vectorize(lambda x: complex(x))
		component = S(component)

		if self.__shape is None:
			self.__shape = component.shape
		elif component.shape != self.__shape:
			raise ValueError("Different components of the same Operator object report different shape.")

		if pam in self.components:
			self.components[pam] += component
		else:
			self.components[pam] = component

	def __extract_components(self, array, pam=None):
		components = {}
		if isinstance(array, np.ndarray) and array.dtype == np.dtype(object):
			# Check for symbolic nested components
			it = np.nditer(array, flags=['multi_index', 'refs_ok'])
			while not it.finished:
				e = it[0][()]
				if isinstance(e, sympy.Expr):
					if e.is_number:
						if None not in components:
							components[None] = self.__zero(array.shape)
						components[None][it.multi_index] += e
					else:
						for coefficient, symbol in getLinearlyIndependentCoeffs(e):
							if symbol is sympy.S.One:
								key = None
							else:
								key = str(symbol)
							if key not in components:
								components[key] = self.__zero(array.shape)
							components[key][it.multi_index] += coefficient
				it.iternext()
		else:
			components[None] = np.array(array)
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
		This method returns a symbolic representation of the Operator object as
		a numpy array with a dtype of `object`, as documented in the general
		class documentation.

		:param params: A dictionary of parameter overrides.
		:type params: dict

		For example:

		>>> op.symbolic(x=2,y='sin(x)')
		'''
		return self.__assemble(symbolic=True, params=params)

	def matrix(self, **params):
		'''
		This method returns a symbolic representation of the Operator object as
		a sympy.Matrix object, as documented in the general class documentation.

		:param params: A dictionary of parameter overrides.
		:type params: dict

		For example:

		>>> op.matrix(x=2,y='sin(x)')
		'''
		return sympy.Matrix(self.__assemble(symbolic=True, params=params))

	def __assemble(self, symbolic=False, params={}, apply_state=None, left=True):
		'''
		This utility function does the grunt work of compiling the various components into
		the required form.
		'''
		R = None

		S = None
		if not symbolic and self.exact:
			S = np.vectorize(lambda x: complex(x))
		elif symbolic and not self.exact:
			S = np.vectorize(lambda x: sympy.S(x))

		if symbolic:
			pam_eval = lambda pam: sympy.S(pam).subs(params)
		else:
			pam_eval = lambda pam: self.p(self.__optimise(pam), **params)

		apply_op = None
		if apply_state is not None:
			apply_op = lambda x: x.dot(self.__np(apply_state)) if left else self.__np(apply_state).dot(x)

		for pam, component in self.components.items():
			if S is not None:
				component = S(component)
			if apply_op is not None:
				component = apply_op(component)
			if R is None:
				R = self.__zero(exact=symbolic, shape=component.shape)
			if pam is None:
				R += component
			else:
				R += pam_eval(pam) * component
		return R

	def apply(self, state, symbolic=False, left=True, params={}):
		'''
		This method returns the object resulting from multiplication by this Operator
		without ever fully constructing the Operator; making it potentially a little
		faster. When using `apply` on symbolic arrays, be sure to set `symbolic` to `True`.

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

		return self.__assemble(symbolic=symbolic, apply_state=state, left=left, params=params)

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

		indices = np.array(indices)

		components = {}

		for pam, component in self.components.items():
			components[pam] = component[indices[:,None], indices]

		return self._new(components)

	############### Other utility methods #############################################

	def __np(self, array, exact=None):
		'''
		A utility function to convert any component to a numpy array.
		'''
		exact = exact if exact is not None else self.exact
		dtype = object if exact else complex

		if type(array) == np.ndarray:
			return array.astype(dtype)
		if isinstance(array, sympy.MatrixBase):
			return np.array(array.tolist(), dtype=dtype)
		if isinstance(array, (tuple, list)):
			return np.array(array, dtype=dtype)
		raise ValueError("Unknown conversion from %s to numpy array." % type(array))

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
			ind = np.where(np.abs(self.components[key]) < threshold)
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

	def __zero(self, shape=None, exact=None):
		exact = exact if exact is not None else self.exact
		shape = shape if shape is not None else self.shape
		return np.zeros(shape, dtype=object) if exact else np.zeros(shape, dtype=complex)

	def __add__(self, other):
		O = self._copy()
		if not isinstance(other, Operator):
			other = self._new(other)
		for pam, component in self.__get_other_operator(other).components.items():
			O.components[pam] = O.components.get(pam, self.__zero()) + component
		return O

	def __radd__(self, other):
		return self.__add__(other)

	def __sub__(self, other):
		O = self._copy()
		if not isinstance(other, Operator):
			other = self._new(other)
		for pam, component in self.__get_other_operator(other).components.items():
			O.components[pam] = O.components.get(pam, self.__zero()) - component
		return O

	def __rsub__(self, other):
		other = self._new(other)
		return other - self

	def __dot(self, one, two):
		if type(one) == type(two) and type(one) == np.ndarray:
			return np.dot(one, two)
		return one * two

	def __mul__(self, other):
		components = {}
		if isinstance(other, (np.ndarray, sympy.MatrixBase)):
			other = self._new(other)
		if isinstance(other, Operator):
			for pam, component in self.components.items():
				for pam_other, component_other in self.__get_other_operator(other).components.items():
					pam = pam if pam is not None else 1
					pam_other = pam_other if pam_other is not None else 1
					mpam = str(sympy.S(pam) * sympy.S(pam_other))
					r = self.__dot(component, component_other)
					if mpam not in components:
						components[mpam] = self.__zero(shape=r.shape, exact=self.exact or other.exact)
					components[mpam] += r
		elif isinstance(other, (str, sympy.expr.Expr)):
			for pam, component in self.components.items():
				components[str(sympy.S(other) * sympy.S(pam))] = component
		elif isinstance(other, (int, float, complex, long)):
			for pam, component in self.components.items():
				components[pam] = other * component
		else:
			raise ValueError("Operator cannot be multiplied by type: %s" % type(other))
		return self._new(components)

	def __rmul__(self, other):
		components = {}
		if isinstance(other, (np.ndarray, sympy.MatrixBase)):
			other = self._new(other)
			return other * self
		else:
			return self * other
		return self._new(components)

	def __div__(self, other):
		components = {}
		if isinstance(other, (str, sympy.expr.Expr)):
			for pam, component in self.components.items():
				components[str(sympy.S(pam)/sympy.S(other))] = component
		elif isinstance(other, (int, float, complex, long)):
			for pam, component in self.components.items():
				components[pam] = component/other
		return self._new(components)

	# TODO: Implement tensor products
	# def tensor(self, other):
	# 	'''
	# 	This method returns a new Operator object that is the component-wise tensor
	# 	(or Kronecker) product of this Operator with *other*.
	#
	# 	:param other: Another Operator with which to perform the tensor product
	# 	:type other: Operator
	# 	'''
	# 	NotImplementedError("Tensor product calculations have yet to be implemented.")

	def block_diag(self, other):
		'''
		This method returns a new Operator object with the `other` Operator appended
		as a diagonal block below this Operator.

		:param other: Another Operator to add as a diagonal block.
		:type other: Operator
		'''
		components = {}

		if not isinstance(other, Operator):
			other = self._new(other)

		for pam, component in self.components.items():
			components[pam] = spla.block_diag(component, self.__zero(other.shape))

		for pam, component in self.__get_other_operator(other).components.items():
			if pam not in components:
				components[pam] = self.__zero(np.array(self.shape) + np.array(component.shape))
			components[pam] += spla.block_diag(sp.zeros(self.shape), component)

		return self._new(components)

	def inverse(self):
		'''
		This method computes and returns the pseudo-inverse of the Operator object. This may be very slow. If you do not need a
		symbolic inversion, then simply numerically evaluate the Operator object and take a numerical inverse using
		numpy.

		.. note:: The pseudo-inverse will equal the normal inverse if it exists.
		'''
		if len(self.shape) != 2:
			raise ValueError("Cannot perform inverse on Operators with dimension other than 2.")
		return self._new(self.matrix().pinv())

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
				components[key] = self.__zero()
			components[key] += contrib

		for pam, component in self.components.items():
			if pam is None:
				add_comp(None, component)
			else:
				new_pam = 0
				for (coeff, indet) in getLinearlyIndependentCoeffs(sympy.S(pam)):
					if len(wrt) > 0 and self.p.is_constant(str(indet), *wrt, **params):
						add_comp(None, coeff * self.p(indet, **params) * component)
					else:
						subs = params.copy()
						if len(wrt) > 0:
							for s in indet.free_symbols:
								if self.p.is_constant(str(s), *wrt, **params):
									subs[s] = self.p(str(s), **params)

						coeff2, indet2 = getLinearlyIndependentCoeffs(indet.subs(subs))[0]
						if isinstance(indet2,sympy.Number):
							add_comp(None, coeff*coeff2*component)
						else:
							new_pam += coeff*coeff2*indet2
				if new_pam != 0:
					add_comp(str(new_pam.simplify()), component)

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
	:type components: dict of Operator instances or None
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

	Subclassing OperatorSet:
		To subclass `OperatorSet` simply override the `init` method to construct whichever
		`Operators` you desire. You should initialise `OperatorSet.components` to be a dictionary;
		but otherwise, there are no constraints. Note that you can add elements to an OperatorSet
		without subclassing it.
	'''

	def __init__(self, components=None, defaults=None, **kwargs):
		self.components = components
		self.defaults = defaults
		if self.components is None:
			self.components = {}
		self.init(**kwargs)

	def init(self):
		'''
		Subclasses may use this hook to initialise themselves. It is called after
		`OperatorSet.components` and `OperatorSet.defaults` are set to their
		passed values, with `Operator.components` guaranteed to be a dictionary.
		'''
		pass

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
			cs.append(self[component])
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
			rs.append(self[component].apply(state, symbolic=symbolic, left=left, params=params))
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
		try:
			return self.components[key]
		except:
			raise ValueError("Invalid operator component: '%s'" % key)

	def __setitem__(self, key, value):
		if not isinstance(value, Operator):
			try:
				value = Operator(value)
			except:
				raise ValueError("Set value is not an Operator instance, nor can it be transformed into one. Received: %s" % value)
		self.components[key] = value
