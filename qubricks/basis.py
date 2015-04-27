from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

import numpy as np
import sympy
import sympy.physics.quantum as sq

from parampy import Parameters

from .operator import Operator


class Basis(object):
	'''
	A Basis instance describes a particular basis, and allows transformations
	of objects (such as `Operator`s) from one basis to another. A Basis 
	is an abstract class, and must be subclassed to be useful.
	
	:param dim: The dimension of the basis. If not specified, the dimension will
		be extracted from the Operator returned by Basis.operator; except during 
		`Basis.init`, where `Basis.dim` will return the raw value stored (e.g. None).
	:type dim: int or None
	:param parameters: A Parameters instance, if required.
	:type parameters: parampy.Parameters
	:param kwargs: Additional keyword arguments to pass to `Basis.init`.
	:type kwargs: dict

	Subclassing Basis:
		Subclasses of Basis must implement the following methods, which function according to 
		their respective documentation below:
		- init
		- operator
		Subclasses may optionally implement:
		- state_info
		- state_toString
		- state_fromString
		- state_latex
		These latter methods are used to allow convenient conversion of strings to states
		and also later representation of states as strings/LaTeX. Otherwise, these
		methods are not required. Since they are not used except when the user desires
		to change the state's representation, the implementer has a lot of freedom
		about the way these functions work, and what they return. The documentation
		for these methods indicates the way in which the original author intended
		for them to function.
	'''
	__metaclass__ = ABCMeta

	def __init__(self, dim=None, parameters=None, **kwargs):
		self.dim = dim
		self.p = parameters
		
		self._basis_initialising = True
		self.init(**kwargs)
		del self._basis_initialising

	@abstractmethod
	def init(self, **kwargs):
		'''
		This method should do whatever is necessary to prepare the
		Basis instance for use. When this method is called by the 
		Python __init__ method, you can use `Basis.dim` to access
		the raw value of `dim`. If `dim` is necessary to construct 
		the operator, and it is not set, this method should raise an
		exception. All keyword arguments except `dim` and `parameters`
		passed to the `Basis` instance constructor will also be passed to 
		this method. 
		'''
		pass

	def __repr__(self):
		return "<%s(Basis) of dimension %d>" % (self.__class__.__name__, self.dim)

	@property
	def dim(self):
		'''
		The dimension of the basis; or equivalently, the number of basis states.
		'''
		if self.__dim == None and not getattr(self,'_basis_initialising',False):
			self.__dim = self.operator().shape[0]
		return self.__dim
	@dim.setter
	def dim(self, dim):
		try:
			if self.__dim is not None:
				raise ValueError("Attempt to change Basis dimension after initialisation. This is not supported.")
		except AttributeError:
			pass
		self.__dim = int(dim) if dim is not None else None

	@property
	def p(self):
		'''
		A reference to the Parameters instance used by this object.
		'''
		if self.__p is None:
			raise ValueError("Parameters instance required by Basis object, a Parameters object has not been configured.")
		return self.__p
	@p.setter
	def p(self, parameters):
		if parameters is not None and not isinstance(parameters, Parameters):
			raise ValueError("Parameters reference must be an instance of Parameters or None.")
		self.__p = parameters

	def __p_ref(self):
		'''
		This method allows child Operator objects to automatically be kept up to date with
		changes to the Parameters instance associated with the StateOperator object. This
		cannot be used directly, but is used by the StateOperator.Operator method.
		'''
		return self.__p
	
	def Operator(self, components, basis=None, exact=False):
		'''
		This method is a shorthand for constructing Operator objects which refer
		to the same Parameters instance as this Basis instance.

		:param components: Specification for Operator.
		:type components: Operator, dict, numpy.ndarray or sympy.MatrixBase
		:param basis: The basis in which the Operator is represented.
		:type basis: Basis or None
		:param exact: True if Operator should maintain exact representations of numbers,
			and False otherwise.
		:type exact: bool

		If *components* is already an Operator object, it is returned with its
		Parameters reference updated to point the Parameters instance associated
		with this Basis instance. Otherwise, a new Operator is constructed according
		to the specifications, again with a reference to this Basis's
		Parameters instance.
		
		For more information, refer to the documentation for Operator.
		'''
		if not isinstance(components, Operator):
			operator = Operator(components)
		operator.p = self.__p_ref
		return operator

	@abstractproperty
	def operator(self):
		'''
		This method should return a two dimensional `Operator` object, with basis states as 
		columns. The `Operator` object should use the `Parameters` instance provided by the
		Basis instance. The simplest way to ensure this is to use the `Basis.Operator` method.
		'''
		raise NotImplementedError("Basis operator has not been implemented.")

	def states(self, **params):
		'''
		This method returns the basis states (columns of the `Operator` returned by 
		`basis.operator`) as a list. The Operator is first evaluated with the 
		parameter overrides in params.
		
		:param params: A dictionary of parameter overrides. (see `parampy.Parameters`)
		:type params: dict
		'''
		i = xrange(self.dim)
		O = self.operator(**params)
		return map(lambda i: O[:, i], i)

	def state_info(self, state, params={}):
		'''
		This method (if implemented) should return a dictionary with more information
		about the state provided. There are no further constraints upon what might be
		returned.
		
		:param state: The state about which information should be returned.
		:type state: str or iterable
		:param params: A dictionary of parameter overrides. (see `parampy.Parameters`)
		:type params: dict
		'''
		return NotImplementedError("Basis.state_info has not been implemented.")

	def state_toString(self, state, params={}):
		'''
		This method (if implemented) should return a string representation of the
		provided state, which should then be able to be converted back into the same
		state using `Basis.state_fromString`.
		
		:param state: The state which should be represented as a string.
		:type state: iterable
		:param params: A dictionary of parameter overrides. (see `parampy.Parameters`)
		:type params: dict
		'''
		raise NotImplementedError("Basis.state_toString has not been implemented.")

	def state_fromString(self, string, params={}):
		'''
		This method (if implemented) should return the state as a numerical array that 
		is represented as a string in `string`. Calling `basis.state_toString` should then
		return the same (or equivalent) string representation.
		
		:param string: A string representation of a state.
		:type state: str
		:param params: A dictionary of parameter overrides. (see `parampy.Parameters`)
		:type params: dict
		'''
		raise NotImplementedError("Basis.state_fromString has not been implemented.")

	def state_latex(self, state, params={}):
		'''
		This method (if implemented) should return string that when compiled by
		LaTeX would represent the state.
		
		:param state: The state which should be represented as a string.
		:type state: iterable
		:param params: A dictionary of parameter overrides. (see `parampy.Parameters`)
		:type params: dict
		'''
		raise NotImplementedError("Basis.state_latex has not been implemented.")

	def state_toSymbolic(self, state):
		'''
		This method is a stub, and may be implemented in the future to provide the 
		logical inverse of `Basis.state_fromSymbolic`.
		'''
		raise NotImplementedError("Conversion of a state to a symbolic representation has not yet been implemented.")

	def state_fromSymbolic(self, expr):
		'''
		This method converts a sympy representation of a quantum state into
		an array or vector (as used by QuBricks). It uses internally `Basis.state_fromString` 
		to recognise ket and bra names, and to substitute them appropriately with the right
		state vectors.
		
		.. warning:: Support for conversion from symbolic representations is not fully
			baked, but seems to work reasonably well.
		'''

		r = np.array(sq.represent(expr, basis=self.__sympy_basis).tolist(), dtype=object)

		if len(r.shape) == 2:
			if r.shape[0] == 1:
				r = r[0, :]
			else:
				r = r[:, 0]
		return r

	@property
	def __sympy_basis(self):
		'''
		Sympy requires an operator instance to determine how states should represent
		themselves. This property provides such an instance.
		'''
		op = QubricksBasis(self.__class__.__name__)
		op.qubricks = self
		return op

	#
	# Transform vector and matrix elements from the standard basis to this basis
	def transform(self, state, inverse=False, threshold=False, params={}):
		'''
		This method allows one to transform states from the standard basis to
		this basis; or, if the inverse flag is provided, to transform from this
		basis to the standard basis. This is chained in the Basis.transform_to and
		Basis.transform_from methods to convert states between bases. State objects
		can be Operator or numpy array objects; and can be one or two dimensional.
		The basis states are evaluated in the parameter context specified in `params` 
		before being used in this method.
		
		This method can automatically try to set elements in the transformed object that 
		are different from zero by some small amount to zero, in the hope of ignoring 
		numerical error. If threshold is `False`, no attempts to clean the transformed state are made.
		If a numerical threshold is provided, any elements of the resulting
		transformed state with amplitude less than the supplied value will be set
		to zero. If threshold is set to True, the transformation operation attempts
		to determine the threshold automatically. This automatic algorithm looks 
		for the smallest entry in `Basis.operator` and then multiplies it by 10**-8. 
		This value is then used as the threshold. One should use this feature with
		caution.
		
		:param state: The state to be transformed.
		:type state: 1D or 2D Operator or numpy.ndarray
		:param inverse: `True` for transformation from this basis to the standard basis,
			and `False` for transformation to this basis from the standard basis.
		:type inverse: bool
		:param threshold: True or False to specify that the threshold should be automatically
			determined or not used respectively. If a float is provided, that value is used as
			the threshold.
		:type threshold: bool or float
		:param params: The parameter overrides to use during the transformation (see `Operator`).
		:type params: dict
		'''

		if isinstance(state, Operator):
			self.operator.basis = "*"

			if len(state.shape) == 1:
				if inverse:
					output = self.operator * state
				else:
					output = self.operator.inverse() * state

			elif len(state.shape) == 2:  # assume square
				if inverse:
					output = self.operator * state * self.operator.inverse()
				else:
					output = self.operator.inverse() * state * self.operator
			else:
				raise ValueError

			return self.__filter(output, state=state, threshold=threshold, params=params)

		elif isinstance(state, str):
			state = self.state_fromString(state, params)

		elif not isinstance(state, np.ndarray):
			state = np.array(state)

		state = np.array(state)
		output = None

		o = self.operator(**params)

		# If input is a vector
		if state.ndim == 1:
			if not inverse:
				o = np.linalg.inv(o)
			output = np.dot(o, state)
		elif state.ndim == 2:
			od = np.linalg.inv(o)
			if inverse:
				output = np.dot(o, np.dot(state, od))
			else:
				output = np.dot(od, np.dot(state, o))
		else:
			raise ValueError("Invalid number of dimensions.")

		return self.__filter(output, state=state, threshold=threshold, params=params)

	def __filter(self, output, state=None, threshold=False, params={}):
		'''
		Basis.__filter is a private method that implements the thresholding described in
		the documentation for Basis.transform .
		'''

		def __filter_threshold(a):
			# TODO: DO THIS MORE NEATLY?
			'''
			Determine the minimum threshold over real and imaginary components which should capture everything.
			'''
			try:
				real_ind = np.where(np.abs(a.real) > 1e-15)
				t_real = np.min(np.abs(a[real_ind])) if len(real_ind[0]) > 0 else np.inf

				imag_ind = np.where(np.abs(a.imag) > 1e-15)
				t_imag = np.min(np.abs(a[imag_ind])) if len(imag_ind[0]) > 0 else np.inf

				return min(t_real, t_imag)
			except:
				return False

		if threshold is False:
			return output

		warnings.warn("Be careful with auto-thresholding.")
		threshold = __filter_threshold(self.operator(**params))
		# print "op threshold", threshold
		if state is not None:
			if isinstance(state, Operator):
				threshold = np.min(map(__filter_threshold, state.components.values()))  # TODO: do more abstractly?
			else:
				threshold = min(__filter_threshold(state), threshold)
		# print "w/ state threshold", threshold
		if threshold is False:
			return output

		threshold *= 1e-8
		# print "final threshold", threshold

		if isinstance(state, Operator):
			output.clean(threshold)
		else:
			ind = np.where(np.abs(output) < threshold)
			output[ind] = 0
		return output

	def transform_from(self, state, basis=None, threshold=False, params={}):
		'''
		This method transforms the given state to this basis from the basis provided in
		`basis` (which must be a Basis instance). If `basis` is note provided, the 
		standard basis is assumed.
		
		:param state: The state to be transformed.
		:type state: 1D or 2D Operator or numpy.ndarray
		:param basis: The basis into which the state should be transformed.
		:type basis: Basis or None
		:param threshold: True or False to specify that the threshold should be automatically
			determined or not used respectively. If a float is provided, that value is used as
			the threshold.
		:type threshold: bool or float
		:param params: The parameter overrides to use during the transformation (see `Operator`).
		:type params: dict
		'''
		if basis is not None:
			if not isinstance(basis, Basis):
				raise ValueError('`basis` must be a Basis object; received `%s` of type `%s`.' % (str(basis), type(basis)))
			state = basis.transform(state, inverse=True, threshold=threshold, params=params)

		return self.transform(state, threshold=threshold, params=params)

	def transform_to(self, state, basis=None, threshold=False, params={}):
		'''
		This method transforms the given state from this basis to the basis provided in
		`basis` (which must be a Basis instance). If `basis` is note provided, the 
		standard basis is assumed.
		
		:param state: The state to be transformed.
		:type state: 1D or 2D Operator or numpy.ndarray
		:param basis: The basis into which the state should be transformed.
		:type basis: Basis or None
		:param threshold: True or False to specify that the threshold should be automatically
			determined or not used respectively. If a float is provided, that value is used as
			the threshold.
		:type threshold: bool or float
		:param params: The parameter overrides to use during the transformation (see `Operator`).
		:type params: dict
		'''
		if basis is not None:
			if not isinstance(basis, Basis):
				raise ValueError('`basis` must be a Basis object.')
			return basis.transform_from(state, basis=self, threshold=threshold, params=params)

		return self.transform(state, inverse=True, threshold=threshold, params=params)

	def transform_op(self, basis=None, inverse=False, threshold=False, params={}):
		'''
		This method returns a function which can be used to transform any 1D or 2D 
		`Operator` or numpy array to (from) this basis from (to) the basis provided 
		in `basis`, if `inverse` is False (True). If basis is not provided, the
		standard basis is assumed.
		
		:param state: The state to be transformed.
		:type state: 1D or 2D Operator or numpy.ndarray
		:param basis: The basis into which the state should be transformed.
		:type basis: Basis or None
		:param inverse: `True` for transformation from this basis to the `basis` provided,
			and `False` for transformation to this basis from the the `basis` provided.
		:type inverse: bool
		:param threshold: True or False to specify that the threshold should be automatically
			determined or not used respectively. If a float is provided, that value is used as
			the threshold.
		:type threshold: bool or float
		:param params: The parameter overrides to use during the transformation (see `Operator`).
		:type params: dict
		
		For example:
		
		>>> f = Basis.transform_op()
		>>> state_transformed = f(state)
		'''
		if inverse:
			return lambda y: self.transform_to(y, basis=basis, threshold=threshold, params=params)
		return lambda y: self.transform_from(y, basis=basis, threshold=threshold, params=params)

############ SYMBOLIC STATE REPRESENTATION HELPERS #################################################


class QubricksBasis(sq.Operator):
	'''
	This object is used internally to support symbolic representations of states.
	'''
	pass


# TODO: Flesh out sympy symbolic representation
class QubricksKet(sq.Ket):
	'''
	This object is used to represent states analytically.
	
	For example:
	
	>>> ket = QubricksKet('0')
	
	These objects then obey standard arithmetic, for example:
	
	>>> 2*ket
	2|0>
	
	You can convert from a symbolic representation of states
	to a QuBricks array using `Basis.state_fromSymbolic`.
	'''
	def _represent_QubricksBasis(self, basis, **options):
		if getattr(basis, 'qubricks') is None:
			raise ValueError("The `qubricks` attribute must be set on the basis object for ket representation.")
		# print str(self)
		# print sympy.Matrix( basis.qubricks.state_fromString(str(self)) ).applyfunc(sympy.nsimplify)
		return sympy.Matrix(basis.qubricks.state_fromString(str(self))).applyfunc(sympy.nsimplify)

	def _eval_innerproduct_QubricksBra(self, bra, **hints):
		return 1 if bra.label == self.label else 0

	def _eval_conjugate(self):
		return QubricksBra(*self.label)

	@property
	def strlabel(self):
		return ",".join(map(str, self.label))

	def __mul__(self, other):
		if isinstance(other, QubricksKet):
			return QubricksKet(",".join((self.strlabel, other.strlabel)))
		return super(QubricksKet, self).__mul__(other)

	def __rmul__(self, other):
		if isinstance(other, QubricksKet):
			return QubricksKet(",".join((other.strlabel, self.strlabel)))
		return super(QubricksKet, self).__rmul__(other)

	def __pow__(self, power):
		r = 1
		for _ in xrange(power):
			r *= self
		return r

	@classmethod
	def dual_class(self):
		return QubricksBra


class QubricksBra(sq.Bra):
	'''
	This object is used to represent states analytically.
	
	For example:
	
	>>> bra = QubricksBra('0')
	
	These objects then obey standard arithmetic, for example:
	
	>>> 2*bra
	2<0|
	
	You can convert from a symbolic representation of states
	to a QuBricks array using `Basis.state_fromSymbolic`.
	'''

	@classmethod
	def dual_class(self):
		return QubricksKet

	def _represent_QubricksBasis(self, basis, **options):
		if 'qubricks_basis' not in options:
			raise ValueError("Qubricks basis object must be passed to ket for representation.")
		basis = options['qubricks_basis']
		l = ",".join(map(str, self.label))
		return sympy.Matrix(basis.state_fromString("|%s>" % l).transpose()).applyfunc(sympy.nsimplify)

