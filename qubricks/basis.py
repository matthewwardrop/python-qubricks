from abc import ABCMeta, abstractmethod, abstractproperty
import math
import re
import warnings

import numpy as np
import sympy
import sympy.physics.quantum as sq

from .operators import Operator


class Basis(object):
	'''
	Basis(name="",dim=None,parameters=None,**kwargs)
	
	An object that represents a choice of basis for the QuBricks library. 
	Basis objects allow one to transform state vectors between bases, and
	to represent state vectors in strings. Basis is an abstract class, and
	must be inherited to be used.
	
	Parameters
	----------
	name : A friend name to be used when error messages are returned to the
		user.
	dim : The dimension of the basis. If not specified, the dimension will 
		be extracted from the Operator returned by Basis.operator.
	parameters : A Parameters instance; which can be shared between all basis
		(and other) objects.
	kwargs : Extra arguments that are passed to the __basis_init function.
	
	Subclasses
	----------
	In order to create a Basis object, one must create a new class that 
	inherits from Basis, and implements at least the following abstract 
	methods:
	- Basis.__basis_init__
	- Basis.operator
	Optionally, one can also implement:
	- Basis.state_toString
	- Basis.state_fromString
	- Basis.state_latex
	
	Documentation for all of the methods of Basis is provided inline.
	'''
	__metaclass__ = ABCMeta

	def __init__(self, name="", dim=None, parameters=None, **kwargs):
		self.__name = name
		self.__dim = dim
		self.__parameters = parameters
		self.init(dim=dim, **kwargs)
	
	@abstractmethod
	def init(self, dim=None, **kwargs):
		'''
		Basis.init_basis is called during the basis initialisation
		routines, allowing Basis subclasses to initialise themselves.
		'''
		pass
	
	@property
	def name(self):
		'''
		A friendly name to be used in errors and user-facing messages.
		'''
		return self.__name
	
	def __repr__(self):
		return "<%s(Basis) '%s'>" % (self.__class__.__name__, self.name)
	
	@property
	def dim(self):
		'''
		The dimension of the basis; or equivalently, the number of basis states.
		'''
		if self.__dim == None:
			self.__dim = self.operator().shape[0]
		return self.__dim
	
	@property
	def p(self):
		'''
		A reference to the Parameters instance used by the Operator objects.
		'''
		return self.__parameters
	
	@abstractproperty
	def operator(self):
		'''
		Basis.operator must return an Operator object with basis states as the 
		columns. The operator should use the parameters instance provided by the
		Basis subclass.
		'''
		raise NotImplementedError, "Basis operator has not been implemented."
	
	def states(self, **params):
		'''
		Returns the columns of Basis.operator (after evaluations with the 
		parameters in `params`) as list. These are the "basis states".
		'''
		i = xrange(self.dim)
		O = self.operator(**params)
		return map(lambda i: O[:, i], i)
	
	def state_info(self, state, params={}):
		'''
		Basis.state_info can be optionally implemented by subclasses to provide
		further information about states in this basis.
		'''
		return NotImplementedError("Basis.state_info has not been implemented.")

	def state_toString(self, state, params={}):
		'''
		Basis.state_toString can be optionally implemented by subclasses to provide
		a mapping from states in this basis to strings.
		'''
		raise NotImplementedError("Basis.state_toString has not been implemented.")

	def state_fromString(self, string, params={}):
		'''
		Basis.state_fromString can be optionally implemented by subclasses to provide
		a mapping from strings to states in this basis.
		'''
		raise NotImplementedError("Basis.state_fromString has not been implemented.")
	
	def state_latex(self, state, params={}):
		'''
		Basis.state_info can be optionally implemented by subclasses to provide
		a latex representation of states in this basis.
		'''
		raise NotImplementedError("Basis.state_latex has not been implemented.")
	
	def state_toSymbolic(self, state):
		'''
		Basis.state_toSymbolic converts state objects to symbolic representations.
		'''
		raise NotImplementedError("Symbolic conversion has not yet been implemented.")
	
	def state_fromSymbolic(self, expr):
		'''
		Basis.state_fromSymbolic converts symbolic representations of states into 
		numerical state vectors.
		'''
		return np.array(sq.represent(expr, basis=self.__sympy_basis, qubricks_basis=self).tolist()[0],dtype=object)
	
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
		Basis.transform allows one to transform states from the standard basis to
		this basis; or, if the inverse flag is provided, to transform from this
		basis to the standard basis. This is chained in the Basis.transform_to and
		Basis.transform_from methods to convert states between bases. State objects 
		can be Operator or numpy array objects; and can be one or two dimensional. 
		The basis states are evaluated at `params` before being used in this method.
		If threshold is False, no attempts to neaten the transformed state are made.
		If a numerical threshold is provided, any elements of the resulting 
		transformed state with amplitude less than the supplied value will be set 
		to zero. If threshold is set to True, the transformation operation attempts
		to determine the threshold automatically. One should use this feature with 
		caution.
		'''
		if isinstance(state, Operator):
			
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
			raise ValueError, "Invalid number of dimensions."

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
		Basis.transform_from transforms states from the basis specified to this
		basis. `basis` must be a Basis instance itself. The other parameters
		are described in the Basis.transform method.
		'''
		if basis is not None:
			if not isinstance(basis, Basis):
				raise ValueError, '`basis` must be a Basis object.'
			state = basis.transform(state, inverse=True, threshold=threshold, params=params)
		
		return self.transform(state, threshold=threshold, params=params)
	
	def transform_to(self, state, basis=None, threshold=False, params={}):
		'''
		Basis.transform_to transforms states to the basis specified from this
		basis. `basis` must be a Basis instance itself. The other parameters
		are described in the Basis.transform method.
		'''
		if basis is not None:
			if not isinstance(basis, Basis):
				raise ValueError, '`basis` must be a Basis object.'
			return basis.transform_from(state, basis=self, threshold=threshold, params=params)
		
		return self.transform(state, inverse=True, threshold=threshold, params=params)
	
	def transform_op(self, basis=None, invert=False, threshold=False, params={}):
		'''
		Basis.transform_op returns a lambdified function which can be later applied to 
		states independently of this Basis instance. If basis is not provided, the
		standard basis is assumed.
		
		e.g.
		>>> f = Basis.transform_op()
		>>> state_transformed = f(state)
		'''
		if invert:
			return lambda y: self.transform_to(y, basis=basis, threshold=threshold, params=params)
		return lambda y: self.transform_from(y, basis=basis, threshold=threshold, params=params)
	

############ SYMBOLIC STATE REPRESENTATION HELPERS #################################################


class QubricksBasis(sq.Operator):
	pass

# TODO: Flesh out sympy symbolic representation
class QubricksKet(sq.Ket):
	def _represent_QubricksBasis(self, basis, **options):
		if 'qubricks_basis' not in options:
			raise ValueError("Qubricks basis object must be passed to ket for representation.")
		basis = options['qubricks_basis']
		return sympy.Matrix( basis.state_fromString(str(self)) ).applyfunc(sympy.nsimplify)

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
	
	@classmethod
	def dual_class(self):
		return QubricksKet

	def _represent_QubricksBasis(self, basis, **options):
		if 'qubricks_basis' not in options:
			raise ValueError("Qubricks basis object must be passed to ket for representation.")
		basis = options['qubricks_basis']
		l = ",".join(map(str, self.label))
		return sympy.Matrix(basis.state_fromString("|%s>" % l).transpose()).applyfunc(sympy.nsimplify)


######## Useful Basis Implementations ###################################################################
	
class StandardBasis(Basis):
	
	def init(self,dim=None):
		if dim is None:
			raise ValueError("Dimension must be specified.")
	
	@property
	def operator(self):
		return Operator(parameters=self.p, components={None:np.eye(self.dim)})


class SpinBasis(StandardBasis):

	def init(self,dim=None):
		if dim is None:
			raise ValueError("Dimension must be specified.")
		if dim % 2 == 1:
			raise ValueError("Dimension must be even.")
	
	def state_info(self, state, params={}):
		'''
		Return a dictionary with the total z-spin projection of the state.
		
		e.g. |uud> -> {'spin': 0.5}
		'''
		totalSpin = 0
		index = state.index(1)
		for _ in xrange(int(math.log(self.dim, 2))):
			mod = (index % 2 - 1.0 / 2.0)
			totalSpin -= mod
			index = (index - index % 2) / 2
		return {'spin':totalSpin}
	
	def state_fromString(self, state, params={}):
		'''
		Convert strings representing sums of basis states of form:
		"<complex coefficient>|<state label>>"
		into a numerical vector.
		
		e.g. "|uuu>" -> [1,0,0,0,0,0,0,0]
			 "0.5|uuu>+0.5|ddd>" -> [0.5,0,0,0,0,0,0,0.5]
			 etc.
		'''
		matches = re.findall("(([\+\-]?(?:[0-9\.]+)?)\|([\,ud]+)\>)", state.replace(",", ""))
		
		ostate = [0.] * self.dim
		
		for m in matches:
			coeff = 1.
			if m[1] != "":
				if m[1] == "+":
					coeff = 1.
				elif m[1] == "-":
					coeff = -1.
				else:
					coeff = float(m[1])
			
			index = 0
			for i in xrange(len(m[2])):
				j = len(m[2]) - i - 1
				
				if m[2][i] == 'd':
					index += 2 ** j
			ostate[index] = coeff
		return np.array(ostate)
	
	def state_toString(self, state, params={}):
		"""
		Applies the inverse map of SpinBasis.state_fromString .
		"""
		s = ''
		for i, v in enumerate(state):
			if v != 0:
				if s != "" and v >= 0:
					s += "+"
				if v == 1:
					s += "|%s>" % self.__state_str(i)
				else:
					if np.imag(v) != 0:
						s += "(%.3f + %.3fi)|%s>" % (np.real(v), np.imag(v), self.__state_str(i))
					else:
						s += "%.3f|%s>" % (np.real(v), self.__state_str(i))
		return s
	
	def state_latex(self, state, params={}):
		"""
		Returns the latex representation of each of the basis states. Note that this 
		method does not deal with arbitrary states, as does SpinBasis.state_toString
		and SpinBasis.state_fromString .
		"""
		spinString = ''
		index = state.index(1)
		for _ in xrange(self.dim / 2):
			mod = index % 2
			if mod == 0:
				spinString = '\\uparrow' + spinString
			else:
				spinString = '\\downarrow' + spinString
			index = (index - index % 2) / 2
		return '\\left|%s\\right>' % (spinString)
	
	def __state_str(self, index):
		"""
		A utility function to convert a basis function index to a string.
		
		e.g. [1,0,0,0,0,0,0,0] -> "uuu"
		"""
		s = ""
		for _ in xrange(int(math.log(self.dim, 2))):
			mod = index % 2
			if mod == 0:
				s = 'u' + s
			else:
				s = 'd' + s
			index = (index - index % 2) / 2
		return s
	
	
