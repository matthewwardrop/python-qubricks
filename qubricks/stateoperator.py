from abc import ABCMeta, abstractmethod, abstractproperty
from parameters import Parameters

from .operator import Operator


class StateOperator(object):
	'''
	StateOperator objects are not to be confused with Operator objects. StateOperator
	objects can encode arbitrary operations on states; including those which cannot
	be described by a linear map. Often, but not always, the internal mechanisms of
	a StateOperator will *use* an Operator object. The StateOperator object itself
	is an abstract class, and must be subclassed before it is used. In order to
	conform to the expectations of the QuBricks framework, StateOperators should
	implement basis transformations and subspace restrictions.

	:param parameters: A referenced to a Parameters instance, which can be shared among
		multiple objects. This is overridden when adding to a QuantumSystem instance,
		but is helpful for testing purposes.
	:type parameters: Parameters
	:param kwargs: Keyword arguments which are passed onto the StateOperator.init method, which must
		be implemented by subclasses.
	:type kwargs: dict

	Subclassing StateOperator:
		A subclass of StateOperator must implement the following methods::
			- init(self, **kwargs)
			- __call__(self, state, t=0, params={})
			- restrict(self, *indices)
			- connected(self, *indices, **params)
			- transform(self, transform_op)

		And the following properties::
			- for_state
			- for_ensemble

		And may, if desired, implement::
			- collapse(self, *wrt, **params)

		For more documentation about what these methods should return, see the
		documentation below.

	Applying a StateOperator instance to a State:
		StateOperator objects are applied to states by calling them with the
		state as a passed parameter. For example:

		>>> stateoperator(state)

		It is also possible to pass a parameter context. Often, such as when
		performing numerical integration, you want to treat time specially, and so
		the time can also be passed on its own.

		>>> stateoperator(state, t=0, params=param_dict)

		Parameters may be supplied in any format that is supported by the Parameters module.

		.. note:: The integration time is passed by the integrator via `t`, and not
			as part of the parameter context.
	'''
	__metaclass__ = ABCMeta

	def __init__(self, parameters=None, basis=None, **kwargs):
		self.__p = parameters
		self.__basis = basis
		self.init(**kwargs)

	@abstractmethod
	def init(self, **kwargs):
		'''
		This method is called when StateOperator subclasses are
		initialised, which allows subclasses to set themselves up as appropriate.

		:param kwargs: The keyword arguments passed to the StateOperator constructor.
		:type kwargs: dict
		'''
		raise NotImplementedError("StateOperator.init is not implemented.")

	@property
	def p(self):
		'''
		Returns a reference to the internal Parameter instance.
		'''
		if self.__p is None:
			raise ValueError("Operator requires use of Parameters object; but none specified.")
		return self.__p
	@p.setter
	def p(self, parameters):
		if not isinstance(parameters, Parameters):
			raise ValueError("You must provide a Parameters instance.")
		self.__p = parameters

	def __p_ref(self):
		'''
		This method allows child Operator objects to automatically be kept up to date with
		changes to the Parameters instance associated with the StateOperator object. This
		cannot be used directly, but is used by the StateOperator.Operator method.
		'''
		return self.__p

	def Operator(self, operator, basis=None, exact=False):
		'''
		This method is a shorthand for constructing Operator objects which refer
		to the same Parameters instance as this StateOperator.

		:param components: Specification for Operator.
		:type components: Operator, dict, numpy.ndarray or sympy.MatrixBase
		:param basis: The basis in which the Operator is represented.
		:type basis: Basis or None
		:param exact: True if Operator should maintain exact representations of numbers,
			and False otherwise.
		:type exact: bool

		If *components* is already an Operator object, it is returned with its
		Parameters reference updated to point the Parameters instance associated
		with this StateOperator. Otherwise, a new Operator is constructed according
		to the specifications, again with a reference to this StateOperator's
		Parameters instance.
		
		For more information, refer to the documentation for Operator.
		'''
		if not isinstance(operator, Operator):
			operator = Operator(operator)
		operator.p = self.__p_ref
		return operator

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
	def restrict(self, *indices):
		'''
		This method should return a new StateOperator restricted to the basis states
		with indices `indices`. See Operator.restrict for more information.
		'''
		raise NotImplementedError("StateOperator.restrict is not implemented.")

	@abstractmethod
	def connected(self, *indices, **params):
		'''
		This method should return the list of basis state indices which would mix
		with the specified basis state indices `indices` under repeated operation of the
		StateOperator. See Operator.connected for more information.
		'''
		raise NotImplementedError("StateOperator.connected is not implemented.")

	def collapse(self, *wrt, **params):
		'''
		This method is a stub to allow subclasses to simplify themselves when
		requested. If implemented, and Operators are used, the `collapse` method
		should be used on them also. See Operator.collapse for more information.
		
		Note that unless this method is overridden, no simplification occurs.
		'''
		return self

	@abstractproperty
	def for_state(self):
		'''
		Should be True if the StateOperator supports 1D vector operations; and False otherwise.
		'''
		raise NotImplementedError("StateOperator.for_state is not implemented.")

	@abstractproperty
	def for_ensemble(self):
		'''
		Should be True if the StateOperator supports 2D ensemble operations; and False otherwise.
		'''
		raise NotImplementedError("StateOperator.process_args is not implemented.")
	
	@abstractproperty
	def is_linear(self):
		'''
		Should be True if the StateOperator is linear. If so, the `Integrator` instance
		may apply the real and imaginary components separately (or any other linear breakdown of the
		state).
		'''
		raise NotImplementedError("StateOperator.process_args is not implemented.")

	############### Basis Transformations ###########################################

	@abstractmethod
	def transform(self, transform_op):
		'''
		This method should transform all future operations on arbitrary input states
		according to the transformation `transform_op`. See Operator.transform
		for more information.
		'''
		raise NotImplementedError("StateOperator.transform is not implemented.")

	@property
	def basis(self):
		'''
		A reference to the current Basis of the Operator; or None if it has no specified
		Basis.
		'''
		return self.__basis
	@basis.setter
	def basis(self, basis):
		'''
		Sets the current basis of the Operator object. Note that this does NOT transform the
		Operator into this basis; but simply identifies the current form the Operator as being
		written in the specified basis. To transform basis, use Operator.change_basis .
		'''
		self.__basis = basis

	def change_basis(self, basis=None, threshold=False):
		'''
		Returns a copy of this StateOperator object expressed in the specified basis. The behaviour of the
		threshold parameter is described in the Basis.transform documentation; but this allows
		elements which differ from zero only by numerical noise to be set to zero.

		:param basis: The basis in which the new StateOperator should be represented.
		:type basis: Basis or None
		:param threshold: Whether a threshold should be used to limit the effects
			of numerical noise (if boolean), or the threshold to use (if float).
			See Basis.transform for more information.
		:type threshold: bool or float
		'''
		if basis == self.__basis:
			return self
		elif basis is None:
			O = self.transform(self.__basis.transform_op(threshold=threshold, inverse=True))
		else:
			O = self.transform(basis.transform_op(basis=self.__basis, threshold=threshold))
		O.basis = basis

		return O
