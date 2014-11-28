from abc import ABCMeta, abstractmethod, abstractproperty
from parameters import Parameters

from .operator import Operator


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

	def __init__(self, parameters=None, basis=None, **kwargs):
		self.__p = parameters
		self.__basis = basis
		self.init(**kwargs)

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

	def Operator(self, operator):
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

	def collapse(self, *wrt, **params):
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
