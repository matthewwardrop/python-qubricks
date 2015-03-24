
from qubricks import StateOperator

class CustomStateOperator(StateOperator):
	'''
	Refer to the API documentation for `StateOperator` for more information.
	'''
	
	def init(self, **kwargs):
		'''
		This method is called when StateOperator subclasses are
		initialised, which allows subclasses to set themselves up as appropriate.

		:param kwargs: The keyword arguments passed to the StateOperator constructor.
		:type kwargs: dict
		'''
		raise NotImplementedError("StateOperator.init is not implemented.")

	def __call__(self, state, t=0, params={}):
		'''
		StateOperator objects are called on states to effect some desired operation.
		States may be 1-dimensional (as state vectors) or 2-dimensional (as quantum ensembles),
		and each subclass should specify in StateOperator.for_state and StateOperator.for_ensemble
		which kinds of states are supported.
		'''
		raise NotImplementedError("StateOperator.__call__ is not implemented.")
	
	def restrict(self, *indices):
		'''
		This method should return a new StateOperator restricted to the basis states
		with indices `indices`. See Operator.restrict for more information.
		'''
		raise NotImplementedError("StateOperator.restrict is not implemented.")
	
	def connected(self, *indices, **params):
		'''
		This method should return the list of basis state indices which would mix
		with the specified basis state indices `indices` under repeated operation of the
		StateOperator. See Operator.connected for more information.
		'''
		raise NotImplementedError("StateOperator.connected is not implemented.")
	
	def for_state(self):
		'''
		Should be True if the StateOperator supports 1D vector operations; and False otherwise.
		'''
		raise NotImplementedError("StateOperator.for_state is not implemented.")

	def for_ensemble(self):
		'''
		Should be True if the StateOperator supports 2D ensemble operations; and False otherwise.
		'''
		raise NotImplementedError("StateOperator.process_args is not implemented.")
	
	def transform(self, transform_op):
		'''
		This method should transform all future operations on arbitrary input states
		according to the transformation `transform_op`. See Operator.transform
		for more information.
		'''
		raise NotImplementedError("StateOperator.transform is not implemented.")
	
#	The following method is optional	
# 	def collapse(self, *wrt, **params):
# 		'''
# 		This method is a stub to allow subclasses to simplify themselves when
# 		requested. If implemented, and Operators are used, the `collapse` method
# 		should be used on them also. See Operator.collapse for more information.
# 		
# 		Note that unless this method is overridden, no simplification occurs.
# 		'''
# 		return self
	
