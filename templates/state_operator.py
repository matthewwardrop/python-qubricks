
from qubricks import StateOperator

class CustomStateOperator(StateOperator):
	
	def init(self, **kwargs):
		'''
		StateOperator.init is called when StateOperator subclasses are 
		initialised, which allows subclasses to set themselves up as appropriate.
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
		StateOperator.restrict should return a new StateOperator restricted to the basis states
		with indices `indices`.
		'''
		raise NotImplementedError("StateOperator.restrict is not implemented.")
	
	def connected(self, *indices, **params):
		'''
		StateOperator.connected should return the list of basis state indices which would mix
		with the specified basis state indices `indices` under repeated operation of the 
		StateOperator.
		'''
		raise NotImplementedError("StateOperator.connected is not implemented.")
	
	def for_state(self):
		'''
		StateOperator.for_state should return True if the StateOperator supports 1D vector
		operations; and False otherwise.
		'''
		raise NotImplementedError("StateOperator.for_state is not implemented.")
	
	def for_ensemble(self):
		'''
		StateOperator.for_ensemble should return True if the StateOperator supports 2D ensemble
		operations; and False otherwise.
		'''
		raise NotImplementedError("StateOperator.process_args is not implemented.")
	
	def transform(self, transform_op):
		'''
		StateOperator.transform should transform all operations on the state
		according to the basis transformation `transform_op`.
		'''
		raise NotImplementedError("StateOperator.transform is not implemented.")
