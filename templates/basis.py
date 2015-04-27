
from qubricks import Basis

class CustomBasis(Basis):
	'''
	Refer to the API documentation of `Basis` for more details.
	'''
	
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
	
	@property
	def operator(self):
		'''
		This method should return a two dimensional `Operator` object, with basis states as 
		columns. The `Operator` object should use the `Parameters` instance provided by the
		Basis instance. The simplest way to ensure this is to use the `Basis.Operator` method.
		'''
		raise NotImplementedError("Basis operator has not been implemented.")
	
#	The following methods are optional; refer to `Basis` documentation for more information.
# 	def state_info(self, state, params={}):
# 		'''
# 		This method (if implemented) should return a dictionary with more information
# 		about the state provided. There are no further constraints upon what might be
# 		returned.
# 		
# 		:param state: The state about which information should be returned.
# 		:type state: str or iterable
# 		:param params: A dictionary of parameter overrides. (see `parampy.Parameters`)
# 		:type params: dict
# 		'''
# 		return NotImplementedError("Basis.state_info has not been implemented.")
# 
# 	def state_toString(self, state, params={}):
# 		'''
# 		This method (if implemented) should return a string representation of the
# 		provided state, which should then be able to be converted back into the same
# 		state using `Basis.state_fromString`.
# 		
# 		:param state: The state which should be represented as a string.
# 		:type state: iterable
# 		:param params: A dictionary of parameter overrides. (see `parampy.Parameters`)
# 		:type params: dict
# 		'''
# 		raise NotImplementedError("Basis.state_toString has not been implemented.")
# 
# 	def state_fromString(self, string, params={}):
# 		'''
# 		This method (if implemented) should return the state as a numerical array that 
# 		is represented as a string in `string`. Calling `basis.state_toString` should then
# 		return the same (or equivalent) string representation.
# 		
# 		:param string: A string representation of a state.
# 		:type state: str
# 		:param params: A dictionary of parameter overrides. (see `parampy.Parameters`)
# 		:type params: dict
# 		'''
# 		raise NotImplementedError("Basis.state_fromString has not been implemented.")
# 
# 	def state_latex(self, state, params={}):
# 		'''
# 		This method (if implemented) should return string that when compiled by
# 		LaTeX would represent the state.
# 		
# 		:param state: The state which should be represented as a string.
# 		:type state: iterable
# 		:param params: A dictionary of parameter overrides. (see `parampy.Parameters`)
# 		:type params: dict
# 		'''
# 		raise NotImplementedError("Basis.state_latex has not been implemented.")