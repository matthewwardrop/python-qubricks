
from qubricks import Basis

class CustomBasis(Basis):
	
	def init(self, dim=None, **kwargs):
		'''
		Basis.init is called during the basis initialisation
		routines, allowing Basis subclasses to initialise themselves.
		'''
		pass
	
	@property
	def operator(self):
		'''
		Basis.operator must return an Operator object with basis states as the 
		columns. The operator should use the parameters instance provided by the
		Basis subclass.
		'''
		raise NotImplementedError, "Basis operator has not been implemented."
