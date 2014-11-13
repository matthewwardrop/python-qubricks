
from qubricks import QuantumSystem

class CustomSystem(QuantumSystem):

	def setup_environment(self, **kwargs):
		'''
		Configure any custom properties/attributes using kwargs passed
		to __init__.
		'''
		raise NotImplementedError

	def setup_parameters(self):
		'''
		After the QuantumSystem parameter initialisation routines
		are run, check that the parameters are initialised correctly.
		'''
		raise NotImplementedError

	def setup_bases(self):
		'''
		Add the bases which are going to be used by this QuantumSystem
		instance.
		'''
		raise NotImplementedError

	def setup_hamiltonian(self):
		'''
		Initialise the Hamiltonian to be used by this QuantumSystem
		instance.
		'''
		raise NotImplementedError

	def setup_states(self):
		'''
		Add the named/important states to be used by this quantum system.
		'''
		raise NotImplementedError

	def setup_measurements(self):
		'''
		Add the measurements to be used by this quantum system instance.
		'''
		raise NotImplementedError

	@property
	def default_derivative_ops(self):
		raise NotImplementedError

	# Optional: can use self.add_derivative_op instead
	def get_derivative_ops(self, components=None):
		'''
		Setup the derivative operators to be implemented on top of the
		basic quantum evolution operator.
		'''
		raise NotImplementedError
