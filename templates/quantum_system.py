
from qubricks import QuantumSystem

class CustomSystem(QuantumSystem):
	'''
	Refer to the API documentation for `QuantumSystem` for more information.
	'''

	def init(self, **kwargs):
		'''
		This method can be used by subclasses to initialise the state
		of the `QuantumSystem` instance. Any excess kwargs beyond `parameters`
		passed to q `QuantumSystem` instance will be passed to this method.
		'''
		pass

	def init_parameters(self):
		'''
		This method can be used by subclasses to add any additional
		parameters required to describe the quantum system. If this 
		method returns a dictionary, then it is used to update the
		parameters stored in the `Parameters` instance. This would
		be equivalent to:
		
		>>> system.p << system.init_parameters()
		
		Parameters may, of course, be set directly by this method, using 
		(for example):
		
		>>> self.p.x = 1
		'''
		pass
	
	def init_hamiltonian(self):
		'''
		This method can be used by subclasses to initialise the Hamiltonian
		to be used to describe the Quantum System. The Hamiltonian can
		either be set directly in this method, using:
		
		>>> self.hamiltonian = <Operator or OperatorSet>
		
		Alternatively, if this method returns an Operator or OperatorSet
		object, then it will be set as the Hamiltonian for this QuantumSystem
		instance.
		'''
		pass

	def init_bases(self):
		'''
		This method can be used by subclasses to initialise the bases to be
		used by this instance of `QuantumSystem`. Bases can be added directly
		using the `add_basis` method; or, if this method returns a dictionary 
		of `Basis` objects (indexed by string names), then they will be added
		as bases of this system.
		'''
		pass

	def init_states(self):
		'''
		This method can be used by subclasses to initialise named states, ensembles,
		and subspaces. This can be done directly, using the corresponding 
		`add_state` and `add_subspace` methods. If a dictionary is returned, then
		it is assumed to be a dictionary of states indexed by names, which are then
		added to the system using `add_state`. Note that this assumes that the 
		states are represented in the standard basis of this `QuantumSystem` object.
		'''
		pass

	def init_measurements(self):
		'''
		This method can be used by subclasses to initialise the measurements that
		will be used with this `QuantumSystem` instance. This can be done directly, 
		using `add_measurement`; or, if this method returns a dictionary of `Measurement`
		objects indexed by string names, then they will be added as potential measurements
		of this quantum system.
		'''
		pass
	
	def init_derivative_ops(self):
		'''
		This method can be used by subclasses to initialise the `StateOperator` objects
		use to describe the time derivative of the evolution of the quantum system
		described by this object. Derivative operators may be added directly using
		`add_derivative_op`, or, if a dictionary of `StateOperator` objects is returned
		indexed with string names, then they are added as derivative operators of this object.
		If the operators depend on the hamiltonian or other properties of the quantum system,
		then the operators should be implemented in `get_derivative_ops` instead.
		
		This method should also initialise the default_derivative_ops property.
		'''
		pass
	
	def get_derivative_ops(self, components=None):
		'''
		This method can be used by subclasses to specify the `StateOperator` objects
		use to describe the time derivative of the evolution of the quantum system
		described by this object. These operators are added just before integration
		the operators described in `init_derivative_ops` and the 'evolution' operator
		describing Schroedinger evolution. Any properties of this
		`QuantumSystem` instance should not change before integration.
		
		:param components: The components activated in the Hamiltonian for this integration.
		:type components: iterable
		'''
		pass
	
