import warnings
import importlib
import types

import numpy as np

from parampy import Parameters

from .basis import Basis
from .operator import Operator, OperatorSet
from .stateoperator import StateOperator


class QuantumSystem(object):
	'''
	The `QuantumSystem` class is used to describe particular quantum systems,
	and provides utility functions which assists in the analysis of these
	systems. While it is possible to directly instantiate a `QuantumSystem`
	object, and to programmatically add the description of the quantum system
	of interest, it is more common to subclass it or to use a ready-made subclass
	from `qubricks.wall.systems`.

	:param parameters: An object used to initialise a Parameters instance for use in this object.
	:type parameters: Parameters, str or None
	:param kwargs: Additional keyword arguments to pass to `init`, which may be useful for subclasses.
	:type kwargs: dict

	Specifying parameters:
		Every QuantumSystem instance requires access to a `Parameters` instance
		in order to manage the physical quantities associated with the
		represented quantum system. When instantiating `QuantumSystem`
		or one of its subclasses, this is managed by passing as a value to the
		`parameters` keyword one of the following:
		- A `Parameters` instance, which is then used directly.
		- A string, in which case QuantumSystem attempts to load a parameters
		  configuration from the path indicated in the string.
		- `None`, in which case an empty `Parameters` instance is constructed.

	Subclassing:
		If you choose to subclass `QuantumSystem` in order to represent a
		quantum system of interest, it should only be necessary to implement some
		or all of the following methods:
			- init(self, **kwargs)
			- init_parameters(self)
			- init_hamiltonian(self)
			- init_bases(self)
			- init_states(self)
			- init_measurements(self)
			- init_derivative_ops(self)
			- get_derivative_ops(self,components=None)
		Documentation pertaining to the behaviour of these methods is available
		below. Importantly, these methods will always be called in the order
		given above.

	Integration and Measurement:
		Perhaps among the most common operations you might want to perform
		with your quantum system is simulated time-evolution and measurement.

		Integration is handled as documented in the `QuantumSystem.integrate` method.

		Measurement is handled using the `measure` attribute, which points to
		a `Measurements` object. For example, if a measurement named 'fidelity'
		has been added to this QuantumSystem, you could use:

		>>> system.measure.fidelity(...)

		For more, see the documentation for `Measurements`.
	'''

	def __init__(self, parameters=None, **kwargs):

		self.__H = None
		self.__derivative_ops = {}
		self.__derivative_ops_default = ['evolution']

		self.__named_states = {}  # A dictionary of named states for easy recollection
		self.__named_ensembles = {}  # A dictionary of named ensembles for density states
		self.__named_subspaces = {}  # A dictionary of named subspaces for easy identification
		self.__named_bases = {}  # A dictionary of named bases for state representation
		self.__basis_default = None

		self.init(**kwargs)

		# Initialise parameters instance
		if isinstance(parameters, str):
			self.p = Parameters.load(parameters, constants=True)
		elif isinstance(parameters, Parameters):
			self.p = parameters
		else:
			self.p = Parameters(constants=True)
		if isinstance(parameters, dict):
			self.p << parameters
		params = self.init_parameters()
		if isinstance(params, dict):
			self.p << params

		# Initialise Hamiltonian
		H = self.init_hamiltonian()
		if isinstance(H, (Operator, OperatorSet)):
			self.hamiltonian = H

		# Initialise Bases
		bases = self.init_bases()
		if isinstance(bases, dict):
			for name, basis in bases.items():
				self.add_basis(name, basis)

		# Initialise named states
		states = self.init_states()
		if isinstance(states, dict):
			for name, state in states.items():
				self.add_state(name, state)

		# Initialise derivative operators
		derivatives = self.init_derivative_ops()
		if isinstance(derivatives, dict):
			for name, derivative in derivatives.items():
				self.add_derivative_op(name, derivative)

		# Initialise measurements
		self.measure = Measurements(self)
		measurements = self.init_measurements()
		if isinstance(measurements, dict):
			for name, measurement in measurements:
				self.add_measurement(name, measurement)

	# Initialisation Methods
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

	# Other overridable methods

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

	# Post construction interrogation and configuration

	@property
	def p(self):
		'''
		A reference to the `Parameters` instance used by this object.
		'''
		if self.__p is None:
			raise ValueError("Parameters instance required by Basis object, a Parameters object has not been configured.")
		return self.__p
	@p.setter
	def p(self, parameters):
		if parameters is not None and not isinstance(parameters, Parameters):
			raise ValueError("Parameters reference must be an instance of Parameters or None.")
		self.__p = parameters

	@property
	def hamiltonian(self):
		'''
		The `Operator` or `OperatorSet` object used to describe this quantum system.
		If not yet specified, this will be `None`. The Hamiltonian can be specified using:

		>>> system.hamiltonian = <Operator or OperatorSet>

		.. note:: The Hamiltonian is expected to be represented such that the basis
				vectors form the standard basis. This is, in practice, not a limitation; but
				is important to remember when transforming bases.
		'''
		return self.__H
	@hamiltonian.setter
	def hamiltonian(self, hamiltonian):
		if len(self.__derivative_ops) > 0:
			warnings.warn("If you are using the Hamiltonian directly in your derivative operators, \
			you may need to reinitialise them now (using `init_derivative_ops`. Alternatively, you can implement \
			the get_derivative_ops method instead of init_derivative_ops.")

		self.__H = hamiltonian
		if hasattr(self, '__dim'):
			del self.__dim

	def H(self, *components):
		'''
		This method returns an `Operator` instance representing the Hamiltonian of the
		system. If `components` is specified, only the components of the OperatorSet object listed are included,
		otherwise the Operator object returns its default set. Note that if the
		Hamiltonian object is simply an Operator, the Operator is simply returned
		as is.

		:param components: Sequence of component names.
		:type components: tuple
		'''
		if self.__H is None:
			raise ValueError("Hamiltonian has not been set, and an attempt was made to access it.")
		if isinstance(self.__H, Operator):
			return self.__H
		else:
			return self.__H(*components)

	@property
	def dim(self):
		'''
		The dimension of this `QuantumSystem`; or equivalently the number of basis vectors.
		'''
		try:
			return self.__dim
		except:
			self.__dim = self.H().shape[0]
			return self.__dim

	@property
	def bases(self):
		'''
		A sorted list of the names of the bases configured for this `QuantumSystem` instance.
		'''
		return sorted(self.__named_bases.keys())

	def basis(self, basis):
		'''
		This method returns the `Basis` object associated with the provided name. If `basis` is
		a `Basis` object, then it is simply returned; and if `basis` is None, a `StandardBasis` is
		returned (or if a StandardBasis instance has been added to this instance, then it is returned instead).

		:param basis: A name of a basis, a `Basis` instance, or None.
		'''
		if isinstance(basis, Basis):
			return basis
		if basis in self.__named_bases:
			return self.__named_bases[basis]
		if basis is None:
			if self.__basis_default is not None:
				return self.__named_bases[self.__basis_default]
			else:
				try:
					basis = self.__basis_default_fallback
					if basis.dim == self.dim:
						return basis
				except:
					self.__basis_default_fallback = StandardBasis(dim=self.dim, parameters=self.p)
					return self.__basis_default_fallback
		return ValueError("Unknown basis by name of `%s`" % basis)

	def add_basis(self, name, basis):
		'''
		This method is used to add a basis. The first `StandardBasis` instance
		to be added will become the default basis used to describe this `QuantumSystem`.

		:param name: The name used to reference this basis in the future.
		:type name: str
		:param basis: A `Basis` instance to be associated with the above name.
		:type basis: Basis
		'''
		name = str(name)
		if not isinstance(basis, Basis):
			raise ValueError("Invalid basis object.")
		if self.__basis_default is None and isinstance(basis, StandardBasis):
			self.__basis_default = name
		self.__named_bases[name] = basis

		if self.__dim is None:
			self.__dim = basis.dim

	@property
	def states(self):
		'''
		A sorted list of the names of the states configured for this `QuantumSystem` instance.
		'''
		return sorted(self.__named_states.keys())

	def state(self, state, input=None, output=None, threshold=False, evaluate=True, params={}):
		'''
		This method returns a state vector (numpy array) that is associated with the input `state`.
		As well as retrieving named states from storage, this method also allows basis conversions of
		the state. Note that states can be 1D or 2D (states or ensembles).

		:param state: The input state. If this is a string, this should refer to a named state as in `states`.
		:type state: str or Operator or sequence convertible to numpy array
		:param input: The basis of the input states.
		:type input: str, Basis or None
		:param output: The basis into which this method should convert output states.
		:type output: str, Basis or None
		:param threshold: Parameter to control thresholding (see `Basis.transform` documentation)
		:type threshold: bool or float
		:param evaluate: If input type is Operator, whether the Operator should be numerically evaluated.
		:type evaluate: bool
		:param params: Parameter overrides to use during evaluation and basis transformation.
		:type params: dict
		'''
		if isinstance(state, str):
			state = self.__named_states[state]
			if isinstance(state, Operator) and evaluate:
				state = state(**params)
			if output is None:
				return state
			else:
				return self.basis(output).transform(state, threshold=threshold, params=params)
		else:
			if not isinstance(state, Operator):
				state = np.array(state)
			elif evaluate:
				state = state(**params)
			if input is None and output is None:
				return state
			elif input is None:
				return self.basis(output).transform(state, threshold=threshold, params=params)
			elif output is None:
				return self.basis(input).transform(state, inverse=True, threshold=threshold, params=params)
			else:
				return self.basis(output).transform_from(state, basis=self.basis(input), threshold=threshold, params=params)

	def add_state(self, name, state, basis=None, params={}):
		'''
		This method allows a string name to be associated with a state. This name can then
		be used in `QuantumSystem.state` to retrieve the state. States are assumed to be in
		the basis specified, and are transformed as necessary to be in the standard basis of this
		`QuantumSystem` instance.

		:param name: String name to be associated with this state.
		:type name: str
		:param state: State or ensemble to be recorded.
		:type state: Operator or iterable object convertible to numpy array
		:param basis: The basis of the input `state`.
		:type basis: str, Basis or None
		:param params: The parameter overrides to use during basis transformation.
		:type params: dict
		'''
		# TODO: check dimensions
		if not isinstance(state, Operator):
			state = np.array(state, dtype=complex)  # Force internally stored named states to be complex arrays

		if len(state.shape) > 1:
			if isinstance(state, np.ndarray):
				state /= float(np.sum(np.diag(state)))
			self.__named_ensembles[name] = self.basis(basis).transform(state, params=params, inverse=True) if basis is not None else state
		else:
			if isinstance(state, np.ndarray):
				state /= np.linalg.norm(state)
			self.__named_states[name] = self.basis(basis).transform(state, params=params, inverse=True) if basis is not None else state

	def state_projector(self, state, **kwargs):
		'''
		This method returns a projector onto the state provided. This method is equivalent to:

		>>> system.subspace_projector([state], **kwargs)

		Refer to `subspace_projector` for more information.
		'''
		return self.subspace_projector([state], **kwargs)

	def state_fromString(self, state, input=None, output=None, threshold=False, params={}):
		'''
		This method creates a state object from a string representation, as interpreted by
		the `Basis.state_fromString` method. Otherwise, this method acts as the `QuantumSystem.state`
		method.

		:param state: The string representation of the state.
		:type state: str
		:param input: The basis to use to interpret the string.
		:type input: str, Basis or None
		:param output: The basis into which this method should convert output states.
		:type output: str, Basis or None
		:param threshold: Parameter to control thresholding (see `Basis.transform` documentation)
		:type threshold: bool or float
		:param params: Parameter overrides to use during evaluation and basis transformation.
		:type params: dict
		'''
		return self.state(self.basis(input).state_fromString(state, params), input=input, output=output, threshold=threshold, params=params, evaluate=True)

	def state_toString(self, state, input=None, output=None, threshold=False, params={}):
		'''
		This method create a string representation of a state object, using `Basis.state_toString`. Otherwise,
		this method acts like `QuantumSystem.state`.

		:param state: The input state. If this is a string, this should refer to a named state as in `states`.
		:type state: str or Operator or sequence convertible to numpy array
		:param input: The basis of the input states.
		:type input: str, Basis or None
		:param output: The basis into which this method should convert output states, and which should be
			used to create the string representation.
		:type output: str, Basis or None
		:param threshold: Parameter to control thresholding (see `Basis.transform` documentation)
		:type threshold: bool or float
		:param params: Parameter overrides to use during evaluation and basis transformation.
		:type params: dict

		Converts a state object to its string representation, as interpreted by the
		Basis.state_toString method. As with QuantumSystem.state, basis conversions can also be
		done.
		'''
		return self.basis(output).state_toString(self.state(state, input=input, output=output, threshold=threshold, params=params, evaluate=True), params)

	@property
	def ensembles(self):
		'''
		A sorted list of names that are associated with ensembles. Ensembles are added using
		`QuantumSystem.add_state`, where a 2D state is automatically identified as an ensemble.
		'''
		return sorted(self.__named_ensembles.keys())

	def ensemble(self, state, input=None, output=None, threshold=False, evaluate=True, params={}):
		'''
		This method returns a ensemble 2D vector (numpy array) that is associated with the input `state`.
		If the state associated with `state` is not an ensemble, then the outer product is taken and returned.
		Just like `QuantumSystem.state`, this method allows basis conversion of states.

		:param state: The input state. If this is a string, this should refer to a named state as in `states` or `ensembles`.
		:type state: str or Operator or sequence convertible to numpy array
		:param input: The basis of the input states.
		:type input: str, Basis or None
		:param output: The basis into which this method should convert output states.
		:type output: str, Basis or None
		:param threshold: Parameter to control thresholding (see `Basis.transform` documentation)
		:type threshold: bool or float
		:param evaluate: If input type is Operator, whether the Operator should be numerically evaluated.
		:type evaluate: bool
		:param params: Parameter overrides to use during evaluation and basis transformation.
		:type params: dict
		'''
		if isinstance(state, str):
			if state in self.__named_ensembles:
				state = self.__named_ensembles[state]
				return self.state(state, output=output, threshold=threshold, params=params, evaluate=evaluate)
			elif state in self.__named_states:
				state = self.__named_states[state]
			else:
				raise KeyError("No such state '%s' known." % state)
		state = self.state(state, input=input, output=output, threshold=threshold, params=params, evaluate=evaluate)
		if len(state.shape) == 1:
			return np.outer(np.array(state).conjugate(), state)
		return state

	@property
	def subspaces(self):
		'''
		A sorted list of named subspaces. The subspaces can be extracted using the `QuantumSystem.subspace`
		method.
		'''
		return sorted(self.__named_subspaces.keys())

	def subspace(self, subspace, input=None, output=None, threshold=False, evaluate=True, params={}):
		'''
		This method returns the subspace associated with the input `subspace`. A subspace is a list of
		states (but *not* ensembles). The subspace is the span of the states. Otherwise, this method
		acts like `QuantumSystem.state`.

		:param subspace: The input subspace. If this is a string, this should refer to a named state as in `states`.
		:type subspace: str or list of str, Operators or sequences convertible to numpy array
		:param input: The basis of the input states.
		:type input: str, Basis or None
		:param output: The basis into which this method should convert output states.
		:type output: str, Basis or None
		:param threshold: Parameter to control thresholding (see `Basis.transform` documentation)
		:type threshold: bool or float
		:param evaluate: If input type is Operator, whether the Operator should be numerically evaluated.
		:type evaluate: bool
		:param params: Parameter overrides to use during evaluation and basis transformation.
		:type params: dict
		'''
		states = []
		if isinstance(subspace, str):
			subspace = self.__named_subspaces[subspace]
			for state in subspace:
				states.append(self.state(state, output=output, threshold=threshold, params=params, evaluate=evaluate))
		else:
			for state in subspace:
				states.append(self.state(state, input=input, output=output, threshold=threshold, params=params, evaluate=evaluate))
		return states

	def add_subspace(self, name, subspace, basis=None, params={}):
		'''
		This method adds a new association between a string and a sequence of states (which
		can be in turn be the names of named states).

		:param name: The name to associate with this subspace.
		:type name: str
		:param subspace: The input subspace to be stored.
		:type subspace: str or list of str, Operators or sequences convertible to numpy array
		:param basis: The basis in which the subspace is represented.
		:type basis: str, Basis or None
		:param params: The parameter overrides to use during basis transformation.
		:type params: dict
		'''
		# TODO: Check dimensions
		fstates = []
		for state in subspace:
			fstates.append(self.state(state, input=basis, params=params, evaluate=False))
		self.__named_subspaces[name] = fstates

	def subspace_projector(self, subspace, input=None, output=None, invert=False, threshold=False, evaluate=True, params={}):
		'''
		This method returns a projector onto the provided subspace (which can be the name of a named subspace, or a list of
		named states. If `invert` is `True`, then the projector returned is onto the subspace orthogonal to the provided one.

		:param subspace: The input subspace. If this is a string, this should refer to a named state as in `states`.
		:type subspace: str or list of str, Operators or sequences convertible to numpy array
		:param input: The basis of the provided subspace.
		:type input: str, Basis or None
		:param output: The basis into which this method should convert output subspaces.
		:type output: str, Basis or None
		:param invert: `True` if the projector returned should be onto the subspace, and `False` if the projector should be off the subspace.
		:type invert: bool
		:param threshold: Parameter to control thresholding (see `Basis.transform` documentation)
		:type threshold: bool or float
		:param evaluate: If input type is Operator, whether the Operator should be numerically evaluated.
		:type evaluate: bool
		:param params: Parameter overrides to use during evaluation and basis transformation.
		:type params: dict
		'''
		if not evaluate:
			raise ValueError("Symbolic subspace projector not yet implemented.")

		states = self.subspace(subspace, input=input, output=output, threshold=threshold, params=params, evaluate=evaluate)

		P = 0
		for state in states:

			state = np.array(state) / np.linalg.norm(state)
			state.shape = (len(state), 1)

			dP = state.dot(state.transpose().conjugate())
			P += dP / np.trace(dP)

		if invert:
			P = np.identity(len(states[0])) - P

		return P

	def derivative_ops(self, ops=None, components=None):
		'''
		This method returns a dictionary of named `StateOperator` objects, which are used
		to calculate the instantaneous time derivative by `QuantumSystem.integrate` and
		related methods. This method picks upon the operators created by `init_derivative_ops`
		and `get_derivative_ops`, as well as the default 'evolution' operator which describes
		evolution under the Schroedinger equation.

		:param ops: A sequence of operators to include in the returned dictionary.
		:type ops: iterable of strings
		:param components: A sequence of component names to enable in the `OperatorSet`
			describing the Hamiltonian (if applicable).
		:type components: iterable of strings
		'''
		o = self.__get_derivative_ops(components)
		if ops is None:
			return o
		for k in o.keys():
			if k not in ops:
				o.pop(k)
		for key in ops:
			if key not in o:
				raise ValueError("Unknown operator: %s" % key)
		return o

	def add_derivative_op(self, name, state_op):
		'''
		This method adds an association between a string `name` and a `StateOperator` object,
		for use as a derivative operator.

		:param name: The name to be used to refer to the provided `StateOperator`.
		:type name: str
		:param state_op: The `StateOperator` to be stored.
		:type state_op: StateOperator
		'''
		if not isinstance(state_op, StateOperator):
			raise ValueError("Supplied operator must be an instance of StateOperator.")
		state_op.p = self.p
		self.__derivative_ops[name] = state_op

	def __get_derivative_ops(self, components=None):
		ops = {}
		if components is None:
			components = tuple()
		ops["evolution"] = SchrodingerStateOperator(parameters=self.p, H=self.H(*components))
		ops.update(self.__derivative_ops)
		ops_user = self.get_derivative_ops(components=components)
		if ops_user is not None:
			if type(ops_user) is not dict:
				raise ValueError("Unknown type returned for: get_derivative_ops()")
			else:
				for name, op in ops_user.items():
					op.p = self.p
					ops[name] = op
		return ops

	@property
	def default_derivative_ops(self):
		'''
		The list (or sequence) of names corresponding to the derivative operators that will be used by default
		(if no operator names are otherwise supplied).
		'''
		return self.__derivative_ops_default
	@default_derivative_ops.setter
	def default_derivative_ops(self, default):
		self.__derivative_ops_default = default



	@property
	def measurements(self):
		'''
		A sorted list of the names of the measurements associated with this `QuantumSystem`.
		'''
		return sorted(self.measure._names)

	def add_measurement(self, name, measurement):
		'''
		This method add a `Measurement` instance to the `Measurements` container associated with this
		Quantum System object. It can then be called using:

		>>> system.measure.<name>

		See `Measurements` for more information.
		'''
		self.measure._add(name, measurement)



	# Utility methods

	def Operator(self, components, basis=None, exact=False):
		'''
		This method is a shorthand way of creating an `Operator` instance that shares
		the same `Parameters` instance as this `QuantumSystem`. This is shorthand for:

		>>> Operator(components, parameters=self.p, basis=self.basis(basis) if basis is not None else None, exact=exact)

		For more documentation, refer to `Operator`.
		'''
		return Operator(components, parameters=self.p, basis=self.basis(basis) if basis is not None else None, exact=exact)

	def OperatorSet(self, operatorMap, defaults=None):
		'''
		This method is a shorthand way of creating an `OperatorSet` instance. This method first
		checks through the values of `operatorMap` and converts anything that is not an `Operator`
		to an `Operator` using:

		>>> system.Operator(operatorMap[key])

		Then, it creates an OperatorSet instance:

		>>> OperatorSet(operatorMap, defaults=defaults)

		For more documentation, see `Operator` and `OperatorSet`.
		'''
		for key in operatorMap.keys():
			if not isinstance(operatorMap[key], Operator):
				operatorMap[key] = self.Operator(operatorMap[key])

		return OperatorSet(operatorMap, defaults=defaults)

	def show(self):
		'''
		This method prints (to stdout) a basic overview of the `QuantumSystem` that includes:
		- the dimension of the model
		- the representation of the `Parameters` instance
		- the names of any bases
		- the names of any states
		- the names of any ensembles
		- the names of any subspaces
		- the names of any derivative operators
		'''
		title = "%s Quantum System" % type(self).__name__
		print "-" * len(title)
		print title
		print "-" * len(title)
		print "\tDimension: %s" % self.dim
		print "\tParameters: %s" % self.p
		print "\tBases: %s" % sorted(self.bases)
		print "\tStates: %s" % sorted(self.states)
		print "\tEnsembles: %s" % sorted(self.ensembles)
		print "\tSubspaces: %s" % sorted(self.subspaces)
		print "\tDerivative Operators: %s" % self.derivative_ops().keys()

	############# INTEGRATION CODE #########################################

	# Integrate hamiltonian forward to describe state at time t ( or times [t_i])
	def integrate(self, times, initial, **kwargs):
		'''
		This method constructs an `Integrator` instance with initial states defined as `initial` using
		`get_integrator`, and then calls `start` on that instance with times specified as `times`.

		:param times: The times for which to return the instantaneous state. All values are
			passed through the `Parameters` instance, allowing for times to be expressions
			of parameters.
		:type times: iterable of floats or str
		:param initial: A sequence of initial states (or ensembles).
		:type initial: list or tuple of numpy arrays

		This method is equivalent to:
		>>> system.get_integrator(initial=initial, **kwargs).start(times)

		For more documentation, see `get_integrator` and `Integrator.start`.
		'''
		return self.get_integrator(initial=initial, **kwargs).start(times)

	def get_integrator(self, initial=None, input=None, output=None, threshold=False, components=None, operators=None, time_ops={}, params={}, integrator='RealIntegrator', **kwargs):
		'''
		This method is shorthand for manually creating an `Integrator` instance. It converts all operators and states
		into a consistent basis and form for use in the integrator.

		:param initial: A sequence of initial states to use (including string names of known states and ensembles; see `QuantumSystem.state`).
		:type initial: iterable
		:param input: Basis of input states (including string name of stored Basis; see `QuantumSystem.basis`).
		:type input: str, Basis or None
		:param output: Basis to use during integration (including string name of stored Basis; see `QuantumSystem.basis`)
		:type output: str, Basis or None
		:param threshold: Parameter to control thresholding during basis transformations (see `Basis.transform`).
		:type threshold: bool or float
		:param components: A sequence of component names to enable for this `Integrator` (see `QuantumSystem.H`).
		:type components: iterable of str
		:param operators: A sequence of operator names to use during integration (see `QuantumSystem.derivative_ops`).
			Additional operators can be added using `Integrator.add_operator` on the returned `Integrator` instance.
		:type operators: iterable of str
		:param time_ops: A dictionary of `StateOperator` instances, or a two-tuple of a StateOperator subclass
			with a dictionary of arguments to pass to the constructor when required, indexed by times (see `Integrator.time_ops`).
			The two-tuple specification is for use with multithreading, where a `StateOperator` instance may not be so easily
			picked. The `StateOperator` instance is initialised before being pass to an `Integrator` instance.
		:type time_ops: dict
		:param params: Parameter overrides to use during basis transformations and integration.
		:type params: dict
		:param integrator: The integrator class to use as a Class; either as a string (in which case it is
			imported from `qubricks.wall`) or as a Class to be instantiated.
		:type integrator: str or classobj
		:param kwargs: Additional keyword arguments to pass to `Integrator` constructor.
		:type kwargs: dict
		'''

		ops = self.__integrator_operators(components=components, operators=operators, basis=output, threshold=threshold)

		for time, op in time_ops.items():
			if not isinstance(op, StateOperator):
				if type(op) == tuple and len(op) == 2 and issubclass(op[0], StateOperator) and type(op[1]) == dict:
					op = op[0](self.p, **op[1])
				else:
					raise ValueError("Invalid StateOperator specification provided.")
			if not (op.basis is None or isinstance(op.basis, Basis)):
				op.basis = self.basis(op.basis)
			time_ops[time] = op.change_basis(basis=self.basis(output), threshold=threshold)
			time_ops[time].p = self.p

		use_ensemble = self.use_ensemble(ops, ensemble=True in [len(np.array(s).shape) == 2 for s in initial])
		# Prepare states
		y_0s = []
		for psi0 in initial:
			if use_ensemble is True:
				y_0s.append(self.ensemble(psi0, input=input, output=output, threshold=threshold))
			else:
				y_0s.append(self.state(psi0, input=input, output=output, threshold=threshold))

		if type(integrator) is types.ClassType:
			IntegratorClass = integrator
		elif type(integrator) is str:
			try:
				IntegratorClass = getattr(importlib.import_module('qubricks.wall'),integrator)
			except AttributeError:
				raise ValueError("Could not find integrator class named '%s' in `qubricks.wall`." % integrator)
		else:
			raise ValueError("QuantumSystem does not know how to convert '%s' of type '%s' to an Integrator instance." % (integrator,type(integrator)))

		return IntegratorClass(parameters=self.p, initial=y_0s, operators=ops, params=params, time_ops=time_ops, **kwargs)

	def __integrator_operators(self, components=None, operators=None, basis=None, threshold=False):
		'''
		An internal method that generates the operators to be used by the integrator. In this method
		derivative StateOperators are converted into the appropriate basis.
		'''
		if basis is not None:
			basis = self.basis(basis)

		ops = []

		if operators is None:
			operators = self.default_derivative_ops

		for _, operator in self.derivative_ops(ops=operators, components=components).items():

			if not (operator.basis is None or isinstance(operator.basis, Basis)):
				operator.basis = self.basis(operator.basis)
			op = operator.change_basis(basis, threshold=threshold)

			ops.append(op)

		return ops

	def use_ensemble(self, ops=None, ensemble=False):
		'''
		This method is used to check whether integration should proceed using
		ensembles or state vectors. It first checks which kind of evolution is
		supported by all of the `StateOperators`, as heralded by the `StateOperator.for_state`
		and `StateOperator.for_ensemble` methods. It then checks that there is a match
		between the type of state being used and the type of states supported
		by the operators. Note that if `ensemble` is `False`, indicating that the
		state is a ordinary state vector, then it is assumed that if this method returns
		`True`, indicating ensemble evolution is to be used, that the states will be
		converted using the outer-product with themselves to ensembles. Note that this
		is already done by `get_integrator`.

		:param ops: A sequence of Operator objects or names of Operators. The list can be mixed.
		:type ops: iterable
		:param ensemble: `True` one or more of the input states are to be ensembles. `False` otherwise.
		:type ensemble: bool
		'''

		op_state = True
		op_ensemble = True

		if ops is None:
			ops = self.default_derivative_ops
		if isinstance(ops[0], str):
			ops = self.derivative_ops(ops=ops).values()

		for operator in ops:
			op_state = op_state and operator.for_state
			op_ensemble = op_ensemble and operator.for_ensemble

		if not ensemble:
			if op_state:
				return False
			if op_ensemble:
				return True
		else:
			if op_ensemble:
				return True

		raise RuntimeError("The StateOperators specified for use do not collectively support the type of state required for time evolution.")



# Cheekily import some classes from wall
from .measurement import Measurements
from .wall import SchrodingerStateOperator, StandardBasis
