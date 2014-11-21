from abc import ABCMeta, abstractmethod, abstractproperty

from parameters import Parameters

import numpy as np

from . import operators
from .basis import Basis,StandardBasis
from .measurement import Measurements
from .operators import Operator, OperatorSet, StateOperator


try:
	import sage
	from .integrator import SageIntegrator as Integrator
except ImportError:
	from .integrator import RealIntegrator as Integrator

class QuantumSystem(object):
	'''
	QuantumSystem (parameters=None,**kwargs)

	QuantumSystem is an abstract class (which cannot therefore be instantiated)
	which is designed to be subclassed to describe particular quantum systems.

	Parameters
	----------
	parameters : An object which is used to initialise the parameters used
		by this object. It can be:
		- A string, in which case it is assumed to be a filename, and
		  an attempt is made to import the parameters from that filename.
		- A Parameters object
		- None; in which case a default parameters object is constructed/
	**kwargs : Any further keyword arguments are passed onto the
		setup_environment(**kwarg) method.

	'''
	__metaclass__ = ABCMeta

	def __init__(self, parameters=None, **kwargs):

		self.__derivative_ops = {}

		self.__named_states = {}  # A dictionary of named states for easy recollection
		self.__named_ensembles = {}  # A dictionary of named ensembles for density states
		self.__named_subspaces = {}  # A dictionary of named subspaces for easy identification
		self.__named_bases = {}  # A dictionary of named bases for state representation
		self.__basis_default = None

		self.setup_environment(**kwargs)

		if isinstance(parameters, str):
			self.set_parameters(Parameters.load(parameters, constants=True))
		elif isinstance(parameters, Parameters):
			self.set_parameters(parameters)
		else:
			self.set_parameters(Parameters(constants=True))

		if isinstance(parameters,dict):
			self.p << parameters

		self.setup_parameters()

		self.set_hamiltonian(self.setup_hamiltonian())

		self.dim = self.H().shape[0]

		self.add_basis(StandardBasis(name="default", dim=self.dim, parameters=self.p))
		self.setup_bases()

		self.setup_states()

		self.measure = Measurements(self)
		self.setup_measurements()

	@abstractmethod
	def setup_environment(self, **kwargs):
		'''
		Configure any custom properties/attributes using kwargs passed
		to __init__.
		'''
		raise NotImplementedError

	@abstractmethod
	def setup_parameters(self):
		'''
		After the QuantumSystem parameter initialisation routines
		are run, check that the parameters are initialised correctly.
		'''
		raise NotImplementedError

	@abstractmethod
	def setup_bases(self):
		'''
		Add the bases which are going to be used by this QuantumSystem
		instance.
		'''
		raise NotImplementedError

	@abstractmethod
	def setup_hamiltonian(self):
		'''
		Initialise the Hamiltonian to be used by this QuantumSystem
		instance.
		'''
		raise NotImplementedError

	@abstractmethod
	def setup_states(self):
		'''
		Add the named/important states to be used by this quantum system.
		'''
		raise NotImplementedError

	@abstractmethod
	def setup_measurements(self):
		'''
		Add the measurements to be used by this quantum system instance.
		'''
		raise NotImplementedError

	@abstractproperty
	def default_derivative_ops(self):
		raise NotImplementedError

	def get_derivative_ops(self, components=None):
		'''
		Setup the derivative operators to be implemented on top of the
		basic quantum evolution operator.
		'''
		pass

	def __get_derivative_ops(self, components=None):
		ops = {}
		if components is None:
			components = tuple()
		ops["evolution"] = operators.SchrodingerOperator(self.p, H=self.H(*components))
		ops.update(self.__derivative_ops)
		ops_user = self.get_derivative_ops(components=components)
		if ops_user is not None:
			if type(ops_user) is not dict:
				raise ValueError("Unknown type returned for: get_derivative_ops()")
			else:
				for name,op in ops_user.items():
					op.p = self.p
					ops[name] = op
		return ops

	############# CONFIGURATION ############################################

	def set_parameters(self, p):
		'''
		Set the parameters reference. This can be retroactively applied after QuantumSystem
		object creation, in the event that a Parameters instance was not provided when it
		was first initialised.
		'''
		self.__parameters = p

	def set_hamiltonian(self, h):
		'''
		Set the Hamiltonian object. This can be retroactively applied after QuantumSystem
		object creation, if one changes one's mind about which Hamiltonian to use.
		'''
		self.__H = h

	def add_measurement(self, name, measurement):
		'''
		Add a Measurement instance to the Measurements container associated with this
		Quantum System object. It can then be called using:
		quantumsystem.measure.<name>
		'''
		self.measure._add(name, measurement)

	def add_derivative_op(self, name, state_op):
		if not isinstance(state_op,StateOperator):
			raise ValueError, "Supplied operator must be an instance of StateOperator."
		state_op._on_attach_to_system(self)
		self.__derivative_ops[name] = state_op

	def add_basis(self, basis):
		'''
		Add a basis to this instance of QuantumSystem. The name used to recall this basis
		is given by Basis.name .
		'''
		if not isinstance(basis, Basis):
			raise ValueError, "Invalid basis object."
		if self.__basis_default is None:
			self.__basis_default = basis.name
		self.__named_bases[basis.name] = basis

	def add_state(self, name, state, basis=None, params={}):
		'''
		Add a named state, provided in some basis `basis`.
		'''
		# TODO: check dimensions
		if not isinstance(state,Operator):
			state = np.array(state,dtype=complex) # Force internally stored named states to be complex arrays

		if len(state.shape) > 1:
			if isinstance(state,np.ndarray):
				state /= float(np.sum(np.diag(state)))
			self.__named_ensembles[name] = self.basis(basis).transform(state, params=params, inverse=True) if basis is not None else state
		else:
			if isinstance(state,np.ndarray):
				state /= np.linalg.norm(state)
			self.__named_states[name] = self.basis(basis).transform(state, params=params, inverse=True) if basis is not None else state

	def add_subspace(self, name, subspace, basis=None, params={}):
		'''
		Add a named subspace, with name `name`. The subspace `subspace` should be provided
		as a list of subspace in a basis `basis`.
		'''
		# TODO: Check dimensions
		fstates = []
		for state in subspace:
			fstates.append(self.state(state, input=basis, params=params,evaluate=False))
		self.__named_subspaces[name] = fstates

	def Operator(self, components, basis=None):
		'''
		A shorthand method for creating an Operator for use in whichever way is appropriate.
		This will create an Operator that shares the same parameters instance as the
		QuantumSystem object; and will ensure the basis is actually a Basis object.
		'''
		return Operator(components, parameters=self.p, basis=self.basis(basis))

	def OperatorSet(self, operatorMap, defaults=None):
		'''
		A shorthand method for creating an Operator for use in whichever way is appropriate.
		Operator objects should be initialised before calling this method. If something else
		is provided instead, QuantumSystem.Operator will be called to create the Operator
		objects. In such a case, the basis is expected to be the default basis of this
		QuantumSystem object.
		'''
		for key in operatorMap.keys():
			if not isinstance(operatorMap[key], Operator):
				operatorMap[key] = self.Operator(operatorMap[key])

		return OperatorSet(operatorMap, defaults=defaults)

	############# INTERROGATION ############################################

	@property
	def p(self):
		'''
		Returns a reference to the internal Parameter object.
		'''
		return self.__parameters

	def H(self, *components):
		'''
		Returns an Operator instance representing the Hamiltonian, which is assumed
		to be written in the "standard basis"; which can represent any basis you like.
		All other basis transformations will be relative to this one. If `components`
		is specified, only the components of the OperatorSet object listed are included,
		otherwise the Operator object returns its default set. Note that if the
		Hamiltonian object is simply an Operator, the Operator is simply returned
		as is.
		'''
		if isinstance(self.__H, Operator):
			return self.__H
		else:
			return self.__H(*components)

	@property
	def bases(self):
		'''
		Returns a list of the names of the bases configured for this QuantumSystem object.
		'''
		return sorted(self.__named_bases.keys())

	def basis(self, basis):
		'''
		Returns the Basis object associated with a name. If a Basis object is provided, this
		method passes that result through.
		'''
		if isinstance(basis, Basis):
			return basis
		return self.__named_bases.get(basis if basis is not None else self.__basis_default)

	@property
	def states(self):
		'''
		Returns a list of the names of the states configured for this QuantumSystem object.
		'''
		return sorted(self.__named_states.keys())

	def state(self, state, input=None, output=None, threshold=False, evaluate=True, params={}):
		'''
		Returns a state vector (numpy array) that is associated with the input state. This
		method allows for basis conversions of states, with `input` being the name of the basis
		(or the Basis object itself) in which `state` is represented; and `output` being
		the desired basis of representation. `threshold` is used as defined in Basis.transform,
		and allows numerical noise in transformations to be suppressed. `params` can be used
		to determine the parameters at which the bases are evaluated.
		'''
		if isinstance(state, str):
			state = self.__named_states[state]
			if isinstance(state,Operator) and evaluate:
				state = state(**params)
			if output is None:
				return state
			else:
				return self.basis(output).transform(state, threshold=threshold, params=params)
		else:
			if not isinstance(state,Operator):
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

	def state_fromString(self, state, input=None, output=None, threshold=False, params={}):
		'''
		Converts a string object to state vector, as interpreted by the Basis.state_fromString method.
		As with QuantumSystem.state, basis conversions can also be done.
		'''
		return self.state(self.basis(input).state_fromString(state, params), input=input, output=output, threshold=threshold, params=params, evaluate=True)

	def state_toString(self, state, input=None, output=None, threshold=False, params={}):
		'''
		Converts a state object to its string representation, as interpreted by the
		Basis.state_toString method. As with QuantumSystem.state, basis conversions can also be
		done.
		'''
		return self.basis(output).state_toString(self.state(state, input=input, output=output, threshold=threshold, params=params, evaluate=True), params)

	@property
	def ensembles(self):
		'''
		Returns a list of the names of the ensembles configured for this QuantumSystem object.
		'''
		return sorted(self.__named_ensembles.keys())

	def ensemble(self, state, input=None, output=None, threshold=False, evaluate=True, params={}):
		'''
		Returns a state matrix (numpy array) that is associated with the input state. This
		method allows for basis conversions of states, with `input` being the name of the basis
		(or the Basis object itself) in which `state` is represented; and `output` being
		the desired basis of representation. `threshold` is used as defined in Basis.transform,
		and allows numerical noise in transformations to be suppressed. `params` can be used
		to determine the parameters at which the bases are evaluated. `state` can be a named
		ensemble, a 2D array, or a 1D vector, in which case the self outer product will be taken.
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
		Returns a list of the names of the subspaces configured for this QuantumSystem object.
		'''
		return sorted(self.__named_subspaces.keys())

	def subspace(self, subspace, input=None, output=None, threshold=False, evaluate=True, params={}):
		'''
		Returns a list of basis states that span the subspace. Basis conversions are also possible; as
		described in QuantumSystem.state. `subspace` can be a named subspace, or a list of states.
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

	def subspace_projector(self, subspace, input=None, output=None, invert=False, threshold=False, evaluate=True, params={}):
		'''
		Returns a projector onto a subspace. Subspace can be any valid input into QuantumSystem.subspace; including
		any output of that function. Basis conversions are possible as for QuantumSystem.state.
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

	def state_projector(self, state, **kwargs):
		return self.subspace_projector([state],**kwargs)

	def derivative_ops(self, ops=None, components=None):
		'''
		Returns a dictionary of StateOperator objects, which are used to calculate the
		instantaneous derivative. These are used in the QuantumSystem.integrate (and
		related) methods. By default, a SchrodingerOperator based around the Hamiltonian
		is created, and called "evolution".
		'''
		o = self.__get_derivative_ops(components)
		if ops is None:
			return o
		for k in o.keys():
			if k not in ops:
				o.pop(k)
		return o

	def show(self):
		'''
		Returns a string overview of this QuantumSystem object.
		'''
		title = "%s Quantum System" % type(self).__name__
		print "-"*len(title)
		print title
		print "-"*len(title)
		print "\tDerivative Operators: %s" % self.derivative_ops
		print "\tBases: %s" % sorted(self.bases)
		print "\tSubspaces: %s" % sorted(self.subspaces)
		print "\tStates: %s" % sorted(self.states)
		print "\tEnsembles: %s" % sorted(self.ensembles)
		print "\tParameters: %s" % self.p

	############# INTEGRATION CODE #########################################

	def use_ensemble(self, ops=None):
		'''
		Returns False if all derivative operators support standard state-vector
		evolution; and True if all support ensemble evolution, and do not all
		support state-vector evolution. In the event that the operators do not
		agree on at least one of state-vector or ensemble evolution, a
		RuntimeError is raised.
		'''

		state = True
		ensemble = True

		if ops is None:
			ops = self.default_derivative_ops
		if isinstance(ops[0], str):
			ops = self.derivative_ops(ops=ops).values()

		for operator in ops:
			state = state and operator.for_state
			ensemble = ensemble and operator.for_ensemble

		if state:
			return False
		if ensemble:
			return True

		raise RuntimeError, "No possible configuration found that supports all operators."

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
			if basis is not None:
				op = operator.change_basis(basis,threshold=threshold)
			else:
				op = operator

			ops.append(op)

		return ops

	# y_0s=None,operators=None,solver='rkf45',error_rel=1e-8,error_abs=1e-8,time_ops={},callback=None,callback_fallback=True
	def get_integrator(self, initial, input=None, output=None, threshold=False, components=None, operators=None, time_ops={}, params={}, **args):
		'''
		Returns an Integrator object, which is configured and ready to start integrating according to the parameters
		passed to this method. `initial` is a list of starting vectors/ensembles. `input` is the basis in which the
		`initial` states are represented, and `output` is the basis in which the integration routines should be run.
		`threshould` is as defined in Basis.transform. `components` is the list of string names which should be activated
		if the Hamiltonian is an OperatorSet object. `operators` is a list of strings (and/or StateOperator objects) identifying
		which derivative operators should be used in this integration. `params` is a list of parameter overrides which will be
		used unless further overridden, as in Measurement.iterate . All other arguments are passed onto the Integrator object.
		'''

		ops = self.__integrator_operators(components=components, operators=operators, basis=output, threshold=threshold)

		for time,op in time_ops.items():
			time_ops[time] = op.change_basis(basis=self.basis(output),threshold=threshold)._on_attach_to_system(self)

		use_ensemble = self.use_ensemble(ops) or True in [len(np.array(s).shape) == 2 for s in initial]
		# Prepare states
		y_0s = []
		for psi0 in initial:
			if use_ensemble is True:
				y_0s.append(self.ensemble(psi0, input=input, output=output, threshold=threshold))
			else:
				y_0s.append(self.state(psi0, input=input, output=output, threshold=threshold))

		return Integrator(parameters=self.p, initial=y_0s, operators=ops, op_params=params, time_ops=time_ops, **args) #

	# Integrate hamiltonian forward to describe state at time t ( or times [t_i])
	def integrate(self, t, psi0s, **kwargs):
		'''
		This is a shorthand notation for initialising the Integrator (as in QuantumSystem.get_integrator); and then running the
		Integrator object. The value returned from the integration routine is returned here.
		'''
		return self.get_integrator(initial=psi0s, **kwargs).start(t)
