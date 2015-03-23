from abc import ABCMeta, abstractmethod
import math
import sys
import types
import warnings

from sympy.core.cache import clear_cache as sympy_clear_cache

import numpy as np
from parameters import Parameters

from .operator import Operator
from .stateoperator import StateOperator


class IntegratorCallback(object):

	#
	# Trigger start
	def onStart(self):
		pass

	#
	# Receives a float value in [0,1].
	def onProgress(self, progress, identifier=None):
		pass

	#
	# Called when function is complete.
	# Level = 0 -> OKAY
	# Level = 1 -> WARN
	# Level = 2 -> ERROR
	# Status = string message
	def onComplete(self, identifier=None, message=None, status=0):
		pass

# Try to use the full implementation of progress bar
try:
	import progressbar as pbar

	class ProgressBarCallback(IntegratorCallback):

		def onStart(self):
			self.pb = pbar.ProgressBar(widgets=['\033[0;34m', pbar.Percentage(), ' ', pbar.Bar(), '\033[0m'])
			self.pb.start()
			self._progress = 0

		def onProgress(self, progress, identifier=None):
			progress *= 100
			if (progress > self._progress):
				self._progress = math.ceil(progress)
				self.pb.update(min(100, progress))

		def onComplete(self, identifier=None, message=None, status=0):
			self.pb.finish()
except:
	class ProgressBarCallback(IntegratorCallback):

		def onStart(self):
			self._progress = 0

		def onProgress(self, progress, identifier=None):
			progress *= 100
			# if (progress > self._progress):
			self._progress = math.ceil(progress)
			sys.stderr.write("\r%3d%%" % min(100, progress))

		def onComplete(self, identifier=None, message=None, status=0):
			print "\n"
			if message:
				print "%s:" % identifier, message


class Progress(object):
	def __init__(self):
		self.run = 0
		self.runs = 1
		self.callback = None


class Integrator(object):
	'''
	
	'''
	__metaclass__ = ABCMeta

	def __init__(self,
				identifier=None,
				initial=None,
				t_offset=0, 
				operators=None, 
				parameters=None, 
				params={}, 
				error_rel=1e-8, 
				error_abs=1e-8, 
				time_ops={}, 
				progress=False, 
				**kwargs):
		
		self.identifier = identifier
		
		# Set up initial conditions
		self.initial = initial
		self.t_offset = t_offset

		# Parameter Object
		self.p = parameters
		
		# Set solver and tolerances
		self.error_abs = error_abs
		self.error_rel = error_rel

		# Set up integration operator
		self.operators = operators
		self.params = params

		# Set up intermediate pulse operators
		self.time_ops = time_ops

		# Set up results cache
		self.results = None

		self.progress_callback = progress
		
		self.int_kwargs = kwargs
	
	@property
	def p(self):
		'''
		A reference to the Parameters instance used by this object.
		'''
		if self.__p is None:
			raise ValueError("Parameters instance required by Integrator object, but a Parameters object has not been configured.")
		return self.__p
	@p.setter
	def p(self, parameters):
		if parameters is not None and not isinstance(parameters, Parameters):
			raise ValueError("Parameters reference must be an instance of Parameters or None.")
		self.__p = parameters
	
	########### CONFIGURATION ##############################################
	
	@property
	def initial(self):
		'''
		The initial state. Can be set using:
		
		>>> integrator.initial = <list of states>
		'''
		return self.__initial
	@initial.setter
	def initial(self, initial):
		self.__initial = initial
	
	@property
	def t_offset(self):
		'''
		The initial time to use in the integration. Ignored in `Integrator.extend`.
		Can be set using:
		
		>>> integrator.t_offset = <time>
		'''
		return self.__t_offset
	@t_offset.setter
	def t_offset(self, t_offset):
		self.__t_offset = t_offset
	
	@property
	def error_rel(self):
		'''
		The maximum relative error permitted in the integrator. This can be set using:
		
		>>> integrator.error_rel = <float>
		'''
		return self.__error_rel
	@error_rel.setter
	def error_rel(self, error_rel):
		self.__error_rel = error_rel
	
	@property
	def error_abs(self):
		'''
		The maximum absolute error permitted in the integrator. This can be set using:
		
		>>> integrator.error_abs = <float>
		'''
		return self.__error_abs
	@error_abs.setter
	def error_abs(self, error_abs):
		self.__error_abs = error_abs
	
	@property
	def results(self):
		'''
		The currently stored results. Used to continue integration in `Integrator.extend`. While
		it is possible to overwrite the results, the new value is not checked, and care should be taken.
		
		>>> integrator.results = <valid results object>
		'''
		return self.__results
	@results.setter
	def results(self, results):
		self.__results = results
	
	def reset(self):
		'''
		This method resets the internal stored `Integrator.results` to `None`, effectively resetting
		the `Integrator` object to its pre-integration status.
		
		>>> integrator.reset()
		'''
		self.results = None
	
	@property
	def time_ops(self):
		'''
		A reference to the dictionary of time operators. Can be updated directly by adding to the dictionary, or 
		(much more safely) using:
		
		>>> self.time_ops = {'T': <StateOperator>, ...}
		
		The above is shorthand for:
		
		>>> for time, time_op in {'T': <StateOperator>, ...}:
		        self.add_time_op(time, time_op) 
		'''
		return self.__time_ops
	@time_ops.setter
	def time_ops(self, time_ops):
		self.__time_ops = {}
		for time, time_op in time_ops.items():
			self.add_time_op(time, time_op)
	
	def add_time_op(self, time, time_op):
		'''
		This method adds a time operator `time_op` at time `time`. Note that there can only be
		one time operator for any given time. A "time operator" is just a `StateOperator` 
		'''
		if not isinstance(time_op, StateOperator):
			raise ValueError("Time operator must be an instance of State Operator.")
		self.__time_ops[time] = time_op
	
	def get_time_ops(self, indices=None):
		'''
		This method returns the time_ops of `Integrator.time_ops` restricted to
		the indicies specified (using `StateOperator.restrict`).
		
		This is used internally by `Integrator` to optimise the integration
		process (by restricting integration to the indices which could possibly 
		affect the state).
		'''
		time_ops = {}
		for time, op in self.time_ops.items():
			time = self.p('t', t=time)  # ,**self.get_op_params()
			if time in time_ops:
				raise ValueError("Timed operators clash. Consider merging them.")
			if indices is None:
				time_ops[time] = op.collapse('t')  # ,**self.get_op_params()
			else:
				time_ops[time] = op.restrict(*indices).collapse('t')  # ,**self.get_op_params()
		return time_ops
	
	@property
	def operators(self):
		'''
		Return a reference to the list of operators (each of which is a `StateOperator`) used internally. 
		To add operators you can directly add to this list, or use (much safer):
		
		>>> self.operators = [<StateOperator>, <StateOperator>, ...]
		'''
		return self.__operators
	@operators.setter
	def operators(self, operators):
		self.__operators = []
		for operator in operators:
			self.add_operator(operator)
	
	def add_operator(self, operator):
		'''
		This method appends the provided `StateOperator` to the list of operators
		to be used during integrations.
		'''
		if not isinstance(operator, StateOperator):
			raise ValueError("Operator must be an instance of State Operator.")
		self.__operators.append(operator)

	def get_operators(self, indices=None):
		'''
		This method returns the operators of `Integrator.operators` restricted to
		the indicies specified (using `StateOperator.restrict`).
		
		This is used internally by `Integrator` to optimise the integration
		process (by restricting integration to the indices which could possibly 
		affect the state).
		'''
		if indices is None:
			return self.operators

		operators = []
		for operator in self.operators:
			operators.append(operator.restrict(*indices).collapse('t'))  # ,**self.get_op_params()
		return operators
	
	@property
	def params(self):
		'''
		Returns a reference to the parameter overrides to be used by the `StateOperator` objects
		that are in turn used by `Integrator`.
		'''
		return self.__operator_params
	@params.setter
	def params(self, params):
		self.__operator_params = params
	
	@property
	def progress_callback(self):
		'''
		Returns the current set progress callback. This can be `True`, in which case the default 
		fallback callback is used; `False`, in which case the callback is disabled; or a 
		manually created instance of `IntegratorCallback`. To retrieve the `IntegratorCallback`
		that will be used (including the fallback), use `Integrator.get_progress_callback`.
		
		The progress callback instance can be set using:
		
		>>> integrator.progress_callback = <True, False, or IntegratorCallback instance>
		'''
		return self.__progress_callback
	@progress_callback.setter
	def progress_callback(self, progress):
		if not isinstance(progress, IntegratorCallback) and not type(progress) == bool:
			raise ValueError("Invalid type '%s' for progress_callback." % type(progress))
		self.__progress_callback = progress
			
	def get_progress_callback(self):
		'''
		Return the `IntegratorCallback` object that will be used by `Integrator`. Note that
		if a callback has not been specified, and `Integrator.progress_callback` is `False`, 
		then an impotent `IntegratorCallback` object is returned, that does nothing when called.
		'''
		if isinstance(self.progress_callback, IntegratorCallback):
			return self.progress_callback
		
		return ProgressBarCallback() if self.progress_callback else IntegratorCallback()
	
	@property
	def int_kwargs(self):
		'''
		A reference to the dictionary of extra keyword arguments to pass to the 
		`_integrator` initialisation method; which in turn can use these keyword
		arguments to initialise the integration. 
		'''
		return self.__int_kwargs
	@int_kwargs.setter
	def int_kwargs(self, kwargs):
		self.__int_kwargs = kwargs

	########## USER METHODS ################################################
	
	def integrate(self, times=None, step=1):
		if self.results is None:
			return self.start(times, step)
		else:
			return self.extend(times, step)
		
	#
	# Start integration and cache results
	def start(self, times=None, step=1):
		self.results, return_results = self.__integrate(self.initial, times=times, step=step, t_offset=self.t_offset)
		return return_results

	#
	# Continue last integration
	def extend(self, times=None, step=1):
		t_offset = self.results[0][-1][0]

		current_states = []
		for result in self.results:
			current_states.append(result[-1][1])

		results, return_results = self.__integrate(current_states, times=times, step=step, t_offset=t_offset)

		for i, result in enumerate(results):
			for j, tuple in enumerate(result):
				if j > 0:
					self.results[i].append((tuple[0] + t_offset, tuple[1]))
		return return_results

	########## INTEGRATION #################################################

	#
	# Convert initial values to a vector twice as long, with the second set referring to imaginary values
	@abstractmethod
	def _state_internal2ode(self, state):
		pass

	@abstractmethod
	def _state_ode2internal(self, state, dimensions):
		pass

	@abstractmethod
	def _derivative(self, t, y, dim):
		pass

	@abstractmethod
	def _integrator(self, f, **kwargs):
		pass

	@abstractmethod
	def _integrate(self, T, y_0, times=None, **kwargs):
		pass

	#
	# Process solution results to convert solution back to complex form
	def __results_ode2internal(self, results, dimensions):
		presults = []
		for cut in results:
			presults.append((cut[0], self._state_ode2internal(cut[1], dimensions)))
		return presults

	def __state_prepare(self, y_0):
		if isinstance(y_0, Operator):
			y_0 = y_0(t=0)  # ,**self.get_op_params()
		nz = np.nonzero(y_0)
		indices = set()
		for n in nz:
			indices.update(list(n))
		indices = self.__get_connected(*indices)
		return self.__get_restricted_state(y_0, indices), indices

	def __get_restricted_state(self, y_0, indices):
		if len(y_0.shape) == 2:
			y_0 = y_0[indices, :]
			return y_0[:, indices]
		elif len(y_0.shape) == 1:
			return y_0[indices]
		raise ValueError("Cannot restrict y_0. Too many dimensions.")

	def __get_connected(self, *indices):
		new = set(indices)

		operators = self.get_operators() + self.get_time_ops().values()
		for operator in operators:
				new.update(operator.connected(*indices))  # ,**self.get_op_params()

		if len(new.difference(indices)) != 0:
			new.update(self.__get_connected(*new))

		return list(new)

	def __state_restore(self, y, indices, shape):
		if len(shape) not in (1, 2):
			raise ValueError("Integrator only knows how to handle 1 and 2 dimensional states.")
		new_y = np.zeros(shape, dtype=np.complex128)
		if len(shape) == 1:
			new_y[indices] = y
		else:
			for i, index in enumerate(indices):
					new_y[index, indices] = np.array(y)[i, :]

		return new_y

	def __results_restore(self, ys, indices, shape):
		new_ys = []
		for y in ys:
			new_ys.append((y[0], self.__state_restore(y[1], indices, shape)))
		return new_ys

	#
	# Integration routine. Should not ordinarily be called directly
	def __integrate(self, y_0s, times=None, step=1, t_offset=0):
		with self.p:
			self.p(**self.params)

			if times is None:
				raise ValueError("Times must be a list of interesting times.")

			if y_0s is None:
				y_0s, t_offset = self.initial, self.t_offset

			callback = self.get_progress_callback()
			callback.onStart()

			results = []

			# Determine the integration sequence
			if isinstance(times, (list, tuple, np.ndarray)):
				times2 = map(lambda x: self.p('t', t=x), times)  # ,**self.get_op_params()
			else:
				times2 = self.p('t', t=times)  # ,**self.get_op_params())

			progress = Progress()
			progress.run = 0
			progress.runs = len(y_0s)
			progress.callback = callback
			progress.max_time = max(times2)

			required = []

			operators = [None]

			def f(t, y):
				# Register progress
				try:
					current_progress = (t / progress.max_time + float(progress.run)) / float(progress.runs)
					progress.callback.onProgress(current_progress, identifier=self.identifier)
				except Exception, e:
					print "Error updating progress.", e

				# Do integrations
				return self._derivative(t, y, dim, operators[0])

			# Initialise ODE Solver
			T = self._integrator(f, **self.int_kwargs)

			for y_orig in y_0s:
				y_0, indices = self.__state_prepare(y_orig)
				new_ops = self.get_operators(indices=indices)
				operators[0] = new_ops
				sequence = self._sequence(t_offset, times2, self.get_time_ops(indices=indices))

				y_0, dim = self._state_internal2ode(y_0)
				solution = []

				for i, segment in enumerate(sequence):
					if isinstance(segment, tuple):
						sol = self._integrate(T, y_0, times=segment[0], max_step=step)
						solution.extend(sol) if len(solution) == 0  else solution.extend(sol[1:])
						y_0 = sol[-1][1]
						required.extend(segment[1])
					elif isinstance(segment, StateOperator):
						y_0, dim = self._state_internal2ode(
														segment(
															state=self._state_ode2internal(y_0, dim),
															t=0 if i == 0 else sequence[i - 1][0][-1],
															# params=self.get_op_params()
															)
														)
					else:
						raise ValueError("Unknown segment type generated by sequence.")

				inner_results = self.__results_ode2internal(solution, dim)
				results.append(self.__results_restore(inner_results, indices, y_orig.shape))
				progress.run += 1

			# since we use sympy objects, potentially with cache enabled, we should clear it lest it build up
			sympy_clear_cache()

			callback.onComplete(identifier=self.identifier)

			# ## Generate results for user
			result_type = [('time', float), ('label', object), ('state', complex, y_0s[0].shape)]
			return_results = np.empty(dtype=result_type, shape=(len(results), len(times)))

			# Create a map of times to original labels
			time_map = {}
			for i, time in enumerate(times):
				if not isinstance(time, (int, long, float, complex)):
					time_map[self.p(time)] = time  # **self.get_op_params()

			# Populate return results
			for i, result_set in enumerate(results):
				k = 0
				for j, (time, y) in enumerate(result_set):
					if required[j]:
						return_results[i, k] = (time, time_map.get(time, None), y)
						k += 1

			return results, return_results

	#
	# Return a sequence of events for easy integration
	def _sequence(self, t_offset, times, time_ops):

		t_offset = float(t_offset)

		sequence = []

		# Value checking (assuming iterable list)
		if (t_offset > np.max(times)):
			raise ValueError("All interesting times before t_offset. Aborting.")

		times = np.sort(np.unique(np.array(times)))
		if times.shape == ():
			times.shape = (1,)

		times = times[np.where(times >= t_offset)]
		optimes = np.array(sorted(time_ops.keys()))
		optimes = optimes[np.where((optimes >= t_offset) & (optimes <= np.max(times)))]

		subtimes = []
		required = []
		for time in np.sort(np.unique(np.append(np.append(times, optimes), t_offset))):
			subtimes.append(time)
			if time in optimes:
				required.append(False)
				if len(subtimes) > 1:
					sequence.append((subtimes, required))
				sequence.append(time_ops[time])
				subtimes = [time]
			elif time in times:
				required.append(True)
			else:
				required.append(False)
		if len(subtimes) > 1:
			sequence.append((subtimes, required))

		if isinstance(sequence[-1], types.FunctionType):
			print "WARNING: Last time_op will not currently appear in results."

		return sequence


class QuantumIntegrator(Integrator):

	def _state_internal2ode(self, state):
		state = np.array(state)
		dim = state.shape
		return np.reshape(state, (np.prod(state.shape),)), dim

	def _state_ode2internal(self, state, dimensions):
		return np.reshape(state, dimensions)

	def _derivative(self, t, y, dim, operators):
		y = y / np.linalg.norm(y)
		dy = np.zeros(dim, dtype='complex')
		# Apply operators
		y.shape = dim

		for operator in operators:
			dy += operator(state=y, t=t)  # ,params=self.get_op_params())

		dy.shape = (np.prod(dy.shape),)
		return dy

	#
	# Set up integrator
	def _integrator(self, f, **kwargs):
		warnings.warn("QuantumIntegrator can sometimes be unreliable. Please use RealIntegrator instead.")
		from scipy.integrate import ode

		defaults = {
			'nsteps': 1e9,
			'with_jacobian': False,
			'method': 'bdf',
		}
		defaults.update(kwargs)

		if 'nsteps' not in kwargs:
			kwargs['nsteps'] = 1e9

		r = ode(f).set_integrator('zvode', atol=self.error_abs, rtol=self.error_rel, **kwargs)
		return r

	#
	# Internal integration function
	# Must return at values at least t_0 and t_f to allow for continuing
	# Format for returned value is: [ [<time>,<vector>], [<time>,<vector>], ... ]
	def _integrate(self, T, y_0, times=None, **kwargs):

		T.set_initial_value(y_0, times[0])
		results = []
		for i, time in enumerate(times):
			if i == times[0]:
				results.append((time, y_0))
			else:
				T.integrate(time)
				if not T.successful():
					raise ValueError("Integration unsuccessful. T = %f" % time)
				results.append((T.t, T.y))

		return results


class RealIntegrator(Integrator):

	def _integrator(self, f, **kwargs):
		from scipy.integrate import ode

		defaults = {
			'nsteps': 1e9,
			'with_jacobian': False,
			'method': 'bdf',
		}
		defaults.update(kwargs)

		if 'nsteps' not in kwargs:
			kwargs['nsteps'] = 1e9

		r = ode(f).set_integrator('vode', atol=self.error_abs, rtol=self.error_rel, **kwargs)
		return r

	def _integrate(self, T, y_0, times=None, **kwargs):
		T.set_initial_value(y_0, times[0])
		results = []
		for i, time in enumerate(times):
			if i == 0:
				results.append((time, y_0))
			else:
				T.integrate(time)
				if not T.successful():
					raise ValueError("Integration unsuccessful. T = %f" % time)
				results.append((T.t, T.y))

		return results

	def _state_internal2ode(self, state):
		dim = state.shape
		state = np.array(state).reshape((np.prod(dim),))
		return np.array(list(np.real(state)) + list(np.imag(state))), dim

	def _state_ode2internal(self, state, dimensions):
		state = np.array(state[:len(state) / 2]) + 1j * np.array(state[len(state) / 2:])
		return np.reshape(state, dimensions)

	def _derivative(self, t, y, dim, operators):
		dy = np.zeros(dim, dtype='complex')
		# Apply operators
		y = self._state_ode2internal(y, dim)

		for operator in operators:
			dy += operator(state=y, t=t)  # ,params=self.get_op_params())

		dy, dim = self._state_internal2ode(dy)
		return dy


try:
	import sage.all as s

	class SageIntegrator(Integrator):

		def _integrator(self, f, **kwargs):
			T = s.ode_solver()
			T.function = f
			T.algorithm = 'rkf45'  # self.solver
			T.error_rel = self.error_rel
			T.error_abs = self.error_abs
			return T

		def _integrate(self, T, y_0, times=None, **kwargs):
			T.ode_solve(y_0=list(y_0), t_span=list(times))
			return T.solution

		def _state_internal2ode(self, state):
			dim = state.shape
			state = np.array(state).reshape((np.prod(dim),))
			return list(np.real(state)) + list(np.imag(state)), dim

		def _state_ode2internal(self, state, dimensions):
			state = np.array(state[:len(state) / 2]) + 1j * np.array(state[len(state) / 2:])
			return np.reshape(state, dimensions)

		def _derivative(self, t, y, dim, operators):
			y = np.array(y[:len(y) / 2]).reshape(dim) + 1j * np.array(y[len(y) / 2:]).reshape(dim)
			dy = np.zeros(dim, dtype='complex')

			for operator in operators:
				dy += operator(state=y, t=t)  # ,params=self.get_op_params())

			dy.shape = (np.prod(dy.shape),)
			dy = list(np.real(dy)) + list(np.imag(dy))
			return dy

except:
	pass
