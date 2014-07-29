import numpy as np
import math
import types
import sys
from abc import ABCMeta, abstractmethod, abstractproperty
import time as proftime
from sympy.core.cache import clear_cache as sympy_clear_cache
import warnings

from .operators import StateOperator

#TODO: Document this function.

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
	def onComplete(self,identifier=None,message=None,status=0):
		pass

# Try to use the full implementation of progress bar
try:
	import progressbar as pbar
	class ProgressBarCallback(IntegratorCallback):
	
		def onStart(self):
		
		
			self.pb = pbar.ProgressBar(widgets=['\033[0;34m',pbar.Percentage(), ' ', pbar.Bar(),'\033[0m'])
			self.pb.start()
			self._progress = 0
	
		def onProgress(self, progress, identifier=None):
			progress *= 100
			if (progress > self._progress):
				self._progress = math.ceil(progress)
				self.pb.update(min(100,progress))
	
		def onComplete(self,identifier=None,message=None,status=0):
			self.pb.finish()
except:
	class ProgressBarCallback(IntegratorCallback):
		
		def onStart(self):
			self._progress = 0
		
		def onProgress(self,progress, identifier=None):
			progress *= 100
			#if (progress > self._progress):
			self._progress = math.ceil(progress)
			sys.stderr.write( "\r%3d%%" % min(100,progress) )
		
		def onComplete(self,identifier=None,message=None,status=0):
			print "\n"
			if message:
				print "Level %d"%level, message

class Progress(object):
	def __init__(self):
		self.run = 0
		self.runs = 1
		self.callback = None

#
# Integrator object
class Integrator(object):
	__metaclass__ = ABCMeta
	
	def __init__(self,identifier=None,initial=None,t_offset=0,operators=None,parameters=None,op_params={},error_rel=1e-8,error_abs=1e-8,time_ops={},callback=None,callback_fallback=True,**kwargs):
		# Set the identifier for this integrator instance
		self.identifier = identifier

		# Set up initial conditions
		self.set_initial(initial,t_offset)
		
		# Parameter Object
		self._p = parameters
		
		# Set up integration operator
		self._operators = []
		self._operators_params = {}
		if isinstance(operators,list):
			self.add_operators(*operators)
		else:
			self.add_operators(operators)
		self.add_op_params(**op_params)
		
		# Set up intermediate pulse operators
		self._time_ops = {}
		self.add_time_ops(**time_ops)
		
		# Set solver and tolerances
		self.set_error(error_rel,error_abs)
		
		# Set up results cache
		self.results = None
		
		self.set_callback(callback, callback_fallback)
		
		self._integrator_args = kwargs
	
	########### CONFIGURATION ##############################################
	def set_initial(self,y_0s,t_offset=0):
		self._initial = y_0s
		self._initial_offset = t_offset
	
	def set_callback(self,callbackObj=None,fallback=True):
		self._callback,self._callback_fallback = callbackObj, fallback
	
	def set_error(self,rel=1e-8,abs=1e-8):
		self._error_rel=rel
		self._error_abs=abs
	
	def add_time_ops(self,**kwargs):
		for time,op in kwargs:
			if not isinstance(op,StateOperator):
				raise ValueError, "Operator must be an instance of State Operator."
			self._time_ops[time] = op
	
	def add_operators(self,*args):
		for arg in args:
			if not isinstance(arg,StateOperator):
				raise ValueError, "Operator must be an instance of State Operator."
			self._operators.append(arg)
	
	def add_op_params(self,**kwargs):
		self._operators_params.update(kwargs)
	
	def reset(self):
		self.results = None
	
	########## INTERROGATION ###############################################
	
	def get_initial(self):
		return (self._initial,self._initial_offset)
	
	def get_callback(self):
		if self._callback is not None:
			return self._callback
		else:
			return ProgressBarCallback() if self._callback_fallback else IntegratorCallback()
	
	def get_error(self):
		return {'rel':self._error_rel,'abs':self._error_abs}
	
	def get_results(self):
		return self._results
	
	def get_operators(self,indicies=None):
		if indicies is None:
			return self._operators
		
		operators = []
		for operator in self._operators:
			operators.append( operator.restrict(*indicies).collapse('t',**self.get_op_params()) )

		return operators
	
	def get_op_params(self,*args):
		if len(args) > 0:
			r = {}
			for arg in args:
				r[arg] = self._operator_params.get(arg)
		return self._operators_params
	
	def get_time_ops(self):
		return self._time_ops
	
	########## USER METHODS ################################################
	
	#
	# Start integration and cache results
	def start(self,times=None,step=1):
		self.results,return_results = self.integrate(self._initial,times=times,step=step,t_offset=self._initial_offset)
		return return_results
	
	#
	# Continue last integration
	def extend(self,times=None,step=1):
		t_offset = self.results[0][-1][0]
		
		current_states = []
		for result in self.results:
			current_states.append(result[-1][1])
		
		results,return_results = self.integrate(current_states,times=times,step=step,t_offset=t_offset)
		
		for i,result in enumerate(results):
			for j,tuple in enumerate(result):
				if j>0:
					self.results[i].append((tuple[0]+t_offset,tuple[1]))
		return return_results
	
	
	########## INTEGRATION #################################################
	
	#
	# Convert initial values to a vector twice as long, with the second set referring to imaginary values
	@abstractmethod
	def _state_internal2ode(self,state):
		pass
	
	@abstractmethod
	def _state_ode2internal(self,state,dimensions):
		pass
	
	@abstractmethod
	def _derivative(self,t,y,dim):
		pass
	
	@abstractmethod
	def _integrator(self,f,**kwargs):
		pass
	
	@abstractmethod
	def _integrate(self,T,y_0,times=None,**kwargs):
		pass
	
	#
	# Process solution results to convert solution back to complex form
	def __results_ode2internal(self,results,dimensions):	
		presults = []
		for cut in results:
			presults.append( (cut[0],self._state_ode2internal(cut[1],dimensions)) )
		return presults

	def __state_prepare(self, y_0):
		nz = np.nonzero(y_0)
		indicies = set()
		for n in nz:
			indicies.update(list(n))
		indicies = self.__get_connected(*indicies)
		return self.__get_restricted_state(y_0,indicies), self.get_operators(indicies), indicies

	def __get_restricted_state(self, y_0, indicies):
		if len(y_0.shape) == 2:
			y_0 = y_0[indicies,:]
			return y_0[:,indicies]
		elif len(y_0.shape) == 1:
			return y_0[indicies]
		raise ValueError("Cannot restrict y_0. Too many dimensions.")


	def __get_connected(self,*indicies):
		new = set(indicies)
		
		for operator in self.get_operators():
				new.update( operator.connected(*indicies,**self.get_op_params()) )
		
		if len( new.difference(indicies) ) != 0:
			new.update( self.__get_connected(*new) )
		
		return list(new)

	def __state_restore(self, y, indicies, shape):
		if len(shape) not in (1,2):
			raise ValueError("Integrator only knows how to handle 1 and 2 dimensional states.")
		new_y = np.zeros(shape,dtype=np.complex128)
		if len(shape) == 1:
			new_y[indicies] = y
		else:
			for i,index in enumerate(indicies):
					new_y[index,indicies] = np.array(y)[i,:]

		return new_y

	def __results_restore(self, ys, indicies, shape):
		new_ys = []
		for y in ys:
			new_ys.append( (y[0], self.__state_restore(y[1],indicies,shape)) )
		return new_ys
	
	#
	# Integration routine. Should not ordinarily be called directly
	def integrate(self,y_0s,times=None,step=1,t_offset=0):
		
		if times is None:
			raise ValueError, "Times must be a list of interesting times."
		
		if y_0s is None:
			y_0s,t_offset = self.get_initial()
		
		callback = self.get_callback()
		callback.onStart()
		
		results = []
		
		# Determine the integration sequence
		if isinstance(times,(list,tuple,np.ndarray)):
			times2 = map( lambda x: self._p('t',t=x,**self.get_op_params()), times)
		else:
			times2 = self._p('t',t=times,**self.get_op_params())
		
		sequence = self._sequence(t_offset,times2,self.get_time_ops())
		
		progress = Progress()
		progress.run = 0
		progress.runs = len(y_0s)
		progress.callback = callback
		progress.max_time = max(times2)
		
		required = []

		operators = [None]
		
		def f(t,y):
			# Register progress
			try:
				current_progress = (t/progress.max_time+float(progress.run))/float(progress.runs)
				progress.callback.onProgress(current_progress, identifier=self.identifier)
			except Exception, e:
				print "Error updating progress.",e
			
			# Do integrations 
			return self._derivative(t,y,dim,operators[0])
		
		#Initialise ODE Solver
		T = self._integrator(f,**self._integrator_args)
		
		for y_orig in y_0s:
			y_0,new_ops,indicies = self.__state_prepare(y_orig)
			operators[0] = new_ops

			y_0,dim = self._state_internal2ode(y_0)
			solution = []
			
			for segment in sequence:
				if isinstance(segment,types.FunctionType):
					y_0,dim = self._state_internal2ode(segment(0,self._state_ode2internal(y_0,dim),self.get_op_params())) # TODO: FIX time
				else:
					sol = self._integrate(T,y_0,times=segment[0],max_step=step)
					solution.extend(sol) if len(sol) == 0 or len(solution) == 0  else solution.extend(sol[1:])
					y_0 = sol[-1][1]
					required.extend(segment[1])
			
			inner_results = self.__results_ode2internal(solution,dim)
			results.append(self.__results_restore(inner_results,indicies,y_orig.shape))
			progress.run += 1
		
		# since we use sympy objects, potentially with cache enabled, we should clear it lest it build up
		sympy_clear_cache()

		callback.onComplete(identifier=self.identifier)
		

		### Generate results for user
		result_type = [('time',float),('label',object),('state',complex,y_0s[0].shape)]
		return_results = np.empty(dtype=result_type,shape=(len(results),len(times)))

		# Create a map of times to original labels
		time_map = {}
		for i,time in enumerate(times):
			if not isinstance(time,(int, long, float, complex)):
				time_map[self._p(time,**self.get_op_params())] = time

		# Populate return results
		for i,result_set in enumerate(results):
			k=0
			for j, (time,y) in enumerate(result_set):
				if required[j]:
					return_results[i,k] =  (time,time_map.get(time,None),y)
					k+=1
			
		return results, return_results
	
	#
	# Return a sequence of events for easy integration
	def _sequence(self,t_offset,times,time_ops):
		
		t_offset = float(t_offset)
	
		sequence = []
	
		# Value checking (assuming iterable list)
		if ( t_offset > np.max(times) ):
			raise ValueError, "All interesting times before t_offset. Aborting."
		
		times = np.sort(np.unique(np.array(times)))
		if times.shape == (): times.shape = (1,)
		
		times = times[np.where(times>=t_offset)]
		optimes = np.array(sorted(time_ops.keys()))
		optimes = optimes[np.where( (optimes >= t_offset) & (optimes <= np.max(times)) )]
	
		i = 0
		j = 0
	
		subtimes = []
		required = []
		for time in np.sort(np.unique(np.append(np.append(times,optimes),t_offset))):
			subtimes.append(time)
			if time in optimes:
				required.append(False)
				if len(subtimes) > 1:
					sequence.append((subtimes,required))
				sequence.append(time_ops[time])
				subtimes = [time]
			elif time in times:
				required.append(True)
			else:
				required.append(False)
		if len(subtimes) > 1:
			sequence.append((subtimes,required))
		
		if isinstance(sequence[-1],types.FunctionType):
			print "WARNING: Last time_op will not currently appear in results."
		
		return sequence

class QuantumIntegrator(Integrator):
	
	def _state_internal2ode(self,state):
		state = np.array(state)
		dim = state.shape
		return np.reshape(state,(np.prod(state.shape),)),dim
	
	def _state_ode2internal(self,state,dimensions):
		return np.reshape(state,dimensions)
	
	def _derivative(self,t,y,dim,operators):
		y = y/np.linalg.norm(y)
		dy = np.zeros(dim,dtype='complex')
		# Apply operators
		y.shape = dim
		
		for operator in operators:
			dy += operator(state=y,t=t,params=self.get_op_params())
		
		dy.shape = (np.prod(dy.shape),)
		return dy
	
	#
	# Set up integrator
	def _integrator(self,f,**kwargs):
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
			
		
		r = ode(f).set_integrator('zvode', atol=self._error_abs, rtol=self._error_rel, **kwargs)
		return r
	
	#
	# Internal integration function
	# Must return at values at least t_0 and t_f to allow for continuing
	# Format for returned value is: [ [<time>,<vector>], [<time>,<vector>], ... ]
	def _integrate(self,T,y_0,times=None,**kwargs):
		
		T.set_initial_value(y_0,times[0])
		results = []
		for i,time in enumerate(times):
			if i == times[0]:
				results.append((time,y_0))
			else:
				T.integrate(time)
				if not T.successful():
					raise ValueError, "Integration unsuccessful. T = %f" % time
				results.append((T.t,T.y))
		
		return results

class RealIntegrator(Integrator):

	def _integrator(self,f,**kwargs):
		from scipy.integrate import ode
		
		defaults = {
			'nsteps': 1e9,
			'with_jacobian': False,
			'method': 'bdf',
		}
		defaults.update(kwargs)
		
		if 'nsteps' not in kwargs:
			kwargs['nsteps'] = 1e9
			
		r = ode(f).set_integrator('vode', atol=self._error_abs, rtol=self._error_rel, **kwargs)
		return r

	def _integrate(self,T,y_0,times=None,**kwargs):
		T.set_initial_value(y_0,times[0])
		results = []
		for i,time in enumerate(times):
			if i == times[0]:
				results.append((time,y_0))
			else:
				T.integrate(time)
				if not T.successful():
					raise ValueError, "Integration unsuccessful. T = %f" % time
				results.append((T.t,T.y))
		
		return results
	
	def _state_internal2ode(self,state):
		dim = state.shape
		state = np.array(state).reshape((np.prod(dim),))
		return np.array(list(np.real(state)) + list(np.imag(state))),dim

	def _state_ode2internal(self,state,dimensions):
		state = np.array(state[:len(state)/2])+1j*np.array(state[len(state)/2:])
		return np.reshape(state,dimensions)
	
	def _derivative(self,t,y,dim,operators):
		dy = np.zeros(dim,dtype='complex')
		# Apply operators
		y = self._state_ode2internal(y,dim)
		
		for operator in operators:
			dy += operator(state=y,t=t,params=self.get_op_params())
		
		dy,dim = self._state_internal2ode(dy)
		return dy


try:
	import sage.all as s

	class SageIntegrator(Integrator):
	
		def _integrator(self,f,**kwargs):
			T = s.ode_solver()
			T.function = f
			T.algorithm='rkf45'#self.solver
			T.error_rel=self._error_rel
			T.error_abs=self._error_abs
			return T
	
		def _integrate(self,T,y_0,times=None,**kwargs):
			T.ode_solve(y_0=list(y_0),t_span=list(times))
			return T.solution
		
		def _state_internal2ode(self,state):
			dim = state.shape
			state = np.array(state).reshape((np.prod(dim),))
			return list(np.real(state)) + list(np.imag(state)),dim
	
		def _state_ode2internal(self,state,dimensions):
			state = np.array(state[:len(state)/2])+1j*np.array(state[len(state)/2:])
			return np.reshape(state,dimensions)
		
		def _derivative(self,t,y,dim,operators):
			y = np.array(y[:len(y)/2]).reshape(dim) + 1j*np.array(y[len(y)/2:]).reshape(dim)
			dy = np.zeros(dim,dtype='complex')
	
			for operator in operators:
				dy += operator(state=y,t=t,params=self.get_op_params())
	
			dy.shape = (np.prod(dy.shape),)
			dy = list(np.real(dy)) + list(np.imag(dy))
			return dy
			
except:
	pass
