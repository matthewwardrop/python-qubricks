
from abc import ABCMeta, abstractmethod
from qubricks.utility.text import colour_text as coloured
import sys

def _model_wrapper(dependency=None, message=None, complete=None):
	
	def private(self, key, default=None):
		return getattr(self,'_%s%s'%(self.__class__.__name__,key),default)
	
	def private_set(self, key, data):
		args = tuple()
		kwargs = {}
		
		if data is None:
			pass
		elif isinstance(data, tuple):
			if len(data) == 2 and isinstance(data[0], tuple) and isinstance(data[1],dict):
				args = data[0]
				kwargs = data[1]
			else:
				args = data
		elif isinstance(data, dict):
			kwargs = data
		else:
			args = (data,)
		
		setattr(self, '_%s%s_args'%(self.__class__.__name__,key), args)
		setattr(self, '_%s%s_kwargs'%(self.__class__.__name__,key), kwargs)
		setattr(self, '_%s%s'%(self.__class__.__name__,key), True)
		
	def public_output(args, kwargs):
		if len(args) == 0 and len(kwargs) == 0:
			return None
		elif len(kwargs) == 0:
			if len(args) == 1:
				return args[0]
			else:
				return args
		elif len(args) == 0:
			return kwargs
		else:
			return args, kwargs
	
	def wrap(f):
		fname = f.__name__
		if fname.startswith('__'):
			fname = fname[2:]
		
		def wrapped(self, *args, **kwargs):
			
			# If arguments provided, use them
			if len(args) > 0 or len(kwargs) > 0:
				return f(self, *args, **kwargs)
			
			# Otherwise, use the existing results, generating them as required
			if not private(self, '__%s'%fname, False):
				if dependency is not None:
					dependency_f = getattr(self, dependency, None)
					if dependency_f is None:
						raise ValueError("Unknown method: %s" % dependency)
					
					if private(self, '__%s'%dependency) is None:
						dependency_f()
				
				if message is not None:
					print coloured(message, "YELLOW", True),
					sys.stdout.flush()
				
				try:
					private_set(self, '__%s'%fname, f(self, *private(self,'__%s_args'%dependency,[]), **private(self,'__%s_kwargs'%dependency,{})) )
				except Exception, e:
					print coloured("Error", "RED", True)
					raise e
				
				if complete is not None:
					print coloured("DONE" if complete is None else complete, "GREEN", True)
					
			return public_output(private(self, '__%s_args'%fname), private(self, '__%s_kwargs'%fname))
		return wrapped
	return wrap

class ModelAnalysis(object):
	'''
	`ModelAnalysis` is a helper class that can simplify the routine of running
	simulations, processing the results, and then plotting (or otherwise outputting)
	them. One simply need subclass `ModelAnalysis`, and implement the following
	methods:
		- prepare(self, *args, **kwargs)
		- simulate(self, *args, **kwargs)
		- process(self, *args, **kwargs)
		- plot(self, *args, **kwargs)
	
	Each of these methods is guaranteed to be called in the order specified above, with
	the return values of the previous method being fed forward to the next.
	Calling `process` (with no arguments), for example, will also call `prepare` and `simulate` in order, with the
	return values of `prepare` being passed to `simulate`, and the return values of `simulate`
	being passed to `process`. If a method is called directly with input values, then this
	chaining does not occur, and the method simply returns what it should.
	
	It is necessary to be a little bit careful about what one returns in these methods.
	In particular, this is the way in which return values are processed:
		- If a tuple is returned of length 2, and the first element is a
		tuple and the second a dict, then it is assumed that these are 
		respectively the `args` and `kwargs` to be fed forward.
		- If a tuple is returned of any other length, or any of the above conditions
		fail, then these are assumed to be the `args` to be fed forward.
		- If a dictionary is returned, then these are assumed to be the 
		`kwargs` to be fed forward.
		- Otherwise, the result is fed forward as the first non-keyword argument.
	
	.. note:: It is not necessary to return values at these steps, if it is unnecessary
		or if you prefer to save your results as attributes.
		
	.. note:: Return values of all of these methods will be cached, so each method
		will only be run once.
	'''
	__metaclass__ = ABCMeta
	
	def __getattribute__(self, name):
		if name in ['prepare', 'process', 'simulate', 'plot']:
			return object.__getattribute__(self, '_ModelAnalysis__%s' % name )
		else:
			return object.__getattribute__(self, name)
	
	@_model_wrapper(dependency=None, message="Preparing...", complete="DONE")
	def __prepare(self, *args, **kwargs):
		return object.__getattribute__(self, 'prepare')(*args, **kwargs)

	@_model_wrapper(dependency="prepare", message="Simulating...", complete="DONE")
	def __simulate(self, *args, **kwargs):
		return object.__getattribute__(self, 'simulate')(*args, **kwargs)

	@_model_wrapper(dependency="simulate", message="Processing...", complete="DONE")
	def __process(self, *args, **kwargs):
		return object.__getattribute__(self, 'process')(*args, **kwargs)

	@_model_wrapper(dependency="process", message="Plotting...", complete="DONE")
	def __plot(self, *args, **kwargs):
		return object.__getattribute__(self, 'plot')(*args, **kwargs)

	def __init__(self, *args, **kwargs):
		self.prepare(*args, **kwargs)
	
	@abstractmethod
	def prepare(self, *args, **kwargs):
		'''
		This method should prepare the `ModelAnalysis` instance for calling the
		rest of the methods. It is invoked on class initialisation, with the 
		arguments passed to the constructor. Any return values will be passed
		onto `simulate` if it is ever called with no arguments.
		'''
		pass

	@abstractmethod
	def simulate(self, *args, **kwargs):
		'''
		This method should perform whichever simulations are required. Any values returned
		will be passed onto `process` if it is ever called with no arguments.
		'''
		pass

	@abstractmethod
	def process(self, *args, **kwargs):
		'''
		This method should perform whatever processing is interesting on return values of
		`simulate`. Any values returned will be passed onto `plot` if it is ever called
		with no arguments.
		'''
		pass

	@abstractmethod
	def plot(self, *args, **kwargs):
		'''
		This method should perform whatever plotting/output is desired based upon return values of
		`process`.
		'''
		pass
