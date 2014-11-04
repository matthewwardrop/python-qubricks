from abc import ABCMeta, abstractmethod, abstractproperty
import copy
import math
import os
import re
import shelve
import sys
import types
import datetime, time

import numpy as np

from .integrator import IntegratorCallback
from .operators import StateOperator
from .utility.text import colour_text as coloured
from .utility import struct_allclose


class Measurement(object):
	'''
	Measurement(system,**kwargs)

	Measurement is an object which wraps around a QuantumSystem object to provide
	a convenient way to procedurally perform measurements. It also provides
	routines to sweep over multi-dimensional regions of parameter space.
	Measurement is an abstract class and should be subclassed to be useful.
	Measurements can be called to perform the measurement routines, or equivalently,
	calls can be made to the Measurement.method routine.

	Parameters
	----------
	system : A reference to a QuantumSystem object.
	kwargs : parameters to be sent through to subclasses of Measurement.
	'''
	__metaclass__ = ABCMeta

	multiprocessing = True
	ranges_type = [('ranges','object')]
	#axis_type = [('label','object'),('units','object'),('values',float)]

	def __init__(self,system,**kwargs):
		self._system = system
		self.init(**kwargs)

	@abstractmethod
	def init(self,**kwargs):
		'''
		Measurement.init should be specified by measurement subclasses.
		'''
		raise NotImplementedError("Measurement.init has not been implemented.")

	def __call__(self,*args,**kwargs):
		'''
		Calling a Measurement object is shorthand for calling the `measure` method.
		'''
		return self.measure(*args,**kwargs)

	@abstractmethod
	def measure(self,times,y_0s,params={},**kwargs):
		'''
		Measurement.measure is where the grunt work is done; and it should return
		a numpy array of type Measurement.result_type, with shape
		Measurement.result_shape . Otherwise, anything is permitted in this method.
		'''
		raise NotImplementedError("Measurement.measure has not been implemented.")

	@abstractmethod
	def result_type(self,*args,**kwargs):
		'''
		Measurement.result_type should return an object suitable for use as the dtype
		argument in numpy. Otherwise, no restrictions; other than that it must also
		agree with the datatype returned by Measurement.measure.
		'''
		raise NotImplementedError("Measurement.result_type has not been implemented.")

	@abstractmethod
	def result_shape(self,*args,**kwargs):
		'''
		Measurement.result_shape should agree with the shape of the numpy array returned
		by Measurement.measure, but otherwise has no restrictions.
		'''
		raise NotImplementedError("Measurement.result_shape has not been implemented.")

	def _iterate_results_init(self,ranges=[],shape=None,params={},*args,**kwargs):
		'''
		This is a generic initialisation for the Measurement object. It can be overloaded
		if necessary.
		'''
		# Initialise empty array to store results
		dtype = self.result_type(ranges=ranges,shape=shape,params=params,*args,**kwargs)
		rshape = self.result_shape(ranges=ranges,shape=shape,params=params,*args,**kwargs)
		if rshape is not None:
			shape = shape+rshape

		a = np.empty(shape, dtype=dtype)
		a.fill(np.nan)

		return a

	def _iterate_results_add(self,resultsObj=None,result=None,indicies=None,params={}):
		'''
		This is a generic update for a particular iteration of the Measurements.iterate. It can be
		overloaded if necessary.
		'''
		resultsObj[indicies] = result

	def iterate_yielder(self,ranges,params={},masks=None,nprocs=None,yield_every=100,results=None,*args,**kwargs):
		'''
		Measurement.iterate performs the calculations in Measurement.measure method
		multiple times, sweeping over multi-dimensional parameter ranges, as specified
		in the `ranges` argument. The format of the ranges argument is described below.
		Where not overridden in ranges, additional parameter overrides can be specified
		in `params`, using any format understood by the Parameters module. This method
		is by default capable of using multiple processes to speed up execution using
		Python's multiprocessing module. `nprocs` specifies the number of threads to
		use; by default, if it is None, one thread less than the number of available
		processors will be used. `args` and `kwargs` will be forwarded to the the
		measure method.

		The range formatting is the same as for the Parameters module. i.e. each
		parameter range can be specified as a list of values ([0,1,2,3]) or a tuple
		of constraints (<start>,<stop>,<count>,<optional sampler>). The sampler
		can be one of: 'linear', 'log', 'invlog' or a function f(<start>,<stop>,<count>).

		For each dimension of parameter sweeping, a dictionary of ranges is added to
		a list; each indexed by the appropriate parameter name. For example:
		[ {'pam1': (0,1,10), 'pam2': range(10)}, {'pam3': (0,1,5), 'pam4': range(5)} ]
		In this example, a 2D sweep takes place; the first dimension having 10 iterations,
		and the second having 5; with pam1, pam2, pam3 and pam4 being updated appropriately.

		`masks` allows one to filter ranges within the parameter ranges to skip. This
		can be useful if those runs have already been collected, or if they are outside
		of the domain in which you are especially interested. `masks` should be a list
		of functions with a declaration similar to:
		mask (indicies, ranges=None, params={})
		Masks should not change any of the lists or dictionaries passed to them; and simply
		return True if this datapoint should be collected. If multiple masks are provided,
		data will only be collected if all masks return True. Iteration numbers are
		zero indexed.

		If `yield_every` is not None, every `yield_every` runs, the method yield the current
		results as a MeasurementResults object. This can be used to save the results iteratively
		as the calculations are performed; which may help to prevent data loss in the event
		of interruptions like power loss. If `yield_every` is supposed to be None, you should
		use `Measurement.iterate` instead.
		'''
		# Check if ranges is just a single range, and if so make it a list
		if isinstance(ranges,dict):
			ranges = [ranges]

		def extend_ranges(ranges_eval,labels,size):
			dtype_delta = [(label,float) for label in labels]
			if ranges_eval is None:
				ranges_eval = np.zeros(size,dtype=dtype_delta)
				ranges_eval.fill(np.nan)
			else:
				final_shape = ranges_eval.shape + (size,)
				ranges_eval = np.array(np.repeat(ranges_eval,size).reshape(final_shape),dtype=ranges_eval.dtype.descr + dtype_delta)
				for label in labels:
					ranges_eval[label].fill(np.nan)
			return ranges_eval

		def vary_pams(level=0,iteration=tuple(),output=[],ranges_eval=None):
			'''
			This method generates a list of different parameter configurations
			'''

			pam_ranges = ranges[level]

			## Interpret ranges
			pam_values = {}
			count = None
			for param, pam_range in pam_ranges.items():
				
				if param in ranges_eval.dtype.fields.keys() and np.all(ranges_eval[param] != np.nan): # If values already exist in ranges_eval, reuse them
					pam_values[param] = ranges_eval[param][iteration + (slice(None),) + tuple([0]*(ranges_eval.ndim-len(iteration)-1))]
				else:
					tparams = params.copy()
					tparams[param] = pam_range
					pam_values[param] = self._system.p.range(param,**tparams)
					
				c = len(pam_values[param])
				count = c if count is None else count
				if c != count:
					raise ValueError, "Parameter ranges for %s are not consistent in count: %s" % (param, pam_ranges)

			if ranges_eval is None or ranges_eval.ndim < level + 1:
				ranges_eval = extend_ranges(ranges_eval, pam_ranges.keys(), count)

			for i in xrange(count):
				current_iteration = iteration + (i,)

				# Generate slice corresponding all the components of range_eval this iteration affects
				s = current_iteration + tuple([slice(None)]*(ranges_eval.ndim-len(current_iteration)))

				# Update parameters
				for param, pam_value in pam_values.items():
					params[param] = pam_value[i]
					ranges_eval[param][s] = pam_value[i]

				if level < len(ranges) - 1:
					# Recurse problem
					ranges_eval,_ = vary_pams(level=level+1,iteration=current_iteration,output=output,ranges_eval=ranges_eval)
				else:
					if masks is not None and isinstance(masks,list):
						skip = False
						for mask in masks:
							if not mask(current_iteration,ranges=ranges,params=params):
								skip = True
								break
						if skip:
							continue


					kwargs2 = copy.copy(kwargs)
					kwargs2['params'] = copy.copy(params)
					kwargs2['callback_fallback'] = False
					output.append( (
							iteration,
							copy.copy(args),
							kwargs2
						) )

			return ranges_eval, output

		ranges_eval,output = vary_pams(ranges_eval=results.ranges_eval if results is not None else None)

		if self.multiprocessing:
			from qubricks.utility.symmetric import AsyncParallelMap
			apm = AsyncParallelMap(self.measure,progress=True,nprocs=nprocs)
			callback = None
		else:
			apm = None
			levels_info = [{'name':','.join(ranges[i].keys()),'count':ranges_eval.shape[i]} for i in xrange(ranges_eval.ndim)]
			callback = IteratorCallback(levels_info,ranges_eval.shape)
			
		data = results.data if results is not None else None
		data_new = self._iterate_results_init(ranges=ranges,shape=ranges_eval.shape,params=params,*args,**kwargs)
		if data is None:
			data = data_new
		else:
			if data.shape != data_new.shape:
				raise ValueError("Invalid results given to continue. Shape %s does not agree with result dimensions %s" % (data.shape,data_new.shape))
			if data.dtype != data_new.dtype:
				raise ValueError("Invalid results given to continue. Type %s does not agree with result type %s" % (data.dtype,data_new.dtype))
		
		if results is None:
			results = MeasurementResults(ranges,ranges_eval,data)
		else:
			if not struct_allclose(ranges_eval,results.ranges_eval,rtol=1e-15,atol=1e-15):
				raise ValueError("Attempted to resume measurement collection on a result set with different parameter ranges.")
			results.update(ranges=ranges, ranges_eval=ranges_eval, data=data)

		def splitlist(l,length=None):
			if length is None:
				yield l
			else:
				for i in xrange(0,len(l),length):
					yield l[i:i+length]

		start_time = datetime.datetime.now()
		for i,tasks in enumerate(splitlist(output,yield_every)):
			t_start = time.time()
			if self.multiprocessing:
				res = apm.map(tasks,count_offset= (yield_every*i if yield_every is not None else None),count_total=len(output),start_time=start_time )
			else:
				res = [ (indicies,self.measure(*args,callback=callback,identifier=indicies,**kwargs)) for indicies,args,kwargs in tasks] # TODO: neaten legacy mode callback
			for indicies,value in res:
				self._iterate_results_add(resultsObj=data,result=value,indicies=indicies)

			yield results.update(data=data,runtime=time.time()-t_start)

	def iterate(self,*args,**kwargs):
		'''
		A wrapper around the `Measurement.iterate_yielder` method in the event that
		one does not want to deal with a generator object. This simply returns
		the final result.
		'''
		from collections import deque
		return deque(self.iterate_yielder(*args, yield_every=None, **kwargs),maxlen=1).pop()

	def iterate_to_file(self,path,samplers={},*args,**kwargs):
		'''
		Measurement.iterate_to_file saves the results of the Measurement.iterate method
		to a python shelve file at `path`; all other arguments are passed through to the
		Measurement.iterate method.
		'''
		
		if os.path.dirname(path) is not "" and not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		elif os.path.exists(os.path.dirname(path)) and os.path.isfile(os.path.dirname(path)):
			raise RuntimeError, "Destination path '%s' is a file."%os.path.dirname(path)

		results = None
		if os.path.isfile(path):
			results = MeasurementResults.load(path,samplers=samplers)
			
			if results.is_complete:
				return results

			masks = kwargs.get('masks',None)
			if masks is None:
				masks = []
			masks.append( results.continue_mask )
			kwargs['masks'] = masks

			coloured("Attempting to continue data collection...","YELLOW",True)
		
		
		for results in self.iterate_yielder(*args,results=results,**kwargs):
			results.save(path=path,samplers=samplers)

		return results

class MeasurementResults(object):

	def __init__(self,ranges,ranges_eval,data,runtime=None,path=None,samplers={}):
		self.ranges = ranges
		self.ranges_eval = ranges_eval
		self.data = data
		self.runtime = 0 if runtime is None else runtime
		self.path = path
		self.samplers = samplers
	
	def update(self,**kwargs):
		for key,value in kwargs.items():
			if key not in ['ranges','ranges_eval','data','runtime','path','samplers']:
				raise ValueError("Invalid update key: %s" % key)
			if key is "runtime":
				self.runtime += value
			else:
				setattr(self,key,value)
		return self
	
	@property
	def is_complete(self):
		return len(np.where(np.isnan(self.data.view('float')))[0]) == 0
	
	@property
	def continue_mask(self):
		def continue_mask(indicies, ranges=None, params={}): # Todo: explore other mask options
				return np.any(np.isnan(self.data[indicies].view('float')))
		return continue_mask
	
	@staticmethod
	def _process_ranges(ranges,defunc=False,samplers={}):
		if ranges is None:
			return ranges
		ranges = copy.deepcopy(ranges)
		for range in ranges:
			for param,spec in range.items():
				if defunc and type(spec[-1]) == types.FunctionType:
					spec = list(spec)
					spec[-1] = spec[-1].__name__
					spec = tuple(spec)
					range[param] = spec
				if not defunc and len(spec) > 3 and type(spec[-1]) == str and spec[-1] in samplers:
					spec = list(spec)
					spec[-1] = samplers[spec[-1]]
					spec = tuple(spec)
					range[param] = spec
		return ranges

	def save(self,path=None,samplers=None):
		if path is None:
			path = self.path
		if samplers is None:
			samplers = self.samplers
		else:
			self.samplers = samplers
		if path is None:
			raise ValueError("Output file was not specified.")

		s = shelve.open(path)
		s['ranges'] = MeasurementResults._process_ranges(self.ranges,defunc=True,samplers=samplers)
		s['ranges_eval'] = self.ranges_eval
		s['results'] = self.data
		s['runtime'] = self.runtime
		s.close()

	@classmethod
	def load(cls,path,samplers={}):
		s = shelve.open(path)
		ranges = MeasurementResults._process_ranges(s.get('ranges'),defunc=False,samplers=samplers)
		ranges_eval = s.get('ranges_eval')
		data = s.get('results')
		runtime = s.get('runtime')
		s.close()
		return cls(ranges=ranges,ranges_eval=ranges_eval,data=data,runtime=runtime,path=path,samplers=samplers)

class Measurements(object):
	'''
	Measurements()

	A Measurements object is simply a hub to which multiple Measurement objects
	can be connected; and from there, easily called.
	'''

	def _add(self,name,measurement):
		if not re.match('^[a-zA-Z][a-zA-Z\_]*$',name):
			raise ValueError, "'%s' Is an invalid name for a measurement." % name

		if not isinstance(measurement,Measurement):
			raise ValueError, "Supplied measurement must be an instance of Measurement."

		setattr(self,name,measurement)

	def _remove(self,name):
		delattr(self,name)

class IteratorCallback(IntegratorCallback):
	'''
	IteratorCallback is an object that handles reporting progress through the ranges,
	in the event that multiprocessing is disabled.
	'''

	def __init__(self,levels_info,counts):
		self.levels_info = levels_info
		self.counts = counts
		self.count = np.prod(counts)
		self._progress = -1

		self.last_identifier = None
	#
	# Trigger start
	def onStart(self):
		pass

	def getCompleted(self):
		completed = 0
		for i,level_info in enumerate(self.levels_info):
			completed += (self.last_identifier[i])*np.prod(self.counts[i+1:])
		return completed

	def getProgress(self,progress):
		return (float(self.getCompleted())+progress)/self.count
	#
	# Receives a float value in [0,1].
	def onProgress(self, progress, identifier=None):
		self.last_identifier = identifier

		if round(progress*100.,0) > self._progress+5 or progress < self._progress:
			self._progress = round(progress*100.,0)
			sys.stderr.write("\r %3d%% | " % (100.*self.getProgress(progress)) )
			for i,level_info in enumerate(self.levels_info):
				sys.stderr.write("%s: "%level_info["name"])
				sys.stderr.write( coloured('%d of %d' % (identifier[i]+1,level_info['count']),"BLUE",True) )
				sys.stderr.write(" | ")

	#
	# Called when function is complete.
	# status = 0 -> OKAY
	# status = 1 -> WARN
	# status = 2 -> ERROR
	# Status = string message
	def onComplete(self,identifier=None,message=None,status=0):
		if self.getCompleted() == self.count:
			print


### Example Measurement Operators

class Amplitude(Measurement):
	'''
	Amplitude is a sample Measurement subclass that measures the amplitude of being
	in certain basis states as function of time throughout some state evolution.
	'''

	def init(self):
		pass

	def result_type(self,*args,**kwargs):
		return [
					('time',float),
					('amplitude',float,(self._system.dim,) )
				]

	def result_shape(self,*args,**kwargs):
		return (len(kwargs['y_0s']),len(kwargs.get('times',0)))

	def measure(self,times,y_0s,params={},subspace=None,**kwargs):
		r = self._system.integrate(times,y_0s,params=params,**kwargs)

		rval = np.empty((len(r),len(times)),dtype=self.result_type(y_0s=y_0s,times=times))

		self.__P = None
		for i,resultset in enumerate(r):
			for j,time in enumerate(resultset['time']):
				rval[i,j] = (time,self.amplitudes(resultset['state'][j]))

		return rval

	def amplitudes(self,state):
		if len(state.shape) > 1:
			return np.abs(np.diag(state))
		return np.abs(state)**2
