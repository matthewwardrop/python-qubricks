from abc import ABCMeta, abstractmethod, abstractproperty
import copy
import math
import os
import re
import shelve
import sys
import datetime

import numpy as np

from .integrator import IntegratorCallback
from .operators import StateOperator
from .utility.text import colour_text as coloured


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
		results as a tuple: (ranges,results) . This can be used to save the results iteratively
		as the calculations are performed; which may help to prevent data loss in the event
		of interruptions like power loss. If `yield_every` is supposed to be None, you should
		use `Measurement.iterate` instead.
		'''
		# Check if ranges is just a single range, and if so make it a list
		if isinstance(ranges,dict):
			ranges = [ranges]
		
		def range_length(param,pam_range):
			if isinstance(pam_range,list):
				return len(pam_range)
			if isinstance(pam_range,tuple):
				assert len(pam_range) >= 3, "Tuple specification incorrect for %s: %s" % (param,pam_range)
				return pam_range[2]
			raise ValueError("Unknown range format for %s: %s" % (param, pam_range))
		
		# Check to see if all parameters in this level have the same number of steps
		counts = []
		for pam_ranges in ranges:
			count = None
			for key in list(pam_ranges.keys()):
				c = range_length(key,pam_ranges[key])
				count = c if count is None else count
				if c != count: 
					raise ValueError, "Parameter ranges are not consistent in count: %s" % pam_ranges[key]
			counts.append( count )
		
		levels_info = [None]*len(ranges)

		if self.multiprocessing:
			from qubricks.utility.symmetric import AsyncParallelMap 
			apm = AsyncParallelMap(self.measure,progress=True,nprocs=nprocs)
			callback = None
		else:
			apm = None
			callback = IteratorCallback(levels_info,counts)
		
		results_new = self._iterate_results_init(ranges=ranges,shape=tuple(counts),params=params,*args,**kwargs)
		if results is None:
			results = results_new
		else:
			if results.shape != results_new.shape:
				raise ValueError("Invalid results given to continue. Shape %s does not agree with result dimensions %s" % (results.shape,results_new.shape))
			if results.dtype != results_new.dtype:
				raise ValueError("Invalid results given to continue. Type %s does not agree with result type %s" % (results.dtype,results_new.dtype))

		def vary_pams(level=0,levels_info=[],output=[]):
			'''
			This method generates a list of different parameter configurations
			'''
			level_info = {
				'name': ",".join(ranges[level].keys()),
				'count': counts[level],
				'iteration': 0,
			}
			levels_info[level] = level_info
			
			pam_ranges = ranges[level]

			## Interpret ranges
			pam_values = {}
			for param, pam_range in pam_ranges.items():
				tparams = params.copy()
				tparams[param] = pam_range
				pam_values[param] = self._system.p.range(param,**tparams)

			for i in xrange(counts[level]):
				level_info['iteration'] = i + 1
				
				# Update parameters
				for param, pam_value in pam_values.items():
					params[param] = pam_value[i]
				
				if level < len(ranges) - 1:
					# Recurse problem
					vary_pams(level=level+1,levels_info=levels_info,output=output)
				else:
					iteration = tuple(x['iteration']-1 for x in levels_info)
					if masks is not None and isinstance(masks,list):
						skip = False
						for mask in masks:
							if not mask(iteration,ranges=ranges,params=params):
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

			if self.multiprocessing:
				return output
		
		output = vary_pams(levels_info=levels_info)
		
		def splitlist(l,length=None):
			if length is None:
				yield l
			else:
				for i in xrange(0,len(l),length):
					yield l[i:i+length]
		
		start_time = datetime.datetime.now()
		for i,tasks in enumerate(splitlist(output,yield_every)):
			if self.multiprocessing:
				res = apm.map(tasks,count_offset= (yield_every*i if yield_every is not None else None),count_total=len(output),start_time=start_time )
			else:
				res = [ (indicies,self.measure(*args,callback=callback,**kwargs)) for indicies,args,kwargs in tasks] # TODO: neaten legacy mode callback
			for indicies,value in res:
				self._iterate_results_add(resultsObj=results,result=value,indicies=indicies)
			
			yield (ranges,results)
	
	def iterate(self,*args,**kwargs):
		'''
		A wrapper around the `Measurement.iterate_yielder` method in the event that 
		one does not want to deal with a generator object. This simply returns
		the final result.
		'''
		from collections import deque
		return deque(self.iterate_yielder(*args, yield_every=None, **kwargs),maxlen=1).pop()

	def iterate_to_file(self,path,*args,**kwargs):
		'''
		Measurement.iterate_to_file saves the results of the Measurement.iterate method
		to a python shelve file at `path`; all other arguments are passed through to the
		Measurement.iterate method.
		'''
		
		def save(ranges,results):
			s = shelve.open(path)
			s['ranges'] = ranges
			s['results'] = results
			s.close()
		
		def get():
			s = shelve.open(path)
			ranges = s['ranges']
			results = s['results']
			s.close()
			return ranges,results

		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		elif os.path.exists(os.path.dirname(path)) and os.path.isfile(os.path.dirname(path)):
			raise RuntimeError, "Destination path '%s' is a file."%os.path.dirname(path)
		
		if os.path.isfile(path):
			ranges,results = get()
			
			if len(np.where(np.isnan(results.view('float')))[0]) == 0:
				return ranges,results
				
			def continue_mask(indicies, ranges=None, params={}): # Todo: explore other mask options
				return np.any(np.isnan(results[indicies].view('float')))
			
			masks = kwargs.get('masks',None)
			if masks is None:
				masks = []
			masks.append( continue_mask )
			kwargs['masks'] = masks
			
			coloured("Attempting to continue data collection...","YELLOW",True)
		else:
			results = None
	
		for ranges,results in self.iterate_yielder(*args,results=results,**kwargs):
			save(ranges,results)

		return ranges,results


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
	#
	# Trigger start
	def onStart(self):
		pass
	
	def getCompleted(self):
		completed = 0
		for i,level_info in enumerate(self.levels_info):
			completed += (level_info['iteration']-1)*np.prod(self.counts[i+1:])
		return completed
	
	def getProgress(self,progress):
		return (float(self.getCompleted())+progress)/self.count
	#
	# Receives a float value in [0,1].
	def onProgress(self, progress):
		if round(progress*100.,0) > self._progress+5 or progress < self._progress:
			self._progress = round(progress*100.,0)
			sys.stderr.write("\r %3d%% | " % (100.*self.getProgress(progress)) )
			for level_info in self.levels_info:
				sys.stderr.write("%s: "%level_info["name"])
				sys.stderr.write( coloured('%d of %d' % (level_info['iteration'],level_info['count']),"BLUE",True) )
				sys.stderr.write(" | ")
	
	#
	# Called when function is complete.
	# Level = 0 -> OKAY
	# Level = 1 -> WARN
	# Level = 2 -> ERROR
	# Status = string message
	def onComplete(self,message=None,level=0):
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
