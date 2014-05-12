from abc import ABCMeta, abstractmethod, abstractproperty
import copy
import math
import os
import re
import shelve
import sys

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

		return a
	
	def _iterate_results_add(self,resultsObj=None,result=None,indicies=None,levels_info=None,params={}):
		'''
		This is a generic update for a particular iteration of the Measurements.iterate. It can be 
		overloaded if necessary.
		'''
		resultsObj[indicies] = result
	
	def iterate(self,ranges,params={},nprocs=None,*args,**kwargs):
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
		
		results = self._iterate_results_init(ranges=ranges,shape=tuple(counts),params=params,*args,**kwargs)

		def vary_pams(level=0,levels_info=[],output=[]):
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
					if self.multiprocessing:
						kwargs2 = copy.copy(kwargs)
						kwargs2['params'] = copy.copy(params)
						kwargs2['callback_fallback'] = False
						output.append( (
								tuple(x['iteration']-1 for x in levels_info),
								copy.copy(args),
								kwargs2
							) )
					else:
						self._iterate_results_add(resultsObj=results,result=self.measure(*args,params=params,callback=callback,**kwargs),indicies=tuple(x['iteration']-1 for x in levels_info),levels_info=levels_info,params=params)

			if self.multiprocessing:
				return output
		
		output = vary_pams(levels_info=levels_info)
		if self.multiprocessing:
			res = apm.map(output)
			for i,value in res:
				self._iterate_results_add(resultsObj=results,result=value,indicies=i)
		return ranges,results

	def iterate_to_file(self,path,*args,**kwargs):
		'''
		Measurement.iterate_to_file saves the results of the Measurement.iterate method
		to a python shelve file at `path`; all other arguments are passed through to the
		Measurement.iterate method.
		'''

		if not os.path.exists(path):

			if not os.path.exists(os.path.dirname(path)):
				os.makedirs(os.path.dirname(path))
			elif os.path.exists(os.path.dirname(path)) and os.path.isfile(os.path.dirname(path)):
				raise RuntimeError, "Destination path '%s' is a file."%os.path.dirname(path)

			ranges,results = self.iterate(*args,**kwargs)

			s = shelve.open(path)
			s['ranges'] = ranges
			s['results'] = results
			s.close()
		else:
			s = shelve.open(path)
			ranges = s['ranges']
			results = s['results']
			s.close()

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
	
	def measure(self,times,y_0s,params={},subspace=None,basis=None,**kwargs):
		r = self._system.integrate(times,y_0s,params=params,basis=basis,**kwargs)

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
