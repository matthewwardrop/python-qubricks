from abc import ABCMeta, abstractmethod
import copy
import time
import inspect
import os
import re
import shelve
import sys
import types

import numpy as np

from .integrator import IntegratorCallback
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
	ranges_type = [('ranges', 'object')]
	# axis_type = [('label','object'),('units','object'),('values',float)]

	def __init__(self, *args, **kwargs):
		self.init(*args, **kwargs)

	@property
	def _system(self):
		return self.__system
	@_system.setter
	def _system(self, value):
		self.__system = value

	@abstractmethod
	def init(self, **kwargs):
		'''
		Measurement.init should be specified by measurement subclasses.
		'''
		raise NotImplementedError("Measurement.init has not been implemented.")

	def __call__(self, *args, **kwargs):
		'''
		Calling a Measurement object is shorthand for calling the `measure` method.
		'''
		return self.measure(*args, **kwargs)

	@abstractmethod
	def measure(self, times, y_0s, params={}, **kwargs):
		'''
		Measurement.measure is where the grunt work is done; and it should return
		a numpy array of type Measurement.result_type, with shape
		Measurement.result_shape . Otherwise, anything is permitted in this method.
		'''
		raise NotImplementedError("Measurement.measure has not been implemented.")

	@abstractmethod
	def result_type(self, *args, **kwargs):
		'''
		Measurement.result_type should return an object suitable for use as the dtype
		argument in numpy. Otherwise, no restrictions; other than that it must also
		agree with the datatype returned by Measurement.measure.
		'''
		raise NotImplementedError("Measurement.result_type has not been implemented.")

	@abstractmethod
	def result_shape(self, *args, **kwargs):
		'''
		Measurement.result_shape should agree with the shape of the numpy array returned
		by Measurement.measure, but otherwise has no restrictions.
		'''
		raise NotImplementedError("Measurement.result_shape has not been implemented.")

	def _iterate_results_init(self, ranges=[], shape=None, params={}, *args, **kwargs):
		'''
		This is a generic initialisation for the Measurement object. It can be overloaded
		if necessary.
		'''
		# Initialise empty array to store results
		dtype = self.result_type(ranges=ranges, shape=shape, params=params, *args, **kwargs)
		rshape = self.result_shape(ranges=ranges, shape=shape, params=params, *args, **kwargs)
		if rshape is not None:
			shape = shape + rshape

		a = np.empty(shape, dtype=dtype)
		a.fill(np.nan)

		return a

	def _iterate_results_add(self, resultsObj=None, result=None, indicies=None, params={}):
		'''
		This is a generic update for a particular iteration of the Measurements.iterate. It can be
		overloaded if necessary.
		'''
		resultsObj[indicies] = result

	@property
	def _integration_independent(self):
		'''
		Override this with a value of True if this Measurement object does all the required integration
		internally.
		'''
		return False


class MeasurementResults(object):

	def __init__(self, ranges, ranges_eval, results, runtime=None, path=None, samplers={}):
		self.ranges = ranges
		self.ranges_eval = ranges_eval
		self.results = results
		self.runtime = 0 if runtime is None else runtime
		self.path = path
		self.samplers = samplers

		self.__check_sanity()

	def __check_sanity(self):
		if type(self.results) is None:
			raise ValueError("Improperly initialised result data.")
		if type(self.results) is not dict:
			new_name = raw_input("Old data format detect. Upgrading: Please enter a name for the current result data (should match the measurement name): ")
			self.results = {new_name: self.results}
			self.save() # Commit changes to new format

	def update(self, **kwargs):
		for key, value in kwargs.items():
			if key not in ['ranges', 'ranges_eval', 'data', 'runtime', 'path', 'samplers']:
				raise ValueError("Invalid update key: %s" % key)
			if key is "runtime":
				self.runtime += value
			else:
				setattr(self, key, value)
		return self

	@property
	def is_complete(self):
		for data in self.results.values():
			if len(np.where(np.isnan(data.view('float')))[0]) != 0:
				return False
		return True

	@property
	def continue_mask(self):
		def continue_mask(indicies, ranges=None, params={}):  # Todo: explore other mask options
			for data in self.results.values():
				if np.any(np.isnan(data[indicies].view('float'))):
					return True
			return False
		return continue_mask

	@staticmethod
	def _process_ranges(ranges, defunc=False, samplers={}):
		if ranges is None:
			return ranges
		if type(ranges) is dict:
			ranges = [copy.deepcopy(ranges)]
		else:
			ranges = copy.deepcopy(ranges)
		for range in ranges:
			for param, spec in range.items():
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

	def save(self, path=None, samplers=None):
		if path is None:
			path = self.path
		if samplers is None:
			samplers = self.samplers
		else:
			self.samplers = samplers
		if path is None:
			raise ValueError("Output file was not specified.")

		s = shelve.open(path)
		s['ranges'] = MeasurementResults._process_ranges(self.ranges, defunc=True, samplers=samplers)
		s['ranges_eval'] = self.ranges_eval
		s['results'] = self.results
		s['runtime'] = self.runtime
		s.close()

	@classmethod
	def load(cls, path, samplers={}):
		s = shelve.open(path)
		ranges = MeasurementResults._process_ranges(s.get('ranges'), defunc=False, samplers=samplers)
		ranges_eval = s.get('ranges_eval')
		results = s.get('results')
		runtime = s.get('runtime')
		s.close()
		return cls(ranges=ranges, ranges_eval=ranges_eval, results=results, runtime=runtime, path=path, samplers=samplers)


class Measurements(object):
	'''
	Measurements()

	A Measurements object is simply a hub to which multiple Measurement objects
	can be connected; and from there, easily called. Used by QuantumSystem objects.
	'''

	def __init__(self, system):
		self.__system = system
		self.__measurements = {}

	def _add(self, name, measurement):
		if not re.match('^[a-zA-Z][a-zA-Z0-9\_]*$', name):
			raise ValueError("'%s' Is an invalid name for a measurement." % name)

		if not isinstance(measurement, Measurement):
			raise ValueError("Supplied measurement must be an instance of Measurement.")

		self.__measurements[name] = measurement
		measurement._system = self.__system

	def _remove(self, name):
		return self.__measurements.pop(name)

	# User interaction
	def __getattr__(self, name):
		return MeasurementWrapper(self.__system, {name: self.__measurements[name]})

	def __call__(self, *names):
		meas = {}
		for name in names:
			meas[name] = self.__measurements[name]
		return MeasurementWrapper(self.__system, meas)


class MeasurementWrapper(object):

	def __init__(self, system, measurements={}):
		self.measurements = {}
		self._system = system

		self.add_measurements(**measurements)

	@property
	def __integration_needed(self):
		for meas in self.measurements.values():
			if not meas._integration_independent:
				return True
		return False

	def add_measurements(self, **measurements):
		self.measurements.update(measurements)

	def on(self, data, **kwargs):

		psi_0s = kwargs.get('psi_0s')
		times = kwargs.get('times')
		if data is not None:
			if psi_0s is None:
				kwargs['psi_0s'] = data['state'][:, 0]
			if times is None:
				kwargs['times'] = data['time'][0, :]
		else:
			if 'psi_0s' in kwargs and psi_0s is None:
				kwargs.pop('psi_0s')
			if 'times' in kwargs and times is None:
				kwargs.pop('times')

		if len(self.measurements) == 1:
			if self.measurements.values()[0]._integration_independent:
				return self.measurements.values()[0].measure(**kwargs)
			else:
				return self.measurements.values()[0].measure(data, **kwargs)

		res = {}
		for name, measurement in self.measurements.items():
			if measurement._integration_independent:
				res[name] = measurement.measure(**kwargs)
			else:
				res[name] = measurement.measure(data, **kwargs)
		return res

	def integrate(self, times=None, psi_0s=None, params={}, **kwargs):
		int_kwargs = {}
		for kwarg in kwargs:
			if kwarg.startswith('int_'):
				int_kwargs[kwarg.replace[4:]] = kwargs.pop(kwarg)
			if kwarg in inspect.getargspec(self._system.get_integrator).args:
				int_kwargs[kwarg] = kwargs[kwarg]

		if self.__integration_needed:
			return self.on(self._system.integrate(times=times, psi_0s=psi_0s, params=params, **int_kwargs), params=params, **kwargs)
		else:
			return self.on(None, times=times, psi_0s=psi_0s, params=params, **kwargs)

	def iterate_yielder(self, ranges, params={}, masks=None, nprocs=None, yield_every=60, results=None, **kwargs):
		# yield_every is the minimum/maximum number of seconds to go without yielding

		iterator = self._system.p.ranges_iterator(ranges)

		kwargs['callback_fallback'] = False

		ranges_eval, indicies = iterator.ranges_expand(masks=masks, ranges_eval=None, params=params)
		if results is None:
			data = {}
			for name, meas in self.measurements.items():
				data[name] = meas._iterate_results_init(ranges=ranges, shape=ranges_eval.shape, params=params, **kwargs)

			results = MeasurementResults(ranges, ranges_eval, data)
		else:
			if not struct_allclose(ranges_eval, results.ranges_eval, rtol=1e-15, atol=1e-15):
				if not raw_input("Attempted to resume measurement collection on a result set with different parameter ranges. Continue anyway? (y/N) ").lower().startswith('y'):
					raise ValueError("Stopping.")
			if type(results.results) != dict:
				results.update(ranges=ranges, ranges_eval=ranges_eval, data={self.measurements.keys()[0]: data})

		def splitlist(l, length=None):
			if length is None:
				yield l
			else:
				for i in xrange(0, len(l), length):
					yield l[i:i + length]

		t_start = time.time()
		data = results.results

		for i, (indicies, result) in enumerate(iterator.iterate(self.integrate, function_kwargs=kwargs, params=params, masks=masks, nprocs=nprocs, ranges_eval=ranges_eval)):
			if type(result) is not dict:
				self.measurements[self.measurements.keys()[0]]._iterate_results_add(resultsObj=data[self.measurements.keys()[0]], result=result, indicies=indicies)
			else:
				for name, value in result.items():
					self.measurements[name]._iterate_results_add(resultsObj=data[name], result=value, indicies=indicies)
			if yield_every is not None and time.time() - t_start >= yield_every:
				yield results.update(data=data, runtime=time.time() - t_start)
				t_start = time.time()

		yield results.update(data=data, runtime=time.time() - t_start)

	def iterate(self, *args, **kwargs):
		'''
		A wrapper around the `Measurement.iterate_yielder` method in the event that
		one does not want to deal with a generator object. This simply returns
		the final result.
		'''
		from collections import deque
		return deque(self.iterate_yielder(*args, yield_every=None, **kwargs), maxlen=1).pop()

	def iterate_to_file(self, path, samplers={}, *args, **kwargs):
		'''
		Measurement.iterate_to_file saves the results of the Measurement.iterate method
		to a python shelve file at `path`; all other arguments are passed through to the
		Measurement.iterate method.
		'''

		if os.path.dirname(path) is not "" and not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		elif os.path.exists(os.path.dirname(path)) and os.path.isfile(os.path.dirname(path)):
			raise RuntimeError("Destination path '%s' is a file." % os.path.dirname(path))

		results = None
		if os.path.isfile(path):
			results = MeasurementResults.load(path, samplers=samplers)
			if results.results is not None:

				if results.is_complete:
					return results

				masks = kwargs.get('masks', None)
				if masks is None:
					masks = []
				masks.append(results.continue_mask)
				kwargs['masks'] = masks

				print coloured("Attempting to continue data collection...", "YELLOW", True)
			else:
				results = None

		for results in self.iterate_yielder(*args, results=results, **kwargs):
			results.save(path=path, samplers=samplers)

		return results


class IteratorCallback(IntegratorCallback):
	'''
	IteratorCallback is an object that handles reporting progress through the ranges,
	in the event that multiprocessing is disabled.
	'''

	def __init__(self, levels_info, counts):
		self.levels_info = levels_info
		self.counts = counts
		self.count = np.prod(counts)
		self._progress = -1

		self.last_identifier = None

	# Trigger start
	def onStart(self):
		pass

	def getCompleted(self):
		completed = 0
		for i, level_info in enumerate(self.levels_info):
			completed += (self.last_identifier[i]) * np.prod(self.counts[i + 1:])
		return completed

	def getProgress(self, progress):
		return (float(self.getCompleted()) + progress) / self.count

	# Receives a float value in [0,1].
	def onProgress(self, progress, identifier=None):
		self.last_identifier = identifier

		if round(progress * 100., 0) > self._progress + 5 or progress < self._progress:
			self._progress = round(progress * 100., 0)
			sys.stderr.write("\r %3d%% | " % (100. * self.getProgress(progress)))
			for i, level_info in enumerate(self.levels_info):
				sys.stderr.write("%s: " % level_info["name"])
				sys.stderr.write(coloured('%d of %d' % (identifier[i] + 1, level_info['count']), "BLUE", True))
				sys.stderr.write(" | ")

	#
	# Called when function is complete.
	# status = 0 -> OKAY
	# status = 1 -> WARN
	# status = 2 -> ERROR
	# Status = string message
	def onComplete(self, identifier=None, message=None, status=0):
		if self.getCompleted() == self.count:
			print
