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
	A Measurement instance is an object which encodes the logic required to extract
	information from a QuantumSystem object. It specifies the algorithm
	to be performed to extract the measurement outcome, the type of the measurement
	results, and also provides methods to initialise and add results to storage
	when performing the same measurement iteratively. Measurement is an
	abstract class, and should be subclassed for each new class of measurements.

	While Measurement objects can be used directly, they are typically used in conjunction
	with Measurements and MeasurementWrapper, as documented in those classes.

	Any arguments and/or keyword arguments passed to the Measurement constructor
	are passed to Measurement.init.

	Subclassing Measurement:
		A subclass of Measurement must implement the following methods:
			- init
			- measure
			- result_type
			- result_shape
		A subclass *may* also override the following methods to further
		customise behaviour:
			- is_independent
			- iterate_results_init
			- iterate_results_add
			- iterate_is_complete
			- iterate_continue_mask
		Documentation for these methods is provided below.

	Applying Measurement instances:
		Although not normally used directly, you can use a Measurement instance
		directly on the results of an QuantumSystem integration, for example:

		>>> measurement(data=system.integrate(...))

		Calling a Measurement instance is an alias for the Measurement.measure
		method. If the measurement instance is configured to perform its own integration:

		>>> measurement(times=..., initial=..., ...)

		Note that if the measurement instance needs access to the QuantumSystem
		instance, you can setup the reference using:

		>>> measurement.system = system
	'''
	__metaclass__ = ABCMeta

	def __init__(self, *args, **kwargs):
		self.__system = None
		self.init(*args, **kwargs)

	def __call__(self, *args, **kwargs):
		return self.measure(*args, **kwargs)

	@property
	def system(self):
		'''
		A reference to a QuantumSystem instance. If a system instance is not provided,
		and an attempt to access this property is made, a RuntimeException is raised.

		You can specify a QuantumSystem using:

		>>> measurement.system = system
		'''
		if self.__system is None:
			raise ValueError("QuantumSystem instance required by Measurement object, but a QuantumSystem object has not been configured.")
		return self.__system
	@system.setter
	def system(self, value):
		if not isinstance(value, QuantumSystem):
			raise ValueError("Specified value must be a `QuantumSystem` instance.")
		self.__system = value

	@abstractmethod
	def init(self):
		'''
		This method should initialise the Measurement instance in whatever way
		is necessary to prepare the instance for use. Note that any arguments
		passed to the Measurement constructor will also be passed to this method.
		There is no restriction on the method signature for the init method.
		'''
		raise NotImplementedError("Measurement.init has not been implemented.")

	@abstractmethod
	def measure(self, data=None, times=None, initial=None, params={}, **kwargs):
		'''
		This method should return the value of a measurement as a numpy array with
		data type and shape as specified in `result_type` and `result_shape` respectively.

		.. note:: It is possible to return types other than numpy array and still
			be compatible with iteration (see MeasurementWrapper) provided you overload
			the `iterate_results_init` and `iterate_results_add` methods.

		Implementations of `measure` will typically be provided by integration data
		by a `MeasurementWrapper` instance (which will be a structured numpy array
		as returned by `Integrator.integrate) as the value for the `data` keyword.
		A consistent set of values for `times` and `initial` will also be passed.

		.. note:: If an implementation of `measure` omits the `data` keyword, QuBricks
			assumes that all integration required by the `measure` operator will be
			performed internally. It can use the reference to a QuantumSystem
			instance at `Measurement.system` for this purpose. If the `data` keyword
			is present (for testing/etc), but pre-computed integration data is undesired,
			override the `is_independent` method to return `True`. If external data
			is *required*, then simply remove the default value of `data`.

		Apart from the required keywords: `data`, `times`, `initial` and `params`; any additional
		keywords can be specified. Refer to the documentation of `MeasurementWrapper` to
		see how their values will filter through.

		.. note:: Although the keywords `times` and `initial` are necessary, it is not
			necessary to use these keywords. As such, Measurement operators need not
			require an integration of the physical system.

		:param data: Data from a QuantumSystem.integrate call, or None.
		:type data: numpy.ndarray or None
		:param times: Sequence of times of interest.
		:type times: iterable
		:param initial: The initial state vectors/ensembles with which to start integrating.
		:type initial: str or iterable
		:param params: Parameter context to use during this measurement. Parameter types can be anything supported by Parameters.
		:type params: dict
		:param kwargs: Any other keyword arguments not collected explicitly.
		:type kwargs: dict
		'''
		raise NotImplementedError("Measurement.measure has not been implemented.")

	@abstractmethod
	def result_type(self, *args, **kwargs):
		'''
		This method should return an object suitable for use as the dtype
		argument in a numpy array constructor. Otherwise, no restrictions; other than that it must also
		agree with the data-type returned by `Measurement.measure`.

		This method will receive all arguments and keyword arguments passed to
		`iterate_results_init`, where it is used to initialise the storage of
		measurement results.
		'''
		raise NotImplementedError("Measurement.result_type has not been implemented.")

	@abstractmethod
	def result_shape(self, *args, **kwargs):
		'''
		This method should return a tuple describing the shape of the numpy array to be returned
		by Measurement.measure.

		This method will receive all arguments and keyword arguments passed to
		`iterate_results_init`, where it is used to initialise the storage of
		measurement results.
		'''
		raise NotImplementedError("Measurement.result_shape has not been implemented.")

	def iterate_results_init(self, ranges=[], shape=None, params={}, *args, **kwargs):
		'''
		This method is called by `MeasurementWrapper.iterate_yielder` to initialise
		the storage of the measurement results returned by this object. By default, this
		method returns a numpy array with dtype as specified by `result_type` and shape
		returned by `result_shape`, with all entries set to np.nan objects. If necessary,
		you can overload this method to provide a different storage container
		This is a generic initialisation for the Measurement object. It can be overloaded
		if necessary.

		:param ranges: The range specifications provided to MeasurementWrapper.iterate_yielder.
		:type range: list of dict
		:param shape: The shape of the resulting evaluated ranges.
		:type shape: tuple
		:param params: The parameter context of the ranges.
		:type params: dict
		:param args: Any additional arguments passed to MeasurementWrapper.iterate_yielder.
		:type args: tuple
		:param kwargs: Any additional keyword arguments passed to MeasurementWrapper.iterate_yielder.
		:type kwargs: dict
		'''
		# Initialise empty array to store results
		dtype = self.result_type(ranges=ranges, shape=shape, params=params, *args, **kwargs)
		rshape = self.result_shape(ranges=ranges, shape=shape, params=params, *args, **kwargs)
		if rshape is not None:
			shape = shape + rshape

		a = np.empty(shape, dtype=dtype)
		a.fill(np.nan)

		return a

	def iterate_results_add(self, results=None, result=None, indices=None, params={}):
		'''
		This method adds a measurement result `result` from `Measurement.measure` to the `results` object
		initialised in `Measurement.iterate_results_init`. It should put this result into storage at the
		appropriate location for the provided `indices`.

		:param results: The storage object in which to place the result (as from `Measurement.iterate_results_init`).
		:type results: object
		:param result: The result to be stored (as from `Measurement.measure`).
		:type result: object
		:param indices: The indices at which to store this result.
		:type indices: tuple
		:param params: The parameter context for this measurement result.
		:type params: dict
		'''
		results[indices] = result

	def iterate_continue_mask(self, results):
		'''
		This method returns a mask function (see `MeasurementWrapper` documentation), which
		in turn based on the `results` object (as initialised by `iterate_results_init`) returns
		True or False for a given set of indices indicating whether there already exists data
		for those indices.

		:param results: The results storage object (see Measurement.iterate_results_init).
		:type results: object
		'''
		def continue_mask(indices, ranges=None, params={}):
			if np.any(np.isnan(results[indices].view('float'))):
				return True
			return False
		return continue_mask

	def iterate_is_complete(self, results):
		'''
		This method returns `True` when the results object is completely specified (results
		have been added for all indices; and `False` otherwise.

		:param results: The results storage object (see Measurement.iterate_results_init).
		:type results: object
		'''
		if len(np.where(np.isnan(results.view('float')))[0]) != 0:
			return False
		return True

	@property
	def is_independent(self):
		'''
		`True` if this Measurement instance does all required integration internally (and so should
		not receive pre-computed integration data). `False` otherwise. The default implementation is
		`False`.
		'''
		return False


class MeasurementIterationResults(object):
	'''
	MeasurementIterationResults is class designed to store the results of measurements applied iteratively
	over a range of different values (see `MeasurementWrapper.iterate_yielder`). Apart from its role as a
	data structure, it also provides methods for saving and loading the data to/from disk.

	:param ranges: The specification of ranges passed ultimately to the Parameters instance.
	:type ranges: dict or list of dict
	:param ranges_eval: The values of the parameters after evaluation from the above specification.
	:type ranges_eval: numpy.ndarray
	:param results: A dictionary of measurement results, with keys of measurement names.
	:type results: dict
	:param runtime: An optional number indicating in seconds the time taken to generate the results.
	:type runtime: float
	:param path: The location to use as a storage location by default.
	:type path: str
	:param samplers: A dictionary of named samplers (see `parampy.Parameters.range`) for future use with `ranges`, since functions cannot be serialised.
	:type samplers: dict of callables

	Constructing a MeasurementIterationResults object:
		Manually constructing a MeasurementIterationResults instance is unusual, since this is handled
		for you by the MeasurementWrapper iteration methods. However, this is possible using:

		>>> results = MeasurementIterationResults(ranges=..., ranges_eval=..., results=..., runtime=1.2, path='data.dat', samplers=...

	Accessing results:
		To access the data stored in a MeasurementIterationResults instance, simply access the relevant
		attributes. The available attributes are:
			- ranges
			- ranges_eval
			- results
			- runtime
			- path
			- samplers
		Each of these attributes corresponds to the documented parameters described above.

		For example:

		>>> mresults = results.results['measurement_name']

		Note that all of these attributes can be freely overridden. Check out the
		`MeasurementIterationResults.update` method for an alternative to updating these
		results.
	'''

	def __init__(self, ranges, ranges_eval, results, params={}, runtime=None, path=None, samplers={}):
		self.ranges = ranges
		self.ranges_eval = ranges_eval
		self.results = results
		self.runtime = 0 if runtime is None else runtime
		self.path = path
		self.samplers = samplers
		self.params = params

		self.__check_sanity()

	def __check_sanity(self):
		if type(self.results) is None:
			raise ValueError("Improperly initialised result data.")
		if type(self.results) is not dict:
			new_name = raw_input("Old data format detect. Upgrading: Please enter a name for the current result data (should match the measurement name): ")
			self.results = {new_name: self.results}
			self.save() # Commit changes to new format

	def update(self, **kwargs):
		'''
		This method allows you to update the stored data of this `MeasurementIterationResults` instance.
		Simply call this method with keyword arguments of the relevant attributes. For example:

		>>> results.update(results=..., path=..., ...)

		Note that you can update multiple attributes at once. The one special argument is
		"runtime", which will increment that attribute with the specified value, rather than
		replacing it. For example:

		>>> results.runtime
		231.211311
		>>> results.update(runtime=2.2)
		>>> results.runtime
		233.411311
		'''
		for key, value in kwargs.items():
			if key not in ['ranges', 'ranges_eval', 'data', 'runtime', 'path', 'samplers', 'params']:
				raise ValueError("Invalid update key: %s" % key)
			if key is "runtime":
				self.runtime += value
			else:
				setattr(self, key, value)
		return self

	def is_complete(self, measurements={}):
		'''
		This method calls the `Measurement.iterate_is_complete` method with the appropriate
		results for each of the measurements provided. If False for any of these
		measurements, False is returned.

		:param measurements: A dictionary of measurement objects with keys indicating their names.
		:type measurements: dict
		'''
		for name, measurement in measurements.items():
			if self.results.get(name,None) is None:
				return False
			if not measurement.iterate_is_complete(self.results[name]):
				return False
		return True

	def continue_mask(self, measurements={}):
		'''
		This method provides a mask for the `parampy.iteration.RangesIterator` instance called in
		`MeasurementWrapper.iterate_yielder`. The provided mask calls the `Measurement.iterate_continue_mask`
		method with the appropriate results for each of the measurements provided in `measurements`.

		:param measurements: A dictionary of measurement objects with keys indicating their names.
		:type measurements: dict
		'''
		def continue_mask(indices, ranges=None, params={}):
			for name, measurement in measurements.items():
				data = self.results.get(name,None)
				if data is None:
					return True
				if measurement.iterate_continue_mask(data)(indices, ranges=ranges, params=params):
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
					sampler = spec[-1]
					for name, fn in samplers.items():
						if sampler == fn:
							sampler = name
							break
					if type(sampler) is not str:
						sampler = sampler.__name__
					spec[-1] = sampler
					spec = tuple(spec)
					range[param] = spec
				if not defunc and len(spec) > 3 and type(spec[-1]) == str and spec[-1] in samplers:
					spec = list(spec)
					spec[-1] = samplers[spec[-1]]
					spec = tuple(spec)
					range[param] = spec
		return ranges

	def save(self, path=None, samplers=None):
		'''
		This method will save this MeasurementIterationResults object to disk at the specified
		path, trading the sampler functions in the ranges attribute with their names extract from samplers
		(if possible), or by using their inspected name (using the `__name__` attribute). The
		resulting file is a "shelf" object from the python `shelve` module.

		:param path: A path to the file's destination. If not provided, the earlier provided path is used.
		:type path: str
		:param samplers: A dictionary of functions (or callables) indexed by string names.
		:type samplers: str

		For example:

		>>> results.save('data.dat')
		'''
		if path is None:
			path = self.path
		if samplers is None:
			samplers = self.samplers
		else:
			self.samplers = samplers
		if path is None:
			raise ValueError("Output file was not specified.")

		s = shelve.open(path)
		s['ranges'] = MeasurementIterationResults._process_ranges(self.ranges, defunc=True, samplers=samplers)
		s['ranges_eval'] = self.ranges_eval
		s['results'] = self.results
		s['runtime'] = self.runtime
		s.close()

	@classmethod
	def load(cls, path, samplers={}):
		'''
		This method will load and populate a new MeasurementIterationResults object from previously
		saved data. If provided, `samplers` will be used to convert string names of samplers in the
		ranges to functions.

		:param path: A path to the file's destination. If not provided, the earlier provided path is used.
		:type path: str
		:param samplers: A dictionary of functions (or callables) indexed by string names.
		:type samplers: str

		For example:

		>>> results = MeasurementIterationResults.load('data.dat')
		'''
		s = shelve.open(path)
		ranges = MeasurementIterationResults._process_ranges(s.get('ranges'), defunc=False, samplers=samplers)
		ranges_eval = s.get('ranges_eval')
		results = s.get('results')
		runtime = s.get('runtime')
		s.close()
		return cls(ranges=ranges, ranges_eval=ranges_eval, results=results, runtime=runtime, path=path, samplers=samplers)


class Measurements(object):
	'''
	Measurements is a designed to simplify the Measurement evaluation process
	by acting as a host for multiple named Measurement objects. This object
	is used as the `measure` attribute of `QuantumSystem` objects.

	:param system: A QuantumSystem instance.
	:type system: QuantumSystem

	Constructing a Measurements object:
		If you want to create a Measurements instance separate from the one hosted
		by `QuantumSystem` objects, use the following:

		>>> measurements = Measurements(system)

	Adding and removing Measurement objects:
		To add a Measurement object to a Measurements instance, you simply provide it
		a string name, and use (for example):

		>>> measurements._add("name", NamedMeasurement)

		where `NamedMeasurement` is a subclass of `Measurement`.

		To remove a `Measurement` from `Measurements`, use:

		>>> measurements._remove("name")

		The underscores preceding these methods' names are designed to prevent
		name clashes with potential Measurement names.

		When a `Measurement` instance is added to `Measurements`, its
		internal "system" attribute is updated to point to the `QuantumSystem`
		used by `Measurements`.

	Extracting a Measurement object:
		Once added to the `Measurements` object, a `Measurement` object can be
		accessed using attribute notation, or by calling the `Measurements`
		instance. For example:

		>>> system.measure.name

		Or to bundle multiple measurements up into the same evaluation:

		>>> system.measure("measurement_1", "measurement_2", ...)

		In both cases, the return type is **not** a `Measurement` instance,
		but rather a `MeasurementWrapper` instance, which can be used to
		perform the `Measurement.measure` operations in a simplified
		and consistent manner. See the `MeasurementWrapper` documentation for
		more information.

	Inspecting a Measurements instance:
		To see a list of the names of measurements stored in a `Measurements`
		instance, you can use:

		>>> measurements._names

		To get a reference of the dictionary internally used by `Measurements`
		to store and retrieve hosted `Measurement` objects, use:

		>>> measurements._measurements
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
		measurement.system = self.__system

	def _remove(self, name):
		return self.__measurements.pop(name)

	@property
	def _names(self):
		'''
		A list of strings corresponding to the names of the measurements hosted in this
		Measurements instances.
		'''
		return self.__measurements.keys()

	@property
	def _measurements(self):
		return self.__measurements

	# User interaction
	def __getattr__(self, name):
		return MeasurementWrapper(self.__system, {name: self.__measurements[name]})

	def __call__(self, *names):
		meas = {}
		for name in names:
			meas[name] = self.__measurements[name]
		return MeasurementWrapper(self.__system, meas)


class MeasurementWrapper(object):
	'''
	The `MeasurementWrapper` class wraps around one or more `Measurement` objects
	to provide a consistent API for performing (potentially) multiple measurements
	at once. There are also performance benefits to be had, since wherever possible
	integration results are shared between the `Measurement` instances.

	:param system: A QuantumSystem instance.
	:type system: QuantumSystem
	:param measurements: A dictionary of `Measurement` objects indexed by a string name.
	:type dict:

	Constructing MeasurementWrapper objects:
		The syntax for creating a `MeasurementWrapper` object is:

		>>> wrapper = MeasurementWrapper(system, {'name': NamedMeasurement, ...})

		where `NamedMeasurement` is a `Measurement` instance.

	Adding Measurement objects:
		If you want to add additional `Measurement` objects after creating the
		`MeasurementWrapper` instance, use the `add_measurements` method. Refer
		to the documentation below for more information.

	Performing Measurements:
		There are three basic procedures which you can use to perform measurements
		on the reference "system" `QuantumSystem` instance.

		The first of these is `MeasurementWrapper.on`, which applies the measurement(s)
		to a pre-computed data. The second is `MeasurementWrapper.integrate`, which
		applies the measurement(s) to data computed on the fly. And the last is
		the iteration procedures: `MeasurementWrapper.iterate`, `MeasurementWrapper.iterate_yielder`,
		and `MeasurementWrapper.iterate_to_file`; each of which allows you to perform
		measurements over a range of parameter contexts.

		Each of these methods is documented in more detail below.
	'''

	def __init__(self, system, measurements={}):
		self.measurements = {}
		self.system = system

		self.add_measurements(**measurements)

	@property
	def __integration_needed(self):
		for meas in self.measurements.values():
			if not meas.is_independent and 'data' in inspect.getargspec(meas.measure)[0]:
				return True
		return False

	def add_measurements(self, **measurements):
		'''
		This method adds named measurements to the `MeasurementWrapper`. The
		syntax for this is:

		>>> wrapper.add_measurements(name=NamedMeasurement, ...)

		where "name" is any valid measurement name, and `NamedMeasurement` is a
		`Measurement` instance.
		'''
		self.measurements.update(measurements)

	def on(self, data, **kwargs):
		'''
		This method applies the `Measurement.measure` method to `data` for
		every `Measurement` stored in this object. If there are two or more
		`Measurement` objects, this method returns a dictionary of `Measurement.measure` results; with
		keys being the measurement names. If there is only one `Measurement` object,
		the return value of `Measurement.measure` is returned.

		:param data: Data from an `QuantumSystem` integration.
		:type data: numpy.ndarray
		:param kwargs: Additional kwargs to pass to `Measurement.measure`.
		:type kwargs: dict

		For example:
		>>> wrapper.on(data, mykey=myvalue)

		Note that if `data` is not `None`, then `initial` and `times` are
		extracted from `data`, and passed to `Measurement.measure` as well.

		Also note that if a measurement has `Measurement.is_independent` being `True`,
		only the `initial` and `times` will be forwarded from `data`.
		'''

		initial = kwargs.get('initial')
		times = kwargs.get('times')
		if data is not None:
			if initial is None:
				kwargs['initial'] = data['state'][:, 0]
			if times is None:
				kwargs['times'] = data['time'][0, :]
		else:
			if 'initial' in kwargs and initial is None:
				kwargs.pop('initial')
			if 'times' in kwargs and times is None:
				kwargs.pop('times')

		if len(self.measurements) == 1:
			measurement = self.measurements.values()[0]
			if measurement.is_independent or 'data' not in inspect.getargspec(measurement.measure)[0]:
				return measurement.measure(**kwargs)
			else:
				return measurement.measure(data=data, **kwargs)

		res = {}
		for name, measurement in self.measurements.items():
			if measurement.is_independent or 'data' not in inspect.getargspec(measurement.measure)[0]:
				res[name] = measurement.measure(**kwargs)
			else:
				res[name] = measurement.measure(data=data, **kwargs)
		return res

	def integrate(self, times=None, initial=None, params={}, **kwargs):
		'''
		This method performs an integration of the `QuantumSystem` referenced when
		this object was constructed, and then calls `Measurement.on` on that data.
		If all `Measurement` objects hosted are "independent" (have `Measurement.is_independent`
		as `True`), then no integration is performed.

		:param times: Times for which to report the state during integration.
		:type times: iterable
		:param initial: Initial state vectors / ensembles for the integration. (See `QuantumSystem.state`.
		:type initial: list
		:param params: Parameter overrides to use during integration. (See `parampy.Parameters` documentation).
		:type param: dict
		:param kwargs: Additional keyword arguments to pass to `QuantumSystem.integrate` and `Measurement.measure`.
		:type kwargs: dict

		.. note:: Only keyword arguments prepended with 'int_' are forwarded to
		`QuantumSystem.integrate`, with the prefix removed. These keywords are not
		also passed to `Measurement.measure`.

		For example:

		>>> wrapper.integrate(times=['T'], initial=['logical0'])
		'''
		int_kwargs = {}
		for kwarg in kwargs.keys():
			if kwarg.startswith('int_'):
				int_kwargs[kwarg[4:]] = kwargs.pop(kwarg)
			if kwarg in inspect.getargspec(self.system.get_integrator).args:
				int_kwargs[kwarg] = kwargs[kwarg]

		if self.__integration_needed:
			return self.on(self.system.integrate(times=times, initial=initial, params=params, **int_kwargs), params=params, **kwargs)
		else:
			return self.on(None, times=times, initial=initial, params=params, **kwargs)

	def iterate_yielder(self, ranges, params={}, masks=None, nprocs=None, yield_every=0, results=None, progress=True, **kwargs):
		'''
		This method iterates over the possible Cartesian products of the parameter ranges provided,
		at each step running the `MeasurementWrapper.integrate` in the resulting parameter context.
		After every `yield_every` seconds, this method will flag that it needs to yield the results currently accumulated (as a
		`MeasurementIterationResults` object) when the next measurement result has finished computing. This means that
		you can, for example, progressively save (or plot) the results as they are taken. Note that if
		the processing of the results is slow, this can greatly increase the time it takes to finish
		the iteration.

		:param ranges: A valid ranges specification (see `parampy.iteration.RangesIterator`)
		:type ranges: list or dict
		:param params: Parameter overrides to use (see `parampy.Parameters.range`)
		:type params: dict
		:param masks: List of masks to use to filter indices to compute. (see `parampy.iteration.RangesIterator`)
		:type masks: list of callables
		:param nprocs: Number of processes to spawn (if 0 or 1 multithreading is not enabled) (see `parampy.iteration.RangesIterator`)
		:type nprocs: number or None
		:param yield_every: Minimum number of seconds to go without returning the next result. To yield the value after
			every successful computation, use yield_every=0 . If yield_every is None, results are returned
			only after every computation has succeeded. By default, yield_every = 0.
		:type yield_every: number or None
		:param results: Previously computed MeasurementIterationResults object to extend.
		:type results: MeasurementIterationResults
		:param progress: Whether progress information should be shown (True or False); or a callable. (see `parampy.iteration.RangesIterator` for more)
		:type progress: bool or callable
		:param kwargs: Additional keyword arguments to be passed to `MeasurementWrapper.integrate` (and also to
			`Measurement.iterate_results_init`.
		:type kwargs: dict
		'''

		# yield_every is the minimum/maximum number of seconds to go without yielding

		iterator = self.system.p.ranges_iterator(ranges, masks=masks, ranges_eval=None if results is None else results.ranges_eval, params=params, function=self.integrate, function_kwargs=kwargs, nprocs=nprocs, progress=progress)

		kwargs['progress'] = False

		ranges_eval, indices = iterator.ranges_expand()
		if results is None:
			data = {}
			for name, meas in self.measurements.items():
				data[name] = meas.iterate_results_init(ranges=ranges, shape=ranges_eval.shape, params=params, **kwargs)

			results = MeasurementIterationResults(ranges, ranges_eval, data, params=params)
		else:
			if not struct_allclose(ranges_eval, results.ranges_eval, rtol=1e-15, atol=1e-15):
				if not raw_input("Attempted to resume measurement collection on a result set with different parameter ranges. Continue anyway? (y/N) ").lower().startswith('y'):
					raise ValueError("Stopping.")
			if type(results.results) != dict:
				results.update(ranges=ranges, ranges_eval=ranges_eval, data={self.measurements.keys()[0]: results.results}, params=params)

		def splitlist(l, length=None):
			if length is None:
				yield l
			else:
				for i in xrange(0, len(l), length):
					yield l[i:i + length]

		t_start = time.time()
		data = results.results

		for i, (indices, result) in enumerate(iterator):
			if type(result) is not dict:
				self.measurements[self.measurements.keys()[0]].iterate_results_add(results=data[self.measurements.keys()[0]], result=result, indices=indices)
			else:
				for name, value in result.items():
					self.measurements[name].iterate_results_add(results=data[name], result=value, indices=indices)
			if yield_every is not None and time.time() - t_start >= yield_every:
				yield results.update(data=data, runtime=time.time() - t_start)
				t_start = time.time()

		yield results.update(data=data, runtime=time.time() - t_start)

	def iterate(self, *args, **kwargs):
		'''
		This method is a wrapper around the `Measurement.iterate_yielder` method in the event that
		one only cares about the final result, and does not want to deal with interim results. This
		method simply waits until the iteration process is complete, and returns the last result.

		All arguments and keyword arguments are passed to `MeasurementWrapper.iterate_yielder`.
		'''
		from collections import deque
		return deque(self.iterate_yielder(*args, yield_every=None, **kwargs), maxlen=1).pop()

	def iterate_to_file(self, path, samplers={}, yield_every=60, *args, **kwargs):
		'''
		This method wraps around `Measurement.iterate_yielder` in order to continue a previous
		measurement collection process (if it did not finish successfully) and to iteratively write the
		most recent results to a file. This method modifies the default `yield_every` of the
		`iterate_yielder` method to 60 seconds, so that file IO is not the limiting factor of performance,
		and so that at most around a minute's worth of processing is lost in the event that something
		goes wrong.

		:param path: The path at which to save results. If this file exists, attempts are made to continue
			the measurement acquisition.
		:type path: str
		:param samplers: A dictionary of samplers to be used when loading and saving the `MeasurementIterationResults`
			object. (see `MeasurementIterationResults.load` and `MeasurementIterationResults.save`)
		:type samplers: dict
		:param yield_every: The minimum time between attempts to save the results. (see `iterate_yielder`)
		:type yield_every: number or None
		:param args: Additional arguments to pass to `iterate_yielder`.
		:type args: tuple
		:param kwargs: Additional keyword arguments to pass to `iterate_yielder`.
		:type kwargs: dict

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
			results = MeasurementIterationResults.load(path=path, samplers=samplers)
			if results.results is not None:

				if results.is_complete(self.measurements):
					return results

				masks = kwargs.get('masks', None)
				if masks is None:
					masks = []
				masks.append(results.continue_mask(self.measurements))
				kwargs['masks'] = masks

				print coloured("Attempting to continue data collection...", "YELLOW", True)
			else:
				results = None

		for results in self.iterate_yielder(*args, results=results, yield_every=yield_every, **kwargs):
			results.save(path=path, samplers=samplers)

		return results

from .system import QuantumSystem
