
from abc import ABCMeta, abstractmethod
from qubricks.utility.text import colour_text as coloured
import sys
import traceback


def d_prepare(prepare):
	def wrapper(self, *args, **kwargs):
		self.__prepared = True
		return prepare(self, *args, **kwargs)
	return wrapper


def d_simulate(simulate):
	def wrapper(self):
		if not getattr(self, '__prepared', False):
			try:
				self.prepare()
			except Exception:
				raise ValueError("Not yet prepared. Attempts to prepare caused an exception:\n%s" % traceback.format_exc())
		if not hasattr(self, '__resulted'):
			print coloured("Simulating...", "YELLOW", True),
			sys.stdout.flush()
			self.__resulted = simulate(self)
			print coloured("DONE", "GREEN", True)
		return self.__resulted
	return wrapper


def d_process(process):
	def wrapper(self, results=None):
		if not hasattr(self, '__processed'):
			if results is not None:
				return process(self, results)
			if not getattr(self, '__resulted', None):
				self.simulate()

			print coloured("Processing data...", "YELLOW", True),
			sys.stdout.flush()
			p = process(self, self.__resulted)
			if type(p) is not dict:
				print coloured("Error", "RED", True)
				raise ValueError("ModelAnalysis.process() must return type dict.")
			print coloured("DONE", "GREEN", True)
			self.__processed = p
		return self.__processed
	return wrapper


def d_plot(plot):
	def wrapper(self, **kwargs):
		if getattr(self, '__processed', None) is None:
			self.process()
		print coloured("Generating plots...", "YELLOW", True),
		sys.stdout.flush()
		r = plot(self, **self.__processed)
		print coloured("DONE", "GREEN", True)
		return r
	return wrapper


class ModelAnalysis(object):
	'''
	ModelAnalysis()

	This is a helper object that can simplify the routine running of simulations.
	One can simply create a subclass of this class, and implement prepare, simulate,
	process, and plot; and this object will ensure that the methods are always run
	in order. Calling `process`, for example, will also call all of the previous methods
	in order.
	'''
	__metaclass__ = ABCMeta

	def __getattribute__(self, name):
		if name in ['prepare', 'process', 'simulate', 'plot']:
			return object.__getattribute__(self, '_%s' % name)
		else:
			return object.__getattribute__(self, name)

	@d_prepare
	def _prepare(self, *args, **kwargs):
		return object.__getattribute__(self, 'prepare')(*args, **kwargs)

	@d_simulate
	def _simulate(self):
		return object.__getattribute__(self, 'simulate')()

	@d_process
	def _process(self, results=None):
		return object.__getattribute__(self, 'process')(results)

	@d_plot
	def _plot(self, **kwargs):
		return object.__getattribute__(self, 'plot')(**kwargs)

	@abstractmethod
	def prepare(self, *args, **kwargs):
		pass

	@abstractmethod
	def simulate(self):
		pass

	@abstractmethod
	def process(self, results):
		pass

	@abstractmethod
	def plot(self, **kwargs):
		pass
