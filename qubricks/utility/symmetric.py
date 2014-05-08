
# Multiprocessing code for qubricks
# Inspired by: http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class

#WARNING: This module is currently under development.

import Queue
import multiprocessing, traceback, logging, resource
import sys, gc
import warnings

heap = None
def set_heap(hp):
	global heap
	heap = hp

def get_heap(hp):
	global heap
	return heap

def error(msg, *args):
	return multiprocessing.get_logger().error(msg, *args)

def warn(msg, *args):
	return multiprocessing.get_logger().warn(msg, *args)

def spawn(f):
	def fun(q_in, q_out):
		warnings.simplefilter("ignore")

		while True:
			i, args, kwargs = q_in.get()
			if i is None:
				break
			
			try:
				r = f(*args, **kwargs)
			except Exception as e:
				error(traceback.format_exc())
				raise e

			q_out.put((i, r))
			
			gc.collect()
			warn('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
			
	return fun

def spawnonce(f):
	def fun(q_in, q_out):
		warnings.simplefilter("ignore")

		i, args, kwargs = q_in.get()
		if i is None:
			return
		
		r = None
		try:
			r = f(*args, **kwargs)
		except Exception as e:
			error(traceback.format_exc())
			raise e

		q_out.put((i, r))
			
	return fun

'''def parmap(f, X, nprocs = multiprocessing.cpu_count()):
	q_in   = multiprocessing.Queue(1)
	q_out  = multiprocessing.Queue()

	proc = [multiprocessing.Process(target=spawn(f),args=(q_in,q_out)) for _ in range(nprocs)]
	for p in proc:
		p.daemon = True
		p.start()

	sent = [q_in.put(x) for x in X]
	[q_in.put((None,None,None)) for _ in range(nprocs)]
	res = [q_out.get() for _ in range(len(sent))]

	[p.join() for p in proc]

	return [(i,x) for i,x in sorted(res)]'''

class AsyncParallelMap(object):

	#DEBUG = False

	def __init__(self, f, progress=False, nprocs=None, spawnonce=True):  # Spawnonce uses less memory. Need to chase memory leaks in spawn.
		multiprocessing.log_to_stderr(logging.WARN)
		self.q_in = multiprocessing.Queue(1 if not spawnonce else nprocs)
		self.q_out = multiprocessing.Queue()
		if nprocs is None:
			self.nprocs = multiprocessing.cpu_count()
		else:
			self.nprocs = multiprocessing.cpu_count() + nprocs if nprocs < 0 else nprocs
		self.proc = []
		self.progress = progress
		self.spawnonce = spawnonce
		self.f = f

		self.reset(f)

	def reset(self, f):
		self.results = []
		while len(self.proc) > 0:
			self.proc.pop().terminate()
		if not self.spawnonce:
			self.proc = [multiprocessing.Process(target=spawn(f), args=(self.q_in, self.q_out)) for _ in range(self.nprocs)]
			for p in self.proc:
				p.daemon = True
				p.start()

	def sweep_results(self):
		while True:
			try:
				self.results.append(self.q_out.get(timeout=0.01))
			except Queue.Empty:
				break
			except:
				break

	def __print_progress(self, count):
		sys.stderr.write("\r %3d%% | %d of %d | Memory usage: %.2f MB" % (float(len(self.results)) / count * 100, len(self.results), count, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.))
		sys.stderr.flush()


	def map(self, X):
		count = len(X)
		if not self.spawnonce:
			X = X + [(None, None, None)] * self.nprocs  # add sentinels

		'''if self.DEBUG:
			import tracemalloc
			tracemalloc.start()
			# from guppy import hpy
			# hp = hpy()
			# before = hp.heap()
			before = tracemalloc.take_snapshot().filter_traces([tracemalloc.Filter(False, tracemalloc.__file__)])
			last = None'''

		for i, x in enumerate(X):
			
			if self.spawnonce and len(self.results) + self.nprocs <= i:  # Wait for processes to finish before starting new ones
				self.results.append(self.q_out.get())
			self.q_in.put(x)
			if self.spawnonce:
				self.proc.append(multiprocessing.Process(target=spawnonce(self.f), args=(self.q_in, self.q_out)))
				self.proc[-1].daemon = False
				self.proc[-1].start()
			while len(self.proc) > 2*self.nprocs:
				p = self.proc.pop(0)
				if not p.is_alive():
					p.terminate()
					del p
			
			gc.collect()
			'''if self.DEBUG:

				after = tracemalloc.take_snapshot().filter_traces([tracemalloc.Filter(False, tracemalloc.__file__)])
				top_stats = after.compare_to(before, 'lineno')

				print("[ Top 10 Leaks ]")
				for stat in top_stats[:10]:
					print(stat)

				
				if last is not None:
					top_stats = after.compare_to(last, 'lineno')

					print("[ Top 10 Leaks since last check ]")
					for stat in top_stats[:10]:
						print(stat)

				last = after'''

			self.sweep_results()
			if self.progress:
				self.__print_progress(count)
			
			gc.collect()

		self.q_in.close()
		
		while len(self.results) < len(X) - (self.nprocs if not self.spawnonce else 0):
			self.results.append(self.q_out.get())
			if self.progress:
				self.__print_progress(count)
		if self.progress:
			print
		
		if not self.spawnonce:
			[p.terminate() if p.is_alive() else False for p in self.proc]
		
		return [(i, x) for i, x in sorted(self.results)]

