import numpy as np
import sympy
import warnings
DEBUG = False


def debug(*messages):
	if DEBUG:
		for message in messages:
			print messages,
	print


class Perturb(object):

	def __init__(self, H_0=None, V=None, type='RSPT', subspace=None):
		self.H_0 = H_0
		self.V = V
		self.type = type
		self.__pts = {}
		self.__subspace = list(subspace) if subspace is not None else None

		self.E0s, self.Psi0s = get_unperturbed_states(H_0, V, subspace=subspace)

	@property
	def dim(self):
		return self.H_0.shape[0]

	def pt(self, index):
		if index not in self.__pts:
			self.__pts[index] = RSPT(self.H_0, self.V, index=self.subspace().index(index), E0s=self.E0s, Psi0s=self.Psi0s)
		return self.__pts[index]

	def subspace(self, subspace=None):
		if subspace is not None:
			return subspace
		if self.__subspace is not None:
			return self.__subspace
		return range(self.dim)

	def energies(self, order=0, cumulative=True, subspace=None):
		Es = []
		for i in self.subspace(subspace):
			if cumulative:
				Es.append(sum([self.pt(i).E(ord) for ord in range(order + 1)]))
			else:
				Es.append(self.pt(i).E(order))
		return np.array(Es, dtype=object)

	def wavefunctions(self, order=0, cumulative=True, subspace=None):
		psis = []
		for i in self.subspace(subspace):
			if cumulative:
				psis.append(sum([self.pt(i).Psi(ord) for ord in range(order + 1)]))
			else:
				psis.append(self.pt(i).Psi(order))
		return np.array(psis, dtype=object)

	def Heff_adiabatic(self, order=0, cumulative=True, subspace=None):
		return np.diag(self.energies(order=order, cumulative=cumulative, subspace=self.subspace(subspace)))

	def Heff(self, order=0, cumulative=True, subspace=None):
		raise NotImplementedError("Effective Hamiltonian is not yet implemented.")

class RSPT(object):
	# Rayleigh-Schroedinger Perturbation Theory

	def __init__(self, H_0=None, V=None, index=0, E0s=None, Psi0s=None):
		self.__Es = {}
		self.__Psis = {}

		self.H_0 = np.array(H_0)
		self.V = np.array(V)

		if E0s is None or Psi0s is None:
			E0s, Psi0s = get_unperturbed_states(H_0, V)
		self.E0s = E0s
		self.Psi0s = Psi0s

		self.index = index

	@property
	def dim(self):
		return self.H_0.shape[0]

	def P(self):  # Assumes diagonal H_0
		a = np.eye(self.dim, dtype=object)
		for i in xrange(self.dim):
			if self.E0s[i] == self.E0s[self.index]:
				a[self.index, self.index] = 0
		debug("P", a)
		return a

	def E(self, order=0):
		if order in self.__Es:
			return self.__Es[order]

		if order == 0:
			debug("E", order, self.E0s[self.index])
			return self.E0s[self.index]

		elif order % 2 == 0:
			r = self.Psi(order / 2).dot(self.V).dot(self.Psi(order / 2 - 1))
			for k in xrange(1, order / 2 + 1):
				for l in xrange(1, order / 2):
					r -= self.E(order - k - l) * self.Psi(k).dot(self.Psi(l))

		else:
			r = self.Psi((order - 1) / 2).dot(self.V).dot(self.Psi((order - 1) / 2))
			for k in xrange(1, (order - 1) / 2 + 1):
				for l in xrange(1, (order - 1) / 2 + 1):
					r -= self.E(order - k - l) * self.Psi(k).dot(self.Psi(l))

		debug("E", order, r)
		self.__Es[order] = r
		return r

	def inv(self):  # Assumes diagonal H_0
		#inv = sympy.Matrix(self.E(0)*np.eye(self.H_0.shape[0]) - self.H_0).inv()
		inv = np.zeros(self.H_0.shape, dtype=object)
		for i in xrange(self.dim):
			if self.E0s[i] != self.E0s[self.index]:
				inv[i, i] = 1 / (self.E(0) - self.H_0[i, i])
		debug("inv", inv)
		return inv

	def Psi(self, order=0):
		if order in self.__Psis:
			return self.__Psis[order]

		if order == 0:
			debug("wf", order, self.Psi0s[self.index])
			return self.Psi0s[self.index]

		b = np.dot(self.V, self.Psi(order - 1))
		for k in xrange(1, order + 1):
			b -= self.E(k) * self.Psi(order - k)

		self.__Psis[order] = self.inv().dot(self.P()).dot(b)
		debug("wf", order, self.__Psis[order])
		return self.__Psis[order]


def get_unperturbed_states(H_0, V, subspace=None):
	warnings.warn("Assuming H_0 is diagonal. If H_0 is not diagonal, these methods will fail.")
	E0s = []
	for i in xrange(H_0.shape[0]):
		E0s.append(H_0[i, i])

	if subspace is None:
		subspace = range(H_0.shape[0])

	done = set()
	psi0s = [None] * len(E0s)
	for i, E0 in enumerate(E0s):
		if i not in done:
			degenerate_subspace = np.where(np.array(E0s) == E0)[0]

			if len(degenerate_subspace) > 1 and not (all(e in subspace for e in degenerate_subspace) or all(e not in subspace for e in degenerate_subspace)):
				warnings.warn("Chosen subspace %s overlaps with degenerate subspace of H_0 %s. Extending the subspace to include these states." % (subspace, degenerate_subspace))
				subspace = set(subspace).union(degenerate_subspace)

			if len(degenerate_subspace) == 1 or i not in subspace:
				v = np.zeros(H_0.shape[0], dtype='object')
				v[i] = sympy.S('1')
				psi0s[i] = v
				done.add(i)
			else:
				m = sympy.Matrix(V)[tuple(degenerate_subspace), tuple(degenerate_subspace)]
				l = 0
				for (_energy, multiplicity, vectors) in m.eigenvects():
					for k in xrange(multiplicity):
						v = np.zeros(H_0.shape[0], dtype=object)
						v[np.array(degenerate_subspace)] = np.array(vectors[k].transpose().normalized()).flatten()
						psi0s[degenerate_subspace[l]] = v
						done.add(degenerate_subspace[l])
						l += 1

	return E0s, psi0s
