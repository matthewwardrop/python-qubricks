import numpy as np
import sympy
import itertools
import math
import mpmath
import warnings
from qubricks.operator import Operator
DEBUG = False


def debug(*messages):
	if DEBUG:
		for message in messages:
			print messages,
		print


class Perturb(object):
	'''
	`Perturb` is a class that allows one to perform degenerate perturbation theory.
	The perturbation theory logic is intentionally separated into a different class for clarity.
	Currently it only supports using `RSPT` for perturbation theory, though in the future
	this may be extended to `Kato` perturbation theory. The advantage of using this class
	as compared to directly using the `RSPT` class is that the energies and eigenstates
	can be computed cumulatively, as well as gaining access to shorthand constructions
	of effective Hamiltonians.

	:param H_0: The unperturbed Hamiltonian to consider.
	:type H_0: Operator, sympy matrix or numpy array
	:param V: The Hamiltonian perturbation to consider.
	:type V: Operator, sympy matrix or numpy array
	:param subspace: The state indices to which attention should be restricted.
	:type subspace: list of int
	'''

	def __init__(self, H_0=None, V=None, subspace=None):
		self.H_0 = H_0
		self.V = V
		self.__subspace_default = list(subspace) if subspace is not None else None
		self.__rspt = RSPT(self.H_0, self.V, self.__subspace())

	@property
	def dim(self):
		'''
		The dimension of :math:`H_0`.
		'''
		return self.H_0.shape[0]

	@property
	def pt(self):
		'''
		A reference to the perturbation calculating object (e.g. RSPT).
		'''
		return self.__rspt

	def __subspace(self, subspace=None):
		if subspace is not None:
			return subspace
		if self.__subspace_default is not None:
			return self.__subspace_default
		return range(self.dim)

	def E(self, index, order=0, cumulative=True):
		'''
		This method returns the `index` th eigenvalue correct to order `order` if
		`cumulative` is `True`; or the the `order` th correction otherwise.

		:param index: The index of the state to be considered.
		:type index: int
		:param order: The order of perturbation theory to apply.
		:type order: int
		:param cumulative: `True` if all order corrections up to `order` should be summed
			(including the initial unperturbed energy).
		:type cumulative: bool
		'''
		if cumulative:
			return sum([self.pt.E(index,ord) for ord in range(order + 1)])
		else:
			return self.pt.E(index,order)

	def Psi(self, index, order=0, cumulative=True):
		'''
		This method returns the `index` th eigenstate correct to order `order` if
		`cumulative` is `True`; or the the `order` th correction otherwise.

		:param index: The index of the state to be considered.
		:type index: int
		:param order: The order of perturbation theory to apply.
		:type order: int
		:param cumulative: `True` if all order corrections up to `order` should be summed
			(including the initial unperturbed state).
		:type cumulative: bool
		'''
		if cumulative:
			return sum([self.pt.Psi(index,ord) for ord in range(order + 1)])
		else:
			return self.pt.Psi(index,order)

	def Es(self, order=0, cumulative=True, subspace=None):
		'''
		This method returns a the energies associated with the indices
		in `subspaces`. Internally this uses `Perturb.E`, passing through
		the keyword arguments `order` and `cumulative` for each index in
		subspace.

		:param order: The order of perturbation theory to apply.
		:type order: int
		:param cumulative: `True` if all order corrections up to `order` should be summed
			(including the initial unperturbed energy).
		:type cumulative: bool
		:param subspace: The set of indices for which to return the associated energies.
		:type subspace: list of int
		'''
		Es = []
		for i in self.__subspace(subspace):
			if cumulative:
				Es.append(sum([self.pt.E(i,ord) for ord in range(order + 1)]))
			else:
				Es.append(self.pt.E(i,order))
		return np.array(Es, dtype=object)

	def Psis(self, order=0, cumulative=True, subspace=None):
		'''
		This method returns a the eigenstates associated with the indices
		in `subspaces`. Internally this uses `Perturb.Psi`, passing through
		the keyword arguments `order` and `cumulative` for each index in
		subspace.

		:param order: The order of perturbation theory to apply.
		:type order: int
		:param cumulative: `True` if all order corrections up to `order` should be summed
			(including the initial unperturbed state).
		:type cumulative: bool
		:param subspace: The set of indices for which to return the associated energies.
		:type subspace: list of int
		'''
		psis = []
		for i in self.__subspace(subspace):
			if cumulative:
				psis.append(sum([self.pt.Psi(i,ord) for ord in range(order + 1)]))
			else:
				psis.append(self.pt.Psi(i,order))
		return np.array(psis, dtype=object)

	def H_eff(self, order=0, cumulative=True, subspace=None, adiabatic=False):
		'''
		This method returns the effective Hamiltonian on the subspace indicated,
		using energies and eigenstates computed using `Perturb.E` and `Perturb.Psi`.
		If `adiabatic` is `True`, the effective Hamiltonian describing the energies of
		the instantaneous eigenstates is returned in the basis of the instantaneous
		eigenstates (i.e. the Hamiltonian is diagonal with energies corresponding to
		the instantaneous energies). Otherwise, the Hamiltonian returned is the sum over
		the indices of the subspace of the perturbed energies multiplied by the outer
		product of the corresponding perturbed eigenstates.

		:param order: The order of perturbation theory to apply.
		:type order: int
		:param cumulative: `True` if all order corrections up to `order` should be summed
			(including the initial unperturbed energies and states).
		:type cumulative: bool
		:param subspace: The set of indices for which to return the associated energies.
		:type subspace: list of int
		:param adiabatic: `True` if the adiabatic effective Hamiltonian (as described above)
			should be returned. `False` otherwise.
		:type adiabatic: bool
		'''
		subspace = self.__subspace(subspace)

		H_eff = np.zeros( (len(subspace),len(subspace)) , dtype=object)
		for index in subspace:
			E = self.E(index, order, cumulative)
			if adiabatic:
				H_eff[index,index] = E
			else:
				psi = self.Psi(index, order, cumulative)
				H_eff += E*np.outer(psi,psi)
		return H_eff


class RSPT(object):
	'''
	This class implements (degenerate) Rayleigh-Schroedinger Perturbation Theory.
	It is geared toward generating symbolic solutions, in the hope that the perturbation
	theory might provide insight into the quantum system at hand. For numerical solutions,
	you are better off simply diagonalising the evaluated Hamiltonian.

	.. warning:: This method currently only supports diagonal :math:`H_0`.

	:param H_0: The unperturbed Hamiltonian to consider.
	:type H_0: Operator, sympy matrix or numpy array
	:param V: The Hamiltonian perturbation to consider.
	:type V: Operator, sympy matrix or numpy array
	:param subspace: The state indices to which attention should be restricted.
	:type subspace: list of int
	'''

	def __init__(self, H_0=None, V=None, subspace=None):
		self.__cache = {
					'Es': {},
					'Psis': {},
					'inv': {}
					}

		self.H_0 = H_0
		self.V = V

		self.subspace = subspace

		self.E0s, self.Psi0s = self.get_unperturbed_states()

	@property
	def H_0(self):
		return self.__H_0
	@H_0.setter
	def H_0(self, H_0):
		if isinstance(H_0, Operator):
			self.__H_0 = np.array(H_0.symbolic())
		else:
			self.__H_0 = np.array(H_0)

	@property
	def V(self):
		return self.__V
	@V.setter
	def V(self, V):
		if isinstance(V, Operator):
			self.__V = np.array(V.symbolic())
		else:
			self.__V = np.array(V)

	def __store(self, store, index, order, value=None):
		storage = self.__cache[store]
		if value is None:
			if index in storage:
				return storage[index].get(order,None)
			return None

		if index not in storage:
			storage[index] = {}
		storage[index][order] = value

	def __Es(self, index, order, value=None):
		return self.__store('Es', index, order, value)

	def __Psis(self, index, order, value=None):
		return self.__store('Psis', index, order, value)

	def get_unperturbed_states(self):
		'''
		This method returns the unperturbed eigenvalues and eigenstates as
		a tuple of energies and state-vectors.

		.. note:: This is the only method that does not support a non-diagonal
			:math:`H_0`. While possible to implement, it is not currently clear
			that a non-diagonal :math:`H_0` is actually terribly useful.
		'''

		# Check if H_0 is diagonal
		if not (self.H_0 - np.diag(self.H_0.diagonal()) == 0).all():
			raise ValueError("Provided H_0 is not diagonal")

		E0s = []
		for i in xrange(self.H_0.shape[0]):
			E0s.append(self.H_0[i, i])

		subspace = self.subspace
		if subspace is None:
			subspace = range(self.H_0.shape[0])

		done = set()
		psi0s = [None] * len(E0s)
		for i, E0 in enumerate(E0s):
			if i not in done:
				degenerate_subspace = np.where(np.array(E0s) == E0)[0]

				if len(degenerate_subspace) > 1 and not (all(e in subspace for e in degenerate_subspace) or all(e not in subspace for e in degenerate_subspace)):
					warnings.warn("Chosen subspace %s overlaps with degenerate subspace of H_0 %s. Extending the subspace to include these states." % (subspace, degenerate_subspace))
					subspace = set(subspace).union(degenerate_subspace)

				if len(degenerate_subspace) == 1 or i not in subspace:
					v = np.zeros(self.H_0.shape[0], dtype='object')
					v[i] = sympy.S('1')
					psi0s[i] = v
					done.add(i)
				else:
					m = sympy.Matrix(self.V)[tuple(degenerate_subspace), tuple(degenerate_subspace)]
					l = 0
					for (_energy, multiplicity, vectors) in m.eigenvects():
						for k in xrange(multiplicity):
							v = np.zeros(self.H_0.shape[0], dtype=object)
							v[np.array(degenerate_subspace)] = np.array(vectors[k].transpose().normalized()).flatten()
							psi0s[degenerate_subspace[l]] = v
							done.add(degenerate_subspace[l])
							l += 1

		return E0s, psi0s

	@property
	def dim(self):
		'''
		The dimension of :math:`H_0`.
		'''
		return self.H_0.shape[0]

	def E(self, index, order=0):
		r'''
		This method returns the `order` th correction to the eigenvalue associated
		with the `index` th state using RSPT.

		The algorithm:
			If `order` is 0, return the unperturbed energy.

			If `order` is even:

			.. math::

				E_n = \left< \Psi_{n/2} \right|  V \left| \Psi_{n/2-1} \right> - \sum_{k=1}^{n/2} \sum_{l=1}^{n/2-1} E_{n-k-l} \left< \Psi_k \big | \Psi_l \right>

			If `order` is odd:

			.. math::

				E_n = \left< \Psi_{(n-1)/2} \right| V \left| \Psi_{(n-1)/2} \right> - \sum_{k=1}^{(n-1)/2} \sum_{l=1}^{(n-1)/2} E_{n-k-l} \left< \Psi_k \big| \Psi_l \right>

			Where subscripts indicate that the subscripted symbol is correct to
			the indicated order in RSPT, and where `n` = `order`.

		:param index: The index of the state to be considered.
		:type index: int
		:param order: The order of perturbation theory to apply.
		:type order: int
		'''
		if self.__Es(index, order) is not None:
			return self.__Es(index, order)

		if order == 0:
			debug("E", order, self.E0s[index])
			return self.E0s[index]

		elif order % 2 == 0:
			r = self.Psi(index, order / 2).dot(self.V).dot(self.Psi(index, order / 2 - 1))
			for k in xrange(1, order / 2 + 1):
				for l in xrange(1, order / 2):
					r -= self.E(index, order - k - l) * self.Psi(index, k).dot(self.Psi(index, l))

		else:
			r = self.Psi(index, (order - 1) / 2).dot(self.V).dot(self.Psi(index, (order - 1) / 2))
			for k in xrange(1, (order - 1) / 2 + 1):
				for l in xrange(1, (order - 1) / 2 + 1):
					r -= self.E(index, order - k - l) * self.Psi(index, k).dot(self.Psi(index, l))

		debug("E", order, r)
		self.__Es(index, order, r)
		return r

	def inv(self, index):
		r'''
		This method returns: :math:`(E_0 - H_0)^{-1} P`, for use in `Psi`,
		which is computed using:

		.. math::
			A_{ij} = \delta_{ij} \delta_{i0} (E^n_0 - E^i_0)^{-1}

		Where `n` = `order`.

		.. note:: In cases where a singularity would result, `0` is used instead.
			This works because the projector off the subspace `P`
			reduces support on the singularities to zero.

		:param index: The index of the state to be considered.
		:type index: int
		'''
		if index in self.__cache['inv']:
			return self.__cache['inv'][index]

		inv = np.zeros(self.H_0.shape, dtype=object)
		for i in xrange(self.dim):
			if self.E0s[i] != self.E0s[index]:
				inv[i, i] = 1 / (self.E(index, 0) - self.E0s[i])
		debug("inv", inv)

		self.__cache['inv'][index] = inv

		return inv

	def Psi(self, index, order=0):
		r'''
		This method returns the `order` th correction to the `index` th eigenstate using RSPT.

		The algorithm:
			If `order` is 0, return the unperturbed eigenstate.

			Otherwise, return:

			.. math::
				\left| \Psi_n \right> = (E_0-H_0)^{-1} P \left( V \left|\Psi_{n-1}\right> - \sum_{k=1}^n E_k \left|\Psi_{n-k}\right> \right)

			Where `P` is the projector off the degenerate subspace enveloping
			the indexed state.

		:param index: The index of the state to be considered.
		:type index: int
		:param order: The order of perturbation theory to apply.
		:type order: int
		'''
		if self.__Psis(index, order) is not None:
			return self.__Psis(index, order)

		if order == 0:
			debug("wf", order, self.Psi0s[index])
			return self.Psi0s[index]

		b = np.dot(self.V, self.Psi(index, order - 1))
		for k in xrange(1, order + 1):
			b -= self.E(index, k) * self.Psi(index, order - k)

		psi = self.inv(index).dot(b)
		self.__Psis(index, order, psi)
		debug("wf", order, psi)
		return psi

class SWPT(object):
	'''
	This class implements (degenerate) Schrieffer-Wolff Perturbation Theory.
	It is geared toward generating symbolic solutions, in the hope that the perturbation
	theory might provide insight into the quantum system at hand. For numerical solutions,
	you are better off simply diagonalising the evaluated Hamiltonian.
	For more details, review:
	 - Bravyi, S., DiVincenzo, D. P., & Loss, D. (2011). Schrieffer-Wolff transformation for
	   quantum many-body systems. Annals of Physics, 326(10), 2793-2826.

	:param H_0: The unperturbed Hamiltonian to consider.
	:type H_0: Operator, sympy matrix or numpy array
	:param V: The Hamiltonian perturbation to consider.
	:type V: Operator, sympy matrix or numpy array
	:param subspace: The state indices to which attention should be restricted.
	:type subspace: list of int
	'''

	def __init__(self, H_0=None, V=None, subspace=None):
		self.__cache = {
					'S': {},
					'S_k': {},
					}

		self.H_0 = H_0
		self.V = V

		self.P_0 = np.zeros(self.H_0.shape)
		self.Q_0 = np.zeros(self.H_0.shape)
		for i in xrange(self.H_0.shape[0]):
			if i in subspace:
				self.P_0[i,i] = 1
			else:
				self.Q_0[i,i] = 1

		self.V_od = self.O(self.V)
		self.V_d = self.D(self.V)

		if subspace is None:
			raise ValueError("Must define low energy subspace.")
		self.subspace = subspace

		self.E0s, self.Psi0s = self.get_unperturbed_states()

	def get_unperturbed_states(self):
		'''
		This method returns the unperturbed eigenvalues and eigenstates as
		a tuple of energies and state-vectors.

		.. note:: This is the only method that does not support a non-diagonal
			:math:`H_0`. While possible to implement, it is not currently clear
			that a non-diagonal :math:`H_0` is actually terribly useful.
		'''

		# Check if H_0 is diagonal
		if not (self.H_0 - np.diag(self.H_0.diagonal()) == 0).all():
			raise ValueError("Provided H_0 is not diagonal")

		E0s = []
		for i in xrange(self.H_0.shape[0]):
			E0s.append(self.H_0[i, i])

		subspace = self.subspace
		if subspace is None:
			subspace = range(self.H_0.shape[0])

		done = set()
		psi0s = [None] * len(E0s)
		for i, E0 in enumerate(E0s):
			if i not in done:
				degenerate_subspace = np.where(np.array(E0s) == E0)[0]

				if len(degenerate_subspace) > 1 and not (all(e in subspace for e in degenerate_subspace) or all(e not in subspace for e in degenerate_subspace)):
					warnings.warn("Chosen subspace %s overlaps with degenerate subspace of H_0 %s. Extending the subspace to include these states." % (subspace, degenerate_subspace))
					subspace = set(subspace).union(degenerate_subspace)

				if len(degenerate_subspace) == 1 or i not in subspace:
					v = np.zeros(self.H_0.shape[0], dtype='object')
					v[i] = sympy.S('1')
					psi0s[i] = v
					done.add(i)
				else:
					m = sympy.Matrix(self.V)[tuple(degenerate_subspace), tuple(degenerate_subspace)]
					l = 0
					for (_energy, multiplicity, vectors) in m.eigenvects():
						for k in xrange(multiplicity):
							v = np.zeros(self.H_0.shape[0], dtype=object)
							v[np.array(degenerate_subspace)] = np.array(vectors[k].transpose().normalized()).flatten()
							psi0s[degenerate_subspace[l]] = v
							done.add(degenerate_subspace[l])
							l += 1

		return E0s, psi0s

	# Utility superoperators

	def O(self, op):
		return self.P_0.dot(op).dot(self.Q_0) + self.Q_0.dot(op).dot(self.P_0)

	def D(self, op):
		return self.P_0.dot(op).dot(self.P_0) + self.Q_0.dot(op).dot(self.Q_0)

	def L(self, op):
		denom = np.array(self.E0s).reshape((self.dim,1)) - np.array(self.E0s).reshape((1,self.dim))
		denom[denom == 0] = 1. #TODO: DO THIS MORE SAFELY
		return self.O(op)/denom

	def hat(self, operator, operand):
		return operator.dot(operand) - operand.dot(operator)

	def S(self, n):
		if n in self.__cache['S']:
			return self.__cache['S'][n]
		self.__cache['S'][n] = self._S(n)
		return self.__cache['S'][n]

	def _S(self, n):
		if n < 1:
			raise ValueError("i must be greater than or equal to zero.")
		elif n == 1:
			return self.L(self.V_od)
		elif n == 2:
			return -self.L(self.hat(self.V_d, self.S(1)))
		else:
			r = -self.L(self.hat(self.V_d, self.S(n-1)))
			# k<=m => j<=(n-1)/2
			for j in xrange(1, int(math.ceil( (n-1)/2 )) + 1 ):
				a = 2**(2*j) * mpmath.bernoulli(2*j) / mpmath.factorial(2*j)
				r += a * self.L(self.S_k(2*j, n-1))
			return r

	def _partition(self, number, count=None):
		if count <= 0:
			return set()
		answer = set()
		if count == 1:
			answer.add((number, ))
		for x in range(1, number):
			ys = self._partition(number - x, count-1 if count is not None else None)
			if len(ys) == 0:
				continue
 			for y in ys:
				answer.add(tuple(sorted((x, ) + y)))
		return answer

	def S_k(self, k, m):
		if (k,m) in self.__cache['S']:
			return self.__cache['S_k'][(k,m)]
		self.__cache['S_k'][(k,m)] = self._S_k(k,m)
		return self.__cache['S_k'][(k,m)]

	def _S_k(self, k, m):
		indices = self._partition(m,k)
		r = np.zeros(self.H_0.shape, dtype=object)
		for indexes in indices:
			for perm in set(itertools.permutations(indexes)):
				rt = self.V_od
				for i in perm: # Can ignore ordering because all permutations are considered
					rt = self.hat(self.S(i), rt)
				r += rt
		return r

	def H_eff(self, order=0, restrict=True):
		H = self.H_0.dot(self.P_0)
		if order >= 1:
			H += self.P_0.dot(self.V).dot(self.P_0)
		for n in xrange(2,order+1):
			H += self.H_eff_n(n)
		H = np.vectorize(sympy.nsimplify)(H)
		if restrict:
			subspace = np.array(self.subspace)
			return H[subspace[:,None], subspace]
		return H

	def H_eff_n(self, n):
		# k<=m => j<=(n)/2
		r = 0
		for j in xrange(1, int(math.ceil( n/2. )) + 1 ):
			b = 2*(2**(2*j)-1)*mpmath.bernoulli(2*j)/mpmath.factorial(2*j)
			r += b*self.P_0.dot(self.S_k(2*j-1,n-1)).dot(self.P_0)
		return r

	@property
	def dim(self):
		return self.H_0.shape[0]
