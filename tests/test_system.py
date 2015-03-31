import numpy as np

import math
import sympy

import unittest

from qubricks import QuantumSystem

class TwoLevel(QuantumSystem):

	def init(self, **kwargs):
		pass

	def init_parameters(self):
		self.p << {'c_hbar': 1.0}

		self.p.B = 1
		self.p.J = 1

	def init_bases(self):
		pass

	def init_hamiltonian(self):
		return self.Operator( {'J': np.array([[0,1],[1,0]]),'B':np.array([[1,0],[0,-1]])})

	def init_states(self):
		'''
		Add the named/important states to be used by this quantum system.
		'''
		self.add_state("up",[1,0])
		self.add_state("down",[0,1])
		self.add_state("+",np.array([1,1])/math.sqrt(2))
		self.add_state("-",np.array([1,-1])/math.sqrt(2))

	def init_measurements(self):
		'''
		Add the measurements to be used by this quantum system instance.
		'''
		pass

	@property
	def default_derivative_ops(self):
		return ['evolution']

	def init_derivative_ops(self):
		'''
		Setup the derivative operators to be implemented on top of the
		basic quantum evolution operator.
		'''
		pass

	def ideal_integration(self,time,state,params={}):
		t,J,B,c_hbar = sympy.var('t,J,B,c_hbar')

		ps = {'B':self.p.B,'J':self.p.J,'t':time,'c_hbar':self.p.c_hbar}
		ps.update(params)

		op = sympy.exp( (-sympy.I/c_hbar*t*sympy.Matrix([[B,J],[J,-B]])).evalf(subs=ps) )

		return np.array( op ).astype(complex).dot(self.state(state))

class TestTwoLevel(unittest.TestCase):

	def setUp(self):
		self.system = TwoLevel()

	def test_evolution(self):
		for time in [1,5,10,20]:
			np.testing.assert_array_almost_equal(self.system.integrate([time], ['up'])['state'][0,0], self.system.ideal_integration(time, 'up'), 5)
			np.testing.assert_array_almost_equal(self.system.integrate([time], ['up'], params={'B':0})['state'][0,0], self.system.ideal_integration(time, 'up', params={'B':0}), 5)
			np.testing.assert_array_almost_equal(self.system.integrate([time], ['up'], params={'J':0})['state'][0,0], self.system.ideal_integration(time, 'up', params={'J':0}), 5)
