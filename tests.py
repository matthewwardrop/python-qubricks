import timeit
import cProfile as profile
import numpy as np

import qubricks
import math
import sympy
from qubricks import  Operator, QuantumSystem
from qubricks.wall import SimpleBasis, SimpleBasis, SpinBasis
from parameters import Parameters

print "Unit Tests"
print "-----------------"
###################### UNIT TESTS ##############################################
import unittest

class TestBasis(unittest.TestCase):

	def setUp(self):
		self.b = SpinBasis(name="TestBasis", dim=2**3)

	def test_properties(self):
		self.assertEqual(self.b.dim, 8)
		self.assertEqual(self.b.name, "TestBasis")
		self.assertIsInstance(self.b.operator, Operator)
		self.assertEqual(len(self.b.states()), 2**3)

	def test_symbolic(self):
		self.skipTest("Symbolic functions are not yet fully baked.")

	def test_transform(self):
		self.assertEqual(self.b.transform([1,0,0,0,0,0,0,0]).tolist(),[1,0,0,0,0,0,0,0])

	def test_repr(self):
		self.assertEqual(self.b.state_fromString("|uuu>").tolist(), [1,0,0,0,0,0,0,0])
		self.assertEqual(self.b.state_fromString("|ddd>").tolist(), [0,0,0,0,0,0,0,1])
		self.assertEqual(self.b.state_fromString("|uuu>+|ddd>").tolist(), [1.,0,0,0,0,0,0,1.])

		self.assertEqual(self.b.state_toString([1,0,0,1,0,0,0,0]),"|uuu>+|udd>")
		self.assertEqual(self.b.state_toString([1,0,0,0,0,0,0,0]),"|uuu>")
		self.assertEqual(self.b.state_toString([0,0,0,0,0,0,0,1]),"|ddd>")

	def test_info(self):
		self.assertEqual(self.b.state_info([1,0,0,0,0,0,0,0]),{'spin':1.5})

class TestOperator(unittest.TestCase):

	def setUp(self):
		self.p = Parameters()
		self.basis_z = SimpleBasis(parameters=self.p,name='basis_z',operator=[[1,0],[0,1]])
		self.basis_x = SimpleBasis(parameters=self.p,name='basis_x',operator=np.sqrt(2)*np.array([[1,1],[1,-1]]))

	def test_arithmetic(self):
		op1 = Operator([[1,0],[0,1]])
		op2 = Operator([[1,1],[1,1]])

		self.assertTrue( np.all(np.array([[2,1],[1,2]]) == (op1+op2)()) )
		self.assertTrue( np.all(np.array([[0,-1],[-1,0]]) == (op1-op2)()) )
		self.assertTrue( np.all(np.array([[1,1],[1,1]]) == (op1*op2)()) )

	def test_auto_basis_transformation(self):
		op1 = Operator([[1,0],[0,1]],basis=self.basis_z)
		op2 = Operator([[0,1],[1,0]],basis=self.basis_x)

		self.assertTrue( np.all(np.array([[2,0],[0,0]]) == (op1+op2)()) )

class TestTwoLevel(unittest.TestCase):

	def setUp(self):
		self.system = TwoLevel()

	def test_evolution(self):
		for time in list(range(1,20)):
			np.testing.assert_array_almost_equal(self.system.integrate([time], ['up'], callback_fallback=False)['state'][0,0], self.system.ideal_integration(time, 'up'), 5)
			np.testing.assert_array_almost_equal(self.system.integrate([time], ['up'], callback_fallback=False, params={'B':0})['state'][0,0], self.system.ideal_integration(time, 'up', params={'B':0}), 5)
			np.testing.assert_array_almost_equal(self.system.integrate([time], ['up'], callback_fallback=False, params={'J':0})['state'][0,0], self.system.ideal_integration(time, 'up', params={'J':0}), 5)

class TwoLevel(QuantumSystem):

	def setup_environment(self, **kwargs):
		pass

	def setup_parameters(self):
		self.p << {'c_hbar': 1.0}

		self.p.B = 1
		self.p.J = 1

	def setup_bases(self):
		pass

	def setup_hamiltonian(self):
		return self.Operator( {'J': np.array([[0,1],[1,0]]),'B':np.array([[1,0],[0,-1]])})

	def setup_states(self):
		'''
		Add the named/important states to be used by this quantum system.
		'''
		self.add_state("up",[1,0])
		self.add_state("down",[0,1])
		self.add_state("+",np.array([1,1])/math.sqrt(2))
		self.add_state("-",np.array([1,-1])/math.sqrt(2))

	def setup_measurements(self):
		'''
		Add the measurements to be used by this quantum system instance.
		'''
		pass

	@property
	def default_derivative_ops(self):
		return ['evolution']

	def setup_derivative_ops(self):
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

if __name__ == '__main__':
	unittest.main()
