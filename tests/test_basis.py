import unittest
import sys
sys.path.insert(0,'..')

import numpy as np

from parampy import Parameters
from qubricks import Operator
from qubricks.wall import SpinBasis, SimpleBasis

class TestBasis(unittest.TestCase):

	def setUp(self):
		self.b = SpinBasis(dim=2**3)

	def test_properties(self):
		self.assertEqual(self.b.dim, 8)
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

class TestOperatorTransform(unittest.TestCase):

	def setUp(self):
		self.p = Parameters()
		self.basis_z = SimpleBasis(parameters=self.p,operator=[[1,0],[0,1]])
		self.basis_x = SimpleBasis(parameters=self.p,operator=np.sqrt(2)*np.array([[1,1],[1,-1]]))

	def test_auto_basis_transformation(self):
		op1 = Operator([[1,0],[0,1]], basis=self.basis_z, parameters=self.p)
		op2 = Operator([[0,1],[1,0]], basis=self.basis_x, parameters=self.p)

		self.assertTrue( np.all(np.array([[2,0],[0,0]]) == (op1+op2)()) )
