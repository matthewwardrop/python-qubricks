import unittest
import sys
sys.path.insert(0,'..')

import sympy
import numpy as np

from parampy import Parameters
from qubricks import Operator

class TestOperator(unittest.TestCase):

	def setUp(self):
		self.p = Parameters()
		self.array = np.array([[1,2],[3,4]])
		z,x = sympy.var('z,x')
		self.matrix = sympy.Matrix([[z, x],[x**2, z**2]])
		self.op1 = Operator({'x': [[1,2],[3,4]]}, parameters=self.p)
		self.op2 = Operator({'y': [[4,3],[2,1]]}, parameters=self.p, exact=True)

	def test_indexing(self):
		# Should support all numpy indexing
		self.assertEqual(self.op1[0,0](x=1), 1)
		self.assertEqual(self.op2[1,1](y=1), 1)

	def test_mul(self):
		self.assertEqual((self.op1*self.op2)[0,0](x=1,y=1), 8)

		self.assertEqual((self.op1*self.array)[0,0](x=1), 7)
		self.assertEqual((self.array*self.op2)[0,0](y=1), 8)

		self.assertEqual((self.op1*self.matrix)[0,0](x=2,z=3), 22)
		# Requires sympy to acknowledge it cannot multiply an Operator object (and so fails)
		# self.assertEqual((self.matrix*self.op2)[0,0](x=1,y=2,z=3), 28)

	def test_add_sub(self):
		self.assertEqual( (self.op1 + self.op2)[0,0](x=2,y=3), 14)
		self.assertEqual( (self.op1 - self.op2)[0,0](x=2,y=3), -10)

		self.assertEqual( (self.op1 + self.array)[0,0](x=2), 3)
		self.assertEqual( (self.array + self.op2)[0,0](y=1), 5)

		self.assertEqual( (self.op1 - self.array)[0,0](x=2), 1)
		self.assertEqual( (self.array - self.op2)[0,0](y=1), -3)

		self.assertEqual( (self.op1 + self.matrix)[0,0](x=2,z=2), 4)
		# Requires sympy to acknowledge it cannot add or subtract an Operator object (and so fails)
		#self.assertEqual( (self.matrix + self.op2)[0,0](y=2,z=2), 10)

		self.assertEqual( (self.op1 - self.matrix)[0,0](x=2,z=2), 0)
		# Requires sympy to acknowledge it cannot add or subtract an Operator object (and so fails)
		#self.assertEqual( (self.matrix - self.op2)[0,0](y=2,z=2), -6)

	# A redundant test to just to make sure everything is consistent
	def test_arithmetic(self):
		op1 = Operator([[1,0],[0,1]], parameters=self.p)
		op2 = Operator([[1,1],[1,1]], parameters=self.p)

		self.assertTrue( np.all(np.array([[2,1],[1,2]]) == (op1+op2)()) )
		self.assertTrue( np.all(np.array([[0,-1],[-1,0]]) == (op1-op2)()) )
		self.assertTrue( np.all(np.array([[1,1],[1,1]]) == (op1*op2)()) )

	def test_scaling(self):
		self.assertEqual( (2j*self.op1)[0,0](x=1), 2j )
		self.assertEqual( (self.op1*2j)[0,0](x=1), 2j )

		self.assertEqual( ('x'*self.op1)[0,0](x=2), 4 )
		self.assertEqual( (self.op1*'x')[0,0](x=2), 4 )

		self.assertEqual( (self.op1/'x')[0,0](x=2), 1 )
		self.assertEqual( (self.op1/'y')[0,0](x=2,y=1j), -2j )

	# Basis transformations are tested in test_basis.py
