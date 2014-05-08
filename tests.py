import timeit
import cProfile as profile
import numpy as np

import qubricks
import math
from qubricks import SpinBasis, Operator

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
		
if __name__ == '__main__':
	unittest.main()
