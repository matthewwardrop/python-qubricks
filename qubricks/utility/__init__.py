import numpy as np
from text import colour_text
import sympy


def getLinearlyIndependentCoeffs(expr):
	def getCoefficient(e):
		return e.as_independent(*e.free_symbols, as_Add=False)
	if type(expr) == sympy.Add:
		result = []
		for term in expr.as_terms()[0]:
			result.append(getCoefficient(term[0]))
		return result
	else:
		return [getCoefficient(expr)]


def dot(*args):
	a = args[0]
	for i in xrange(1, len(args)):
		a = a.dot(args[i])
	return a


def tensor(*args):
	a = args[0]
	for i in xrange(1, len(args)):
		a = np.kron(a, args[i])
	return a


def struct_allclose(a, b, rtol=1e-05, atol=1e-08):
	if set(a.dtype.names) != set(a.dtype.names):
		return False
	for name in a.dtype.names:
		if not np.allclose(a[name], b[name], rtol=rtol, atol=atol):
			return False
	return True
