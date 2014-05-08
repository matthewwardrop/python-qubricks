from tools import ModelAnalysis

import sympy
def getFreeSymbolCoefficients(expr):
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
    for i in xrange(1,len(args)):
        a = a.dot(args[i])
    return a