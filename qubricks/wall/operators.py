import re

import numpy as np

from ..operator import OperatorSet, Operator
from ..utility import tensor

class PauliOperatorSet(OperatorSet):
    '''
    This subclass of `OperatorSet` does not store any components
    directly, but generates tensor products of Pauli I,X,Y and Z upon request. 
    For example:
    
    >>> p = PauliOperatorSet()
    >>> p['X']()
    array([[ 0.+0.j,  1.+0.j],
           [ 1.+0.j,  0.+0.j]])
    >>> p['XX']()
    array([[ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
           [ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]])
    
    As with any `OperatorSet`, the return type is an `Operator` instance.
    
    .. note:: You can still use the aggregation and `apply` methods, such as
        
        >>> p('XX', 'YY', 'ZZ')
        
        will sum p['XX'], p['YY'] and p['ZZ'] together.
    '''
    
    def init(self):
        self.__values = {}
        self.__values['I'] = np.eye(2)
        self.__values['X'] = np.array([[0,1],[1,0]])
        self.__values['Y'] = np.array([[0,-1j],[1j,0]])
        self.__values['Z'] = np.array([[1,0],[0,-1]])

    def __getitem__(self, key):
        try:
            return self.components[key]
        except:
            if re.match("^[I,X,Y,Z]+$", key):
                return Operator(tensor(*map(lambda x: self.__values[x], key)))
            raise ValueError("Unknown component: %s" % key)