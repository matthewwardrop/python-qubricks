import numpy as np
import math
import re

from ..basis import Basis


class StandardBasis(Basis):
    '''
    `StandardBasis` is a simple subclass of `Basis` that describes the 
    standard basis; that is, presents a basis that looks like the identity
    operator. An instance can be created using:
    
    >>> StandardBasis(parameters=<Parameters instance>, dim=<dimension of basis>)
    '''

    def init(self):
        if self.dim is None:
            raise ValueError("Dimension must be specified.")
        self.__operator = self.Operator(np.eye(self.dim))

    @property
    def operator(self):
        return self.__operator


class SimpleBasis(Basis):
    '''
    `SimpleBasis` is a subclass of `Basis` that allows a Basis object to 
    be created on the fly from an `Operator`, a numpy array or a list instance.
    For example:
    
    >>> SimpleBasis(parameters=<Parameters intance>, operator=<Operator, numpy.ndarray or list instance>)
    '''

    def init(self, operator=None):
        self.__operator = self.Operator(operator)
        shape = self.__operator.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Basis operators must be square. Check your input.")

    @property
    def operator(self):
        return self.__operator


class SpinBasis(StandardBasis):
    '''
    `SpinBasis` is a subclass of `StandardBasis` that associates each element 
    of the standard basis with a spin configuration. It assumes that there are
    `n` spin-1/2 particles in the system, and thus requires the dimension to be
    :math:`2^n`. It also implements conversion to and from string representation 
    of the states.
    '''

    def init(self):
        if self.dim is None:
            raise ValueError("Dimension must be specified.")
        power = math.log(self.dim,2)
        if power != round(power):
            raise ValueError("Dimension must be a power of 2.")
        super(SpinBasis, self).init()
    
    @property
    def operator(self):
        return super(SpinBasis, self).operator

    def state_info(self, state, params={}):
        '''
        Return a dictionary with the total z-spin projection of the state.

        e.g. |uud> -> {'spin': 0.5}
        '''
        totalSpin = 0
        index = state.index(1)
        for _ in xrange(int(math.log(self.dim, 2))):
            mod = (index % 2 - 1.0 / 2.0)
            totalSpin -= mod
            index = (index - index % 2) / 2
        return {'spin': totalSpin}

    def state_fromString(self, state, params={}):
        '''
        Convert strings representing sums of basis states of form:
        "<complex coefficient>|<state label>>"
        into a numerical vector.

        e.g. "|uuu>" -> [1,0,0,0,0,0,0,0]
             "0.5|uuu>+0.5|ddd>" -> [0.5,0,0,0,0,0,0,0.5]
             etc.
        '''
        matches = re.findall("(([\+\-]?(?:[0-9\.]+)?)\|([\,ud]+)\>)", state.replace(",", ""))

        ostate = [0.] * self.dim

        for m in matches:
            coeff = 1.
            if m[1] != "":
                if m[1] == "+":
                    coeff = 1.
                elif m[1] == "-":
                    coeff = -1.
                else:
                    coeff = float(m[1])

            index = 0
            for i in xrange(len(m[2])):
                j = len(m[2]) - i - 1

                if m[2][i] == 'd':
                    index += 2 ** j
            ostate[index] = coeff
        return np.array(ostate)

    def state_toString(self, state, params={}):
        """
        Applies the inverse map of SpinBasis.state_fromString .
        """
        s = ''
        for i, v in enumerate(state):
            if v != 0:
                if s != "" and v >= 0:
                    s += "+"
                if v == 1:
                    s += "|%s>" % self.__state_str(i)
                else:
                    if np.imag(v) != 0:
                        s += "(%.3f + %.3fi)|%s>" % (np.real(v), np.imag(v), self.__state_str(i))
                    else:
                        s += "%.3f|%s>" % (np.real(v), self.__state_str(i))
        return s

    def state_latex(self, state, params={}):
        """
        Returns the latex representation of each of the basis states. Note that this
        method does not deal with arbitrary states, as does SpinBasis.state_toString
        and SpinBasis.state_fromString .
        """
        spinString = ''
        index = state.index(1)
        for _ in xrange(self.dim / 2):
            mod = index % 2
            if mod == 0:
                spinString = '\\uparrow' + spinString
            else:
                spinString = '\\downarrow' + spinString
            index = (index - index % 2) / 2
        return '\\left|%s\\right>' % (spinString)

    def __state_str(self, index):
        """
        A utility function to convert a basis function index to a string.

        e.g. [1,0,0,0,0,0,0,0] -> "uuu"
        """
        s = ""
        for _ in xrange(int(math.log(self.dim, 2))):
            mod = index % 2
            if mod == 0:
                s = 'u' + s
            else:
                s = 'd' + s
            index = (index - index % 2) / 2
        return s
