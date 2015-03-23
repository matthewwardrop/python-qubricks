import numpy as np
import math
import re

from ..basis import Basis


class StandardBasis(Basis):

    def init(self):
        if self.dim is None:
            raise ValueError("Dimension must be specified.")

    @property
    def operator(self):
        return self.Operator(np.eye(self.dim))


class SimpleBasis(Basis):

    def init(self, dim=None, operator=None):
        '''
        Basis.init_basis is called during the basis initialisation
        routines, allowing Basis subclasses to initialise themselves.
        '''
        self.__operator = self.Operator(operator)

    @property
    def operator(self):
        '''
        Basis.operator must return an Operator object with basis states as the
        columns. The operator should use the parameters instance provided by the
        Basis subclass.
        '''
        return self.__operator


class SpinBasis(StandardBasis):

    def init(self, dim=None):
        if self.dim is None:
            raise ValueError("Dimension must be specified.")
        if self.dim % 2 == 1:
            raise ValueError("Dimension must be even.")

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
