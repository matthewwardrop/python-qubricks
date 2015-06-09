### Example Measurementf Operators
from ..measurement import Measurement
import numpy as np


class AmplitudeMeasurement(Measurement):
    '''
    Amplitude is a sample Measurement subclass that measures the amplitude of being
    in certain basis states as function of time throughout some state evolution.
    The basis states of interest should be identified using the `subspace` keyword 
    to the measure function.
    '''

    def init(self):
        pass

    def result_type(self, *args, **kwargs):
        return [
                    ('time', float),
                    ('amplitude', float, (self.system.dim,))
                ]

    def result_shape(self, *args, **kwargs):
        return (len(kwargs['int_initial']), len(kwargs.get('int_times', 0)))

    def measure(self, data, subspace=None, params={}, int_kwargs={}, **kwargs):
        
        times = int_kwargs.get('times',[])

        rval = np.empty((len(data), len(times)), dtype=self.result_type())

        self.__P = None
        for i, resultset in enumerate(data):
            for j, time in enumerate(resultset['time']):
                rval[i, j] = (time, self.amplitudes(resultset['state'][j]))

        return rval

    def amplitudes(self, state):
        if len(state.shape) > 1:
            return np.abs(np.diag(state))
        return np.abs(state) ** 2


class ExpectationMeasurement(Measurement):
    '''
    ExpectationMeasurement measures the expectation values of a particular set of operators applied to
    a system state. It can be initialised using:
    
    >>> ExpectationMeasurement(<Operator 1>, <Operator 2>, ...)
    '''

    def init(self, *ops):
        self.ops = ops

    def result_type(self, *args, **kwargs):
        return [
                    ('time', float),
                    ('expectation', float, (len(self.ops),))
                ]

    def result_shape(self, *args, **kwargs):
        return (len(kwargs['int_initial']), len(kwargs.get('int_times', 0)))

    def measure(self, data, subspace=None, params={}, int_kwargs={}, **kwargs):
        
        times = int_kwargs.get('times',[])
        
        rval = np.empty((len(data), len(times)), dtype=self.result_type())

        self.__P = None
        for i, resultset in enumerate(data):
            for j, time in enumerate(resultset['time']):
                rval[i, j] = (time, self.expectations(resultset['state'][j]))

        return rval

    def expectations(self, state):
        es = []
        for op in self.ops:
            if len(state.shape) == 1:
                state = np.outer(state, state)
            es.append(np.trace(np.array(op).dot(state)))
        return es


class LeakageMeasurement(Measurement):
    '''
    Leakage measures the probability of a quantum system being outside of a specified
    subspace. The subspace of interest should be identified using the `subspace` keyword 
    to the measure function.
    '''

    def init(self):
        pass

    def result_type(self, *args, **kwargs):
        return [
                    ('time', float),
                    ('leakage', float)
                ]

    def result_shape(self, *args, **kwargs):
        return (len(kwargs['int_initial']), len(kwargs.get('int_times', 0)))

    @property
    def result_units(self):
        return None

    def measure(self, data, subspace=None, params={}, int_kwargs={}, **kwargs):
        
        times = int_kwargs.get('times',[])
        initial = int_kwargs.get('times',[])
        output = int_kwargs.get('output',None)

        rval = np.empty((len(data), len(times)), dtype=self.result_type(int_initial=initial, int_times=times))

        self.__P = None
        for i, resultset in enumerate(data):
            for j, time in enumerate(resultset['time']):
                rval[i, j] = (time, self.leakage(resultset['state'][j], subspace, output, params))

        return rval

    def leakage(self, state, subspace, output, params):
        if subspace is None:
            raise ValueError("Subspace must be non-empty")

        if self.__P is None:
            self.__P = self.system.subspace_projector(subspace, invert=True, output=output, params=params)
        P = self.__P
        if len(state.shape) > 1:
            return np.trace(np.dot(np.dot(P, state), P))
        return np.linalg.norm(np.dot(P, state)) ** 2
