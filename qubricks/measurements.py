### Example Measurementf Operators
from .measurement import Measurement
import numpy as np

class Amplitude(Measurement):
    '''
    Amplitude is a sample Measurement subclass that measures the amplitude of being
    in certain basis states as function of time throughout some state evolution.
    '''

    def init(self):
        pass

    def result_type(self,*args,**kwargs):
        return [
                    ('time',float),
                    ('amplitude',float,(self._system.dim,) )
                ]

    def result_shape(self,*args,**kwargs):
        return (len(kwargs['psi_0s']),len(kwargs.get('times',0)))

    def measure(self,r,times,psi_0s,params={},subspace=None,**kwargs):

        rval = np.empty((len(r),len(times)),dtype=self.result_type(psi_0s=psi_0s,times=times))

        self.__P = None
        for i,resultset in enumerate(r):
            for j,time in enumerate(resultset['time']):
                rval[i,j] = (time,self.amplitudes(resultset['state'][j]))

        return rval

    def amplitudes(self,state):
        if len(state.shape) > 1:
            return np.abs(np.diag(state))
        return np.abs(state)**2

class Expectation(Measurement):
    '''
    Expectation measures the expectation value of a particular operator applied to
    a system state.
    '''

    def init(self,*ops):
        self.ops = ops

    def result_type(self,*args,**kwargs):
        return [
                    ('time',float),
                    ('expectation',float,(len(self.ops),) )
                ]

    def result_shape(self,*args,**kwargs):
        return (len(kwargs['psi_0s']),len(kwargs.get('times',0)))

    def measure(self,r,params={},subspace=None,psi_0s=None,times=None,**kwargs):

        rval = np.empty((len(r),len(times)),dtype=self.result_type(psi_0s=psi_0s,times=times))

        self.__P = None
        for i,resultset in enumerate(r):
            for j,time in enumerate(resultset['time']):
                rval[i,j] = (time,self.expectations(resultset['state'][j]))

        return rval

    def expectations(self,state):
        es = []
        for op in self.ops:
            if len(state.shape) == 1:
                state = np.outer(state,state)
            es.append(np.trace(np.array(op).dot(state)))
        return es


class Leakage(Measurement):
    '''
    Leakage measures the probability of a quantum system being outside of a specified
    subspace.
    '''

    def init(self):
        pass

    def result_type(self,*args,**kwargs):
        return [
                    ('time',float),
                    ('leakage',float)
                ]

    def result_shape(self,*args,**kwargs):
        return (len(kwargs['psi_0s']),len(kwargs.get('times',0)))

    @property
    def result_units(self):
        return None

    def measure(self,r,times,psi_0s,params={},subspace=None,input=None,output=None,**kwargs):

        rval = np.empty((len(r),len(times)),dtype=self.result_type(psi_0s=psi_0s,times=times))

        self.__P = None
        for i,resultset in enumerate(r):
            for j,time in enumerate(resultset['time']):
                rval[i,j] = (time,self.leakage(resultset['state'][j],subspace,output,params))

        return rval

    def leakage(self,state,subspace,output,params):
        if subspace is None:
            raise ValueError, "Subspace must be non-empty"

        if self.__P is None:
            self.__P = self._system.subspace_projector(subspace,invert=True,output=output,params=params)
        P = self.__P
        if len(state.shape) > 1:
            return np.trace(np.dot(np.dot(P,state),P))
        return np.linalg.norm(np.dot(P,state))**2
