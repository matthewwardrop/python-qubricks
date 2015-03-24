from qubricks import Integrator

class CustomIntegrator(Integrator):
    '''
    Refer to the API documentation for `Integrator` for more information.
    '''
    
    def _integrator(self, f, **kwargs):
        '''
        This method should return the object(s) necessary to perform
        the integration step. `f` is the the function which will return
        the derivative at each step.
        
        :param f: A function with signature f(t,y) which returns the derivative 
            at time `t` for the state `y`. Note that the derivative that is returned
            is that of `_derivative`, but `f` also handles progress reporting.
        :type f: function
        :param kwargs: Any additional keyword arguments passed to the `Integrator`
            constructor.
        :type kwargs: dict
        '''
        pass

    def _integrate(self, integrator, initial, times=None, **kwargs):
        '''
        This method should perform the integration using `integrator`, and
        return a list of two-tuples, each containing
        a time and a corresponding state. The times should be those listed in times,
        which will have been processed into floats.
        
        :param integrator: Whichever value was returned from `_integrator`.
        :type integrator: object
        :param initial: The state at which to start integrating. Will be the type
            returned by `_state_internal2ode`.
        :type initial: object
        :param times: A sequence of times for which to return the state.
        :type times: list of float
        :param kwargs: Additional keyword arguments passed to `Integrator.start`
            and/or `Integrator.extend`.
        :type kwargs: dict
        '''
        pass
    
    def _derivative(self, t, y, dim):
        '''
        This method should return the instantaneous derivative at time `t` 
        with current state `y` with dimensions `dim` (as returned by 
        `_state_internal2ode`. The derivative should be expressed
        in a form understood by the integrator returned by `_integrator`
        as used in `_integrate`.
        
        :param t: The current time.
        :type t: float
        :param y: The current state (in whatever form is returned by the integrator).
        :type y: object
        :param dim: The original dimensions of the state (as returned by `_state_internal2ode`).
        :type dim: object
        '''
        pass
    
    def _state_internal2ode(self, state):
        '''
        This method should return a tuple of a state and its original dimensions in some form.
        The state should be in a form understandable by the integrator returned by `_integrator`,
        and the derivative returned by `_derivative`.
        
        :param state: The state represented as a numpy array. Maybe 1D or 2D.
        :type state: numpy.ndarray
        '''
        pass

    def _state_ode2internal(self, state, dimensions):
        '''
        This method should restore and return the state (currently represented in the form used by the integrator
        returned by `_integrator`) to its representation as a numpy array using the 
        `dimensions` returned by `_state_internal2ode`.
        
        :param state: The state to re-represented as a numpy array.
        :type state: object
        :param dimensions: The dimensions returned by `_state_internal2ode`.
        :type dimensions: object
        '''
        pass