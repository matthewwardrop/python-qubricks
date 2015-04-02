import numpy as np
import warnings

from ..integrator import Integrator


class RealIntegrator(Integrator):
    '''
    `RealIntegrator` is a subclass of `Integrator` that wraps around
    `scipy.integrate.ode`. Any arguments supported by `scipy.integrate.ode.set_integrator`
    are supported as extra arguments (except 'name'). By default, the 'vode' integrator is use with
    the 'bdf' method and a maximum number of steps of '1e9'.
    
    At every derivative step, the representation of the state has to be transformed
    from a vector of real and imaginary components, to a vector of complex numbers.
    This workaround defaults the inaccuracies that occur with the 'zvode' integrator, 
    as is used by the `QuantumIntegrator`.
    '''

    def _integrator(self, f, **kwargs):
        from scipy.integrate import complex_ode

        defaults = {
            'nsteps': 1e9,
            'with_jacobian': False,
            'method': 'bdf',
        }
        defaults.update(kwargs)

        r = complex_ode(f).set_integrator('vode', atol=self.error_abs, rtol=self.error_rel, **defaults)
        return r

    def _integrate(self, T, y_0, times=None):
        T.set_initial_value(y_0, times[0])
        results = []
        for i, time in enumerate(times):
            if i == 0:
                results.append((time, y_0))
            else:
                T.integrate(time)
                if not T.successful():
                    raise ValueError("Integration unsuccessful. T = %f" % time)
                results.append((T.t, T.y))

        return results

    def _state_internal2ode(self, state):
        dim = state.shape
        return state.reshape((np.prod(dim),)), dim

    def _state_ode2internal(self, state, dimensions):
        return state.reshape(dimensions)

    def _derivative(self, t, y, dim, operators):
        y = y.reshape(dim)
        dy = sum(map(lambda op: op(state=y,t=t), operators))
        dy.shape = (np.prod(dim),)
        return dy


class QuantumIntegrator(Integrator):
    '''
    `QuantumIntegrator` is a subclass of `Integrator` that wraps around
    `scipy.integrate.ode`. Any arguments supported by `scipy.integrate.ode.set_integrator`
    are supported as extra arguments (except 'name'). By default, the 'zvode' integrator is use with
    the 'bdf' method and a maximum number of steps of '1e9'.
    
    .. warning:: For stiff problems, this integrator can be unreliable. It is recommend that you
        use `RealIntegrator`.
    '''

    def _state_internal2ode(self, state):
        dim = state.shape
        return np.reshape(state, (np.prod(state.shape),)), dim

    def _state_ode2internal(self, state, dimensions):
        return np.reshape(state, dimensions)

    def _derivative(self, t, y, dim, operators):
        y = y.reshape(dim)
        dy = sum(map(lambda op: op(state=y,t=t), operators))
        dy.shape = (np.prod(dy.shape),)
        return dy

    #
    # Set up integrator
    def _integrator(self, f, **kwargs):
        warnings.warn("QuantumIntegrator can sometimes be unreliable. Please use RealIntegrator instead.")
        from scipy.integrate import ode

        defaults = {
            'nsteps': 1e9,
            'with_jacobian': False,
            'method': 'bdf',
        }
        defaults.update(kwargs)

        r = ode(f).set_integrator('zvode', atol=self.error_abs, rtol=self.error_rel, **defaults)
        return r

    #
    # Internal integration function
    # Must return at values at least t_0 and t_f to allow for continuing
    # Format for returned value is: [ [<time>,<vector>], [<time>,<vector>], ... ]
    def _integrate(self, T, y_0, times=None, **kwargs):

        T.set_initial_value(y_0, times[0])
        results = []
        for i, time in enumerate(times):
            if i == times[0]:
                results.append((time, y_0))
            else:
                T.integrate(time)
                if not T.successful():
                    raise ValueError("Integration unsuccessful. T = %f" % time)
                results.append((T.t, T.y))

        return results


try:
    import sage.all as s

    class SageIntegrator(Integrator):
        '''
        `SageIntegrator` is a subclass of `Integrator` that wraps around
        `sage.ode_solver`. This is only available when QuBricks is used 
        within Sage (http://sagemath.org/).
        '''

        def _integrator(self, f, **kwargs):
            T = s.ode_solver()
            T.function = f
            T.algorithm = 'rkf45'  # self.solver
            T.error_rel = self.error_rel
            T.error_abs = self.error_abs
            return T

        def _integrate(self, T, y_0, times=None, **kwargs):
            T.ode_solve(y_0=list(y_0), t_span=list(times))
            return T.solution

        def _state_internal2ode(self, state):
            dim = state.shape
            state = np.array(state).reshape((np.prod(dim),))
            return list(np.real(state)) + list(np.imag(state)), dim

        def _state_ode2internal(self, state, dimensions):
            state = np.array(state[:len(state) / 2]) + 1j * np.array(state[len(state) / 2:])
            return np.reshape(state, dimensions)

        def _derivative(self, t, y, dim, operators):
            y = np.array(y[:len(y) / 2]).reshape(dim) + 1j * np.array(y[len(y) / 2:]).reshape(dim)
            dy = np.zeros(dim, dtype='complex')

            for operator in operators:
                dy += operator(state=y, t=t)  # ,params=self.get_op_params())

            dy.shape = (np.prod(dy.shape),)
            dy = list(np.real(dy)) + list(np.imag(dy))
            return dy

except:
    pass