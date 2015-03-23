from ..system import QuantumSystem


class SimpleQuantumSystem(QuantumSystem):

    def init(self, **kwargs):
        '''
        Configure any custom properties/attributes using kwargs passed
        to __init__.
        '''
        self.kwargs = kwargs

    def init_parameters(self):
        '''
        After the QuantumSystem parameter initialisation routines
        are run, check that the parameters are initialised correctly.
        '''
        pass

    def init_bases(self):
        '''
        Add the bases which are going to be used by this QuantumSystem
        instance.
        '''
        pass

    def init_hamiltonian(self):
        '''
        Initialise the Hamiltonian to be used by this QuantumSystem
        instance.
        '''
        if self.kwargs.get('hamiltonian') is None:
            raise ValueError("A Hamiltonian was not specified (and is required) for this system.")
        return self.Operator(self.kwargs['hamiltonian'])

    def init_states(self):
        '''
        Add the named/important states to be used by this quantum system.
        '''
        pass

    def init_measurements(self):
        '''
        Add the measurements to be used by this quantum system instance.
        '''
        for name, meas in self.kwargs.get('measurements', {}).items():
            self.add_measurement(name, meas)

    @property
    def default_derivative_ops(self):
        return ['evolution'] + self.get_derivative_ops().keys()

    def init_derivative_ops(self, components=None):
        '''
        Setup the derivative operators to be implemented on top of the
        basic quantum evolution operator.
        '''
        for name, op in self.kwargs.get('operators', {}).items():
            self.add_derivative_op(name, op)
