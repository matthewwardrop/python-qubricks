from ..system import QuantumSystem

class SimpleQuantumSystem(QuantumSystem):

    def setup_environment(self, **kwargs):
        '''
        Configure any custom properties/attributes using kwargs passed
        to __init__.
        '''
        self.kwargs = kwargs

    def setup_parameters(self):
        '''
        After the QuantumSystem parameter initialisation routines
        are run, check that the parameters are initialised correctly.
        '''
        pass

    def setup_bases(self):
        '''
        Add the bases which are going to be used by this QuantumSystem
        instance.
        '''
        pass

    def setup_hamiltonian(self):
        '''
        Initialise the Hamiltonian to be used by this QuantumSystem
        instance.
        '''
        return self.Operator(self.kwargs['hamiltonian'])

    def setup_states(self):
        '''
        Add the named/important states to be used by this quantum system.
        '''
        pass

    def setup_measurements(self):
        '''
        Add the measurements to be used by this quantum system instance.
        '''
        for name,meas in self.kwargs.get('measurements',{}).items():
            self.add_measurement(name, meas)

    @property
    def default_derivative_ops(self):
        return ['evolution']+self.get_derivative_ops().keys()

    def get_derivative_ops(self, components=None):
        '''
        Setup the derivative operators to be implemented on top of the
        basic quantum evolution operator.
        '''
        return self.kwargs.get('operators',{})
