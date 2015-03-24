from ..system import QuantumSystem


class SimpleQuantumSystem(QuantumSystem):
    '''
    `SimpleQuantumSystem` is a subclass of `QuantumSystem` that enables you to 
    initialise a `QuantumSystem` instance in one line, by passing keyword arguments
    to the constructor. Otherwise, it is indistinguishable.
    
    :param hamiltonian: The Hamiltonian to use for this QuantumSystem. Can be an Operator or an array.
    :type hamiltonian: Operator or numpy.array or list
    :param bases: A dictionary of bases to add to the QuantumSystem.
    :type bases: dict of Basis
    :param states: A dictionary of states to add to the QuantumSystem.
    :type states: dict of arrays
    :param measurements: A dictionary of `Measurement`s to add to the QuantumSystem.
    :type measurements: dict of Measurement
    :param derivative_ops: A dictionary of `StateOperator`s to add to the QuantumSystem.
    :type derivative_ops: dict of StateOperator
    
    For more documentation, see `QuantumSystem`.
    '''

    def init(self, hamiltonian=None, bases=None, states=None, measurements=None, derivative_ops=None):
        '''
        Configure any custom properties/attributes using kwargs passed
        to __init__.
        '''
        self.kwargs = {
                       'hamiltonian': hamiltonian,
                       'bases': bases,
                       'states': states,
                       'measurements': measurements,
                       'derivative_ops': derivative_ops,
                       }

    def init_hamiltonian(self):
        if self.kwargs.get('hamiltonian') is None:
            raise ValueError("A Hamiltonian was not specified (and is required) for this system.")
        return self.Operator(self.kwargs['hamiltonian'])

    def init_bases(self):
        for name, basis in self.kwargs.get('bases', {}).items():
            self.add_basis(name, basis)

    def init_states(self):
        for name, state in self.kwargs.get('states', {}).items():
            self.add_state(name, state)

    def init_measurements(self):
        for name, meas in self.kwargs.get('measurements', {}).items():
            self.add_measurement(name, meas)

    def init_derivative_ops(self, components=None):
        '''
        Setup the derivative operators to be implemented on top of the
        basic quantum evolution operator.
        '''
        for name, op in self.kwargs.get('derivative_ops', {}).items():
            self.add_derivative_op(name, op)
