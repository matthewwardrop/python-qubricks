import numpy as np
import warnings

def energy_spectrum(system, states, ranges, input=None, output=None, threshold=False, hamiltonian=None, components=[], params={}, hamiltonian_init=None, components_init=None, params_init=None, complete=False, derivative_decimals=8):
    '''
    This function returns a list of sequence which are the energy eigenvalues of the 
    states which map adiabatically to those provided in `states`. Consequently, the provided
    states should be eigenstates of the Hamiltonian (determined by `components_init` or 
    `hamiltonian_init`) when the parameters are set according to `params_init`. Where the initial
    conditions are not set, the states are assumed to be eigenstates of the Hamiltonian provided 
    for analysis (`hamiltonian` or `components`) in the corresponding parameter context `params`.
    
    :param system: A QuantumSystem instance.
    :type system: QuantumSystem
    :param states: A list of states (assumed to be eigenstates as noted above) for which we are
        interested in examining the eigen-spectrum.
    :type states: list of object
    :param ranges: A range specification for iteration (see `Parameters.range`).
    :type ranges: dict
    :param input: The basis of the specified states.
    :type input: str, Basis or None
    :param output: The basis in which to perform the calculations.
    :type output: str, Basis or None
    :param threshold: Whether to use a threshold (if boolean) or the threshold to use in basis transformations. (See `Basis.transform`.)
    :type threshold: bool or float
    :param hamiltonian: The Hamiltonian for which a spectrum is desired.
    :type hamiltonian: Operator or None
    :param components: If `hamiltonian` is `None`, the components to use from the provided
        `QuantumSystem` (see `QuantumSystem.H`).
    :type components: list of str
    :param params: The parameter context in which to perform calculations.
    :type params: dict
    :param hamiltonian_init: The Hamiltonian for which provided states are eigenstates. If not
        provided, and `components_init` is also not provided, this defaults to the value of `hamiltonian`.
    :type hamiltonian_init: Operator
    :param components_init: The components to enable such that the provided states are eigenstates.
        If not provided, this defaults to the value of `components`. (see `QuantumSystem.H`)
    :type components: list of str
    :param params_init: The parameter context to be used such that the provided states are eigenstates
        of the initial Hamiltonian. If not provided, defaults to the value of `params`.
    :type params_init: dict
    :param complete: If `True`, then the eigen-spectrum of the remaining states not specifically requested
        are appended to the returned results.
    :type complete: bool
    
    .. warning:: This method tracks eigenvalues using the rule that the next accepted eigenvalue should be
        the one with minimal absolute value of the second derivative (or equivalently, the one with the most 
        continuous gradient). If you find that this causes unexpected jumps in your plot, please try
        decreasing the granularity of your trace before reporting a bug.
    '''

    if hamiltonian_init is None:
        if hamiltonian is None:
            hamiltonian_init = system.H(*(components_init if components_init is not None else components))
        else:
            hamiltonian_init = hamiltonian
    hamiltonian_init = hamiltonian_init.change_basis(system.basis(output), params=params, threshold=threshold)

    states = system.subspace(states, input=input, output=output, params=params, threshold=threshold)

    if type(ranges) is not dict:
        raise ValueError("Multi-dimensional ranges are not supported; and so ranges must be a dictionary.")

    # IDENTIFY WHICH STATES BELONG TO WHICH LABELS BY PROJECTING THEM ONTO EIGENSTATES
    if params_init is None:
        params_init = params
    evals,evecs = np.linalg.eig(hamiltonian_init(**params_init))
    evecs = evecs[:,np.argsort(evals)]
    evals = np.sort(evals)

    indices = np.argmax(np.array(states).dot(evecs),axis=1)
    if len(set(indices)) != len(indices):
        warnings.warn("Could not form bijective map between states and eigenvalues. Consider changing the initial conditions. Labelling may not work.")

    # Now iterate over the ranges provided, allocating state labels according
    # to the state which adiabatically maps to the current state. This assumes no
    # level crossings.
    if hamiltonian is None:
        hamiltonian = system.H(*components)
    hamiltonian = hamiltonian.change_basis(system.basis(output), params=params, threshold=threshold)

    # Generate values to iterate over
    f_ranges = params.copy() # merge ranges and params to ensure consistency
    f_ranges.update(ranges)
    rvals = system.p.range(ranges.keys(),**f_ranges)

    if not complete:
        results = np.zeros((len(states),len(rvals.values()[0])))
    else:
        results = np.zeros((system.dim,len(rvals.values()[0])))

    # Iterate over the value returned by `Parameters.range`.
    for i in xrange(len(rvals.values()[0])):
        vals = params.copy()
        for val in rvals:
            vals[val] = rvals[val][i]
        evals = sorted(np.linalg.eigvals(hamiltonian(**vals)).real)
        
        # Add the reported eigenvalue to the trace with least difference in the gradient (i.e. least second derivative)
        
        if i >= 2:
            # Update the indicies according to the rule that it should preserve least second derivative
            for j in xrange(len(indices)):
                derivatives = map(lambda x: abs(x - 2*results[j][i-1] + results[j][i-2]), evals)
                if round(derivatives[np.argmin(derivatives)],derivative_decimals) < round(derivatives[indices[j]],derivative_decimals):
                    indices[j] = np.argmin(derivatives)

        results[:len(indices),i] = [evals[indices[j]] for j in xrange(len(indices))]
        if complete:
            count = 0
            for k in xrange(system.dim):
                if k not in indices:
                    results[len(indices)+count,i] = evals[k]
                    count += 1

    return results
