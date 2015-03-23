import numpy as np
import warnings

def energy_spectrum(system, states, ranges, components=[], hamiltonian=None, input=None, output=None, params={}, components_init=None, hamiltonian_init=None, params_init=None, complete=False):
    '''
    Returns the energy eigenvalues of the states which map adiabatically to
    those provided. The states provided should ideally be eigenstates when the
    parameters are set according to `params_init` (but close enough is good enough).
    Since this method uses the ordering of the eigenvalues to detect which eigenvalues
    belong to which eigenstates, this method does not work in cases when the adiabatic
    theorem is violated (i.e. when energy levels cross).
    '''

    if hamiltonian_init is None:
        if hamiltonian is None:
            hamiltonian_init = system.H(*(components_init if components_init is not None else components))
        else:
            hamiltonian_init = hamiltonian
    hamiltonian_init = hamiltonian_init.change_basis(system.basis(output), params=params)

    state_specs = states
    states = system.subspace(states, input=input, output=output, params=params)

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
    hamiltonian = hamiltonian.change_basis(system.basis(output), params=params)

    # Generate values to iterate over
    f_ranges = params.copy() # merge ranges and params to ensure consistency
    f_ranges.update(ranges)
    rvals = system.p.range(*ranges.keys(),**f_ranges)
    if type(rvals) != dict:
        rvals = {ranges.keys()[0]: rvals}

    if not complete:
        results = np.zeros((len(states),len(rvals.values()[0])))
    else:
        results = np.zeros((system.dim,len(rvals.values()[0])))

    for i in xrange(len(rvals.values()[0])):
        vals = params.copy()
        for val in rvals:
            vals[val] = rvals[val][i]
        evals = sorted(np.linalg.eigvals(hamiltonian(**vals)))

        results[:len(indices),i] = [evals[indices[j]] for j in xrange(len(indices))]
        if complete:
            count = 0
            for k in xrange(system.dim):
                if k not in indices:
                    results[len(indices)+count,i] = evals[k]
                    count += 1

    return results
