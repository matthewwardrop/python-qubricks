import numpy as np

def energy_spectrum(system, states, ranges, components=[], hamiltonian=None, input=None, output=None, params={}, params_init=None):
    '''
    Returns the energy eigenvalues of the states which map adiabatically to
    those provided. The states provided should ideally be eigenstates when the
    parameters are set according to `params_init` (but close enough is good enough).
    Since this method uses the ordering of the eigenvalues to detect which eigenvalues
    belong to which eigenstates, this method does not work in cases when the adiabatic
    theorem is violated (i.e. when energy levels cross).
    '''

    if hamiltonian is None:
        hamiltonian = system.H(*components)
    hamiltonian = hamiltonian.change_basis(system.basis(output), params=params)

    states = system.subspace(states, input=input, output=output, params=params)

    if type(ranges) is not dict:
        raise ValueError("Multi-dimensional ranges are not supported; and so ranges must be a dictionary.")

    # IDENTIFY WHICH STATES BELONG TO WHICH LABELS BY PROJECTING THEM ONTO EIGENSTATES
    if params_init is None:
        params_init = params
    evals,evecs = np.linalg.eig(hamiltonian(**params_init))
    evecs = evecs[:,np.argsort(evals)]
    evals = np.sort(evals)
    indicies = []

    for state in states:
        indicies.append(np.argmax(np.dot(state,evecs)))


    # Now iterate over the ranges provided, allocating state labels according
    # to the state which adiabatically maps to the current state. This assumes no
    # level crossings.
    f_ranges = params.copy() # merge ranges and params to ensure consistency
    f_ranges.update(ranges)
    rvals = system.p.range(*ranges.keys(),**f_ranges)
    if type(rvals) != dict:
        rvals = {ranges.keys()[0]: rvals}

    results = np.zeros((len(states),len(rvals.values()[0])))
    for i in xrange(len(rvals.values()[0])):
        vals = {}
        for val in rvals:
            vals[val] = rvals[val][i]
        evals = sorted(np.linalg.eigvals(hamiltonian(**vals)))
        results[:,i] = [evals[indicies[j]] for j in xrange(len(indicies))]

    return results
