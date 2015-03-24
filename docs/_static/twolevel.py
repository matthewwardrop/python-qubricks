from qubricks.wall import SimpleQuantumSystem, LeakageMeasurement, ExpectationMeasurement, LindbladStateOperator
import numpy as np

q = SimpleQuantumSystem(
  hamiltonian={
    'B':[[1,0],[0,-1]],
    'J':[[0,1],[1,0]]
  },
  parameters={
    'B':(40,'neV'),
    'J':(40,'neV'),
    'T_2': (400,'ns'),
    'D': (lambda T_2,c_hbar: 0.5*c_hbar**2/T_2, '{mu}J^2*ns')
  },
  measurements={
    'E': ExpectationMeasurement( [[1,0],[0,-1]], [[0,1],[1,0]], [[0,-1j],[1j,0]] ),
  },
  derivative_ops={
    'J_noise': LindbladStateOperator(coefficient='D',operator=[[0,1],[1,0]])
  }
)

ts = np.linspace(0,1e-6,1000)
r = q.measure.E.integrate(psi_0s=[ [1,0] ],times=ts, operators=['evolution','J_noise'])

import matplotlib.pyplot as plt
from mplstyles import SampleStyle

style = SampleStyle()

with style:
	# If B=0, plot theoretical exponential decay curve
	if q.p.B == 0:
	    p_D = lambda t,D,c_hbar: np.exp(-2*D/c_hbar**2*t)
	    plt.plot(ts*1e9,q.p.range(p_D,t=ts),linestyle='--')

	plt.plot(r['time'][0]*1e9,r['expectation'][0,:,0],label="$\\left<Z\\right>$")
	plt.plot(r['time'][0]*1e9,r['expectation'][0,:,1],label="$\\left<X\\right>$")
	plt.plot(r['time'][0]*1e9,r['expectation'][0,:,2],label="$\\left<Y\\right>$")

	# Formatting options
	plt.grid()
	plt.legend(loc=0)
	plt.hlines([np.exp(-1),-np.exp(-1)],*plt.xlim())
	plt.xlabel('$\mathrm{Time}\,(ns)$')
	plt.ylabel("$E_Z$")

	style.savefig('results.pdf', polish=False, )
