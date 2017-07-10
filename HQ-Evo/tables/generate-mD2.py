#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import bisect as bisect

c4d9 = 4./9.
c1d9 = 1./9.
c16pi = 16.*np.pi
c48pi = 48.*np.pi;
c16pi2 = 16.*np.pi*np.pi;
c64d9pi2 = 64./9.*np.pi*np.pi;
c256pi4 = 256.*np.pi**4;
Nc = 3
nf = 3;
pf_g = 4.*np.pi/3.*(Nc + nf/2.)
#const double pf_g = 8./np.pi*(Nc + nf);
pf_q = np.pi/2.*(Nc*Nc - 1)/2./Nc;
alpha0 = 4.*np.pi/(11. - 2./3.*nf);
Lambda = 0.2
Lambda2 = Lambda*Lambda;
Q2cut_l = -Lambda2*np.exp(alpha0)
Q2cut_h = Lambda2*np.exp(np.pi*np.tan(np.pi*(0.5-1./alpha0)))
print (Q2cut_h)
def alpha_s(Q2):
	if Q2<Q2cut_l:
		return alpha0/np.log(-Q2/Lambda2)
	elif Q2 <= Q2cut_h:
		return 1.
	else:
		return alpha0 * (0.5 - np.arctan(np.log(Q2/Lambda2)/np.pi)/np.pi)
alpha_s = np.vectorize(alpha_s)

def equation(mD2, T):
	return pf_g*alpha_s(-mD2) - mD2/T/T

T = np.linspace(0.1, 1.0, 100)
mD2 = np.array([bisect(equation, 0., 10., args=(t)) for t in T])
plt.plot(T, mD2)
plt.show()
f = h5py.File('mD2.hdf5', 'w')
ds = f.create_dataset('mD2', data=mD2)
ds.attrs.create('TL', 0.1)
ds.attrs.create('TH', 1.0)
ds.attrs.create('NT', 100, dtype=int)
print(mD2) 
f.close()
