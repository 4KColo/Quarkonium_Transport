#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
from Quarkonium_evolution import QQbar_evol
import h5py


#### ------------------ some constants --------------- ####
alpha_s = 0.3 # for bottomonium
#alpha_s = 0.55 # for charmonium
N_C = 3.0
T_F = 0.5
M = 4650.0 # MeV b-quark
#M = 1270.0  # MeV c-quark
rho_c = 1.0/(N_C**2-1.0)
C_F = 4.0/3.0
a_B = 2.0/(alpha_s*C_F*M)
E_1S = alpha_s*C_F/(2.0*a_B)  # here is magnitude, true value is its negative
C1 = 197.327				  # 197 MeV*fm = 1
M_1S = M*2-E_1S

#### -------------- end of some constants ------------- ####



#### ------------ multiple runs averaged and compare ---------------- ####
N_ave = 1000	# #of parallel runnings
T0 = 499.9		# initial temperature
Tf = 155.0		# hadronization temperature
tau0 = 1.0 		# fm/c, time for the initial collision to reach hydro region
vs2 = 1.0/3.0	# speed of sound squared		
tauf = tau0/(Tf/T0)**(1.0/vs2)

dt = 0.04
N_step = int((tauf - tau0)/dt)
tmax = tau0 + N_step*dt
t = np.linspace(tau0, tmax, N_step+1)
T_t = T0*(tau0/t)**vs2


Nq0 = 50		# initial number of Q or Qbar
N1s0 = 0		# initial number of U1s
N1s_t = []		# to store number of U1s in each time step
Nq_t = []		# to store number of Q or Qbar in each time step



## initialize N_ave number of events
events = []
for i in range(N_ave):
	events.append(QQbar_evol('static', temperature = T0, HQ_scat = False))
	events[i].initialize(N_Q = Nq0, N_U1S = N1s0)


## next store the N(t), px, py, pz into arrays


for i in range(N_step+1):
	len_u1s_tot = 0.0
	
	for j in range(N_ave):
		len_u1s_tot += len(events[j].U1Slist.p)		# length of U1S list in one event
		events[j].run(temp_run = T_t[i])
	N1s_t.append(len_u1s_tot/N_ave)		# store the average Num. of U1S	


N1s_t = np.array(N1s_t)			# time-sequenced particle number
Nq_t = Nq0 + N1s0 - N1s_t		# time-sequenced particle number

N1s_r = N1s_t/(Nq0+N1s0+0.0)		# ratio
Nq_r = 1.0 - N1s_r				# ratio





#### ------------ end of multiple runs averaged and compare ---------- ####





#### ------------ save the data in a h5py file ------------- ####
Ep = []
for i in range(0, N_ave):
	if len(events[i].U1Slist.p) != 0:
		if len(Ep) == 0:
			Ep = events[i].U1Slist.p
		else:
			Ep = np.append(Ep, events[i].U1Slist.p, axis=0)


file1 = h5py.File('Bjorken_N_ave='+str(N_ave)+'T0='+str(T0)+'tau0='+str(tau0)+'Nb0='+str(Nq0)+'Nu0='+str(N1s0)+'.hdf5')
file1.create_dataset('percentage', data = N1s_r)
file1.create_dataset('time', data = t)
file1.create_dataset('4momentum', data = Ep)
file1.close()

#### ------------ end of saving the file ------------- ####





















