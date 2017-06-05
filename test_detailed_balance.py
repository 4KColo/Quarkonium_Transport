#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
from Quarkonium_evolution import QQbar_evol


#### ------ theory prediction of equilibrium percentage of each particle ---- ####

def I_p(m, T):	#relativistic energy
	mbar = m/T
	integral = si.quad(lambda x: x**2*np.exp(-np.sqrt(x**2+mbar**2)), 0.0, 10.0*mbar, epsabs = 10**(-20) )[0]
	return T**3*integral/(2.0*np.pi**2)


def Inr_p(m, T):	#non-relativistic energy
	mbar = m/T
	integral = si.quad(lambda x: x**2*np.exp(-mbar-x**2/(2.0*mbar)), 0.0, 10.0*mbar, epsabs = 10**(-20) )[0]
	return T**3*integral/(2.0*np.pi**2)



alpha_s = 0.3 # for bottomonium
#alpha_s = 0.55 # for charmonium
N_C = 3.0
M = 4650.0 # MeV b-quark
#M = 1270.0  # MeV c-quark
rho_c = 1.0/(N_C**2-1.0)
C_F = 4.0/3.0
a_B = 2.0/(alpha_s*C_F*M)
E_1S = alpha_s*C_F/(2.0*a_B)  # here is magnitude, true value is its negative
C1 = 197.327				  # 197 MeV*fm = 1
M_1S = M*2-E_1S

def pct(N_tot, T, L):		# DO NOT forget the color and spin multiplies, but we do not have here
	# L is the side length in fm of the box
	# solve for fugacity first
	Vol = (L/C1)**3		#in MeV^-3
	a = I_p(M_1S, T)
	b = I_p(M, T)
	c = -N_tot/Vol
	fugacity = ( -b+np.sqrt(b**2-4.0*a*c) )/(2.0*a)
	r = fugacity*a/(fugacity*a+b)
	return r	#the percentage of #Upsilon


def pct_nr(N_tot, T, L):		# DO NOT forget the color and spin multiplies, but we do not have here
	# L is the side length in fm of the box
	# solve for fugacity first
	Vol = (L/C1)**3		#in MeV^-3
	a = Inr_p(M_1S, T)
	b = Inr_p(M, T)
	c = -N_tot/Vol
	fugacity = ( -b+np.sqrt(b**2-4.0*a*c) )/(2.0*a)
	r = fugacity*a/(fugacity*a+b)
	return r	#the percentage of #Upsilon

#### ------------------------ end of theory prediction ---------------- ####




#### ------------- single time run and compare --------------- ####
'''

T = 450.0
N_step = 7500
dt = 0.04
tmax = N_step*dt
t = np.linspace(0.0, tmax, N_step+1)
Nq = 50
N1s = 0
Nq_t = []
N1s_t = []


a = QQbar_evol('static',temperature = T)

a.initialize(N_Q = Nq, N_U1S = N1s)

for i in range(N_step+1):
	Nq_t.append(len(a.Qlist.x))
	N1s_t.append(len(a.U1Slist.x))
	#if len(a.U1Slist.p):
	#	print np.sqrt(np.sum((a.U1Slist.p[-1]/np.sqrt(np.sum(a.U1Slist.p[-1]**2)+M_1S**2))**2))
	a.run()

Nq_t = np.array(Nq_t)
N1s_t = np.array(N1s_t)

Nq_r = Nq_t/(Nq+N1s+0.0)
N1s_r = N1s_t/(Nq+N1s+0.0)


### equilibrium prediction (finite fugacity)
r_U1s = pct(Nq+N1s, T, 10.0)
r_Q = 1.0-r_U1s

r_U1s_nr = pct_nr(Nq+N1s, T, 10.0)
r_Q_nr = 1.0-r_U1s_nr

plt.figure()
#plt.plot(t, Nq_r, linewidth = 2.0, color='black', label=r'$Q$ or $\bar{Q}$ simulation')
#plt.plot(t, t*0.0+r_Q, linewidth = 2.0, color='blue', label=r'$Q$ or $\bar{Q}$ prediction')
#plt.plot(t, t*0.0+r_Q_nr, linewidth = 2.0, color='purple', linestyle='--', label=r'$Q$ or $\bar{Q}$ NR prediction')
plt.plot(t, N1s_r, linewidth = 2.0, color='red', label = r'$\Upsilon(1S)$ simulation')
plt.plot(t, t*0.0+r_U1s, linewidth = 2.0, color='green', label = r'$\Upsilon(1S)$ prediction')
plt.plot(t, t*0.0+r_U1s_nr, linewidth = 2.0, color='orange', linestyle='--', label=r'$Q$ or $\bar{Q}$ NR prediction')
plt.xlabel('$t$ (fm)', size = 20)
plt.ylabel(r'$N/N_{tot}$', size = 20)
#plt.ylim([0.0, 1.0])
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig('test_plot/detban_T='+str(T)+'Nb='+str(Nq)+'Nu='+str(N1s)+'.eps')
plt.show()

'''
#### ------------- end of single time run and compare --------------- ####







#### ------------ multiple runs averaged and compare ---------------- ####
N_ave = 49		# #of parallel runnings
T = 250.0		
N_step = 7500	
dt = 0.04
tmax = N_step*dt
t = np.linspace(0.0, tmax, N_step+1)
Nq = 50		# initial number of Q or Qbar
N1s = 0		# initial number of U1s
N1s_t = []	# to store number of U1s in each time step
Nq_t = []	# to store number of Q or Qbar in each time step


## initialize N_ave number of events
events = []
for i in range(N_ave):
	events.append(QQbar_evol('static',temperature = T))
	events[i].initialize(N_Q = Nq, N_U1S = N1s)

for i in range(N_step+1):
	len_u1s = 0.0
	for j in range(N_ave):
		len_u1s += len(events[j].U1Slist.x)
		events[j].run()
	N1s_t.append(len_u1s/N_ave)
	Nq_t.append(Nq-len_u1s/N_ave)

Nq_t = np.array(Nq_t)
N1s_t = np.array(N1s_t)

Nq_r = Nq_t/(Nq+N1s+0.0)
N1s_r = N1s_t/(Nq+N1s+0.0)


### equilibrium prediction (finite fugacity)
r_U1s = pct(Nq+N1s, T, 10.0)
r_Q = 1.0-r_U1s

r_U1s_nr = pct_nr(Nq+N1s, T, 10.0)
r_Q_nr = 1.0-r_U1s_nr


plt.figure(1)
#plt.plot(t, Nq_r, linewidth = 2.0, color='black', label=r'$Q$ or $\bar{Q}$ simulation')
#plt.plot(t, t*0.0+r_Q, linewidth = 2.0, color='blue', label=r'$Q$ or $\bar{Q}$ prediction')
#plt.plot(t, t*0.0+r_Q_nr, linewidth = 2.0, color='purple', linestyle='--', label=r'$Q$ or $\bar{Q}$ NR prediction')
plt.plot(t, N1s_r, linewidth = 2.0, color='red', label = r'$\Upsilon(1S)$ simulation')
plt.plot(t, t*0.0+r_U1s, linewidth = 2.0, color='green', label = r'$\Upsilon(1S)$ prediction')
plt.plot(t, t*0.0+r_U1s_nr, linewidth = 2.0, color='orange', linestyle='--', label=r'$Q$ or $\bar{Q}$ NR prediction')
plt.xlabel('$t$ (fm)', size = 20)
plt.ylabel(r'$N/N_{tot}$', size = 20)
#plt.ylim([0.0, 1.0])
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig('test_plot/detban_ave_T='+str(T)+'Nb='+str(Nq)+'Nu='+str(N1s)+'.eps')
plt.show()



















#### ------------ end of multiple runs averaged and compare ---------- ####

