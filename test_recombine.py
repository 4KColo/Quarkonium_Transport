#!/usr/bin/env python

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.integrate as si
from Quarkonium_evolution import QQbar_evol


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


#### ----------- recombination cross section * v_rel ---------- ####
#---- first define moving frame factor ----
def fac1(z):
	return np.log(1.0-np.exp(-z))


# -------- first we assume the potential is unscreened Coulomb ------
#---- matrix element squared of 1S: |< 1S|r| Psi_p >|^2 ----
def matrix_1S(p):
	eta = alpha_s*M/(4.0*N_C*p)
	Nume = 2.0**9*3.1416**2*a_B**5*eta*(2.0+rho_c)**2*(rho_c**2+(a_B*p)**2)*np.exp(4.0*eta*np.arctan(a_B*p))
	return Nume/( (1.0+(a_B*p)**2)**6 * (np.exp(2.0*3.1416*eta)-1.0) )


# --- formation cross section times relative velocity in the QQbar CM frame ------
#------ p is the relative momentum between the QQbar in the QQbar CM frame ------
def frateV_1S(v, T, p): 	# rate * Vol = sigma * v_rel
	q = p**2/M + E_1S
	if v == 0.0:
		return 4.0*alpha_s*T_F/N_C/3.0*( 1.0+1.0/(np.exp(q/T)-1.0) )*q**3*matrix_1S(p)
	else:
		gamma = 1.0/np.sqrt(1.0-v**2)
		angle_part = 2.0 + T/(gamma*q*v)*( fac1(q*gamma*(1+v)/T)-fac1(q*gamma*(1-v)/T) )
		return 2.0*alpha_s*T_F/N_C/3.0*q**3*matrix_1S(p)*angle_part
		
		
#### -------- end of recombination cross section * v_rel ---------- ####



####-------- define a function for Lorentz transformation --------- ####
def lorentz(p4, v):
#----- first we add some error checking
	if len(p4) != 4:
		raise ValueError('first argument of lorentz transformation must be a 4-vector')
	if len(v) !=3:
		raise ValueError('second argument of lorentz transformation must be a 3-vector')
	v_sq = v[0]**2+v[1]**2+v[2]**2
	if v_sq == 0.0:
		return np.array([p4[1],p4[2],p4[3]])
	else:
		gamma = 1.0/np.sqrt(1.0-v_sq)
		vdotp = v[0]*p4[1]+v[1]*p4[2]+v[2]*p4[3]
		E = gamma*( p4[0] - vdotp )
		px = -gamma*v[0]*p4[0] + p4[1] +(gamma-1.0)*v[0]*vdotp/v_sq
		py = -gamma*v[1]*p4[0] + p4[2] +(gamma-1.0)*v[1]*vdotp/v_sq
		pz = -gamma*v[2]*p4[0] + p4[3] +(gamma-1.0)*v[2]*vdotp/v_sq
		return np.array([px,py,pz])
		#return [E,px,py,pz]


'''
def test_lorentz(p1, p2, mass):
	p1 = np.array(p1)
	E1 = np.sqrt( np.sum(p1**2) + mass**2 )
	p2 = np.array(p2)
	E2 = np.sqrt( np.sum(p2**2) + mass**2 )
	p_com = p1 + p2
	E = np.sqrt( np.sum(p_com**2) + (2.0*mass)**2 )
	v3 = p_com/E
	p1_com = lorentz(np.append(E1, p1), v3)
	p2_com = lorentz(np.append(E2, p2), v3)
	print p1_com, p2_com
	
test_lorentz([100.0, 2000.0, 2000.0], [200.0, 2000.0, 2000.0], M)	
'''	
	
	

####---------- end of lorentz transformation function ----------- ####




#### ------------ thermal distribution sampling ------------- ####
# thermal distribution function W/O the fugacity factor
def thermal_dist(temp, mass, momentum):
	return momentum**2*np.exp(-np.sqrt(mass**2+momentum**2)/temp)

# sample according to the thermal distribution function
def thermal_sample(temp, mass):
	p_max = np.sqrt(2.0*temp**2 + 2.0*temp*np.sqrt(temp**2+mass**2))	
	# most probably momentum
	p_uplim = 10.0*p_max
	y_uplim = thermal_dist(temp, mass, p_max)
	while True:
		p_try = rd.uniform(0.0, p_uplim)
		y_try = rd.uniform(0.0, y_uplim)
		if y_try < thermal_dist(temp, mass, p_try):
			break
	
	cos_theta = rd.uniform(-1.0, 1.0)
	sin_theta = np.sqrt(1.0-cos_theta**2)
	phi = rd.uniform(0.0, 2.0*np.pi)
	cos_phi = np.cos(phi)
	sin_phi = np.sin(phi)
	return np.array([ p_try*sin_theta*cos_phi, p_try*sin_theta*sin_phi, p_try*cos_theta ])	



'''
##------------------ test on the sampling ---------------- ##
N_test = 10000
T_test = 250.0
p_test = []
E_test = []
for i in range(N_test):
	a = np.sqrt(np.sum(thermal_sample(T_test, M)**2))
	p_test.append(a)
	E_test.append(np.sqrt(a**2+M**2))

norm = si.quad(lambda x: x**2*np.exp(-np.sqrt(x**2+M**2)/T_test), 0.0, 1000.0*T_test)[0]
p_plot = np.linspace(0.0, 5000.0, 1000)
E_plot = np.sqrt(p_plot**2+M**2)
y_p = p_plot**2*np.exp(-E_plot/T_test)/norm
y_E = p_plot*E_plot*np.exp(-E_plot/T_test)/norm

plt.figure(0)
plt.hist(p_test, bins = 50, normed = 1, facecolor='blue', edgecolor='black')
plt.plot(p_plot, y_p)
plt.xlabel(r'$p(MeV)$', size = 20)
plt.figure(1)
plt.hist(E_test, bins = 50, normed = 1, facecolor='blue', edgecolor='black')
plt.plot(E_plot, y_E)
plt.xlabel(r'$E(MeV)$', size = 20)
plt.show()
## --------------------- end of test plot ------------------ ##
'''

#### ------------ end of thermal distribution sampling ------------- ####

def frate_event(temp):
	# calculate formation rate of one recombination event
	# sampled from thermal distribution
	p1 = thermal_sample(temp, M)	# b-quark 3-momentum
	p2 = thermal_sample(temp, M)	# bbar-quark 3-momentum
	E1 = np.sqrt( np.sum(p1**2) + M**2 )
	E2 = np.sqrt( np.sum(p2**2) + M**2 )
	p_com = p1 + p2
	E_com = np.sqrt( np.sum(p_com**2) + (2.0*M)**2 )
	v_vec = p_com/E_com
	v_abs = np.sqrt(np.sum(v_vec**2))
	p1_com = lorentz(np.append(E1, p1), v_vec)
	p2_com = lorentz(np.append(E2, p2), v_vec)
	p_rel = (p1_com - p2_com)/2.0
	pr_abs = np.sqrt( np.sum(p_rel**2) )
	return frateV_1S(v_abs, temp, pr_abs)


Vbox = 1000.0	# fm^3, cubic box with side length 10 fm
Nb = 50		

N_sample = 15000
def frate_normed(temp, number_b, volume):		# average rate for one b quark
	# volume is in fm^3
	norm = number_b / volume * C1**3
	# use Monte Carlo integration
	rate_f = []
	for i in range(N_sample):
		rate_f.append( frate_event(temp) )
	rate_f = np.array(rate_f)
	ave_rate_f = np.sum(rate_f)/N_sample
	variance = np.sum( (rate_f - ave_rate_f)**2 )/(N_sample - 1.0)
	return norm * ave_rate_f, norm * np.sqrt(variance/N_sample)	# need to multiply two factors of norm_dist and divide one




#### -------- Monte Carlo integration plot --------- ####
T = np.linspace(150.0, 500.0, 25)

len_T =len(T)
formation_rate_b_up = []	# averaged formation rate of a b quark in the medium
formation_rate_b_down = []
for j in range (len_T):
	rate_ave, rate_sigma = frate_normed(T[j], Nb, Vbox) 
	formation_rate_b_up.append(rate_ave+rate_sigma)
	formation_rate_b_down.append(rate_ave-rate_sigma)




#### ------------ multiple runs averaged and compare ---------------- ####
T_test = np.linspace(150.0, 450.0, 4)
len_T = len(T_test)
N_ave = 1000		# #of parallel runnings		
N_step = 3000	
dt = 0.04
tmax = N_step*dt
t = np.linspace(0.0, tmax, N_step+1)
Nq = 50		# initial number of Q or Qbar
N1s = 0		# initial number of U1s

ratef_T = []
ratef_var = []

## initialize N_ave number of events
for j in range(len_T):
	events = []
	for i in range(N_ave):
		events.append(QQbar_evol('static', temperature = T_test[j]))
		events[i].initialize(N_Q = Nq, N_U1S = N1s)
	
	test_rate = []
	for i in range(N_step+1):
		for j in range(N_ave):
			test_rate.append(events[j].testrun())
	
	test_rate = np.array(test_rate)
	test_rate_ave = np.sum(test_rate)/N_ave/(N_step+1.0)
	test_rate_var = np.sum( (test_rate - test_rate_ave)**2 )/N_ave/(N_step+1.0)
	ratef_T.append(test_rate_ave)
	ratef_var.append(test_rate_var)

ratef_var = np.array(ratef_var)

plt.figure(0)
plt.fill_between(T, formation_rate_b_down, formation_rate_b_up, color='blue')
plt.errorbar(T_test, ratef_T, yerr = np.sqrt(ratef_var), color='black', fmt='o')
plt.xlabel(r'$T(MeV)$', size = 20)
plt.ylabel(r'$\bar{\Gamma^f_{1S}}(MeV)$', size = 20)
plt.show()


# T = 150: 0.08448
# T = 250: 0.08140
# T = 350: 0.07479
# T = 450: 0.06977





		
		
		







		
