#!/usr/bin/env python

import numpy as np
import pylab as plt
import mpmath as mp
import cmath as cm
import csv
import scipy.optimize as so
import scipy.integrate as si
import scipy.special as ss
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import h5py


alpha_s = 0.3 # for bottomonium
#alpha_s = 0.55 # for charmonium
N_C = 3.0
M = 4650.0 # MeV b-quark
#M = 1270.0  # MeV c-quark
rho_c = 1.0/(N_C**2-1.0)
C_F = 4.0/3.0
a_B = 2.0/(alpha_s*C_F*M)
E_1S = alpha_s*C_F/(2.0*a_B)  # here is magnitude, true value is its negative



###------- first we assume the potential is unscreened Coulomb ------
#---- matrix element squared of 1S: |< 1S|r| Psi_p >|^2
def matrix_1S(p):
	eta = alpha_s*M/(4.0*N_C*p)
	Nume = 2.0**9*3.1416**2*a_B**5*eta*(2.0+rho_c)**2*(rho_c**2+(a_B*p)**2)*np.exp(4.0*eta*np.arctan(a_B*p))
	return Nume/( (1.0+(a_B*p)**2)**6 * (np.exp(2.0*3.1416*eta)-1.0) )


#---- cross section of 1S
def cros_1S(q):
	assert isinstance(q,float)
	t1 = np.sqrt(q/E_1S-1.0) #-- the value of q must be larger than E_1S
	Const = alpha_s*C_F/3.0 *2.0**10*3.1416**2 * rho_c*(2.0+rho_c)**2
	return Const*E_1S**4/(M*q**5)*(t1**2+rho_c**2)*np.exp(4.0*rho_c/t1*np.arctan(t1))/(np.exp(2.0*3.1416*rho_c/t1)-1.0)


#---- one can also use the following cross section expression to test the matrix element is correct
def cros_1S_test(q):
	p = np.sqrt(M*(q-E_1S))
	return 2.0/3.0*alpha_s*C_F*q*M*p*matrix_1S(p)


#---- moving frame factor
#---- PAY ATTNENTION to how spence is defined in scipy.special
def fac1(z):
	return z*np.log(1.0-np.exp(-z))-ss.spence(1.0-np.exp(-z))


#---- decay rate of 1S in the medium frame, v = c.o.m. velocity of quarkonium
#---- T is the medium temperature at the position of quarkonium
def drate_1S(v,T):
	gamma = 1.0/np.sqrt(1.0-v**2)
	I = si.quad(lambda q: cros_1S(q)*( fac1(q*gamma*(1+v)/T)-fac1(q*gamma*(1-v)/T) ), E_1S, 100*E_1S )[0]
	return I * T**2/(4.0*3.1416**2*gamma*v)

#---- integrand of dq, used in the initial state sampling
def initsam(v,T,q):
	gamma = 1.0/np.sqrt(1.0-v**2)
	return cros_1S(q)*( fac1(q*gamma*(1+v)/T)-fac1(q*gamma*(1-v)/T) ) * T**2/(4.0*3.1416**2*gamma*v)



#---- formation cross section in the rest frame of the QQbar pair
#---- p is the relative momentum between the QQbar
def fcros_1S(p, T):
	v_rel = p/np.sqrt(p**2/4.0+M**2)
	q = p**2/M + E_1S
	return 4.0*alpha_s*C_F/(3.0*v_rel)*q**3/(1.0-np.exp(-q/T))*matrix_1S(p)







#---- build the dissociation rate table ---
g_disso = []

v = np.linspace(0.01,0.99,50)
T = np.linspace(150.0, 500.0, 101)
len_v = len(v)
len_T = len(T)

X1, Y1 =np.meshgrid(v,T)
vT = np.array([X1.flatten(),Y1.flatten()]).T

#print vT


for i in range(len_T):
	for j in range(len_v):
		g_disso.append( [drate_1S(v[j],T[i])] ) #--- for the necessity of "[]", see the following codes
												#--- unit is MeV

g_disso = np.array(g_disso)
table_g_disso = np.append(vT,g_disso,axis=1)


f1 = h5py.File('b_g_disso.hdf5')
dset1 = f1.create_dataset('ds1',data=table_g_disso)
f1.close()



#---- construct the dissociation initial sampling table ----
q = np.linspace(E_1S+0.01, 10*E_1S+0.01, 101)
len_q = len(q)

array_vTq = np.vstack(np.meshgrid(v,T,q)).reshape(3,-1).T
#### Attention! index of q changes first, then that of v and finally that of T
sam_g_disso = [] #to construct the table for initial sampling
max_sam_g_disso = [] # obtain the max value when sampling, which depends on v,T

for i in range(len_T):
	for j in range(len_v):
		sam_q = []
		for k in range(len_q):
			sam_q.append(initsam(v[j],T[i],q[k]))
			sam_g_disso.append( [initsam(v[j],T[i],q[k])] )
		max_sam_g_disso.append( [max(sam_q)] )
		


sam_g_disso = np.array(sam_g_disso)
table_sam_g_disso = np.append(array_vTq, sam_g_disso, axis=1)	


max_sam_g_disso = np.array(max_sam_g_disso)
table_max_sam_g_disso = np.append(vT, max_sam_g_disso, axis=1)


f3 = h5py.File('sam_g_disso.hdf5')
dset3 = f3.create_dataset('ds3',data=table_sam_g_disso)
f3.close()


f4 = h5py.File('max_sam_g_disso.hdf5')
dset4 = f4.create_dataset('ds4',data=table_max_sam_g_disso)
f4.close()




#---- build the formation cross section table ---
g_form = []		
#p_rel = np.exp(np.linspace(3.9, 8.9, 100))  # for bottomonium
p_rel = np.exp(np.linspace(3.0, 8.0, 101))  # for charmonium
len_p = len(p_rel)

X2, Y2 = np.meshgrid(p_rel,T)
pT = np.array([X2.flatten(),Y2.flatten()]).T


for i in range(len_T):
	for j in range(len_p):
		g_form.append( [fcros_1S(p_rel[j],T[i])*197.326**2] )  ### convert from MeV^-2 to fm^2 

g_form = np.array(g_form)
table_g_form = np.append(pT,g_form,axis=1)




f2 = h5py.File('b_g_form.hdf5')
dset2 = f2.create_dataset('ds2',data=table_g_form)
f2.close()




'''
#####-----------------#####################
#----- this part of code is to explore how to form a 3-D array in a useful way ----
#----- check how to extend arrays
x = np.linspace(0.1,1.0,10)
y = np.linspace(5, 50, 10)
X, Y =np.meshgrid(x,y)

xy = np.array([X.flatten(),Y.flatten()]).T
p = []
for i in range(10):
	for j in range(10):
		p.append([y[i]*x[j]])

p = np.array(p)

A = np.append(xy,p,axis=1)
print A
#######---------------##################
'''



'''
rate_1 = []
rate_5 = []
rate_9 = []
fcros = []
T = np.linspace(170.0, 450.0, 1000)
p = np.linspace(1.0, 10000.0, 1000)
length = len(T)
for i in range(length):
	fcros.append(fcros_1S(p[i],300.0)*197.326**2)
	#rate_1.append(drate_1S(0.1,T[i]))
	#rate_5.append(drate_1S(0.5,T[i]))
	#rate_9.append(drate_1S(0.9,T[i]))


plt.figure()
plt.plot(p, fcros, linewidth=2.0, color='blue',label='$T=300.0$ MeV')
#plt.plot(T, rate_1, linewidth=2.0, color='blue',label='$v=0.1$')
#plt.plot(T, rate_5, linewidth=2.0, color='red',label='$v=0.5$')
#plt.plot(T, rate_9, linewidth=2.0, color='green',label='$v=0.9$')
#plt.xlim([0.001, 0.01])
#plt.xlabel(r'$T(MeV)$', size=20)
#plt.ylabel(r'$\Gamma^d_{1S}(MeV)$', size=20)
plt.xscale('log')
plt.legend(loc='upper left')
#plt.savefig('1S_test.eps')
plt.show()
'''


