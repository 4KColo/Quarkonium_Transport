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
C1 = 197.327				  # 197 MeV*fm = 1

v_min = 0.0
v_max = 0.99
T_min = 150.0
T_max = 500.0
q_min = E_1S+0.01 # for sample initial gluon in the gluo-dissociation
q_max = 10*E_1S+0.01
p_rel_min = 4.0 #ln value of p_rel min and max, used in the formation cross section
p_rel_max = 8.6

N_v = 99.0   # this is the number of spacings, real number is N+1
N_T = 100.0
N_q = 100.0
N_pr = 100.0

### ----------------- first we assume the potential is unscreened Coulomb ------
#---- matrix element squared of 1S: |< 1S|r| Psi_p >|^2
def matrix_1S(p):
	eta = alpha_s*M/(4.0*N_C*p)
	Nume = 2.0**9*3.1416**2*a_B**5*eta*(2.0+rho_c)**2*(rho_c**2+(a_B*p)**2)*np.exp(4.0*eta*np.arctan(a_B*p))
	return Nume/( (1.0+(a_B*p)**2)**6 * (np.exp(2.0*3.1416*eta)-1.0) )


#---- decay cross section of 1S ----
def dcros_1S(q):
	assert isinstance(q,float)
	t1 = np.sqrt(q/E_1S-1.0) #-- the value of q must be larger than E_1S
	Const = alpha_s*C_F/3.0 *2.0**10*3.1416**2 * rho_c*(2.0+rho_c)**2
	return Const*E_1S**4/(M*q**5)*(t1**2+rho_c**2)*np.exp(4.0*rho_c/t1*np.arctan(t1))/(np.exp(2.0*3.1416*rho_c/t1)-1.0)


#---- one can also use the following cross section expression to test the matrix element is correct
def dcros_1S_test(q):
	p = np.sqrt(M*(q-E_1S))
	return 2.0/3.0*alpha_s*C_F*q*M*p*matrix_1S(p)


### ----------------- end of unscreened Coulomb potential results --------



### ------------------------ gluo-dissociation decay rate -------------------- ###
#---- first define moving frame factor ----
def fac1(z):
	return np.log(1.0-np.exp(-z))


#---- decay rate of 1S in the medium frame, v = c.o.m. velocity of quarkonium
#---- T is the medium temperature at the position of quarkonium
def drate_1S(v,T):
	if v == 0.0:
		I = si.quad(lambda q: q**2*dcros_1S(q)/(np.exp(q/T)-1.0), E_1S, 100*E_1S )[0]
		return I / (2.0*np.pi**2)
	else:
		gamma = 1.0/np.sqrt(1.0-v**2)
		I = si.quad(lambda q: q*dcros_1S(q)*( fac1(q*gamma*(1+v)/T)-fac1(q*gamma*(1-v)/T) ), E_1S, 100*E_1S )[0]
		return I * T / (4.0*3.1416**2*v*gamma**2)


#---- integrand of dq, used in the initial state sampling -----
def initsam(v,T,q):
	if v == 0:
		return q**2*dcros_1S(q)/(np.exp(q/T)-1.0) / (2.0*np.pi**2)
	else:
		gamma = 1.0/np.sqrt(1.0-v**2)
		return q*dcros_1S(q)*( fac1(q*gamma*(1+v)/T)-fac1(q*gamma*(1-v)/T) ) * T / (4.0*3.1416**2*v*gamma**2)


### ------------------------ end of gluo-dissociation -------------------- ###



### ------- formation cross section times relative velocity in the QQbar CM frame ------
#------ p is the relative momentum between the QQbar in the QQbar CM frame ------
def fcros_1S(v, T, p):
	q = p**2/M + E_1S
	if v == 0.0:
		return 4.0*alpha_s*C_F/3.0*( 1.0+1.0/(np.exp(q/T)-1.0) )*q**3*matrix_1S(p)
	else:
		gamma = 1.0/np.sqrt(1.0-v**2)
		angle_part = 2.0 + T/(gamma*q*v)*( fac1(q*gamma*(1+v)/T)-fac1(q*gamma*(1-v)/T) )
		return 2.0*alpha_s*C_F/3.0*q**3*matrix_1S(p)*angle_part




'''
#---- moving frame factor
#---- PAY ATTNENTION to how spence is defined in scipy.special
def fac1(z):
	return z*np.log(1.0-np.exp(-z))-ss.spence(1.0-np.exp(-z))


#---- decay rate of 1S in the medium frame, v = c.o.m. velocity of quarkonium
#---- T is the medium temperature at the position of quarkonium
def drate_1S(v,T):
	gamma = 1.0/np.sqrt(1.0-v**2)
	I = si.quad(lambda q: dcros_1S(q)*( fac1(q*gamma*(1+v)/T)-fac1(q*gamma*(1-v)/T) ), E_1S, 100*E_1S )[0]
	return I * T**2/(4.0*3.1416**2*gamma*v)

#---- integrand of dq, used in the initial state sampling
def initsam(v,T,q):
	gamma = 1.0/np.sqrt(1.0-v**2)
	return dcros_1S(q)*( fac1(q*gamma*(1+v)/T)-fac1(q*gamma*(1-v)/T) ) * T**2/(4.0*3.1416**2*gamma*v)



#------ formation cross section times relative velocity in the QQbar CM frame ------
#------ p is the relative momentum between the QQbar in the QQbar CM frame ------
def fcros_1S(p, T):
	q = p**2/M + E_1S
	return 4.0*alpha_s*C_F/3.0*q**3/(1.0-np.exp(-q/T))*matrix_1S(p)
'''






####-------------- build the dissociation rate table ---------#####
g_disso = []

v = np.linspace(v_min, v_max, N_v+1)
T = np.linspace(T_min, T_max, N_T+1)
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
f1.create_dataset('ds',data=table_g_disso)
f1.close()



####--------- construct the gluo-dissociation initial sampling table ----
q = np.linspace(q_min, q_max, N_q+1)
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


f2 = h5py.File('sam_g_disso.hdf5')
f2.create_dataset('ds',data=table_sam_g_disso)
f2.close()


f3 = h5py.File('max_sam_g_disso.hdf5')
f3.create_dataset('ds',data=table_max_sam_g_disso)
f3.close()


##### ------------- end of the dissociation rate table ---------#####




####  ---------- build the formation cross section * V table -------- ######
g_form = []		
p_rel = np.exp(np.linspace(p_rel_min, p_rel_max, N_pr+1))  # for bottomonium
len_p = len(p_rel)

array_vTp = np.vstack(np.meshgrid(v,T,p_rel)).reshape(3,-1).T
## Attention! index of p_rel changes first, then v's  and finally T's
## the -1 in the reshape means the length of second axis is calculated accordingly

#print array_vTp


for i in range(len_T):
	for j in range(len_v):
		for k in range(len_p):
			g_form.append( [C1**3*fcros_1S(v[j],T[i],p_rel[k])] ) 
g_form = np.array(g_form)
table_g_form = np.append(array_vTp,g_form,axis=1)



f2 = h5py.File('b_g_form.hdf5')
dset2 = f2.create_dataset('ds',data=table_g_form)
f2.close()


####  ---------- end of formation cross section * V table -------- ######




'''
#####-----------------#####################
####----- this part of code is to explore how to form a 3-D array in a useful way ----
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
### ------------- the following are for plots --------------####

rate_1 = []
rate_5 = []
rate_9 = []
fcros0 = []
fcros_1 = []
fcros_5 = []
fcros_9 = []
T = np.linspace(100.0, 450.0, 101)
p = np.linspace(60.0, 3500.0, 101)
length = len(T)
for i in range(length):
	rate_1.append(drate_1S(0.1,T[i]))
	rate_5.append(drate_1S(0.5,T[i]))
	rate_9.append(drate_1S(0.9,T[i]))
	fcros0.append(C1**3*fcros_1S(0.1,T[i],1/a_B))
	fcros_1.append(C1**3*fcros_1S(0.1,250.0,p[i]))
	fcros_5.append(C1**3*fcros_1S(0.5,250.0,p[i]))
	fcros_9.append(C1**3*fcros_1S(0.9,250.0,p[i]))


plt.figure(0)
plt.plot(T, rate_1, linewidth=2.0, color='blue',label='$v=0.1$')
plt.plot(T, rate_5, linewidth=2.0, color='red',label='$v=0.5$')
plt.plot(T, rate_9, linewidth=2.0, color='green',label='$v=0.9$')
plt.xlabel(r'$T(MeV)$', size=20)
plt.ylabel(r'$\Gamma^d_{1S}(MeV)$', size=20)
plt.legend(loc='upper left')
plt.savefig('1S_gluo-decay.eps')
plt.show()

plt.figure(1)
plt.plot(p, fcros_1, linewidth=2.0, color='blue',label='$v=0.1,T=250\ MeV$')
plt.plot(p, fcros_5, linewidth=2.0, color='red',label='$v=0.5,T=250\ MeV$')
plt.plot(p, fcros_9, linewidth=2.0, color='green',label='$v=0.9,T=250\ MeV$')
#plt.xlim([0.001, 0.01])
plt.xlabel(r'$p(MeV)$', size=20)
plt.ylabel(r'$V\Gamma^f_{1S}(MeV\cdot fm^3)$', size=20)
#plt.xscale('log')
plt.legend(loc='upper right')
plt.savefig('1S_gluo-form.eps')
plt.show()


plt.figure(0)
plt.plot(T, fcros0, linewidth=2.0, color='blue',label='$p=1/a_B$')
plt.xlabel(r'$T(MeV)$', size=20)
plt.ylabel(r'$V\Gamma^f_{1S}(MeV\cdot fm^3)$', size=20)
plt.legend(loc='upper left')
plt.savefig('1S_gluo-form2.eps')
plt.show()
'''