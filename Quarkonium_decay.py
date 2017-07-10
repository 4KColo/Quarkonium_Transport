#!/usr/bin/env python
import numpy as np
import random as rd
import h5py

alpha_s = 0.3 # for bottomonium
N_C = 3.0
T_F = 0.5
M = 4650.0 # MeV b-quark
rho_c = 1.0/(N_C**2-1.0)
C_F = 4.0/3.0
a_B = 2.0/(alpha_s*C_F*M)
E_1S = alpha_s*C_F/(2.0*a_B)  # here is magnitude, true value is its negative
M_1S = M*2.0 - E_1S  		  # mass of Upsilon(1S)
C1 = 197.327				  # 197 MeV*fm = 1


v_min = 0.0
v_max = 0.99
T_min = 150.0
T_max = 500.0
q_min = E_1S+0.01 # for sample initial gluon in the gluo-dissociation
q_max = 10*E_1S+0.01

N_v = 99.0
N_T = 100.0
N_q = 100.0

dv = (v_max - v_min)/N_v
dT = (T_max - T_min)/N_T
dq = (q_max - q_min)/N_q


#----- open files to store tables: T_decay is the decay rate of a quarkonium
#----- T_sam is the integrand of dq , the integral gives decay rate
#----- T_maxsam is the maximum value of the integrand, used to sample a uniform distribution
f_decay = h5py.File('b_g_disso.hdf5','r')
f_sam = h5py.File('sam_g_disso.hdf5','r')
f_maxsam = h5py.File('max_sam_g_disso.hdf5','r')

T_decay = f_decay['ds'].value
T_sam = f_sam['ds'].value
T_maxsam = f_maxsam['ds'].value

f_decay.close()
f_sam.close()
f_maxsam.close()

def angular_sample(A, v, y):
# we use the inverse transform method to sample the incoming gluon angles
# here A = q/T, v is the velocity of quarkonium, y is a random number [0,1]
	if v == 0.0:
		return 2.0*y - 1.0
	elif y == 1.0:
		return 1.0
	else:
		gamma_v = 1.0/np.sqrt(1.0-v**2)
		B = A*gamma_v
		C = y*np.log(1.0-np.exp(-B*(1.0+v))) + (1.0-y)*np.log(1.0-np.exp(-B*(1.0-v)))
		if C == 0.0:
			return 2.0*y - 1.0
		else:
			return (-np.log(1.0-np.exp(C))/B-1.0 )/v



class QQbar_decay:
	def __init__(self, com_momentum, temperature):
	# com_momentum is the 4-momentum vector of the quarkonium in the hydro cell frame
	#---------- and it has to be a NUMPY array ------ !!!!!!!!
	# initialize the quarkonium velocity in the hydro cell frame and the temperature
		self.v3 = com_momentum[1:]/com_momentum[0]
		self.v = np.sqrt(np.sum(self.v3**2))
		self.T = temperature
		self.ind_v = int((self.v-v_min)/dv)
		self.ind_T = int((self.T-T_min)/dT)
		
		self.ind_decay_00 = int(self.ind_T*(N_v+1.0)+self.ind_v)
		self.ind_decay_01 = 1 + self.ind_decay_00
		self.ind_decay_10 = int( N_v+1.0 + self.ind_decay_00 )
		self.ind_decay_11 = 1 + self.ind_decay_10
		
		self.z_v = (self.v - v_min)/dv - self.ind_v
		self.z_T = (self.T - T_min)/dT - self.ind_T
		
		
	def decay_rate(self):
		# will do a linear interpolation in temperature
		a0 = T_decay[self.ind_decay_00][2]*(1-self.z_v) + T_decay[self.ind_decay_01][2]*self.z_v
		a1 = T_decay[self.ind_decay_10][2]*(1-self.z_v) + T_decay[self.ind_decay_11][2]*self.z_v
		return a0*(1-self.z_T) + a1*self.z_T
		
		
	def sample_init(self): # this step is in the hydro cell
		maxsam = ( (T_maxsam[self.ind_decay_00][2]*(1-self.z_v) + T_maxsam[self.ind_decay_01][2]*self.z_v)*(1-self.z_T)
		+ (T_maxsam[self.ind_decay_10][2]*(1-self.z_v) + T_maxsam[self.ind_decay_11][2]*self.z_v)*self.z_T )
		####--- max value of the sampling function (integrand of dq) given a v and T
		while True:
			q = rd.uniform(q_min, q_max)
			ind_q = int((q-q_min)/dq)
			z_q = (q-q_min)/dq - ind_q
			
			f_q = rd.uniform(0.0, maxsam)
			
			index = int( int((q-q_min)/dq)  + self.ind_v*(N_q+1.0) + self.ind_T*(N_q+1.0)*(N_v+1.0) )
			index000 = int( ind_q + self.ind_decay_00*(N_q+1.0) )
			index001 = 1 + index000
			index010 = int( ind_q + self.ind_decay_01*(N_q+1.0) )
			index011 = 1 + index010
			index100 = int( ind_q + self.ind_decay_10*(N_q+1.0) )
			index101 = 1 + index100
			index110 = int( ind_q + self.ind_decay_11*(N_q+1.0) )
			index111 = 1 + index110
			
			f_00 = T_sam[index000][3]*(1-z_q)+T_sam[index001][3]*z_q
			f_01 = T_sam[index010][3]*(1-z_q)+T_sam[index011][3]*z_q
			f_10 = T_sam[index100][3]*(1-z_q)+T_sam[index101][3]*z_q
			f_11 = T_sam[index110][3]*(1-z_q)+T_sam[index111][3]*z_q
			f_sample = ( f_00*(1-self.z_v) + f_01*self.z_v )*(1-self.z_T)+( f_10*(1-self.z_v) + f_11*self.z_v )*self.z_T
			
			if f_q <= f_sample:
				break
		
		####--- then sample x = cos(theta) and phi
		y = rd.uniform(0.0,1.0)
		x = angular_sample(q/self.T, self.v, y)
		
		phi = rd.uniform(0.0, 2.0*np.pi)
			
		return [q, x, phi]
#### return the energy of the gluon and the cos(theta) w.r.t. to the direction of v
#### the azimuthal angle phi
#### this q is already in the rest frame of quarkonium
		
		

	def sample_final(self, q): # this step is in the quarkonium rest frame
#### here q is the boosted value from the sample_init method
####-------------- q must be larger than E_1S !!!!! -----------
#### returns need to be boosted back to the hydro cell frame 
		x = rd.uniform(-1.0, 1.0)	#cos(theta) is evenly distributed
		p_rel = np.sqrt( M*(q-E_1S) )
		phi = rd.uniform(0.0, 2.0*np.pi)
		return [p_rel, x, phi]
		### remember that one Q has p_rel the other has -p_rel
	
	
	