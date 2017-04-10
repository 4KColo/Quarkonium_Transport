#!/usr/bin/env python
import numpy as np
import random as rd
import h5py

alpha_s = 0.3 # for bottomonium
N_C = 3.0
M = 4650.0 # MeV b-quark
rho_c = 1.0/(N_C**2-1.0)
C_F = 4.0/3.0
a_B = 2.0/(alpha_s*C_F*M)
E_1S = alpha_s*C_F/(2.0*a_B)  # here is magnitude, true value is its negative
M_1S = M*2.0 - E_1S  		  # mass of Upsilon(1S)
C1 = 197.327				  # 197 MeV*fm = 1


#----- open files to store tables: T_decay is the decay rate of a quarkonium
#----- T_sam is the integrand of dq , the integral gives decay rate
#----- T_maxsam is the maximum value of the integrand, used to sample a uniform distribution
f_decay = h5py.File('b_g_disso.hdf5','r')
f_sam = h5py.File('sam_g_disso.hdf5','r')
f_maxsam = h5py.File('max_sam_g_disso.hdf5','r')

T_decay = f_decay['ds1'].value
T_sam = f_sam['ds3'].value
T_maxsam = f_maxsam['ds4'].value

f_decay.close()
f_sam.close()
f_maxsam.close()


class QQbar_decay:
	def __init__(self, com_momentum, temperature):
	# com_momentum is the momentum vector of the quarkonium in the hydro cell frame
	#---------- and it has to be a NUMPY array ------ !!!!!!!!
	# initialize the quarkonium velocity in the hydro cell frame and the temperature
		self.v3 = com_momentum/np.sqrt(np.sum(com_momentum**2)+M_1S**2)
		self.v = np.sqrt(np.sum(self.v3**2))
		self.T = temperature
		self.ind_v = int((self.v-0.01)/0.02)
		self.ind_T = int((self.T-150)/3.5)
		self.ind_decay = self.ind_T*50+self.ind_v
		
		
	def decay_rate(self):
		return T_decay[self.ind_decay][2]

			
	def sample_init(self): # this step is in the hydro cell
		maxsam = T_maxsam[self.ind_decay][2] 
		####--- max value of the sampling function (integrand of dq) given a v and T
		while True:
			q = rd.uniform(E_1S+0.01, 10*E_1S+0.01)
			f_q = rd.uniform(0.0, maxsam)
			index = int( (q-E_1S-0.01)/(E_1S/100.0) ) + self.ind_v*101 + self.ind_T*101*50
			if f_q < T_sam[index][3]:
				break
		while True:
			x = rd.uniform(-1.0, 1.0)  #sample angular part: cos(angle)
			ang_sam = rd.uniform(0.0, (1.0-self.v)/( np.exp(q/np.sqrt(1.0-self.v**2)/self.T*(1.0-self.v))-1.0 ) )
			### sample the angular distribution according to the max value at x = cos() = -1
			f_x = (1.0+x*self.v)/( np.exp(q/np.sqrt(1.0-self.v**2)/self.T*(1.0+x*self.v))-1.0 )
			if ang_sam < f_x:
				break
		phi = rd.uniform(0.0, 2.0*np.pi)
			
		return [q, x, phi]
#### return the energy of the gluon and the cos(theta) w.r.t. to the direction of v
#### the azimuthal angle phi
#### this q is already in the rest frame of quarkonium
		
		

	def sample_final(self, q): # this step is in the quarkonium rest frame
#### here q is the boosted value from the sample_init method
####-------------- q must be larger than E_1S !!!!! -----------
#### returns need to be boosted back to the hydro cell frame 
		theta = rd.uniform(-np.pi,np.pi)
		p_rel = np.sqrt( M*(q-E_1S) )
		phi = rd.uniform(0.0, 2.0*np.pi)
		return [p_rel, np.cos(theta), phi]
		### remember that one Q has p_rel/2 the other has -p_rel/2
	
	
	