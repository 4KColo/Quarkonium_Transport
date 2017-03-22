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

T_min = 150.0
T_max = 500.0
N_T = 100.0
dT = (T_max-T_min)/N_T
p_rel_min = 4.0 #ln value of p_rel min and max, used in the formation cross section
p_rel_max = 8.6
N_pr = 100.0
dp_r = (p_rel_max - p_rel_min)/N_pr
f_form = h5py.File('b_g_form.hdf5','r')

class QQbar_form:
	def __init__(self, p1, p2, x1, x2, temperature):
		self.p_rel = 0.5*np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
		self.p_com = [ p1[0]+p2[0], p1[1]+p2[1] ]
		self.r = np.sqrt( (x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 )
		self.R = [ 0.5*(x1[0]+x2[0]), 0.5*(x1[1]+x2[1]) ]
		self.T = temperature
		self.ind_T = int((self.T-T_min)/dT)
		pr_ln = np.log(self.p_rel)
		self.ind_pr = int((pr_ln - p_rel_min)/dp_r)
		self.ind_form = self.ind_T*50+self.ind_pr
		
		
	def form_cros(self):
	# give the recombination cross section based on p_rel and T
		return f_form['ds2'].value[self.ind_form][2]
		
	def check(self):
		if 3.1416*self.r**2 > f_form['ds2'].value[self.ind_form][2]:
			return False
		else:
			return True
		
		
		
	def sample_final(self): # if the recombination occurs, i.e., if 'check' method returns True
	#in the com frame of QQbar, sample final gluon direction
		q_g = self.p_rel**2/M + E_1S
		theta = rd.uniform(-np.pi, np.pi)
		return [q_g, np.cos(theta)]
		# the return value is the outgoing gluon energy and cos(theta) direction.
		# in the com frame of QQbar, the formed quarkonium moves in the opposite direction
			
		
		
		
		
		
		
		