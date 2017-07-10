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
N_v = 99.0
dv = (v_max-v_min)/N_v

T_min = 150.0
T_max = 500.0
N_T = 100.0
dT = (T_max-T_min)/N_T

p_rel_min = 4.0 #ln value of log(p_rel) min and max, used in the formation cross section
p_rel_max = 8.6
N_pr = 100.0
dp_r = (p_rel_max - p_rel_min)/N_pr




#### ------- open file of formation rate * V and read in ---------
f_form = h5py.File('b_g_form.hdf5','r')
T_form = f_form['ds'].value
f_form.close()



####----- define a function for Lorentz transformation ---------
def lorentz(p4, v):
#----- first we add some error checking
	if len(p4) != 4:
		raise ValueError('first argument of lorentz transformation must be a 4-vector')
	if len(v) !=3:
		raise ValueError('second argument of lorentz transformation must be a 3-vector')
	v_sq = v[0]**2+v[1]**2+v[2]**2
	if v_sq == 0.0:
		return p4
	else:
		gamma = 1.0/np.sqrt(1.0-v_sq)
		vdotp = v[0]*p4[1]+v[1]*p4[2]+v[2]*p4[3]
		E = gamma*( p4[0] - vdotp )
		px = -gamma*v[0]*p4[0] + p4[1] +(gamma-1.0)*v[0]*vdotp/v_sq
		py = -gamma*v[1]*p4[0] + p4[2] +(gamma-1.0)*v[1]*vdotp/v_sq
		pz = -gamma*v[2]*p4[0] + p4[3] +(gamma-1.0)*v[2]*vdotp/v_sq
		#return np.array([px,py,pz])
		return np.array([E,px,py,pz])

####------ end of lorentz transformation function -----------




####------ define heavy Q distribution used in recombination ----####
def dist_heavyQ(x):
	if len(x) != 3:
		raise ValueError('the argument in the heavy Q distribution function has to be 3-vector')
	x_sqrd = np.sum(x**2)
	radius = a_B*C1
	return np.exp(-x_sqrd/(2.0*radius**2))/(np.sqrt(2.0*np.pi)*radius)**3*2.0
# the factor 2 normalization is for the theta function of xdotp
	#return np.exp(-x_sqrd/(2.0*radius**2))/(np.sqrt(2.0*np.pi)*radius)**3
# the second one has no xdotp < 0 constraint in the evolution

####----------- end of heavy Q distribution function ----------#####



class QQbar_form:
	def __init__(self, x1, p1, x2, p2, temperature):
	# p1, p2, x1, x2 are all in the medium frame
	# p1 and p2 are 4-momentum, x1 and x2 are 3-position
		p1 = np.array(p1)
		p2 = np.array(p2)
		x1 = np.array(x1)
		x2 = np.array(x2)
	## assume all vectors are numpy array
		p_com =  p1[1:] + p2[1:] 	# com momentum in medium frame, corresponding to mass = 2M
		# if one uses p_com = (p1 + p2)/2, then should use mass = M
		
		self.r = x1 - x2
		self.R = 0.5*( x1 + x2 )
		self.rdotp = self.r[0]*(p1[1]-p2[1]) + self.r[1]*(p1[2]-p2[2]) + self.r[2]*(p1[3]-p2[3])
		
		#next step check: they must have passed each other (NOT NEEDED!)
		#r_next = self.r + p1/np.sqrt(np.sum(p1**2)+M**2) - p2/np.sqrt(np.sum(p2**2)+M**2)
		#self.rdotp_next = r_next[0]*(p1[0]-p2[0]) + r_next[1]*(p1[1]-p2[1]) + r_next[2]*(p1[2]-p2[2])
		
		self.T = temperature
		
		p_comsqd = np.sum(p_com**2)
		E_com = np.sqrt(p_comsqd + (2.0*M)**2)
		self.v = np.sqrt(p_comsqd)/E_com
		self.v3 = p_com/E_com

		p1_cm = lorentz(p1, self.v3)
		p2_cm = lorentz(p2, self.v3)
		
		p_rel = (p1_cm[1:] - p2_cm[1:])/2.0
		self.pr = np.sqrt( np.sum(p_rel**2) )
		
		
		
	def form_rate(self):
	# give the recombination rate based on v, T and p_rel
	# will do a linear interpolation
		ind_T = int((self.T - T_min)/dT)			# index of T
		z_T = (self.T - T_min)/dT - ind_T

		ind_v = int((self.v - v_min)/dv)			# index of v
		z_v = (self.v - v_min)/dv - ind_v
		
		ind_form00x = int( ind_v*(N_pr+1) + ind_T*(N_v+1)*(N_pr+1) )
		ind_form01x = ind_form00x + int(N_pr+1)
		ind_form10x = ind_form00x + int((N_v+1)*(N_pr+1))
		ind_form11x = ind_form10x + int(N_pr+1)
		
		pr_log = np.log(self.pr)
		if pr_log <= p_rel_min:
			ind_pr = 0
			formation_rate = (( T_form[ind_form00x][3]*(1.0-z_v) + T_form[ind_form01x][3]*z_v )*(1-z_T)
			+ ( T_form[ind_form10x][3]*(1.0-z_v) + T_form[ind_form11x][3]*z_v )*z_T)*self.pr/np.exp(p_rel_min)
			
		elif pr_log >= p_rel_max:
			ind_pr = int(N_pr)
			formation_rate = (( T_form[ind_form00x+ind_pr][3]*(1.0-z_v) + T_form[ind_form01x+ind_pr][3]*z_v )*(1-z_T)
			+ ( T_form[ind_form10x+ind_pr][3]*(1.0-z_v) + T_form[ind_form11x+ind_pr][3]*z_v )*z_T)
			
		else:
			ind_pr = int((pr_log - p_rel_min)/dp_r)		#index of pr
			z_pr = (pr_log - p_rel_min)/dp_r - ind_pr
			formation_rate_low = (( T_form[ind_form00x+ind_pr][3]*(1.0-z_v) + T_form[ind_form01x+ind_pr][3]*z_v )*(1-z_T)
			+ ( T_form[ind_form10x+ind_pr][3]*(1.0-z_v) + T_form[ind_form11x+ind_pr][3]*z_v )*z_T)
			formation_rate_upp = (( T_form[ind_form00x+ind_pr+1][3]*(1.0-z_v) + T_form[ind_form01x+ind_pr+1][3]*z_v )*(1-z_T)
			+ ( T_form[ind_form10x+ind_pr+1][3]*(1.0-z_v) + T_form[ind_form11x+ind_pr+1][3]*z_v )*z_T)
			
			formation_rate = formation_rate_low*(1-z_pr) + formation_rate_upp*z_pr
			
		return formation_rate * dist_heavyQ(self.r)
		
		#vol = np.abs(self.r[0]*self.r[1]*self.r[2])	# use 1/V as distribution function
		#return T_form[ind_form][3]/vol		# this is a bad choice since r[i] may be 0, i = x,y,z
		#return T_form[ind_form][3] * dist_heavyQ(self.r)
		#return T_form[ind_form][3]/4.0*3.0/np.pi*2.0		# for flat distribution
		
		
		
	def sample_final(self): # if the recombination occurs
	#in the CM frame of QQbar, sample final gluon direction w.r.t. the velocity direction (v // z axis)
		q_g = self.pr**2/M + E_1S
		while True:
			x = rd.uniform(-1.0, 1.0)		# x = cos(theta)
			x_sam = rd.uniform(0.0, 1.0/(1.0-np.exp(-q_g/np.sqrt(1.0-self.v**2)*(1.0-self.v)/self.T)) )
			f_x = 1.0/(1.0-np.exp(-q_g/np.sqrt(1.0-self.v**2)*(1.0+x*self.v)/self.T) )

			if x_sam <= f_x:
				break

		phi = rd.uniform(0.0, 2.0*np.pi)
		return [q_g, -x, phi]
		# the return value is the outgoing gluon energy, cos(theta) and phi w.r.t. velocity direction.
		# in the CM frame of QQbar, the formed quarkonium moves in the opposite direction of gluon
		# therefore we add a '-' in x
			
		
		
		
		
		
		
		