#!/usr/bin/env python
import numpy as np
import h5py
import random as rd
from scipy.spatial import cKDTree
from particle import Particlelist
from Quarkonium_decay import QQbar_decay
from Quarkonium_recombine import QQbar_form
from HQ_momentum import HQ_p_update


#### --------- some constants -----------------------------
alpha_s = 0.3 # for bottomonium
N_C = 3.0
T_F = 0.5
M = 4650.0 # MeV b-quark
rho_c = 1.0/(N_C**2-1.0)
C_F = 4.0/3.0
a_B = 2.0/(alpha_s*C_F*M)
E_1S = alpha_s*C_F/(2.0*a_B)  # Upsilon(1S), here is magnitude, true value is its negative
M_1S = M*2.0 - E_1S  		  # mass of Upsilon(1S)
C1 = 197.327				  # 197 MeV*fm = 1
R_search = 1.0				  # (fm), pair-search radius in the recombination


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



####------ define a rotation matrix that transforms initial ------
####------ and final state to the medium frame
def rotation(vec, theta, phi):
	if len(vec) !=3:
		raise ValueError('first argument of rotation must be a 3-vector')
	v1 = np.cos(theta)*np.cos(phi)*vec[0] - np.sin(phi)*vec[1] + np.sin(theta)*np.cos(phi)*vec[2]
	v2 = np.cos(theta)*np.sin(phi)*vec[0] + np.cos(phi)*vec[1] + np.sin(theta)*np.sin(phi)*vec[2]
	v3 = -np.sin(theta)*vec[0] + np.cos(theta)*vec[2]
	return np.array([v1,v2,v3])
	#return vec


####------ end of rotation matrix ------------------------



####------ define a function that gives the angle theta and phi ------
def angle(vec):
	if len(vec) !=3:
		raise ValueError('first argument of angle must be a 3-vector')
	len_vec = np.sqrt(np.sum(np.array(vec)**2))
	
	if len_vec == 0.0:
		theta = 0.0
	else:
		theta = np.arccos(vec[2]/len_vec)
	if vec[0] > 0:
		phi = np.arctan(vec[1]/vec[0])
	elif vec[0] == 0:
		phi = np.pi/2.0*(vec[1]>0)
	else:
		phi = np.arctan(vec[1]/vec[0]) + np.pi
	return [theta, phi]
	

####------ end of angle function --------------------




####--------- initial sample of heavy Q and Qbar using thermal distribution -------- ####
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
	
	E = np.sqrt(p_try**2+mass**2)
	cos_theta = rd.uniform(-1.0, 1.0)
	sin_theta = np.sqrt(1.0-cos_theta**2)
	phi = rd.uniform(0.0, 2.0*np.pi)
	cos_phi = np.cos(phi)
	sin_phi = np.sin(phi)
	#return np.array([ p_try*sin_theta*cos_phi, p_try*sin_theta*sin_phi, p_try*cos_theta ])
	return np.array([ E, p_try*sin_theta*cos_phi, p_try*sin_theta*sin_phi, p_try*cos_theta ])

####---- end of initial sample of heavy Q and Qbar using thermal distribution ---- ####




####--------- initial sample of heavy Q and Qbar using uniform distribution ------ ####
def uniform_sample(Pmax, mass):
	px, py, pz = (np.random.rand(3)-0.5)*2*Pmax
	E = np.sqrt(mass**2 + px**2 + py**2 + pz**2)
	return np.array([E, px, py, pz])
	
####---- end of initial sample of heavy Q and Qbar using uniform distribution ---- ####




class QQbar_evol:
#---- input the medium_type when calling the class ----
	def __init__(self, medium_type, temperature = 300.0, recombine = True, HQ_scat = False):
		self.type = medium_type		# for static type, the input temperature will be used
		self.T = temperature
		self.recombine = recombine
		self.HQ_scat = HQ_scat


#---- initialize the Q, Qbar and Quarkonium -- currently we only study Upsilon(1S)
	def initialize(self, N_Q = 100, N_U1S = 10, Lmax = 10.0, thermal_dist = True, decaytestmode = False, Pmax = 5000.0, P_decaytest = [0.0, 0.0, 5000.0]):
		self.Lmax = 10.0
#---- first determine medium type and then initialize
		if self.type == 'static':
	#---- static medium!! parameters only make sense in STATIC mode------
	#------ Lmax = 10 fm, 3-D box size  --------
	# if thermal_dist = False and decay_test_mode = False, 
	# then initial distribution is flat in all three directions, uniform in [0, Pmax]
	# if decaytestmode = True, give quarkonium initial [Px, Py, Pz] in the initialization

			self.Qlist = Particlelist()
			self.Qbarlist = Particlelist()
			self.U1Slist = Particlelist()
			
			self.Qlist.id = 5			# b quark
			self.Qbarlist.id = -5		# anti b quark
			self.U1Slist.id = 533		# Upsilon(1S)
	
			#--- sample their initial positions x
			self.Qlist.x = np.array([np.random.rand(3)*Lmax for i in range(N_Q)])
			self.Qbarlist.x = np.array([np.random.rand(3)*Lmax for i in range(N_Q)])
			self.U1Slist.x = np.array([np.random.rand(3)*Lmax for i in range(N_U1S)])
			
			
			#--- sample their initial momenta p
			if thermal_dist == True:
			# thermal distribution of initial momenta
				self.Qlist.p = np.array([ thermal_sample(self.T, M) for i in range(N_Q)])
				self.Qbarlist.p = np.array([ thermal_sample(self.T, M) for i in range(N_Q)])
				self.U1Slist.p = np.array([ thermal_sample(self.T, M_1S) for i in range(N_U1S)])
			
			elif decaytestmode == True:
			# the following is used in test of decay rates
				E_decaytest = np.sqrt(M**2 + P_decaytest[0]**2 + P_decaytest[1]**2 + P_decaytest[2]**2)
				self.Qlist.p = np.array([ [M, 0.0, 0.0, 0.0] for i in range(N_Q)])
				self.Qbarlist.p = np.array([ [M, 0.0, 0.0, 0.0] for i in range(N_Q)])
				self.U1Slist.p = np.array([ np.append(E_decaytest, P_decaytest) for i in range(N_U1S)])

			else:
			# uniform distribution of initial momenta
				self.Qlist.p = np.array([ uniform_sample(Pmax, M) for i in range(N_Q)])
				self.Qbarlist.p = np.array([ uniform_sample(Pmax, M) for i in range(N_Q)])
				self.U1Slist.p = np.array([ uniform_sample(Pmax, M_1S) for i in range(N_U1S)])
			

			

#####------------ evolution function -------------- #####
	def run(self, dt = 0.04, temp_run = -1.0):		#--- time step is universal since we include recombination
		len_U1S = len(self.U1Slist.x)
		len_Qbar = len(self.Qbarlist.x)		#notice len_Qbar = len_Q
		if temp_run != -1.0:
			self.T = temp_run
			
		### --------- update HQ momentum if HQ_scatter is switched on --------
		if self.HQ_scat == True:
			HQ_event = HQ_p_update()
			self.Qlist.p = np.array([ HQ_event.update(HQ_Ep = self.Qlist.p[i], Temp = self.T/1000.0, time_step = dt) for i in range(len_Qbar)])
			self.Qbarlist.p = np.array([ HQ_event.update(HQ_Ep = self.Qbarlist.p[i], Temp = self.T/1000.0, time_step = dt) for i in range(len_Qbar)])

		### --------- end of update HQ momentum ----------
		
		
		
		### --------- then free stream Q, Qbar, U1S ------------
		if len_U1S != 0:
			v_U1S = np.array( [self.U1Slist.p[i][1:]/self.U1Slist.p[i][0] for i in range(len_U1S)] )
			self.U1Slist.x = (self.U1Slist.x + dt*v_U1S )%self.Lmax
			
		if len_Qbar != 0:
			v_Q = np.array( [self.Qlist.p[i][1:]/self.Qlist.p[i][0] for i in range(len_Qbar)] )
			v_Qbar = np.array( [self.Qbarlist.p[i][1:]/self.Qbarlist.p[i][0] for i in range(len_Qbar)] )
			
			self.Qlist.x = (self.Qlist.x + dt*v_Q )%self.Lmax
			self.Qbarlist.x = (self.Qbarlist.x + dt*v_Qbar )%self.Lmax
			
		### ----------- end of free stream ---------------------
		
		
		
		### --------then consider the decay of U1S-------------
		### -------- start here -------------------------------
		
		delete_U1S = []
		add_pQ = []
		add_pQbar = []
		add_xQ = []
		#add_xQbar = [] the positions of Q and Qbar are the same
		for i in range(len_U1S):
			evt = QQbar_decay(com_momentum = self.U1Slist.p[i], temperature = self.T) #call the decay class
			
			# if the decay happens
			if evt.decay_rate()*dt/C1 >= np.random.rand(1):
				delete_U1S.append(i) # store the indexes that should be deleted later

			#else: nothing happens	
			
				# ----- sample the initial gluon and final QQbar -------
				q, costheta1, phi1 = evt.sample_init()  # gluon energy, angles in the quarkonium rest frame
				p_rel, costheta2, phi2 = evt.sample_final(q)  # QQbar in the same frame
				sintheta1 = np.sqrt(1.0-costheta1**2)
				sintheta2 = np.sqrt(1.0-costheta2**2)
			#--- all the following three momentum components are in the quarkonium rest frame
				tempmomentum_g = np.array([q*sintheta1*np.cos(phi1), q*sintheta1*np.sin(phi1), q*costheta1])
				tempmomentum_Q = np.array([p_rel*sintheta2*np.cos(phi2), p_rel*sintheta2*np.sin(phi2), p_rel*costheta2])
			#	tempmomentum_Qbar = -tempmomentum_Q (true in the quarkonium rest frame)
				
				#---- add the recoil momentum from the gluon
				recoil_p_Q = 0.5*tempmomentum_g + tempmomentum_Q
				recoil_p_Qbar = 0.5*tempmomentum_g - tempmomentum_Q
				
				#---- energy of Q and Qbar
				E_Q = np.sqrt(  np.sum(recoil_p_Q**2) + M**2  )
				E_Qbar = np.sqrt(  np.sum(recoil_p_Qbar**2) + M**2  )
				
			#--- the Q, Qbar momenta need to be rotated from the v = z axis to the medium frame
				#---- first get the rotation matrix angles
				theta_rot, phi_rot = angle(evt.v3)
				rotmomentum_Q = rotation(recoil_p_Q, theta_rot, phi_rot)
				rotmomentum_Qbar = rotation(recoil_p_Qbar, theta_rot, phi_rot)
				
				
			#--- we now transform them back to the hydro cell frame
				momentum_Q = lorentz(np.append(E_Q, rotmomentum_Q), -evt.v3)			# final momentum of Q
				momentum_Qbar = lorentz(np.append(E_Qbar, rotmomentum_Qbar), -evt.v3)		# final momentum of Qbar
				position_Q = self.U1Slist.x[i]
			#	position_Qbar = position_Q
			
			#--- add x and p for the QQbar to the temporary list
				add_pQ.append(momentum_Q)
				add_pQbar.append(momentum_Qbar)
				add_xQ.append(position_Q)
			#	add_xQbar.append(position_Qbar)
			
				
		###--------------- end of decay part -----------------------
		
		
		
		
		###--------------- recombination part (new) ----------------
		if self.recombine == True:
			delete_Qbar = []
		
			for i in range(len_Qbar):
				rate_f = []		# to store the formation rate from each pair
				Qbar_x, Qbar_y, Qbar_z = self.Qbarlist.x[i]
				a_low = R_search
				a_upp = self.Lmax - R_search
				lenQ_temp = len(self.Qlist.x)
				
				#--- search pairs first: determine the boundary condition and define KDtree ----
				if a_low <= Qbar_x <= a_upp:
					if a_low <= Qbar_y <= a_upp:
				
						if a_low <= Qbar_z <= a_upp:
							Q_xlist = self.Qlist.x
							Q_plist = self.Qlist.p				
				
						elif Qbar_z < a_low:
							Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[0.0, 0.0, -self.Lmax], axis=0)
							Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)		
				
						else:
							Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[0.0, 0.0, self.Lmax], axis=0)
							Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)
			
				
					elif Qbar_y < a_low:
				
						if a_low <= Qbar_z <= a_upp:
							Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[0.0, -self.Lmax, 0.0], axis=0)
							Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)		
				
						elif Qbar_z < a_low:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, -self.Lmax, -self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
						else:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, -self.Lmax, self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)	
	
				
					else:
				
						if a_low <= Qbar_z <= a_upp:
							Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[0.0, self.Lmax, 0.0], axis=0)
							Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)		
				
						elif Qbar_z < a_low:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, self.Lmax, -self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
						else:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, self.Lmax, self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)	
	
				
				
				elif Qbar_x < a_low:
					if a_low <= Qbar_y <= a_upp:
				
						if a_low <= Qbar_z <= a_upp:
							Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], axis=0)
							Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)				
				
						elif Qbar_z < a_low:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[-self.Lmax, 0.0, -self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
						else:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[-self.Lmax, 0.0, self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)	
			
				
					elif Qbar_y < a_low:
				
						if a_low <= Qbar_z <= a_upp:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[-self.Lmax, -self.Lmax, 0.0]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
						elif Qbar_z < a_low:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[-self.Lmax, -self.Lmax, 0.0],
							self.Qlist.x+[-self.Lmax, 0.0, -self.Lmax], self.Qlist.x+[0.0, -self.Lmax, -self.Lmax], self.Qlist.x+[-self.Lmax, -self.Lmax, -self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
				
						else:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[-self.Lmax, -self.Lmax, 0.0],
							self.Qlist.x+[-self.Lmax, 0.0, self.Lmax], self.Qlist.x+[0.0, -self.Lmax, self.Lmax], self.Qlist.x+[-self.Lmax, -self.Lmax, self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
	
				
					else:
				
						if a_low <= Qbar_z <= a_upp:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[-self.Lmax, self.Lmax, 0.0]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
						elif Qbar_z < a_low:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[-self.Lmax, self.Lmax, 0.0],
							self.Qlist.x+[-self.Lmax, 0.0, -self.Lmax], self.Qlist.x+[0.0, self.Lmax, -self.Lmax], self.Qlist.x+[-self.Lmax, self.Lmax, -self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
				
						else:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[-self.Lmax, self.Lmax, 0.0],
							self.Qlist.x+[-self.Lmax, 0.0, self.Lmax], self.Qlist.x+[0.0, self.Lmax, self.Lmax], self.Qlist.x+[-self.Lmax, self.Lmax, self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)						
			
			
				else:
					if a_low <= Qbar_y <= a_upp:
				
						if a_low <= Qbar_z <= a_upp:
							Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], axis=0)
							Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)				
				
						elif Qbar_z < a_low:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[self.Lmax, 0.0, -self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
						else:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[self.Lmax, 0.0, self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)	
			
				
					elif Qbar_y < a_low:
				
						if a_low <= Qbar_z <= a_upp:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[self.Lmax, -self.Lmax, 0.0]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
						elif Qbar_z < a_low:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[self.Lmax, -self.Lmax, 0.0],
							self.Qlist.x+[self.Lmax, 0.0, -self.Lmax], self.Qlist.x+[0.0, -self.Lmax, -self.Lmax], self.Qlist.x+[self.Lmax, -self.Lmax, -self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
				
						else:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[self.Lmax, -self.Lmax, 0.0],
							self.Qlist.x+[self.Lmax, 0.0, self.Lmax], self.Qlist.x+[0.0, -self.Lmax, self.Lmax], self.Qlist.x+[self.Lmax, -self.Lmax, self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
	
				
					else:
				
						if a_low <= Qbar_z <= a_upp:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[self.Lmax, self.Lmax, 0.0]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
						elif Qbar_z < a_low:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[self.Lmax, self.Lmax, 0.0],
							self.Qlist.x+[self.Lmax, 0.0, -self.Lmax], self.Qlist.x+[0.0, self.Lmax, -self.Lmax], self.Qlist.x+[self.Lmax, self.Lmax, -self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
				
						else:
							Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[self.Lmax, self.Lmax, 0.0],
							self.Qlist.x+[self.Lmax, 0.0, self.Lmax], self.Qlist.x+[0.0, self.Lmax, self.Lmax], self.Qlist.x+[self.Lmax, self.Lmax, self.Lmax]), axis=0)
							Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)						
							
							
				tree_Q = cKDTree(data = Q_xlist)
				list = tree_Q.query_ball_point(self.Qbarlist.x[i], r = R_search)	## the distance r = 1 fm can be changed
				
				
				for each in list:
					evt_f = QQbar_form(Q_xlist[each], Q_plist[each], self.Qbarlist.x[i], self.Qbarlist.p[i], self.T)
					
					## first implementation has xdotp < 0
					if evt_f.rdotp < 0.0:
						rate_f.append(evt_f.form_rate())
					else:
						rate_f.append(0.0)
						# since only half position space contribute due to the xdotp<0
						# normalization needs doubling the rate, which is already included in the form_rate
			
					## second on has no xdotp
					#rate_f.append(evt_f.form_rate())
					
				prob_random = np.random.rand(1)
				prob_f = 8.0/9.0*np.array(rate_f)*dt/C1
				len_list = len(prob_f)
				totalprob_f = np.sum(prob_f)

				if prob_random <= totalprob_f:
					delete_Qbar.append(i)
					a = 0.0
					for j in range(len_list):
						if a <= prob_random <= a+prob_f[j]:
							k = j
							break
						a += prob_f[j]
					
					evt_f = QQbar_form(Q_xlist[list[k]], Q_plist[list[k]], self.Qbarlist.x[i], self.Qbarlist.p[i], self.T)
					q_U1S, costhetaU, phiU = evt_f.sample_final() 	## sample U1S momentum
					sinthetaU = np.array(1.0-costhetaU**2)
					# get the 3-component of U1S momentum, where v = z axis
					tempmomentum_U = np.array([q_U1S*sinthetaU*np.cos(phiU), q_U1S*sinthetaU*np.sin(phiU), q_U1S*costhetaU])
					E_U1S = np.sqrt( np.sum(tempmomentum_U**2)+M_1S**2 )
					
					# need to rotate the vector, v is not the z axis in medium frame
					theta_rot, phi_rot = angle(evt_f.v3)
					rotmomentum_U = rotation(tempmomentum_U, theta_rot, phi_rot)

					# lorentz back to the medium frame
					momentum_U1S = lorentz( np.append(E_U1S, rotmomentum_U), -evt_f.v3 )
					position_U1S = evt_f.R%self.Lmax
					
					# update the lists
					self.Qlist.x = np.delete(self.Qlist.x, list[k]%lenQ_temp, axis=0)
					self.Qlist.p = np.delete(self.Qlist.p, list[k]%lenQ_temp, axis=0)
					## DO NOT change the Qbarlist here, STILL inside the LOOP of Qbarlist !!!
					
					if len_U1S == 0:
						self.U1Slist.x = np.array([position_U1S])
						self.U1Slist.p = np.array([momentum_U1S])
					else:
						self.U1Slist.x = np.append(self.U1Slist.x, [position_U1S], axis=0)
						self.U1Slist.p = np.append(self.U1Slist.p, [momentum_U1S], axis=0)
					
							
			# ---- now update the Qbarlist from delete_Qbar ----
			self.Qbarlist.x = np.delete(self.Qbarlist.x, delete_Qbar, axis=0)
			self.Qbarlist.p = np.delete(self.Qbarlist.p, delete_Qbar, axis=0)
			
			
		###--------------- end of recombination (new) ---------------------			

		
		
		
		
		### -------------- Update all three lists from decay ------------------
		add_pQ = np.array(add_pQ)
		add_pQbar = np.array(add_pQbar)
		add_xQ = np.array(add_xQ)
		if len(add_pQ):	
		### if there is at least quarkonium decays, we need to update all the three lists
			self.U1Slist.x = np.delete(self.U1Slist.x, delete_U1S, axis=0) # delete along the axis = 0
			self.U1Slist.p = np.delete(self.U1Slist.p, delete_U1S, axis=0)
			
			if len(self.Qbarlist.x) == 0:
				self.Qlist.x = np.array(add_xQ)
				self.Qlist.p = np.array(add_pQ)
				self.Qbarlist.x = np.array(add_xQ)
				self.Qbarlist.p = np.array(add_pQbar)
			else:
				self.Qlist.x = np.append(self.Qlist.x, add_xQ, axis=0)
				self.Qlist.p = np.append(self.Qlist.p, add_pQ, axis=0)
				self.Qbarlist.x = np.append(self.Qbarlist.x, add_xQ, axis=0)
				self.Qbarlist.p = np.append(self.Qbarlist.p, add_pQbar, axis=0)
			
		### --------------- end of the list updates of decay --------------------
##### ------------ end of evolution function ---------------#####	
			
			
			

##### -------- a test evolution function for recombination --------#####			
	def testrun(self, dt = 0.04):		#--- time step is universal since we include recombination
		
		### --------- free stream Q, Qbar, U1S first ------------
		len_U1S = len(self.U1Slist.x)
		len_Qbar = len(self.Qbarlist.x)		#notice len_Qbar = len_Q
		'''
		if len_U1S != 0:
			v_U1S = np.array( [self.U1Slist.p[i]/np.sqrt(np.sum(self.U1Slist.p[i]**2)+M_1S**2) for i in range(len_U1S)] )
			self.U1Slist.x = (self.U1Slist.x + dt*v_U1S )%self.Lmax
			
		if len_Qbar != 0:
			v_Q = np.array( [self.Qlist.p[i]/np.sqrt(np.sum(self.Qlist.p[i]**2)+M**2) for i in range(len_Qbar)] )
			v_Qbar = np.array( [self.Qbarlist.p[i]/np.sqrt(np.sum(self.Qbarlist.p[i]**2)+M**2) for i in range(len_Qbar)] )
			
			self.Qlist.x = (self.Qlist.x + dt*v_Q )%self.Lmax
			self.Qbarlist.x = (self.Qbarlist.x + dt*v_Qbar )%self.Lmax
		'''	
		### ----------- end of free stream ---------------------
		
		
		total_rate_f = 0.0
		count = 0.0
		
		### -------- next return the averaged recombination rate -------
		for i in range(len_Qbar):
			rate_f = []		# to store the formation rate from each pair
			Qbar_x, Qbar_y, Qbar_z = self.Qbarlist.x[i]
			a_low = R_search
			a_upp = self.Lmax - R_search
			
		#--- search pairs first: determine the boundary condition and define KDtree ----
			if a_low <= Qbar_x <= a_upp:
				if a_low <= Qbar_y <= a_upp:
				
					if a_low <= Qbar_z <= a_upp:
						Q_xlist = self.Qlist.x
						Q_plist = self.Qlist.p				
				
					elif Qbar_z < a_low:
						Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[0.0, 0.0, -self.Lmax], axis=0)
						Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)		
				
					else:
						Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[0.0, 0.0, self.Lmax], axis=0)
						Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)
			
				
				elif Qbar_y < a_low:
				
					if a_low <= Qbar_z <= a_upp:
						Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[0.0, -self.Lmax, 0.0], axis=0)
						Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)		
				
					elif Qbar_z < a_low:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, -self.Lmax, -self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
					else:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, -self.Lmax, self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)	
	
				
				else:
				
					if a_low <= Qbar_z <= a_upp:
						Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[0.0, self.Lmax, 0.0], axis=0)
						Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)		
				
					elif Qbar_z < a_low:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, self.Lmax, -self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
					else:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, self.Lmax, self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)	
	
				
				
			elif Qbar_x < a_low:
				if a_low <= Qbar_y <= a_upp:
				
					if a_low <= Qbar_z <= a_upp:
						Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], axis=0)
						Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)				
				
					elif Qbar_z < a_low:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[-self.Lmax, 0.0, -self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
					else:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[-self.Lmax, 0.0, self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)	
			
				
				elif Qbar_y < a_low:
				
					if a_low <= Qbar_z <= a_upp:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[-self.Lmax, -self.Lmax, 0.0]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
					elif Qbar_z < a_low:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[-self.Lmax, -self.Lmax, 0.0],
						self.Qlist.x+[-self.Lmax, 0.0, -self.Lmax], self.Qlist.x+[0.0, -self.Lmax, -self.Lmax], self.Qlist.x+[-self.Lmax, -self.Lmax, -self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
				
					else:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[-self.Lmax, -self.Lmax, 0.0],
						self.Qlist.x+[-self.Lmax, 0.0, self.Lmax], self.Qlist.x+[0.0, -self.Lmax, self.Lmax], self.Qlist.x+[-self.Lmax, -self.Lmax, self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
	
				
				else:
				
					if a_low <= Qbar_z <= a_upp:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[-self.Lmax, self.Lmax, 0.0]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
					elif Qbar_z < a_low:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[-self.Lmax, self.Lmax, 0.0],
						self.Qlist.x+[-self.Lmax, 0.0, -self.Lmax], self.Qlist.x+[0.0, self.Lmax, -self.Lmax], self.Qlist.x+[-self.Lmax, self.Lmax, -self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
				
					else:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[-self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[-self.Lmax, self.Lmax, 0.0],
						self.Qlist.x+[-self.Lmax, 0.0, self.Lmax], self.Qlist.x+[0.0, self.Lmax, self.Lmax], self.Qlist.x+[-self.Lmax, self.Lmax, self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)						
			
			
			else:
				if a_low <= Qbar_y <= a_upp:
				
					if a_low <= Qbar_z <= a_upp:
						Q_xlist = np.append(self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], axis=0)
						Q_plist = np.append(self.Qlist.p, self.Qlist.p, axis=0)				
				
					elif Qbar_z < a_low:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[self.Lmax, 0.0, -self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
					else:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[self.Lmax, 0.0, self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)	
			
				
				elif Qbar_y < a_low:
				
					if a_low <= Qbar_z <= a_upp:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[self.Lmax, -self.Lmax, 0.0]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
					elif Qbar_z < a_low:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[self.Lmax, -self.Lmax, 0.0],
						self.Qlist.x+[self.Lmax, 0.0, -self.Lmax], self.Qlist.x+[0.0, -self.Lmax, -self.Lmax], self.Qlist.x+[self.Lmax, -self.Lmax, -self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
				
					else:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, -self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[self.Lmax, -self.Lmax, 0.0],
						self.Qlist.x+[self.Lmax, 0.0, self.Lmax], self.Qlist.x+[0.0, -self.Lmax, self.Lmax], self.Qlist.x+[self.Lmax, -self.Lmax, self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
	
				
				else:
				
					if a_low <= Qbar_z <= a_upp:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[self.Lmax, self.Lmax, 0.0]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)		
				
					elif Qbar_z < a_low:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, -self.Lmax], self.Qlist.x+[self.Lmax, self.Lmax, 0.0],
						self.Qlist.x+[self.Lmax, 0.0, -self.Lmax], self.Qlist.x+[0.0, self.Lmax, -self.Lmax], self.Qlist.x+[self.Lmax, self.Lmax, -self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)				
				
					else:
						Q_xlist = np.concatenate((self.Qlist.x, self.Qlist.x+[self.Lmax, 0.0, 0.0], self.Qlist.x+[0.0, self.Lmax, 0.0], self.Qlist.x+[0.0, 0.0, self.Lmax], self.Qlist.x+[self.Lmax, self.Lmax, 0.0],
						self.Qlist.x+[self.Lmax, 0.0, self.Lmax], self.Qlist.x+[0.0, self.Lmax, self.Lmax], self.Qlist.x+[self.Lmax, self.Lmax, self.Lmax]), axis=0)
						Q_plist = np.concatenate((self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p, self.Qlist.p), axis=0)						
			
			
			
			tree_Q = cKDTree(data = Q_xlist)
			list = tree_Q.query_ball_point(self.Qbarlist.x[i], r = R_search)	## the distance r = 1 fm can be changed
			
			for each in list:
				evt_f = QQbar_form(Q_xlist[each], Q_plist[each], self.Qbarlist.x[i], self.Qbarlist.p[i], self.T)
				
				## first implementation has xdotp < 0
				if evt_f.rdotp < 0.0:
					rate_f.append(evt_f.form_rate())
					# since only half position space contribute due to the xdotp<0
					# normalization needs changing, factor of 2 already added in the recombination rate
				
				## second one has no xdotp
				#rate_f.append(evt_f.form_rate())
				
			len_ratef = len(rate_f)
			if len_ratef != 0:
				rate_f = np.array(rate_f)
				total_rate_f += np.sum(rate_f)


		return total_rate_f/len_Qbar
		
##### -------- end of test evolution function for recombination --------#####