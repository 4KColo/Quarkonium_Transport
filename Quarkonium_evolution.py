#!/usr/bin/env python
import numpy as np
import h5py
from scipy.spatial import cKDTree
from particle import Particlelist
from Quarkonium_decay import QQbar_decay
from Quarkonium_recombine import QQbar_form

#### --------- some constants -----------------------------
alpha_s = 0.3 # for bottomonium
N_C = 3.0
M = 4650.0 # MeV b-quark
rho_c = 1.0/(N_C**2-1.0)
C_F = 4.0/3.0
a_B = 2.0/(alpha_s*C_F*M)
E_1S = alpha_s*C_F/(2.0*a_B)  # Upsilon(1S), here is magnitude, true value is its negative
M_1S = M*2.0 - E_1S  		  # mass of Upsilon(1S)
C1 = 197.327				  # 197 MeV*fm = 1



####----- define a function for Lorentz transformation ---------
def lorentz(p4, v):
#----- first we add some error checking
	if len(p4) != 4:
		raise ValueError('first argument of lorentz transformation must be a 4-vector')
	if len(v) !=3:
		raise ValueError('second argument of lorentz transformation must be a 3-vector')
	v_sq = v[0]**2+v[1]**2+v[2]**2
	gamma = 1.0/np.sqrt(1.0-v_sq)
	vdotp = v[0]*p4[1]+v[1]*p4[2]+v[2]*p4[3]
	E = gamma*( p4[0] - vdotp )
	px = -gamma*v[0]*p4[0] + p4[1] +(gamma-1.0)*v[0]*vdotp/v_sq
	py = -gamma*v[1]*p4[0] + p4[2] +(gamma-1.0)*v[1]*vdotp/v_sq
	pz = -gamma*v[2]*p4[0] + p4[3] +(gamma-1.0)*v[2]*vdotp/v_sq
	return np.array([px,py,pz])
	#return [E,px,py,pz]

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


####------ end of rotation matrix ------------------------



####------ define a function that gives the angle theta and phi ------
def angle(vec):
	if len(vec) !=3:
		raise ValueError('first argument of angle must be a 3-vector')
	len_vec = np.sqrt(np.sum(np.array(vec)**2))
	theta = np.arccos(vec[2]/len_vec)
	if vec[0] > 0:
		phi = np.arctan(vec[1]/vec[0])
	elif vec[0] == 0:
		phi = np.pi/2.0*(vec[1]>0)
	else:
		phi = np.arctan(vec[1]/vec[0]) + np.pi
	return [theta, phi]
	

####------ end of angle function --------------------





class QQbar_evol:
#---- input the medium_type when calling the class ----
	def __init__(self, medium_type, temperature = 300.0):
		self.type = medium_type		# for static type, the input temperature will be used
		self.T = temperature


#---- initialize the Q, Qbar and Quarkonium -- currently we only study Upsilon(1S)
	def initialize(self, N_Q = 100, N_U1S = 10, Lmax = 10.0, Pmax = 15000.0):
		self.Lmax = 10.0
#---- first determine medium type and then initialize
		if self.type == 'static':
	#---- static medium!! parameters only make sense in STATIC mode------
	#------ Lmax = 10 fm, 3-D box size, Pmax = 15 GeV ------
			self.Qlist = Particlelist()
			self.Qbarlist = Particlelist()
			self.U1Slist = Particlelist()
			
			self.Qlist.id = 5			# b quark
			self.Qbarlist.id = -5		# anti b quark
			self.U1Slist.id = 533		# Upsilon(1S)
	
			#--- sample their initial positions x and momentum p
			self.Qlist.x = np.array([np.random.rand(3)*Lmax for i in range(N_Q)])
			self.Qbarlist.x = np.array([np.random.rand(3)*Lmax for i in range(N_Q)])
			self.U1Slist.x = np.array([np.random.rand(3)*Lmax for i in range(N_U1S)])
			
			self.Qlist.p = np.array([(np.random.rand(3)-0.5)*2*Pmax for i in range(N_Q)])
			self.Qbarlist.p = np.array([(np.random.rand(3)-0.5)*2*Pmax for i in range(N_Q)])
			self.U1Slist.p = np.array([(np.random.rand(3)-0.5)*2*Pmax for i in range(N_U1S)])

#####----- evolution function --------
	def run(self, dt = 0.04):		#--- time step is universal since we include recombination
	
		### --------- free stream Q, Qbar, U1S first ------------
		self.Qlist.x = (self.Qlist.x + dt*self.Qlist.p/np.sqrt(self.Qlist.p**2+M**2) )%self.Lmax
		self.Qbarlist.x = (self.Qbarlist.x + dt*self.Qbarlist.p/np.sqrt(self.Qbarlist.p**2+M**2) )%self.Lmax
		self.U1Slist.x = (self.U1Slist.x + dt*self.U1Slist.p/np.sqrt(self.U1Slist.p**2+M**2) )%self.Lmax
		### ----------- end of free stream ---------------------
		
		
		### --------then consider the decay of U1S-------------
		### -------- start here -------------------------------
		
		len_U1S = len(self.U1Slist.p)
		delete_U1S = []
		add_pQ = []
		add_pQbar = []
		add_xQ = []
		#add_xQbar = [] the positions of Q and Qbar are the same
		for i in range(len_U1S):
			evt = QQbar_decay(com_momentum = self.U1Slist.p[i], temperature = self.T) #call the decay class
			
			# if the decay happens
			if evt.decay_rate()*dt/C1 > np.random.rand(1):
				delete_U1S.append(i) # store the indexes that should be deleted later

			#else: nothing happens	
			
				# ----- sample the initial gluon and final QQbar -------
				q, costheta1, phi1 = evt.sample_init()  # gluon energy, angles in the quarkonium rest frame
				p_rel, costheta2, phi2 = evt.sample_final(q)  # QQbar in the same frame
				sintheta1 = np.sqrt(1.0-costheta1**2)
				sintheta2 = np.sqrt(1.0-costheta2**2)
			#--- all the following three momentum components are in the quarkonium rest frame
				tempmomentum_g = np.array([q*sintheta1*np.cos(phi1), q*sintheta1*np.sin(phi1), q*costheta1])
				tempmomentum_Q = np.array([0.5*p_rel*sintheta2*np.cos(phi2), 0.5*p_rel*sintheta2*np.sin(phi2), 0.5*p_rel*costheta2])
			#	tempmomentum_Qbar = -tempmomentum_Q (true in the quarkonium rest frame)
				
				#---- energy of Q and Qbar
				E_Q = np.sqrt(  np.sum((0.5*tempmomentum_g + tempmomentum_Q)**2)+M_1S**2  )
				E_Qbar = np.sqrt(  np.sum((0.5*tempmomentum_g - tempmomentum_Q)**2)+M_1S**2  )
				
			#--- the tempmomentum needs to be rotated from the v = z axis to the medium frame
				#---- first get the rotation matrix angles
				theta_rot, phi_rot = angle(evt.v3)
				rotmomentum_g = rotation(tempmomentum_g, theta_rot, phi_rot)
				rotmomentum_Q = rotation(tempmomentum_Q, theta_rot, phi_rot)
				
				
			#--- we now transform them back to the hydro cell frame
				momentum_Q = lorentz(np.append(E_Q, 0.5*rotmomentum_g + rotmomentum_Q), -evt.v3)			# final momentum of Q
				momentum_Qbar = lorentz(np.append(E_Qbar, 0.5*rotmomentum_g - rotmomentum_Q), -evt.v3)		# final momentum of Qbar
				position_Q = self.U1Slist.x[i]
			#	position_Qbar = position_Q
			
			#--- add x and p for the QQbar to the temporary list
				add_pQ.append(momentum_Q)
				add_pQbar.append(momentum_Qbar)
				add_xQ.append(position_Q)
			#	add_xQbar.append(position_Qbar)
			
				
		###--------------- end of decay part -----------------------
		
		
		
		
		
		###--------------- recombination part ----------------------
		len_Qbar = len(self.Qbarlist.x)
		delete_Qbar = []

		for i in range(len_Qbar):
		#--- search pairs first ----
			tree_Q = cKDTree(data = self.Qlist.x)
			list = tree_Q.query_ball_point(self.Qbarlist.x[i], r=1.0)	## the distance r = 1 fm can be changed
			
			for each in list:
				evt_f = QQbar_form(self.Qlist.x[each], self.Qlist.p[each], self.Qbarlist.x[i], self.Qbarlist.p[i], self.T)
				
				if 8/9.0*min(evt_f.form_rate()*dt/C1, 1.0) > np.random.rand(1):
					delete_Qbar.append(i)
					#print 'bang!'
					q_U1S, costhetaU, phiU = evt_f.sample_final() 	## sample U1S momentum
					sinthetaU = np.array(1.0-costhetaU**2)
					# get the 3-component of U1S momentum, where v = z axis
					tempmomentum_U = np.array([q_U1S*sinthetaU*np.cos(phiU), q_U1S*sinthetaU*np.sin(phiU), q_U1S*costhetaU])
					E_U1S = np.sqrt( np.sum(tempmomentum_U**2)+M_1S**2 )
					
					# need to rotate the vector, v is not the z axis in medium frame
					theta_rot, phi_rot = angle(evt_f.v3)
					rotmomentum_U = rotation(tempmomentum_U, theta_rot, phi_rot)

					#lorentz back to the medium frame
					momentum_U1S = lorentz( np.append(E_U1S, rotmomentum_U), -evt_f.v3 )
					position_U1S = evt_f.R
					
					# update the lists
					self.Qlist.x = np.delete(self.Qlist.x, each, axis=0)
					self.Qlist.p = np.delete(self.Qlist.p, each, axis=0)
					## DO NOT change the Qbarlist here, STILL inside the LOOP of Qbarlist !!!
					self.U1Slist.x = np.append(self.U1Slist.x, [position_U1S], axis=0)
					self.U1Slist.p = np.append(self.U1Slist.p, [momentum_U1S], axis=0)
					
					break
					
		# ----now update the Qbarlist from delete_Qbar ----
		self.Qbarlist.x = np.delete(self.Qbarlist.x, delete_Qbar, axis=0)
		self.Qbarlist.p = np.delete(self.Qbarlist.p, delete_Qbar, axis=0)
		###--------------- end of recombination ---------------------
		
		
		
		
		
		### -------------- Update all three lists from decay ------------------
		add_pQ = np.array(add_pQ)
		add_pQbar = np.array(add_pQbar)
		add_xQ = np.array(add_xQ)
		if len(add_pQ):	
		### if there is at least quarkonium decays, we need to update all the three lists
			self.U1Slist.x = np.delete(self.U1Slist.x, delete_U1S, axis=0) # delete along the axis = 0
			self.U1Slist.p = np.delete(self.U1Slist.p, delete_U1S, axis=0)
			self.Qlist.x = np.append(self.Qlist.x, add_xQ, axis=0)
			self.Qlist.p = np.append(self.Qlist.p, add_pQ, axis=0)
			self.Qbarlist.x = np.append(self.Qbarlist.x, add_xQ, axis=0)
			self.Qbarlist.p = np.append(self.Qbarlist.p, add_pQbar, axis=0)
			
		### --------------- end of the list updates of decay --------------------
			
			
			
			
			



