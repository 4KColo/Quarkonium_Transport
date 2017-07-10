# cython: c_string_type=str, c_string_encoding=ascii
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport fmin
import cython
import os

#------------------Import C++ fucntions and class for Xsection and rates------------------
cdef extern from "../src/matrix_elements.h":
	cdef double dX_Qq2Qq_dPS(double * PS, size_t n_dims, void * params)  
	cdef double dX_Qg2Qg_dPS(double * PS, size_t n_dims, void * params) 

	cdef double M2_Qq2Qqg(double * x_, size_t n_dims_, void * params_)  
	cdef double M2_Qg2Qgg(double * x_, size_t n_dims_, void * params_)  

	cdef double Ker_Qqg2Qq(double * x_, size_t n_dims_, void * params_)   
	cdef double Ker_Qgg2Qg(double * x_, size_t n_dims_, void * params_) 

cdef extern from "../src/Xsection.h":
	cdef cppclass Xsection_2to2 :
		Xsection_2to2(double (*dXdPS_)(double *, size_t, void *), double M1_, string name_, bool refresh)  
		void sample_dXdPS(double * arg, vector[ vector[double] ] & FS)  
	
	cdef cppclass Xsection_2to3 :
		Xsection_2to3(double (*dXdPS_)(double *, size_t, void *), double M1_, string name_, bool refresh)  
		void sample_dXdPS(double * arg, vector[ vector[double] ] & FS)  

	cdef cppclass f_3to2 :
		f_3to2(double (*dXdPS_)(double *, size_t, void *), double M1_, string name_, bool refresh)  
		void sample_dXdPS(double * arg, vector[ vector[double] ] & FS)  

cdef extern from "../src/rates.h":
	cdef cppclass rates_2to2 :
		rates_2to2(Xsection_2to2 * Xprocess_, int degeneracy_, double eta_2_, string name_, bool refresh)  
		double interpR(double * arg)  
		void sample_initial(double * arg, vector[ vector[double] ] & IS)  

	cdef cppclass rates_2to3 :
		rates_2to3(Xsection_2to3 * Xprocess_, int degeneracy_, double eta_2_, string name_, bool refresh)  
		double interpR(double * arg)  
		void sample_initial(double * arg, vector[ vector[double] ] & IS)  

	cdef cppclass rates_3to2 :
		rates_3to2(f_3to2 * Xprocess_, int degeneracy_, double eta_2_, double eta_k_, string name_, bool refresh) 
		double interpR(double * arg)  
		void sample_initial(double * arg, vector[ vector[double] ] & IS)  


#-------------Heavy quark evolution class------------------------
cdef double dtmin = 0.1

cdef class HqEvo(object):
	cdef bool elastic, inelastic, detailed_balance #2->2, 2->3, 3->2
	cdef Xsection_2to2 * x_Qq_Qq
	cdef Xsection_2to2 * x_Qg_Qg
	cdef Xsection_2to3 * x_Qq_Qqg
	cdef Xsection_2to3 * x_Qg_Qgg
	cdef f_3to2 * x_Qqg_Qq
	cdef f_3to2 * x_Qgg_Qg
	cdef rates_2to2 * r_Qq_Qq
	cdef rates_2to2 * r_Qg_Qg
	cdef rates_2to3 * r_Qq_Qqg
	cdef rates_2to3 * r_Qg_Qgg
	cdef rates_3to2 * r_Qqg_Qq
	cdef rates_3to2 * r_Qgg_Qg
	cdef size_t Nchannels, Nf
	cdef double mass, Kfactor
	cdef public vector[vector[double]] IS, FS
	
	def __cinit__(self, options, table_folder='./tables', refresh_table=False):
		self.elastic=options['2->2']
		self.inelastic=options['2->3']
		self.detailed_balance=options['3->2']
		self.Nchannels = 0
		self.mass = options['mass']
		self.Nf = options['Nf']
		self.Kfactor = options['Kfactor']
		
		if not os.path.exists(table_folder):
			os.makedirs(table_folder)

		if self.elastic:
			self.x_Qq_Qq = new Xsection_2to2(&dX_Qq2Qq_dPS, self.mass, "%s/XQq2Qq.hdf5"%table_folder, refresh_table)
			self.x_Qg_Qg = new Xsection_2to2(&dX_Qg2Qg_dPS, self.mass, "%s/XQg2Qg.hdf5"%table_folder, refresh_table)
			self.r_Qq_Qq = new rates_2to2(self.x_Qq_Qq, 12*self.Nf, 0., "%s/RQq2Qq.hdf5"%table_folder, refresh_table)
			self.r_Qg_Qg = new rates_2to2(self.x_Qg_Qg, 16, 0., "%s/RQg2Qg.hdf5"%table_folder, refresh_table)
			self.Nchannels += 2
			
		if self.inelastic:
			self.x_Qq_Qqg = new Xsection_2to3(&M2_Qq2Qqg, self.mass, "%s/XQq2Qqg.hdf5"%table_folder, refresh_table)
			self.x_Qg_Qgg = new Xsection_2to3(&M2_Qg2Qgg, self.mass, "%s/XQg2Qgg.hdf5"%table_folder, refresh_table)
			self.r_Qq_Qqg = new rates_2to3(self.x_Qq_Qqg, 12*self.Nf, 0., "%s/RQq2Qqg.hdf5"%table_folder, refresh_table)
			self.r_Qg_Qgg = new rates_2to3(self.x_Qg_Qgg, 16/2, 0., "%s/RQg2Qgg.hdf5"%table_folder, refresh_table)
			self.Nchannels += 2

		if self.detailed_balance:
			self.x_Qqg_Qq = new f_3to2(&Ker_Qqg2Qq, self.mass, "%s/XQqg2Qq.hdf5"%table_folder, refresh_table)
			self.x_Qgg_Qg = new f_3to2(&Ker_Qgg2Qg, self.mass, "%s/XQgg2Qg.hdf5"%table_folder, refresh_table)
			self.r_Qqg_Qq = new rates_3to2(self.x_Qqg_Qq, 12*self.Nf*16, 0., 0., "%s/RQqg2Qq.hdf5"%table_folder, refresh_table)
			self.r_Qgg_Qg = new rates_3to2(self.x_Qgg_Qg, 16*16/2, 0., 0., "%s/RQgg2Qg.hdf5"%table_folder, refresh_table)
			self.Nchannels += 2
			
		print "Number of Channels", self.Nchannels

	cpdef (double, double) sample_channel(self, double E1, double T, double dt23, double dt32):	
		cdef double r, psum = 0.0, dt, Pmax = 0.1, R1, R2, Relastic
		cdef int i=0
		cdef int channel_index = -1
		cdef double p[6]
		cdef double * arg = <double*>malloc(3*sizeof(double))
		arg[0] = E1; arg[1] = T;
		# channel < 0 : freestream
		# 0:	Qq->Qq
		# 1: 	Qg->Qg
		# 2:	Qq->Qqg
		# 3: 	Qg->Qgg
		# 4:	Qqg->Qq
		# 5: 	Qgg->Qg
		if self.elastic:
			arg[2] = 0.
			psum += self.r_Qq_Qq.interpR(arg)
			p[i] = psum; i += 1
			psum += self.r_Qg_Qg.interpR(arg)
			p[i] = psum; i += 1
			Relastic = psum
		if self.inelastic:
			arg[2] = max(dt23, dtmin)
			R1 = self.r_Qq_Qqg.interpR(arg)
			R2 = self.r_Qg_Qgg.interpR(arg)
			psum += R1
			p[i] = psum; i += 1
			psum += R2
			p[i] = psum; i += 1
		if self.detailed_balance:
			arg[2] = max(dt32, dtmin)
			R1 = self.r_Qqg_Qq.interpR(arg)
			R2 = self.r_Qgg_Qg.interpR(arg)
			psum += R1
			p[i] = psum; i += 1
			psum += R2 
			p[i] = psum; i += 1
		free(arg)
		dt = Pmax/psum
		r = (<double>rand())/dt/RAND_MAX
		if r >= p[self.Nchannels-1]:
			return -1, dt
		for i in range(self.Nchannels):
			if r < p[i]:
				channel_index = i
				break
		return channel_index, dt/self.Kfactor

	cpdef sample_initial(self, int channel, double E1, double T, double dt23, double dt32):
		cdef double * arg = <double*>malloc(3*sizeof(double))
		arg[0] = E1; arg[1] = T; arg[2] = 0.;
		if channel == 0:
			self.r_Qq_Qq.sample_initial(arg, self.IS)
		elif channel == 1:
			self.r_Qg_Qg.sample_initial(arg,  self.IS)
		elif channel == 2:
			arg[2] = max(dt23, dtmin)
			self.r_Qq_Qqg.sample_initial(arg,  self.IS)
		elif channel == 3:
			arg[2] = max(dt23, dtmin)
			self.r_Qg_Qgg.sample_initial(arg,  self.IS)
		elif channel == 4:
			arg[2] = max(dt32, dtmin)
			self.r_Qqg_Qq.sample_initial(arg,  self.IS)
		elif channel == 5:
			arg[2] = max(dt32, dtmin)
			self.r_Qgg_Qg.sample_initial(arg,  self.IS)
		else:
			pass
		free(arg)

	cpdef sample_final(self, int channel, double s, double T, double dt23, double a1, double a2):
		cdef double * arg = <double*>malloc(4*sizeof(double))
		arg[0] = s; arg[1] = T; arg[2] = 0.; arg[3] = 0.
		if channel == 0:
			self.x_Qq_Qq.sample_dXdPS(arg,  self.FS)
		elif channel == 1:
			self.x_Qg_Qg.sample_dXdPS(arg,  self.FS)
		elif channel == 2:
			arg[2] = max(dt23, dtmin)
			self.x_Qq_Qqg.sample_dXdPS(arg,  self.FS)
		elif channel == 3:
			arg[2] = max(dt23, dtmin)
			self.x_Qg_Qgg.sample_dXdPS(arg,  self.FS)
		elif channel == 4:
			arg[2] = a1; arg[3] = a2
			self.x_Qqg_Qq.sample_dXdPS(arg,  self.FS)
		elif channel == 5:
			arg[2] = a1; arg[3] = a2
			self.x_Qgg_Qg.sample_dXdPS(arg,  self.FS)
		else:
			pass
		free(arg)

		
			

							
		










