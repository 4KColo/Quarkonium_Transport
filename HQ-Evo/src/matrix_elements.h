#ifndef MATRIX_ELEMENTS_H
#define MATRIX_ELEMENTS_H
#include <cstdlib>

//======running coupling=======================================================
double alpha_s(double Q2);

class self_consistent_mD2{
private:
	const double TL, TH, dT;
	const size_t NT;
	double * mD2;
public:	
	self_consistent_mD2(const double TL, const double TH, const size_t NT);
	~self_consistent_mD2(){delete[] mD2;};
	double get_mD2(double T);
};

//=============Baisc function for Q+q --> Q+q==================================
double M2_Qq2Qq(double t, void * params);
double dX_Qq2Qq_dPS(double * PS, size_t n_dims, void * params);
double dqhat_Qq2Qq_dPS(double* PS, size_t ndims, void* params);

//=============Baisc function for Q+g --> Q+g==================================
double M2_Qg2Qg(double t, void * params);
double dX_Qg2Qg_dPS(double * PS, size_t n_dims, void * params);
double dqhat_Qg2Qg_dPS(double* PS, size_t ndims, void* params);

//=============Baisc function for Q+q --> Q+q+g==================================
double M2_Qq2Qqg(double * x_, size_t n_dims_, void * params_);
//=============Baisc function for Q+g --> Q+g+g==================================
double M2_Qg2Qgg(double * x_, size_t n_dims_, void * params_);

//=============Baisc function for Q+q+g --> Q+q==================================
double Ker_Qqg2Qq(double * x_, size_t n_dims_, void * params_);
//=============Baisc function for Q+g+g --> Q+g==================================
double Ker_Qgg2Qg(double * x_, size_t n_dims_, void * params_);


#endif
