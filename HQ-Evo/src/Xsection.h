#ifndef XSECTION_H
#define XSECTION_H

#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <boost/multi_array.hpp>
#include "sample_methods.h"


/* all the differential Xsection function are declared by type "double f(double * arg, size_t n_dims, void * params)"
	This type is natural for gsl vegas integration, but does not match the type for gsl 1d integration which requires double f(double x, void * params)
	Therefore, for 1d integration, use the struct below. It stores both the function call and the numerical parameters:
	
	double f_1d(double x, void * p_) {
		Mygsl_integration_params * p = static_cast<Mygsl_integration_params *> p;
		return p->f(x, 1, p->params)
	}
*/
struct Mygsl_integration_params{
	double (*f)(double * arg, size_t n_dims, void * params);
	double * params;
};

//=============Xsection base class===================================================
// This is the base class for 2->2 and 2->3 cross-sections.
// It takes care of the tabulating details and the tabulating routines, also the interpolation process
// The actually total Xsection calcuate function and final state sample function are virtual functions, because 2->2 and 2->3 uses quite different techniques to do these jobs.
class Xsection{
protected:
	virtual void tabulate(size_t T_start, size_t dnT) = 0;
	virtual void save_to_file(std::string filename, std::string datasetname) = 0;
	virtual void read_from_file(std::string filename, std::string datasetname) = 0;
	double (*dXdPS)(double * PS, size_t n_dims, void * params);
	double M1;
public:
	Xsection(double (*dXdPS_)(double *, size_t, void *), double M1_, std::string name_, bool refresh);
	double get_M1(void) {return M1;};
	// arg = [s, T] fot X22, arg = [s, T, dt] for X23, arg = [s, T, s1k, s2k] for f32
	virtual double interpX(double * arg) = 0; 
	virtual double calculate(double * arg) = 0;
	virtual void sample_dXdPS(double * arg, std::vector< std::vector<double> > & FS) = 0;
};

//============Derived 2->2 Xsection class============================================
class Xsection_2to2 : public Xsection{
private:
	rejection_1d sampler1d;
	std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist_phi3;
	void tabulate(size_t T_start, size_t dnT);
	void save_to_file(std::string filename, std::string datasetname);
	void read_from_file(std::string filename, std::string datasetname);
	size_t Nsqrts, NT;
	double sqrtsL, sqrtsH, dsqrts,
		   TL, TH, dT;
	boost::multi_array<double, 2> Xtab;
public:
    Xsection_2to2(double (*dXdPS_)(double *, size_t, void *), double M1_, std::string name_, bool refresh);
	double interpX(double * arg);
    double calculate(double * arg);
	void sample_dXdPS(double * arg, std::vector< std::vector<double> > & FS);
};

//============Derived 2->3 Xsection class============================================
class Xsection_2to3 : public Xsection{
private:
	AiMS sampler;
	std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist_phi4;
	void tabulate(size_t T_start, size_t dnT);
	void save_to_file(std::string filename, std::string datasetname);
	void read_from_file(std::string filename, std::string datasetname);
	size_t Nsqrts, NT, Ndt;
	double sqrtsL, sqrtsH, dsqrts,
				 TL, TH, dT,
				 dtL, dtH, ddt;
	boost::multi_array<double, 3> Xtab;
public:
    Xsection_2to3(double (*dXdPS_)(double *, size_t, void *), double M1_, std::string name_, bool refresh);
	double interpX(double * arg);
    double calculate(double * arg);
	void sample_dXdPS(double * arg, std::vector< std::vector<double> > & FS);
};

//============Derived 3->2 Xsection class============================================
class f_3to2 : public Xsection{
private:
	//rejection_1d sampler1d;
	AiMS sampler;
	std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dist_phi4;
	void tabulate(size_t T_start, size_t dnT);
	void save_to_file(std::string filename, std::string datasetname);
	void read_from_file(std::string filename, std::string datasetname);
	size_t Nsqrts, NT, Na1, Na2;
	double sqrtsL, sqrtsH, dsqrts,
				 TL, TH, dT,
				 a1L, a1H, da1,
				 a2L, a2H, da2;
	boost::multi_array<double, 4> Xtab;

public:
    f_3to2(double (*dXdPS_)(double *, size_t, void *), double M1_, std::string name_, bool refresh);
	double interpX(double * arg);
    double calculate(double * arg);
	void sample_dXdPS(double * arg, std::vector< std::vector<double> > & FS);
};


#endif
