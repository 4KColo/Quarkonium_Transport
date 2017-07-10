#ifndef RATE_H
#define RATE_H

#include <iostream>
#include <random>
#include <functional>
#include <vector>
#include <string>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <boost/multi_array.hpp>

#include "Xsection.h"

double f_0(double x, double xi);
double fy_wrapper22(double y, void * params_);
double fx_wrapper22(double x, void * px_);
double fy_wrapper23(double y, void * params_);
double fx_wrapper23(double x, void * px_);

struct integrate_params_2{
	std::function<double(double*)> f;   // store a call to member funtion (interp) and object
	double * params;  // other double type parameters
};

double Vegas_func_wrapper(double * var, long unsigned int n_dims, void *params);

class rates{
protected:
	std::random_device rd;
    std::mt19937 gen;
    std::gamma_distribution<double> dist_x, dist_xcorr;
	std::uniform_real_distribution<double> dist_norm_y;
	std::uniform_real_distribution<double> dist_reject;
	virtual void tabulate_E1_T(size_t T_start, size_t dnT) = 0;
	virtual void save_to_file(std::string filename, std::string datasetname) = 0;
	virtual void read_from_file(std::string filename, std::string datasetname) = 0;
public:
	rates(std::string name_);
	virtual double calculate(double * arg) = 0;
	virtual double interpR(double * arg) = 0;
	virtual void sample_initial(double * arg, std::vector< std::vector<double> > & IS) = 0;
};

class rates_2to2 : public rates{
private:
	Xsection_2to2 * Xprocess;
	const double M;
	const int degeneracy;
	const double eta_2;
	size_t NE1, NT;
	double E1L, E1H, TL, TH,
		   dE1, dT;
	boost::multi_array<double, 2> Rtab;
	void tabulate_E1_T(size_t T_start, size_t dnT);
	void save_to_file(std::string filename, std::string datasetname);
	void read_from_file(std::string filename, std::string datasetname);
public:
	rates_2to2(Xsection_2to2 * Xprocess_, int degeneracy_, double eta_2_, std::string name_, bool refresh);
	double calculate(double * arg);
	double interpR(double * arg);
	void sample_initial(double * arg, std::vector< std::vector<double> > & IS);
};

class rates_2to3 : public rates{
private:
	Xsection_2to3 * Xprocess;
	const double M;
	const int degeneracy;
	const double eta_2;
	size_t NE1, NT, Ndt;
	double E1L, E1H, TL, TH, dtL, dtH,
		   dE1, dT, ddt;
	boost::multi_array<double, 3> Rtab;
	void tabulate_E1_T(size_t T_start, size_t dnT);
	void save_to_file(std::string filename, std::string datasetname);
	void read_from_file(std::string filename, std::string datasetname);
public:
	rates_2to3(Xsection_2to3 * Xprocess_, int degeneracy_, double eta_2_, std::string name_, bool refresh);
	double calculate(double * arg);
	double interpR(double * arg);
	void sample_initial(double * arg, std::vector< std::vector<double> > & IS);
};

class rates_3to2 : public rates{
private:
	f_3to2 * Xprocess;
	const double M;
	const int degeneracy;
	const double eta_2, eta_k;
	size_t NE1, NT, Ndt;
	double E1L, E1H, TL, TH, dtL, dtH,
		   dE1, dT, ddt;
	boost::multi_array<double, 3> Rtab;
	AiMS sampler;
	void tabulate_E1_T(size_t T_start, size_t dnT);
	void save_to_file(std::string filename, std::string datasetname);
	void read_from_file(std::string filename, std::string datasetname);
public:
	rates_3to2(f_3to2 * Xprocess_, int degeneracy_, double eta_2_, double eta_k_, std::string name_, bool refresh);
	double calculate(double * arg);
	double interpR(double * arg);
	void sample_initial(double * arg, std::vector< std::vector<double> > & IS);
};



#endif
