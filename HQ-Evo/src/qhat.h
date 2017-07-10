#ifndef QHAT_H
#define QHAT_H

#include <iostream>
#include <functional>
#include <vector>
#include <string>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

#include <boost/multi_array.hpp>


#include "qhat_Xsection.h"

struct integrate_params_2_YX
{
        std::function<double(double*)> f;
        double *params;
};


class Qhat
{
protected:
        virtual void tabulate_E1_T(size_t T_start, size_t dnT) = 0;
        virtual void save_to_file(std::string filename, std::string datasetname) = 0;
        virtual void read_from_file(std::string filename, std::string datasetname) = 0;

public:
        Qhat(std::string name_);
        virtual double calculate(double* args) = 0;
        virtual double interpQ(double* args) = 0;
};


class Qhat_2to2: public Qhat
{
private:
        QhatXsection_2to2 * Xprocess;
        const double M;
        const int degeneracy;
        const double eta_2;
        size_t NE, NT;
        double E1L, E1M, E1H, TL, TH, dE1, dE2, dT;
        boost::multi_array<double, 3> QhatTab;
        void tabulate_E1_T(size_t T_start, size_t dnT);
        void save_to_file(std::string filename, std::string datasetname);
        void read_from_file(std::string filename, std::string datasetname);

public:
        Qhat_2to2(QhatXsection_2to2 * Xprocess_, int degeneracy_, double eta_2_, std::string name_, bool refresh);
        double calculate(double *args);
        double interpQ(double *args);
};


#endif
