#ifndef QHAT_XSECTION_H
#define QHAT_XSECTION_H

#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <boost/multi_array.hpp>


struct YXgsl_integration_params
{
        double(*f)(double* args, size_t ndims, void* params);
        double* params;
};



//==== qhat_Xsection base class
class QhatXsection
{
protected:
        virtual void tabulate(size_t T_start, size_t dnT) = 0;
        virtual void save_to_file(std::string filename, std::string datasetname) = 0;
        virtual void read_from_file(std::string filename, std::string datasetname) =0;
        double (*dXdPS)(double * PS, size_t ndims, void* params);
        double M1;
public:
        QhatXsection(double (*dXdPS_)(double*, size_t, void*), double M1_, std::string name_, bool refresh);
        double get_M1(void) {return M1;};
        //args = [s, T, index] for X22
        virtual double interpX(double *args) = 0;
        virtual double calculate(double *args) = 0;
};






// ==== derived 2->2 Xsection class
class QhatXsection_2to2: public QhatXsection
{
private:
        void tabulate(size_t T_start, size_t dnT);
        void save_to_file(std::string filename, std::string datasetname);
        void read_from_file(std::string filename, std::string datasetname);
        size_t Nsqrts, NT;
        double sqrtsL, sqrtsM, sqrtsH, dsqrts1, dsqrts2, 
                TL, TH, dT;
        boost::multi_array<double, 3> QhatXtab;
public:
        QhatXsection_2to2(double (*dXdPS_)(double*, size_t, void*), double M1_, std::string name_, bool refresh);
        double interpX(double *args);
        double calculate(double *args);
};

#endif
