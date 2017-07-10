#ifndef TLORENTZ_H
#define TLORENTZ_H


#include <iostream>
#include <stdlib.h>

double* rot(double theta, double phi, double* p);
double* rot2(double cos_theta, double sin_theta, double cos_phi, double sin_phi, double* p);
double* bos(double* beta, double* p);
double* transform_to_CoM(double* p1, double* p2, double* vec);
double* transform_from_CoM(double* p1, double* p2, double* vec);
double** transform_from_CoM_array(double* p1, double* p2);

#endif
