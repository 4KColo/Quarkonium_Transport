#ifndef LANGEVIN_H
#define LANGEVIN_H

#include <vector>
#include "qhat.h"

void Langevin_pre(double p_length, double M, double temp, double drag, double kperp, double kpara, double delta_lrf, std::vector<double> & pre_result);

void Langevin_post(double p_length, double M, double temp, double drag, double kperp, double kpara, double delta_lrf, std::vector<double> const& pre_result, std::vector<double> & post_result);


#endif
