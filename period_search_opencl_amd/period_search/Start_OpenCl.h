#pragma once
#include <CL/cl.h>
#include "mfile.h"

//extern cl_int ClPrepare(cl_int deviceId, cl_double* beta_pole, cl_double* lambda_pole, cl_double* par, cl_double cl, cl_double Alambda_start, cl_double Alambda_incr,
//	cl_double ee[][3], cl_double ee0[][3], cl_double* tim, cl_double Phi_0, cl_int checkex, cl_int ndata);

cl_int ClPrepare(cl_int deviceId, cl_double* beta_pole, cl_double* lambda_pole, cl_double* par, cl_double cl, cl_double Alambda_start, cl_double Alambda_incr,
	std::vector<std::vector<double>>& ee, std::vector<std::vector<double>>& ee0, std::vector<double>& tim,
    cl_double Phi_0, cl_int checkex, cl_int ndata, struct globals& gl);

//extern cl_int ClPrecalc(cl_double freq_start, cl_double freq_end, cl_double freq_step, cl_double stop_condition, cl_int n_iter_min, cl_double* conw_r,
//	cl_int ndata, cl_int* ia, cl_int* ia_par, cl_int* new_conw, cl_double* cg_first, cl_double* sig, cl_int Numfac, cl_double* brightness, cl_double lcoef, int n_coef);

cl_int ClPrecalc(cl_double freq_start, cl_double freq_end, cl_double freq_step, cl_double stop_condition, cl_int n_iter_min, cl_double* conw_r,
	cl_int ndata, std::vector<int>& ia, cl_int* ia_par, cl_int* new_conw, std::vector<double>& cg_first, std::vector<double>& sig, cl_int Numfac,
                 std::vector<double> &brightness, struct globals &gl, cl_double lcoef, int n_coef);

//extern int ClStart(int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
//	int ndata, int* ia, int* ia_par, double* cg_first, MFILE& mf, double escl, double* sig, int Numfac, double* brightness);

cl_int ClStart(int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
	int ndata, std::vector<int>& ia, int* ia_par, std::vector<double>& cg_first, MFILE& mf, double escl, std::vector<double>& sig, int Numfac,
    std::vector<double>& brightness, struct globals& gl);

int DoCheckpoint(MFILE &mf, int nlines, int newConw, double conwr);

//void PrepareBufferFromFlatenArray(size_t CUDA_grid_dim_precalc, globals &gl, cl_mem &bufJpScale, cl_int &err);

void ReleaseGlobalClObjects();
