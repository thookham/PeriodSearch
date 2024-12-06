#pragma once

#include <memory>
#include <vector>

//int CUDAPrepare(int cudadev,double *beta_pole, double *lambda_pole, double *par, double cl,
//				double Alamda_start, double Alamda_incr, double** ee, double** ee0, double *tim,
//				double Phi_0, int checkex, int ndata, struct globals& gl);

int CUDAPrepare(int cudadev, double* beta_pole, double* lambda_pole, double* par, double cl, double Alamda_start, double Alamda_incr,
    std::vector<std::unique_ptr<double[]>>& ee, std::vector<std::unique_ptr<double[]>>& ee0, std::unique_ptr<double[]>& tim,
    double Phi_0, int checkex, int ndata, struct globals& gl);

int CUDAPrecalc(int cudadev, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double* conw_r,
    int ndata, int* ia, int* ia_par, int* new_conw, std::unique_ptr<double[]>& cg_first, double* sig, int Numfac, double* brightness, struct globals& gl);

int CUDAStart(int cudadev, int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
    int ndata, int* ia, int* ia_par, std::unique_ptr<double[]>& cg_first, MFILE& mf, double escl, double* sig, int Numfac, double* brightness, struct globals& gl);

int DoCheckpoint(MFILE& mf, int nlines, int newConw, double conwr);

void CUDAUnprepare(void);

void GetCUDAOccupancy(const int cudaDevice);
