#pragma once
#if defined __GNUC__
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#else //_WIN32
#if defined INTEL
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#elif defined AMD
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#endif
#endif

#include <CL/cl.h>

#include <cstdio>
#include "constants.h"
#include <string>
#include <vector>

extern cl_int l_max, m_max, n_iter, last_call,
n_coef, num_fac, n_ph_par,
deallocate; // l_curves,

extern cl_double o_chi_square, chi_square, a_lambda, a_lamda_incr, a_lamda_start, scale, // phi_0,
area[MAX_N_FAC + 1], d_area[MAX_N_FAC + 1], // sclnw[MAX_LC + 1],
//y_out[MAX_N_OBS + 1],
f_c[MAX_N_FAC + 1][MAX_LM + 1], f_s[MAX_N_FAC + 1][MAX_LM + 1],
t_c[MAX_N_FAC + 1][MAX_LM + 1], t_s[MAX_N_FAC + 1][MAX_LM + 1],
d_sphere[MAX_N_FAC + 1][MAX_N_PAR + 1], d_g[MAX_N_FAC + 1][MAX_N_PAR + 1],
normal[MAX_N_FAC + 1][3], bl_matrix[4][4],
pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
d_bl_matrix[3][4][4];
//weight[MAX_N_OBS + 1];

// OpenCL
extern size_t CUDA_grid_dim;
extern cl_program program;

//extern std::vector<cl_int2, int> texture;

//extern cl_int max_l_points;
extern cl_double phi_0;
extern cl_double Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1], Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1], Dg[MAX_N_FAC + 1][MAX_N_PAR + 1];
extern cl_double Area[MAX_N_FAC + 1], Darea[MAX_N_FAC + 1];
//extern cl_int l_points[MAX_LC + 1]; //, in_rel[MAX_LC + 1];
//extern cl_double weight[MAX_N_OBS + 1];

extern std::string kernelCurv, kernelDaveFile, kernelSig2wghtFile;

struct globalsCl
{
#ifdef __GNUC__
    double Nor[3][MAX_N_FAC + 8]{0.0};
    double Area[MAX_N_FAC + 8]{0.0};
    double Darea[MAX_N_FAC + 8]{0.0};
    double dyda[MAX_N_PAR + 16]{0.0};
    double Dg[MAX_N_FAC + 16][MAX_N_PAR + 8]{{0.0}};

    std::vector<std::vector<double>> covar;
    std::vector<std::vector<double>> alpha;
    // AlignedOuterVector covar __attribute__((aligned(64)));
    // AlignedOuterVector alpha __attribute__((aligned(64)));
#else
#if _MSC_VER >= 1900 // Visual Studio 2015 or later
    // NOTE: About MSVC - https://learn.microsoft.com/en-us/cpp/cpp/alignment-cpp-declarations?view=msvc-170
    double Nor[3][MAX_N_FAC + 8] = {};
    double Area[MAX_N_FAC + 8] = {};
    double Darea[MAX_N_FAC + 8] = {};
    double Dg[MAX_N_FAC + 16][MAX_N_PAR + 8] = {};
    double dyda[MAX_N_PAR + 16] = {};
    std::vector<std::vector<double>> covar;
    std::vector<std::vector<double>> alpha;
#else
    __declspec(align(64)) double Nor[3][MAX_N_FAC + 8];
    __declspec(align(64)) double Area[MAX_N_FAC + 8];
    __declspec(align(64)) double Darea[MAX_N_FAC + 8];
    __declspec(align(64)) double Dg[MAX_N_FAC + 16][MAX_N_PAR + 8];
    __declspec(align(64)) double dyda[MAX_N_PAR + 16];
    __declspec(align(64)) std::vector<std::vector<double>> covar;
    __declspec(align(64)) std::vector<std::vector<double>> alpha;
#endif
#endif

    int Lcurves = 0;
    int maxLcPoints = 0;	// replaces macro MAX_LC_POINTS
    int maxDataPoints = 0;	// replaces macro MAX_N_OBS
    int dytemp_sizeX = 0;
    int dytemp_sizeY = 0;

    // points in every lightcurve
    std::vector<int> Lpoints;
    std::vector<int> Inrel;

    double ymod;
    double wt;
    double sig2i;
    double dy;
    double coef;
    double wght;
    double ave;
    double xx1[4]{0.0};
    double xx2[4]{0.0};
    double dave[MAX_N_PAR + 1 + 4]{0.0};
    std::vector<double> ytemp;
    std::vector<double> Weight;
    std::vector<std::vector<double>> dytemp;

    globalsCl()
    {
    }
};
