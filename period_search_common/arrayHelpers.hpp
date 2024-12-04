#pragma once
#include "constants.h"
#include <vector>
#include <memory>

void init3Darray(std::vector<double>& vector, int dim1, int dim2, int dim3);
void init3Darray(std::unique_ptr<std::unique_ptr<std::unique_ptr<double[]>[]>[]>& de, int dim1, int dim2, int dim3);
void init2Darray(std::vector<std::unique_ptr<double[]>>& matrix, int xSize, int ySize);
void init2Darray(double**& matrix, int dytemp_siszeX, int dytemp_sizeY);
void delete2Darray(double**& ary, int sizeY);
void printArray(int array[], int iMax, char msg[]);
void printArray(double array[], int iMax, char msg[]);
void printArray(double** array, int iMax, int jMax, char msg[]);
void printArray(double*** array, int iMax, int jMax, int kMax, char msg[]);

struct globals
{
//#if defined CUDA_VERSION
//
//#else
#ifdef __GNUC__
    double Nor[3][MAX_N_FAC + 8] __attribute__((aligned(64))),
        Area[MAX_N_FAC + 8] __attribute__((aligned(64))),
        Darea[MAX_N_FAC + 8] __attribute__((aligned(64))),
        Dg[MAX_N_FAC + 16][MAX_N_PAR + 8] __attribute__((aligned(64)));
    double dyda[MAX_N_PAR + 16] __attribute__((aligned(64)));
#else
    // NOTE: About MSVC - https://learn.microsoft.com/en-us/cpp/cpp/alignment-cpp-declarations?view=msvc-170
    alignas(64) double Nor[3][MAX_N_FAC + 8];
    alignas(64) double Area[MAX_N_FAC + 8];
    alignas(64) double Darea[MAX_N_FAC + 8];
    alignas(64) double Dg[MAX_N_FAC + 16][MAX_N_PAR + 8];
    alignas(64) double dyda[MAX_N_PAR + 16];
#endif
//#endif

    int Lcurves;        // replaces MAX_LC
    int maxLcPoints;	// replaces MAX_LC_POINTS
    int maxDataPoints;	// replaces MAX_N_OBS
    int dytemp_sizeX;
    int dytemp_sizeY;
    std::unique_ptr<int[]> Lpoints;	    // int*
    std::unique_ptr<int[]> Inrel;		// int*

    double ymod;
    double wt;
    double sig2i;
    double dy;
    double coef;
    double wght;
    double ave;
    double xx1[4];
    double xx2[4];
    double dave[MAX_N_PAR + 1 + 4];
    std::unique_ptr<double[]> ytemp;
    std::unique_ptr<double[]> Weight;
    std::vector<std::unique_ptr<double[]>> dytemp;
};
