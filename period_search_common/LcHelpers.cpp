#include "pch.h"
#include <string>
#include <fstream>
#include <numeric>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <memory>
#include <iostream>

#if defined __GNU__
#include <bits/stdc++.h>
#endif

#include "constants.h"
#include "arrayHelpers.hpp"

/* number of lightcurves and the first realtive one */
void processLine15(struct globals& gl, const char* line, int& err) {
    err = sscanf(line, "%d", &gl.Lcurves);
    if (err != 1) {
        err = -1;
        return;
    }

    //gl.Lpoints = std::make_unique<int[]>(gl.Lcurves + 2); // +2 instead of +1+1
    //std::fill_n(gl.Lpoints.get(), gl.Lcurves + 2, 0);
    //gl.Lpoints = new int[gl.Lcurves + 2](); // +2 instead of +1+1, zero-initialized, f.e. {0, 310, 1142, 15, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...}
    gl.Lpoints.resize(gl.Lcurves + 2, 0);
}

void processLine16(struct globals& gl, const char* line, int& err, int& offset, int& i, int& i_temp, int lineNumber) {
    err = sscanf(line, "%d %d", &gl.Lpoints[i], &i_temp);
    if (err != 2) {
        err = -1;
        return;
    }
    //fprintf(stderr, "[LcHelpers] Lpoints[%d] %d\n", i, gl.Lpoints[i]);

    offset = lineNumber;
    i++;
}

void processSubsequentLines(struct globals& gl, const char* line, int& err, int& offset, int& i, int& i_temp, int lineNumber) {
    err = sscanf(line, "%d %d", &gl.Lpoints[i], &i_temp);
    if (err != 2) {
        err = -1;
        return;
    }
    //fprintf(stderr, "[LcHelpers] Lpoints[%d] %d\n", i, gl.Lpoints[i]);
    offset = lineNumber;
    i++;
}

// Template function to initialize a 2D vector (matrix) in any structure
template <typename T>
void init_matrix(std::vector<std::vector<T>>& matrix, int rows, int cols, T init_value = T{})
{
    matrix.resize(rows + 1); // Resize the outer vector

    for (int i = 0; i < rows + 1; ++i) {
        matrix[i].resize(cols + 1, init_value); // Resize and initialize each inner vector with the specified value
    }
}

/// Convexity regularization: make one last 'lightcurve' that
/// consists of the three comps.of the residual non-convex vectors
/// that should all be zero
/// @param gl struct of globals
void MakeConvexityRegularization(struct globals& gl)
{
    gl.Lcurves = gl.Lcurves + 1;
    gl.Lpoints[gl.Lcurves] = 3;
    gl.Inrel[gl.Lcurves] = 0;

    //gl.maxDataPoints = std::accumulate(gl.Lpoints.get(), gl.Lpoints.get() + gl.Lcurves + 1, 0);
    //gl.maxDataPoints = std::accumulate(gl.Lpoints, gl.Lpoints + gl.Lcurves + 1, 0);    // keep it '+ 1' instead of ' + 2' as the gl.Lcurves has been incremented by 1 already!
    gl.maxDataPoints = std::accumulate(gl.Lpoints.begin(), gl.Lpoints.end(), 0);

    //for (auto q = 0; q <= gl.Lcurves; q++)
    //    fprintf(stderr, "Lpoints[%d] %d\n", q, gl.Lpoints[q]);
}

///< summary>
/// Performs the first loop over lightcurves to find all data points (replacing MAX_LC_POINTS, MAX_N_OBS, etc.)
///</summary>
///< param name="gl"></param>
///< param name="filename"></param>
int PrepareLcData(struct globals& gl, const char* filename)
{
    int i_temp;
    int err = 0;

    std::ifstream file(filename);
    std::string lineStr;
    gl.Lcurves = 0;

    if (!file.is_open())
    {
        return 2;
    }

    int lineNumber = 0;
    int offset = 0;
    int i = 1;

    std::unordered_map<int, std::function<void(const char*, int&)>> actions;
    actions[15] = [&](const char* line, int& err) { processLine15(gl, line, err); };
    actions[16] = [&](const char* line, int& err) { processLine16(gl, line, err, offset, i, i_temp, lineNumber); };

    while (std::getline(file, lineStr))
    {
        char line[MAX_LINE_LENGTH];
        std::strcpy(line, lineStr.c_str());
        lineNumber++;

        if (actions.find(lineNumber) != actions.end()) {
            actions[lineNumber](line, err);
            if (err <= 0) {
                file.close();
                return err;
            }
        }

        if (lineNumber <= 16)
        {
            continue;
        }

        if (lineNumber == offset + 1 + gl.Lpoints[i - 1])
        {
            processSubsequentLines(gl, line, err, offset, i, i_temp, lineNumber);
            if (err <= 0) {
                file.close();
                return err;
            }
            if (i > gl.Lcurves)
            {
                break;
            }
        }
    }

    file.close();

    //gl.Inrel = std::make_unique<int[]>(gl.Lcurves + 1);
    //std::fill_n(gl.Inrel.get(), gl.Lcurves + 1, 0);
    //gl.Inrel = new int[gl.Lcurves + 2]();   // Same, code runs 1-bazed arrays so if we have let's say 3 LC, then we need 4 + 1 size - one extra element for Convexity regularization
    gl.Inrel.resize(gl.Lcurves + 2, 0);

    //gl.maxLcPoints = *(std::max_element(gl.Lpoints.get(), gl.Lpoints.get() + gl.Lcurves + 1)) + 1;
    //gl.maxLcPoints = *(std::max_element(gl.Lpoints, gl.Lpoints + gl.Lcurves + 2));
    gl.maxLcPoints = *(std::max_element(gl.Lpoints.begin(), gl.Lpoints.end()));
    
    //gl.ytemp = std::make_unique<double[]>(gl.maxLcPoints + 1);
    //std::fill_n(gl.ytemp.get(), gl.maxLcPoints + 1, 0.0);
    //gl.ytemp = new double[gl.maxLcPoints + 2]();        // Not used in CUDA
    gl.ytemp.resize(gl.maxLcPoints + 2, 0.0);        // Not used in CUDA

    gl.dytemp_sizeY = MAX_N_PAR + 1 + 4;
    gl.dytemp_sizeX = gl.maxLcPoints + 2;
    //init2Darray(gl.dytemp, gl.dytemp_sizeX, gl.dytemp_sizeY);  // Not used in CUDA
    init_matrix(gl.dytemp, gl.dytemp_sizeX, gl.dytemp_sizeY, 0.0);  // Not used in CUDA

    //gl.maxDataPoints = std::accumulate(gl.Lpoints.get(), gl.Lpoints.get() + gl.Lcurves + 2, 0);
    //gl.maxDataPoints = std::accumulate(gl.Lpoints, gl.Lpoints + gl.Lcurves + 2, 0);   // OK
    gl.maxDataPoints = std::accumulate(gl.Lpoints.begin(), gl.Lpoints.end(), 0);   // OK

    //gl.Weight = std::make_unique<double[]>(gl.maxDataPoints + 1);
    //std::fill_n(gl.Weight.get(), gl.maxDataPoints + 1, 0.0);
    //gl.Weight = new double[gl.maxDataPoints + 1]();
    gl.Weight.resize(gl.maxDataPoints + 1 + 4, 0.0);

    gl.ave = 0.0;

    //MakeConvexityRegularization(gl);

    return 1;
}
