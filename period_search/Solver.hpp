#ifndef PERIOD_SEARCH_SOLVER_HPP
#define PERIOD_SEARCH_SOLVER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

// Forward declarations for legacy globals
// (We still use them because mrqmin.c expects them)
extern int Lmax, Mmax, Niter, Lastcall, Ncoef, Numfac, Lcurves, Nphpar;
extern double Ochisq, Chisq, Alamda, Alamda_incr, Alamda_start, Phi_0, Scale;
extern int* Inrel;
extern int* Lpoints;

class PeriodSearchSolver {
public:
    PeriodSearchSolver();
    ~PeriodSearchSolver();

    // Replaces the legacy main() logic
    int Run();

private:
    // Helper methods
    void LoadData(const std::string& input_path);
    void Initialize();
    void Cleanup();
    
    // Checkpointing helper from original code
    int DoCheckpoint(int nlines);

    // Context for local variables in main() loop
    std::vector<int> ia;
    std::vector<double> t, f, at, af;
    // ... add more as we refactor logic
};

#endif // PERIOD_SEARCH_SOLVER_HPP
