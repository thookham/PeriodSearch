# Notes for PeriodSearch

## Initial Observations

- Repository cloned successfully.
- Contains legacy C/C++ code for the Asteroids@home BOINC project.

## Legacy Code Analysis (`period_search/period_search_BOINC.cpp`)

- **Structure**: Monolithic C-style C++ file.
- **Dependencies**: `boinc_api`, `gsl` (implied or internal math libs like `mrqmin`).
- **Algorithm**:
    1. Reads lightcurve data from `period_search_in`.
    2. Iterates through frequency steps (`freq_start` to `freq_end`).
    3. Uses Levenberg-Marquardt optimization (`mrqmin`) to fit asteroid shape models to the lightcurves.
    4. Outputs best period and pole info to `period_search_out`.
- **Code Quality**:
  - Heavy use of global variables.
  - Mix of C (`FILE*`, `malloc`) and C++ (`std::string`).
  - "Dark facet" optimization logic handles physical plausibility.

## Next Steps

- **Refactor**: Encapsulate global state into a `Context` or `Solver` class to make it testable.
  - *Action*: Created `Solver.hpp` and `Solver.cpp`.
  - *Action*: Moved `main` logic to `PeriodSearchSolver::Run()`.
  - *Action*: Encapsulated globals in `Solver.cpp` (logic) while exposing them `extern` for legacy C linkage.
  - *Note*: `period_search_BOINC.cpp` is now a clean entry point. **IMPORTANT**: `Solver.cpp` must be added to the build targets (VS Solution / Makefile).
- **Modernize**: Replace `FILE*` with `<iostream>` or `<fstream>`, and raw arrays with `std::vector`.
  - *Action*: Partially implemented `std::ifstream` logic in `Solver::Run` (skeleton). Fully replaced raw allocation macros with `std::vector` in `PeriodSearchSolver` constructor.
