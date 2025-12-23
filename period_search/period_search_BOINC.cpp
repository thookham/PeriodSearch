/* refined and modernized version of period_search_BOINC
   Uses PeriodSearchSolver class for logic encapsulation.
*/

#include "Solver.hpp"
#include "declarations.h" // For globals extern if needed, though Solver handles it
#include "boinc_api.h"

#ifdef _WIN32
#include "boinc_win.h"
#endif

// The globals are defined in Solver.cpp to satisfy the linker for legacy C files.

int main(int argc, char** argv) {
    // Instantiate the Solver
    PeriodSearchSolver solver;

    // Run the solver
    // Logic is encapsulated in Solver::Run()
    return solver.Run();
}

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR Args, int WinMode) {
    LPSTR command_line;
    char* argv[100];
    int argc;

    command_line = GetCommandLine();
    argc = parse_command_line( command_line, argv );
    return main(argc, argv);
}
#endif
