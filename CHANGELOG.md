# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0-antigravity] - 2025-12-21
### Added
- **Modern Build System**: Added `Makefile.modern` to support GCC/Dynamic Linking on Linux.
- **Docs**: Updated README to recommend Linux/WSL.

### Fixed
- Removed hardcoded paths to static libraries in build system.

## [Unreleased]

### Added

- **Refactoring**: Introduced `PeriodSearchSolver` class in `period_search/Solver.hpp` and `Solver.cpp` to encapsulate global state and modernized the legacy C codebase.
- **Modernization**: Replaced monolithic `main()` in `period_search_BOINC.cpp` with object-oriented instantiation.
- **Simulation**: Added `SCAFFOLD/experiments/generate_input.py` to generate valid `period_search_in` input files.
- **Documentation**: Updated `SCAFFOLD/NOTES.md` with legacy code analysis.

- Feature X for performance optimization.

## [1.0.0] - 2025-12-21

### Added

- Initial architectural scaffold.
- Documentation suite implementation.
