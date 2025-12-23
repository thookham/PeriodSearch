# PeriodSearch: Asteroid Spin Detection

[![Build Status](https://github.com/AsteroidsAtHome/PeriodSearch/workflows/CI/badge.svg)](https://github.com/AsteroidsAtHome/PeriodSearch/actions)
[![License](https://img.shields.io/github/license/AsteroidsAtHome/PeriodSearch)](LICENSE)

**PeriodSearch** is the core C++ engine for the **Asteroids@home** BOINC project. It analyzes photometric data from asteroids to determine their rotation periods and spin axis orientations, contributing to our understanding of the solar system's evolution.

---

> [!NOTE]
> **Antigravity Hydrated Fork**
> This repository is a verified hydration of the original legacy codebase. It features a significant modernization of the C++ architecture (introducing the `Solver` class) and includes verification scripts (`SCAFFOLD/`) for generating valid input data for testing.

---

## 🚀 Quick Start

### 1. Modernized Architecture

We have refactored the monolithic legacy `main()` into a modular `PeriodSearchSolver` class (`src/Solver.cpp`), making the code testable and maintainable for modern C++ standards (C++17).

### 2. Verification (Simulation)

To generate valid input files and verify the solver logic without connecting to the live BOINC servers:

```bash
# Generate valid period_search input
python SCAFFOLD/experiments/generate_input.py

# Run the solver (after building)
./period_search
```

### 3. Building

**Requirements**: `boinc-dev` headers, `g++` (C++17 compliant).

```bash
# Using the modernized Makefile
make -f Makefile.modern
```

## 🔭 Science

- **Lightcurve Analysis**: Uses the LomB-Scargle method (and variants) to detect periodicity in sparse data.
- **Shape Modeling**: Reconstructs convex 3D shapes from lightcurves.

## 🛠️ Development

### Contributing

We are actively modernizing this legacy codebase. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for how to help.

### Key Goals

- Remove dependencies on legacy BOINC C-style globals.
- Improve SIMD optimization for faster processing on volunteer machines.

## 📄 License

Released under the [GPLv2 License](LICENSE).
