# Contributing to PeriodSearch

Help us modernize the engine searching for asteroids!

## 🚀 Getting Started

### 1. The Mission: Modernization

This codebase has legacy roots (BOINC C-style). Our goal is to refactor it into modern C++17 while maintaining bit-wise compatibility with the original scientific results.
**Check `src/Solver.cpp` to see the new modular class structure.**

### 2. Antigravity Verification

Before submitting a PR, verify your changes using the simulation tools:

- **Generate Input**: `python SCAFFOLD/experiments/generate_input.py`
- *Note*: This creates valid input files (`stdin`) that the binary expects.

### 3. Submission Process

1. **Fork** the repository.
2. Create a **feature branch**.
3. **Commit** your changes.
4. **Push** to your fork and submit a **Pull Request**.

### 4. Code Style

- **C++17** Standard.
- Avoid global variables where possible (move them into `PeriodSearchSolver`).
- Use descriptive naming (snake_case for variables, PascalCase for classes).

## 🧪 Testing

- The project currently relies on integration testing using the generated input files.
- Unit tests are planned for the new `Solver` methods.

## 🤝 Community

- [Asteroids@home Website](https://asteroidsathome.net)

Thank you for helping us map the solar system!
