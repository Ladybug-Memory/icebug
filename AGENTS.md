# AGENTS.md - Icebug Development Guide

This file provides essential guidelines for agentic coding agents working on the Icebug codebase.

## Project Overview

Icebug is a high-performance graph analytics library written in C++ with Python bindings via Cython. The project is a fork of NetworKit and supports read-only graphs in CSR format built on top of Apache Arrow.

## Build Commands

### Building the Python Module

```bash
git clone https://github.com/Ladybug-Memory/icebug icebug
cd icebug
uv pip install -e ".[test]"         # Install with test dependencies
```

### Running Tests

**Python tests (pytest):**
```bash
uv run pytest networkit/test/                           # Run all tests
uv run pytest networkit/test/test_graph.py              # Run specific test file
uv run pytest networkit/test/test_graph.py::TestGraph   # Run specific test class
uv run pytest networkit/test/test_graph.py::TestGraph::testAddNodes  # Run single test
uv run pytest -k "testAddNodes"                         # Run tests matching pattern
```

### Building with CMake (C++ core only)

```bash
mkdir build && cd build
cmake -DNETWORKIT_BUILD_TESTS=ON ..
make -jX
```

**C++ unit tests:**
```bash
# Build tests first (requires CMake with -DNETWORKIT_BUILD_TESTS=ON)
./networkit_tests                                    # Run all C++ tests
./networkit_tests --gtest_filter=CentralityGTest.testBetweennessCentrality  # Single test
./networkit_tests --loglevel=INFO                   # With log level
```

### Linting and Formatting

```bash
clang-format -style=file <code-file>   # Format a single file
# Add // networkit-format comment before includes to enable formatting
```

## Code Style Guidelines

### C++ Code

- **Indentation:** 4 spaces (no tabs)
- **Column limit:** 100 characters
- **Standard:** C++11

**Naming Conventions:**
- Classes: `UpperCamelCase` (e.g., `Graph`, `GraphBuilder`)
- Functions/Methods: `lowerCamelCase` (e.g., `addNode()`, `numberOfEdges()`)
- Variables: `lowerCamelCase` (e.g., `nodeCount`)
- Class members: lowerCamelCase with trailing underscore (e.g., `nodeCount_`)
- Constants: `kCamelCase` or `UPPER_SNAKE_CASE`

**Includes Order** (per `.clang-format`):
1. `<networkit/...>` - project headers (priority 4)
2. `<tlx/...>`, `<googletest/...>`, `<ttmath/...>` - external libs (priority 3)
3. `"/..."` - local headers (priority 2)
4. `"..."` - main header (priority 1)

**Other Guidelines:**
- Use `nullptr` instead of `NULL`
- Use `override` keyword when overriding virtual methods
- Prefer `final` for classes not meant to be inherited

### Python Code

- **Indentation:** 4 spaces
- **Testing:** Use unittest framework with class-based tests

**Naming Conventions:**
- Classes: `UpperCamelCase`
- Functions/Methods: `snake_case`
- Variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

**Imports:** Group by standard library, third-party, then local:
```python
import unittest
import random

import networkit as nk
import numpy as np
import scipy as sc
```

### Error Handling

- **C++:** Use exceptions (`throw std::runtime_error(...)`)
- **Python:** Use `with self.assertRaises(RuntimeError):`
- Provide meaningful error messages

### Documentation

- Use Doxygen-style comments for C++ classes and methods
- Document public APIs thoroughly
- Include examples in docstrings for Python APIs

## Reference Documents

- [Project Structure](docs/agent/structure.md)
- [Dependencies](docs/agent/dependencies.md)
- [Development Workflow](docs/agent/workflow.md)
