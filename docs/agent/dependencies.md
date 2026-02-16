# Dependencies

## C++ Build

- **Compiler**: GCC >= 8.1 or Clang >= 6.0
- **CMake**: >= 3.6
- **Build System**: Make or Ninja
- **OpenMP**: For parallelism (ships with compiler)
- **Apache Arrow**: libarrow-dev (for columnar memory support)

## Python Build

- **Python**: >= 3.9
- **Cython**: >= 0.29, < 3.1.0
- **NumPy**: For array operations
- **SciPy**: For scientific computing
- **PyArrow**: For Arrow integration
- **setuptools**: For package building
- **wheel**: For wheel building

## Runtime Dependencies

```toml
dependencies = [
    "scipy",
    "numpy",
    "pyarrow"
]
```

## Testing Dependencies

- **pytest**: For Python tests
- **googletest**: Built-in for C++ tests (in extlibs/)

## Optional

- **clang-format**: For code formatting
- **address/leak sanitizers**: For debugging (via CMake flags)
