# Project Structure

```
icebug/
├── include/networkit/     # C++ headers
├── networkit/cpp/        # C++ implementation
├── networkit/            # Python package
│   └── test/             # Python tests
├── extlibs/              # External libraries (tlx, ttmath, googletest)
├── setup.py              # Python build script
├── CMakeLists.txt        # C++ build configuration
└── .clang-format         # C++ formatting rules
```

## Key Directories

- `include/networkit/` - Public C++ headers
- `networkit/cpp/` - C++ implementation files
- `networkit/` - Python module (cython wrappers)
- `networkit/test/` - Python unit tests
- `extlibs/` - Third-party dependencies (tlx, ttmath, googletest)

## Build Outputs

- Python extension: `networkit/*.so`
- C++ core: `build/` directory with `libnetworkit.*`
- C++ tests: `./networkit_tests` (when built with tests)
