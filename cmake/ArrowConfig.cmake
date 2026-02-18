# Minimal ArrowConfig.cmake for pyarrow bundled libraries
set(ARROW_FOUND TRUE)
set(ARROW_VERSION "23.0.1")

# Set paths based on pyarrow installation
set(PYARROW_DIR "/home/ubuntu/work/.venv/lib/python3.13/site-packages/pyarrow")
set(ARROW_INCLUDE_DIR "${PYARROW_DIR}/include")
set(ARROW_LIBRARY "${PYARROW_DIR}/libarrow.so.2300")
set(ARROW_LIB_DIR "${PYARROW_DIR}")

# Create imported target
add_library(Arrow::arrow_shared SHARED IMPORTED)
set_target_properties(Arrow::arrow_shared PROPERTIES
    IMPORTED_LOCATION "${ARROW_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ARROW_INCLUDE_DIR}"
)

# Alias for compatibility
add_library(Arrow::arrow ALIAS Arrow::arrow_shared)

# Provide variables that CMake expects
set(ARROW_LIBRARIES Arrow::arrow_shared)
set(ARROW_INCLUDE_DIRS "${ARROW_INCLUDE_DIR}")

mark_as_advanced(ARROW_INCLUDE_DIR ARROW_LIBRARY)
