#!/bin/bash
set -e
set -o pipefail

$CXX --version

if [[ "$(uname)" == "Darwin" ]]; then
  CMAKE_PREFIX_PATH="-DCMAKE_PREFIX_PATH=/opt/homebrew/opt/libomp"
fi

mkdir debug_test && cd "$_"
cmake -GNinja -DNETWORKIT_BUILD_TESTS=ON -DNETWORKIT_MONOLITH=$MONOLITH -DNETWORKIT_CXX_STANDARD=$CXX_STANDARD -DNETWORKIT_WARNINGS=ON -DCMAKE_BUILD_TYPE=Debug -DNETWORKIT_SANITY_CHECKS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache $CMAKE_PREFIX_PATH ..
ninja

ctest -V
