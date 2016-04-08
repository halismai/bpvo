#!/bin/sh

# from https://github.com/jsteemann/travis-cxx11/blob/master/build.sh

echo "Environment: `uname -a`"
echo "Compiler: `$CXX --version`"
echo "CMake: `cmake --version`"

cmake .
make -j2
