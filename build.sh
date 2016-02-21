#!/bin/sh


mkdir -p build && cd build && cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_TEST=ON                   \
  -DWITH_TBB=ON                     \
  -DWITH_TCMALLOC=OFF               \
  -DENABLE_OMIT_FRAME_POINTER=ON    \
  -DENABLE_FAST_MATH=ON             \
  -DBUILD_STATIC=ON                 \
  -DWITH_BITPLANES=ON && make -j2

