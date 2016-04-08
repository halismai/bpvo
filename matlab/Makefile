CXX = c++ # /opt/intel/bin/icpc
CC  = c++ # /opt/intel/bin/icc
LD  = c++ #/opt/intel/bin/xiar
MEX = /usr/local/MATLAB/R2015a/bin/mex
MEX_SUFFIX = $(shell /usr/local/MATLAB/R2015a/bin/mexext)
MEX_OUT_DIR = .

MEX_SRC = $(wildcard *_mex.cc)
LIB_SRC = $(filter-out $(MEX_SRC) $(wildcard *.cc,*.h))
LIB_OBJ = $(patsubst %.cc,%.o, $(LIB_SRC))

INCLUDES = -I/home/halismai/code/mexmat \
					 -I/usr/local/MATLAB/R2015a/extern/include/ \
					 -I/usr/local/include/eigen3/ \
					 -I/home/halismai/code/bpvo/

LD_FLAGS = /home/halismai/code/bpvo/build/bin/libbpvo.a \
					 /home/halismai/code/bpvo/build/bin/libbpvo_utils.a \
					 -lgomp -ltbb -lopencv_core \
					 -L/usr/local/lib/ -lopencv_core -lopencv_imgproc

CXX_FLAGS = -fPIC -O2 -fopenmp -DMEXMAT_WITH_EIGEN -DMEXMAT_WITH_OPENCV -mtune=native -funroll-loops -std=c++11 -DNDEBUG

MEX_FLAGS = -cxx CC='$(CC)' CXX='$(CXX)' LD='$(LD)'

all: $(patsubst %.cc,%.$(MEX_SUFFIX), $(MEX_SRC))

%.$(MEX_SUFFIX):%.o
	$(MEX) $(MEX_FLAGS) $(LD_FLAGS) -output $(MEX_OUT_DIR)/$@ $^

%.o: %.cc
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $^

clean::
	rm -rf *.o *.$(MEX_SUFFIX)

