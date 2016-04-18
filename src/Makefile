CXX=g++
CC=gcc
LD=$(CXX)
LFLAGS=-L /usr/local/lib -L /opt/OpenBLAS/lib
FF=gfortran
MYLIBS=-lm -lc -lboost_math_tr1 -lgcc -lopenblas -lgfortran -lrt
MPI_INCLUDES=-I /usr/include/boost -I /usr/lib/mpich/include
#find the type of system - specifically to fix MacOS issues
UNAME := $(shell uname)
CLANG := $(shell gcc -v 2>&1 | grep clang )
#if MacOS and clang aliased to gcc find the real gcc 
ifeq ($(UNAME), Darwin)
 MAC_INCLUDES= -I /usr/include
 OS=-D MAC_OS=1
 OPENBLASDIR :=  $(shell dirname $$(find /usr 2>/dev/null | grep openblas.dylib) )
 LIBDIRS=-L/usr/local/lib/gcc -L$(OPENBLASDIR)
 MYLIBS=-lm -lc -lboost_math_tr1 -lgcc -lopenblas
 ifneq ($(CLANG),)
	ifneq ($(shell ls /usr/bin/gcc-4* 2>/dev/null),)
		CXX=$(shell ls /usr/bin/g++-4* )
		CC++=$(shell ls /usr/bin/gcc-4*)
	else ifneq ($(shell ls /usr/local/bin/gcc-4* 2>/dev/null),)
		CXX=$(shell ls /usr/local/bin/g++-4*)
		CC=$(shell ls /usr/local/bin/gcc-4*)
	else ifneq ($(shell ls /usr/bin/gcc-5* 2>/dev/null),)
		CXX=$(shell ls /usr/bin/g++-5*)
		CC=$(shell ls /usr/bin/gcc-5*)
	else ifneq ($(shell ls /usr/local/bin/gcc-5* 2>/dev/null),)
		CXX=$(shell ls /usr/local/bin/g++-5*)
		CC=$(shell ls /usr/local/bin/gcc-5*)
	endif	
   endif
endif
LOPENMP= -lgomp
ifdef USEMPI
LMPI= -lmpi -lboost_serialization -lboost_mpi
DEFINES= -DUSEMPI
endif
OPT_FLAGS= -Ofast -ftree-vectorize -funroll-all-loops -m64 -msse3 -mavx
ifdef DEBUG
 OPT_FLAGS= -ggdb -m64 -msse3 -mavx
endif
FFLAGS=-fimplicit-none

CFLAGS= -I ./ -I ./tclap-1.2.1/include $(MAC_INCLUDES) $(MPI_INCLUDES) $(OPT_FLAGS) -std=c++11 -Wno-write-strings -fexpensive-optimizations -fopenmp -ffast-math -finline-functions -frerun-loop-opt -static $(HDRS) $(DEFINES) $(OS)
LIBS= $(LIBDIRS) $(MYLIBS) $(LOPENMP) $(LMPI)

# object files needed

OBJ_fastBMA = main.o\
	MurmurHash3.o

fastBMA :$(OBJ_fastBMA)
	$(LD) -o fastBMA $(LFLAGS) $(OBJ_fastBMA) $(LIBS)
	
main.o: main.cpp
	$(CXX) $(CFLAGS) -c main.cpp
MurmurHash3.o: MurmurHash3.cpp
	$(CXX) $(CFLAGS) -c MurmurHash3.cpp	
clean:
	rm *.o
	rm core.* 
# all together now...
default: fastBMA
print-% : ; @echo $* = $($*)
