# fastBMA
Fast, scalable, parallel and distributed inference of very large networks by Bayesian Model Averaging

##Summary#
###Motivation:#
Inferring genetic networks from genome-wide expression data is extremely demanding computationally. We have developed fastBMA, a distributed, parallel and scalable implementation of Bayesian model averaging (BMA) for this purpose. fastBMA also includes a novel and computationally efficient method for eliminating redundant indirect edges in the network. fastBMA is orders of magnitude faster than existing fast methods such as ScanBMA and LASSO. fastBMA also has a much smaller memory footprint and produces more accurate and compact networks. A 100 gene network is obtained in 0.1 seconds and a complete 10,000-gene regulation network can be obtained in a matter of hours. 
###Results:#
We evaluated the performance of fastBMA on synthetic data and real genome-wide yeast and human datasets. When using a single CPU core, fastBMA is 30 times faster than ScanBMA and up to 100 times faster than LASSO with increased accuracy. The new transitive reduction algorithm is fast and increases the accuracy of the most confidently predicted edges. fastBMA is memory efficient and can be run on multiple instances for the increased speed nec-essary for genome-wide analyses.
###Availability:#
fastBMA is available as a standalone function or as part of the networkBMA R package. The binaries are also distributed in portable software containers, for reproducible deployment on Linux/Mac/Windows machines, cloud instances and clusters. The source code is open source (M.I.T. license). Downloads are available through a GitHub and Docker Hub repository. 

##Installation#
###Compilation from source#

The compilation is relatively straightforward for Linux and MacOS and should work with MinGW with some minor modifications to the Makefile. However, it is much easier to use the Docker container especailly if you want to set up a distributed cloud network to run fastBMA.  Even simpler is the R package. However the R version lacks some minor features and does not use OpenBLAS or MPI.

fastBMA uses [OpenBLAS](http://www.openblas.net/) and [mpich](https://www.mpich.org/)/boost-mpi(http://www.boost.org/doc/libs/1_60_0/doc/html/mpi.html). mpich2 can be installed as a package using apt-get/yum/dnf/brew. However, OpenBLAS must be compiled from source, as does boost if MPI is to be used. Compilation instructions for boost/boost-mpi can be found [here](http://kratos-wiki.cimne.upc.edu/index.php/How_to_compile_the_Boost_if_you_want_to_use_MPI).

Once the necessary libraries are installed, clone the repository or download the zip file and extract the contents and change into the CD. There are two Makefiles, one for a typical Fedora installation and one for a Ubuntu installation. The non-MPI installation has also been tested on MacOS-Yosemite and it probably will compile under MinGW on Windows with minor changes to the Makefile. Once in the src directory.

    cp Makefile.Ubuntu Makefile
    make clean; make <FLAGS>
  
If MPI is desired the <FLAGS> should include USEMPI=1.
For MACOS <FLAGs> should include MACOS=1

After compilation you can run the provided test scripts runFastBMA.sh and runfastBMAMPI.sh which should infer a network from a 100 variable time series.

###Installation using Docker#

###Installation using R#

##Sample usage#



##Data File Formats#

##Use with MPI#

##Use with R#

##Algorithm Documentation#

##Benchmarks#
