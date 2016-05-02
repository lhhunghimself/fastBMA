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
###Compilation from source

The compilation is relatively straightforward for Linux and MacOS and should work with MinGW with some minor modifications to the Makefile. However, it is much easier to use the Docker container especailly if you want to set up a distributed cloud network to run fastBMA.  Even simpler is the R package. However the R version lacks some minor features and does not use OpenBLAS or MPI.

fastBMA uses [OpenBLAS](http://www.openblas.net/) and [mpich]/boost-mpi. These need to be

. [OpenBLAS](http://www.openblas.net/) needs to be installed as does OpenMPI. It probably will work with MinGW by changing the Makefile to point to the correct headers for the boost and MPI libraries (if desired). 

However, it is **MUCH** easier to use the Docker repositories with very little overhead especially for Linux. The R package is 
Clone the repository or download the zip file and extract the contents. There are two Makefiles, one for a typical Fedora installation and one for a Ubuntu installation. The openBLAS headers are included but openBLAS itself must be installed. 

##Sample usage#

##Data File Formats#

##Use with MPI#

##Use with R#

##Algorithm Documentation#

##Benchmarks#
