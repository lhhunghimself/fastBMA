# fastBMA
Fast, scalable, parallel and distributed inference of very large networks by Bayesian Model Averaging
##Acknowledgements#
The core approach used by fastBMA is based on [ScanBMA](http://bmcsystbiol.biomedcentral.com/articles/10.1186/1752-0509-8-47) by William Chad Young, Adrian E Raftery and Ka Yee Yeung . For the R implementation, Kaiyuan Shi wrote and tested the Rcpp glue code. LHH wrote the configuration scripts for the R package. Daniel Kristiyanto wrote and tested the initial version of the Dockerfile. This work was supported by the National Institutes of Health U54HL127624 to KaYee Yeung, R01HD054511 to Adrian Raftery, R01HD070936 to Adrian Rafftery, and the Microsoft Azure for Research Award. 
##Summary#
###Motivation:#
Inferring genetic networks from genome-wide expression data is extremely demanding computationally. We have developed fastBMA, a distributed, parallel and scalable implementation of Bayesian model averaging (BMA) for this purpose. fastBMA also includes a novel and computationally efficient method for eliminating redundant indirect edges in the network. fastBMA is orders of magnitude faster than existing fast methods such as ScanBMA and LASSO. fastBMA also has a much smaller memory footprint and produces more accurate and compact networks. A 100 gene network is obtained in 0.1 seconds and a complete 10,000-gene regulation network can be obtained in a matter of hours. 
###Results:#
We evaluated the performance of fastBMA on synthetic data and real genome-wide yeast and human datasets. When using a single CPU core, fastBMA is 30 times faster than ScanBMA and up to 100 times faster than LASSO with increased accuracy. The new transitive reduction algorithm is fast and increases the accuracy of the most confidently predicted edges. fastBMA is memory efficient and can be run on multiple instances for the increased speed nec-essary for genome-wide analyses.
###Availability:#
fastBMA is available as a standalone function or as part of the networkBMA R package. The binaries are also distributed in portable software containers, for reproducible deployment on Linux/Mac/Windows machines, cloud instances and clusters. The source code is open source (M.I.T. license). 

##Installation#
###Compilation from source#

The compilation is relatively straightforward for Linux and MacOS and should work with MinGW with some minor modifications to the Makefile. However, it is much easier to use the Docker container especailly if you want to set up a distributed cloud network to run fastBMA.  Even simpler is the R package. However the R version lacks some minor features and does not use OpenBLAS or MPI.

fastBMA uses [OpenBLAS](http://www.openblas.net/) and [mpich](https://www.mpich.org/)/boost-mpi(http://www.boost.org/doc/libs/1_60_0/doc/html/mpi.html). mpich2 can be installed as a package using apt-get/yum/dnf/brew. However, Boost must be compiled from source if MPI is to be used. Compilation instructions for boost/boost-mpi can be found [here](http://kratos-wiki.cimne.upc.edu/index.php/How_to_compile_the_Boost_if_you_want_to_use_MPI).

Once the necessary libraries are installed, clone the repository or download the zip file and extract the contents and change into the CD. There are two Makefiles, one for a typical Fedora installation and one for a Ubuntu installation. The non-MPI installation has also been tested on MacOS-Yosemite and it probably will compile under MinGW on Windows with minor changes to the Makefile. Once in the src directory.

    cp Makefile.Ubuntu Makefile
    make clean; make <FLAGS>
  
If MPI is desired the <FLAGS> should include USEMPI=1.
For MACOS <FLAGs> should include MACOS=1

After compilation you can run the provided test scripts runFastBMA.sh and runfastBMAMPI.sh which should infer a network from a 100 variable time series. This should take less than a second. 

###Installation using Docker#
A dockerFile is included starting with from an Ubuntu image with OpenBLAS. Unfortunately, for the version of MPI used, mpich2, the boost libraries must be compiled from source in order for boost-mpi to work properly. So it may take a while to generate the initial image. 
###Installation using R#
fastBMA has been incorporated into the networkBMA package. However, the package is in beta and can only be installed on Linux systems. A demo is available [here](https://github.com/lhhunghimself/fastBMARdemo) This is due to the requirement for OpenBLAS which is difficult to provide in R without recompiling R on Windows. We plan to provide a version without OpenBLAS for Bioconductor in the future. There is also no MPI support for the R version but multithreading is available through openMP. 
##Sample usage#
Sample usage is provided in the two shell scripts, one for MPI and one for OpenMP only. fastBMA is very customizable with a myriad of flags. To get a list of flags and a summary of what they do type

    fastBMA --help
##Data File Formats#
fastBMA has been extensively tested with time series data. Examples of the data files are provided in the package.

##Use with MPI#
MPI jobs are run using mpiexec or mpirun. Documentation on running MPI apps can be found [here](https://www.mpich.org/documentation/guides/)
OpenMP can be used at the same time by using the -n flag to set the number of cores used. For some reason, even for single machines, MPI is considerably more efficient that OpenMP for managing separate fastBMA threads. This is despite trying several different approaches to improve OpenMP performance.

##Use with R#
A demo is available [here](https://github.com/lhhunghimself/fastBMARdemo). 
##Algorithm Documentation#
There are 4 major algorithmic improvements that increase the speed, scalability and accuracy of fastBMA relative to its predecessor ScanBMA
1.	Parallel and distributed implementation
2.	Faster regression by updating previous solutions
3.	Probabilistic hashing
4.	Post-processing with transitive reduction

These are described in detail in an upcoming paper.

##Benchmarks#
