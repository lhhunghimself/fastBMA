FROM ogrisel/openblas
MAINTAINER lhhung
RUN apt-get update && apt-get install -y libmpich-dev wget g++ && wget https://sourceforge.net/projects/boost/files/boost/1.57.0/boost_1_57_0.tar.bz2/download\
    && tar -xjvf download && rm download -rf &&  apt-get remove -y wget && cd ./boost_1_57_0 && /bin/bash -c './bootstrap.sh' && echo 'using mpi ;' >> project-config.jam\
    && /bin/bash -c './b2 -j8 -target=shared,static; exit 0;' && /bin/bash -c './b2 install; exit 0;' && rm ./boost_1_57_0 -rf && apt-get remove -y g++
COPY src /src
WORKDIR /src
RUN apt-get install -y libmpich-dev g++ && make clean && cp Makefile-Ubuntu Makefile && make MPI=1 && cp fastBMA /bin/fastBMA && apt-get remove -y g++ nano
WORKDIR /
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
 
