bool doneEarlier(int index,int nGenes,int rank,bool *doneMatrix);
template <class T> void findEdges(string evalSubsetString,string matrixFile,string priorsMatrixFile,string priorsListFile,string residualsFile,bool timeSeries,bool useResiduals,bool dynamicScheduling,bool noHeader,bool rankOnly,bool selfie,bool showPrune,bool noPrune,int nVars,int nThreads,int optimizeBits,int maxOptimizeCycles,float twoLogOR,float gPrior,float pruneEdgeMin,float pruneFilterMin,float edgeMin,float edgeTol,float uPrior,float timeout);

template <class T> void findEdges(string evalSubsetString,string matrixFile,string priorsMatrixFile,string priorsListFile,string residualsFile,bool timeSeries,bool useResiduals,bool dynamicScheduling,bool noHeader,bool rankOnly,bool selfie,bool showPrune,bool noPrune,int nVars,int nThreads,int optimizeBits,int maxOptimizeCycles,float twoLogOR,float gPrior,float pruneEdgeMin,float pruneFilterMin,float edgeMin,float edgeTol,float uPrior,float timeout){
	
 namespace mpi = boost::mpi;
 mpi::communicator(world);
	vector<string> headers;
	vector<uint32_t>evalSubset;
	if(evalSubsetString != ""){
		uint32_t start,end;
		if(sscanf(evalSubsetString.c_str(),"%u:%u",&start,&end) != 2){
			cerr << "format of subset indices is <start>:<finish> instead the input was" << endl;
			cerr <<evalSubsetString << endl;
			exit(0); 
		}
		for(int i=start;i<=end;i++){
			evalSubset.push_back(i);	
		}
	}
	T **rProbs=0,**data=0;
	int nGenes=0,nRows=0,nTimes=0,nSamples=0;
	if(priorsMatrixFile != ""){
		//probs are directly read in if in matrix format - otherwise the priorsList is passed
		//matrix form is only if the complete set of priors (all possible pairs) is meant to be passed
		//use the priorsList to pass a partial set
		if(world.rank() ==0)rProbs=readPriorsMatrix<T>(priorsMatrixFile,nGenes);
		//broadcast arraySize
		broadcast(world,nGenes,0);
		if(world.rank() !=0){
			rProbs=new T*[nGenes];
			rProbs[0]=new T[nGenes*nGenes];
			for(int k=1;k<nGenes;k++)rProbs[k]=rProbs[k-1]+nGenes;
		}
		broadcast(world,rProbs[0],nGenes*nGenes,0);						
	}
 const T uniform_prob=uPrior;
 if(world.rank() ==0){
		if(timeSeries)data=readTimeData<T>(matrixFile,headers,nGenes,nSamples,nRows,nTimes,noHeader,useResiduals,residualsFile);
		else data=readData<T>(matrixFile,headers,nGenes,nSamples,noHeader);
	}
	broadcast(world,nGenes,0);
	broadcast(world,nSamples,0); //nRows == nSamples for non-timeSeries data
	 //now we that we know number of genes we set evalSubset to the identity set if no subset is defined 
 if(!evalSubset.size()){
		for(int i=0;i<nGenes;i++){
			evalSubset.push_back(i);
		}	
 }
	if(timeSeries){ 	
	 broadcast(world,nRows,0);
  broadcast(world,nTimes,0);
	}
	else nRows=nSamples;
 if(world.rank() !=0){
		data=new T*[nGenes];
		data[0]=new T[nSamples*nGenes];
	}	
 broadcast(world,data[0],nGenes*nSamples,0);
 if(world.rank() !=0){
		for(int k=1;k<nGenes;k++){
		 data[k]=data[k-1]+nSamples;
		}
	}	
	if(priorsMatrixFile == "" && priorsListFile != ""){
		rProbs=new T*[nGenes];
		rProbs[0]=new T[nGenes*nGenes];
		for (int i=1;i<nGenes;i++){
			rProbs[i-1]=rProbs[i]+nGenes;
		}
		if(world.rank()==0)readPriorsList(priorsListFile,headers,rProbs,uniform_prob);
		broadcast(world,rProbs[0],nGenes*nGenes,0);
	}
	if(gtime)current_utc_time(&start_time);
	//initialize variables
	T g= (gPrior)? gPrior : sqrt((double)nRows);
	T *A=new T [(nGenes+1)*nRows];
	T *ATA=new T[(nGenes+1)*(nGenes+1)];
 const int ATAldr=nGenes+1;
 const int Aldr=nRows;
	initRegressParms<T>(A,ATA,data,nGenes,nRows,nSamples,nTimes,nVars,nThreads,timeSeries);
 vector <float> children;
	vector <int> indexLock(evalSubset.size());

	//calculate start and end - have to wait until the regressio
	vector <pair<int,int>> indices(world.size());
	for(int i=0;i<world.size();i++){
		indices[i]=make_pair(0,-1);
	}
	int startIndex=0;
	int finishIndex=-1;	
	for(int i=0;i<=world.rank();i++){
		startIndex=finishIndex+1;
		if(startIndex >=evalSubset.size())break;
		if(i == world.size()-1)finishIndex=evalSubset.size()-1;
		else finishIndex=startIndex+(evalSubset.size()/(float)world.size()+.5)-1;
		if(finishIndex<startIndex)finishIndex=startIndex;
	 indices[i]=make_pair(startIndex,finishIndex);
	}
	//thread variables for openMP
	vector<bool> evaluated(nGenes); 
 double **weights=new double*[nGenes];	
 weights[0]=new double[nGenes*nGenes];
	for(int i=1;i<nGenes;i++){
		weights[i]=weights[i-1]+nGenes;
	}
	for(int i=0;i<nGenes;i++){
 	for(int j=0;j<nGenes;j++){
		 weights[i][j]=0;
		}
	}	
	
	if(dynamicScheduling){
		bool *locked=new bool[nGenes];	
	 bool *doneMatrix=new bool[nGenes*world.size()];
	 memset(locked,0,sizeof(bool)*nGenes);	
	 memset(doneMatrix,0,sizeof(bool)*nGenes*world.size());
			//may have race conditions but the design minimizes that and we remove duplicates at the end
		//use when the calculatons will be very long 
		#pragma omp parallel num_threads(nThreads) 
		{
			char inbuffer[64];
			const int th=omp_get_thread_num();
			for(int k=indices[world.rank()].first+th;k<=indices[world.rank()].second;k+=nThreads){
				if(locked[k]){
					//get messages and break
					#pragma omp critical
					while(boost::optional<mpi::status> rv=world.iprobe(mpi::any_source,mpi::any_tag)){
						world.irecv(rv->source(),rv->tag(),inbuffer);
						doneMatrix[rv->tag()]=1;
						locked[rv->tag()%nGenes]=1;
					}
					continue;		
			 }	
			 int nEdges;
			 #pragma omp critical
				while(boost::optional<mpi::status> rv=world.iprobe(mpi::any_source,mpi::any_tag)){
					world.irecv(rv->source(),rv->tag(),inbuffer);
					doneMatrix[rv->tag()]=1;
					locked[rv->tag()%nGenes]=1;
				}
				if(!locked[k]){
					locked[k]=1;
					#pragma omp critical
					for(int i=0;i<world.size();i++){
						if(i!=world.rank()){
							world.isend(i,k+i*nGenes);
						}
				 }
					#pragma omp critical
					while(boost::optional<mpi::status> rv=world.iprobe(mpi::any_source,mpi::any_tag)){
						world.irecv(rv->source(),rv->tag(),inbuffer);
						doneMatrix[rv->tag()]=1;
						locked[rv->tag()%nGenes]=1;
					}
				 nEdges=findRegulators(g,optimizeBits,maxOptimizeCycles,uniform_prob,twoLogOR,nVars,nThreads,rankOnly,evalSubset[k],data,rProbs,0,weights[evalSubset[k]],A,ATA, Aldr,ATAldr, nGenes,nRows,nSamples,nTimes,timeout);
				 evaluated[evalSubset[k]]=1;
				}
			}
		}
		for(int n=world.rank()+1;n<world.rank()+world.size();n++){
			//steal jobs from the others if possible
	 		const int r=n%world.size();
	 		#pragma omp parallel num_threads(nThreads) 
				{
					const int th=omp_get_thread_num();
					char inbuffer[64];
	 		 for(int k=indices[r].second-th;k>=indices[r].first;k-=nThreads){
						if(locked[k]){
							//get messages and break
							#pragma omp critical
							while(boost::optional<mpi::status> rv=world.iprobe(mpi::any_source,mpi::any_tag)){
								world.irecv(rv->source(),rv->tag(),inbuffer);
								doneMatrix[rv->tag()]=1;
							 locked[rv->tag()%nGenes]=1;
							}
							continue;		
						}	
			   int nEdges;
			   #pragma omp critical
						while(boost::optional<mpi::status> rv=world.iprobe(mpi::any_source,mpi::any_tag)){
							world.irecv(rv->source(),rv->tag(),inbuffer);
							doneMatrix[rv->tag()]=1;
							locked[rv->tag()%nGenes]=1;
						}
						if(!locked[k]){
							locked[k]=1;
							#pragma omp critical
							for(int i=0;i<world.size();i++){
							 if(i!=world.rank()){
									world.isend(i,k+i*nGenes);
								}
							}
							#pragma omp critical
							while(boost::optional<mpi::status> rv=world.iprobe(mpi::any_source,mpi::any_tag)){
								world.irecv(rv->source(),rv->tag(),inbuffer);
								doneMatrix[rv->tag()]=1;
							 locked[rv->tag()%nGenes]=1;
							}
				 	 nEdges=findRegulators(g,optimizeBits,maxOptimizeCycles,uniform_prob,twoLogOR,nVars,nThreads,rankOnly,evalSubset[k],data,rProbs,0 ,weights[evalSubset[k]],A,ATA, Aldr,ATAldr, nGenes,nRows,nSamples,nTimes,timeout);
				 	 evaluated[evalSubset[k]]=1;
				  }
				 }
				}
			}
			//cerr << world.rank() <<" reached barrier "<<endl;
			world.barrier();	//wait for all threads to finish
		//check messages one more time
		 {
				char inbuffer[64];
		  while(boost::optional<mpi::status> rv=world.iprobe(mpi::any_source,mpi::any_tag)){
		  	world.irecv(rv->source(),rv->tag(),inbuffer);
		  	doneMatrix[rv->tag()]=1;
		  }
		 }
		 //cerr << world.rank() <<" checked messages "<<endl;
		 //now check each
		 for(int i=0;i<nGenes;i++){
		  if(evaluated[i]){
				 if(doneEarlier(i,nGenes,world.rank(),doneMatrix)){
						evaluated[i]=0;
					}		
				}	
	  }
 //cerr << world.rank() <<" checked done"<<endl;
 			 world.barrier();
			delete[] locked;
			delete[] doneMatrix;		
		}
	 else{
   #pragma omp parallel for schedule(dynamic) num_threads(nThreads) 
			for(int k=indices[world.rank()].first;k<=indices[world.rank()].second;k++){
				const int th=omp_get_thread_num();
			 int nEdges;
			 nEdges=findRegulators(g,optimizeBits,maxOptimizeCycles,uniform_prob,twoLogOR,nVars,nThreads,rankOnly,evalSubset[k],data,rProbs,0,weights[evalSubset[k]],A,ATA, Aldr,ATAldr, nGenes,nRows,nSamples,nTimes,timeout);
			 evaluated[evalSubset[k]]=1;
		 }
		}
		vector<int> transferChildren;
		vector<float> transferWeights;
		int nEvaluated=0;
		for(int i=0;i<nGenes;i++){
			if(evaluated[i]){
				transferChildren.push_back(i);
				for (int j=0;j<nGenes;j++){
					transferWeights.push_back(weights[i][j]);
				}	
			}	
		}
	 delete[] weights[0];
		delete[] weights;
	 if(world.rank() ==0){
			vector<int> childrenSizes(world.size());
			vector<vector<float>>allWeights(world.size());
			vector<vector<int>>allChildren(world.size());
			//sizes of vectors not always transmitted properly - bug in boost::MPI?
			gather(world,(int)transferChildren.size(),childrenSizes,0);
   gather(world,transferChildren,allChildren,0);
   gather(world,transferWeights,allWeights,0);
			//make into connectivity matrix
			float **finalWeights=new float*[nGenes];	
   finalWeights[0]=new float[nGenes*nGenes];
   for(int i=1;i<nGenes;i++)finalWeights[i]=finalWeights[i-1]+nGenes;
   for(int i=0;i<nGenes;i++)
    for(int j=0;j<nGenes;j++)
     finalWeights[i][j]=0;
   for(int i=0;i<world.size();i++){
				for(int j=0;j<childrenSizes[i];j++){
					const int c=allChildren[i][j];
			 	memmove(finalWeights[c],&(allWeights[i][0])+(j*nGenes),nGenes*sizeof(float));
				}	
			}
	
	  EdgeList edgeList(nGenes,edgeMin/4.0,finalWeights);
	  delete[] finalWeights[0];
	  delete[] finalWeights;
   if(!noPrune){
		  EdgeList nonSelfList=edgeList.nonSelfList();
    nonSelfList.prune_edges(pruneFilterMin,edgeTol);
    if(selfie)edgeList.printSelfEdges(edgeMin,headers,showPrune,0);
   	 nonSelfList.printEdges(edgeMin,headers,selfie,showPrune,pruneEdgeMin);
	  	}
	  	else{
	    edgeList.printEdges(edgeMin,headers,selfie,showPrune,pruneEdgeMin);
			 }
			 if(gtime && world.rank()==0){
		   current_utc_time(&end_time);
     cerr << "elapsed time: "<< get_elapsed_time(&start_time, &end_time) << " seconds"<<endl;
     //cerr << "hash time: " << hashTime <<endl;
    }	   
			}
		 else{
    gather(world,(int)transferChildren.size(),0); 
    gather(world,transferChildren,0);
    gather(world,transferWeights,0);
			}
	  if(rProbs){
		  delete[]rProbs[0];
		  delete[]rProbs;
		 }
	 	delete[] hashLUT;
   delete[] A;
	  delete[] ATA;
	  delete[] data[0];
	  delete[] data;	

}

bool doneEarlier(int index,int nGenes,int rank,bool *doneMatrix){//checks index to see if it has been previously evaluated
	for(int i=0;i<rank;i++){
		if(doneMatrix[i*nGenes+index])return(1);
	}	
	return(0);
}	
	
