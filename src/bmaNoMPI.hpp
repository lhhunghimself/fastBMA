
template <class T> void findEdges(string evalSubsetString,string matrixFile,string priorsMatrixFile,string priorsListFile,string residualsFile,bool timeSeries,bool useResiduals,bool dynamicScheduling,bool noHeader,bool rankOnly,bool selfie,bool showPrune,bool noPrune,int nVars,int nThreads,int optimizeBits,int maxOptimizeCycles,float twoLogOR,float gPrior,float pruneEdgeMin,float pruneFilterMin,float edgeMin,float edgeTol,float uPrior,float timeout);
template <class T> void findDenseEdges(string evalSubsetString,string matrixFile,string priorsMatrixFile,string priorsListFile,string residualsFile,bool timeSeries,bool useResiduals,bool dynamicScheduling,bool noHeader,bool rankOnly,bool selfie,bool showPrune,bool noPrune,int nVars,int nThreads,int optimizeBits,int maxOptimizeCycles,float twoLogOR,float gPrior,float pruneEdgeMin,float pruneFilterMin,float edgeMin,float edgeTol,float uPrior,float timeout);
template <class T> void findEdges(string evalSubsetString,string matrixFile,string priorsMatrixFile,string priorsListFile,string residualsFile,bool timeSeries,bool useResiduals,bool dynamicScheduling,bool noHeader,bool rankOnly,bool selfie,bool showPrune,bool noPrune,int nVars,int nThreads,int optimizeBits,int maxOptimizeCycles,float twoLogOR,float gPrior,float pruneEdgeMin,float pruneFilterMin,float edgeMin,float edgeTol,float uPrior,float timeout){
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
	if(priorsMatrixFile != "")	rProbs=readPriorsMatrix<T>(priorsMatrixFile,nGenes);			
	//probs are directly read in if in matrix format - otherwise the priorsList is passed
	//matrix form is only if the complete set of priors (all possible pairs) is meant to be passed
	//use the priorsList to pass a partial set
	 
 const T uniform_prob=uPrior;
	if(timeSeries)data=readTimeData<T>(matrixFile,headers,nGenes,nSamples,nRows,nTimes,noHeader,useResiduals,residualsFile);
	else data=readData<T>(matrixFile,headers,nGenes,nSamples,noHeader);
	//now we that we know number of genes we set evalSubset to the identity set if no subset is defined 
 if(!evalSubset.size()){
		for(int i=0;i<nGenes;i++){
			evalSubset.push_back(i);
		}	
 }
	if(!timeSeries)nRows=nSamples;	
	if(priorsMatrixFile == "" && priorsListFile != ""){
		rProbs=new T*[nGenes];
		rProbs[0]=new T [nGenes*nGenes];
		for (int i=1;i<nGenes;i++){
			rProbs[i]=rProbs[i]+nGenes;
		}
		readPriorsList(priorsListFile,headers,rProbs,uniform_prob);
 }
	if(gtime)current_utc_time(&start_time);
	
	//initialize variables
	T g= (gPrior)? gPrior : sqrt((double)nRows);
	T *A=new T [(nGenes+1)*nRows];
	T *ATA=new T[(nGenes+1)*(nGenes+1)];
 const int ATAldr=nGenes+1;
 const int Aldr=nRows;
	initRegressParms<T>(A,ATA,data,nGenes,nRows,nSamples,nTimes,nVars,nThreads,timeSeries);
 vector <int> parents,children,edgeCounts;  
 vector <double> weights;  
	//thread variables for openMP
	int **parentsSlice=new int*[nThreads];
	double **weightsSlice=new double*[nThreads];
	parentsSlice[0]=new int[nGenes*nThreads];
	weightsSlice[0]=new double[nGenes*nThreads];	 
 vector<int> *thParents=new vector<int>[nThreads];
	vector<double> *thWeights=new vector<double> [nThreads];
	vector<int> *thChildren=new vector<int> [nThreads];
	vector<int> *thEdgeCounts=new vector<int> [nThreads];
	for(int i=1;i<nThreads;i++){
		parentsSlice[i]=parentsSlice[i-1]+nGenes;
		weightsSlice[i]=weightsSlice[i-1]+nGenes;
	}	
 #pragma omp parallel for schedule(dynamic) num_threads(nThreads) 
	for(int k=0;k<evalSubset.size();k++){
		const int th=omp_get_thread_num();
		int nEdges=findRegulators(g,optimizeBits,maxOptimizeCycles,uniform_prob,twoLogOR,nVars,nThreads,rankOnly,evalSubset[k],data,rProbs,parentsSlice[th] ,weightsSlice[th],A,ATA, Aldr,ATAldr, nGenes,nRows,nSamples,nTimes,timeout);
		int goodEdges=0;
		for(int i=0;i<nEdges;i++){
			if(weightsSlice[th][i] > pruneFilterMin){
				thParents[th].push_back(parentsSlice[th][i]);
				goodEdges++;
				thWeights[th].push_back(weightsSlice[th][i]);
			}
		}
		if(goodEdges){
			thChildren[th].push_back(evalSubset[k]);
			thEdgeCounts[th].push_back(goodEdges);
		}
	}
	set<pair<pair<int,int>,float>> edgeSet;	 
	//reduction of thread variables
	for(int th=0;th<nThreads;th++){
		int n=0;
  for(int j=0;j<thChildren[th].size();j++){
			const int c=thChildren[th][j];
			for(int k=0;k<thEdgeCounts[th][j];k++){
			 edgeSet.insert(make_pair(make_pair(thParents[th][n],c),(float)thWeights[th][n]));
			 n++;
			}
		}
		thParents[th].clear();thChildren[th].clear();thWeights[th].clear();thEdgeCounts[th].clear();
	} 
	delete[]thParents;	    
	delete[]thChildren;		   
	delete[]thWeights;		   
	delete[]thEdgeCounts;
	delete[]parentsSlice[0];
	delete[]parentsSlice;
	delete[]weightsSlice[0];
	delete[]weightsSlice;
	if(rProbs){
		delete[]rProbs[0];
		delete[]rProbs;
	}	
	EdgeList edgeList(nGenes,edgeSet);
 if(!noPrune){
		EdgeList nonSelfList=edgeList.nonSelfList();
  nonSelfList.prune_edges(pruneFilterMin,edgeTol);
  if(selfie)edgeList.printSelfEdges(edgeMin,headers,showPrune,0);
  nonSelfList.printEdges(edgeMin,headers,selfie,showPrune,pruneEdgeMin);
	}
	else{
	 edgeList.printEdges(edgeMin,headers,selfie,showPrune,pruneEdgeMin);
	}
	if(gtime){
		current_utc_time(&end_time);
  cerr << "elapsed time: "<< get_elapsed_time(&start_time, &end_time) << " seconds"<<endl;
     //cerr << "hash time: " << hashTime <<endl;
 }
 delete[] hashLUT;
 delete[] A;
	delete[] ATA;
	delete[] data[0];
	delete[] data;			   
}

