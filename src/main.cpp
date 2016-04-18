#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <tclap/CmdLine.h>
#include "fastBMA.hpp"
#ifdef USEMPI
#include "bmaMPI.hpp"
#else
#include "bmaNoMPI.hpp"
#endif
//main function reads from stdin

using namespace std;
 
int tvalue_to_tindex(int tvalue,int *tvalues,int ntimes);
void current_utc_time(struct timespec *ts);
int main(int argc, char *argv[]){
	#ifdef USEMPI
	 mpi::environment env{argc, argv};
	 mpi::communicator(world);
	#endif
  string edgeListFile="";
  string priorsListFile="";
  string matrixFile="";
  string priorsMatrixFile="";
  string residualsFile="";
  string evalSubsetString="";
  unsigned int nThreads=1;
  float edgeMin=.5,pruneEdgeMin=0,pruneFilterMin;
  float gPrior=0;
  float uPrior=UNIFORM_PRIOR;
  float oddsRatio=10000.0f;
  float timeout=0;
  int optimizeBits=0;
  int maxOptimizeCycles=20;
  int nVars=0;
  float edgeTol=-1;
  bool timeSeries=0;
  bool noHeader=0;
  bool noPrune=0;
  bool showPrune=0;
  bool selfie=0;
  bool singlePrecision=false;
  bool verbose=0;
  bool useResiduals=0;
  bool rankOnly=0;
  bool dynamicScheduling=0;
		try {  
   using namespace TCLAP;
	  CmdLine cmd("fastBMA flags", ' ', "0.9");
	  ValueArg<unsigned int> nThreadsArg ("n","nThreads","Number of threads to use",false,1,"unsigned int");  
	  ValueArg<uint32_t> maxKeptModelsArg ("","maxKeptModels","Maximum umber of models kept",false,100000,"unsigned int");
	  ValueArg<uint32_t> maxActiveModelsArg ("","maxActiveModels","Maximum number of active models to add and delete variables from",false,100000,"unsigned int");
	  
	  ValueArg<string> edgeListFileArg("e","edgeList","edgeList file with network to be pruned",false,"","string");
	  ValueArg<string> priorsListArg ("p","priorsList","priors file in list form",false,"","string");
	  ValueArg<string> priorsMatrixArg ("","priorsMatrix","priors file in matrix form",false,"","string");
	  ValueArg<string> matrixArg ("m","matrix","matrix file",false,"","string");	  
	  ValueArg<string>residualsFileArg ("","residualsFile","save transformed timeSeries to file",false,"","string");
	  ValueArg<string>indicesArg ("i","indicesSubset","subset of indicesm to be evaluated in format <start>:<end> eg. 0:0 or 3:5 ",false,"","string");
	  ValueArg<float> edgeMinArg ("","edgeMin","minimum posterior prob (0-1) to draw edge",false,0.5f,"float");
	  ValueArg<float> pruneEdgeMinArg ("","pruneEdgeMin","minimum posterior prob (0-1) before an edge will be eliminated",false,0,"float");
	  ValueArg<float> pruneFilterMinArg ("","pruneFilterMin","minimum posterior prob (0-1) before an edge will be included in the network to be pruned",false,0,"float");
	  ValueArg<float> gPriorArg ("g","gPrior","Zellner g prior",false,0,"float");
	  ValueArg<float> edgeDensityArg ("","edgeDensity","expected number of edges divide by total possible number of edges",false,-1,"float");
	  ValueArg<float> oddsRatioArg ("","oddsRatio","odds ratio - determines the size of Occam's window",false,10000.0f,"float");
	  ValueArg<int> optimizeBitsArg ("o","optimizeBits","optimize bits - determines how accurate the optimization of g is",false,0,"int");	  
	  ValueArg<int> maxOptimizeCyclesArg ("","maxOptimizeCycles","determines maximum number of cycles to search for optimal g",false,20,"int");
	  ValueArg<int> nVarsArg ("v","nVars","the number of variables analyzed",false,0,"int");
	  ValueArg<int> nVarArg ("","nVar","the number of variables analyzed",false,0,"int");
	  ValueArg<float> edgeTolArg("","edgeTol","the error tolerance for determining whether an indirect path is as good as a direct path",false,-1,"float");
	  ValueArg<float> timeoutArg ("","timeout","maximum number of seconds for the regression before it stops the search",false,0,"int");
   SwitchArg rankOnlyArg("","rankOnly","use priors to rank variables but uniform prior otherwise",cmd,false);
	  SwitchArg noPruneArg ("","noPrune","indicate that we do not try to prune redundant edges",cmd,false);	  
	  SwitchArg showPruneArg ("","showPrune","keep edges and add an extra column showing redundant edges" ,cmd,false);
	  SwitchArg noHeaderArg ("","noHeader","indicates there is a no header in the data",cmd,false);
	  SwitchArg timeSeriesArg ("t","timeSeries","treat matrix as time series ",cmd,false);	  
	  SwitchArg singlePrecisionArg ("s","singlePrecision","use single precision instead of double precision ",cmd,false);	  
	  SwitchArg timerArg ("","time","shows execution time in seconds not including file i/o ",cmd,false);
	  SwitchArg selfArg ("","self","output edges to self ",cmd,false);
	  SwitchArg useResidualsArg ("","residuals","transform timeSeries to minimize impact of self-regulation",cmd,false);
	  SwitchArg verboseArg("","verbose","show more parameter information",cmd,false);
	  SwitchArg dynamicArg("d","dynamic","use dynamic scheduling for MPI",cmd,false);	  
	  cmd.add(nThreadsArg);
	  cmd.add(priorsListArg),
	  cmd.add(priorsMatrixArg),
	  cmd.add(gPriorArg),
	  cmd.add(edgeListFileArg),
	  cmd.add(matrixArg); 
	  cmd.add(edgeMinArg);
	  cmd.add(pruneEdgeMinArg);
	  cmd.add(pruneFilterMinArg);
	  cmd.add(edgeDensityArg);	  
	  cmd.add(oddsRatioArg);
	  cmd.add(nVarsArg);	  
	  cmd.add(nVarArg);
	  cmd.add(optimizeBitsArg);	  
	  cmd.add(maxOptimizeCyclesArg);
	  cmd.add(edgeTolArg);
	  cmd.add(residualsFileArg);
	  cmd.add(maxKeptModelsArg);
	  cmd.add(maxActiveModelsArg);
	  cmd.add(indicesArg);
	  cmd.add(timeoutArg);
	  cmd.parse( argc, argv );

	  // Get the value parsed by each arg
	  priorsListFile= priorsListArg.getValue();
	  priorsMatrixFile= priorsMatrixArg.getValue();
  	matrixFile= matrixArg.getValue();
  	residualsFile=residualsFileArg.getValue();
   nThreads=nThreadsArg.getValue(); 
   noHeader=noHeaderArg.getValue();   
   noPrune=noPruneArg.getValue();   
   showPrune=showPruneArg.getValue();
   selfie=selfArg.getValue();
			edgeMin=edgeMinArg.getValue();
			edgeTol=edgeTolArg.getValue();
			gPrior=gPriorArg.getValue();			
		 oddsRatio=oddsRatioArg.getValue();
			singlePrecision=singlePrecisionArg.getValue();
			timeSeries=timeSeriesArg.getValue();
			pruneEdgeMin=pruneEdgeMinArg.getValue();
			pruneFilterMin=pruneFilterMinArg.getValue();			
		 gMaxActiveModels=maxActiveModelsArg.getValue();		 
		 gMaxKeptModels=maxKeptModelsArg.getValue();
   evalSubsetString=indicesArg.getValue();
			{
				//either nVar or nVars is acceptable - both are not
			 int temp1=nVarsArg.getValue();
			 int temp2=nVarArg.getValue();
			 if(temp1 && temp2){
					cerr << "only one of flags nVar or nVars can be used" << endl;
				 exit(0);
				}
				if(temp1)nVars=temp1;
				if(temp2)nVars=temp2;	
			}
			verbose=verboseArg.getValue();
			optimizeBits=optimizeBitsArg.getValue();			
			maxOptimizeCycles=maxOptimizeCyclesArg.getValue();
			edgeListFile=edgeListFileArg.getValue();
			useResiduals=useResidualsArg.getValue();
			rankOnly=rankOnlyArg.getValue();
			dynamicScheduling=dynamicArg.getValue();
   timeout=timeoutArg.getValue();
			if(edgeDensityArg.getValue() >= 0){
				uPrior=edgeDensityArg.getValue();
			}	
			gtime=timerArg.getValue();
			if(edgeListFile == "" && matrixFile == ""){
				cerr << "must provide either an edgelist with a network to be pruned or a data file to infer a network from" << endl;
				exit(0);
			}	
			if(edgeListFile != "" && matrixFile != ""){
				cerr << "Both edgelist and data file given - only one is needed" << endl;
				exit(0);
			}
			if(noPrune && showPrune){
				cerr << "Warning: both noPrune and showPrune chosen - showPrune flag ignored"<<endl;
				exit(0); 
			}
			if(priorsListFile != "" && priorsMatrixFile != ""){
				cerr << "two files specified for priors - they can be in a list form i.e -p --priorList OR in a matrix form --priorMatrix"<<endl;
				exit(0);
			}		
	 // Do what you intend. 
	 #ifdef USEMPI
	 if(verbose && world.rank() ==0){
		#else
	 if(verbose){		
		#endif
	  cerr << "Input Parameters: "<<endl;
	  if(priorsListFile != "")cerr << "priorsList: " << priorsListFile << endl;
		 if(priorsMatrixFile != "") cerr << "priorsMatrix: " << priorsMatrixFile << endl;
		 cerr << "matrixFile: " << matrixFile << endl;		 
	  cerr << "nThreads: " << nThreads << endl;
	  cerr << "minimum edgeWeight: " << edgeMin << endl;
	  if(gPrior) cerr << "g: " << gPrior << endl;
	  else cerr << "g set to number of sqrt(nsamples)"<<endl;
	  if(edgeTol >= 0 && !noPrune){
				cerr<< "edgeTol: " <<edgeTol <<endl; 
			}	
	  if(optimizeBits){
				cerr << "g optimized: "<< optimizeBits << " bits" <<endl;
				cerr << "g maxOptimizeCycles: " <<maxOptimizeCycles <<endl;
			}
			if(timeSeries){
				cerr << "matrix interpreted as time series"<<endl;	
			}		 
	  cerr << "Uninformed Prior: " << uPrior << endl;
	  cerr << "oddsRatio: " << oddsRatio << endl;
	  if(selfie) cerr << "Show edges to same node" <<endl;
	  else cerr << "Do not show edges to same node" << endl;
	  if(noHeader){
	   cerr << "noHeader: " << noHeader << endl;
		 }
		 if(nVars){
				cerr<< "nVars: " <<nVars<<endl;
			}
			else{
				cerr << "All variables used" <<endl;
			}		
		 if(verbose){
				cerr <<"verbose on" <<endl;
			} 	 
		 if(noPrune){
	   cerr << "Do no remove redundant direct edges " << endl;
		 }
		 else if(showPrune){
				 cerr << "Do no remove but indicate redundant direct edges " << endl;
			}	
		 else {
		 	cerr << "Remove redundant direct edges " << endl;
		 }
		} 	
	} 
	catch (TCLAP::ArgException &e)  // catch any exceptions
	{ cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
 //constants
 if(!pruneFilterMin){
		pruneFilterMin=edgeMin/4.0;
	}
	if(noPrune){
		pruneFilterMin=edgeMin;
	}	
 const float twoLogOR=2.0 * log(oddsRatio);
	if(matrixFile != ""){
		if(singlePrecision){
			findEdges<float>(evalSubsetString,matrixFile,priorsMatrixFile,priorsListFile,residualsFile,timeSeries,useResiduals,dynamicScheduling,noHeader,rankOnly,selfie,showPrune,noPrune,nVars,nThreads,optimizeBits,maxOptimizeCycles,twoLogOR,gPrior,pruneEdgeMin,pruneFilterMin,edgeMin,edgeTol,uPrior,timeout);
			
		}
		else{	
			findEdges<double>(evalSubsetString,matrixFile,priorsMatrixFile,priorsListFile,residualsFile,timeSeries,useResiduals,dynamicScheduling,noHeader,rankOnly,selfie,showPrune,noPrune,nVars,nThreads,optimizeBits,maxOptimizeCycles,twoLogOR,gPrior,pruneEdgeMin,pruneFilterMin,edgeMin,edgeTol,uPrior,timeout);
		}
 }
	else{
		#ifdef USEMPI
		if(world.rank() >0)return(0);
		#endif
	 vector<string> headers;
		//edgeList
		EdgeList *edgeList=readEdgeListFile(edgeListFile,headers); 
		if(gtime){
		 current_utc_time(&start_time);
  } 	 
		EdgeList nonSelfList=edgeList->nonSelfList();
		if(!pruneFilterMin)pruneFilterMin=edgeMin/4.0;
  nonSelfList.prune_edges(pruneFilterMin,edgeTol);
  if(selfie)edgeList->printSelfEdges(edgeMin,headers,showPrune,0);
  nonSelfList.printEdges(edgeMin,headers,selfie,showPrune,pruneEdgeMin);
		if(gtime){
		 current_utc_time(&end_time);
   cerr << "elapsed time: "<< get_elapsed_time(&start_time, &end_time) << " seconds"<<endl;
  }
		if(edgeList)delete edgeList;			
	}
}

