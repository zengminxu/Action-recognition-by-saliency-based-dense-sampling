#ifdef _WIN32 // Windows version
	#include "../CmInclude.h"
#else // Linux version
	#include "CmInclude.h"
	#include <math.h>
	#include <stdio.h>
	#include <stdlib.h>
	#include <sys/types.h>
	#include <unistd.h>
	#include <dlfcn.h>
#endif
#include "CmAPCluster.h"

CmAPCluster::CmAPCluster(void) : dlh(NULL)
{
#ifdef _WIN32 // Windows version
	if (!(dlh=LoadLibrary("apclusterwin.dll")))
		printf("LoadLibrary() failed: %d.\n", GetLastError()); 
#else // Linux version
	if (!(dlh=dlopen("./apclusterunix64.so", RTLD_LAZY))) {
		printf("%s\n",dlerror());}
#endif

	apoptions.cbSize = sizeof(APOPTIONS);
	apoptions.lambda = 0.9;
	apoptions.minimum_iterations = 1;
	apoptions.converge_iterations = 200;
	apoptions.maximum_iterations = 2000;
	apoptions.nonoise = 0;
	apoptions.progress = NULL; //callback; 
	apoptions.progressf = NULL;

#ifdef _WIN32 // Windows version
	apFun = (apcluster32)GetProcAddress(dlh, "apcluster32");
	kcFun = (kcenters32)GetProcAddress(dlh, "kcenters32");
	if (kcFun == NULL || apFun == NULL)
		printf("GetProcAddress() failed: %d\n", GetLastError());
#else // Linux version
	apFun = (int (*)(double*,unsigned int*, unsigned int*, unsigned int, int*, double*, APOPTIONS*))dlsym(dlh, "apcluster32");
	kcFun = (int (*)(double*,unsigned int*, unsigned int*, unsigned int, unsigned int, int*, double*, KCOPTIONS*))dlsym(dlh, "kcenters32");
	if (kcFun == NULL || apFun == NULL)
		printf("GetProcAddress() failed. \n");
#endif

}

CmAPCluster::~CmAPCluster(void)
{
#ifdef _WIN32
	FreeLibrary(dlh);
#else
	dlclose(dlh);
#endif
}

int CmAPCluster::callback(double *a, double *r, int N, int *idx, int I, double netsim, double dpsim, double expref, int iter)
{
	static double netsimOld = 0;
	if (netsimOld == netsim)
		printf(".");
	else
		printf(" %g ", netsim), netsimOld = netsim;

	return(0); /* 0=continue apcluster */
}

int CmAPCluster::ReMapIdx(vecI &mapIdx)
{
	int N = (int)mapIdx.size(), newCount = 0;
	map<int, int> idxCount, oldNewIdx;
	vecI newIdx(N);
	for (int i = 0; i < N; i++){
		if (idxCount.find(mapIdx[i]) == idxCount.end())
			oldNewIdx[mapIdx[i]] = newCount++, idxCount[mapIdx[i]]++;
		mapIdx[i] = oldNewIdx[mapIdx[i]];
	}
	return (int)idxCount.size();
}
