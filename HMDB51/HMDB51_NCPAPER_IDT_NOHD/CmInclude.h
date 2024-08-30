#pragma once
#pragma warning(disable: 4996)
#pragma warning(disable: 4995)
#pragma warning(disable: 4805)
#pragma warning(disable: 4267)
#define _CRT_SECURE_NO_DEPRECATE

//#include <SDKDDKVer.h>
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers

#ifdef _WIN32 // Windows version
	#include <xstring>
	#include <atlstr.h>
	#include <atltypes.h>
#endif

#include <stdio.h>
#include <assert.h>
#include <string>
//#include <xstring>
#include <map>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <exception>
#include <cmath>
#include <time.h>
#include <set>
#include <queue>
#include <list>
#include <limits>
#include <fstream>
#include <sstream>
#include <random>
//#include <atlstr.h>
//#include <atltypes.h>
#include <omp.h>
#include <strstream>
using namespace std;


//#include <Eigen/Dense> //Eigen 3
#ifdef _DEBUG
#define lnkLIB(name) name "d"
#else
#define lnkLIB(name) name
#endif


#include <opencv2/opencv.hpp> 
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#define cvLIB(name) lnkLIB("opencv_" name CV_VERSION_ID)

#pragma comment( lib, cvLIB("core"))
#pragma comment( lib, cvLIB("imgproc"))
#pragma comment( lib, cvLIB("highgui"))
using namespace cv;


#ifdef _WIN32 // Windows version
	// CmLib Basic coding help
	#include "./Basic/CmDefinition.h"
	#include "./Basic/CmTimer.h"
	#include "./Basic/CmFile.h"
	// For illustration
	//#include "./Illustration/CmEvaluation.h"
	// Clustering algorithms
	#include "./Cluster/CmAPCluster.h"
	#include "./Cluster/CmColorQua.h"
	#include "./Cluster/CmGMM.h"
#else // Linux version
	// CmLib Basic coding help
	#include "CmDefinition.h"
	#include "CmTimer.h"
	#include "CmFile.h"
	// For illustration
	//#include "CmEvaluation.h"
	// Clustering algorithms
	#include "CmAPCluster.h"
	#include "CmColorQua.h"
	#include "CmGMM.h"
#endif


#define ToDo printf("To be implemented, %d:%s\n", __LINE__, __FILE__)

extern bool dbgStop;
#define DBG_POINT if (dbgStop) printf("%d:%s\n", __LINE__, __FILE__);
