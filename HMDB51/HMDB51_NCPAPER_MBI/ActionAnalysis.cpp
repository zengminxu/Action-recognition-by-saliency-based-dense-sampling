#include "ActionAnalysis.h"

using namespace std;

extern string spritLabel;	//使用DenseTrackStab.cpp的全局变量

// 多线程同时对HOG/HOF/MBH 各通道进行码本训练
#ifdef _WIN32 // Windows version
	volatile LONG semaphore_trainMultiChCodebook = 0;
	DWORD WINAPI trainMultiChCodebook( LPVOID lpParameter )
#else // Linux version
	sem_t semaphore_trainMultiChCodebook;
	static void *trainMultiChCodebook(void *lpParameter)
#endif
{	
	CodebookParam *lp = (CodebookParam*)lpParameter;

	// 准备训练码本
	int retries = 8;  
	int flags = KMEANS_PP_CENTERS; 
	TermCriteria tc(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 10, 0.001); 
	BOWKMeansTrainer bowTrainer(DICTIONARY_SIZE, tc, retries, flags);	

	// 写入每类动作指定大小的特征
	FileStorage fs;
	string cb_file = lp->resultPath + "bow" + lp->cbname + ".xml";
	fs.open(cb_file, FileStorage::WRITE);
	if( !fs.isOpened() )
	{
		cout << "Error: Could not open features file in trainMultiChCodebook()" << endl;
		delete lp;
#ifdef _WIN32 // Windows version
		// 结束提取特征，信号量-1
		_InterlockedDecrement( &semaphore_trainMultiChCodebook );
#else // Linux version
		sem_wait( &semaphore_trainMultiChCodebook );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}

	// 添加各通道的训练数据
	if( lp->cbname == "HOG" )
		bowTrainer.add(lp->featuresMat.colRange(0,96));
	else if( lp->cbname == "HOF" )
		bowTrainer.add(lp->featuresMat.colRange(96,204));
	else if( lp->cbname == "MBH" )
		bowTrainer.add(lp->featuresMat.colRange(204,396));

	// 训练HOG/HOF/MBH 码本
	Mat codebook = bowTrainer.cluster(); 
	
	// 保存生成的码本
	fs  << "codebook" << codebook;

	// 释放资源.....
	fs.release();
	delete lp;

#ifdef _WIN32 // Windows version
	// 结束提取特征，信号量-1
	_InterlockedDecrement( &semaphore_trainMultiChCodebook );
#else // Linux version
	sem_wait( &semaphore_trainMultiChCodebook );
	pthread_detach(pthread_self());
#endif
	
	return 0;
}


// 先读取待训练的特征向量，写入txt文件，作为码本的输入数据
#ifdef _WIN32 // Windows version
	volatile LONG semaphore_featuresFormEachAction = 0;
	DWORD WINAPI featuresFormEachAction( LPVOID lpParameter )
#else // Linux version
	sem_t semaphore_featuresFormEachAction;
	static void *featuresFormEachAction(void *lpParameter)
#endif
{	
	ThreadParam *lp = (ThreadParam*)lpParameter;

	// 先读取待训练的特征向量，存于bin文件，作为码本的输入数据
	ActionAnalysis action;
	Mat des, tmpMat;

	// 从目录下顺序读取所有特征描述符，并合并到一个矩阵中	
	multimap<string, string>::iterator iter;
	for(iter = lp->actionSet.begin(); iter != lp->actionSet.end(); iter++)
	{
		if( iter->first == lp->actionTypeStr )
		{
			string file = lp->featurePath + iter->second + ".bin";
			action.bin2Mat(file, tmpMat, "features");
			if( tmpMat.empty() ){
				cout << "Read Error: features is empty in " << file << endl;
				continue;		
			}

			// 每段视频，指定采样sampleNum=GMM_SAMPLE_NUM/videoNum个特征点，videoNum指每类动作的视频数
			if( tmpMat.rows <= lp->sampleNum )
				des.push_back(tmpMat);
			else
				action.randSampling(tmpMat, des, lp->sampleNum, "oneVideo");

		}//if iter->first
		tmpMat.release();
	}// for iter	

	// 写入每类动作指定大小的特征
	FileStorage fs;
	string action_files = lp->resultPath + lp->actionTypeStr + ".xml";
	fs.open(action_files, FileStorage::WRITE);
	if( !fs.isOpened() )
	{
		cout << "Error: Could not open features file in readFeaturesFormEachAction()" << endl;
		delete lp;
#ifdef _WIN32 // Windows version
		// 结束提取特征，信号量-1
		_InterlockedDecrement( &semaphore_featuresFormEachAction );
#else // Linux version
		sem_wait( &semaphore_featuresFormEachAction );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}
	fs << "features" << des;

	// 释放资源.....
	des.release();
	fs.release();
	delete lp;

#ifdef _WIN32 // Windows version
	// 结束提取特征，信号量-1
	_InterlockedDecrement( &semaphore_featuresFormEachAction );
#else // Linux version
	sem_wait( &semaphore_featuresFormEachAction );
	pthread_detach(pthread_self());
#endif
	
	return 0;
}


// 视频量化：BoF编码
#ifdef _WIN32 // Windows version
	volatile LONG semaphore_quantization = 0;
	DWORD WINAPI featuresQuantizationProcess( LPVOID lpParameter )
#else // Linux version
	sem_t semaphore_quantization;
	static void *featuresQuantizationProcess(void *lpParameter)
#endif
{
	ThreadParam *lp = (ThreadParam*)lpParameter;

	// 开始量化特征
	string hogcb_file = lp->resultPath + "bowHOG.xml";
	string hofcb_file = lp->resultPath + "bowHOF.xml";
	string mbhcb_file = lp->resultPath + "bowMBH.xml";
	FileStorage fscodebook, fscodebook2, fscodebook3;
	fscodebook.open( hogcb_file.c_str(), FileStorage::READ);
	fscodebook2.open(hofcb_file.c_str(), FileStorage::READ);
	fscodebook3.open(mbhcb_file.c_str(), FileStorage::READ);
	if( !fscodebook.isOpened() || !fscodebook2.isOpened() || !fscodebook3.isOpened() ){
		cout << "Error: Could not open codebook file in featuresQuantizationProcess()." << endl;
		delete lp;
#ifdef _WIN32 // Windows version
		// 结束提取特征，信号量-1
		_InterlockedDecrement( &semaphore_quantization );
#else // Linux version
		sem_wait( &semaphore_quantization );
		pthread_detach(pthread_self());
#endif		
		return 0;		
	}

	ActionAnalysis action;
	Mat hogcodebook, hofcodebook, mbhcodebook;
	fscodebook["codebook"] >> hogcodebook;
	fscodebook2["codebook"] >> hofcodebook;
	fscodebook3["codebook"] >> mbhcodebook;

	// 读取每个视频的XML特征描述符
	ifstream fsread;
	FileStorage fswrite;
	string feature_files = lp->featurePath + lp->filename + ".bin";
	// 写入每个视频量化后的特征
	string vidcode_files = lp->vidcodePath + lp->filename + "_quantization.xml";
	fsread.open(feature_files.c_str(), ios::in);
	fswrite.open(vidcode_files, FileStorage::WRITE);
	if(!fsread.is_open() || !fswrite.isOpened())
	{
		cout << "Error: Could not open feature/quantization file in featuresQuantizationProcess()." << endl;
		delete lp;
#ifdef _WIN32 // Windows version
		// 结束提取特征，信号量-1
		_InterlockedDecrement( &semaphore_quantization );
#else // Linux version
		sem_wait( &semaphore_quantization );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}

	//cout << "video quantization "<< vidcode_files << endl;	
	Mat code(1, DICTIONARY_SIZE, CV_32FC3, Scalar::all(0));	//针对HOG、HOF、MBH 这三种Codebook的组合
	const int channels = code.channels(); 
	// 每段视频统计一次
	switch(channels)
	{
	case 1:
		{
			Mat tmp;
			action.bin2Mat(feature_files, tmp, "features");	
			CV_Assert( !tmp.empty() );
			vector<DMatch> matches;
			Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );
			descriptorMatcher->match(tmp, hogcodebook, matches);
			for(int i = 0; i < matches.size(); i++)
				code.at<float>(0, matches[i].trainIdx)++;
			tmp.release();
			matches.clear();
			break;
		}// case 1
	case 3:
		{
			Mat tmp;				
			action.bin2Mat(feature_files, tmp, "features");	
			if( tmp.empty() )
			{
				cout << "Quantization Error: features is empty in " << feature_files << endl;
				// 视频对应的特征子标号
				fswrite << lp->actionTypeStr;
				// 将空的码字写入xml
				fswrite << tmp;
				fswrite.release();
				delete lp;
		#ifdef _WIN32 // Windows version
				// 结束提取特征，信号量-1
				_InterlockedDecrement( &semaphore_quantization );
		#else // Linux version
				sem_wait( &semaphore_quantization );
				pthread_detach(pthread_self());
		#endif		
				return 0;
			}

			// VQ: VectorQuantization
			//action.hardVote(tmp, code, hogcodebook, hofcodebook, mbhcodebook);

			// SA-k5
			int knn = 5;
			action.softAssignKnn(tmp, code, knn, hogcodebook, hofcodebook, mbhcodebook);

			tmp.release();
			break;
		}// case 3
	}// switch
	
	// 视频对应的特征子标号
	fswrite << lp->actionTypeStr;
	// 将视频编码后的码字写入xml
	fswrite << code;

	fsread.close();
	fswrite.release();
	fscodebook.release();
	fscodebook2.release();
	fscodebook3.release();
	hogcodebook.release();
	hofcodebook.release();
	mbhcodebook.release();
	code.release();
	delete lp;

#ifdef _WIN32 // Windows version
	// 结束提取特征，信号量-1
	_InterlockedDecrement( &semaphore_quantization );
#else // Linux version
	sem_wait( &semaphore_quantization );
	pthread_detach(pthread_self());
#endif
	
	return 0;
}

// 将特征文件、码本文件从bin转换为Mat格式
int ActionAnalysis::bin2Mat(string file, Mat& tmpMat, string exType)
{
	FILE *fx = fopen(file.c_str(), "rb");
	if( fx == NULL ){
		std::cout << "Error: Could not open file '" << file << "'." << std::endl;
		return 0;
	}		
	vector<float> vec;
	float tmpData = 0;
	while(fread(&tmpData,sizeof(float),1,fx)!=0){
		vec.push_back(tmpData);
	}
	fclose(fx);
	if( vec.size()%DIMENSION !=0 ){
		std::cout << "Error: wrong dimension! " << DIMENSION << "is required!" << file << std::endl;
		return -1;
	}	
	
	int rows = vec.size()/DIMENSION;
	Mat tmp(rows, DIMENSION, CV_32FC1, Scalar(0));
	for(int i = 0; i < tmp.rows; i++)
		for(int j = 0; j < tmp.cols; j++)
			tmp.at<float>(i,j) = vec[i*tmp.cols+j];

	if( exType == "features" )			// 只取最后396位的HOG/HOF/MBH数据训练码本
		tmpMat = tmp.colRange(42,DIMENSION);
	else if( exType == "denseSample" ){// 只取帧数、轨迹mean_x、mean_y、396位的HOG/HOF/MBH数据
		Mat tmpFrame = tmp.colRange(0,1);
		Mat tmpxyPos = tmp.colRange(1,3);
		Mat tmpData  = tmp.colRange(42,DIMENSION);	
		Mat tmpMatrix(rows, 399, CV_32FC1);
		for(int i = 0; i < tmpData.rows; i++)
			for(int j = 0; j < tmpData.cols; j++){
				tmpMatrix.at<float>(i,0) = tmpFrame.at<float>(i,0);
				tmpMatrix.at<float>(i,1) = tmpxyPos.at<float>(i,0);
				tmpMatrix.at<float>(i,2) = tmpxyPos.at<float>(i,1);
				tmpMatrix.at<float>(i,j+3) = tmpData.at<float>(i,j);
			}
		tmpMat = tmpMatrix;
	}
	else if( exType == "drawTrack" ){	// 只取帧数、尺度、轨迹的16个坐标点	
		Mat tmpFrame = tmp.colRange(0,1);
		Mat tmpScale = tmp.colRange(6,7);
		Mat tmpData  = tmp.colRange(10,42);	
		Mat tmpMatrix(rows, 34, CV_32FC1);
		for(int i = 0; i < tmpData.rows; i++)
			for(int j = 0; j < tmpData.cols; j++){
				tmpMatrix.at<float>(i,0) = tmpFrame.at<float>(i,0);
				tmpMatrix.at<float>(i,1) = tmpScale.at<float>(i,0);
				tmpMatrix.at<float>(i,j+2) = tmpData.at<float>(i,j);
			}
		tmpMat = tmpMatrix;
	}

	tmp.release();
	vec.clear();
	return 0;
}

// 将特征文件、码本文件从txt转换为Mat格式
int ActionAnalysis::txt2Mat(string file, Mat& tmpMat, string exType)
{
	ifstream dataFile(file.c_str());
	string line;
	vector<float> vec;					
	while(getline(dataFile, line)) 
	{	
		int counter = 0;
		istringstream iss(line);
		string tmpData;

		while( iss >> tmpData ){
			if( exType == "features" ){
				if( counter++ > 41 )	// 只取最后396位的HOG/HOF/MBH数据训练码本
					vec.push_back(atof(tmpData.c_str()));
			}
			else if( exType == "denseSample" ){
				// 只取帧数、轨迹mean_x、mean_y、396位的HOG/HOF/MBH数据
				if( counter == 0 || (counter>0 && counter<3) || counter > 41 )					
					vec.push_back(atof(tmpData.c_str()));
				counter++;
			}
			else if( exType == "drawTrack" ){
				// 只取帧数、尺度、轨迹的16个坐标点
				if( counter == 0 || counter == 6 || (counter>9 && counter<42) )					
					vec.push_back(atof(tmpData.c_str()));
				counter++;
			}
			else
				vec.push_back(atof(tmpData.c_str()));
		}

		Mat tmpRow(1, vec.size(), CV_32FC1, Scalar(0));
		for(int j = 0; j < tmpRow.cols; j++)
			tmpRow.at<float>(0, j) = vec[j];
		tmpMat.push_back(tmpRow);
		tmpRow.release();
		vec.clear();
	}// while getline

	dataFile.close();
	return 0;
}

// 从split文件中读取训练集和测试集
int ActionAnalysis::readFileSets(string file, multimap<string, string> &dataSets, string processType)
{
	ifstream configFile(file.c_str());
    string line;
    while(getline(configFile, line)) 
	{
		istringstream iss(line);
		string actionType;
		if (!(iss >> actionType))
			continue;		
		string videoname;
		if (!(iss >> videoname))
			continue;
		int value;
		if (!(iss >> value))
			continue;
		if (processType == "train"){
			if (value == 1){
				int index = videoname.find_last_of('.');
				string tmpStr = videoname.substr(0, index);
				dataSets.insert(pair<string, string>(actionType, tmpStr));
			}
		}
		else if (processType == "test"){
			if (value == -1){
				int index = videoname.find_last_of('.');
				string tmpStr = videoname.substr(0, index);
				dataSets.insert(pair<string, string>(actionType, tmpStr));
			}
		}
	}

	configFile.close();
	return 0;
}


int ActionAnalysis::getMultiCodebook(vector<string> actionType, multimap<string, string> actionSet, 
									 string resultPath)
{
#ifdef _WIN32 // Windows version
#else // Linux version
	struct sysinfo s_info;
    int error = sysinfo(&s_info);
#endif

	cout << "enter getMultiCodebook，准备读取所有视频的txt特征文件........" << endl;

	// 先算出每类动作应该采样的特征数	
	int videoNum = 0;
	map<string, float> videoSampling;
	vector<string>::iterator itype;
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{	
		videoNum = 0;
		string actionTypeStr = *itype;
		multimap<string, string>::iterator iter;
		for(iter = actionSet.begin(); iter != actionSet.end(); iter++)
			if( iter->first == actionTypeStr )
				videoNum++;
		videoSampling.insert(map<string, float>::value_type(actionTypeStr, SAMPLE_NUM/videoNum+1));
	}

	FileStorage fs;
	Mat des, tmpMat, dstMat;	
	int video_num = 0, totalframes = 0;
	int retries = 8, cnt = 0;  
/*	int flags = KMEANS_PP_CENTERS; 
	TermCriteria tc(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 10, 0.001); 
	BOWKMeansTrainer bowTrainer(DICTIONARY_SIZE, tc, retries, flags);
	BOWKMeansTrainer bowTrainer2(DICTIONARY_SIZE, tc, retries, flags);
	BOWKMeansTrainer bowTrainer3(DICTIONARY_SIZE, tc, retries, flags);
*/
	// Current time tick
	double t1 = double( getTickCount() );	

	// 从目录下顺序读取所有特征描述符，并合并到一个矩阵中	
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{	
		float sampleNum = videoSampling[*itype];	
		ThreadParam *thrd = new ThreadParam();
		thrd->actionSet = actionSet;
		thrd->resultPath = resultPath;
		thrd->featurePath = resultPath + "features" + spritLabel + *itype + spritLabel;
		thrd->actionTypeStr = *itype;
		thrd->sampleNum = sampleNum;

#ifdef _WIN32 // Windows version
		SYSTEM_INFO theSystemInfo;
		::GetSystemInfo(&theSystemInfo);
		while( semaphore_featuresFormEachAction >= theSystemInfo.dwNumberOfProcessors)
			Sleep( 1000 );

		HANDLE hThread = CreateThread(NULL, 0, featuresFormEachAction, thrd, 0, NULL);
		if(hThread == NULL){
			cout << "Create Thread failed in getMultiCodebook() !" << endl;
			delete thrd;
			return -1;
		}
		_InterlockedIncrement( &semaphore_featuresFormEachAction );
#else // Linux version
		int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
		int semaNum;
		sem_getvalue(&semaphore_featuresFormEachAction, &semaNum);
		while( semaNum >= NUM_PROCS){
			sleep( 1 );
			sem_getvalue(&semaphore_featuresFormEachAction, &semaNum);
		}

		pthread_t pthID;
		int ret = pthread_create(&pthID, NULL, featuresFormEachAction, thrd);
		if(ret)	{
			cout << "Create Thread failed in trainGMM() !" << endl;
			delete thrd;
			return -1;
		}
		sem_post( &semaphore_featuresFormEachAction );
#endif
	}// for itype
	
	// 防止以上for循环结束后程序马上结束的情况。因为信号量semaphore可能还不为0，
	// 此时有部分线程仍在工作未释放信号量，所以应该以信号量是否为0，判断主程序是否结束。
#ifdef _WIN32 // Windows version
	while( semaphore_featuresFormEachAction )
		Sleep( 1000 );
#else // Linux version
	int semaNum;
	sem_getvalue(&semaphore_featuresFormEachAction, &semaNum);
	while( semaNum ){
		sleep( 1 );
		sem_getvalue(&semaphore_featuresFormEachAction, &semaNum);
	}
#endif

	// Current time tick
	double t2 = double( getTickCount() );
	t1 = ( (double)getTickCount() - t1 ) / getTickFrequency();
	cout << endl << "all action features have been put into memory ...." << endl;	
	cout << endl << "ready for bowTrainer.add(dstMat) ...." << endl;

	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{	
		string file = resultPath + *itype + ".xml";
		fs.open(file, FileStorage::READ);
		fs["features"] >> tmpMat;
		CV_Assert( !tmpMat.empty() );		
		// 每类动作，指定采样SAMPLE_NUM/actionType.size()+1个特征点
		randSampling(tmpMat, dstMat, SAMPLE_NUM/actionType.size()+1, "actionType");
		tmpMat.release();
		// delete txt for saving disk storage
		string cmdDel = "rm " + resultPath + *itype + ".xml";
		system(cmdDel.c_str());
	}

	// 多通道分别训练码本	
	vector<string> cb_name; 
	cb_name.push_back("HOG");
	cb_name.push_back("HOF");
	cb_name.push_back("MBH");
	vector<string>::iterator itcb;
	for(itcb = cb_name.begin(); itcb != cb_name.end(); itcb++)
	{	
		CodebookParam *thrd = new CodebookParam();
		thrd->cbname = *itcb;
		thrd->featuresMat = dstMat;
		thrd->resultPath = resultPath;

#ifdef _WIN32 // Windows version
		SYSTEM_INFO theSystemInfo;
		::GetSystemInfo(&theSystemInfo);
		while( semaphore_trainMultiChCodebook >= theSystemInfo.dwNumberOfProcessors)
			Sleep( 1000 );

		HANDLE hThread = CreateThread(NULL, 0, trainMultiChCodebook, thrd, 0, NULL);
		if(hThread == NULL){
			cout << "Create Thread failed in getMultiCodebook() !" << endl;
			delete thrd;
			return -1;
		}
		_InterlockedIncrement( &semaphore_trainMultiChCodebook );
#else // Linux version
		int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
		int semaNum;
		sem_getvalue(&semaphore_trainMultiChCodebook, &semaNum);
		while( semaNum >= NUM_PROCS){
			sleep( 1 );
			sem_getvalue(&semaphore_trainMultiChCodebook, &semaNum);
		}
		pthread_t pthID;
		int ret = pthread_create(&pthID, NULL, trainMultiChCodebook, thrd);
		if(ret)	{
			cout << "Create Thread failed in getMultiCodebook() !" << endl;
			delete thrd;
			return -1;
		}
		sem_post( &semaphore_trainMultiChCodebook );
#endif
	}// for itcb
	
	// 防止以上for循环结束后程序马上结束的情况。因为信号量semaphore可能还不为0，
	// 此时有部分线程仍在工作未释放信号量，所以应该以信号量是否为0，判断主程序是否结束。
#ifdef _WIN32 // Windows version
	while( semaphore_trainMultiChCodebook )
		Sleep( 1000 );
#else // Linux version
	sem_getvalue(&semaphore_trainMultiChCodebook, &semaNum);
	while( semaNum ){
		sleep( 1 );
		sem_getvalue(&semaphore_trainMultiChCodebook, &semaNum);
	}
#endif
/*******
	// 训练HOG 码本
	bowTrainer.add(dstMat.colRange(0,96));
	cout << endl << "making hogcodebook ...." << endl;
	Mat codebook = bowTrainer.cluster(); 

	// 训练HOF 码本
	bowTrainer2.add(dstMat.colRange(96,204));
	cout << "making hofcodebook ...." << endl;
	Mat codebook2 = bowTrainer2.cluster(); 

	// 训练MBH 码本
	bowTrainer3.add(dstMat.colRange(204,396));
	cout << "making mbhcodebook ...." << endl;
	Mat codebook3 = bowTrainer3.cluster(); 

	string hogcb_file = resultPath + "HOG.xml";
	string hofcb_file = resultPath + "HOF.xml";
	string mbhcb_file = resultPath + "MBH.xml";
	FileStorage fscodebook, fscodebook2, fscodebook3;
	fscodebook.open( hogcb_file.c_str(), FileStorage::WRITE);
	fscodebook2.open(hofcb_file.c_str(), FileStorage::WRITE);
	fscodebook3.open(mbhcb_file.c_str(), FileStorage::WRITE);
	if( !fscodebook.isOpened() || !fscodebook2.isOpened() || !fscodebook3.isOpened() ){
		cout << "Error: Could not open codebook file in ActionAnalysis::getMultiCodebook()." << endl;
		return -1;		
	}

	// 保存生成的码本
	fscodebook  << "codebook" << codebook;
	fscodebook2 << "codebook" << codebook2;
	fscodebook3 << "codebook" << codebook3;
*******/
	// 记录编码本的时间开销	
	FileStorage fslog;
	string file_loger= resultPath + "log_BOW.xml";
	t2 = ( (double)getTickCount() - t2 ) / getTickFrequency();
	fslog.open(file_loger, FileStorage::WRITE);
	fslog << "FulliScale_BOW" << "{" ;
	//fslog << "total_rows" << bowTrainer.descripotorsCount();
	fslog << "read_time_hours" << t1/3600.f;
	fslog << "bow_time_hours" << t2/3600.f;
	fslog << "video_num" << video_num;
	fslog << "}";

	// 释放内存
	dstMat.release();
/*	codebook.release();
	codebook2.release();
	codebook3.release();
	fscodebook.release();
	fscodebook2.release();
	fscodebook3.release();
*/	fs.release();
	fslog.release();

	return 0;
}

// 根据视频特征描述符，由码本得到对应的码字
int ActionAnalysis::getVideoCode(vector<string> actionType, multimap<string, string> actionSet, 
								 string vidcodePath, string resultPath, string processType)
{
	float totalframes = 0;		
	int video_num = 0;
	// Current time tick
	double t = double( getTickCount() );

	vector<string>::iterator itype;
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{		
		string actionTypeStr = *itype;
		multimap<string, string>::iterator iter;
		for(iter = actionSet.begin(); iter != actionSet.end(); iter++)
		{
			if( iter->first == actionTypeStr )
			{
				//if( iter->second != "person01_boxing_d1_uncomp")
				//	continue;
				ThreadParam *thrd = new ThreadParam();
				thrd->featurePath = resultPath + "features" + spritLabel + actionTypeStr + spritLabel;
				thrd->vidcodePath = vidcodePath;
				thrd->resultPath = resultPath;
				thrd->filename = iter->second;
				thrd->actionTypeStr = actionTypeStr;

	#ifdef _WIN32 // Windows version
				SYSTEM_INFO theSystemInfo;
				::GetSystemInfo(&theSystemInfo);
				while( semaphore_quantization >= theSystemInfo.dwNumberOfProcessors)
					Sleep( 1000 );

				HANDLE hThread = CreateThread(NULL, 0, featuresQuantizationProcess, thrd, 0, NULL);
				if(hThread == NULL){
					cout << "Create Thread failed in featuresQuantizationProcess !" << endl;
					delete thrd;
					return -1;
				}
				_InterlockedIncrement( &semaphore_quantization );
	#else // Linux version
				int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
				int semaNum;
				sem_getvalue(&semaphore_quantization, &semaNum);
				while( semaNum >= NUM_PROCS){
					sleep( 1 );
					sem_getvalue(&semaphore_quantization, &semaNum);
				}

				pthread_t pthID;
				int ret = pthread_create(&pthID, NULL, featuresQuantizationProcess, thrd);
				if(ret){
					cout << "Create Thread failed in featuresQuantizationProcess !" << endl;
					delete thrd;
					return -1;
				}
				sem_post( &semaphore_quantization );
	#endif
				// 统计已处理的帧数和视频数
				//totalframes += frame_counter[idx];
				video_num++;	
			}// if iter->first
		}//	for iter		
	}// for itype

	// 防止以上for循环结束后程序马上结束的情况。因为信号量semaphore可能还不为0，
	// 此时有部分线程仍在工作未释放信号量，所以应该以信号量是否为0，判断主程序是否结束。
#ifdef _WIN32 // Windows version
	while( semaphore_quantization )
		Sleep( 1000 );
#else // Linux version
	int semaNum;
	sem_getvalue(&semaphore_quantization, &semaNum);
	while( semaNum ){
		sleep( 1 );
		sem_getvalue(&semaphore_quantization, &semaNum);
	}
#endif
	// 记录训练集量化的时间开销
	FileStorage fslog;
	string file_loger= resultPath + "log_" + processType + "VideoCode.xml";
	t = ( (double)getTickCount() - t ) / getTickFrequency();
	fslog.open(file_loger, FileStorage::WRITE);
	fslog << "getVideoCode" << "{" ;
	fslog << "time_hours" << t/3600;
	fslog << "video_num" << video_num;
	//fslog << "average_FPS" << (int)( totalframes / t );
	fslog << "}";
	fslog.release();

	return 0;
}

// VectorQuantization 硬投票
int ActionAnalysis::hardVote(Mat tmp, Mat& code, Mat hogcodebook, Mat hofcodebook, Mat mbhcodebook)
{
	vector<DMatch> matches;
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );

	// channel[0] for hog									
	descriptorMatcher->match(tmp.colRange(0,96), hogcodebook, matches);
	for(int k = 0; k < matches.size(); k++)
		code.at<Vec3f>(0, matches[k].trainIdx)[0]++;
	matches.clear();

	// channel[1] for hof					
	descriptorMatcher->match(tmp.colRange(96,204), hofcodebook, matches);
	for(int k = 0; k < matches.size(); k++)
		code.at<Vec3f>(0, matches[k].trainIdx)[1]++;
	matches.clear();

	// channel[2] for mbh
	descriptorMatcher->match(tmp.colRange(204,396), mbhcodebook, matches);
	for(int k = 0; k < matches.size(); k++)
		code.at<Vec3f>(0, matches[k].trainIdx)[2]++;
	matches.clear();

	return 0;
}

// SA-k knn近邻软投票 默认knn=5 
int ActionAnalysis::softAssignKnn(Mat tmp, Mat& code, int knn, Mat hogcodebook, Mat hofcodebook, Mat mbhcodebook)
{
	int channels = code.channels();
	vector<float> maxPooling(channels);
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( "BruteForce" );

	// channel[0] for hog
	float tmpDis = 0, sumDis = 0;
	vector<vector<DMatch> > hogmatches;
	descriptorMatcher->knnMatch(tmp.colRange(0,96), hogcodebook, hogmatches, knn);
	for( int m = 0; m < hogmatches.size(); m++ )
	{
		sumDis = 0;
		for( int fk = 0; fk < hogmatches[m].size(); fk++ )
		{						
			DMatch forward = hogmatches[m][fk];
			tmpDis = exp(-1*forward.distance);
			sumDis += tmpDis;	
			tmpDis = 0;
		}
		for( int fk = 0; fk < hogmatches[m].size(); fk++ )
		{						
			DMatch forward = hogmatches[m][fk];
			tmpDis = exp(-1*forward.distance);
			if(code.at<Vec3f>(0, forward.trainIdx)[0] < tmpDis/sumDis)
				code.at<Vec3f>(0, forward.trainIdx)[0] = tmpDis/sumDis;
			//code.at<Vec3f>(0, forward.trainIdx)[0] += tmpDis/sumDis;
		}
	}
	hogmatches.clear();
			
	// channel[1] for hof
	tmpDis = 0, sumDis = 0;
	vector<vector<DMatch> > hofmatches;
	descriptorMatcher->knnMatch(tmp.colRange(96,204), hofcodebook, hofmatches, knn);
	for( int m = 0; m < hofmatches.size(); m++ )
	{
		sumDis = 0;
		for( int fk = 0; fk < hofmatches[m].size(); fk++ )
		{						
			DMatch forward = hofmatches[m][fk];
			tmpDis = exp(-1*forward.distance);
			sumDis += tmpDis;	
			tmpDis = 0;
		}
		for( int fk = 0; fk < hofmatches[m].size(); fk++ )
		{						
			DMatch forward = hofmatches[m][fk];
			tmpDis = exp(-1*forward.distance);
			if(code.at<Vec3f>(0, forward.trainIdx)[1] < tmpDis/sumDis)
				code.at<Vec3f>(0, forward.trainIdx)[1] = tmpDis/sumDis;
			//code.at<Vec3f>(0, forward.trainIdx)[1] += tmpDis/sumDis;
		}
	}
	hofmatches.clear();
			
	// channel[2] for mbh
	tmpDis = 0, sumDis = 0;
	vector<vector<DMatch> > mbhmatches;
	descriptorMatcher->knnMatch(tmp.colRange(204,396), mbhcodebook, mbhmatches, knn);
	for( int m = 0; m < mbhmatches.size(); m++ )
	{
		sumDis = 0;
		for( int fk = 0; fk < mbhmatches[m].size(); fk++ )
		{						
			DMatch forward = mbhmatches[m][fk];
			tmpDis = exp(-1*forward.distance);
			sumDis += tmpDis;	
			tmpDis = 0;
		}
		for( int fk = 0; fk < mbhmatches[m].size(); fk++ )
		{						
			DMatch forward = mbhmatches[m][fk];
			tmpDis = exp(-1*forward.distance);
			if(code.at<Vec3f>(0, forward.trainIdx)[2] < tmpDis/sumDis)
				code.at<Vec3f>(0, forward.trainIdx)[2] = tmpDis/sumDis;
			//code.at<Vec3f>(0, forward.trainIdx)[2] += tmpDis/sumDis;
		}
	}
	mbhmatches.clear();

	return 0;
}

int ActionAnalysis::randSampling(Mat src, Mat& dst, int sampleNum, string sampleType)
{
	vector<int> sampledRows;
	vector<int> unSampledRows;
	vector<int>::iterator irows;
	
	// 将所要提取特征的行数加入未采样队列容器
	for(int i=0; i<src.rows; i++)
		unSampledRows.push_back(i);

	// "真"随机数，只针对训练码本前读取每类动作的XML特征文件，SAMPLE_NUM中只出现一个重复点可忽略不计
	if( sampleType == "actionType" )
	{
		int sampleCount = 0;
		// 指定特征点采样数量
		while(sampleCount < sampleNum)
		{	//	根据时间产生随机数
			srand(getTickCount());	
			// 指定采样特征点的范围，从(0, 未采样队列的剩余点数]
			unsigned int num = rand()%unSampledRows.size();
			for(irows=unSampledRows.begin(); irows!=unSampledRows.end(); irows++){
				if( *irows == num ){
					unSampledRows.erase(irows);
					break;
				}
			}
			sampleCount++;
			dst.push_back(src.row(num));
		}
	}
	// 真随机数，只针对单个视频提取指定数量的特征数，只从sampledRows中取数据
	else if( sampleType == "oneVideo" )
	{
		int sampleCount = 0;
		// 指定特征点采样数量
		while(sampleCount < sampleNum)
		{
			srand(getTickCount());	
			// 指定采样特征点的范围，从(0, 未采样队列的剩余点数]
			unsigned int num = rand()%unSampledRows.size();
			for(irows=unSampledRows.begin(); irows!=unSampledRows.end(); irows++){
				if( *irows == num ){
					unSampledRows.erase(irows);
					break;
				}
			}
			if( sampleCount == 0){
				sampleCount++;
				sampledRows.push_back(num);
				dst.push_back(src.row(num));
			}
			for(irows=sampledRows.begin(); irows!=sampledRows.end(); irows++){
				// 如果特征点的行数之前没有出现过，就取该点作为有效随机数，并压入已采样队列容器；否则重新生成随机数
				if( *irows != num ){
					sampleCount++;
					sampledRows.push_back(num);
					dst.push_back(src.row(num));
					break;
				}
			}
		}
	}// else if

	// 释放资源
	sampledRows.clear();
	unSampledRows.clear();

	return 0;
}