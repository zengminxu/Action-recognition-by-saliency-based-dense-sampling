#include "ActionAnalysis.h"

using namespace std;

extern string spritLabel;	//ʹ��DenseTrackStab.cpp��ȫ�ֱ���

// ���߳�ͬʱ��HOG/HOF/MBH ��ͨ�������뱾ѵ��
#ifdef _WIN32 // Windows version
	volatile LONG semaphore_trainMultiChCodebook = 0;
	DWORD WINAPI trainMultiChCodebook( LPVOID lpParameter )
#else // Linux version
	sem_t semaphore_trainMultiChCodebook;
	static void *trainMultiChCodebook(void *lpParameter)
#endif
{	
	CodebookParam *lp = (CodebookParam*)lpParameter;

	// ׼��ѵ���뱾
	int retries = 8;  
	int flags = KMEANS_PP_CENTERS; 
	TermCriteria tc(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 10, 0.001); 
	BOWKMeansTrainer bowTrainer(DICTIONARY_SIZE, tc, retries, flags);	

	// д��ÿ�ද��ָ����С������
	FileStorage fs;
	string cb_file = lp->resultPath + "bow" + lp->cbname + ".xml";
	fs.open(cb_file, FileStorage::WRITE);
	if( !fs.isOpened() )
	{
		cout << "Error: Could not open features file in trainMultiChCodebook()" << endl;
		delete lp;
#ifdef _WIN32 // Windows version
		// ������ȡ�������ź���-1
		_InterlockedDecrement( &semaphore_trainMultiChCodebook );
#else // Linux version
		sem_wait( &semaphore_trainMultiChCodebook );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}

	// ��Ӹ�ͨ����ѵ������
	if( lp->cbname == "HOG" )
		bowTrainer.add(lp->featuresMat.colRange(0,96));
	else if( lp->cbname == "HOF" )
		bowTrainer.add(lp->featuresMat.colRange(96,204));
	else if( lp->cbname == "MBH" )
		bowTrainer.add(lp->featuresMat.colRange(204,396));

	// ѵ��HOG/HOF/MBH �뱾
	Mat codebook = bowTrainer.cluster(); 
	
	// �������ɵ��뱾
	fs  << "codebook" << codebook;

	// �ͷ���Դ.....
	fs.release();
	delete lp;

#ifdef _WIN32 // Windows version
	// ������ȡ�������ź���-1
	_InterlockedDecrement( &semaphore_trainMultiChCodebook );
#else // Linux version
	sem_wait( &semaphore_trainMultiChCodebook );
	pthread_detach(pthread_self());
#endif
	
	return 0;
}


// �ȶ�ȡ��ѵ��������������д��txt�ļ�����Ϊ�뱾����������
#ifdef _WIN32 // Windows version
	volatile LONG semaphore_featuresFormEachAction = 0;
	DWORD WINAPI featuresFormEachAction( LPVOID lpParameter )
#else // Linux version
	sem_t semaphore_featuresFormEachAction;
	static void *featuresFormEachAction(void *lpParameter)
#endif
{	
	ThreadParam *lp = (ThreadParam*)lpParameter;

	// �ȶ�ȡ��ѵ������������������bin�ļ�����Ϊ�뱾����������
	ActionAnalysis action;
	Mat des, tmpMat;

	// ��Ŀ¼��˳���ȡ�������������������ϲ���һ��������	
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

			// ÿ����Ƶ��ָ������sampleNum=GMM_SAMPLE_NUM/videoNum�������㣬videoNumָÿ�ද������Ƶ��
			if( tmpMat.rows <= lp->sampleNum )
				des.push_back(tmpMat);
			else
				action.randSampling(tmpMat, des, lp->sampleNum, "oneVideo");

		}//if iter->first
		tmpMat.release();
	}// for iter	

	// д��ÿ�ද��ָ����С������
	FileStorage fs;
	string action_files = lp->resultPath + lp->actionTypeStr + ".xml";
	fs.open(action_files, FileStorage::WRITE);
	if( !fs.isOpened() )
	{
		cout << "Error: Could not open features file in readFeaturesFormEachAction()" << endl;
		delete lp;
#ifdef _WIN32 // Windows version
		// ������ȡ�������ź���-1
		_InterlockedDecrement( &semaphore_featuresFormEachAction );
#else // Linux version
		sem_wait( &semaphore_featuresFormEachAction );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}
	fs << "features" << des;

	// �ͷ���Դ.....
	des.release();
	fs.release();
	delete lp;

#ifdef _WIN32 // Windows version
	// ������ȡ�������ź���-1
	_InterlockedDecrement( &semaphore_featuresFormEachAction );
#else // Linux version
	sem_wait( &semaphore_featuresFormEachAction );
	pthread_detach(pthread_self());
#endif
	
	return 0;
}


// ��Ƶ������BoF����
#ifdef _WIN32 // Windows version
	volatile LONG semaphore_quantization = 0;
	DWORD WINAPI featuresQuantizationProcess( LPVOID lpParameter )
#else // Linux version
	sem_t semaphore_quantization;
	static void *featuresQuantizationProcess(void *lpParameter)
#endif
{
	ThreadParam *lp = (ThreadParam*)lpParameter;

	// ��ʼ��������
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
		// ������ȡ�������ź���-1
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

	// ��ȡÿ����Ƶ��XML����������
	ifstream fsread;
	FileStorage fswrite;
	string feature_files = lp->featurePath + lp->filename + ".bin";
	// д��ÿ����Ƶ�����������
	string vidcode_files = lp->vidcodePath + lp->filename + "_quantization.xml";
	fsread.open(feature_files.c_str(), ios::in);
	fswrite.open(vidcode_files, FileStorage::WRITE);
	if(!fsread.is_open() || !fswrite.isOpened())
	{
		cout << "Error: Could not open feature/quantization file in featuresQuantizationProcess()." << endl;
		delete lp;
#ifdef _WIN32 // Windows version
		// ������ȡ�������ź���-1
		_InterlockedDecrement( &semaphore_quantization );
#else // Linux version
		sem_wait( &semaphore_quantization );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}

	//cout << "video quantization "<< vidcode_files << endl;	
	Mat code(1, DICTIONARY_SIZE, CV_32FC3, Scalar::all(0));	//���HOG��HOF��MBH ������Codebook�����
	const int channels = code.channels(); 
	// ÿ����Ƶͳ��һ��
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
				// ��Ƶ��Ӧ�������ӱ��
				fswrite << lp->actionTypeStr;
				// ���յ�����д��xml
				fswrite << tmp;
				fswrite.release();
				delete lp;
		#ifdef _WIN32 // Windows version
				// ������ȡ�������ź���-1
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
	
	// ��Ƶ��Ӧ�������ӱ��
	fswrite << lp->actionTypeStr;
	// ����Ƶ����������д��xml
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
	// ������ȡ�������ź���-1
	_InterlockedDecrement( &semaphore_quantization );
#else // Linux version
	sem_wait( &semaphore_quantization );
	pthread_detach(pthread_self());
#endif
	
	return 0;
}

// �������ļ����뱾�ļ���binת��ΪMat��ʽ
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

	if( exType == "features" )			// ֻȡ���396λ��HOG/HOF/MBH����ѵ���뱾
		tmpMat = tmp.colRange(42,DIMENSION);
	else if( exType == "denseSample" ){// ֻȡ֡�����켣mean_x��mean_y��396λ��HOG/HOF/MBH����
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
	else if( exType == "drawTrack" ){	// ֻȡ֡�����߶ȡ��켣��16�������	
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

// �������ļ����뱾�ļ���txtת��ΪMat��ʽ
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
				if( counter++ > 41 )	// ֻȡ���396λ��HOG/HOF/MBH����ѵ���뱾
					vec.push_back(atof(tmpData.c_str()));
			}
			else if( exType == "denseSample" ){
				// ֻȡ֡�����켣mean_x��mean_y��396λ��HOG/HOF/MBH����
				if( counter == 0 || (counter>0 && counter<3) || counter > 41 )					
					vec.push_back(atof(tmpData.c_str()));
				counter++;
			}
			else if( exType == "drawTrack" ){
				// ֻȡ֡�����߶ȡ��켣��16�������
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

// ��split�ļ��ж�ȡѵ�����Ͳ��Լ�
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

	cout << "enter getMultiCodebook��׼����ȡ������Ƶ��txt�����ļ�........" << endl;

	// �����ÿ�ද��Ӧ�ò�����������	
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

	// ��Ŀ¼��˳���ȡ�������������������ϲ���һ��������	
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
	
	// ��ֹ����forѭ��������������Ͻ������������Ϊ�ź���semaphore���ܻ���Ϊ0��
	// ��ʱ�в����߳����ڹ���δ�ͷ��ź���������Ӧ�����ź����Ƿ�Ϊ0���ж��������Ƿ������
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
		// ÿ�ද����ָ������SAMPLE_NUM/actionType.size()+1��������
		randSampling(tmpMat, dstMat, SAMPLE_NUM/actionType.size()+1, "actionType");
		tmpMat.release();
		// delete txt for saving disk storage
		string cmdDel = "rm " + resultPath + *itype + ".xml";
		system(cmdDel.c_str());
	}

	// ��ͨ���ֱ�ѵ���뱾	
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
	
	// ��ֹ����forѭ��������������Ͻ������������Ϊ�ź���semaphore���ܻ���Ϊ0��
	// ��ʱ�в����߳����ڹ���δ�ͷ��ź���������Ӧ�����ź����Ƿ�Ϊ0���ж��������Ƿ������
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
	// ѵ��HOG �뱾
	bowTrainer.add(dstMat.colRange(0,96));
	cout << endl << "making hogcodebook ...." << endl;
	Mat codebook = bowTrainer.cluster(); 

	// ѵ��HOF �뱾
	bowTrainer2.add(dstMat.colRange(96,204));
	cout << "making hofcodebook ...." << endl;
	Mat codebook2 = bowTrainer2.cluster(); 

	// ѵ��MBH �뱾
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

	// �������ɵ��뱾
	fscodebook  << "codebook" << codebook;
	fscodebook2 << "codebook" << codebook2;
	fscodebook3 << "codebook" << codebook3;
*******/
	// ��¼���뱾��ʱ�俪��	
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

	// �ͷ��ڴ�
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

// ������Ƶ���������������뱾�õ���Ӧ������
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
				// ͳ���Ѵ����֡������Ƶ��
				//totalframes += frame_counter[idx];
				video_num++;	
			}// if iter->first
		}//	for iter		
	}// for itype

	// ��ֹ����forѭ��������������Ͻ������������Ϊ�ź���semaphore���ܻ���Ϊ0��
	// ��ʱ�в����߳����ڹ���δ�ͷ��ź���������Ӧ�����ź����Ƿ�Ϊ0���ж��������Ƿ������
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
	// ��¼ѵ����������ʱ�俪��
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

// VectorQuantization ӲͶƱ
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

// SA-k knn������ͶƱ Ĭ��knn=5 
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
	
	// ����Ҫ��ȡ��������������δ������������
	for(int i=0; i<src.rows; i++)
		unSampledRows.push_back(i);

	// "��"�������ֻ���ѵ���뱾ǰ��ȡÿ�ද����XML�����ļ���SAMPLE_NUM��ֻ����һ���ظ���ɺ��Բ���
	if( sampleType == "actionType" )
	{
		int sampleCount = 0;
		// ָ���������������
		while(sampleCount < sampleNum)
		{	//	����ʱ����������
			srand(getTickCount());	
			// ָ������������ķ�Χ����(0, δ�������е�ʣ�����]
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
	// ���������ֻ��Ե�����Ƶ��ȡָ����������������ֻ��sampledRows��ȡ����
	else if( sampleType == "oneVideo" )
	{
		int sampleCount = 0;
		// ָ���������������
		while(sampleCount < sampleNum)
		{
			srand(getTickCount());	
			// ָ������������ķ�Χ����(0, δ�������е�ʣ�����]
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
				// ��������������֮ǰû�г��ֹ�����ȡ�õ���Ϊ��Ч���������ѹ���Ѳ������������������������������
				if( *irows != num ){
					sampleCount++;
					sampledRows.push_back(num);
					dst.push_back(src.row(num));
					break;
				}
			}
		}
	}// else if

	// �ͷ���Դ
	sampledRows.clear();
	unSampledRows.clear();

	return 0;
}