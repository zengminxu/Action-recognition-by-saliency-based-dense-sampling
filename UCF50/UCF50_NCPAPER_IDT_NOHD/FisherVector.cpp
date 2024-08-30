#include "ActionAnalysis.h"
#include "FisherVector.h"

using namespace cv;
using namespace std;

extern string spritLabel;	//ʹ��DenseTrackStab.cpp��ȫ�ֱ���

// ���߳�ͬʱ��HOG/HOF/MBHx/MBHy ��ͨ�������뱾ѵ��
#ifdef _WIN32 // Windows version
	volatile LONG semaphore_trainMulChGMM = 0;
	DWORD WINAPI trainMulChGMM( LPVOID lpParameter )
#else // Linux version
	sem_t semaphore_trainMulChGMM;
	static void *trainMulChGMM(void *lpParameter)
#endif
{	
	CodebookParam *lp = (CodebookParam*)lpParameter;
	
	FisherVector fisher;
	Mat PCAwhiten, GMMdes;
	int pcaNum = 0;
	
	// ��Ӹ�ͨ����ѵ������
	if( lp->manifoldType == "raw")
	{
		if( lp->cbname == "Combine" )
			GMMdes.push_back(lp->featuresMat);
		else if( lp->cbname == "HOG" )
			GMMdes.push_back(lp->featuresMat.colRange(0,96));
		else if( lp->cbname == "HOF" )
			GMMdes.push_back(lp->featuresMat.colRange(96,204));
		else if( lp->cbname == "MBH" )
			GMMdes.push_back(lp->featuresMat.colRange(204,396));
		else if( lp->cbname == "MBHx" )
			GMMdes.push_back(lp->featuresMat.colRange(204,300));
		else if( lp->cbname == "MBHy" )
			GMMdes.push_back(lp->featuresMat.colRange(300,396));

		// L2��һ�����Ż�ŷ�Ͼ���Ӷ�����GMMѵ�����(��Ϊ����GMM��ͨ��ŷ�Ͼ���������������й����)
		if(lp->cbname == "Combine"){
			for(int r = 0; r < GMMdes.rows; r++){
				normalize(GMMdes.row(r).colRange(0,96), GMMdes.row(r).colRange(0,96), 1.0, 0.0, NORM_L2);
				normalize(GMMdes.row(r).colRange(96,204), GMMdes.row(r).colRange(96,204), 1.0, 0.0, NORM_L2);
				normalize(GMMdes.row(r).colRange(204,396), GMMdes.row(r).colRange(204,396), 1.0, 0.0, NORM_L2);
			}
		}
		else
			for(int r = 0; r < GMMdes.rows; r++)
				normalize(GMMdes.row(r), GMMdes.row(r), 1.0, 0.0, NORM_L2);

		// δPCA�׻���ѵ���뱾��GMM���ࣨ396άHOG+HOF+MBH��
		fisher.gmmCluster(GMMdes, lp->resultPath, lp->gmmNum, lp->cbname, lp->manifoldType); 
		cout << lp->cbname << ":GMMdes.rows:" << GMMdes.rows << ", GMMdes.cols:" << GMMdes.cols << endl;
	}
	else if( lp->manifoldType == "pca")
	{
		if( lp->cbname == "Combine" ){
			GMMdes.push_back(lp->featuresMat);
			pcaNum = 200;
		}
		else if( lp->cbname == "HOG" ){
			GMMdes.push_back(lp->featuresMat.colRange(0,96));
			pcaNum = 48;
		}
		else if( lp->cbname == "HOF" ){
			GMMdes.push_back(lp->featuresMat.colRange(96,204));
			pcaNum = 54;
		}
		else if( lp->cbname == "MBH" ){
			GMMdes.push_back(lp->featuresMat.colRange(204,396));
			pcaNum = 96;
		}
		else if( lp->cbname == "MBHx" ){
			GMMdes.push_back(lp->featuresMat.colRange(204,300));
			pcaNum = 48;
		}
		else if( lp->cbname == "MBHy" ){
			GMMdes.push_back(lp->featuresMat.colRange(300,396));
			pcaNum = 48;
		}

		// �ȶ�����Ԥ����PCA�׻�
		fisher.pcaWhiten(GMMdes, PCAwhiten, lp->resultPath, pcaNum, lp->cbname, "CV_PCA_DATA_AS_ROW");
		// L2��һ�����Ż�ŷ�Ͼ���Ӷ�����GMMѵ�����(��Ϊ����GMM��ͨ��ŷ�Ͼ���������������й����)
		for(int r = 0; r < PCAwhiten.rows; r++)
			normalize(PCAwhiten.row(r), PCAwhiten.row(r), 1.0, 0.0, NORM_L2);
		// ��PCA�׻���ѵ���뱾��GMM����
		fisher.gmmCluster(PCAwhiten, lp->resultPath, lp->gmmNum, lp->cbname, lp->manifoldType);
		cout << lp->cbname << ":PCAwhiten.rows:" << PCAwhiten.rows << ", PCAwhiten.cols:" << PCAwhiten.cols << endl;
	}
	
	// �ͷ���Դ.....
	delete lp;

#ifdef _WIN32 // Windows version
	// ������ȡ�������ź���-1
	_InterlockedDecrement( &semaphore_trainMulChGMM );
#else // Linux version
	sem_wait( &semaphore_trainMulChGMM );
	pthread_detach(pthread_self());
#endif
	
	return 0;
}


// Data to protect with the interlocked functions.
#ifdef _WIN32 // Windows version
	volatile LONG semaphore_readFeaturesFormEachAction = 0;
	DWORD WINAPI readFeaturesFormEachAction( LPVOID lpParameter )
#else // Linux version
	sem_t semaphore_readFeaturesFormEachAction;
	static void *readFeaturesFormEachAction(void *lpParameter)
#endif
{	
	ThreadParam *lp = (ThreadParam*)lpParameter;

	// �ȶ�ȡ��ѵ������������������txt�ļ�����Ϊgmm����������
	FisherVector fisher;
	ActionAnalysis action;
	Mat des, tmpMat, pcaMat;

	// ��Ŀ¼��˳���ȡ�������������������ϲ���һ��������	
	multimap<string, string>::iterator iter;
	for(iter = lp->actionSet.begin(); iter != lp->actionSet.end(); iter++)
	{
		if( iter->first == lp->actionTypeStr )
		{
			//if(iter->second != "-_FREE_HUGS_-_Abrazos_Gratis_www_abrazosgratis_org_hug_u_cm_np2_le_goo_11")
			//	continue;
			string file = lp->featurePath + iter->second + ".bin";
			action.bin2Mat(file, tmpMat, "features");
			if( tmpMat.empty() ){
				cout << "Read Error: features is empty in " << file << endl;
				continue;		
			}

			RNG rng(getTickCount());	// ȡ��ǰϵͳʱ����Ϊ���������
			Mat randMat(tmpMat.rows, 1, CV_32S);
			for(int r=0; r<randMat.rows; r++)
				randMat.at<int>(r,0) = r;
			randShuffle(randMat, 1, &rng);
	
			if( tmpMat.rows > lp->sampleNum ){	// ÿ�ද��ֻȡSAMPLE_NUM/CATEGORY_NUM��������
				for(int r=0; r<lp->sampleNum; r++){	
					int randRow = randMat.at<int>(r,0);
					des.push_back(tmpMat.row(randRow));
				}
				cout << lp->actionTypeStr << ", ���̶߳�ȡ������Ƶʱ�������������..." << ", tmpMat.rows:" << tmpMat.rows << ", lp->sampleNum:" << lp->sampleNum << endl;
			}
			else
				des.push_back(tmpMat);

		}//if iter->first
		pcaMat.release();
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
		_InterlockedDecrement( &semaphore_readFeaturesFormEachAction );
#else // Linux version
		sem_wait( &semaphore_readFeaturesFormEachAction );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}

	// ��ÿ�ද��������д����Ӧ��xml�ļ�
	fs << "features" << des;
	fs.release();

	// �ͷ���Դ.....
	des.release();
	fs.release();
	delete lp;

#ifdef _WIN32 // Windows version
	// ������ȡ�������ź���-1
	_InterlockedDecrement( &semaphore_readFeaturesFormEachAction );
#else // Linux version
	sem_wait( &semaphore_readFeaturesFormEachAction );
	pthread_detach(pthread_self());
#endif
	
	return 0;
}
	

// Data to protect with the interlocked functions.
#ifdef _WIN32 // Windows version
	volatile LONG semaphore_fisher_quantization = 0;
	DWORD WINAPI featuresFisherQuantization( LPVOID lpParameter )
#else // Linux version
	sem_t semaphore_fisher_quantization;
	static void *featuresFisherQuantization(void *lpParameter)
#endif
{	
	ThreadParam *lp = (ThreadParam*)lpParameter;

	// ��ʼ��������
	FileStorage fs;
	FisherVector fisher;
	ActionAnalysis action;

	// ��ȡÿ����Ƶ��TXT����������
	ifstream fsread;
	FileStorage fswrite;
	string feature_files = lp->featurePath + lp->filename + ".bin";
	// д��ÿ����Ƶ�����������
	string vidcode_files = lp->vidcodePath + lp->filename + "_quantization.xml";
	fsread.open(feature_files.c_str(), ios::in);
	fswrite.open(vidcode_files, FileStorage::WRITE);

	if(!fsread.is_open() || !fswrite.isOpened())
	{
		cout << "Error: Could not open feature/quantization file in featuresFisherQuantization()." << endl;
		delete lp;
#ifdef _WIN32 // Windows version
		// ������ȡ�������ź���-1
		_InterlockedDecrement( &semaphore_fisher_quantization );
#else // Linux version
		sem_wait( &semaphore_fisher_quantization );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}
	
	Mat tmpMat, denseMat, localMat; 
	Mat hogPCAwhiten, hofPCAwhiten, mbhPCAwhiten, mbhxPCAwhiten, mbhyPCAwhiten, combinePCAwhiten;
	Mat hogCode, hofCode, mbhCode, mbhxCode, mbhyCode, combineCode, localCode;

	// ��ȡ396άHOG+HOF+MBH����
	action.bin2Mat(feature_files, tmpMat, "features");
	// ���ܲ�����ʱ�վ�
	//action.bin2Mat(feature_files, tmpMat, "denseSample");
	
	if( tmpMat.empty() )
	{
		cout << "Warning: features is empty. ";
		int fvdim = 0;
		if( lp->descriptorType == "d-Fusion" )
			fvdim = 200*2*lp->gmmNum;
		else if( lp->descriptorType == "r3-Fusion" || lp->descriptorType == "r4-Fusion")
			fvdim = 198*2*lp->gmmNum;
		// ��Ƶ��Ӧ�������ӱ��
		fswrite << lp->actionTypeStr;
		Mat decimalMat(1, fvdim, CV_32FC1, Scalar(0));
		// ��ӣ�����������һ���ض���С��
		decimalMat = decimalMat + 1./fvdim;
		fswrite << decimalMat;
		cout << "code 1row x " << decimalMat.cols << "cols has been filled decimal in " << feature_files << endl;
		fswrite.release();
		delete lp;
#ifdef _WIN32 // Windows version
		// ������ȡ�������ź���-1
		_InterlockedDecrement( &semaphore_fisher_quantization );
#else // Linux version
		sem_wait( &semaphore_fisher_quantization );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}
	
 /////////////////////////////////////////////////////////// 
	//// ���ܲ�����ʱ�վ�
	//VideoCapture capture;
	//string video = lp->datasetPath + lp->actionTypeStr + spritLabel + lp->filename + ".avi";
	//capture.open(video);
	//int frameH = 0, frameW = 0, frameNum = 0;
	//frameH = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	//frameW = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	//frameNum = capture.get(CV_CAP_PROP_FRAME_COUNT);

	//int t_ind = 0, t_stride = 10, t_start = 0, t_end = 0, frameFlag = 0;
	//int tsize_3 = frameNum / 3;
	////for(int tsize = 0; tsize < frameNum; tsize += tsize_3 ){	// ��forѭ�������ʱ�����϶���߶ȵķ���

	//t_end = tsize_3;	
	//// ֻȡʱ�����ϵ�һ���߶�
	//do{		 
	//	if( t_end >= frameNum ){ // �趨��ʱ�վ�β����������Ƶ���֡��
	//		t_end = frameNum;
	//		frameFlag = 1;
	//	}
	//	for(int r = 0; r < tmpMat.rows; r++){	// ���������켣��֡���г��ܲ���
	//		t_ind = tmpMat.at<float>(r,0);
	//		if( t_start<t_ind && t_ind<=t_end )
	//			denseMat.push_back(tmpMat.row(r));
	//	}// for r
	//	if(denseMat.rows == 0)	// �ų�������Ƶ���ܲ���Ϊ0�����
	//		denseMat = tmpMat.clone();
	//	// PCA�׻�
	//	fisher.newSamplePCAWhiten(denseMat.colRange(3,denseMat.cols), combinePCAwhiten, lp->resultPath, "Combine", "CV_PCA_DATA_AS_ROW");
	//	//fisher.newSamplePCAWhiten(tmpMat.colRange(0,96), hogPCAwhiten, lp->resultPath, "HOG", "CV_PCA_DATA_AS_ROW");
	//	//fisher.newSamplePCAWhiten(tmpMat.colRange(96,204), hofPCAwhiten, lp->resultPath, "HOF", "CV_PCA_DATA_AS_ROW");
	//	//fisher.newSamplePCAWhiten(tmpMat.colRange(204,396), mbhPCAwhiten, lp->resultPath, "MBH", "CV_PCA_DATA_AS_ROW");
	//	//fisher.newSamplePCAWhiten(tmpMat.colRange(204,300), mbhxPCAwhiten, lp->resultPath, "MBHx", "CV_PCA_DATA_AS_ROW");
	//	//fisher.newSamplePCAWhiten(tmpMat.colRange(300,396), mbhyPCAwhiten, lp->resultPath, "MBHy", "CV_PCA_DATA_AS_ROW");
	//				
	//	localMat.push_back(combinePCAwhiten);
	//	
	//	t_start += t_stride;
	//	t_end += t_stride;
	//	denseMat.release();
	//}while(frameFlag != 1);	// ��ʱ�վ�β��������Ƶ���֡��ʱֹͣ����

	//if( localMat.rows == 0 )
	//	cout << video << ", localMat.cols:" << localMat.cols << endl;

	//// Fisher���루PCA�׻���
	//fisher.fisherEncode(localMat, localCode, lp->resultPath, lp->gmmNum, "Combine", manifoldType);

 ///////////////////////////////////////////////////////////
	// ����ʱ������ܲ�����PCA�׻�
	//fisher.newSamplePCAWhiten(tmpMat.colRange(3,tmpMat.cols), combinePCAwhiten, lp->resultPath, "Combine", "CV_PCA_DATA_AS_ROW");

	// ���û�н�������Ԥ����������������Ƶ��������
	if( lp->manifoldType == "raw" )
	{
		// L2��һ�����Ż�ŷ�Ͼ���Ӷ�����GMMѵ�����(��Ϊ����GMM��ͨ��ŷ�Ͼ���������������й����)
		for(int r = 0; r < tmpMat.rows; r++){
			normalize(tmpMat.row(r).colRange(0,96), tmpMat.row(r).colRange(0,96), 1.0, 0.0, NORM_L2);
			normalize(tmpMat.row(r).colRange(96,204), tmpMat.row(r).colRange(96,204), 1.0, 0.0, NORM_L2);
			normalize(tmpMat.row(r).colRange(204,396), tmpMat.row(r).colRange(204,396), 1.0, 0.0, NORM_L2);
		}

		// ��δ�����νṹԤ�����L2��һ��������ֱ�ӽ���Fisher���루396άHOG+HOF+MBH��
		if( lp->descriptorType == "d-Fusion" )
			fisher.fisherEncode(tmpMat, combineCode, lp->resultPath, lp->gmmNum, "Combine", lp->manifoldType);
		else if( lp->descriptorType == "r3-Fusion" ){
			fisher.fisherEncode(tmpMat.colRange(0,96), hogCode, lp->resultPath, lp->gmmNum, "HOG", lp->manifoldType);
			fisher.fisherEncode(tmpMat.colRange(96,204), hofCode, lp->resultPath, lp->gmmNum, "HOF", lp->manifoldType);
			fisher.fisherEncode(tmpMat.colRange(204,396), mbhCode, lp->resultPath, lp->gmmNum, "MBH", lp->manifoldType);
		}
	}
	// �������Ԥ����ʱ���νṹΪPCA �������������Ƶ��������
	else if( lp->manifoldType == "pca" )
	{
		// �ȶ�����Ԥ����PCA�׻�
		if( lp->descriptorType == "d-Fusion" )
			fisher.newSamplePCAWhiten(tmpMat, combinePCAwhiten, lp->resultPath, "Combine", "CV_PCA_DATA_AS_ROW");
		else if( lp->descriptorType == "r3-Fusion" ){
			fisher.newSamplePCAWhiten(tmpMat.colRange(0,96), hogPCAwhiten, lp->resultPath, "HOG", "CV_PCA_DATA_AS_ROW");
			fisher.newSamplePCAWhiten(tmpMat.colRange(96,204), hofPCAwhiten, lp->resultPath, "HOF", "CV_PCA_DATA_AS_ROW");
			fisher.newSamplePCAWhiten(tmpMat.colRange(204,396), mbhPCAwhiten, lp->resultPath, "MBH", "CV_PCA_DATA_AS_ROW");
		}
		else if( lp->descriptorType == "r4-Fusion" ){
			fisher.newSamplePCAWhiten(tmpMat.colRange(0,96), hogPCAwhiten, lp->resultPath, "HOG", "CV_PCA_DATA_AS_ROW");
			fisher.newSamplePCAWhiten(tmpMat.colRange(96,204), hofPCAwhiten, lp->resultPath, "HOF", "CV_PCA_DATA_AS_ROW");
			fisher.newSamplePCAWhiten(tmpMat.colRange(204,300), mbhxPCAwhiten, lp->resultPath, "MBHx", "CV_PCA_DATA_AS_ROW");
			fisher.newSamplePCAWhiten(tmpMat.colRange(300,396), mbhyPCAwhiten, lp->resultPath, "MBHy", "CV_PCA_DATA_AS_ROW");
		}

		// L2��һ�����Ż�ŷ�Ͼ���Ӷ�����GMMѵ�����(��Ϊ����GMM��ͨ��ŷ�Ͼ���������������й����)
		if( lp->descriptorType == "d-Fusion" )
			for(int r = 0; r < combinePCAwhiten.rows; r++)
				normalize(combinePCAwhiten.row(r), combinePCAwhiten.row(r), 1.0, 0.0, NORM_L2);
		else if( lp->descriptorType == "r3-Fusion" ){
			for(int r = 0; r < hogPCAwhiten.rows; r++){	// ���ڸ���������������ͬ������r<hogPCAwhiten�ɴ���ѭ������
				normalize(hogPCAwhiten.row(r), hogPCAwhiten.row(r), 1.0, 0.0, NORM_L2);
				normalize(hofPCAwhiten.row(r), hofPCAwhiten.row(r), 1.0, 0.0, NORM_L2);
				normalize(mbhPCAwhiten.row(r), mbhPCAwhiten.row(r), 1.0, 0.0, NORM_L2);
			}
		}
		else if( lp->descriptorType == "r4-Fusion" ){
			for(int r = 0; r < hogPCAwhiten.rows; r++){
				normalize(hogPCAwhiten.row(r), hogPCAwhiten.row(r), 1.0, 0.0, NORM_L2);
				normalize(hofPCAwhiten.row(r), hofPCAwhiten.row(r), 1.0, 0.0, NORM_L2);
				normalize(mbhxPCAwhiten.row(r), mbhxPCAwhiten.row(r), 1.0, 0.0, NORM_L2);
				normalize(mbhyPCAwhiten.row(r), mbhyPCAwhiten.row(r), 1.0, 0.0, NORM_L2);
			}
		}

		// �ٶ���PCA�׻���L2��һ����������������Fisher����
		if( lp->descriptorType == "d-Fusion" )
			fisher.fisherEncode(combinePCAwhiten, combineCode, lp->resultPath, lp->gmmNum, "Combine", lp->manifoldType);
		else if( lp->descriptorType == "r3-Fusion" ){
			fisher.fisherEncode(hogPCAwhiten, hogCode, lp->resultPath, lp->gmmNum, "HOG", lp->manifoldType );
			fisher.fisherEncode(hofPCAwhiten, hofCode, lp->resultPath, lp->gmmNum, "HOF", lp->manifoldType);
			fisher.fisherEncode(mbhPCAwhiten, mbhCode, lp->resultPath, lp->gmmNum, "MBH", lp->manifoldType);
		}
		else if( lp->descriptorType == "r4-Fusion" ){
			fisher.fisherEncode(hogPCAwhiten, hogCode, lp->resultPath, lp->gmmNum, "HOG", lp->manifoldType);
			fisher.fisherEncode(hofPCAwhiten, hofCode, lp->resultPath, lp->gmmNum, "HOF", lp->manifoldType);
			fisher.fisherEncode(mbhxPCAwhiten, mbhxCode, lp->resultPath, lp->gmmNum, "MBHx", lp->manifoldType);
			fisher.fisherEncode(mbhyPCAwhiten, mbhyCode, lp->resultPath, lp->gmmNum, "MBHy", lp->manifoldType);
		}
	}

	Mat code;
	// ���HOG+HOF+MBH��396ά��ͨ������������Ϊһ�������ӵ����
	if( lp->descriptorType == "d-Fusion" )
		code = combineCode;
	// ���ʱ������ܲ�����ȫ��+�ֲ�ʱ�վ� ���������������
	if( lp->descriptorType == "r2-Fusion" ){
		int colCount = combineCode.cols + localCode.cols;
		code.create(1, colCount, CV_32FC1);
		for(int j=0; j<combineCode.cols; j++)
			code.at<float>(0, j) = combineCode.at<float>(0, j);
		for(int j=0; j<localCode.cols; j++)
			code.at<float>(0, j + combineCode.cols) = localCode.at<float>(0, j);
	}

	// ���HOG��HOF��MBH ���������������
	if( lp->descriptorType == "r3-Fusion" ){
		int colCount = hogCode.cols + hofCode.cols + mbhCode.cols;
		code.create(1, colCount, CV_32FC1);
		for(int j=0; j<hogCode.cols; j++)
			code.at<float>(0, j) = hogCode.at<float>(0, j);
		for(int j=0; j<hofCode.cols; j++)
			code.at<float>(0, j + hogCode.cols) = hofCode.at<float>(0, j);
		for(int j=0; j<mbhCode.cols; j++)
			code.at<float>(0, j + hogCode.cols + hofCode.cols) = mbhCode.at<float>(0, j);
	}

	// ���HOG��HOF��MBHx��MBHy ���������������
	if( lp->descriptorType == "r4-Fusion" ){
		int colCount = hogCode.cols + hofCode.cols + mbhxCode.cols + mbhyCode.cols;
		code.create(1, colCount, CV_32FC1);
		for(int j=0; j<hogCode.cols; j++)
			code.at<float>(0, j) = hogCode.at<float>(0, j);
		for(int j=0; j<hofCode.cols; j++)
			code.at<float>(0, j + hogCode.cols) = hofCode.at<float>(0, j);
		for(int j=0; j<mbhxCode.cols; j++)
			code.at<float>(0, j + hogCode.cols + hofCode.cols) = mbhxCode.at<float>(0, j);
		for(int j=0; j<mbhyCode.cols; j++)
			code.at<float>(0, j + hogCode.cols + hofCode.cols + mbhxCode.cols) = mbhyCode.at<float>(0, j);
	}
 ///////////////////////////////////////////////////////////
	
	// ��Ƶ��Ӧ�������ӱ��
	fswrite << lp->actionTypeStr;
	// ����Ƶ����������д��xml
	fswrite << code;

	// �ͷ���Դ.....
	fsread.close();
	fswrite.release();	
	code.release();
	tmpMat.release(); denseMat.release(); localMat.release();
	hogCode.release(); hofCode.release(); mbhCode.release(); 
	mbhxCode.release(); mbhyCode.release(); combineCode.release();
	hogPCAwhiten.release();	hofPCAwhiten.release();	mbhPCAwhiten.release();
	mbhxPCAwhiten.release();mbhyPCAwhiten.release();combinePCAwhiten.release(); localCode.release();
	delete lp;

#ifdef _WIN32 // Windows version
	// ������ȡ�������ź���-1
	_InterlockedDecrement( &semaphore_fisher_quantization );
#else // Linux version
	sem_wait( &semaphore_fisher_quantization );
	pthread_detach(pthread_self());
#endif
		
	return 0;
}
	

//��GMM�õ�ѵ�����ݵĸ��ʷֲ�����������Ӧ��fisher vector
int FisherVector::trainGMM(vector<string> actionType, multimap<string, string> actionSet, 
						   int gmmNum, string resultPath, string descriptorType, string manifoldType)
{
	cout << "enter trainGMM��׼����ȡ������Ƶ��txt�����ļ�........" << endl;

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
		videoSampling.insert(map<string, float>::value_type(actionTypeStr, GMM_SAMPLE_NUM/videoNum+1));
	}

	FileStorage fs, fslog;
	ActionAnalysis action;
	Mat des, tmpMat, GMMdes;
	
	// Current time tick
	double t1 = double( getTickCount() );	
	// ���̶߳�ȡÿ�ද����������������д���Ӧ������xml��ʱ�ļ�
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{	
		unsigned int sampleNum = videoSampling[*itype];	
		ThreadParam *thrd = new ThreadParam();
		thrd->actionSet = actionSet;
		thrd->resultPath = resultPath;
		thrd->featurePath = resultPath + "features" + spritLabel + *itype + spritLabel;
		thrd->actionTypeStr = *itype;
		thrd->sampleNum = sampleNum;

#ifdef _WIN32 // Windows version
		SYSTEM_INFO theSystemInfo;
		::GetSystemInfo(&theSystemInfo);
		while( semaphore_readFeaturesFormEachAction >= theSystemInfo.dwNumberOfProcessors)
			Sleep( 1000 );
		HANDLE hThread = CreateThread(NULL, 0, readFeaturesFormEachAction, thrd, 0, NULL);
		if(hThread == NULL){
			cout << "Create Thread failed in trainGMM() !" << endl;
			delete thrd;
			return -1;
		}
		_InterlockedIncrement( &semaphore_readFeaturesFormEachAction );
#else // Linux version
		int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
		int semaNum;
		sem_getvalue(&semaphore_readFeaturesFormEachAction, &semaNum);
		while( semaNum >= NUM_PROCS){
			sleep( 1 );
			sem_getvalue(&semaphore_readFeaturesFormEachAction, &semaNum);
		}
		pthread_t pthID;
		int ret = pthread_create(&pthID, NULL, readFeaturesFormEachAction, thrd);
		if(ret){
			cout << "Create Thread failed in trainGMM() !" << endl;
			delete thrd;
			return -1;
		}
		sem_post( &semaphore_readFeaturesFormEachAction );
#endif
	}// for itype

	// ��ֹ����forѭ��������������Ͻ������������Ϊ�ź���semaphore���ܻ���Ϊ0��
	// ��ʱ�в����߳����ڹ���δ�ͷ��ź���������Ӧ�����ź����Ƿ�Ϊ0���ж��������Ƿ������
#ifdef _WIN32 // Windows version
	while( semaphore_readFeaturesFormEachAction )
		Sleep( 1000 );
#else // Linux version
	int semaNum;
	sem_getvalue(&semaphore_readFeaturesFormEachAction, &semaNum);
	while( semaNum ){
		sleep( 1 );
		sem_getvalue(&semaphore_readFeaturesFormEachAction, &semaNum);
	}
#endif

	// Current time tick
	double t2 = double( getTickCount() );	
	t1 = ( (double)getTickCount() - t1 ) / getTickFrequency();
	cout << endl << "�ѽ����ж����Ĳ�������ȫ��д��xml�ļ�......" << endl;	
	// ��ȡÿ�ද��������xml��ʱ�ļ�����GMM_SAMPLE_NUM�������ŵ��ڴ棬ѵ��GMM�뱾
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{	
		//if(*itype != "climb")
		//	continue;
		string file = resultPath + *itype + ".xml";
		fs.open(file, FileStorage::READ);
		fs["features"] >> tmpMat;
		//CV_Assert( !tmpMat.empty() );	

		RNG rng(getTickCount());	// ȡ��ǰϵͳʱ����Ϊ���������
		Mat randMat(tmpMat.rows, 1, CV_32S);
		for(int r=0; r<randMat.rows; r++)
			randMat.at<int>(r,0) = r;
		randShuffle(randMat, 1, &rng);

		// ÿ�ද����ָ������GMM_SAMPLE_NUM/CATEGORY_NUM+1�������㣨���ѡ̫������ѵ���뱾���ڴ�װ���»������
		int CATEGORY_NUM = actionType.size();
		int loop = GMM_SAMPLE_NUM/CATEGORY_NUM+1;
		cout << "loop:" << loop << ", tmpMat.rows:" << tmpMat.rows << endl;
		if( loop < tmpMat.rows ){	
			for(int r=0; r<loop; r++){	
				int randRow = randMat.at<int>(r,0);
				GMMdes.push_back(tmpMat.row(randRow));
			}
			cout << *itype << ", sampleNum(loop) < tmpMat.rows, ��ʼ���������" << endl;
		}
		else
			GMMdes.push_back(tmpMat);
		
		tmpMat.release();
		// delete txt for saving disk storage
		string cmdDel = "rm " + resultPath + *itype + ".xml";
		system(cmdDel.c_str());
	}

	cout << "�ѽ����ж��������������ڴ棬׼��ѵ��gmm......"<< endl;
	cout << "GMMdes.rows:" << GMMdes.rows << ", GMMdes.cols:" << GMMdes.cols << endl;
//////////////////////////////////////////////////////////////////
	// ��ͨ���ֱ�ѵ���뱾	
	vector<string> cb_name; 

	if( descriptorType == "d-Fusion" )
		cb_name.push_back("Combine");
	else if( descriptorType == "r3-Fusion" ){
		cb_name.push_back("HOG");
		cb_name.push_back("HOF");
		cb_name.push_back("MBH");
	}
	else if( descriptorType == "r4-Fusion" ){
		cb_name.push_back("HOG");
		cb_name.push_back("HOF");
		cb_name.push_back("MBHx");
		cb_name.push_back("MBHy");
	}
	vector<string>::iterator itcb;
	for(itcb = cb_name.begin(); itcb != cb_name.end(); itcb++)
	{	
		CodebookParam *thrd = new CodebookParam();
		thrd->cbname = *itcb;
		thrd->featuresMat = GMMdes;
		thrd->resultPath = resultPath;
		thrd->manifoldType = manifoldType;
		thrd->gmmNum = gmmNum;

#ifdef _WIN32 // Windows version
		SYSTEM_INFO theSystemInfo;
		::GetSystemInfo(&theSystemInfo);
		while( semaphore_trainMulChGMM >= theSystemInfo.dwNumberOfProcessors)
			Sleep( 1000 );

		HANDLE hThread = CreateThread(NULL, 0, trainMulChGMM, thrd, 0, NULL);
		if(hThread == NULL){
			cout << "Create Thread failed in trainGMM() !" << endl;
			delete thrd;
			return -1;
		}
		_InterlockedIncrement( &semaphore_trainMulChGMM );
#else // Linux version
		int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
		int semaNum;
		sem_getvalue(&semaphore_trainMulChGMM, &semaNum);
		while( semaNum >= NUM_PROCS){
			sleep( 1 );
			sem_getvalue(&semaphore_trainMulChGMM, &semaNum);
		}
		pthread_t pthID;
		int ret = pthread_create(&pthID, NULL, trainMulChGMM, thrd);
		if(ret)	{
			cout << "Create Thread failed in trainGMM() !" << endl;
			delete thrd;
			return -1;
		}
		sem_post( &semaphore_trainMulChGMM );
#endif
	}// for itcb
	
	// ��ֹ����forѭ��������������Ͻ������������Ϊ�ź���semaphore���ܻ���Ϊ0��
	// ��ʱ�в����߳����ڹ���δ�ͷ��ź���������Ӧ�����ź����Ƿ�Ϊ0���ж��������Ƿ������
#ifdef _WIN32 // Windows version
	while( semaphore_trainMulChGMM )
		Sleep( 1000 );
#else // Linux version
	sem_getvalue(&semaphore_trainMulChGMM, &semaNum);
	while( semaNum ){
		sleep( 1 );
		sem_getvalue(&semaphore_trainMulChGMM, &semaNum);
	}
#endif
//////////////////////////////////////////////////////////////////
	//Mat hogPCAwhiten, hofPCAwhiten, mbhPCAwhiten, mbhxPCAwhiten, mbhyPCAwhiten, combinePCAwhiten;

	//// �ȶ���������Ԥ����PCA
	//pcaWhiten(GMMdes, combinePCAwhiten, resultPath, 200, "Combine", "CV_PCA_DATA_AS_ROW");
	////pcaWhiten(GMMdes.colRange(0,96), hogPCAwhiten, resultPath, 48, "HOG", "CV_PCA_DATA_AS_ROW");
	////pcaWhiten(GMMdes.colRange(96,204), hofPCAwhiten, resultPath, 54, "HOF", "CV_PCA_DATA_AS_ROW");
	////pcaWhiten(GMMdes.colRange(204,396), mbhPCAwhiten, resultPath, 96, "MBH", "CV_PCA_DATA_AS_ROW");
	////pcaWhiten(GMMdes.colRange(204,300), mbhxPCAwhiten, resultPath, 48, "MBHx", "CV_PCA_DATA_AS_ROW");
	////pcaWhiten(GMMdes.colRange(300,396), mbhyPCAwhiten, resultPath, 48, "MBHy", "CV_PCA_DATA_AS_ROW");

	//// ��ѵ���뱾��GMM����
	//gmmCluster(combinePCAwhiten, resultPath, gmmNum, "Combine", manifoldType);
	////gmmCluster(hogPCAwhiten, resultPath, gmmNum, "HOG", manifoldType);
	////gmmCluster(hofPCAwhiten, resultPath, gmmNum, "HOF", manifoldType);
	////gmmCluster(mbhPCAwhiten, resultPath, gmmNum, "MBH", manifoldType);
	////gmmCluster(mbhxPCAwhiten, resultPath, gmmNum, "MBHx", manifoldType);
	////gmmCluster(mbhyPCAwhiten, resultPath, gmmNum, "MBHy", manifoldType);

	//// ��ѵ���뱾��GMM���ࣨδPCA�׻���
	////gmmCluster(GMMdes, resultPath, gmmNum, "Combine", manifoldType); 
	////gmmCluster(GMMdes.colRange(0,96), resultPath, gmmNum, "HOG", manifoldType);
	////gmmCluster(GMMdes.colRange(96,204), resultPath, gmmNum, "HOF", manifoldType);
	////gmmCluster(GMMdes.colRange(204,396), resultPath, gmmNum,"MBH", manifoldType);

	//cout << "Combine:combinePCAwhiten.rows:" << combinePCAwhiten.rows << ", combinePCAwhiten.cols:" << combinePCAwhiten.cols << endl;
	////cout << "HOG:hogPCAwhiten.rows:" << hogPCAwhiten.rows << ", hogPCAwhiten.cols:" << hogPCAwhiten.cols << endl;
	////cout << "HOF:hofPCAwhiten.rows:" << hofPCAwhiten.rows << ", hofPCAwhiten.cols:" << hofPCAwhiten.cols << endl;
	////cout << "MBH:mbhPCAwhiten.rows:" << mbhPCAwhiten.rows << ", mbhPCAwhiten.cols:" << mbhPCAwhiten.cols << endl;
	////cout << "MBH:mbhxPCAwhiten.rows:" << mbhxPCAwhiten.rows << ", mbhxPCAwhiten.cols:" << mbhxPCAwhiten.cols << endl;
	////cout << "MBH:mbhyPCAwhiten.rows:" << mbhyPCAwhiten.rows << ", mbhyPCAwhiten.cols:" << mbhyPCAwhiten.cols << endl;
//////////////////////////////////////////////////////////////////

	// ��¼���뱾��ʱ�俪��
	t2 = ( (double)getTickCount() - t2 ) / getTickFrequency();
	stringstream strGmmNum;
	strGmmNum << gmmNum; 
	string file_loger= resultPath + "log_GMM" + strGmmNum.str() + ".xml";
	fslog.open(file_loger, FileStorage::WRITE);
	fslog << "FulliScale_GMM" << "{" ;
	fslog << "read_time_hours" << t1/3600;
	fslog << "gmm_time_hours" << t2/3600;
	fslog << "}";

	// �ͷ���Դ
	GMMdes.release();
	//hogPCAwhiten.release();	hofPCAwhiten.release();	mbhPCAwhiten.release();
	//mbhxPCAwhiten.release();mbhyPCAwhiten.release();combinePCAwhiten.release();
	return 0;
}

int FisherVector::getFisherVector(vector<string> actionType, multimap<string, string> actionSet, 
								  int gmmNum, string datasetPath,string vidcodePath, string resultPath, 
								  string processType, string descriptorType, string manifoldType)
{
	float totalframes = 0;		
	int video_num = 0;
	// Current time tick
	double t = double( getTickCount() );

	cout << processType << "Set begin for getFisherVector...." << endl;

	vector<string>::iterator itype;
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{		
		string actionTypeStr = *itype;
		multimap<string, string>::iterator iter;
		for(iter = actionSet.begin(); iter != actionSet.end(); iter++)
		{
			if( iter->first == actionTypeStr )
			{
				//if(iter->second != "Finding_Forrester_1_shoot_ball_f_nm_np1_le_bad_3" &&  
				//	iter->second != "Re-_Show_your_Smile!_7_smile_h_nm_np1_fr_med_0")
				//	continue;
				//if(iter->second != "-_FREE_HUGS_-_Abrazos_Gratis_www_abrazosgratis_org_hug_u_cm_np2_le_goo_11")
				//	continue;
				ThreadParam *thrd = new ThreadParam();
				thrd->datasetPath = datasetPath;
				thrd->featurePath = resultPath + "features" + spritLabel + actionTypeStr + spritLabel;
				thrd->vidcodePath = vidcodePath;
				thrd->resultPath = resultPath;
				thrd->filename = iter->second;
				thrd->actionTypeStr = actionTypeStr;
				thrd->descriptorType = descriptorType;
				thrd->manifoldType = manifoldType;
				thrd->gmmNum = gmmNum;

	#ifdef _WIN32 // Windows version
				SYSTEM_INFO theSystemInfo;
				::GetSystemInfo(&theSystemInfo);
				while( semaphore_fisher_quantization >= theSystemInfo.dwNumberOfProcessors)
					Sleep( 1000 );

				HANDLE hThread = CreateThread(NULL, 0, featuresFisherQuantization, thrd, 0, NULL);
				if(hThread == NULL)	{
					cout << "Create Thread failed in featuresFisherQuantization !" << endl;
					delete thrd;
					return -1;
				}
				_InterlockedIncrement( &semaphore_fisher_quantization );
	#else // Linux version
				int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
				int semaNum;
				sem_getvalue(&semaphore_fisher_quantization, &semaNum);
				while( semaNum >= NUM_PROCS){
					sleep( 1 );
					sem_getvalue(&semaphore_fisher_quantization, &semaNum);
				}
				pthread_t pthID;
				int ret = pthread_create(&pthID, NULL, featuresFisherQuantization, thrd);
				if(ret)	{
					cout << "Create Thread failed in featuresFisherQuantization !" << endl;
					delete thrd;
					return -1;
				}
				sem_post( &semaphore_fisher_quantization );
	#endif
				// ͳ���Ѵ����֡������Ƶ��
				//totalframes += frame_counter[idx];
				video_num++;	
			}// if iter->first
		}//	for iter		
	}// for itype

	cout << processType << "Set end of getFisherVector...." << endl;

	// ��ֹ����forѭ��������������Ͻ������������Ϊ�ź���semaphore���ܻ���Ϊ0��
	// ��ʱ�в����߳����ڹ���δ�ͷ��ź���������Ӧ�����ź����Ƿ�Ϊ0���ж��������Ƿ������
#ifdef _WIN32 // Windows version
	while( semaphore_fisher_quantization )
		Sleep( 1000 );
#else // Linux version
	int semaNum;
	sem_getvalue(&semaphore_fisher_quantization, &semaNum);
	while( semaNum ){
		sleep( 1 );
		sem_getvalue(&semaphore_fisher_quantization, &semaNum);
	}
#endif
	// ��¼ѵ����������ʱ�俪��
	FileStorage fslog;
	string file_loger= resultPath + "log_" + processType + "FisherCode.xml";
	t = ( (double)getTickCount() - t ) / getTickFrequency();
	fslog.open(file_loger, FileStorage::WRITE);
	fslog << "getFisherVector" << "{" ;
	fslog << "time_hours" << t/3600;
	fslog << "video_num" << video_num;
	//fslog << "average_FPS" << totalframes / t;
	fslog << "}";
	fslog.release();

	return 0;
}

void FisherVector::bulidGMM_Data(Mat src, float* data){
	CV_Assert( !src.empty() );
	for(int i = 0; i < src.rows; i++)
		for(int j = 0; j < src.cols; j++)
			*(data + i*src.cols + j) = src.at<float>(i, j);		
}

void FisherVector::setGMM(string file, VlGMM* gmm, string descriptorType, string manifoldType)
{	
	FileStorage fs;
	int dimension = vl_gmm_get_dimension(gmm);
	int numClusters = vl_gmm_get_num_clusters(gmm);

	//�����õ�3��������means, covariances, priors
	fs.open(file + "gmm" + manifoldType + descriptorType + ".xml",FileStorage::WRITE);
	Mat mat;
	mat.create(1,dimension*numClusters,CV_32F);
	float *means= (float *)vl_gmm_get_means(gmm);
	for(int i=0;i<dimension*numClusters;i++)
		mat.at<float>(0,i)=*(means+i);
	fs<<"means"<<mat;
	mat.release();

	mat.create(1,dimension*numClusters,CV_32F);
	float *covariances=(float *)vl_gmm_get_covariances(gmm);
	for(int i=0;i<dimension*numClusters;i++)
		mat.at<float>(0,i)=*(covariances+i);
	fs<<"covariances"<<mat;
	mat.release();
	
	mat.create(1,numClusters,CV_32F);
	float *priors=(float *)vl_gmm_get_priors(gmm);
	for(int i=0;i<numClusters;i++)
		mat.at<float>(0,i)=*(priors+i);
	fs<<"priors"<<mat;

	mat.release();
	fs.release();
}

void FisherVector::getGMM(string file, VlGMM* &gmm, string descriptorType, string manifoldType)
{
	cout << "enter getGMM()..." << endl;

	FileStorage fs;
	//�����õ�3��������means, covariances, priors
	fs.open(file + "gmm" + manifoldType + descriptorType + ".xml", FileStorage::READ);
	Mat mat;
	fs["means"]>>mat;
	float *means=(float*)malloc(sizeof(float)*mat.cols);
	for(int i=0;i<mat.cols;i++)
		*(means+i)=mat.at<float>(0,i);
	vl_gmm_set_means(gmm,means);

	fs["covariances"]>>mat;
	float *covariances=(float*)malloc(sizeof(float)*mat.cols);
	for(int i=0;i<mat.cols;i++)
		*(covariances+i)=mat.at<float>(0,i);
	vl_gmm_set_covariances(gmm,covariances);

	fs["priors"]>>mat;
	float *priors=(float*)malloc(sizeof(float)*mat.cols);
	for(int i=0;i<mat.cols;i++)
		*(priors+i)=mat.at<float>(0,i);
	vl_gmm_set_priors(gmm,priors);

	mat.release();
	fs.release();
	free(means);
	free(covariances);
	free(priors);
}

int FisherVector::fisherEncode(Mat src, Mat& dst, string resultPath, int gmmNum, string descriptorType, string manifoldType)
{
	FisherVector fisher;

	//��ѵ����ת�浽�����ռ�
	float *data = (float*)malloc(src.cols*sizeof(float)*src.rows);
	fisher.bulidGMM_Data(src, data);
	//����������ά��
	int dimension = src.cols;
	//��ϸ�˹��ά��K����gmm�������ĵĸ���
	int numClusters = gmmNum;//64;
	VlGMM *gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numClusters);

	fisher.getGMM(resultPath, gmm, descriptorType, manifoldType);

	//fisher ����������
	float* enc;
	//�����������������
	int numDataToEncode = src.rows;
	// ����ռ䣬���ڴ洢һ����Ƶ��Ӧ��fisher��������
	enc = (float*) vl_malloc(sizeof(float) * 2 * dimension * numClusters);

	cout << "111111111..." << endl;

	Mat code(1, 2*dimension*numClusters, CV_32FC1, Scalar(0));	//���FisherVector
	
	vl_fisher_encode(enc, VL_TYPE_FLOAT,
		vl_gmm_get_means(gmm), dimension, numClusters,
		vl_gmm_get_covariances(gmm),
		vl_gmm_get_priors(gmm),
		data, numDataToEncode,
		VL_FISHER_FLAG_IMPROVED
		) ;

	cout << "2222222222..." << endl;

	for(int i=0;i<code.cols;i++)
		code.at<float>(0,i) = *(enc+i);

	dst = code;
	
	// �ͷ���Դ.....
	vl_free(enc);
	free(data);		
	free(gmm);
	code.release();

	return 0;
}

int FisherVector::gmmCluster(Mat src, string resultPath, int gmmNum, string descriptorType, string manifoldType)
{
	FisherVector fisher;

	//��ѵ����ת�浽�����ռ�
	float *data = (float*)malloc(sizeof(float)*src.rows*src.cols);
	fisher.bulidGMM_Data(src, data);

	//����������ά������TXT�����ļ��ж�ȡ��
	int dimension = src.cols;
	//��ϸ�˹��ά��
	int numClusters = gmmNum;//64;
	//����ѵ����GMM
	VlGMM *gmm= vl_gmm_new(VL_TYPE_FLOAT, dimension, numClusters);

	//������������ѵ�����õ���ϸ�˹�ֲ�
	vl_size numData = src.rows;
	src.release();

	cout << descriptorType << "��ʼgmmѵ��......";
	vl_gmm_cluster (gmm, data, numData);
	cout << descriptorType << "gmmѵ�����!" << endl;
	free(data);

	//�ɻ�ϸ�˹ģ�ͼ���ʱ��ϳ�������������л����ļ��У��Ա�����õ�ʱ�ָ�
	fisher.setGMM(resultPath, gmm, descriptorType, manifoldType);

	return 0;
}

// PCA�׻���������ת����PCA�׻��������ռ䣩��������Ƶ�ڱ������������У����ݶ�Ӧ������descriptorType��PCA�׻�ͶӰ������������ռ�ת��
// ע�⣺����PCA_DATA_TYPEֻ�����Ѹú���ֻ������������������δ�õ��ò���
int FisherVector::newSamplePCAWhiten(Mat src, Mat& dst, string resultPath, string descriptorType, string PCA_DATA_TYPE)
{
	//PCA�׻��õ�3��������means��ֵ, diag�ԽǾ���, projͶӰ����
	Mat meanMat, diagMat, U;
	FileStorage fspca;
	string pca_file = resultPath + "pca" + descriptorType + ".xml";
	fspca.open(pca_file, FileStorage::READ);
	fspca["means"] >> meanMat;
	fspca["diag"] >> diagMat;
	fspca["proj"] >> U;

    // ����ȥ��ֵ���γ��������ֵ����f
	for(int i=0; i<src.rows; i++)
	for(int j=0; j<src.cols; j++)
		src.at<float>(i,j) -= meanMat.at<float>(0,j);

	// PCA�׻������ f �����������洢���������ֵ����������׻���ʽΪ��  X = U'*f*��    ��2��
	// U ΪͶӰת������fΪ��������src��ÿ��Ԫ�ؼ�ȥÿ�еľ�ֵ�õ����¾���.
    dst = src * U * diagMat;
	//dst = src * U; // ���׻�

	fspca.release();
	meanMat.release();
	diagMat.release();
	src.release();
	U.release();
	return 0;
}

// PCA�׻���ָ����ά��pcaNum��ά������srcת����PCA�׻��������ռ��γ�dst���ú���Ŀǰֻ֧��������������PCA����������������PCA��ο�pcaWhitenOld������
int FisherVector::pcaWhiten(Mat src, Mat& dst, string resultPath, int pcaNum, string descriptorType, string PCA_DATA_TYPE)
{
	Mat projMat, diagMat;
	PCA pca(src, noArray(), CV_PCA_DATA_AS_ROW, pcaNum);
    pca.project(src, projMat);

    Mat sqrtEigMat(pca.eigenvalues.rows, pca.eigenvalues.cols, CV_32F);
    float eplison = 0.00001;    //eplisonΪ��ֹ��ĸΪ0

    // �γ��������ֵ�ĶԽ��߾������diagMatΪ�ԽǾ����
	// ������˵������ֵ����pca.eigenvalues����ָsrcȥ��ֵ��õ��ľ���f ��Э������������ֵ��
	// ͶӰ����U������������pca.eigenvectors.t()����ָָsrcȥ��ֵ��õ��ľ���f ��Э������������������
    for(int i=0; i<pca.eigenvalues.rows; i++)
        for(int j=0; j<pca.eigenvalues.cols; j++)
            sqrtEigMat.at<float>(i,j) = 1.0 / sqrt(pca.eigenvalues.at<float>(i,j) + eplison);
    diagMat = Mat::diag(sqrtEigMat);

	// �׻��������������f �����������洢������������׻���ʽΪ��  X = U'*f*��    ��2��
	dst = projMat * diagMat;
	//dst = projMat; // ���׻�

	FileStorage fspca;
	string pca_file = resultPath + "pca" + descriptorType + ".xml";
	//�����õ�3��������means, �ԽǾ���diag, ͶӰ����U
	fspca.open(pca_file, FileStorage::WRITE);
	fspca << "means" << pca.mean;
	fspca << "diag" << diagMat;
	fspca << "proj" << pca.eigenvectors.t();

	//fstream fsProj;	
	//string fileRot = resultPath + "features" + spritLabel + "00PCAwhiten.txt"; 
	//fsProj.open(fileRot, ios::out);	
	//fsProj<<dst;
	//fsProj.close();

	fspca.release();
	projMat.release();
	diagMat.release();
	sqrtEigMat.release();
	return 0;
}