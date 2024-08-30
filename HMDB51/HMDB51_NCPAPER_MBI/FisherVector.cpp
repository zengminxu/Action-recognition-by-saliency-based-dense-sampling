#include "ActionAnalysis.h"
#include "FisherVector.h"

using namespace cv;
using namespace std;

extern string spritLabel;	//使用DenseTrackStab.cpp的全局变量

// 多线程同时对HOG/HOF/MBHx/MBHy 多通道进行码本训练
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
	
	// 添加各通道的训练数据
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

		// L2归一化：优化欧氏距离从而改善GMM训练结果(因为猜想GMM是通过欧氏距离来将样本点进行归类的)
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

		// 未PCA白化的训练码本：GMM聚类（396维HOG+HOF+MBH）
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

		// 先对样本预处理：PCA白化
		fisher.pcaWhiten(GMMdes, PCAwhiten, lp->resultPath, pcaNum, lp->cbname, "CV_PCA_DATA_AS_ROW");
		// L2归一化：优化欧氏距离从而改善GMM训练结果(因为猜想GMM是通过欧氏距离来将样本点进行归类的)
		for(int r = 0; r < PCAwhiten.rows; r++)
			normalize(PCAwhiten.row(r), PCAwhiten.row(r), 1.0, 0.0, NORM_L2);
		// 已PCA白化的训练码本：GMM聚类
		fisher.gmmCluster(PCAwhiten, lp->resultPath, lp->gmmNum, lp->cbname, lp->manifoldType);
		cout << lp->cbname << ":PCAwhiten.rows:" << PCAwhiten.rows << ", PCAwhiten.cols:" << PCAwhiten.cols << endl;
	}
	
	// 释放资源.....
	delete lp;

#ifdef _WIN32 // Windows version
	// 结束提取特征，信号量-1
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

	// 先读取待训练的特征向量，存于txt文件，作为gmm的输入数据
	FisherVector fisher;
	ActionAnalysis action;
	Mat des, tmpMat, pcaMat;

	// 从目录下顺序读取所有特征描述符，并合并到一个矩阵中	
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

			RNG rng(getTickCount());	// 取当前系统时间作为随机数种子
			Mat randMat(tmpMat.rows, 1, CV_32S);
			for(int r=0; r<randMat.rows; r++)
				randMat.at<int>(r,0) = r;
			randShuffle(randMat, 1, &rng);
	
			if( tmpMat.rows > lp->sampleNum ){	// 每类动作只取SAMPLE_NUM/CATEGORY_NUM个特征点
				for(int r=0; r<lp->sampleNum; r++){	
					int randRow = randMat.at<int>(r,0);
					des.push_back(tmpMat.row(randRow));
				}
				cout << lp->actionTypeStr << ", 多线程读取单个视频时，遇到随机采样..." << ", tmpMat.rows:" << tmpMat.rows << ", lp->sampleNum:" << lp->sampleNum << endl;
			}
			else
				des.push_back(tmpMat);

		}//if iter->first
		pcaMat.release();
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
		_InterlockedDecrement( &semaphore_readFeaturesFormEachAction );
#else // Linux version
		sem_wait( &semaphore_readFeaturesFormEachAction );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}

	// 将每类动作的特征写入相应的xml文件
	fs << "features" << des;
	fs.release();

	// 释放资源.....
	des.release();
	fs.release();
	delete lp;

#ifdef _WIN32 // Windows version
	// 结束提取特征，信号量-1
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

	// 开始量化特征
	FileStorage fs;
	FisherVector fisher;
	ActionAnalysis action;

	// 读取每个视频的TXT特征描述符
	ifstream fsread;
	FileStorage fswrite;
	string feature_files = lp->featurePath + lp->filename + ".bin";
	// 写入每个视频量化后的特征
	string vidcode_files = lp->vidcodePath + lp->filename + "_quantization.xml";
	fsread.open(feature_files.c_str(), ios::in);
	fswrite.open(vidcode_files, FileStorage::WRITE);

	if(!fsread.is_open() || !fswrite.isOpened())
	{
		cout << "Error: Could not open feature/quantization file in featuresFisherQuantization()." << endl;
		delete lp;
#ifdef _WIN32 // Windows version
		// 结束提取特征，信号量-1
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

	// 读取396维HOG+HOF+MBH特征
	action.bin2Mat(feature_files, tmpMat, "features");
	// 稠密采样子时空卷
	//action.bin2Mat(feature_files, tmpMat, "denseSample");
	
	if( tmpMat.empty() )
	{
		cout << "Warning: features is empty. ";
		int fvdim = 0;
		if( lp->descriptorType == "d-Fusion" )
			fvdim = 200*2*lp->gmmNum;
		else if( lp->descriptorType == "r3-Fusion" || lp->descriptorType == "r4-Fusion")
			fvdim = 198*2*lp->gmmNum;
		// 视频对应的特征子标号
		fswrite << lp->actionTypeStr;
		Mat decimalMat(1, fvdim, CV_32FC1, Scalar(0));
		// 点加：将空码字填一个特定的小数
		decimalMat = decimalMat + 1./fvdim;
		fswrite << decimalMat;
		cout << "code 1row x " << decimalMat.cols << "cols has been filled decimal in " << feature_files << endl;
		fswrite.release();
		delete lp;
#ifdef _WIN32 // Windows version
		// 结束提取特征，信号量-1
		_InterlockedDecrement( &semaphore_fisher_quantization );
#else // Linux version
		sem_wait( &semaphore_fisher_quantization );
		pthread_detach(pthread_self());
#endif		
		return 0;
	}
	
 /////////////////////////////////////////////////////////// 
	//// 稠密采样子时空卷
	//VideoCapture capture;
	//string video = lp->datasetPath + lp->actionTypeStr + spritLabel + lp->filename + ".avi";
	//capture.open(video);
	//int frameH = 0, frameW = 0, frameNum = 0;
	//frameH = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	//frameW = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	//frameNum = capture.get(CV_CAP_PROP_FRAME_COUNT);

	//int t_ind = 0, t_stride = 10, t_start = 0, t_end = 0, frameFlag = 0;
	//int tsize_3 = frameNum / 3;
	////for(int tsize = 0; tsize < frameNum; tsize += tsize_3 ){	// 该for循环是针对时间轴上多个尺度的方法

	//t_end = tsize_3;	
	//// 只取时间轴上的一个尺度
	//do{		 
	//	if( t_end >= frameNum ){ // 设定子时空卷尾部不超过视频最大帧数
	//		t_end = frameNum;
	//		frameFlag = 1;
	//	}
	//	for(int r = 0; r < tmpMat.rows; r++){	// 根据特征轨迹的帧进行稠密采样
	//		t_ind = tmpMat.at<float>(r,0);
	//		if( t_start<t_ind && t_ind<=t_end )
	//			denseMat.push_back(tmpMat.row(r));
	//	}// for r
	//	if(denseMat.rows == 0)	// 排除个别视频稠密采样为0的情况
	//		denseMat = tmpMat.clone();
	//	// PCA白化
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
	//}while(frameFlag != 1);	// 子时空卷尾部超过视频最大帧数时停止采样

	//if( localMat.rows == 0 )
	//	cout << video << ", localMat.cols:" << localMat.cols << endl;

	//// Fisher编码（PCA白化）
	//fisher.fisherEncode(localMat, localCode, lp->resultPath, lp->gmmNum, "Combine", manifoldType);

 ///////////////////////////////////////////////////////////
	// 考虑时间轴稠密采样：PCA白化
	//fisher.newSamplePCAWhiten(tmpMat.colRange(3,tmpMat.cols), combinePCAwhiten, lp->resultPath, "Combine", "CV_PCA_DATA_AS_ROW");

	// 针对没有进行特征预处理的情况，进行视频编码量化
	if( lp->manifoldType == "raw" )
	{
		// L2归一化：优化欧氏距离从而改善GMM训练结果(因为猜想GMM是通过欧氏距离来将样本点进行归类的)
		for(int r = 0; r < tmpMat.rows; r++){
			normalize(tmpMat.row(r).colRange(0,96), tmpMat.row(r).colRange(0,96), 1.0, 0.0, NORM_L2);
			normalize(tmpMat.row(r).colRange(96,204), tmpMat.row(r).colRange(96,204), 1.0, 0.0, NORM_L2);
			normalize(tmpMat.row(r).colRange(204,396), tmpMat.row(r).colRange(204,396), 1.0, 0.0, NORM_L2);
		}

		// 对未经流形结构预处理的L2归一化特征，直接进行Fisher编码（396维HOG+HOF+MBH）
		if( lp->descriptorType == "d-Fusion" )
			fisher.fisherEncode(tmpMat, combineCode, lp->resultPath, lp->gmmNum, "Combine", lp->manifoldType);
		else if( lp->descriptorType == "r3-Fusion" ){
			fisher.fisherEncode(tmpMat.colRange(0,96), hogCode, lp->resultPath, lp->gmmNum, "HOG", lp->manifoldType);
			fisher.fisherEncode(tmpMat.colRange(96,204), hofCode, lp->resultPath, lp->gmmNum, "HOF", lp->manifoldType);
			fisher.fisherEncode(tmpMat.colRange(204,396), mbhCode, lp->resultPath, lp->gmmNum, "MBH", lp->manifoldType);
		}
	}
	// 针对特征预处理时流形结构为PCA 的情况，进行视频编码量化
	else if( lp->manifoldType == "pca" )
	{
		// 先对特征预处理：PCA白化
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

		// L2归一化：优化欧氏距离从而改善GMM训练结果(因为猜想GMM是通过欧氏距离来将样本点进行归类的)
		if( lp->descriptorType == "d-Fusion" )
			for(int r = 0; r < combinePCAwhiten.rows; r++)
				normalize(combinePCAwhiten.row(r), combinePCAwhiten.row(r), 1.0, 0.0, NORM_L2);
		else if( lp->descriptorType == "r3-Fusion" ){
			for(int r = 0; r < hogPCAwhiten.rows; r++){	// 由于各描述符的行数相同，所以r<hogPCAwhiten可代表循环次数
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

		// 再对已PCA白化、L2归一化的新特征，进行Fisher编码
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
	// 针对HOG+HOF+MBH（396维单通道长向量）作为一个描述子的情况
	if( lp->descriptorType == "d-Fusion" )
		code = combineCode;
	// 针对时间轴稠密采样：全局+局部时空卷 这两种描述子组合
	if( lp->descriptorType == "r2-Fusion" ){
		int colCount = combineCode.cols + localCode.cols;
		code.create(1, colCount, CV_32FC1);
		for(int j=0; j<combineCode.cols; j++)
			code.at<float>(0, j) = combineCode.at<float>(0, j);
		for(int j=0; j<localCode.cols; j++)
			code.at<float>(0, j + combineCode.cols) = localCode.at<float>(0, j);
	}

	// 针对HOG、HOF、MBH 这三种描述子组合
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

	// 针对HOG、HOF、MBHx、MBHy 这四种描述子组合
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
	
	// 视频对应的特征子标号
	fswrite << lp->actionTypeStr;
	// 将视频编码后的码字写入xml
	fswrite << code;

	// 释放资源.....
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
	// 结束提取特征，信号量-1
	_InterlockedDecrement( &semaphore_fisher_quantization );
#else // Linux version
	sem_wait( &semaphore_fisher_quantization );
	pthread_detach(pthread_self());
#endif
		
	return 0;
}
	

//用GMM得到训练数据的概率分布，再求编码对应的fisher vector
int FisherVector::trainGMM(vector<string> actionType, multimap<string, string> actionSet, 
						   int gmmNum, string resultPath, string descriptorType, string manifoldType)
{
	cout << "enter trainGMM，准备读取所有视频的txt特征文件........" << endl;

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
		videoSampling.insert(map<string, float>::value_type(actionTypeStr, GMM_SAMPLE_NUM/videoNum+1));
	}

	FileStorage fs, fslog;
	ActionAnalysis action;
	Mat des, tmpMat, GMMdes;
	
	// Current time tick
	double t1 = double( getTickCount() );	
	// 多线程读取每类动作的特征，并将其写入对应动作的xml临时文件
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

	// 防止以上for循环结束后程序马上结束的情况。因为信号量semaphore可能还不为0，
	// 此时有部分线程仍在工作未释放信号量，所以应该以信号量是否为0，判断主程序是否结束。
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
	cout << endl << "已将所有动作的部分特征全部写入xml文件......" << endl;	
	// 读取每类动作的特征xml临时文件，将GMM_SAMPLE_NUM个特征放到内存，训练GMM码本
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{	
		//if(*itype != "climb")
		//	continue;
		string file = resultPath + *itype + ".xml";
		fs.open(file, FileStorage::READ);
		fs["features"] >> tmpMat;
		//CV_Assert( !tmpMat.empty() );	

		RNG rng(getTickCount());	// 取当前系统时间作为随机数种子
		Mat randMat(tmpMat.rows, 1, CV_32S);
		for(int r=0; r<randMat.rows; r++)
			randMat.at<int>(r,0) = r;
		randShuffle(randMat, 1, &rng);

		// 每类动作，指定采样GMM_SAMPLE_NUM/CATEGORY_NUM+1个特征点（如果选太多特征训练码本，内存装不下会崩溃）
		int CATEGORY_NUM = actionType.size();
		int loop = GMM_SAMPLE_NUM/CATEGORY_NUM+1;
		cout << "loop:" << loop << ", tmpMat.rows:" << tmpMat.rows << endl;
		if( loop < tmpMat.rows ){	
			for(int r=0; r<loop; r++){	
				int randRow = randMat.at<int>(r,0);
				GMMdes.push_back(tmpMat.row(randRow));
			}
			cout << *itype << ", sampleNum(loop) < tmpMat.rows, 开始随机采样！" << endl;
		}
		else
			GMMdes.push_back(tmpMat);
		
		tmpMat.release();
		// delete txt for saving disk storage
		string cmdDel = "rm " + resultPath + *itype + ".xml";
		system(cmdDel.c_str());
	}

	cout << "已将所有动作的特征放入内存，准备训练gmm......"<< endl;
	cout << "GMMdes.rows:" << GMMdes.rows << ", GMMdes.cols:" << GMMdes.cols << endl;
//////////////////////////////////////////////////////////////////
	// 多通道分别训练码本	
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
	
	// 防止以上for循环结束后程序马上结束的情况。因为信号量semaphore可能还不为0，
	// 此时有部分线程仍在工作未释放信号量，所以应该以信号量是否为0，判断主程序是否结束。
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

	//// 先对样本进行预处理：PCA
	//pcaWhiten(GMMdes, combinePCAwhiten, resultPath, 200, "Combine", "CV_PCA_DATA_AS_ROW");
	////pcaWhiten(GMMdes.colRange(0,96), hogPCAwhiten, resultPath, 48, "HOG", "CV_PCA_DATA_AS_ROW");
	////pcaWhiten(GMMdes.colRange(96,204), hofPCAwhiten, resultPath, 54, "HOF", "CV_PCA_DATA_AS_ROW");
	////pcaWhiten(GMMdes.colRange(204,396), mbhPCAwhiten, resultPath, 96, "MBH", "CV_PCA_DATA_AS_ROW");
	////pcaWhiten(GMMdes.colRange(204,300), mbhxPCAwhiten, resultPath, 48, "MBHx", "CV_PCA_DATA_AS_ROW");
	////pcaWhiten(GMMdes.colRange(300,396), mbhyPCAwhiten, resultPath, 48, "MBHy", "CV_PCA_DATA_AS_ROW");

	//// 再训练码本：GMM聚类
	//gmmCluster(combinePCAwhiten, resultPath, gmmNum, "Combine", manifoldType);
	////gmmCluster(hogPCAwhiten, resultPath, gmmNum, "HOG", manifoldType);
	////gmmCluster(hofPCAwhiten, resultPath, gmmNum, "HOF", manifoldType);
	////gmmCluster(mbhPCAwhiten, resultPath, gmmNum, "MBH", manifoldType);
	////gmmCluster(mbhxPCAwhiten, resultPath, gmmNum, "MBHx", manifoldType);
	////gmmCluster(mbhyPCAwhiten, resultPath, gmmNum, "MBHy", manifoldType);

	//// 或训练码本：GMM聚类（未PCA白化）
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

	// 记录编码本的时间开销
	t2 = ( (double)getTickCount() - t2 ) / getTickFrequency();
	stringstream strGmmNum;
	strGmmNum << gmmNum; 
	string file_loger= resultPath + "log_GMM" + strGmmNum.str() + ".xml";
	fslog.open(file_loger, FileStorage::WRITE);
	fslog << "FulliScale_GMM" << "{" ;
	fslog << "read_time_hours" << t1/3600;
	fslog << "gmm_time_hours" << t2/3600;
	fslog << "}";

	// 释放资源
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
				// 统计已处理的帧数和视频数
				//totalframes += frame_counter[idx];
				video_num++;	
			}// if iter->first
		}//	for iter		
	}// for itype

	cout << processType << "Set end of getFisherVector...." << endl;

	// 防止以上for循环结束后程序马上结束的情况。因为信号量semaphore可能还不为0，
	// 此时有部分线程仍在工作未释放信号量，所以应该以信号量是否为0，判断主程序是否结束。
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
	// 记录训练集量化的时间开销
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

	//编码用到3个参数：means, covariances, priors
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
	//编码用到3个参数：means, covariances, priors
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

	//将训练集转存到连续空间
	float *data = (float*)malloc(src.cols*sizeof(float)*src.rows);
	fisher.bulidGMM_Data(src, data);
	//特征描述符维数
	int dimension = src.cols;
	//混合高斯的维度K，即gmm聚类中心的个数
	int numClusters = gmmNum;//64;
	VlGMM *gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, numClusters);

	fisher.getGMM(resultPath, gmm, descriptorType, manifoldType);

	//fisher 编码后的向量
	float* enc;
	//待编码的描述符数量
	int numDataToEncode = src.rows;
	// 分配空间，用于存储一段视频对应的fisher编码向量
	enc = (float*) vl_malloc(sizeof(float) * 2 * dimension * numClusters);

	cout << "111111111..." << endl;

	Mat code(1, 2*dimension*numClusters, CV_32FC1, Scalar(0));	//针对FisherVector
	
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
	
	// 释放资源.....
	vl_free(enc);
	free(data);		
	free(gmm);
	code.release();

	return 0;
}

int FisherVector::gmmCluster(Mat src, string resultPath, int gmmNum, string descriptorType, string manifoldType)
{
	FisherVector fisher;

	//将训练集转存到连续空间
	float *data = (float*)malloc(sizeof(float)*src.rows*src.cols);
	fisher.bulidGMM_Data(src, data);

	//特征描述符维数，从TXT特征文件中读取。
	int dimension = src.cols;
	//混合高斯的维度
	int numClusters = gmmNum;//64;
	//用于训练的GMM
	VlGMM *gmm= vl_gmm_new(VL_TYPE_FLOAT, dimension, numClusters);

	//对描述符进行训练，得到混合高斯分布
	vl_size numData = src.rows;
	src.release();

	cout << descriptorType << "开始gmm训练......";
	vl_gmm_cluster (gmm, data, numData);
	cout << descriptorType << "gmm训练完毕!" << endl;
	free(data);

	//由混合高斯模型计算时间较长，其结果最好序列化到文件中，以便后期用到时恢复
	fisher.setGMM(resultPath, gmm, descriptorType, manifoldType);

	return 0;
}

// PCA白化（新样本转换到PCA白化的特征空间）：单个视频在编码量化过程中，根据对应描述符descriptorType的PCA白化投影矩阵进行特征空间转换
// 注意：参数PCA_DATA_TYPE只是提醒该函数只处理行向量，函数中未用到该参数
int FisherVector::newSamplePCAWhiten(Mat src, Mat& dst, string resultPath, string descriptorType, string PCA_DATA_TYPE)
{
	//PCA白化用到3个参数：means均值, diag对角矩阵, proj投影矩阵
	Mat meanMat, diagMat, U;
	FileStorage fspca;
	string pca_file = resultPath + "pca" + descriptorType + ".xml";
	fspca.open(pca_file, FileStorage::READ);
	fspca["means"] >> meanMat;
	fspca["diag"] >> diagMat;
	fspca["proj"] >> U;

    // 样本去均值，形成样本零均值矩阵f
	for(int i=0; i<src.rows; i++)
	for(int j=0; j<src.cols; j++)
		src.at<float>(i,j) -= meanMat.at<float>(0,j);

	// PCA白化。如果 f 是以行向量存储特征的零均值样本矩阵，则白化公式为：  X = U'*f*△    （2）
	// U 为投影转换矩阵，f为样本矩阵src中每个元素减去每列的均值得到的新矩阵.
    dst = src * U * diagMat;
	//dst = src * U; // 不白化

	fspca.release();
	meanMat.release();
	diagMat.release();
	src.release();
	U.release();
	return 0;
}

// PCA白化：指定降维到pcaNum的维数，将src转换到PCA白化的特征空间形成dst（该函数目前只支持行向量的样本PCA，列向量的样本做PCA请参考pcaWhitenOld函数）
int FisherVector::pcaWhiten(Mat src, Mat& dst, string resultPath, int pcaNum, string descriptorType, string PCA_DATA_TYPE)
{
	Mat projMat, diagMat;
	PCA pca(src, noArray(), CV_PCA_DATA_AS_ROW, pcaNum);
    pca.project(src, projMat);

    Mat sqrtEigMat(pca.eigenvalues.rows, pca.eigenvalues.cols, CV_32F);
    float eplison = 0.00001;    //eplison为防止分母为0

    // 形成最大特征值的对角线矩阵△，diagMat为对角矩阵△
	// 这里所说的特征值（即pca.eigenvalues）是指src去均值后得到的矩阵f 的协方差矩阵的特征值！
	// 投影矩阵U（即特征向量pca.eigenvectors.t()）是指指src去均值后得到的矩阵f 的协方差矩阵的特征向量！
    for(int i=0; i<pca.eigenvalues.rows; i++)
        for(int j=0; j<pca.eigenvalues.cols; j++)
            sqrtEigMat.at<float>(i,j) = 1.0 / sqrt(pca.eigenvalues.at<float>(i,j) + eplison);
    diagMat = Mat::diag(sqrtEigMat);

	// 白化。如果样本矩阵f 是以行向量存储特征样本，则白化公式为：  X = U'*f*△    （2）
	dst = projMat * diagMat;
	//dst = projMat; // 不白化

	FileStorage fspca;
	string pca_file = resultPath + "pca" + descriptorType + ".xml";
	//编码用到3个参数：means, 对角矩阵diag, 投影矩阵U
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