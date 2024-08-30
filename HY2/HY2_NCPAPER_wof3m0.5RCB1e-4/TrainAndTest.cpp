#define mySign(x) (x>=0?1:-1)

#include "ActionAnalysis.h"

using namespace cv;
using namespace std;

extern string spritLabel;

int constructSVMSamples(vector<string> actionType, multimap<string, string> actionSet, 
						Mat& actionTypeMat, Mat& actionSample, string resultPath);
int normSVMSamples(Mat trainTypeMat, Mat testTypeMat, Mat trainSample, Mat testSample, 
				   int gmmNum, string nameTrain, string nameTest, string descriptorType);

#ifdef _WIN32 // windows version
	volatile LONG semTrainAndTest = 0;
	DWORD WINAPI trainAndtestByMultiThread( LPVOID lpParameter )
#else // linux version
	// Data to protect with the interlocked functions
	sem_t semTrainAndTest;
	static void *trainAndtestByMultiThread(void *lpParameter)
#endif
{
	SVMThreadParam *lp = (SVMThreadParam *)lpParameter;

	string resultPath = lp->resultPath;
	string descriptorType = lp->descriptorType;
	vector<string> actionType = lp->actionType;
	multimap<string, string> trainSet = lp->trainSet;
	multimap<string, string> testSet = lp->testSet;
	int gmmNum = lp->gmmNum;
	int splitIdx = lp->splitIdx;
	int cost = lp->cost;
	int costIdx = lp->costIdx;
	float (*accuracy)[50] = lp->accuracy;
	stringstream strSplitIdx, strCost, strCostIdx;
	strSplitIdx << splitIdx; 
	strCost << cost; 
	strCostIdx << costIdx;

	// Construct libSVM samples for Train
	Mat trainSample, testSample;
	Mat trainTypeMat, testTypeMat;
	if( 0 != constructSVMSamples(actionType, trainSet, trainTypeMat, trainSample, resultPath) )
		return 0;
	if( 0 != constructSVMSamples(actionType, testSet,  testTypeMat,  testSample,  resultPath) )
		return 0;

	// xzm
	string nameTrain = resultPath + "svm" + spritLabel + "distanceForTrain_Split" + strSplitIdx.str() + "_Cost" + strCostIdx.str() + ".txt";
	string nameTest  = resultPath + "svm" + spritLabel + "distanceForTest_Split" + strSplitIdx.str() + "_Cost" + strCostIdx.str() + ".txt";
	
	// 将多通道的码字串联成一个长向量，按SVM指定格式将数据写入TXT文件，然后用线性SVM快速求解分类超平面(线性核u'*v)
	if( 0 != normSVMSamples(trainTypeMat, testTypeMat, trainSample, testSample, gmmNum, nameTrain, nameTest, descriptorType) )
		return 0;	

	string tPara1 = "svm-train -s 0 -t 0 -c ";
	string tPara2 = strCost.str();
	string tPara3 = " " + resultPath + "svm" + spritLabel + "distanceForTrain_Split" + strSplitIdx.str() + "_Cost" + strCostIdx.str() + ".txt ";
	string tPara4 = resultPath + "svm" + spritLabel + "trainedModel_Split" + strSplitIdx.str() + "_Cost" + strCostIdx.str() + ".txt > ";
	string tPara5 = resultPath + "svm" + spritLabel + "tmpTrainOutput_Split" + strSplitIdx.str() + "_Cost" + strCostIdx.str() + ".txt";
	string cmdTrain = tPara1 + tPara2 + tPara3 + tPara4 + tPara5;

	string pPara1 = "svm-predict ";
	string pPara2 = resultPath + "svm" + spritLabel + "distanceForTest_Split" + strSplitIdx.str() + "_Cost" + strCostIdx.str() + ".txt ";
	string pPara3 = resultPath + "svm" + spritLabel + "trainedModel_Split" + strSplitIdx.str() + "_Cost" + strCostIdx.str() + ".txt ";
	string pPara4 = resultPath + "svm" + spritLabel + "testResult_Split" + strSplitIdx.str() + "_Cost" + strCostIdx.str() + ".txt > ";
	string pPara5 = resultPath + "svm" + spritLabel + "tmpTestOutput_Split" + strSplitIdx.str() + "_Cost" + strCostIdx.str() + ".txt";
	string cmdTest = pPara1 + pPara2 + pPara3 + pPara4 + pPara5;

	system(cmdTrain.c_str());
	system(cmdTest.c_str());

	fstream fpTestResult;
	fpTestResult.open(pPara5.c_str(), ios::in);
	if(!fpTestResult){
		cout << "Error: Cannot open text file '" << pPara5 << "'." << endl;
		delete lp;
	#ifdef _WIN32 // Windows version
		// 结束提取特征，信号量-1
		_InterlockedDecrement( &semTrainAndTest );
	#else // Linux version
		sem_wait(&semTrainAndTest);
		pthread_detach(pthread_self());
	#endif
		return 0;
	}

	string r1, r2, r3, r4;
	fpTestResult >> r1 >> r2 >> r3 >> r4;
	fpTestResult.close();
//	cout << "r1:" << r1 << ", r2:" << r2 << ", r3:" << r3 << ", r4:" << r4 << endl;
	r3.erase(r3.length()-1, 1);
	accuracy[splitIdx][costIdx] = atof(r3.c_str());

	// delete txt for saving disk storage
	string cmdDel = "rm " + resultPath + "svm" + spritLabel + "*_Split" + strSplitIdx.str() + "_Cost" + strCostIdx.str() + ".txt";
	system(cmdDel.c_str());

	delete lp;
#ifdef _WIN32 // Windows version
	// 结束提取特征，信号量-1
	_InterlockedDecrement( &semTrainAndTest );
#else // Linux version
	sem_wait(&semTrainAndTest);
	pthread_detach(pthread_self());
#endif

	return 0;
}

// use specific kernel of SVM in OpenCV
int ActionAnalysis::trainAndtest(vector<string> actionType, int splitNum, int gmmNum,
								 string datasetName, string resultPath, string descriptorType)
{
	cout << "Train and Testing ..." << endl;
	cout << "Split group : " << splitNum << endl;
	double t = double(getTickCount());	// Current time tick

	float accuracy[50][50];
	float tmpAccuary = 0;
	ActionAnalysis action;
	multimap<string, string> trainSet, testSet;

	int startCost = 100;
	int costStep = 100;
	int endCost = 100;
	int valCost[50];
	int costTryTimes = (int)((endCost - startCost) / costStep + 1);
	cout << "Cost value from " << startCost << " to " << endCost << ", Step:" << costStep << endl;
	cout << "Cost Try times: " << costTryTimes << endl;
	int realCost = startCost;

	for(int i = 0; i < splitNum ; i++)		
	{
		realCost = startCost;
		for(int j = 0; j < costTryTimes; j++)
		{
			stringstream ss;	
			ss << i + 1; // i+1 是要对应上每种数据集的split分组文件名，分组序号从1开始例如*_splits_1.txt
			string splitFile = resultPath + datasetName + "_splits_" + ss.str() + ".txt";
			ss.str("");

			if ( 0 != action.readFileSets(splitFile, trainSet, "train") )
				return -1;
			if ( 0 != action.readFileSets(splitFile, testSet,  "test") )
				return -1;

			accuracy[i][j] = 0;
			valCost[j] = realCost;
			SVMThreadParam *temp = new SVMThreadParam();
			temp->resultPath = resultPath;
			temp->descriptorType = descriptorType;
			temp->actionType = actionType; 
			temp->gmmNum = gmmNum;
			temp->trainSet = trainSet;
			temp->testSet = testSet;
			temp->splitIdx = i;
			temp->cost = realCost;
			temp->costIdx = j;
			temp->accuracy = accuracy;

	#ifdef _WIN32 // Windows version
			SYSTEM_INFO theSystemInfo;
			::GetSystemInfo(&theSystemInfo);
			while( semTrainAndTest >= theSystemInfo.dwNumberOfProcessors)
				Sleep( 1000 );

			HANDLE hThread = CreateThread(NULL, 0, trainAndtestByMultiThread, temp, 0, NULL);
			if(hThread == NULL)	{
				cout << "Create Thread failed in featuresQuantizationProcess !" << endl;
				delete temp;
				return -1;
			}
			_InterlockedIncrement( &semTrainAndTest );
	#else // Linux version
			int NUM_PROCS = get_nprocs();
			int semaNum;
			sem_getvalue(&semTrainAndTest, &semaNum);
			while(semaNum >= NUM_PROCS){
				sleep( 1 );
				sem_getvalue(&semTrainAndTest, &semaNum);
			}

			pthread_t pthID;
			int ret = pthread_create(&pthID, NULL, trainAndtestByMultiThread, temp);
			if(ret){
				cout << "Error: Failed to create thread." << endl;
				return -1;
			}
			sem_post(&semTrainAndTest);	
	#endif
			realCost += costStep;
			trainSet.clear();
			testSet.clear();
		}// for costIdx
	}// for splitIdx
	
#ifdef _WIN32 // Windows version
	while( semTrainAndTest )
		Sleep( 1000 );
#else // Linux version
	int semaNum;
	sem_getvalue(&semTrainAndTest, &semaNum);
	while(semaNum){
		sleep(1);
		sem_getvalue(&semTrainAndTest, &semaNum);
	}
#endif

	float maxAccuracy = 0, sumAccuracy = 0, avgAccuracy = 0;
	float bestSplit = 0, tmpMaxAccuracy[50] = {0}; // 数组元素必须全部初始化为0
	int bestCost = 0;

	for(int i = 0; i < splitNum; i++)
	{
		cout << "splitIdx : " << i+1 << ", accuracy : ";
		for(int j = 0; j < costTryTimes; j++){
			// 输出每个分组每个cost的精度
			cout << accuracy[i][j] << "\t";
			if(maxAccuracy < accuracy[i][j]){
				maxAccuracy = accuracy[i][j];
				bestSplit = i;
				bestCost = valCost[j];
			}
			// 找出每个分组的最大精度
			if(tmpMaxAccuracy[i] < accuracy[i][j])
				tmpMaxAccuracy[i] = accuracy[i][j];
		}
		cout << endl;
		sumAccuracy += tmpMaxAccuracy[i];
	}
	// 求所有分组的平均精度
	avgAccuracy = sumAccuracy/splitNum;

	cout << endl << "Average accuracy is " << avgAccuracy << ", splitNum is " << splitNum << endl;
	cout << "Max accuracy is " << maxAccuracy << ", the best split is " << bestSplit << ", the best cost is " << bestCost << endl;

	t = ((double)getTickCount()-t) / getTickFrequency();
	cout<< "Done Train and Testing. Time-consuming: " << t/3600.f << " hours." <<endl;

	string cmdDel = "rm " + resultPath + datasetName + "_splits_*" + ".txt";
	system(cmdDel.c_str());

	return 0;
}

// Get Train samples
int constructSVMSamples(vector<string> actionType,	multimap<string, string> actionSet, 
					    Mat& actionTypeMat, Mat& actionSample, string resultPath){
	
	int actionid = 0;
	vector<string>::iterator itype;
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{
		actionid++;
		string actionTypeStr = *itype;
		multimap<string, string>::iterator iter;
		for(iter = actionSet.begin(); iter != actionSet.end(); iter++)
		{
			if( iter->first == actionTypeStr )
			{
/**				// 排除jhmdb库中8段提取不出IDT的视频
				if( iter->second == "Arrasando_no_Le_Parkour_jump_f_cm_np1_ri_bad_1" ||
					iter->second == "ArcheryFastShooting_shoot_bow_u_nm_np1_fr_med_17" ||
					iter->second == "LearnToShootFromTheMaster_catch_f_nm_np1_ba_med_1" ||
					iter->second == "LearnToShootFromTheMaster_shoot_ball_f_nm_np1_ba_med_4" ||
					iter->second == "Nike_Soccer_Commercial_-_Good_vs__Evil_kick_ball_f_cm_np1_fr_med_0" ||
					iter->second == "NoCountryForOldMen_pick_u_nm_np1_le_goo_0" ||
					iter->second == "prelinger_LetsPlay1949_sit_u_cm_np1_le_med_9" ||
					iter->second == "THE_PROTECTOR_run_f_cm_np1_ba_med_35"
					)
					continue;
**/		
				//// 排除hmdb51库中MBI提取不出IDT的视频
				//if( iter->second == "baseballpitchslowmotion_throw_f_nm_np1_fr_med_0" ||
				//	iter->second == "Hittingadouble(BXbaseball)_swing_baseball_f_nm_np1_ba_bad_2" ||
				//	iter->second == "likebeckam_run_f_nm_np1_fr_bad_12"                     
				//	)
				//	continue;
				//// 排除hmdb51库中RC-map提取不出IDT的视频
				//if( iter->second == "AmericanGangster_eat_u_cm_np1_fr_bad_62" ||
				//	iter->second == "Basketball_Dribbling_-_Basketball_Dribbling-_Finger_Pads_dribble_f_nm_np2_le_med_6" ||
				//	iter->second == "Fellowship_5_jump_f_nm_np1_ba_bad_12" ||
				//	iter->second == "Fellowship_5_shoot_bow_u_cm_np1_fr_med_13" ||
				//	iter->second == "Finding_Forrester_3_stand_u_cm_np1_fr_bad_10" ||
				//	iter->second == "IndianaJonesandTheTempleofDoom_stand_f_nm_np1_ri_med_3" || 
				//	iter->second == "KOREAN_TEA_CEREMONY_-_Simple_Demo_pour_u_nm_np1_fr_med_0" || 
				//	iter->second == "TrumanShow_run_f_nm_np1_le_med_11" ||  
				//	iter->second == "TrumanShow_turn_f_nm_np1_ri_med_10" 
				//	)
				//	continue;
				//// 排除hmdb51库中RCB-map提取不出IDT的视频
				//if( iter->second == "Fellowship_5_jump_f_nm_np1_ba_bad_12" ||
				//	iter->second == "IndianaJonesandTheTempleofDoom_stand_f_nm_np1_ri_med_3" ||
				//	iter->second == "TrumanShow_run_f_nm_np1_le_med_11"                     
				//	)
				//	continue;				

				Mat code;
				Mat temp(1, 1, CV_32SC1);
				FileStorage fs;
				string file = resultPath + "quantization" + spritLabel + iter->second + "_quantization.xml";
				fs.open(file, FileStorage::READ);
				if(!fs.isOpened()){
					cout << "Error: Could not open file '" << file << "'." << endl;
					return -1;		
				}

				FileNode fn = fs.root();			
				FileNodeIterator itfn = fn.begin();
				*itfn >> code;
				actionSample.push_back(code);

				//fstream fsProj;	
				//string fileRot = "D:\\code.txt"; 
				//fsProj.open(fileRot, ios::out);	
				//fsProj<<code;
				//fsProj.close();

				temp.at<int>(0,0) = actionid;
				actionTypeMat.push_back(temp);
				fs.release();
				temp.release();
				code.release();
			}// if iter
		}// for iter
	}// for it

	return 0;
}

int normSVMSamples(Mat trainTypeMat, Mat testTypeMat, Mat trainSample, Mat testSample, 
				   int gmmNum, string nameTrain, string nameTest, string descriptorType)
{
	fstream trainFile, testFile;
	trainFile.open(nameTrain.c_str(), ios::out);
	testFile.open( nameTest.c_str(),  ios::out);
	if(!trainFile || !testFile){
		cout << "ERROR: Cannot open svm file to write in normSVMSamples(). " << endl;
		return -1;
	}

	// Power 向量内部归一化
	for(int r = 0; r < trainSample.rows; r++)
	for(int c = 0; c < trainSample.cols; c++){
		float z = trainSample.at<float>(r,c);
		trainSample.at<float>(r,c) = mySign(z)*sqrt(abs(z));
	}
	for(int r = 0; r < testSample.rows; r++)
	for(int c = 0; c < testSample.cols; c++){
		float z = testSample.at<float>(r,c);
		testSample.at<float>(r,c) = mySign(z)*sqrt(abs(z));
	}

	// 考虑HOG+HOF+MBH整体组合的码字归一化，而对每段视频码字(即trainSample的一行)统一归一化
	if(descriptorType == "d-Fusion"){
		for(int j = 0; j < trainSample.rows; j++)
			normalize(trainSample.row(j), trainSample.row(j), 1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
		for(int j = 0; j < testSample.rows; j++)
			normalize(testSample.row(j),  testSample.row(j),  1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
	}

	// 考虑时间轴稠密采样：全局+局部时空卷整体组合的码字归一化，而对每段视频码字(即trainSample的一行)统一归一化
	if(descriptorType == "r2-Fusion"){
		int combine = 2*200*gmmNum, local = 2*200*gmmNum;
		for(int j = 0; j < trainSample.rows; j++){
			normalize(trainSample.colRange(0,combine).row(j), trainSample.colRange(0,combine).row(j), 1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(trainSample.colRange(combine, combine+local).row(j), trainSample.colRange(combine, combine+local).row(j), 1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
		}
		for(int j = 0; j < testSample.rows; j++){
			normalize(testSample.colRange(0,combine).row(j),  testSample.colRange(0,combine).row(j),  1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(testSample.colRange(combine, combine+local).row(j),  testSample.colRange(combine, combine+local).row(j),  1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
		}
	}

	// 分别考虑HOG、HOF、MBH的归一化，逐行每个样本对每种描述子单独进行归一化
	if(descriptorType == "r3-Fusion"){
		int hog = 2*48*gmmNum, hof = 2*54*gmmNum, mbh = 2*96*gmmNum;
		for(int j = 0; j < trainSample.rows; j++){
			normalize(trainSample.colRange(0,hog).row(j), trainSample.colRange(0,hog).row(j), 1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(trainSample.colRange(hog, hog+hof).row(j), trainSample.colRange(hog, hog+hof).row(j), 1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(trainSample.colRange(hog+hof, hog+hof+mbh).row(j), trainSample.colRange(hog+hof, hog+hof+mbh).row(j), 1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
		}
		for(int j = 0; j < testSample.rows; j++){
			normalize(testSample.colRange(0,hog).row(j),  testSample.colRange(0,hog).row(j),  1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(testSample.colRange(hog, hog+hof).row(j),  testSample.colRange(hog, hog+hof).row(j),  1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(testSample.colRange(hog+hof, hog+hof+mbh).row(j),  testSample.colRange(hog+hof, hog+hof+mbh).row(j),  1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
		}
	}

	// 分别考虑HOG、HOF、MBHx、MBHy的归一化，逐行每个样本对每种描述子单独进行归一化
	if(descriptorType == "r4-Fusion"){	
		int hog = 2*48*gmmNum, hof = 2*54*gmmNum, mbhx = 2*48*gmmNum, mbhy = 2*48*gmmNum;
		for(int j = 0; j < trainSample.rows; j++){
			normalize(trainSample.colRange(0,hog).row(j), trainSample.colRange(0,hog).row(j), 1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(trainSample.colRange(hog, hog+hof).row(j), trainSample.colRange(hog, hog+hof).row(j), 1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(trainSample.colRange(hog+hof, hog+hof+mbhx).row(j), trainSample.colRange(hog+hof, hog+hof+mbhx).row(j), 1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(trainSample.colRange(hog+hof+mbhx, hog+hof+mbhx+mbhy).row(j), trainSample.colRange(hog+hof+mbhx, hog+hof+mbhx+mbhy).row(j), 1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
		}
		for(int j = 0; j < testSample.rows; j++){
			normalize(testSample.colRange(0,hog).row(j),  testSample.colRange(0,hog).row(j),  1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(testSample.colRange(hog, hog+hof).row(j),  testSample.colRange(hog, hog+hof).row(j),  1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(testSample.colRange(hog+hof, hog+hof+mbhx).row(j),  testSample.colRange(hog+hof, hog+hof+mbhx).row(j),  1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
			normalize(testSample.colRange(hog+hof+mbhx, hog+hof+mbhx+mbhy).row(j),  testSample.colRange(hog+hof+mbhx, hog+hof+mbhx+mbhy).row(j),  1.0, 0.0, NORM_L2);//NORM_L1);//NORM_MINMAX);
		}
	}

	// 将FisherVector 这一个通道的码字按SVM的格式写入文件
	for(int j = 0; j < trainTypeMat.rows; j++)
	{
		trainFile << trainTypeMat.at<int>(j, 0) << " ";
		for(int k = 0; k < trainSample.cols; k++)
			trainFile << k+1 << ":" << trainSample.at<float>(j, k) << " ";
		trainFile << "\n";
	}

	for(int j = 0; j < testTypeMat.rows; j++) 
	{
		testFile << testTypeMat.at<int>(j, 0) << " ";
		for(int k = 0; k < testSample.cols; k++)
			testFile << k+1 << ":" << testSample.at<float>(j, k) << " ";
		testFile << "\n";
	}

	trainFile.close();
	testFile.close();
	return 0;
}
