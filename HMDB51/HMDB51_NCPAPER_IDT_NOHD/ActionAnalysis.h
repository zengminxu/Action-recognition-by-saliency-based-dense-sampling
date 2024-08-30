#ifndef _ACTIONANALYSIS_H_
#define _ACTIONANALYSIS_H_

#define SAMPLE_NUM 100000		// BOW 随机取样100000个特征点
#define GMM_SAMPLE_NUM 256000	// FV  随机取样256000个特征点
#define DICTIONARY_SIZE 4000	// KMEANS  字典4000个聚类中心
#define DIMENSION 438			// IDT 特征维度438(含32维轨迹位移点坐标)
#define SAMPLE_VIDEO 3000		// TRAIN 训练码本不超过3000段视频

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <fstream>
#include <time.h>
#ifdef _WIN32 // Windows version
	#include <windows.h>
#else // Linux version
	#include <pthread.h>
	#include <semaphore.h>
	#include <sys/sysinfo.h>
#endif

using namespace cv;
using namespace std;

typedef struct
{
	unsigned int sampleNum;
	string datasetPath;
	string featurePath;
	string vidcodePath;
	string resultPath;
	string actionTypeStr;
	multimap<string, string> actionSet;
	string filename;
	string extname;	// video format
	string bb_file_thrd;
	string descriptorType;
	string manifoldType;
	int gmmNum;
}ThreadParam;

typedef struct
{
	string cbname;		// codebook name
	string resultPath;
	string manifoldType;
	Mat featuresMat; 
	int gmmNum;
}CodebookParam;

typedef struct
{
	string resultPath;
	string descriptorType;
	vector<string> actionType; 
	multimap<string, string> trainSet;
	multimap<string, string> testSet;
	int gmmNum;
	int cost;
	int splitIdx;
	int costIdx;
	float (*accuracy)[50];
}SVMThreadParam;

class ActionAnalysis
{
public:
	ActionAnalysis(){};
	~ActionAnalysis(){};
		
	// 根据视频的特征向量，得到码本
	int getBOFcodebookHY2(vector<string> actionType, multimap<string, string> actionSet, string resultPath);
	int getBOFcodebookHMDB(vector<string> actionType, multimap<string, string> actionSet, string resultPath);
	int getMultiCodebook(vector<string> actionType, multimap<string, string> actionSet, string resultPath);	

	// 根据视频特征描述符，由码本得到对应的码字
	int getVideoCode(vector<string> actionType, multimap<string, string> actionSet,
					 string vidcodePath, string resultPath, string processType);

	// VectorQuantization 硬投票
	int hardVote(Mat tmp, Mat& code, Mat hogcodebook, Mat hofcodebook, Mat mbhcodebook);
	// SA-k knn近邻软投票 默认knn=5 
	int softAssignKnn(Mat tmp, Mat& code, int knn, Mat hogcodebook, Mat hofcodebook, Mat mbhcodebook);

	// 将特征文件、码本文件从bin转换为Mat格式
	int bin2Mat(string file, Mat& dstMat, string exType);
	// 将特征文件、码本文件从txt转换为Mat格式
	int txt2Mat(string file, Mat& dstMat, string exType);
	// 从split文件中读取训练集和测试集
	int readFileSets(string file, multimap<string, string> &dataSets, string processType);
	// 不重复特征点的随机采样，指定采样sampleNum个特征点
	int randSampling(Mat src, Mat& dst, int sampleNum, string sampleType);

	// 不考虑非线性核，仅用线性核且数据集多分组的libsvm训练和预测，其中splitNum为分组数
	int trainAndtest(vector<string> actionType, int splitNum, int gmmNum,
					 string datasetName, string resultPath, string descriptorType);
};

#endif