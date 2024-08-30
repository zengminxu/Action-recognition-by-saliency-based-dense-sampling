#ifndef _ACTIONANALYSIS_H_
#define _ACTIONANALYSIS_H_

#define SAMPLE_NUM 100000		// BOW ���ȡ��100000��������
#define GMM_SAMPLE_NUM 256000	// FV  ���ȡ��256000��������
#define DICTIONARY_SIZE 4000	// KMEANS  �ֵ�4000����������
#define DIMENSION 438			// IDT ����ά��438(��32ά�켣λ�Ƶ�����)
#define SAMPLE_VIDEO 3000		// TRAIN ѵ���뱾������3000����Ƶ

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
		
	// ������Ƶ�������������õ��뱾
	int getBOFcodebookHY2(vector<string> actionType, multimap<string, string> actionSet, string resultPath);
	int getBOFcodebookHMDB(vector<string> actionType, multimap<string, string> actionSet, string resultPath);
	int getMultiCodebook(vector<string> actionType, multimap<string, string> actionSet, string resultPath);	

	// ������Ƶ���������������뱾�õ���Ӧ������
	int getVideoCode(vector<string> actionType, multimap<string, string> actionSet,
					 string vidcodePath, string resultPath, string processType);

	// VectorQuantization ӲͶƱ
	int hardVote(Mat tmp, Mat& code, Mat hogcodebook, Mat hofcodebook, Mat mbhcodebook);
	// SA-k knn������ͶƱ Ĭ��knn=5 
	int softAssignKnn(Mat tmp, Mat& code, int knn, Mat hogcodebook, Mat hofcodebook, Mat mbhcodebook);

	// �������ļ����뱾�ļ���binת��ΪMat��ʽ
	int bin2Mat(string file, Mat& dstMat, string exType);
	// �������ļ����뱾�ļ���txtת��ΪMat��ʽ
	int txt2Mat(string file, Mat& dstMat, string exType);
	// ��split�ļ��ж�ȡѵ�����Ͳ��Լ�
	int readFileSets(string file, multimap<string, string> &dataSets, string processType);
	// ���ظ�����������������ָ������sampleNum��������
	int randSampling(Mat src, Mat& dst, int sampleNum, string sampleType);

	// �����Ƿ����Ժˣ��������Ժ������ݼ�������libsvmѵ����Ԥ�⣬����splitNumΪ������
	int trainAndtest(vector<string> actionType, int splitNum, int gmmNum,
					 string datasetName, string resultPath, string descriptorType);
};

#endif