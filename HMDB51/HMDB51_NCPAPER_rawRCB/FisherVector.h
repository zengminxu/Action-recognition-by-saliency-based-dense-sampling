#ifndef _FISHERVECTOR_H_
#define _FISHERVECTOR_H_

// for fisher
extern "C" {	
#include "fisher.h"
#include "gmm.h"
#include "mathop.h"
}

using namespace cv;
using namespace std;

class FisherVector
{
public:
	FisherVector(){};
	~FisherVector(){};

	// 将编码用到3个参数,means,covariances,priors存入XML文件file中
	void setGMM(string file, VlGMM* gmm, string descriptorType, string manifoldType);
	// 从XML文件file中获取编码用到3个参数,means,covariances,priors
	// 加了&表示函数操作将会返回到gmm（其实有了*也可以不用&，对gmm的操作都会返回该参数的）
	void getGMM(string file, VlGMM* &gmm, string descriptorType, string manifoldType);	
	// 训练GMM码本，gmmNum为聚类中心数量
	int trainGMM(vector<string> actionType, multimap<string, string> actionSet, 
				 int gmmNum, string resultPath, string descriptorType, string manifoldType);
	// 创建GMM码本，将生成的GMM模型（均值、协方差和权重）存入gmm.xml
	int gmmCluster(Mat src, string resultPath, int gmmNum, string descriptorType, string manifoldType);	
	// 从GMM序列化文件中恢复GMM，根据GMM对每段视频编码为fisher vector
	int getFisherVector(vector<string> actionType, multimap<string, string> actionSet,
					    int gmmNum, string datasetPath, string vidcodePath, string resultPath, 
						string processType, string descriptorType, string manifoldType);
    // FisherVector编码
	int fisherEncode(Mat src, Mat& dst, string resultPath, int gmmNum, string descriptorType, string manifoldType);
	// 从src中获取数据赋值给连续地址空间的des
	void bulidGMM_Data(Mat src, float* data);

	// PCA白化：指定降维到pcaNum的维数，将src转换到PCA白化的特征空间形成dst
	int pcaWhiten(Mat src, Mat& dst, string resultPath, int pcaNum, string descriptorType, string PCA_DATA_TYPE);
	// PCA白化（新样本转换到PCA白化的特征空间）：单个视频在编码量化过程中，根据对应描述符descriptorType的PCA白化投影矩阵进行特征空间转换
	int newSamplePCAWhiten(Mat src, Mat& dst, string resultPath, string descriptorType, string PCA_DATA_TYPE);
};

#endif