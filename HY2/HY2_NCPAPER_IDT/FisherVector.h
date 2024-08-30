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

	// �������õ�3������,means,covariances,priors����XML�ļ�file��
	void setGMM(string file, VlGMM* gmm, string descriptorType, string manifoldType);
	// ��XML�ļ�file�л�ȡ�����õ�3������,means,covariances,priors
	// ����&��ʾ�����������᷵�ص�gmm����ʵ����*Ҳ���Բ���&����gmm�Ĳ������᷵�ظò����ģ�
	void getGMM(string file, VlGMM* &gmm, string descriptorType, string manifoldType);	
	// ѵ��GMM�뱾��gmmNumΪ������������
	int trainGMM(vector<string> actionType, multimap<string, string> actionSet, 
				 int gmmNum, string resultPath, string descriptorType, string manifoldType);
	// ����GMM�뱾�������ɵ�GMMģ�ͣ���ֵ��Э�����Ȩ�أ�����gmm.xml
	int gmmCluster(Mat src, string resultPath, int gmmNum, string descriptorType, string manifoldType);	
	// ��GMM���л��ļ��лָ�GMM������GMM��ÿ����Ƶ����Ϊfisher vector
	int getFisherVector(vector<string> actionType, multimap<string, string> actionSet,
					    int gmmNum, string datasetPath, string vidcodePath, string resultPath, 
						string processType, string descriptorType, string manifoldType);
    // FisherVector����
	int fisherEncode(Mat src, Mat& dst, string resultPath, int gmmNum, string descriptorType, string manifoldType);
	// ��src�л�ȡ���ݸ�ֵ��������ַ�ռ��des
	void bulidGMM_Data(Mat src, float* data);

	// PCA�׻���ָ����ά��pcaNum��ά������srcת����PCA�׻��������ռ��γ�dst
	int pcaWhiten(Mat src, Mat& dst, string resultPath, int pcaNum, string descriptorType, string PCA_DATA_TYPE);
	// PCA�׻���������ת����PCA�׻��������ռ䣩��������Ƶ�ڱ������������У����ݶ�Ӧ������descriptorType��PCA�׻�ͶӰ������������ռ�ת��
	int newSamplePCAWhiten(Mat src, Mat& dst, string resultPath, string descriptorType, string PCA_DATA_TYPE);
};

#endif