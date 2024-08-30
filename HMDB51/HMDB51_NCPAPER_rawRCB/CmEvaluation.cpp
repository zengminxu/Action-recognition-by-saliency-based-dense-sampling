#ifdef _WIN32 // Windows version
	#include "../CmInclude.h"
#else // Linux version
	#include "CmInclude.h"
#endif
#include "CmEvaluation.h"

int CmEvaluation::STEP = 1;
const int CmEvaluation::MI =  COLOR_NUM / STEP + 1;

void CmEvaluation::Evaluate(CStr gtW, CStr &salDir, CStr &resName, vecS &des)
{
	int TN = des.size(); // Type Number of different methods
	vector<vecD> precision(TN), recall(TN), tpr(TN), fpr(TN);
	static const int CN = 21; // Color Number 
	static const char* c[CN] = {"'k'", "'b'", "'g'", "'r'", "'c'", "'m'", "'y'",
		"':k'", "':b'", "':g'", "':r'", "':c'", "':m'", "':y'", 
		"'--k'", "'--b'", "'--g'", "'--r'", "'--c'", "'--m'", "'--y'"
	};
	FILE* f = fopen(_S(resName), "w");
	CV_Assert(f != NULL);
	fprintf(f, "clear;\nclose all;\nclc;\nsubplot('121');\nhold on;\n\n\n");
	vecD thr(MI);
	for (int i = 0; i < MI; i++)
		thr[i] = i * STEP;
	PrintVector(f, thr, "Threshold");
	fprintf(f, "\n");

	vecI changeT(TN, -1);
	for (int i = 0; i < TN; i++)
		changeT[i] = Evaluate_(gtW, salDir, des[i] + ".png", precision[i], recall[i], tpr[i], fpr[i]); //Evaluate(salDir + "*" + des[i] + ".png", gtW, val[i], recall[i], t);

	string leglendStr("legend(");
	vecS strPre(TN), strRecall(TN), strTpr(TN), strFpr(TN);
	for (int i = 0; i < TN; i++){
		strPre[i] = format("Precision%s", _S(des[i]));
		strRecall[i] = format("Recall%s", _S(des[i]));
		strTpr[i] = format("TPR%s", _S(des[i]));
		strFpr[i] = format("FPR%s", _S(des[i]));
		PrintVector(f, recall[i], strRecall[i]);
		PrintVector(f, precision[i], strPre[i]);
		PrintVector(f, tpr[i], strTpr[i]);
		PrintVector(f, fpr[i], strFpr[i]);
		fprintf(f, "plot(%s, %s, %s, 'linewidth', 2);\n", _S(strRecall[i]), _S(strPre[i]), c[i % CN]);
		leglendStr += format("'%s', ",  _S(des[i].substr(1)));
	}
	leglendStr.resize(leglendStr.size() - 2);
	leglendStr += ");";
	for (int i = 0; i < TN; i++)
		if (changeT[i] != -1)
			fprintf(f, "plot(%s(%d), %s(%d), 'r*', 'linewidth', 2);\n", _S(strRecall[i]), changeT[i], _S(strPre[i]), changeT[i]);
	string xLabel = "label('Recall');\n";
	string yLabel = "label('Precision')\n";
	fprintf(f, "hold off;\nx%sy%s\n%s\ngrid on;\naxis([0 1 0 1]);\n", _S(xLabel), _S(yLabel), _S(leglendStr));


	fprintf(f, "\n\nsubplot('122');\nhold on;\n");
	for (int i = 0; i < TN; i++)
		fprintf(f, "plot(%s, %s,  %s, 'linewidth', 2);\n", _S(strFpr[i]), _S(strTpr[i]), c[i % CN]);
	xLabel = "label('False positive rate');\n";
	yLabel = "label('True positive rate')\n";
	fprintf(f, "hold off;\nx%sy%s\n%s\ngrid on;\naxis([0 1 0 1]);\n\n\n", _S(xLabel), _S(yLabel), _S(leglendStr));


	for (int i = 0; i < TN; i++){
		double areaROC = 0;
		CV_Assert(fpr[i].size() == tpr[i].size());
		for (size_t t = 1; t < fpr[i].size(); t++)
			areaROC += (tpr[i][t] + tpr[i][t - 1]) * (fpr[i][t - 1] - fpr[i][t]) / 2.0;
		fprintf(f, "%%ROC%s: %g\n", _S(des[i]), areaROC);
	}


	fclose(f);
	printf("%-70s\r", "");
}

void CmEvaluation::PrintVector(FILE *f, const vecD &v, CStr &name)
{
	fprintf(f, "%s = [", name.c_str());
	for (size_t i = 0; i < v.size(); i++)
		fprintf(f, "%g ", v[i]);
	fprintf(f, "];\n");
}

// Return the threshold when significant amount of recall reach 0
int CmEvaluation::Evaluate_(CStr &gtImgW, CStr &inDir, CStr& resExt, vecD &precision, vecD &recall, vecD &tpr, vecD &fpr)
{
	vecS names;
	string truthDir, gtExt;
	int imgNum = CmFile::GetNamesNE(gtImgW, names, truthDir, gtExt); 
	precision.resize(MI, 0);
	recall.resize(MI, 0);
	tpr.resize(MI, 0);
	fpr.resize(MI, 0);
	vecD val2(MI, 0);
	if (imgNum == 0){
		printf("Can't load ground truth images %s\n", _S(gtImgW));
		return -1;
	}
	else
		printf("Evaluating %d saliency maps ... \n", imgNum);

	for (int i = 0; i < imgNum; i++){
		printf("Evaluating %03d/%d %-40s\r", i, imgNum, _S(names[i] + resExt));
		Mat resS = imread(inDir + names[i] + resExt, CV_LOAD_IMAGE_GRAYSCALE);
		//CV_Assert_(resS.data != NULL, ("Can't load saliency map: %s\n", _S(names[i]) + resExt));
		Mat gtFM = imread(truthDir + names[i] + gtExt, CV_LOAD_IMAGE_GRAYSCALE), gtBM;
		if (gtFM.data == NULL) 
			continue;
		CV_Assert_(resS.size() == gtFM.size(), ("Saliency map and ground truth image size mismatch: %s\n", _S(names[i])));
		compare(gtFM, 128, gtFM, CMP_GT);
		bitwise_not(gtFM, gtBM);
		double gtF = sum(gtFM).val[0];
		double gtB = resS.cols * resS.rows * 255 - gtF;

#pragma omp parallel for
		for (int thr = 0; thr < MI; thr++){
			Mat resM, tpM, fpM;
			compare(resS, thr * STEP, resM, CMP_GE);
			bitwise_and(resM, gtFM, tpM);
			bitwise_and(resM, gtBM, fpM);
			double tp = sum(tpM).val[0]; 
			double fp = sum(fpM).val[0];
			double fn = gtF - tp;
			double tn = gtB - fp;

			recall[thr] += tp/(gtF+EPS);
			double total = EPS + tp + fp;
			precision[thr] += (tp+EPS)/total;
			val2[thr] += tp/total;

			tpr[thr] += (tp + EPS) / (tp + fn + EPS);
			fpr[thr] += (fp + EPS) / (fp + tn + EPS);
		}
	}

	int thrS = 0, thrE = MI, thrD = 1;
	int res = -1;
	for (int thr = thrS; thr != thrE; thr += thrD){
		precision[thr] /= imgNum;
		recall[thr] /= imgNum;
		val2[thr] /= imgNum;
		tpr[thr] /= imgNum;
		fpr[thr] /= imgNum;
		if (precision[thr] - val2[thr] > 0.01 && res == -1)//Too many recall = 0 maps after this threshold
			res = thr;
	}
	return res;
}


void CmEvaluation::EvalueMask(CStr gtW, CStr &maskDir, CStr &gtExt, CStr &maskExt, bool back, bool alertNul)
{
	vecS names;
	string gtDir;
	int imgNum = CmFile::GetNames(gtW, names, gtDir);
	int count = 0, gtExtLen = (int)gtExt.size();
	double p = 0, r = 0, fn = 0;
	for (int i = 0; i < imgNum; i++){
		printf("Processing %-40s\r", _S(names[i]));
		Mat truM = imread(gtDir + names[i], CV_LOAD_IMAGE_GRAYSCALE), truInvM;
		names[i].resize(names[i].size() - gtExtLen);
		Mat res = imread(maskDir + names[i] + maskExt, CV_LOAD_IMAGE_GRAYSCALE);
		if (truM.data == NULL || res.data == NULL || truM.size != res.size){
			if (alertNul)
				printf("Truth(%d, %d), Res(%d, %d): %s\n", truM.cols, truM.rows, res.cols, res.rows, _S(names[i] + maskExt));
			continue;
		}
		compare(truM, 128, truM, back ? CMP_LE : CMP_GE);
		compare(res, 128, res, back ? CMP_LE : CMP_GE);
		Mat commMat;
		bitwise_and(truM, res, commMat);
		double commV = sum(commMat).val[0];
		p += commV/(sum(res).val[0] + EPS);
		r += commV/(sum(truM).val[0] + EPS);

		bitwise_not(truM, truInvM);
		bitwise_and(truInvM, res, commMat);
		commV = sum(commMat).val[0];
		fn += commV / (sum(truInvM).val[0] + EPS);
		count++;
	}	
	p /= count, r /= count, fn /= count;
	double f = 1.3 * p * r / (0.3 * p + r);
	printf("%s: precision = %.3g, recall = %.3g, FMeasure = %.3g, InvFalseNegative = %.3g\n", _S(maskExt), p, r, f, fn);
}

void CmEvaluation::MeanAbsoluteError(CStr &inDir, CStr &salDir, vecS &des)
{
	vecS namesNE;
	int imgNum = CmFile::GetNamesNE(inDir + "*.jpg", namesNE);
	vecD costs(des.size());
	for (int i = 0; i < imgNum; i++){
		//Mat gt = imread(inDir + namesNE[i] + ".png", CV_LOAD_IMAGE_GRAYSCALE);
		Mat gt = imread(inDir + namesNE[i] + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		//printf("%s.jpg ", _S(namesNE[i]));
		gt.convertTo(gt, CV_32F, 1.0/255);
		for (size_t j = 0; j < des.size(); j++){
			Mat res = imread(salDir + namesNE[i] + des[j] + ".png", CV_LOAD_IMAGE_GRAYSCALE);
			CV_Assert_(res.data != NULL, ("Can't load file %s\n", _S(namesNE[i] + des[j] + ".png")));
			if (res.size != gt.size){
				printf("size don't match %s\n", _S(namesNE[i] + des[j] + ".png"));
				resize(res, res, gt.size());
				imwrite(string("C:/WkDir/Saliency/DataFT/Out/") + namesNE[i] + des[j] + ".png", res);
			}
			res.convertTo(res, CV_32F, 1.0/255);
			cv::absdiff(gt, res, res);
			costs[j] += sum(res).val[0] / (gt.rows * gt.cols);
		}
	}

	for (size_t j = 0; j < des.size(); j++)
		printf("%s:%g	", _S(des[j]), costs[j]/imgNum);
	printf("\n");
}
