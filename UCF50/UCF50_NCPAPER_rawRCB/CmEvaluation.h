#pragma once
#ifndef _WIN32
	#include <iostream>
	#include <stdlib.h>
	#include <sys/stat.h>
	#include <dirent.h>
#endif
/************************************************************************/
/* For educational and research use only; commercial use are forbidden.	*/
/* Download more source code from: http://mmcheng.net/					*/
/* If you use any part of the source code, please cite related papers:	*/
/* [1] SalientShape: Group Saliency in Image Collections. M.M. Cheng,	*/
/*	 N.J. Mitra, X. Huang, S.M. Hu. The Visual Computer, 2013.			*/
/* [2] Efficient Salient Region Detection with Soft Image Abstraction.	*/
/*	 M.M. Cheng, J. Warrell, W.Y. Lin, S. Zheng, V. Vineet, N. Crook.	*/
/*	 IEEE ICCV, 2013.													*/
/* [3] Salient Object Detection and Segmentation. M.M. Cheng, N.J.		*/
/*   Mitra, X. Huang, P.H.S. Torr, S.M. Hu. Submitted to IEEE TPAMI		*/
/*	 (TPAMI-2011-10-0753), 2011.										*/
/* [4] Global Contrast based Salient Region Detection, Cheng et. al.,	*/
/*	   CVPR 2011.														*/
/************************************************************************/

struct CmEvaluation
{
	// Save the precision recall curve, and ROC curve to a Matlab file: resName
	// Return area under ROC curve
	static void Evaluate(CStr gtW, CStr &salDir, CStr &resName, vecS &des); 
	static void Evaluate(CStr gtW, CStr &salDir, CStr &resName, CStr &des) {vecS descri(1); descri[0] = des; Evaluate(gtW, salDir, resName, descri);} 

	static void EvalueMask(CStr gtW, CStr &maskDir, CStr &gtExt, CStr &maskExt, bool back = false, bool alertNul = true);

	static void MeanAbsoluteError(CStr &inDir, CStr &salDir, vecS &des);

	static int STEP; // Evaluation threshold density
private:
	static const int COLOR_NUM = 255;  
	static const int MI;  // Number of difference threshold

	static void PrintVector(FILE *f, const vecD &v, CStr &name);

	static int Evaluate_(CStr &gtImgW, CStr &inDir, CStr& resExt, vecD &precision, vecD &recall, vecD &tpr, vecD &fpr);
};

