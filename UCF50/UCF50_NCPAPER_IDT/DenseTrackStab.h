#ifndef _DENSETRACKSTAB_H_
#define _DENSETRACKSTAB_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <ctype.h>
#ifdef _WIN32 // Windows version
	#include <windows.h>
	#include "getopt.h"
#else // Linux version
	#include <unistd.h>
#endif
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

// xzm
int idt_num = 0;			// -p 指定初始的IDT采样数
int mvs_min_distance = 16;	// CAMHID论文中的N
int mvf_length = 11;		// CAMHID论文中的k+1
const float min_sample_ratio = 0.5; // 采样点＜0.2*IDT数时，自动补全IDT
const float min_warp_flow = 0.0001;  // 0.01 for wofRCB drawImage, 0.0001 for wofRCB recognition

// pxj
char* outfile;		// -o [output filename], The output file for Improved Dense trajectory (default: None)\n");
int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

int start_frame = 0;
int end_frame = INT_MAX;
int scale_num = 8;
#ifdef _WIN32 // Windows version
const float scale_stride = sqrt(double(2));
#else // Linux version
const float scale_stride = sqrt(2);
#endif
char* bb_file = NULL;

// parameters for descriptors
int patch_size = 32;
int nxy_cell = 2;
int nt_cell = 3;
float epsilon = 0.05;
const float min_flow = 0.4;

// parameters for tracking
double quality = 0.001;
int min_distance = 5;
int init_gap = 1;
int track_length = 15;

// parameters for rejecting trajectory
#ifdef _WIN32 // Windows version
const float min_var = sqrt(double(3));
#else // Linux version
const float min_var = sqrt(3);
#endif
const float max_var = 50;
const float max_dis = 20;

typedef struct {
	int x;       // top left corner
	int y;
	int width;
	int height;
}RectInfo;

typedef struct {
    int width;   // resolution of the video
    int height;
    int length;  // number of frames
}SeqInfo;

typedef struct {
    int length;  // length of the trajectory
    int gap;     // initialization gap for feature re-sampling 
}TrackInfo;

// xzm
typedef struct {
    int length;  // length of the CAMHID's MVF
    int gap;     // initialization gap for feature re-sampling 
}MVFInfo;

typedef struct {
    int nBins;   // number of bins for vector quantization
    bool isHof; 
    int nxCells; // number of cells in x direction
    int nyCells; 
    int ntCells;
    int dim;     // dimension of the descriptor
    int height;  // size of the block for computing the descriptor
    int width;
}DescInfo; 

// integral histogram for the descriptors
typedef struct {
    int height;
    int width;
    int nBins;
    float* desc;
}DescMat;

// xzm
class MVF
{
public:
    std::vector< vector<Point2f> > mvf;
    int index;

    MVF(const vector<Point2f>& mvs_, const MVFInfo& mvfInfo) : mvf(mvfInfo.length){
        index = 0;
		mvf[index] = mvs_;	// xzm加了这句话会对对每个iMVF轨迹图的第0张MVF图赋值2次！先在mvf_tracks.push_back(MVF(mvs, mvfInfo));赋值一次
							// 然后在iMVF->addMVFs(prev_mvf);处，又对对每个iMVF轨迹图的第0张MVF图进行赋值。去掉不会影响效果。
    }

    void addMVFs(const vector<Point2f>& mvs_){
		mvf[index] = mvs_;	
		index++;
    }
};

class Track
{
public:
    std::vector<Point2f> point;
    std::vector<Point2f> disp;
    std::vector<float> hog;
    std::vector<float> hof;
    std::vector<float> mbhX;
    std::vector<float> mbhY;
    int index;

    Track(const Point2f& point_, const TrackInfo& trackInfo, const DescInfo& hogInfo,
          const DescInfo& hofInfo, const DescInfo& mbhInfo)
        : point(trackInfo.length+1), disp(trackInfo.length), hog(hogInfo.dim*trackInfo.length),
          hof(hofInfo.dim*trackInfo.length), mbhX(mbhInfo.dim*trackInfo.length), mbhY(mbhInfo.dim*trackInfo.length)
    {
        index = 0;
        point[0] = point_;
    }

    void addPoint(const Point2f& point_)
    {
        index++;
        point[index] = point_;
    }
};

class BoundBox
{
public:
	Point2f TopLeft;
	Point2f BottomRight;
	float confidence;

	BoundBox(float a1, float a2, float a3, float a4, float a5)
	{
		TopLeft.x = a1;
		TopLeft.y = a2;
		BottomRight.x = a3;
		BottomRight.y = a4;
		confidence = a5;
	}
};

class Frame
{
public:
	int frameID;
	std::vector<BoundBox> BBs;
	
	Frame(const int& frame_)
	{
		frameID = frame_;
		BBs.clear();
	}
};

#endif /*DENSETRACKSTAB_H_*/
