#ifndef _DESCRIPTORS_H_
#define _DESCRIPTORS_H_

#include "DenseTrackStab.h"
extern void xzmOptIDTmap(Mat flow, Mat& wofIDTmap);  // xzm 20160306
extern void xzmOptRCBmap(Mat flow, Mat salMask, Mat& wofRCBmap);  //使用DrawOpticalFlow.h的函数
extern void imshowMany(const string&, const vector<Mat>&, Mat&);

using namespace cv;
using namespace std;

// get the rectangle for computing the descriptor
void GetRect(const Point2f& point, RectInfo& rect, const int width, const int height, const DescInfo& descInfo)
{
	int x_min = descInfo.width/2;
	int y_min = descInfo.height/2;
	int x_max = width - descInfo.width;
	int y_max = height - descInfo.height;

	rect.x = std::min<int>(std::max<int>(cvRound(point.x) - x_min, 0), x_max);
	rect.y = std::min<int>(std::max<int>(cvRound(point.y) - y_min, 0), y_max);
	rect.width = descInfo.width;
	rect.height = descInfo.height;
}

// compute integral histograms for the whole image
void BuildDescMat(const Mat& xComp, const Mat& yComp, float* desc, const DescInfo& descInfo)
{
	float maxAngle = 360.f;
	int nDims = descInfo.nBins;
	// one more bin for hof
	int nBins = descInfo.isHof ? descInfo.nBins-1 : descInfo.nBins;
	const float angleBase = float(nBins)/maxAngle;

	int step = (xComp.cols+1)*nDims;
	int index = step + nDims;
	for(int i = 0; i < xComp.rows; i++, index += nDims) {
		const float* xc = xComp.ptr<float>(i);
		const float* yc = yComp.ptr<float>(i);

		// summarization of the current line
		std::vector<float> sum(nDims);
		for(int j = 0; j < xComp.cols; j++) {
			float x = xc[j];
			float y = yc[j];
			float mag0 = sqrt(x*x + y*y);
			float mag1;
			int bin0, bin1;

			// for the zero bin of hof
			if(descInfo.isHof && mag0 <= min_flow) {
				bin0 = nBins; // the zero bin is the last one
				mag0 = 1.0;
				bin1 = 0;
				mag1 = 0;
			}
			else {
				float angle = fastAtan2(y, x);
				if(angle >= maxAngle) angle -= maxAngle;

				// split the mag to two adjacent bins
				float fbin = angle * angleBase;
				bin0 = cvFloor(fbin);
				bin1 = (bin0+1)%nBins;

				mag1 = (fbin - bin0)*mag0;
				mag0 -= mag1;
			}

			sum[bin0] += mag0;
			sum[bin1] += mag1;

			for(int m = 0; m < nDims; m++, index++)
				desc[index] = desc[index-step] + sum[m];
		}
	}
}

// get a descriptor from the integral histogram
void GetDesc(const DescMat* descMat, RectInfo& rect, DescInfo descInfo, std::vector<float>& desc, const int index)
{
	int dim = descInfo.dim;
	int nBins = descInfo.nBins;
	int height = descMat->height;
	int width = descMat->width;

	int xStride = rect.width/descInfo.nxCells;
	int yStride = rect.height/descInfo.nyCells;
	int xStep = xStride*nBins;
	int yStep = yStride*width*nBins;

	// iterate over different cells
	int iDesc = 0;
	std::vector<float> vec(dim);
	for(int xPos = rect.x, x = 0; x < descInfo.nxCells; xPos += xStride, x++)
	for(int yPos = rect.y, y = 0; y < descInfo.nyCells; yPos += yStride, y++) {
		// get the positions in the integral histogram
		const float* top_left = descMat->desc + (yPos*width + xPos)*nBins;
		const float* top_right = top_left + xStep;
		const float* bottom_left = top_left + yStep;
		const float* bottom_right = bottom_left + xStep;

		for(int i = 0; i < nBins; i++) {
			float sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
			vec[iDesc++] = std::max<float>(sum, 0) + epsilon;
		}
	}

	float norm = 0;
	for(int i = 0; i < dim; i++)
		norm += vec[i];
	if(norm > 0) norm = 1./norm;

	int pos = index*dim;
	for(int i = 0; i < dim; i++)
		desc[pos++] = sqrt(vec[i]*norm);
}

// for HOG descriptor
void HogComp(const Mat& img, float* desc, DescInfo& descInfo)
{
	Mat imgX, imgY;
	Sobel(img, imgX, CV_32F, 1, 0, 1);
	Sobel(img, imgY, CV_32F, 0, 1, 1);
	BuildDescMat(imgX, imgY, desc, descInfo);
}

// for HOF descriptor
void HofComp(const Mat& flow, float* desc, DescInfo& descInfo)
{
	Mat flows[2];
	split(flow, flows);
	BuildDescMat(flows[0], flows[1], desc, descInfo);
}

// for MBI
void MbhComp(const Mat& flow, Mat& MBI, float* descX, float* descY, DescInfo& descInfo)
{
	Mat flows[2];
	split(flow, flows);

	Mat flowXdX, flowXdY, flowYdX, flowYdY;
	Sobel(flows[0], flowXdX, CV_32F, 1, 0, 1);
	Sobel(flows[0], flowXdY, CV_32F, 0, 1, 1);
	Sobel(flows[1], flowYdX, CV_32F, 1, 0, 1);
	Sobel(flows[1], flowYdY, CV_32F, 0, 1, 1);

	BuildDescMat(flowXdX, flowXdY, descX, descInfo);
	BuildDescMat(flowYdX, flowYdY, descY, descInfo);

	int w = flow.cols;
 	int h = flow.rows;
 	Mat temp32f(h, w, CV_32FC1);	
	if(MBI.empty())
		MBI.create(h, w, CV_8UC1);	

 	for(int i = 0; i < h; i++) {
		const float* xcomp = flowXdX.ptr<float>(i);
		const float* ycomp = flowXdY.ptr<float>(i);
 		const float* Yxcomp = flowYdX.ptr<float>(i);
 		const float* Yycomp = flowYdY.ptr<float>(i);
 		for(int j = 0; j < w; j++) {
 			float shiftX = xcomp[j];
 			float shiftY = ycomp[j];
 			float magnitude0 = sqrt(shiftX*shiftX+shiftY*shiftY);
 			shiftX = Yxcomp[j];  
			shiftY = Yycomp[j];
 			float magnitude1 = sqrt(shiftX*shiftX + shiftY*shiftY);
			// pxj 取光流场中x方向和y方向上MBH幅值较大的作为采样点的一个依据
 			temp32f.at<float>(i, j) = magnitude0 > magnitude1 ? magnitude0 : magnitude1;
 		}
 	} 

	normalize(temp32f, temp32f, 1, 0, NORM_MINMAX); 
	convertScaleAbs(temp32f, MBI, 255, 0);
	//if(!notshow)  cvSaveImage("mbhsample.bmp",_MBI);
	//cvThreshold(_MBI,_MBI,100,255,CV_THRESH_BINARY|CV_THRESH_OTSU);
	//if(!notshow)
	//{ 		
	//	//cvCanny(temp8u,temp8u,100,200,3);CV_THRESH_OTSU
	//	cvShowImage("temp",temp8u); cvWaitKey(10);		
	//}	

	flowXdX.release();
 	flowXdY.release();
 	flowYdX.release();
 	flowYdY.release();
	temp32f.release();
}

// for MBH descriptor
void MbhComp(const Mat& flow, float* descX, float* descY, DescInfo& descInfo)
{
	Mat flows[2];
	split(flow, flows);

	Mat flowXdX, flowXdY, flowYdX, flowYdY;
	Sobel(flows[0], flowXdX, CV_32F, 1, 0, 1);
	Sobel(flows[0], flowXdY, CV_32F, 0, 1, 1);
	Sobel(flows[1], flowYdX, CV_32F, 1, 0, 1);
	Sobel(flows[1], flowYdY, CV_32F, 0, 1, 1);

	BuildDescMat(flowXdX, flowXdY, descX, descInfo);
	BuildDescMat(flowYdX, flowYdY, descY, descInfo);
}

// check whether a trajectory is valid or not
bool IsValid(std::vector<Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length)
{
	int size = track.size();
	float norm = 1./size;
	for(int i = 0; i < size; i++) {
		mean_x += track[i].x;
		mean_y += track[i].y;
	}
	mean_x *= norm;
	mean_y *= norm;

	for(int i = 0; i < size; i++) {
		float temp_x = track[i].x - mean_x;
		float temp_y = track[i].y - mean_y;
		var_x += temp_x*temp_x;
		var_y += temp_y*temp_y;
	}
	var_x *= norm;
	var_y *= norm;
	var_x = sqrt(var_x);
	var_y = sqrt(var_y);

	// remove static trajectory
	if(var_x < min_var && var_y < min_var)
		return false;
	// remove random trajectory
	if( var_x > max_var || var_y > max_var )
		return false;

	float cur_max = 0;
	for(int i = 0; i < size-1; i++) {
		track[i] = track[i+1] - track[i];
		float temp = sqrt(track[i].x*track[i].x + track[i].y*track[i].y);

		length += temp;
		if(temp > cur_max)
			cur_max = temp;
	}

	if(cur_max > max_dis && cur_max > length*0.7)
		return false;

	track.pop_back();
	norm = 1./length;
	// normalize the trajectory
	for(int i = 0; i < size-1; i++)
		track[i] *= norm;

	return true;
}

bool IsCameraMotion(std::vector<Point2f>& disp)
{
	float disp_max = 0;
	float disp_sum = 0;
	for(int i = 0; i < disp.size(); ++i) {
		float x = disp[i].x;
		float y = disp[i].y;
		float temp = sqrt(x*x + y*y);

		disp_sum += temp;
		if(disp_max < temp)
			disp_max = temp;
	}

	if(disp_max <= 1)
		return false;

	float disp_norm = 1./disp_sum;
	for (int i = 0; i < disp.size(); ++i)
		disp[i] *= disp_norm;

	return true;
}

// detect new feature points in an image without overlapping to previous points
void DenseSample(const Mat& grey, std::vector<Point2f>& points, const double quality, const double min_distance)
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);

	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters(width*height);
	int x_max = min_distance*width;
	int y_max = min_distance*height;

	for(int i = 0; i < points.size(); i++) {
		Point2f point = points[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters[y*width+x]++;
	}
	
	points.clear();
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters[index] > 0)
			continue;

		int x = j*min_distance+offset;
		int y = i*min_distance+offset;
		// 如果角点坐标x,y的特征值大于T才能作为强角点进行采样
		if(eig.at<float>(y, x) > threshold)
			points.push_back(Point2f(float(x), float(y)));
	}
}

// 针对SCI第0帧的IDT-wofRCB采样
void wofDenseSample(const Mat& grey, std::vector<Point2f>& points_idt, std::vector<Point2f>& points_opt, 
					const double quality, const int min_distance, const Mat& RCBmap)
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);
	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters_idt(width*height), counters_opt(width*height);
	int x_max = min_distance*width;
	int y_max = min_distance*height;

	// 遍历视频帧中存在強角点的网格，将网格编号存入counters[width*height]
	for(int i = 0; i < points_idt.size(); i++) {
		Point2f point = points_idt[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_idt[y*width+x]++;
	}
	for(int i = 0; i < points_opt.size(); i++) {
		Point2f point = points_opt[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_opt[y*width+x]++;
	}

	// 清空上一次采样的特征点(第0帧不存在上一次采样的特征点，在这里只是与非0帧代码统一写法)
	points_idt.clear();
	points_opt.clear();
	
	// 遍历视频帧中未放入強角点的空网格，将新的強角点放入空网格counters[index]中
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters_idt[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			// 如果角点坐标x,y的特征值大于T才能作为强角点
			//if(eig.at<float>(y,x) > threshold)
			if(eig.at<float>(y,x) > threshold && RCBmap.at<uchar>(y,x)) // xzm 20160306
				points_idt.push_back(Point2f(float(x), float(y)));
		}
		if(counters_opt[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			// 如果角点坐标x,y的特征值大于T才能作为强角点，且该点坐标属于前景掩码才进行采样
			if(eig.at<float>(y,x) > threshold && RCBmap.at<uchar>(y,x)) // 由于第0帧没有光流所以采样RCB做掩码
				points_opt.push_back(Point2f(float(x), float(y)));
		}
	}
}

// 针对SCI所有非0帧的IDT-wofRCB采样
void wofDenseSample(const Mat& grey, const Mat& warp_flow, std::vector<Point2f>& points_idt, 
					std::vector<Point2f>& points_opt, const double quality, 
					const int min_distance, const Mat& RCBmap)
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);
	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters_idt(width*height), counters_opt(width*height);
	int x_max = min_distance*width;
	int y_max = min_distance*height;

	for(int i = 0; i < points_idt.size(); i++) {
		Point2f point = points_idt[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_idt[y*width+x]++;
	}
	for(int i = 0; i < points_opt.size(); i++) {
		Point2f point = points_opt[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_opt[y*width+x]++;
	}
	
	// 将光流场flow_warp_pyr_noHD[0]限定在显著图salMask范围内
	Mat wofRCBmap, wofIDTmap;	// 初始化：直接由xzmOptRCBmap()赋值为双通道掩码矩阵
	xzmOptIDTmap(warp_flow, wofIDTmap); // xzm 20160306 特供idt使用，限制补idt的特征点
	xzmOptRCBmap(warp_flow, RCBmap, wofRCBmap);

	// 清空上一次采样的特征点
	points_idt.clear();
	points_opt.clear();

	// 遍历视频帧中未放入強角点的空网格，将新的強角点放入空网格counters[index]中
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters_idt[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			//if(eig.at<float>(y,x) > threshold)
			// 专供idt使用，限制补idt的特征点：只取warp_flow＞最小光流幅值的点给idt用
			if(eig.at<float>(y,x) > threshold && (wofIDTmap.at<Vec2f>(y,x)[0] && wofIDTmap.at<Vec2f>(y,x)[1]))
				points_idt.push_back(Point2f(float(x), float(y)));
		}
		if(counters_opt[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && (wofRCBmap.at<Vec2f>(y,x)[0] && wofRCBmap.at<Vec2f>(y,x)[1]))
				points_opt.push_back(Point2f(float(x), float(y)));
		}
	}
}

void InitPry(const Mat& frame, std::vector<float>& scales, std::vector<Size>& sizes)
{
	int rows = frame.rows, cols = frame.cols;
	float min_size = std::min<int>(rows, cols);

	int nlayers = 0;
	while(min_size >= patch_size) {
		min_size /= scale_stride;
		nlayers++;
	}

	if(nlayers == 0) nlayers = 1; // at least 1 scale 

	scale_num = std::min<int>(scale_num, nlayers);

	scales.resize(scale_num);
	sizes.resize(scale_num);

	scales[0] = 1.;
	sizes[0] = Size(cols, rows);

	for(int i = 1; i < scale_num; i++) {
		scales[i] = scales[i-1] * scale_stride;
		sizes[i] = Size(cvRound(cols/scales[i]), cvRound(rows/scales[i]));
	}
}

void BuildPry(const std::vector<Size>& sizes, const int type, std::vector<Mat>& grey_pyr)
{
	int nlayers = sizes.size();
	grey_pyr.resize(nlayers);

	for(int i = 0; i < nlayers; i++)
		grey_pyr[i].create(sizes[i], type);
}

// xzm
void BuildPry(const std::vector<Size>& sizes, const int type, const std::vector<Mat>& grey_pyr, 
			  std::vector< vector<Point2f> >& mvs_pyr)
{
	int nlayers = sizes.size();
	mvs_pyr.resize(nlayers);

	for(int i = 0; i < nlayers; i++){
		int height = grey_pyr[i].rows/mvs_min_distance;
		int width = grey_pyr[i].cols/mvs_min_distance;
		mvs_pyr[i].resize(height*width);
	}	
}

void DrawTrack(const std::vector<Point2f>& point, const int index, const float scale, Mat& image)
{
	Point2f point0 = point[0];
	point0 *= scale;

	for (int j = 1; j <= index; j++) {
		Point2f point1 = point[j];
		point1 *= scale;

		line(image, point0, point1, Scalar(0,cvFloor(255.0*(j+1.0)/float(index+1.0)),0), 2, 8, 0);
		point0 = point1;
	}
	circle(image, point0, 1, Scalar(0,0,255), -1, 8, 0);
	//imshow( "DenseTrackStab", image);
	//waitKey(0);
}

// 画出前后两帧模板块发生了位移的新坐标（在ComputeMVs()函数中用到） 
void DrawTrack(Mat& img, std::vector<Point2f> mvs)	// xzm
{
	int width = img.cols/mvs_min_distance;
	int height = img.rows/mvs_min_distance;

	Point2f point0, point1;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++) {
		if( mvs[i*width+j].x != 0 || mvs[i*width+j].y != 0){
			point0.x = j*mvs_min_distance;				
			point0.y = i*mvs_min_distance;
			point1.x = point0.x + mvs_min_distance;		
			point1.y = point0.y + mvs_min_distance;
			rectangle(img, point0, point1, Scalar(0,0,255), 1, 8, 0);
		}
	}

	imshow( "DenseTrackStab", img);
	waitKey(0);
}

// 测试 MatchingMethod() 在邻域同色情况下匹配不正常的情况。按说同色邻域如天空时不会发生大的变化，但匹配结果却总会出现较大位移
// 测试foreman头盔位移变化（在ComputeMVs()函数中用到）
void DrawTrack(Mat& img, Point2f prev_point, Point2f point)	// xzm
{
	Point2f point0, point1;	

	prev_point.x = prev_point.x + mvs_min_distance/2;
	prev_point.y = prev_point.y + mvs_min_distance/2;
	point.x = point.x + mvs_min_distance/2;
	point.y = point.y + mvs_min_distance/2;	
	line(img, prev_point, point, Scalar(0,255,0), 2, 8, 0);
	circle(img, prev_point, 1, Scalar(0,0,255), -1, 8, 0);

	point0.x = point.x - mvs_min_distance/2;	
	point0.y = point.y - mvs_min_distance/2;	
	point1.x = point0.x + mvs_min_distance;
	point1.y = point0.y + mvs_min_distance;
	rectangle(img, point0, point1, Scalar(0,0,255), 1, 8, 0);

	imshow( "DenseTrackStab", img);
	waitKey(0);
}

// 画出CAMHID论文图3的模板块连续k帧的位移梯度方差（在ComputeVariance()后用到）
void DrawTrack(Mat& img, std::vector<Point2f> mvi, string type)	// xzm
{	
	if(type != "fig3")
		return;

	int width = img.cols/mvs_min_distance;
	int height = img.rows/mvs_min_distance;

	Point2f point0, point1;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++) {
		if( mvi[i*width+j].x != 0 || mvi[i*width+j].y != 0)
		{	
			point0.x = j*mvs_min_distance + mvs_min_distance/2;				
			point0.y = i*mvs_min_distance + mvs_min_distance/2;
			point1.x = point0.x + mvi[i*width+j].x;
			point1.y = point0.y + mvi[i*width+j].y;	
			
			Point2f tmp = mvi[i*width+j];
			float stax = point0.x - tmp.x/2.;
			float stay = point0.y - tmp.y/2.;
			float endx = point1.x - tmp.x/2.;
			float endy = point1.y - tmp.y/2.;
			float color = sqrt((mvi[i*width+j].x * mvi[i*width+j].x) + (mvi[i*width+j].y * mvi[i*width+j].y));
			// 画连续mvf_Length-1帧的位移方差
			line(img, Point(stax,stay), Point(endx,endy), Scalar(0,cvFloor(255.0*color),0), 2, 8, 0);
			// 画网格中心点
			circle(img, point0, 1, Scalar(0,0,255), -1, 8, 0);
		}// if mvi
	}// for i,j

	imshow( "DenseTrackStab", img);
	waitKey(10);
}

//void PrintDesc(std::vector<float>& desc, DescInfo& descInfo, TrackInfo& trackInfo)
void PrintDesc(std::vector<float>& desc, DescInfo& descInfo, TrackInfo& trackInfo, std::FILE* fx) // xzm
{
	int tStride = cvFloor(trackInfo.length/descInfo.ntCells);
	float norm = 1./float(tStride);
	int dim = descInfo.dim;
	int pos = 0;
	for(int i = 0; i < descInfo.ntCells; i++) {
		std::vector<float> vec(dim);
		for(int t = 0; t < tStride; t++)
			for(int j = 0; j < dim; j++)
				vec[j] += desc[pos++];
		for(int j = 0; j < dim; j++){
			float norm_vec = vec[j]*norm;			// average 
			fwrite(&norm_vec,sizeof(float),1,fx);	// xzm
			//printf("%.7f\t", vec[j]*norm);
		}
	}
}

// 计算CAMHID论文的公式(2)，注意：du,dv∈[0,9]，最后一个数没有存入前后两帧相减的新数据
void PrintDesc(std::vector< vector<Point2f> >& mvf, Mat img, int frame_num, std::fstream& fs) // xzm
{
	int width = img.cols/mvs_min_distance;
	int height = img.rows/mvs_min_distance;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++) { 
		fs << frame_num << "\t" << i << "\t" << j;
		for(int k = 1; k < mvf.size(); k++){			
			mvf[k-1][i*width+j].x = mvf[k][i*width+j].x - mvf[k-1][i*width+j].x;
			mvf[k-1][i*width+j].y = mvf[k][i*width+j].y - mvf[k-1][i*width+j].y;
			// output the value of du,dv
			fs << "\t" << mvf[k-1][i*width+j].x << "," << mvf[k-1][i*width+j].y; 
		}// for it
		fs << "\n";
	}// for i,j
				
	fs.close();
}

//void LoadBoundBox(char* file, std::vector<Frame>& bb_list) 
void LoadBoundBox(std::string file, std::vector<Frame>& bb_list)// xzm
{
	// load the bouding box file
    //std::ifstream bbFile(file);
	std::ifstream bbFile(file.c_str()); // xzm
    std::string line;

    while(std::getline(bbFile, line)) {
		 std::istringstream iss(line);

		int frameID;
		if (!(iss >> frameID))
			continue;

		Frame cur_frame(frameID);

		float temp;
		std::vector<float> a(0);
		while(iss >> temp)
			a.push_back(temp);

		int size = a.size();

		if(size % 5 != 0)
			fprintf(stderr, "Input bounding box format wrong!\n");

		for(int i = 0; i < size/5; i++)
			cur_frame.BBs.push_back(BoundBox(a[i*5], a[i*5+1], a[i*5+2], a[i*5+3], a[i*5+4]));

		bb_list.push_back(cur_frame);
    }
}

void InitMaskWithBox(Mat& mask, std::vector<BoundBox>& bbs)
{
	int width = mask.cols;
	int height = mask.rows;

	for(int i = 0; i < height; i++) {
		uchar* m = mask.ptr<uchar>(i);
		for(int j = 0; j < width; j++)
			m[j] = 1;
	}

	for(int k = 0; k < bbs.size(); k++) {	// bbs.size()表示当前帧有几个行人目标框，框内赋值为0
		BoundBox& bb = bbs[k];
		for(int i = cvCeil(bb.TopLeft.y); i <= cvFloor(bb.BottomRight.y); i++) {
			uchar* m = mask.ptr<uchar>(i);
			for(int j = cvCeil(bb.TopLeft.x); j <= cvFloor(bb.BottomRight.x); j++)
				m[j] = 0;
		}
	}
}

// 对程明明GC显著图进行掩码处理，用于获取RCBmap
void InitSalMask(Mat& mask)
{
	double smean = 0;	
	smean = mean(mask).val[0];	// 计算程明明GC显著图的灰度平均值
	// 当显著图全白为255(即均值为255)时，不做IDT-RCB，避免图像二值化后显著图掩码全黑为0
	if( smean != 255 ){
		// 将显著度掩码矩阵mask浮点型数据转换成8位无符号整型的灰度图
		mask.convertTo(mask, CV_8U);			// 将浮点型数据转换成标准的8位无符号整型
		//convertScaleAbs(mask, mask, 1, 0);	// 使用线性变换浮点型数据转换成8位无符号整型
		// 用阈值操作得到二值化图像，此时第二个参数mask即为RCmap
		cv::threshold(mask, mask, smean, 255, THRESH_BINARY);	// 原始灰度平均值作为二值化阈值，显著图稳定性好
		//cv::threshold(mask, mask, 0, 255, THRESH_BINARY|THRESH_OTSU); // 二值化阈值由OTSU算法决定，稳定性稍差
		// elemKernel：定义的结构元素
		Mat elemKernel(5,5,CV_8U,Scalar(1));// MORPH_GRADIENT：对图像进行2次梯度运算，即膨胀减去腐蚀
		// 将第一个参数mask(即RCmap)，用形态学梯度变化得到第二个参数mask(即RCBmap)
		cv::morphologyEx(mask, mask, MORPH_GRADIENT, elemKernel, Point(-1,-1), 3);
	}

	//// 未转opencv默认格式的以上代码，前景为255，背景为0.
	//// 转成opencv默认0,1掩码格式：前景为0，背景为1.
	//int width = mask.cols;
	//int height = mask.rows;
	//for(int i = 0; i < height; i++) {
	//	uchar* m = mask.ptr<uchar>(i);
	//	for(int j = 0; j < width; j++)
	//		if(m[j] == 255)
	//			m[j] = 0;
	//		else
	//			m[j] = 1;
	//}	
}	

static void MyWarpPerspective(Mat& prev_src, Mat& src, Mat& dst, Mat& M0, int flags = INTER_LINEAR,
	            			 int borderType = BORDER_CONSTANT, const Scalar& borderValue = Scalar())
{
	int width = src.cols;
	int height = src.rows;
	dst.create( height, width, CV_8UC1 );

	Mat mask = Mat::zeros(height, width, CV_8UC1);
	const int margin = 5;

    const int BLOCK_SZ = 32;
    short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    double M[9];
    Mat matM(3, 3, CV_64F, M);
    M0.convertTo(matM, matM.type());
    if( !(flags & WARP_INVERSE_MAP) )
         invert(matM, matM);

    int x, y, x1, y1;

   int bh0 = std::min<int>(BLOCK_SZ/2, height);			// ??
    int bw0 = std::min<int>(BLOCK_SZ*BLOCK_SZ/bh0, width);	// bw0=64
    bh0 = std::min<int>(BLOCK_SZ*BLOCK_SZ/bw0, height);		// bh0=16

	for( y = 0; y < height; y += bh0 ) {
    for( x = 0; x < width; x += bw0 ) {
		int bw = std::min<int>( bw0, width - x);
        int bh = std::min<int>( bh0, height - y);

	    Mat _XY(bh, bw, CV_16SC2, XY);
		Mat matA;
        Mat dpart(dst, Rect(x, y, bw, bh));

		for( y1 = 0; y1 < bh; y1++ ) {

			short* xy = XY + y1*bw*2;
            double X0 = M[0]*x + M[1]*(y + y1) + M[2];
            double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
            double W0 = M[6]*x + M[7]*(y + y1) + M[8];
            short* alpha = A + y1*bw;

            for( x1 = 0; x1 < bw; x1++ ) {

                double W = W0 + M[6]*x1;
                W = W ? INTER_TAB_SIZE/W : 0;

				double fX = std::max<double>((double)INT_MIN, std::min<double>((double)INT_MAX, (X0 + M[0]*x1)*W));
                double fY = std::max<double>((double)INT_MIN, std::min<double>((double)INT_MAX, (Y0 + M[3]*x1)*W));

				double _X = fX/double(INTER_TAB_SIZE);
				double _Y = fY/double(INTER_TAB_SIZE);

				if( _X > margin && _X < width-1-margin && _Y > margin && _Y < height-1-margin )
					mask.at<uchar>(y+y1, x+x1) = 1;

                int X = saturate_cast<int>(fX);
                int Y = saturate_cast<int>(fY);

                xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
                xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
                alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (X & (INTER_TAB_SIZE-1)));
            }
        }

        Mat _matA(bh, bw, CV_16U, A);
        remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
    }
    }

	for( y = 0; y < height; y++ ) {
		const uchar* m = mask.ptr<uchar>(y);
		const uchar* s = prev_src.ptr<uchar>(y);
		uchar* d = dst.ptr<uchar>(y);
		for( x = 0; x < width; x++ ) {
			if(m[x] == 0)
				d[x] = s[x];
		}
	}
}

void ComputeMatch(const std::vector<KeyPoint>& prev_kpts, const std::vector<KeyPoint>& kpts,
				  const Mat& prev_desc, const Mat& desc, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts)
{
	prev_pts.clear();
	pts.clear();

	if(prev_kpts.size() == 0 || kpts.size() == 0)
		return;

	Mat mask = windowedMatchingMask(kpts, prev_kpts, 25, 25);

	BFMatcher desc_matcher(NORM_L2);
	std::vector<DMatch> matches;

	desc_matcher.match(desc, prev_desc, matches, mask);
	
	prev_pts.reserve(matches.size());
	pts.reserve(matches.size());

	for(size_t i = 0; i < matches.size(); i++) {
		const DMatch& dmatch = matches[i];
		// get the point pairs that are successfully matched
		prev_pts.push_back(prev_kpts[dmatch.trainIdx].pt);
		pts.push_back(kpts[dmatch.queryIdx].pt);
	}

	return;
}

void MergeMatch(const std::vector<Point2f>& prev_pts1, const std::vector<Point2f>& pts1,
				const std::vector<Point2f>& prev_pts2, const std::vector<Point2f>& pts2,
				std::vector<Point2f>& prev_pts_all, std::vector<Point2f>& pts_all)
{
	prev_pts_all.clear();
	prev_pts_all.reserve(prev_pts1.size() + prev_pts2.size());

	pts_all.clear();
	pts_all.reserve(pts1.size() + pts2.size());

	for(size_t i = 0; i < prev_pts1.size(); i++) {
		prev_pts_all.push_back(prev_pts1[i]);
		pts_all.push_back(pts1[i]);
	}

	for(size_t i = 0; i < prev_pts2.size(); i++) {
		prev_pts_all.push_back(prev_pts2[i]);
		pts_all.push_back(pts2[i]);	
	}

	return;
}

void MatchFromFlow(const Mat& prev_grey, const Mat& flow, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask)
{
	int width = prev_grey.cols;
	int height = prev_grey.rows;
	prev_pts.clear();
	pts.clear();

	const int MAX_COUNT = 1000;
	goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);
	
	if(prev_pts.size() == 0)
		return;

	for(int i = 0; i < prev_pts.size(); i++) {
		int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
		int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

		const float* f = flow.ptr<float>(y);
		pts.push_back(Point2f(x+f[2*x], y+f[2*x+1]));
	}
}

void MatchingMethod(Mat& img, Mat& templ, Rect& rect, Point& matchLoc)
{
	/// 将被显示的原图像
	Mat img_display, result;
	img.copyTo( img_display );
	//result.create(rect.size(), CV_8UC1); 

	int match_method = CV_TM_SQDIFF;//CV_TM_SQDIFF_NORMED;//CV_TM_CCORR;//CV_TM_CCOEFF;//CV_TM_CCOEFF_NORMED;//CV_TM_CCORR_NORMED;
	/// 进行匹配和标准化
	matchTemplate( img(rect), templ, result, match_method );
	normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

	/// 通过函数 minMaxLoc 定位最匹配的位置
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	/// 通过函数 minMaxLoc 定位最匹配的位置
	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
	
	/// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好
	if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{ matchLoc = minLoc; }
	else
	{ matchLoc = maxLoc; }	
	matchLoc.x = matchLoc.x + rect.x;
	matchLoc.y = matchLoc.y + rect.y;

	//// 最终结果
	//rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar(0,0,255), 1, 8, 0 );

	//// 多图同时显示
	//vector<Mat> manyMat;  Mat showManyMat;
	//manyMat.push_back(img_display);
	//imshowMany("showManyImg", manyMat, showManyMat);	
	//manyMat.clear();
}

// 计算CAMHID论文的公式(1)
void ComputeMVs(Mat& prev_grey, Mat grey, std::vector<Point2f>& dst, Mat frame, int frame_num)
{
	int width = grey.cols/mvs_min_distance;
	int height = grey.rows/mvs_min_distance;
	
	int x_max = mvs_min_distance*width;
	int y_max = mvs_min_distance*height;
	
	int cnt = 0;

	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++) { 
		int x = j*mvs_min_distance;
		int y = i*mvs_min_distance;
		
		Point matchLoc, prev_point;
		prev_point.x = x; prev_point.y = y;	// 小心Point对象的成员x,y分别对应列和行
		// 指定前一帧查询模板的ROI区域
		Mat templ = prev_grey(Rect(x, y, mvs_min_distance, mvs_min_distance));  
		// 指定待匹配的当前帧ROI区域
		int xstart = std::max<int>(x-mvs_min_distance, 0);// 求出像素坐标(x,y)，防止越界
		int ystart = std::max<int>(y-mvs_min_distance, 0);
		int xend = std::min<int>(x+2*mvs_min_distance, x_max);
		int yend = std::min<int>(y+2*mvs_min_distance, y_max);
		Rect rect = Rect(xstart, ystart, xend-xstart, yend-ystart);
		
		// 在当前帧grey的ROI区域rect中，匹配上一帧的模板templ，将最佳匹配位置返回当前帧的坐标matchLoc
		MatchingMethod(grey, templ, rect, matchLoc);

		dst[i*width+j].x = prev_point.x - matchLoc.x;
		dst[i*width+j].y = prev_point.y - matchLoc.y;
		if(dst[i*width+j].x != 0 || dst[i*width+j].y != 0)
			cnt++;
	
		//// 测试foreman头盔位移：在连续10帧中看前后两帧的变化
		//if( (i==1&&j==11) || (i==1&&j==12) || (i==1&&j==13) || (i==2&&j==9) || (i==2&&j==10) 
		//	|| (i==2&&j==11) || (i==2&&j==12) || (i==2&&j==13) || (i==2&&j==14) || (i==3&&j==8) 
		//	|| (i==3&&j==9) || (i==3&&j==10) || (i==3&&j==11) || (i==3&&j==12) || (i==3&&j==13) || (i==3&&j==14)
		//	|| (i==4&&j==9) || (i==4&&j==10) || (i==4&&j==11) || (i==4&&j==12) || (i==4&&j==13) || (i==4&&j==14)
		//	)
		//{
		//	// 测试头盔位移：在连续10帧中看前后两帧的变化
		//	DrawTrack(frame, prev_point, matchLoc);	
		//}
	}// for i,j

	// 画出前后两帧模板块发生了位移的新坐标 
	//cout << "sub " << frame_num << ":" << cnt << endl;
	//DrawTrack(frame, dst);	
}

// 计算CAMHID论文的公式(2)中du,dv的标准差，以及公式(3)中mvi的联合概率密度pb_u,pb_v // xzm
void ComputeVariance( std::vector< vector<Point2f> >& mvf, std::vector<Point2f>& dst, Mat& grey)
{
	int size = mvf.size()-1;
	int width = grey.cols/mvs_min_distance;
	int height = grey.rows/mvs_min_distance;

	dst.resize(width*height);
	std::vector<Point2f> mean(width*height);
	std::vector<Point2f> var(width*height);
	std::vector<Point2f> mean_var(width*height);
	std::vector<Mat> sigma(width*height);

	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++) { 	
	float sum_du = 0, sum_dv = 0;
		for(int k = 0; k < size; k++){
			sum_du += mvf[k][i*width+j].x;
			sum_dv += mvf[k][i*width+j].y;
		}
		float mean_du = sum_du/size; 
		float mean_dv = sum_dv/size; 

		mean[i*width+j].x = sum_du/size; 
		mean[i*width+j].y = sum_dv/size; 

		float var_du = 0, var_dv = 0;
		for(int k = 0; k < size; k++){
			float tmp_du = mvf[k][i*width+j].x - mean_du;
			float tmp_dv = mvf[k][i*width+j].y - mean_dv;
			var_du += tmp_du*tmp_du;
			var_dv += tmp_dv*tmp_dv;
		}	

		// 标准差
		//mvf[size][i*width+j].x = sqrt(var_du/size-1);
		//mvf[size][i*width+j].y = sqrt(var_dv/size-1);
		dst[i*width+j].x = sqrt(var_du/(size-1));
		dst[i*width+j].y = sqrt(var_dv/(size-1));
		//dst[i*width+j].x = mvf[0][i*width+j].x - mean_du;
		//dst[i*width+j].y = mvf[0][i*width+j].y - mean_dv;
		var[i*width+j].x = sqrt(var_du/(size-1));
		var[i*width+j].y = sqrt(var_dv/(size-1));
	}// for i,j
	
	//// MTA_CAMHID 双参数方法（不好用）
	//for(int i = 0; i < height; i++)
	//for(int j = 0; j < width; j++) { 	
	//	float mean_dist = 0, tmp_du = 0, tmp_dv = 0;
	//	float sd = 0, sd_du = 0, sd_dv = 0;
	//	for(int k = 0; k < size; k++){
	//		tmp_du = mvf[k][i*width+j].x;
	//		tmp_dv = mvf[k][i*width+j].y;
	//		tmp_du = tmp_du*tmp_du;
	//		tmp_dv = tmp_dv*tmp_dv;
	//		mean_dist += sqrt(tmp_du+tmp_dv);
	//	}
	//	mean_dist = mean_dist/size;

	//	sd_du = var[i*width+j].x;
	//	sd_dv = var[i*width+j].y;
	//	sd = sqrt(sd_du*sd_du + sd_dv*sd_dv);

	//	if(mean_dist < 1.5)// && sd > 0.35)
	//	{	dst[i*width+j].x = 0;	dst[i*width+j].y = 0;	}			
	//}// for i,j



	//// 针对所有网格连续10帧的方差，公式(3)的联合概率密度函数
	//float sum_var_du = 0, sum_var_dv = 0;	
	//for(int i = 0; i < height; i++)
	//for(int j = 0; j < width; j++) { 	
	//	sum_var_du += var[i*width+j].x;
	//	sum_var_dv += var[i*width+j].y;
	//}
	//float mean_var_du = sum_var_du/(width*height); 
	//float mean_var_dv = sum_var_dv/(width*height); 

	//float var_var_du = 0, var_var_dv = 0;
	//float mvs_max_var_du = 0, mvs_max_var_dv = 0;
	//for(int i = 0; i < height; i++)
	//for(int j = 0; j < width; j++) { 
	//	float tmp_du = var[i*width+j].x - mean_var_du;
	//	float tmp_dv = var[i*width+j].y - mean_var_dv;
	//	var_var_du += tmp_du*tmp_du;
	//	var_var_dv += tmp_dv*tmp_dv;

	//	if(var[i*width+j].x > mvs_max_var_du)
	//		mvs_max_var_du = var[i*width+j].x;
	//	if(var[i*width+j].y > mvs_max_var_dv)
	//		mvs_max_var_dv = var[i*width+j].y;
	//}	
	//var_var_du = sqrt(var_var_du/(width*height-1));
	//var_var_dv = sqrt(var_var_dv/(width*height-1));

	//float pb = 0, pb_xy = 0, epsilon = 0.000001;
	//for(int i = 0; i < height; i++)
	//for(int j = 0; j < width; j++) { 	
	//	float var_du = var[i*width+j].x+epsilon;
	//	float var_dv = var[i*width+j].y+epsilon;

	//	sigma[i*width+j].create(2, 2, CV_32FC1);
	//	Mat tmp_sigma = sigma[i*width+j];
	//	tmp_sigma.at<float>(0,0) = var_var_du*var_var_du;
	//	tmp_sigma.at<float>(0,1) = var_var_du*var_var_dv;
	//	tmp_sigma.at<float>(1,0) = var_var_dv*var_var_du;
	//	tmp_sigma.at<float>(1,1) = var_var_dv*var_var_dv;		
	//	float det_sigma = abs(determinant(tmp_sigma));
	//	//float delta_du = (mvs_max_var_du-0)/(width*height);
	//	//float delta_dv = (mvs_max_var_dv-0)/(width*height);
	//	float delta = 1;//sqrt(delta_du*delta_du + delta_dv*delta_dv);

	//	float tmp_u = var[i*width+j].x - mean_var_du;
	//	float tmp_v = var[i*width+j].y - mean_var_dv;
	//	Mat xsbm(2, 1, CV_32FC1);
	//	xsbm.at<float>(0,0) = tmp_u;
	//	xsbm.at<float>(1,0) = tmp_v;
	//	Mat tmp_exp_uv = -0.5*xsbm.t()*tmp_sigma.inv()*xsbm;
	//	float exp_uv = tmp_exp_uv.at<float>(0,0);	

	//	// 正态分布概率密度函数
	//	pb += delta*1/(sqrt((2*3.1415926)*(2*3.1415926)*det_sigma+epsilon))*exp(exp_uv);				
	//}// for i,j

	//for(int i = 0; i < height; i++)
	//for(int j = 0; j < width; j++) { 
	//	float var_du = var[i*width+j].x+epsilon;
	//	float var_dv = var[i*width+j].y+epsilon;

	//	sigma[i*width+j].create(2, 2, CV_32FC1);
	//	Mat tmp_sigma = sigma[i*width+j];
	//	tmp_sigma.at<float>(0,0) = var_var_du*var_var_du;
	//	tmp_sigma.at<float>(0,1) = var_var_du*var_var_dv;
	//	tmp_sigma.at<float>(1,0) = var_var_dv*var_var_du;
	//	tmp_sigma.at<float>(1,1) = var_var_dv*var_var_dv;		
	//	float det_sigma = abs(determinant(tmp_sigma));
	//	//float delta_du = (mvs_max_var[i*width+j].x-0)/(width*height);
	//	//float delta_dv = (mvs_max_var[i*width+j].y-0)/(width*height);
	//	float delta = 1;//sqrt(delta_du*delta_du + delta_dv*delta_dv);

	//	float tmp_u = var[i*width+j].x - mean_var_du;
	//	float tmp_v = var[i*width+j].y - mean_var_dv;
	//	Mat xsbm(2, 1, CV_32FC1);
	//	xsbm.at<float>(0,0) = tmp_u;
	//	xsbm.at<float>(1,0) = tmp_v;
	//	Mat tmp_exp_uv = -0.5*xsbm.t()*tmp_sigma.inv()*xsbm;
	//	float exp_uv = tmp_exp_uv.at<float>(0,0);	

	//	// 正态分布概率密度函数
	//	pb_xy = delta*1/(sqrt((2*3.1415926)*(2*3.1415926)*det_sigma+epsilon))*exp(exp_uv);
	//	pb = pb_xy/(pb+epsilon);
	//	if(pb > 0.25){
	//		dst[i*width+j].x = 0;
	//		dst[i*width+j].y = 0;
	//	}// if pb	
	//}// for i,j



	// （已成功）针对每个网格连续10帧的方差，公式(3)的联合概率密度函数
	// 先对横坐标x,y进行升序排列，然而△x,△y经常为0。
	// 因为x,y坐标值都是整数，所以当没有位移时△x=0,△y=0，导致△x,△y都设为1	
	vector<float> sort_x(0), sort_y(0);
	vector<Point2f> mvi(width*height);
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++) { 
		// 保存连续10帧中的第1帧网格坐标
		mvi[i*width+j].x = mvf[0][i*width+j].x;
		mvi[i*width+j].y = mvf[0][i*width+j].y;
		// 将原始数据放入排序容器
		for(int k = 0; k < size; k++){
			sort_x.push_back(mvf[k][i*width+j].x);
			sort_y.push_back(mvf[k][i*width+j].y);
		}	
		// 升序排列
		sort(sort_x.begin(),sort_x.end());
		sort(sort_y.begin(),sort_y.end());
		// 重新赋值
		for(int k = 0; k < size; k++){
			mvf[k][i*width+j].x = sort_x[k];
			mvf[k][i*width+j].y = sort_y[k];			
		}	
		// 清空排序结果
		sort_x.clear();
		sort_y.clear();
	}

	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++) { 	
		float var_du = var[i*width+j].x+epsilon;
		float var_dv = var[i*width+j].y+epsilon;

		sigma[i*width+j].create(2, 2, CV_32FC1);
		Mat tmp_sigma = sigma[i*width+j];
		tmp_sigma.at<float>(0,0) = var_du;//*var_du;
		tmp_sigma.at<float>(0,1) = var_du*var_dv;
		tmp_sigma.at<float>(1,0) = var_dv*var_du;
		tmp_sigma.at<float>(1,1) = var_dv;//*var_dv;		
		float det_sigma = abs(determinant(tmp_sigma));
		float pb = 0, pb_k = 0, epsilon = 0.000001;

		for(int ku = 0; ku < size; ku++){
			float tmp_u = mvf[ku][i*width+j].x - mean[i*width+j].x;
			float delta_du = 1;//abs(mvf[ku+1][i*width+j].x-mvf[ku][i*width+j].x);
			for(int kv = 0; kv < size; kv++){
				float tmp_v = mvf[kv][i*width+j].y - mean[i*width+j].y;
				float delta_dv = 1;//abs(mvf[kv+1][i*width+j].y-mvf[kv][i*width+j].y);
				Mat xsbm(1, 2, CV_32FC1);
				xsbm.at<float>(0,0) = tmp_u;
				xsbm.at<float>(0,1) = tmp_v;
				Mat tmp_exp_uv = -0.5*xsbm*tmp_sigma.inv()*xsbm.t();
				float exp_uv = tmp_exp_uv.at<float>(0,0);
				// 正态分布概率密度函数
				pb = 1/(sqrt((2*3.1415926)*(2*3.1415926)*det_sigma+epsilon))*exp(-0.5*exp_uv);

				pb += pb*delta_du*delta_dv;
			}// for kv
		}// for ku		
	
		for(int ku = 0; ku < 1; ku++){
			float tmp_u = mvi[i*width+j].x - mean[i*width+j].x;
			float delta_du = 1;//abs(mvf[ku+1][i*width+j].x-mvf[ku][i*width+j].x);
			for(int kv = 0; kv < size; kv++){
				float tmp_v = mvi[i*width+j].y - mean[i*width+j].y;		
				float delta_dv = 1;//abs(mvi[i*width+j].y-mvi[i*width+j].y);
				Mat xsbm(1, 2, CV_32FC1);
				xsbm.at<float>(0,0) = tmp_u;
				xsbm.at<float>(0,1) = tmp_v;
				Mat tmp_exp_uv = -0.5*xsbm*tmp_sigma.inv()*xsbm.t();
				float exp_uv = tmp_exp_uv.at<float>(0,0);

				// 正态分布概率密度函数
				pb_k = 1/(sqrt((2*3.1415926)*(2*3.1415926)*det_sigma+epsilon))*exp(exp_uv);
				pb_k += pb_k*delta_du*delta_dv;	
			}
		}

		pb = pb_k/(pb+epsilon);
		if(pb > 0.25){
			dst[i*width+j].x = 0;
			dst[i*width+j].y = 0;
		}// if pb					
	}// for i,j



	//// 公式(3)的独立概率密度函数
	//for(int i = 0; i < height; i++)
	//for(int j = 0; j < width; j++) { 	
	//	float var_du = var[i*width+j].x+epsilon;
	//	float var_dv = var[i*width+j].y+epsilon;
	//	float var_duv = var_du*var_dv+epsilon; 
	//	float pb_u = 0, pb_v = 0, pb_uv = 0, pb_vu = 0, pb = 0, tmp_pb_u = 0, tmp_pb_v = 0, tmp_pb_uv = 0, tmp_pb_vu = 0, epsilon = 0.000001; 
	//	for(int k = 0; k < size; k++){
	//		float tmp_u = mvf[k][i*width+j].x - mean[i*width+j].x;
	//		float tmp_v = mvf[k][i*width+j].y - mean[i*width+j].y;
	//		float exp_u = -0.5*tmp_u*tmp_u/(var_du*var_du);
	//		float exp_v = -0.5*tmp_v*tmp_v/(var_dv*var_dv);
	//		float exp_uv = -0.5*tmp_u*tmp_u/(var_duv*var_duv);	// 正态分布概率密度函数
	//		float exp_vu = -0.5*tmp_v*tmp_v/(var_duv*var_duv);	// 正态分布概率密度函数

	//		tmp_pb_u += 1/(sqrt(2*3.1415926)*var_du+epsilon)*exp(exp_u);
	//		tmp_pb_v += 1/(sqrt(2*3.1415926)*var_dv+epsilon)*exp(exp_v);
	//		tmp_pb_uv += 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_uv);
	//		tmp_pb_vu += 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_vu);
	//	}// for k	

	//	pb = tmp_pb_u*tmp_pb_v;//*tmp_pb_uv*tmp_pb_uv;

	//	for(int k = 0; k < size; k++){
	//		float tmp_u = mvf[k][i*width+j].x - mean[i*width+j].x;
	//		float tmp_v = mvf[k][i*width+j].y - mean[i*width+j].y;
	//		float exp_u = -0.5*tmp_u*tmp_u/(var_du*var_du);
	//		float exp_v = -0.5*tmp_v*tmp_v/(var_dv*var_dv);
	//		float exp_uv = -0.5*tmp_u*tmp_u/(var_duv*var_duv);	// 正态分布概率密度函数
	//		float exp_vu = -0.5*tmp_v*tmp_v/(var_duv*var_duv);	// 正态分布概率密度函数
	//		pb_u = 1/(sqrt(2*3.1415926)*var_du+epsilon)*exp(exp_u);
	//		pb_v = 1/(sqrt(2*3.1415926)*var_dv+epsilon)*exp(exp_v);
	//		pb_uv = 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_uv);
	//		pb_vu = 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_vu);
	//		pb = pb_u*pb_v/(pb+epsilon);
	//		if(pb > 0.25){
	//			dst[i*width+j].x = 0;
	//			dst[i*width+j].y = 0;
	//		}// if pb	
	//	}// for k				
	//}// for i,j


	// 针对每个网格连续10帧的方差，公式(3)的独立概率密度函数	
	// 先对横坐标x,y进行升序排列，然而△x,△y经常为0。
	// 因为x,y坐标值都是整数，所以当没有位移时△x=0,△y=0，导致△x,△y都设为1	
	//vector<float> sort_x(0), sort_y(0);
	//vector<Point2f> mvi(width*height);
	//for(int i = 0; i < height; i++)
	//for(int j = 0; j < width; j++) { 
	//	// 保存连续10帧中的第1帧网格坐标
	//	mvi[i*width+j].x = mvf[0][i*width+j].x;
	//	mvi[i*width+j].y = mvf[0][i*width+j].y;

	//	for(int k = 0; k < size; k++){
	//		sort_x.push_back(mvf[k][i*width+j].x);
	//		sort_y.push_back(mvf[k][i*width+j].y);
	//	}	
	//	// 升序排列
	//	sort(sort_x.begin(),sort_x.end());
	//	sort(sort_y.begin(),sort_y.end());
	//	// 重新赋值
	//	for(int k = 0; k < size; k++){
	//		mvf[k][i*width+j].x = sort_x[k];
	//		mvf[k][i*width+j].y = sort_y[k];			
	//	}	
	//	// 清空排序结果
	//	sort_x.clear();
	//	sort_y.clear();
	//}

	//for(int i = 0; i < height; i++)
	//for(int j = 0; j < width; j++) { 	
	//	float var_du = var[i*width+j].x+epsilon;
	//	float var_dv = var[i*width+j].y+epsilon;
	//	//float var_duv = var_du*var_dv+epsilon; 
	//	float pb_u = 0, pb_v = 0, pb_uv = 0, pb_vu = 0, pb = 0, tmp_pb = 0, tmp_pb_u = 0, tmp_pb_v = 0, tmp_pb_uv = 0, tmp_pb_vu = 0, epsilon = 0.000001; 
	//	for(int k = 0; k < size; k++){
	//		float tmp_u = mvf[k][i*width+j].x - mean[i*width+j].x;
	//		float tmp_v = mvf[k][i*width+j].y - mean[i*width+j].y;
	//		float exp_u = -0.5*tmp_u*tmp_u/(var_du*var_du+epsilon);
	//		float exp_v = -0.5*tmp_v*tmp_v/(var_dv*var_dv+epsilon);
	//		//float exp_uv = -0.5*tmp_u*tmp_u/(var_duv*var_duv*var_duv*var_duv+epsilon);	// 正态分布概率密度函数
	//		//float exp_vu = -0.5*tmp_v*tmp_v/(var_duv*var_duv*var_duv*var_duv+epsilon);	// 正态分布概率密度函数
	//		tmp_pb_u += 1/(sqrt(2*3.1415926)*var_du+epsilon)*exp(exp_u);
	//		tmp_pb_v += 1/(sqrt(2*3.1415926)*var_dv+epsilon)*exp(exp_v);
	//		//tmp_pb_uv = 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_uv);
	//		//tmp_pb_vu = 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_vu);
	//		//tmp_pb += tmp_pb_u*tmp_pb_v;//*tmp_pb_uv*tmp_pb_vu;
	//	}// for k	

	//	//tmp_pb = tmp_pb/size;
	//				
	//	float tmp_u = mvi[i*width+j].x - mean[i*width+j].x;
	//	float tmp_v = mvi[i*width+j].y - mean[i*width+j].y;
	//	float exp_u = -0.5*tmp_u*tmp_u/(var_du*var_du+epsilon);
	//	float exp_v = -0.5*tmp_v*tmp_v/(var_dv*var_dv+epsilon);
	//	//float exp_uv = -0.5*tmp_u*tmp_u/(var_duv*var_duv*var_duv*var_duv+epsilon);	// 正态分布概率密度函数
	//	//float exp_vu = -0.5*tmp_v*tmp_v/(var_duv*var_duv*var_duv*var_duv+epsilon);	// 正态分布概率密度函数
	//	pb_u = 1/(sqrt(2*3.1415926)*var_du+epsilon)*exp(exp_u);
	//	pb_v = 1/(sqrt(2*3.1415926)*var_dv+epsilon)*exp(exp_v);
	//	//pb_uv = 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_uv);
	//	//pb_vu = 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_vu);
	//	//pb = pb_u*pb_v;//*pb_uv*pb_vu;
	//	//pb = pb/(tmp_pb+epsilon);
	//	pb_u = pb_u/(tmp_pb_u+epsilon);
	//	pb_v = pb_v/(tmp_pb_v+epsilon);

	//	//if(pb < 1./size){
	//	//if(pb >= 0.25*1./size){
	//	if(pb_u >= 0.25 || pb_v >= 0.25){
	//		dst[i*width+j].x = 0;
	//		dst[i*width+j].y = 0;
	//	}// if pb				
	//}// for i,j


	//// 针对每个网格第一帧的情况，公式(3)的独立概率密度函数
	//for(int i = 0; i < height; i++)
	//for(int j = 0; j < width; j++) { 	
	//	float var_du = var[i*width+j].x;
	//	float var_dv = var[i*width+j].y;
	//	float var_duv = var_du*var_dv; 
	//	float pb_u = 0, pb_v = 0, pb = 0, pb_uv = 0, pb_vu = 0; 
	//	for(int k = 0; k < size; k++){
	//		float tmp_u = mvf[k][i*width+j].x - mean[i*width+j].x;
	//		float tmp_v = mvf[k][i*width+j].y - mean[i*width+j].y;
	//		float exp_u = -0.5*tmp_u*tmp_u/(var_du*var_du+epsilon);
	//		float exp_v = -0.5*tmp_v*tmp_v/(var_dv*var_dv+epsilon);
	//		float exp_uv = -0.5*tmp_u*tmp_u/(var_duv*var_duv+epsilon);	// 正态分布概率密度函数
	//		float exp_vu = -0.5*tmp_v*tmp_v/(var_duv*var_duv+epsilon);	// 正态分布概率密度函数
	//		pb_u += 1/(sqrt(2*3.1415926)*var_du+epsilon)*exp(exp_u);
	//		pb_v += 1/(sqrt(2*3.1415926)*var_dv+epsilon)*exp(exp_v);
	//		pb_uv += 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_uv);
	//		pb_vu += 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_vu);
	//	}// for k			
	//	
	//	float tmp_u = mvf[0][i*width+j].x - mean[i*width+j].x;
	//	float tmp_v = mvf[0][i*width+j].y - mean[i*width+j].y;
	//	float exp_u = -0.5*tmp_u*tmp_u/(var_du*var_du+epsilon);
	//	float exp_v = -0.5*tmp_v*tmp_v/(var_dv*var_dv+epsilon);
	//	//float exp_uv = -0.5*tmp_u*tmp_u/(var_duv*var_duv+epsilon);	// 正态分布概率密度函数
	//	//float exp_vu = -0.5*tmp_v*tmp_v/(var_duv*var_duv+epsilon);	// 正态分布概率密度函数
	//	float first_pb_u = 1/(sqrt(2*3.1415926)*var_du+epsilon)*exp(exp_u);
	//	float first_pb_v = 1/(sqrt(2*3.1415926)*var_dv+epsilon)*exp(exp_v);
	//	//float first_pb_uv = 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_uv);
	//	//float first_pb_vu = 1/(sqrt(2*3.1415926)*var_duv+epsilon)*exp(exp_vu);
	//	
	//	pb = (first_pb_u/pb_u+epsilon) * (first_pb_v/pb_v+epsilon);		
	//	//pb = (pb_u+pb_uv)*(pb_v+pb_vu);		// 联合概率密度函数
	//	if(pb > 0.024){
	//		dst[i*width+j].x = 0;
	//		dst[i*width+j].y = 0;
	//	}// if pb
	//}// for i,j
}

#endif /*DESCRIPTORS_H_*/
