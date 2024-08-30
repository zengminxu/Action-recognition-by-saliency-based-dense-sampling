#ifndef _DRAWTRACKS_H_
#define _DRAWTRACKS_H_

#include "CmSaliencyGC.h"

using namespace std;
using namespace cv;

extern string spritLabel;	//使用DenseTrackStab.cpp的全局变量
extern void xzmOptRCBmap(Mat flow, Mat salMask, Mat& wofRCBmap);  //使用DrawOpticalFlow.h的函数

// 针对画图时在第0帧稠密采样的情况
void xzmDrawSample(const Mat& grey, std::vector<Point2f>& points, 
					const double quality, const int min_distance, Mat& salMask);
// 针对画8个图时针对帧数<15时的情况：不用画图，因为轨迹特征长度定为15帧
void xzmDrawSample(const Mat& grey, std::vector<Point2f>& points_idt, std::vector<Point2f>& points_rc,
					std::vector<Point2f>& points_rcb, std::vector<Point2f>& points_mbi, 
					const double quality, const int min_distance, Mat& salMask, Mat& MBI);
// 针对画8个图时针对帧数>=15时的情况
void xzmDrawSample(const Mat& grey, std::vector<Point2f>& points_idt, std::vector<Point2f>& points_rc,
					std::vector<Point2f>& points_rcb, std::vector<Point2f>& points_mbi, 
					const double quality, const int min_distance, Mat& salMask, Mat& MBI,
					Mat& RCmapMulCh, Mat& RCBmapMulCh, Mat& MBIMulCh);
// 针对画10个图时针对帧数<15时的情况：不用画图，因为轨迹特征长度定为15帧
void xzmDrawSample(const Mat& grey, std::vector<Point2f>& points_idt, std::vector<Point2f>& points_rc,
					std::vector<Point2f>& points_rcb, std::vector<Point2f>& points_mbi, 
					std::vector<Point2f>& points_opt, const double quality, const int min_distance,
					Mat& salMask, Mat& MBI, const Mat& warp_flow);			
// 针对画10个图时帧数>=15时的情况
void xzmDrawSample(const Mat& grey, std::vector<Point2f>& points_idt, std::vector<Point2f>& points_rc,
					std::vector<Point2f>& points_rcb, std::vector<Point2f>& points_mbi, 
					std::vector<Point2f>& points_opt, const double quality, const int min_distance,
					Mat& salMask, Mat& MBI, const Mat& warp_flow, Mat& RCmapMulCh, Mat& RCBmapMulCh, 
					Mat& MBIMulCh, Mat& optRCBMulCh);
// 生成轨迹特征的同时，直接画出10张图：原图、MBI、RC-map、RCB-map、wofRCB-map；IDT、IDT-MBI、IDT-RC、IDT-RCB、IDT-wofRCB
int myDrawImage(vector<string>, multimap<string, string>, 
					string, string, string, string, string, string);
// （早期函数仅供参考）根据已生成的轨迹特征文件，画出1-2张IDT轨迹图
int myDrawTrack(vector<string>, multimap<string, string>, 
					string, string, string, string, string, string);
void DrawBoundBox(vector<BoundBox>, Mat&);
void motionToColor(Mat flow, Mat &color);
void DrawManyTrack(const vector<vector<Point2f>>&, const int, const float, Mat&);
void DrawManyTrack(const vector<vector<Point2f>>&, const vector<vector<Point2f>>&, 
					const int, const float, Mat&, Mat&, int);
void imshowMany(const string&, const vector<Mat>&, Mat&);

#ifdef _WIN32 // Windows version
volatile LONG semaphore_image = 0;
DWORD WINAPI drawImageMultiThread( LPVOID lpParameter )
#else // Linux version
sem_t semaphore_image;
static void *drawImageMultiThread(void *lpParameter)
#endif
{
	ThreadParam *lp = (ThreadParam*)lpParameter;
	
	VideoCapture capture;
	string video = lp->datasetPath + lp->filename + lp->extname;
	capture.open(video);

	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);				// 初始化轨迹信息，每个特征点跟踪15帧，间隔1帧就重新提取特征点
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);// 时空卷描述符赋初值，除了HOF是9bins外，其他都是8bins直方图
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);

	vector<Frame> bb_list;
	string str_bb_file = lp->bb_file_thrd;
	//if(bb_file){ 
	if( !str_bb_file.empty() ){	
		LoadBoundBox(str_bb_file, bb_list);
		assert(bb_list.size() == seqInfo.length);
	}

	// 初始化SURF特征点
	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);

	// 初始化前后两帧的特征点容器
	vector<Point2f> prev_pts_flow, pts_flow;
	vector<Point2f> prev_pts_surf, pts_surf;
	vector<Point2f> prev_pts_all, pts_all;

	// 初始化前后两帧的特征"点对"
	vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;
	
	Mat image, prev_grey, grey;
	Mat image_rc, image_rcb, image_mbi, image_opt;

	vector<float> fscales(0);
	vector<Size> sizes(0);

	vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);	// 0表示该Mat矩阵初始化元素为0
	vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);				// for optical flow

	// xzm MBI和显著度掩码金字塔
	vector<Mat> salMask_pyr(0), MBI_pyr(0);
	//保存前15帧图像
	queue<Mat> image_que, image_rc_que, image_rcb_que, image_mbi_que, image_opt_que;
	// xzm 计算特征点数和运行时间
	int cnt = 0; 
	double t = double( getTickCount() );
	// 不同尺度下特征点跟踪轨迹的容器. xyScaleTracks[i]表示第i个尺度下的跟踪轨迹，每条轨迹上有多个被跟踪的特征点
	vector<list<Track> > xyScaleTracks, xyScaletracks_rc, xyScaletracks_rcb, xyScaleTracks_mbi, xyScaleTracks_opt;
	int init_counter = 0; // indicate when to detect new feature points

	while(true) {		
		Mat frame, salMask, RCmapMulCh, RCBmapMulCh, MBIMulCh, optRCBMulCh;
		int i;

		capture >> frame;
		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}		

		int rawRows = frame.rows;
		int rawCols = frame.cols;
		int extRows = frame.rows * 2 + 9;		
		//int extCols = frame.cols * 4 + 15;	// for  8个图
		int extCols = frame.cols * 5 + 18;		// for 10个图
		Mat compFrame = Mat::zeros(Size(extCols, extRows), CV_8UC3);

		// Rect(y, x, width, height) 注意坐标是(列，行)顺序！
		Mat roiRawImage(compFrame, Rect(3, 3, rawCols, rawRows));
		Mat roiMBI(compFrame, Rect(rawCols+6, 3, rawCols, rawRows));
		Mat roiRCmap(compFrame, Rect(rawCols*2+9, 3, rawCols, rawRows));
		Mat roiRCBmap(compFrame, Rect(rawCols*3+12, 3, rawCols, rawRows));
		Mat roiwofRCBmap(compFrame, Rect(rawCols*4+15, 3, rawCols, rawRows));
		Mat roiIDTTrack(compFrame, Rect(3, rawRows+6, rawCols, rawRows));
		Mat roiIDTMBITrack(compFrame, Rect(rawCols+6, rawRows+6, rawCols, rawRows));
		Mat roiIDTRCTrack(compFrame, Rect(rawCols*2+9, rawRows+6, rawCols, rawRows));
		Mat roiIDTRCBTrack(compFrame, Rect(rawCols*3+12, rawRows+6, rawCols, rawRows));
		Mat roiIDTOptRCBTrack(compFrame, Rect(rawCols*4+15, rawRows+6, rawCols, rawRows));

		Mat marUpRow(compFrame, Rect(0, 0, extCols, 3));
		Mat marMidRow(compFrame, Rect(0, rawRows+3, extCols, 3));
		Mat marDownRow(compFrame, Rect(0, rawRows*2+6, extCols, 3));
		Mat marLefCol(compFrame, Rect(0, 0, 3, extRows));
		Mat marMidoneCol(compFrame, Rect(rawCols+3, 0, 3, extRows));
		Mat marMidtwoCol(compFrame, Rect(rawCols*2+6, 0, 3, extRows));
		Mat marMidthreeCol(compFrame, Rect(rawCols*3+9, 0, 3, extRows));
		Mat marMidfourCol(compFrame, Rect(rawCols*4+12, 0, 3, extRows));
		Mat marRigCol(compFrame, Rect(rawCols*5+15, 0, 3, extRows));
		marUpRow = Scalar(255,255,255);
		marMidRow = Scalar(255,255,255);
		marDownRow = Scalar(255,255,255);
		marLefCol = Scalar(255,255,255);
		marMidoneCol = Scalar(255,255,255);
		marMidtwoCol = Scalar(255,255,255);
		marMidthreeCol = Scalar(255,255,255);
		marMidfourCol = Scalar(255,255,255);
		marRigCol = Scalar(255,255,255);

		if(frame_num == start_frame) {	// 先对第一帧进行初始化操作
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			// 初始化图像金字塔（注：InitPry应改为InitPyr才准确！）
			InitPry(frame, fscales, sizes);	// 将视频帧划分多个层次的尺度和大小分别存入fscales和sizes

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);

			// xzm
			BuildPry(sizes, CV_8UC1, salMask_pyr);
			BuildPry(sizes, CV_8UC1, MBI_pyr);

			xyScaleTracks.resize(scale_num);
			xyScaletracks_rc.resize(scale_num);
			xyScaletracks_rcb.resize(scale_num);
			xyScaleTracks_mbi.resize(scale_num);
			xyScaleTracks_opt.resize(scale_num);

			// xzm
			CmSaliencyGC::XZM(frame, salMask);
			image_que.push(frame);		image_rc_que.push(frame); 
			image_rcb_que.push(frame);	image_mbi_que.push(frame);	image_opt_que.push(frame);
			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

			// 将每个视频帧在所有尺度下的图像进行密集采样
			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0){		// 金字塔最底层，图像分辨率最高
					prev_grey.copyTo(prev_grey_pyr[0]);
					salMask.copyTo(salMask_pyr[0]);
				}
				else{					// 金字塔通过线性插值往上层缩小图像尺寸
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
					resize(salMask_pyr[iScale-1], salMask_pyr[iScale], salMask_pyr[iScale].size(), 0, 0, INTER_LINEAR);
				}

				// 在当前帧对所有尺度进行特征点密集采样
				std::vector<Point2f> points(0),points_rc(0),points_rcb(0),points_mbi(0),points_opt(0);
				// 根据邻域大小W=5进行特征点密集采样. min_distance就是论文的W.
				xzmDrawSample(prev_grey_pyr[iScale], points, points_rc, points_rcb, points_mbi, 
								quality, min_distance, salMask_pyr[iScale], MBI_pyr[iScale]);

				// 保存特征点(的轨迹).因为每个特征点都有一条轨迹，所以也可以认为是保存轨迹
				std::list<Track>& tracks = xyScaleTracks[iScale];
				std::list<Track>& tracks_rc = xyScaletracks_rc[iScale];
				std::list<Track>& tracks_rcb = xyScaletracks_rcb[iScale];
				std::list<Track>& tracks_mbi = xyScaleTracks_mbi[iScale];
				std::list<Track>& tracks_opt = xyScaleTracks_opt[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
				for(i = 0; i < points_rc.size(); i++)
					tracks_rc.push_back(Track(points_rc[i], trackInfo, hogInfo, hofInfo, mbhInfo));
				for(i = 0; i < points_rcb.size(); i++)
					tracks_rcb.push_back(Track(points_rcb[i], trackInfo, hogInfo, hofInfo, mbhInfo));
				for(i = 0; i < points_mbi.size(); i++)
					tracks_mbi.push_back(Track(points_mbi[i], trackInfo, hogInfo, hofInfo, mbhInfo));
				// 第一帧没有光流，直接用RCBmap作为wofRCBmap
				tracks_opt = tracks_rcb;
			}

			// compute polynomial expansion
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			// 使用行人检测器human_mask
			human_mask = Mat::ones(frame.size(), CV_8UC1);
			//if(bb_file)
			if(!str_bb_file.empty())
				InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
			// 提取当前帧的SURF，与下一帧的SURF点一起作为可能存在的背景噪声点对，排除行人检测器human_mask中的特征点
			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);

			frame_num++;
			continue;
		}

		// 开始处理第二帧	
		init_counter++;
		CmSaliencyGC::XZM(frame, salMask);
		if( frame_num > mvf_length+1 ){
			image_que.pop(); image_rc_que.pop(); image_rcb_que.pop(); image_mbi_que.pop(); image_opt_que.pop();
		}
		image_que.push(frame);		image_rc_que.push(frame); 
		image_rcb_que.push(frame);	image_mbi_que.push(frame);	image_opt_que.push(frame);
		frame.copyTo(image);		frame.copyTo(image_rc);
		frame.copyTo(image_rcb);	frame.copyTo(image_mbi);	frame.copyTo(image_opt);
		cvtColor(image, grey, CV_BGR2GRAY);	
//////////////////////////////////////

		// 使用行人检测器human_mask
		//if(bb_file)
		if(!str_bb_file.empty())
			InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
		// 提取当前帧的SURF，与前一帧的SURF点一起作为可能存在的背景噪声点对，排除行人检测器human_mask中的特征点
		detector_surf.detect(grey, kpts_surf, human_mask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

		// 一次过计算所有尺度下的光流场
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		// 从稠密光流中通过前后两帧的Strong Corner，提取可能存在的背景噪声点对，排除行人检测器human_mask中的特征点
		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);
		// 将前后两帧提取出的SURF和Strong Corner特征点对合并在一起，准备计算透视变换矩阵
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

		// 通过SURF和Strong Corner的点对计算透视变换矩阵H，并用RANSAC对点对进行提纯输出到match_mask
		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_all.size() > 50) {
			vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if(countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}

		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // 用透视变换矩阵H_inv对第二帧进行透视变换

		// 重新计算第二帧所有尺度下的光流
		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);
		
		// 在每个尺度iScale下跟踪特征点。测试视频person01.avi被分为4个尺度
		for(int iScale = 0; iScale < scale_num; iScale++) {
			if(iScale == 0){
				grey.copyTo(grey_pyr[0]);
				salMask.copyTo(salMask_pyr[0]);
			}
			else{
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
				resize(salMask_pyr[iScale-1], salMask_pyr[iScale], salMask_pyr[iScale].size(), 0, 0, INTER_LINEAR);
			}

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;
			
			// IDT：分别计算HOG/HOF/MBH三种特征的直方图，注意HOG用的是梯度信息，而HOF/MBH用的是经过透视变换重新计算的光流场
			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo);

			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);	
			//MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);
			// pxj比wh多加了一个参数MBI
			MbhComp(flow_pyr[iScale], MBI_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);
			
			// tracks对应一种尺度下所有特征点所形成的所有轨迹(链表形式存储)
			list<Track>& tracks = xyScaleTracks[iScale];
			for (list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;							// iTrack为当前尺度下某个被跟踪的特征点指针，index表示该点被跟踪到第index帧
				Point2f prev_point = iTrack->point[index];			// 特指一个被跟踪的特征点
				int x = min<int>(max<int>(cvRound(prev_point.x), 0), width-1);// 近似求出当前特征点的坐标(x,y)，防止越界
				int y = min<int>(max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];	// Pt+1 = (xt+1, yt+1) = Pt + 光流场（见论文公式1）
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				// 将重新计算后的光流场作为可能的背景噪声位移存入iTrack特征点第index个新坐标
				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];	
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
				
				// 获取轨迹特征的描述符！
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);	// 根据前一个特征点坐标，定位时空卷描述符的ROI区域放入rect
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				
				// 将特征点point当前跟踪的帧数index+1，再将该点加入iTrack->point[index]
				iTrack->addPoint(point);

				// 如果轨迹的连续帧数达到L = 15，则将该轨迹0-15连续16帧的坐标放入轨迹容器
				if(iTrack->index >= trackInfo.length) {
					vector<Point2f> trajectory(trackInfo.length+1);// 此length是连续跟踪的帧数L=15，与下面的轨迹长度length不同
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];
				
					vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];
	
					// 初始化为0. 注意：此length为轨迹长度，初始化为0后作参数输入IsValid函数
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);										
					// 检查是否动作轨迹。如果轨迹中单次位移方差小于1个像素就认为是相机运动产生的轨迹，否则才是人体动作轨迹
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {	
						DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);
					}

					iTrack = tracks.erase(iTrack);// 删除当前特征点
					continue;
				}
				++iTrack;
			}// for iTrack

			// 程明明显著图掩码RC-map：在每个尺度iScale下跟踪特征点。
			list<Track>& tracks_rc = xyScaletracks_rc[iScale];		
			for (list<Track>::iterator iTrack = tracks_rc.begin(); iTrack != tracks_rc.end();) {
				int index = iTrack->index;							// iTrack为当前尺度下某个被跟踪的特征点指针，index表示该点被跟踪到第index帧
				Point2f prev_point = iTrack->point[index];			// 特指一个被跟踪的特征点
				int x = min<int>(max<int>(cvRound(prev_point.x), 0), width-1);// 近似求出当前特征点的坐标(x,y)，防止越界
				int y = min<int>(max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];	// Pt+1 = (xt+1, yt+1) = Pt + 光流场（见论文公式1）
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks_rc.erase(iTrack);
					continue;
				}

				// 将重新计算后的光流场作为可能的背景噪声位移存入iTrack特征点第index个新坐标
				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];	
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
				
				// 获取轨迹特征的描述符！
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);	// 根据前一个特征点坐标，定位时空卷描述符的ROI区域放入rect
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				
				// 将特征点point当前跟踪的帧数index+1，再将该点加入iTrack->point[index]
				iTrack->addPoint(point);

				// 如果轨迹的连续帧数达到L = 15，则将该轨迹0-15连续16帧的坐标放入轨迹容器
				if(iTrack->index >= trackInfo.length) {
					vector<Point2f> trajectory(trackInfo.length+1);// 此length是连续跟踪的帧数L=15，与下面的轨迹长度length不同
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];
				
					vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];
	
					// 初始化为0. 注意：此length为轨迹长度，初始化为0后作参数输入IsValid函数
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);										
					// 检查是否动作轨迹。如果轨迹中单次位移方差小于1个像素就认为是相机运动产生的轨迹，否则才是人体动作轨迹
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {	
						DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image_rc);
					}

					iTrack = tracks_rc.erase(iTrack);// 删除当前特征点
					continue;
				}
				++iTrack;
			}// for iTrack

			// 改进的显著图掩码RCB-map：在每个尺度iScale下跟踪特征点。
			list<Track>& tracks_rcb = xyScaletracks_rcb[iScale];		
			for (list<Track>::iterator iTrack = tracks_rcb.begin(); iTrack != tracks_rcb.end();) {
				int index = iTrack->index;							// iTrack为当前尺度下某个被跟踪的特征点指针，index表示该点被跟踪到第index帧
				Point2f prev_point = iTrack->point[index];			// 特指一个被跟踪的特征点
				int x = min<int>(max<int>(cvRound(prev_point.x), 0), width-1);// 近似求出当前特征点的坐标(x,y)，防止越界
				int y = min<int>(max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];	// Pt+1 = (xt+1, yt+1) = Pt + 光流场（见论文公式1）
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks_rcb.erase(iTrack);
					continue;
				}

				// 将重新计算后的光流场作为可能的背景噪声位移存入iTrack特征点第index个新坐标
				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];	
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
				
				// 获取轨迹特征的描述符！
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);	// 根据前一个特征点坐标，定位时空卷描述符的ROI区域放入rect
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				
				// 将特征点point当前跟踪的帧数index+1，再将该点加入iTrack->point[index]
				iTrack->addPoint(point);

				// 如果轨迹的连续帧数达到L = 15，则将该轨迹0-15连续16帧的坐标放入轨迹容器
				if(iTrack->index >= trackInfo.length) {
					vector<Point2f> trajectory(trackInfo.length+1);// 此length是连续跟踪的帧数L=15，与下面的轨迹长度length不同
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];
				
					vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];
	
					// 初始化为0. 注意：此length为轨迹长度，初始化为0后作参数输入IsValid函数
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);										
					// 检查是否动作轨迹。如果轨迹中单次位移方差小于1个像素就认为是相机运动产生的轨迹，否则才是人体动作轨迹
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {	
						DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image_rcb);
					}

					iTrack = tracks_rcb.erase(iTrack);// 删除当前特征点
					continue;
				}
				++iTrack;
			}// for iTrack

			// MBI显著图：在每个尺度iScale下跟踪特征点。
			list<Track>& tracks_mbi = xyScaleTracks_mbi[iScale];		
			for (list<Track>::iterator iTrack = tracks_mbi.begin(); iTrack != tracks_mbi.end();) {
				int index = iTrack->index;							// iTrack为当前尺度下某个被跟踪的特征点指针，index表示该点被跟踪到第index帧
				Point2f prev_point = iTrack->point[index];			// 特指一个被跟踪的特征点
				int x = min<int>(max<int>(cvRound(prev_point.x), 0), width-1);// 近似求出当前特征点的坐标(x,y)，防止越界
				int y = min<int>(max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];	// Pt+1 = (xt+1, yt+1) = Pt + 光流场（见论文公式1）
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks_mbi.erase(iTrack);
					continue;
				}

				// 将重新计算后的光流场作为可能的背景噪声位移存入iTrack特征点第index个新坐标
				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];	
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
				
				// 获取轨迹特征的描述符！
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);	// 根据前一个特征点坐标，定位时空卷描述符的ROI区域放入rect
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				
				// 将特征点point当前跟踪的帧数index+1，再将该点加入iTrack->point[index]
				iTrack->addPoint(point);

				// 如果轨迹的连续帧数达到L = 15，则将该轨迹0-15连续16帧的坐标放入轨迹容器
				if(iTrack->index >= trackInfo.length) {
					vector<Point2f> trajectory(trackInfo.length+1);// 此length是连续跟踪的帧数L=15，与下面的轨迹长度length不同
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];
				
					vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];
	
					// 初始化为0. 注意：此length为轨迹长度，初始化为0后作参数输入IsValid函数
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);										
					// 检查是否动作轨迹。如果轨迹中单次位移方差小于1个像素就认为是相机运动产生的轨迹，否则才是人体动作轨迹
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {	
						DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image_mbi);
					}

					iTrack = tracks_mbi.erase(iTrack);// 删除当前特征点
					continue;
				}
				++iTrack;
			}// for iTrack

			// SCI：修正光流场+显著图掩码optRCB-map：在每个尺度iScale下跟踪特征点
			list<Track>& tracks_opt = xyScaleTracks_opt[iScale];		
			for (list<Track>::iterator iTrack = tracks_opt.begin(); iTrack != tracks_opt.end();) {
				int index = iTrack->index;							// iTrack为当前尺度下某个被跟踪的特征点指针，index表示该点被跟踪到第index帧
				Point2f prev_point = iTrack->point[index];			// 特指一个被跟踪的特征点
				int x = min<int>(max<int>(cvRound(prev_point.x), 0), width-1);// 近似求出当前特征点的坐标(x,y)，防止越界
				int y = min<int>(max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];	// Pt+1 = (xt+1, yt+1) = Pt + 光流场（见论文公式1）
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks_opt.erase(iTrack);
					continue;
				}

				// 将重新计算后的光流场作为可能的背景噪声位移存入iTrack特征点第index个新坐标
				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];	
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
				
				// 获取轨迹特征的描述符！
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);	// 根据前一个特征点坐标，定位时空卷描述符的ROI区域放入rect
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				
				// 将特征点point当前跟踪的帧数index+1，再将该点加入iTrack->point[index]
				iTrack->addPoint(point);

				// 如果轨迹的连续帧数达到L = 15，则将该轨迹0-15连续16帧的坐标放入轨迹容器
				if(iTrack->index >= trackInfo.length) {
					vector<Point2f> trajectory(trackInfo.length+1);// 此length是连续跟踪的帧数L=15，与下面的轨迹长度length不同
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];
				
					vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];
	
					// 初始化为0. 注意：此length为轨迹长度，初始化为0后作参数输入IsValid函数
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);										
					// 检查是否动作轨迹。如果轨迹中单次位移方差小于1个像素就认为是相机运动产生的轨迹，否则才是人体动作轨迹
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {	
						DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image_opt);
					}

					iTrack = tracks_opt.erase(iTrack);// 删除当前特征点
					continue;
				}
				++iTrack;
			}// for iTrack

			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			// 每间隔init_gap帧就重新检测新的特征点。
			if(init_counter != trackInfo.gap)
				continue;

			// 把根据光流场计算出的原特征点的新坐标和跟踪的帧数，压入一个新的特征点容器points。
			vector<Point2f> points(0), points_rc(0), points_rcb(0), points_mbi(0), points_opt(0);
			for(list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);
			for(list<Track>::iterator iTrack = tracks_rc.begin(); iTrack != tracks_rc.end(); iTrack++)
				points_rc.push_back(iTrack->point[iTrack->index]);
			for(list<Track>::iterator iTrack = tracks_rcb.begin(); iTrack != tracks_rcb.end(); iTrack++)
				points_rcb.push_back(iTrack->point[iTrack->index]);
			for(list<Track>::iterator iTrack = tracks_mbi.begin(); iTrack != tracks_mbi.end(); iTrack++)
				points_mbi.push_back(iTrack->point[iTrack->index]);
			for(list<Track>::iterator iTrack = tracks_opt.begin(); iTrack != tracks_opt.end(); iTrack++)
				points_opt.push_back(iTrack->point[iTrack->index]);

			// xzm比wh多加入一个MBH参数MBI_pyr[]和一个掩码参数salMask_pyr[]
			// 在当前帧对所有尺度再次进行网格密集采样，在没有特征点的网格中检测新的特征点points（新的特征点个数会少于轨迹tracks的原特征点个数）
			if(frame_num >= track_length){
				//xzmDrawSample(grey_pyr[iScale], points, points_rc, points_rcb, points_mbi, 
				//				quality, min_distance, salMask_pyr[iScale], MBI_pyr[iScale], 
				//				RCmapMulCh, RCBmapMulCh, MBIMulCh);
				xzmDrawSample(grey_pyr[iScale], points, points_rc, points_rcb, points_mbi,
								points_opt,	quality, min_distance, salMask_pyr[iScale], 
								MBI_pyr[iScale], flow_warp_pyr[iScale], RCmapMulCh, RCBmapMulCh, 
								MBIMulCh, optRCBMulCh);

				// 显示原图和各种显著图
				frame.copyTo(roiRawImage);
				RCmapMulCh.copyTo(roiRCmap);
				RCBmapMulCh.copyTo(roiRCBmap);
				MBIMulCh.copyTo(roiMBI);
				optRCBMulCh.copyTo(roiwofRCBmap);
				image.copyTo(roiIDTTrack);
				image_rc.copyTo(roiIDTRCTrack);
				image_rcb.copyTo(roiIDTRCBTrack);
				image_mbi.copyTo(roiIDTMBITrack);
				image_opt.copyTo(roiIDTOptRCBTrack);
			}
			else
				//xzmDrawSample(grey_pyr[iScale], points, points_rc, points_rcb, points_mbi, 
				//				quality, min_distance, salMask_pyr[iScale], MBI_pyr[iScale]);
				xzmDrawSample(grey_pyr[iScale], points, points_rc, points_rcb, points_mbi,
			    			  points_opt, quality, min_distance, salMask_pyr[iScale], 
							  MBI_pyr[iScale], flow_warp_pyr[iScale]);

			// 将新的特征点信息压入轨迹tracks末尾，之前已被跟踪了若干帧(<15帧)的特征点仍然有效，在下一次循环中继续定位每个特征点的坐标和轨迹，直至该点连续跟踪超过15帧后被移除.
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			for(i = 0; i < points_mbi.size(); i++)
				tracks_mbi.push_back(Track(points_mbi[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			for(i = 0; i < points_rc.size(); i++)
				tracks_rc.push_back(Track(points_rc[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			for(i = 0; i < points_rcb.size(); i++)
				tracks_rcb.push_back(Track(points_rcb[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			// 当IDT-wofRCB采样特征点过少时，直接用idt特征填补！
			if(points_opt.size() < min_sample_ratio*points.size()){
				points_opt.clear();
				points_opt = points;}
			for(i = 0; i < points_opt.size(); i++)
				tracks_opt.push_back(Track(points_opt[i], trackInfo, hogInfo, hofInfo, mbhInfo));

			//if(frame_num >= 30){
			//	stringstream ss, sscale;
			//	ss << frame_num;  sscale << iScale;
			//	string fullFilename = "D:\\XZM\\Data\\" + lp->filename + "_s" + sscale.str() + "_" + ss.str() + ".jpg";
			//	vector<int> comp_paras;
			//	comp_paras.push_back(95); 
			//	try{ 
			//		imwrite(fullFilename, compFrame, comp_paras);
			//	}
			//	catch (runtime_error& ex){
			//		cerr << "Exception converting image to PNGformat: %s" << endl;
			//		return 0;
			//	}
			//	comp_paras.clear();			
			//}

		}// for iScale

/////////////////////////////////////

		// 间隔1帧画1张轨迹图
		if(frame_num >= track_length){// && frame_num%2 == 1){
			stringstream ss;
			ss << frame_num;
			string fullFilename = "D:\\XZM\\Data\\" + lp->filename + "_0_" + ss.str() + ".jpg";
			vector<int> comp_paras;
			//comp_paras.push_back(CV_IMWRITE_JPEG_QUALITY); //CV_IMWRITE_PNG_COMPRESSION);
			comp_paras.push_back(95); //9);
			try{ 
				imwrite(fullFilename, compFrame, comp_paras);
			}
			catch (runtime_error& ex){
				cerr << "Exception converting image to PNGformat: %s" << endl;
				return 0;
			}
			comp_paras.clear();
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		// xzm
		salMask.release();
		RCmapMulCh.release();
		RCBmapMulCh.release();
		MBIMulCh.release();
		optRCBMulCh.release();

		frame_num++;

	}// while(true)

	// 释放资源
	delete lp;

#ifdef _WIN32 // Windows version
	// 结束提取特征，信号量-1
	_InterlockedDecrement( &semaphore_image );
#else // Linux version
	sem_wait( &semaphore_image );
	pthread_detach(pthread_self());
#endif

	return 0;

}

// （早期函数仅供参考）根据已生成的轨迹特征文件，画出1-2张IDT轨迹图
int myDrawTrack(vector<string> actionType, multimap<string, string> actionSet,
					 string datasetPath, string resultPath, string bbPath,
					 string featurePath, string processType, string datasetName)
{
	int frame_num = 0;
	VideoCapture capture;
	ActionAnalysis action;
	Mat tmpMat_0, tmpMat_1;
	int once = 0;

	// 找出每类动作在训练时所用到的视频
	vector<string>::iterator itype;
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{	
		string actionTypeStr = *itype;
		multimap<string, string>::iterator iter;
		for(iter = actionSet.begin(); iter != actionSet.end(); iter++)
		{
			//if( iter->first != "brush_hair") // 提高运行速度
			//	continue;
			if( iter->first == actionTypeStr )
			{
				if( iter->second != "Brushing_my_Long_Hair__February_2009_brush_hair_u_nm_np1_ba_goo_1" && iter->second != "Brushing_my_Long_Hair__February_2009_brush_hair_u_nm_np1_ba_goo_2")//"AHF_Rapier_fencing_f_cm_np2_le_bad_0" && iter->second != "AHF_Rapier_fencing_f_cm_np2_le_bad_1")//"v_BaseballPitch_g03_c03")//"#20_Rhythm_clap_u_nm_np1_fr_goo_1")//"#122_Cleaning_Up_The_Beach_In_Chiba__Japan_pick_f_nm_np1_le_bad_2")//"Documentario_Le_Parkour_Londrina_jump_f_nm_np1_ba_bad_1")//"Documentario_Le_Parkour_Londrina_jump_f_cm_np1_le_bad_11")//"show_your_smile_-)_smile_h_nm_np1_fr_med_0"))
					continue;
				      
				if(datasetName == "hmdb51")
					if(once++ < 1)	// 避免datasetPath 累加几次
						datasetPath = datasetPath + "hmdb51_org" + spritLabel + actionTypeStr + spritLabel;
				else if(datasetName == "Hollywood2")
					datasetPath = datasetPath + "AVIClips" + spritLabel;
				else if(datasetName == "jhmdb")
					datasetPath = datasetPath + "jhmdb_org" + spritLabel + actionTypeStr + spritLabel;
				else if(datasetName == "UCF50" || datasetName == "UCF_Sports" || 
						datasetName == "KTH" || datasetName == "weizemann")
					datasetPath = datasetPath + actionTypeStr + spritLabel;
				
				string video_file = featurePath + iter->second + ".bin";
				if( iter->second == "Brushing_my_Long_Hair__February_2009_brush_hair_u_nm_np1_ba_goo_1")//"AHF_Rapier_fencing_f_cm_np2_le_bad_0" )
					action.bin2Mat(video_file, tmpMat_0, "drawTrack");
				if( iter->second == "Brushing_my_Long_Hair__February_2009_brush_hair_u_nm_np1_ba_goo_2")//"AHF_Rapier_fencing_f_cm_np2_le_bad_1" )
					action.bin2Mat(video_file, tmpMat_1, "drawTrack");

			}// if(iter->first)
		}// for(iter)
	}// for(itype)

	string video = datasetPath + "Brushing_my_Long_Hair__February_2009_brush_hair_u_nm_np1_ba_goo_1.avi";//"AHF_Rapier_fencing_f_cm_np2_le_bad_0.avi";
	capture.open(video);
	vector<vector<Point2f>> trajectories_0(0), trajectories_1(0);
	while(true) {		
		Mat frame;
		int i, j, c;					
		vector<Point2f> points(0);

		// get a new frame
		capture >> frame;
		if(frame.empty())
			break;

		// 因为IDT轨迹是取连续15帧的角点坐标位移信息，所以只需要画出第15帧之后的轨迹
		if(frame_num < start_frame || frame_num > end_frame || frame_num < 15) {
			frame_num++;
			continue;
		}
				
		for(int i=0; i<tmpMat_0.rows; i++){
			Point2f point;
			// 找到当前帧，只取尺度为1的轨迹做测试
			if( tmpMat_0.at<float>(i,0) == frame_num && tmpMat_0.at<float>(i,1) == 1 ){
				for(int j=2; j<tmpMat_0.cols; j+=2){
					point.x = tmpMat_0.at<float>(i,j);
					point.y = tmpMat_0.at<float>(i,j+1);
					points.push_back(point);
				}

				trajectories_0.push_back(points);
			}			
			points.clear();
			continue;	// 找到当前帧后无须再遍历其他帧
		}

		for(int i=0; i<tmpMat_1.rows; i++){
			Point2f point;
			// 找到当前帧，只取尺度为1的轨迹做测试
			if( tmpMat_1.at<float>(i,0) == frame_num && tmpMat_1.at<float>(i,1) == 1 ){
				for(int j=2; j<tmpMat_1.cols; j+=2){
					point.x = tmpMat_1.at<float>(i,j);
					point.y = tmpMat_1.at<float>(i,j+1);
					points.push_back(point);
				}

				trajectories_1.push_back(points);
			}			
			points.clear();
			continue;	// 找到当前帧后无须再遍历其他帧
		}

		Mat frame_1 = frame.clone(); // 在DrawManyTrack()中分别显示两幅图像 
		DrawManyTrack(trajectories_0, trajectories_1, 15, 1, frame, frame_1, frame_num);

		trajectories_0.clear();
		trajectories_1.clear();
		frame_num++;
	}// while(true)

	return 0;
}

void DrawBoundBox(vector<BoundBox> bbs, Mat& image)
{
	for(int k=0; k<bbs.size(); k++)
	{	
		BoundBox& bb = bbs[k];				
		Point2f point0 = bbs[k].TopLeft;
		Point2f point1 = bbs[k].BottomRight;
		rectangle(image, point0, point1, Scalar(0,255,0), 2, 8, 0);
	}
	//imshow( "DenseTrackStab", image);
	//waitKey(0);
}

// 生成轨迹特征的同时，直接画出10张图：原图、MBI、RC-map、RCB-map、wofRCB-map；IDT、IDT-MBI、IDT-RC、IDT-RCB、IDT-wofRCB
int myDrawImage(vector<string> actionType, multimap<string, string> actionSet,
					 string datasetPath, string resultPath, string bbPath,
					 string featurePath, string processType, string datasetName)
{
	// 每类动作随机取sampleNum段视频
	int sampleNum = 20;
	vector<string> tmpActionSet, randActionSet;

	// 找出每类动作在训练时所用到的视频
	vector<string>::iterator itype;
	for(itype=actionType.begin(); itype<actionType.end(); itype++) 
	{	
		multimap<string, string>::iterator iter;
		for(iter = actionSet.begin(); iter != actionSet.end(); iter++){
			if( iter->first == *itype )
				tmpActionSet.push_back(iter->second);
		}
	}// for(itype)
					
	RNG rng(getTickCount());	// 取当前系统时间作为随机数种子
	Mat randMat(tmpActionSet.size(), 1, CV_32S);
	for(int r=0; r<randMat.rows; r++)
		randMat.at<int>(r,0) = r;
	randShuffle(randMat, 1, &rng);
	
	for(int r=0; r<sampleNum; r++){	// 每类动作随机取sampleNum段视频
		int randRow = randMat.at<int>(r,0);
		randActionSet.push_back(tmpActionSet[randRow]);
	}			

	//for(int r=0; r<sampleNum; r++)	// 每类动作只画sampleNum段视频的轨迹图
	{
		vector<string>::iterator itype;
		for(itype=actionType.begin(); itype<actionType.end(); itype++)
		{	
			string actionTypeStr = *itype;
			multimap<string, string>::iterator iter;
			for(iter = actionSet.begin(); iter != actionSet.end(); iter++)
			{									 // 每次只取1段指定的随机视频randActionSet[r]
				if( iter->first == actionTypeStr)// && iter->second == randActionSet[r] )
				{
					//// for UCF50 hard
					//if(	iter->second != "v_Rowing_g13_c03" && iter->second != "v_Skiing_g02_c05" 					
					//	&& iter->second != "v_Biking_g16_c04" && iter->second != "v_BreastStroke_g02_c03"
					//	&& iter->second != "v_PoleVault_g02_c05" && iter->second != "v_BenchPress_g01_c05"
					//	)
					//	continue;
						
					//// for HMDB51 hard
					//if( iter->second != "Faith_Rewarded_catch_f_cm_np1_fr_med_10" && iter->second != "Fellowship_5_shoot_bow_u_cm_np1_fr_med_13"
					//	&& iter->second != "boden__bung_gaurunde_cartwheel_f_cm_np1_le_med_0" && iter->second != "Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0" 
					//	&& iter->second != "Sexy_girl_on_the_bed_teasing_chew_u_nm_np1_fr_med_1" && iter->second != "Sit_ups_situp_f_nm_np1_ri_goo_2"
					//	&& iter->second != "Amazing_Wall_Climber_(Must_be_Seen_to_Be_Believed!)_climb_f_cm_np1_ba_bad_1"
					//	)
					//	continue;

					// for HMDB51 Test
					if( iter->second != "Amazing_Wall_Climber_(Must_be_Seen_to_Be_Believed!)_climb_f_cm_np1_ba_bad_1"//"Sexy_girl_on_the_bed_teasing_chew_u_nm_np1_fr_med_1"
						)
						continue;

					//// for Hollywood2 random
					//if(	iter->second != "actioncliptest00143" && iter->second != "actioncliptest00148"
					//	&& iter->second != "actioncliptest00112" && iter->second != "actioncliptest00878"
					//	)
					//	continue;

					ThreadParam *thrd = new ThreadParam();
					thrd->filename = iter->second;
					thrd->extname = ".avi";

					if(datasetName == "hmdb51"){
						thrd->datasetPath = datasetPath + "hmdb51_org" + spritLabel + actionTypeStr + spritLabel;
						thrd->bb_file_thrd = bbPath + iter->second + ".bb";
					}
					else if(datasetName == "UCF50"){
						thrd->datasetPath = datasetPath + actionTypeStr + spritLabel;
						thrd->bb_file_thrd = bbPath + iter->second + ".bb";
					}
					else if(datasetName == "Hollywood2"){
						thrd->datasetPath = datasetPath + "AVIClips" + spritLabel;
						thrd->bb_file_thrd = bbPath + iter->second + ".bb";
					}
					else if(datasetName == "jhmdb")
						thrd->datasetPath = datasetPath + "jhmdb_org" + spritLabel + actionTypeStr + spritLabel;
					else if(datasetName == "UCF_Sports" || datasetName == "KTH" || datasetName == "weizemann")
						thrd->datasetPath = datasetPath + actionTypeStr + spritLabel;

				
		#ifdef _WIN32 // Windows version
					SYSTEM_INFO theSystemInfo;
					::GetSystemInfo(&theSystemInfo);
					while( semaphore_image >= theSystemInfo.dwNumberOfProcessors)
						Sleep( 1000 );
			
					HANDLE hThread = CreateThread(NULL, 0, drawImageMultiThread, thrd, 0, NULL);
					if(hThread == NULL)	{
						cout << "Create Thread failed in drawImageMultiThread !" << endl;
						delete thrd;
						return -1;
					}
					_InterlockedIncrement( &semaphore_image );
		#else // Linux version
					int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
					int semaNum;
					sem_getvalue(&semaphore_image, &semaNum);
					while( semaNum >= NUM_PROCS ){
						sleep( 1 );
						sem_getvalue(&semaphore_image, &semaNum);
					}

					pthread_t pthID;
					int ret = pthread_create(&pthID, NULL, drawImageMultiThread, thrd);
					if(ret)	{
						cout << "Create Thread failed in drawImageMultiThread !" << endl;
						delete thrd;
						return -1;
					}
					sem_post( &semaphore_image );
		#endif

				}// if(iter->first)
			}// for(iter)
		}// for(itype)
	}// for(r)

	// 防止以上for循环结束后程序马上结束的情况。因为信号量semaphore可能还不为0，
	// 此时有部分线程仍在工作未释放信号量，所以应该以信号量是否为0，判断主程序是否结束。
#ifdef _WIN32 // Windows version
	while( semaphore_image )
		Sleep( 1000 );
#else // Linux version
	int semaNum;
	sem_getvalue(&semaphore_image, &semaNum);
	while( semaNum ){
		sleep( 1 );
		sem_getvalue(&semaphore_image, &semaNum);
	}
#endif	

}

void DrawManyTrack(const vector<vector<Point2f>>& trajectories, const int index, const float scale, Mat& image)
{
	for(int i=0; i<trajectories.size(); i++)
	{
		for(int j=0; j<trajectories[i].size(); j++)
		{
			Point2f point0 = trajectories[i][0];
			point0 *= scale;

			for (int j = 1; j <= index; j++) {
				Point2f point1 = trajectories[i][j];
				point1 *= scale;

				line(image, point0, point1, Scalar(0,cvFloor(255.0*(j+1.0)/float(index+1.0)),0), 2, 8, 0);
				point0 = point1;
			}
			circle(image, point0, 1, Scalar(0,0,255), -1, 8, 0);
		}
	}
	imshow( "DenseTrackStab", image);
	waitKey(0);
}

void DrawManyTrack(const vector<vector<Point2f>>& trajectories_0, const vector<vector<Point2f>>& trajectories_1, const int index, const float scale, Mat& image_0, Mat& image_1, int frame_num)
{
	for(int i=0; i<trajectories_0.size(); i++)
	{
		for(int j=0; j<trajectories_0[i].size(); j++)
		{
			Point2f point0 = trajectories_0[i][0];
			point0 *= scale;

			for (int j = 1; j <= index; j++) {
				Point2f point1 = trajectories_0[i][j];
				point1 *= scale;

				line(image_0, point0, point1, Scalar(0,cvFloor(255.0*(j+1.0)/float(index+1.0)),0), 2, 8, 0);
				point0 = point1;
			}
			circle(image_0, point0, 1, Scalar(0,0,255), -1, 8, 0);
		}
	}

	for(int i=0; i<trajectories_1.size(); i++)
	{
		for(int j=0; j<trajectories_1[i].size(); j++)
		{
			Point2f point0 = trajectories_1[i][0];
			point0 *= scale;

			for (int j = 1; j <= index; j++) {
				Point2f point1 = trajectories_1[i][j];
				point1 *= scale;

				line(image_1, point0, point1, Scalar(0,cvFloor(255.0*(j+1.0)/float(index+1.0)),0), 2, 8, 0);
				point0 = point1;
			}
			circle(image_1, point0, 1, Scalar(0,0,255), -1, 8, 0);
		}
	}

	// 多图同时显示效果
	vector<Mat> manyMat;  Mat showManyMat;
	manyMat.push_back(image_0); manyMat.push_back(image_1);
	imshowMany("showManyImg", manyMat, showManyMat);	manyMat.clear();	
}

// yang_xian521
void imshowMany(const string& _winName, const vector<Mat>& _imgs, Mat& dispImg)  
{  
    int nImg = (int)_imgs.size();        
    //Mat dispImg;   
    int size;		// 被缩小后的图像大小
    int x, y;    
    int w, h;		// w 表示一行显示多少张图，h 表示共有多少行 
    float scale;	// scale - 缩放比例 How much we have to resize the image  
    int max;  
  
    if (nImg <= 0){  
        printf("Number of arguments too small....\n");  
        return;  
    }  
    else if (nImg > 12){  
        printf("Number of arguments too large....\n");  
        return;  
    }  
      
    else if (nImg == 1){  
        w = h = 1;  
        size = 300;  
    }  
    else if (nImg == 2){  
        w = 2; h = 1;  
        size = 300;  
    }  
    else if (nImg == 3){  
        w = 3; h = 1;  
        size = 300;  
    }  
    else if (nImg == 4){  
        w = 4; h = 1;  
        size = 200;  
    }  
    else if ( nImg == 5 || nImg == 6){  
        w = 3; h = 2;  
        size = 200;  
    }  
    else if (nImg == 7 || nImg == 8){  
        w = 4; h = 2;  
        size = 200;  
    }  
    else{  
        w = 4; h = 3;  
        size = 150;  
    }  
    // 注意输入的grey和grey_warp均为单通道灰度图，如果定义为CV_8UC3则无法正常用imshow显示
    dispImg.create(Size(100 + size*w, 60 + size*h), CV_8UC3);//CV_8UC1); 
	//Mat whiteMat(dispImg.rows, dispImg.cols, CV_8UC3, Scalar::all(255));
	//dispImg = whiteMat;
  
    for (int i= 0, m=20, n=20; i<nImg; i++, m+=(20+size)){  
        x = _imgs[i].cols;  
        y = _imgs[i].rows;  
  
        max = (x > y)? x: y;  
        scale = (float) ( (float)max/size );  
  
        if (i%w==0 && m!=20){  
            m = 20;  
			//n += 20+size;			// 旧版行距=图宽=size，画出来不好看；唯一优点是可以兼容宽高不同的图像！
            n += 20+(int)(y/scale); // xzm 行距紧凑  
        }  
		
		// xzm 不缩放比例
		//scale =1;	

        Mat imgROI = dispImg(Rect(m, n, (int)(x/scale), (int)(y/scale)));  
        resize(_imgs[i], imgROI, Size((int)(x/scale), (int)(y/scale)));  
        //Mat imgROI = dispImg(Rect(m, n, x, y));  
        //resize(_imgs[i], imgROI, Size(x,y));  
    }  
  
    //namedWindow(_winName);  
    //imshow(_winName, dispImg);  waitKey(10);
}  

// 针对画图时在第0帧稠密采样的情况
void xzmDrawSample(const Mat& grey, std::vector<Point2f>& points_opt, 
					const double quality, const int min_distance, Mat& salMask)
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);
	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters_opt(width*height);
	int x_max = min_distance*width;
	int y_max = min_distance*height;

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

	vector<Point2f> points_idt(0); // 临时的idt特征，防止IDT-wofRCB采样特征点数为零。
	points_opt.clear();
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters_opt[index] > 0)
			continue;
		int x = j*min_distance+offset;
		int y = i*min_distance+offset;
		// 如果角点坐标x,y的特征值大于T才能作为IDT的强角点
		if(eig.at<float>(y,x) > threshold)
			points_idt.push_back(Point2f(float(x), float(y)));
		// 如果角点坐标x,y的特征值大于T才能作为强角点，且该点坐标属于前景掩码才进行采样
		if(eig.at<float>(y,x) > threshold && salMask.at<uchar>(y,x))
			points_opt.push_back(Point2f(float(x), float(y)));
	}

	// 当IDT-wofRCB采样特征点过少时，直接用idt特征！
	if(points_opt.size() < min_sample_ratio*points_idt.size()){
		points_opt.clear();
		points_opt = points_idt;}
}

// 针对画8个图时帧数<15时的情况：不用画图，因为轨迹特征长度定为15帧
void xzmDrawSample(const Mat& grey, std::vector<Point2f>& points_idt, std::vector<Point2f>& points_rc,
					std::vector<Point2f>& points_rcb, std::vector<Point2f>& points_mbi, 
					const double quality, const int min_distance, Mat& salMask, Mat& MBI)
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);
	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters_idt(width*height),counters_mbi(width*height),counters_rc(width*height),counters_rcb(width*height);
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
	for(int i = 0; i < points_mbi.size(); i++) {
		Point2f point = points_mbi[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_mbi[y*width+x]++;
	}
	for(int i = 0; i < points_rc.size(); i++) {
		Point2f point = points_rc[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_rc[y*width+x]++;
	}
	for(int i = 0; i < points_rcb.size(); i++) {
		Point2f point = points_rcb[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_rcb[y*width+x]++;
	}

	Mat RCmap(grey.size(), CV_8UC1, Scalar::all(255)); 
	Mat RCBmap(grey.size(), CV_8UC1, Scalar::all(255));
	double smean = 0;	
	smean = mean(salMask).val[0];	// 计算程明明GC显著图的灰度平均值
	// 当显著图全白为255(即均值为255)时，不做IDT-RCB，避免图像二值化后显著图掩码全黑为0
	if( smean != 255 ){
		// 使用归一化可以减少噪声点的干扰
		normalize(salMask, salMask, 1, 0, NORM_MINMAX);  
		// 使用线性变换将显著度掩码矩阵salMask转成8位无符号整型的灰度图像，然后才用阈值操作得到二值化图像
		convertScaleAbs(salMask, salMask, 255, 0);	
		cv::threshold(salMask, RCmap, smean, 255, THRESH_BINARY);	// 原始灰度平均值作为二值化阈值，显著图稳定性好
		//cv::threshold(salMask, RCmap, 0, 255, THRESH_BINARY|THRESH_OTSU); // 二值化阈值由OTSU算法决定，稳定性较差
		// elemKernel：定义的结构元素
		Mat elemKernel(5,5,CV_8U,Scalar(1));// MORPH_GRADIENT：对图像进行3次梯度运算，即膨胀减去腐蚀
		cv::morphologyEx(RCmap, RCBmap, MORPH_GRADIENT, elemKernel, Point(-1,-1), 3);
	}
	
	cv::threshold(MBI, MBI, 100, 255, THRESH_BINARY|THRESH_OTSU);

	// 清空上一次采样的特征点
	points_idt.clear();
	points_mbi.clear();
	points_rc.clear();
	points_rcb.clear();	

	// 遍历视频帧中未放入強角点的空网格，将新的強角点放入空网格counters[index]中
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters_idt[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold)
				points_idt.push_back(Point2f(float(x), float(y)));
		}
		if(counters_mbi[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && MBI.at<uchar>(y,x))
				points_mbi.push_back(Point2f(float(x), float(y)));
		}
		if(counters_rc[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && RCmap.at<uchar>(y,x))
				points_rc.push_back(Point2f(float(x), float(y)));
		}
		if(counters_rcb[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && RCBmap.at<uchar>(y,x))
				points_rcb.push_back(Point2f(float(x), float(y)));
		}
	}
}

// 针对画8个图时帧数>=15时的情况
void xzmDrawSample(const Mat& grey, std::vector<Point2f>& points_idt, std::vector<Point2f>& points_rc,
					std::vector<Point2f>& points_rcb, std::vector<Point2f>& points_mbi, 
					const double quality, const int min_distance, Mat& salMask, Mat& MBI,
					Mat& RCmapMulCh, Mat& RCBmapMulCh, Mat& MBIMulCh)
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);
	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters_idt(width*height),counters_mbi(width*height),counters_rc(width*height),counters_rcb(width*height);
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
	for(int i = 0; i < points_mbi.size(); i++) {
		Point2f point = points_mbi[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_mbi[y*width+x]++;
	}
	for(int i = 0; i < points_rc.size(); i++) {
		Point2f point = points_rc[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_rc[y*width+x]++;
	}
	for(int i = 0; i < points_rcb.size(); i++) {
		Point2f point = points_rcb[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_rcb[y*width+x]++;
	}

	Mat RCmap(grey.size(), CV_8UC1, Scalar::all(255)); 
	Mat RCBmap(grey.size(), CV_8UC1, Scalar::all(255));
	double smean = 0;	
	smean = mean(salMask).val[0];	// 计算程明明GC显著图的灰度平均值
	// 当显著图全白为255(即均值为255)时，不做IDT-RCB，避免图像二值化后显著图掩码全黑为0
	if( smean != 255 ){
		// 使用归一化可以减少噪声点的干扰
		normalize(salMask, salMask, 1, 0, NORM_MINMAX);  
		// 使用线性变换将显著度掩码矩阵salMask转成8位无符号整型的灰度图像，然后才用阈值操作得到二值化图像
		convertScaleAbs(salMask, salMask, 255, 0);
		cv::threshold(salMask, RCmap, smean, 255, THRESH_BINARY);	// 原始灰度平均值作为二值化阈值，显著图稳定性好
		//cv::threshold(salMask, RCmap, 0, 255, THRESH_BINARY|THRESH_OTSU); // 二值化阈值由OTSU算法决定，稳定性较差
		// elemKernel：定义的结构元素
		Mat elemKernel(5,5,CV_8U,Scalar(1));// MORPH_GRADIENT：对图像进行3次梯度运算，即膨胀减去腐蚀
		cv::morphologyEx(RCmap, RCBmap, MORPH_GRADIENT, elemKernel, Point(-1,-1), 3);
	}	

	cv::threshold(MBI, MBI, 100, 255, THRESH_BINARY|THRESH_OTSU);

	// 显示程明明显著图RC-map（将CV_8UC1单通道灰度图转为CV_8UC3的三通道彩色图）
	vector<Mat> salMaskVector;
	salMaskVector.push_back(RCmap);
	salMaskVector.push_back(RCmap);
	salMaskVector.push_back(RCmap);
	merge(salMaskVector, RCmapMulCh);
	salMaskVector.clear();

	// 显示ISal显著图RCB-map（将CV_8UC1单通道灰度图转为CV_8UC3的三通道彩色图）
	salMaskVector.push_back(RCBmap);
	salMaskVector.push_back(RCBmap);
	salMaskVector.push_back(RCBmap);
	merge(salMaskVector, RCBmapMulCh);
	salMaskVector.clear();

	// 显示MBI显著图（将CV_8UC1单通道灰度图转为CV_8UC3的三通道彩色图）
	salMaskVector.push_back(MBI);
	salMaskVector.push_back(MBI);
	salMaskVector.push_back(MBI);
	merge(salMaskVector, MBIMulCh);
	salMaskVector.clear();

	// 清空上一次采样的特征点
	points_idt.clear();
	points_mbi.clear();
	points_rc.clear();
	points_rcb.clear();
	
	// 遍历视频帧中未放入強角点的空网格，将新的強角点放入空网格counters[index]中
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters_idt[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold)
				points_idt.push_back(Point2f(float(x), float(y)));
		}
		if(counters_mbi[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && MBI.at<uchar>(y,x))
				points_mbi.push_back(Point2f(float(x), float(y)));
		}
		if(counters_rc[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && RCmap.at<uchar>(y,x))
				points_rc.push_back(Point2f(float(x), float(y)));
		}
		if(counters_rcb[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && RCBmap.at<uchar>(y,x))
				points_rcb.push_back(Point2f(float(x), float(y)));
		}
	}
}


// 针对画10个图时帧数<15时的情况：不用画图，因为轨迹特征长度定为15帧
void xzmDrawSample(const Mat& grey, std::vector<Point2f>& points_idt, std::vector<Point2f>& points_rc,
					std::vector<Point2f>& points_rcb, std::vector<Point2f>& points_mbi, 
					std::vector<Point2f>& points_opt, const double quality, const int min_distance,
					Mat& salMask, Mat& MBI, const Mat& warp_flow)	
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);
	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters_idt(width*height),counters_mbi(width*height),counters_rc(width*height),counters_rcb(width*height),counters_opt(width*height);
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
	for(int i = 0; i < points_mbi.size(); i++) {
		Point2f point = points_mbi[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_mbi[y*width+x]++;
	}
	for(int i = 0; i < points_rc.size(); i++) {
		Point2f point = points_rc[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_rc[y*width+x]++;
	}
	for(int i = 0; i < points_rcb.size(); i++) {
		Point2f point = points_rcb[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_rcb[y*width+x]++;
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

	Mat RCmap(grey.size(), CV_8UC1, Scalar::all(255)); 
	Mat RCBmap(grey.size(), CV_8UC1, Scalar::all(255));
	double smean = 0;	
	smean = mean(salMask).val[0];	// 计算程明明GC显著图的灰度平均值
	// 当显著图全白为255(即均值为255)时，不做IDT-RCB，避免图像二值化后显著图掩码全黑为0
	if( smean != 255 ){
		// 使用归一化可以减少噪声点的干扰
		normalize(salMask, salMask, 1, 0, NORM_MINMAX);  
		// 使用线性变换将显著度掩码矩阵salMask转成8位无符号整型的灰度图像，然后才用阈值操作得到二值化图像
		convertScaleAbs(salMask, salMask, 255, 0);	
		cv::threshold(salMask, RCmap, smean, 255, THRESH_BINARY);	// 原始灰度平均值作为二值化阈值，显著图稳定性好
		//cv::threshold(salMask, RCmap, 0, 255, THRESH_BINARY|THRESH_OTSU); // 二值化阈值由OTSU算法决定，稳定性较差
		// elemKernel：定义的结构元素
		Mat elemKernel(5,5,CV_8U,Scalar(1));// MORPH_GRADIENT：对图像进行3次梯度运算，即膨胀减去腐蚀
		cv::morphologyEx(RCmap, RCBmap, MORPH_GRADIENT, elemKernel, Point(-1,-1), 3);
	}
	
	cv::threshold(MBI, MBI, 100, 255, THRESH_BINARY|THRESH_OTSU);

	// 将光流场flow_warp_pyr_noHD[0]限定在显著图salMask范围内
	Mat wofRCBmap;	// 初始化：直接由xzmOptRCBmap()赋值为双通道掩码矩阵
	xzmOptRCBmap(warp_flow, RCBmap, wofRCBmap);

	// 清空上一次采样的特征点
	points_idt.clear();
	points_mbi.clear();
	points_rc.clear();
	points_rcb.clear();
	points_opt.clear();

	// 遍历视频帧中未放入強角点的空网格，将新的強角点放入空网格counters[index]中
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters_idt[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold)
				points_idt.push_back(Point2f(float(x), float(y)));
		}
		if(counters_mbi[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && MBI.at<uchar>(y,x))
				points_mbi.push_back(Point2f(float(x), float(y)));
		}
		if(counters_rc[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && RCmap.at<uchar>(y,x))
				points_rc.push_back(Point2f(float(x), float(y)));
		}
		if(counters_rcb[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && RCBmap.at<uchar>(y,x))
				points_rcb.push_back(Point2f(float(x), float(y)));
		}
		if(counters_opt[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && (wofRCBmap.at<Vec2f>(y,x)[0] && wofRCBmap.at<Vec2f>(y,x)[1]))
				points_opt.push_back(Point2f(float(x), float(y)));
		}
	}
}

// 针对画10个图时帧数>=15时的情况
void xzmDrawSample(const Mat& grey, std::vector<Point2f>& points_idt, std::vector<Point2f>& points_rc,
					std::vector<Point2f>& points_rcb, std::vector<Point2f>& points_mbi, 
					std::vector<Point2f>& points_opt, const double quality, const int min_distance,
					Mat& salMask, Mat& MBI, const Mat& warp_flow, Mat& RCmapMulCh, Mat& RCBmapMulCh, 
					Mat& MBIMulCh, Mat& optRCBMulCh)
{
	int width = grey.cols/min_distance;
	int height = grey.rows/min_distance;

	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);
	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	std::vector<int> counters_idt(width*height),counters_mbi(width*height),counters_rc(width*height),counters_rcb(width*height),counters_opt(width*height);
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
	for(int i = 0; i < points_mbi.size(); i++) {
		Point2f point = points_mbi[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_mbi[y*width+x]++;
	}
	for(int i = 0; i < points_rc.size(); i++) {
		Point2f point = points_rc[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_rc[y*width+x]++;
	}
	for(int i = 0; i < points_rcb.size(); i++) {
		Point2f point = points_rcb[i];
		int x = cvFloor(point.x);
		int y = cvFloor(point.y);
		if(x >= x_max || y >= y_max)
			continue;
		x /= min_distance;
		y /= min_distance;
		counters_rcb[y*width+x]++;
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

	Mat RCmap(grey.size(), CV_8UC1, Scalar::all(255)); 
	Mat RCBmap(grey.size(), CV_8UC1, Scalar::all(255));
	double smean = 0;	
	smean = mean(salMask).val[0];	// 计算程明明GC显著图的灰度平均值
	// 当显著图全白为255(即均值为255)时，不做IDT-RCB，避免图像二值化后显著图掩码全黑为0
	if( smean != 255 ){
		// 使用归一化可以减少噪声点的干扰
		normalize(salMask, salMask, 1, 0, NORM_MINMAX);  
		// 使用线性变换将显著度掩码矩阵salMask转成8位无符号整型的灰度图像，然后才用阈值操作得到二值化图像
		convertScaleAbs(salMask, salMask, 255, 0);
		cv::threshold(salMask, RCmap, smean, 255, THRESH_BINARY);	// 原始灰度平均值作为二值化阈值，显著图稳定性好
		//cv::threshold(salMask, RCmap, 0, 255, THRESH_BINARY|THRESH_OTSU); // 二值化阈值由OTSU算法决定，稳定性较差
		// elemKernel：定义的结构元素
		Mat elemKernel(5,5,CV_8U,Scalar(1));// MORPH_GRADIENT：对图像进行3次梯度运算，即膨胀减去腐蚀
		cv::morphologyEx(RCmap, RCBmap, MORPH_GRADIENT, elemKernel, Point(-1,-1), 3);
	}	

	cv::threshold(MBI, MBI, 100, 255, THRESH_BINARY|THRESH_OTSU);

	// 显示程明明显著图RC-map（将CV_8UC1单通道灰度图转为CV_8UC3的三通道彩色图）
	vector<Mat> salMaskVector;
	salMaskVector.push_back(RCmap);
	salMaskVector.push_back(RCmap);
	salMaskVector.push_back(RCmap);
	merge(salMaskVector, RCmapMulCh);
	salMaskVector.clear();

	// 显示ISal显著图RCB-map（将CV_8UC1单通道灰度图转为CV_8UC3的三通道彩色图）
	salMaskVector.push_back(RCBmap);
	salMaskVector.push_back(RCBmap);
	salMaskVector.push_back(RCBmap);
	merge(salMaskVector, RCBmapMulCh);
	salMaskVector.clear();

	// 显示MBI显著图（将CV_8UC1单通道灰度图转为CV_8UC3的三通道彩色图）
	salMaskVector.push_back(MBI);
	salMaskVector.push_back(MBI);
	salMaskVector.push_back(MBI);
	merge(salMaskVector, MBIMulCh);
	salMaskVector.clear();

	// 将光流场flow_warp_pyr_noHD[0]限定在显著图salMask范围内
	Mat wofRCBmap;//(grey.size(), CV_8UC1, Scalar::all(255));
	xzmOptRCBmap(warp_flow, RCBmap, wofRCBmap);
	// 必须释放optRCBMulCh，否则生成的图会显示成叠加了多个尺度的效果（以上MBIMulCh用merge()函数已有释放功能）
	optRCBMulCh.release();
	// 显示修正光流场+ISal显著图optRCB-map（将光流场转为三通道彩色图）
	motionToColor(wofRCBmap, optRCBMulCh);

	// 清空上一次采样的特征点
	points_idt.clear();
	points_mbi.clear();
	points_rc.clear();
	points_rcb.clear();
	points_opt.clear();

	// 遍历视频帧中未放入強角点的空网格，将新的強角点放入空网格counters[index]中
	int index = 0;
	int offset = min_distance/2;
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++, index++) {
		if(counters_idt[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold)
				points_idt.push_back(Point2f(float(x), float(y)));
		}
		if(counters_mbi[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && MBI.at<uchar>(y,x))
				points_mbi.push_back(Point2f(float(x), float(y)));
		}
		if(counters_rc[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && RCmap.at<uchar>(y,x))
				points_rc.push_back(Point2f(float(x), float(y)));
		}
		if(counters_rcb[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && RCBmap.at<uchar>(y,x))
				points_rcb.push_back(Point2f(float(x), float(y)));
		}
		if(counters_opt[index] == 0) {		
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;
			if(eig.at<float>(y,x) > threshold && (wofRCBmap.at<Vec2f>(y,x)[0] && wofRCBmap.at<Vec2f>(y,x)[1]))
				points_opt.push_back(Point2f(float(x), float(y)));
		}
	}	
}

// zxy09
// Color encoding of flow vectors from:  
// http://members.shaw.ca/quadibloc/other/colint.htm  
// This code is modified from:  
// http://vision.middlebury.edu/flow/data/  
void makecolorwheel(vector<Scalar> &colorwheel)  
{  
    int RY = 15, YG = 6, GC = 4, CB = 11, BM = 13, MR = 6;    
    int i;  
  
    for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255,       255*i/RY,     0));  
    for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255-255*i/YG, 255,       0));  
    for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0,         255,      255*i/GC));  
    for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0,         255-255*i/CB, 255));  
    for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255*i/BM,      0,        255));  
    for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255,       0,        255-255*i/MR));  
}  
// zxy09  
void motionToColor(Mat flow, Mat &color)  
{  
    if (color.empty())  
        color.create(flow.rows, flow.cols, CV_8UC3);  
  
    static vector<Scalar> colorwheel; //Scalar r,g,b  
    if (colorwheel.empty())  
        makecolorwheel(colorwheel);  
  
    // determine motion range:  
    float maxrad = -1;  
  
    // Find max flow to normalize fx and fy  
    for (int i= 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
            float fx = flow_at_point[0];  
            float fy = flow_at_point[1];  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
                continue;  
            float rad = sqrt(fx * fx + fy * fy);  
            maxrad = maxrad > rad ? maxrad : rad;  
        }  
    }  

    for (int i = 0; i < flow.rows; ++i)   
    {  
        for (int j = 0; j < flow.cols; ++j)   
        {  
            uchar *data = color.data + color.step[0] * i + color.step[1] * j;  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
  
            float fx = flow_at_point[0] / maxrad;  // 对x方向上的光流进行最大值归一化
            float fy = flow_at_point[1] / maxrad;  // 对y方向上的光流进行最大值归一化
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
            {  
                data[0] = data[1] = data[2] = 0;  
                continue;  
            }  
            float rad = sqrt(fx * fx + fy * fy);  
			//if (rad <= 0.4)	// xzm 20151227 测试：光流幅值越大，则光流图颜色越深（已成功）
			//	rad = 0;		// 只取幅值小于阈值的光流作为光流显著度

            float angle = atan2(-fy, -fx) / CV_PI;  
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size()-1);  
            int k0 = (int)fk;  
            int k1 = (k0 + 1) % colorwheel.size();  
            float f = fk - k0;  
            //f = 0; // uncomment to see original color wheel  
  
            for (int b = 0; b < 3; b++)   
            {  
                float col0 = colorwheel[k0][b] / 255.0;  
                float col1 = colorwheel[k1][b] / 255.0;  
                float col = (1 - f) * col0 + f * col1;  
                if (rad <= 1)  
                    col = 1 - rad * (1 - col); // increase saturation with radius  
                else  
                    col *= .75; // out of range  
                data[2 - b] = (int)(255.0 * col);  
            }  								
        }  				
    }  
	//imshow("color", color);waitKey(0);
}  
#endif /*DRAWTRACKS_H_*/