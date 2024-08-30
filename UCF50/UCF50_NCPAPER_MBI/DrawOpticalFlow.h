#ifndef _DRAWOPTICALFLOW_H_
#define _DRAWOPTICALFLOW_H_

#include "CmSaliencyGC.h"

using namespace std;
using namespace cv;

extern string spritLabel;	//使用DenseTrackStab.cpp的全局变量

// 画出8张图：原图、RCB-map、原光流场、修正光流场；带行人检测框HD的原图、空白、原光流场、带HD的修正光流场
int drawOpticalFlow(vector<string>, multimap<string, string>, 
					string, string, string, string, string, string);
// 专供idt使用，限制补idt的特征点：只取warp_flow＞最小光流幅值的点给idt用
void xzmOptIDTmap(Mat flow, Mat& wofIDTmap);
// 只取大于阈值的修正光流作为光流显著度，修正光流显著度 + 全局对比显著度 = 双通道wofRCBmap
void xzmOptRCBmap(Mat flow, Mat salMask, Mat& wofRCBmap);

// Data to protect with the interlocked functions.
#ifdef _WIN32 // Windows version
volatile LONG semaphore_opticalflow = 0;
DWORD WINAPI drawOpticalFlowProcess( LPVOID lpParameter )
#else // Linux version
sem_t semaphore_opticalflow;
static void *drawOpticalFlowProcess(void *lpParameter)
#endif
{
	ThreadParam *lp = (ThreadParam*)lpParameter;
	
	// 开始提取特征
	string feature_file = lp->featurePath + lp->filename + ".bin";
	fstream fs;
	fs.open(feature_file.c_str(), ios::out);
	if(!fs.is_open()){
		cout << "Error: Could not open file '" << feature_file << "'." << endl;
		return 0;		
	}
	cout << "extract features " << feature_file << "...." << endl;

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

	std::vector<Frame> bb_list;
	std::string str_bb_file = lp->bb_file_thrd;
	//if(bb_file){ 
	if( !str_bb_file.empty() ){	
		LoadBoundBox(str_bb_file, bb_list);
		assert(bb_list.size() == seqInfo.length);
	}
	
	// 初始化SURF特征点
	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);
	// 初始化前后两帧的特征点容器
	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;
	// 初始化前后两帧的特征"点对"
	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;	
	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);	// 0表示该Mat矩阵初始化元素为0
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);				// for optical flow

	// xzm MBI和显著度掩码金字塔
	vector<Mat> salMask_pyr(0);
	Mat prev_RCmap, prev_RCBmap;
	// xzm 初始化前后两帧未用HD的特征点容器
	std::vector<Point2f> prev_pts_flow_noHD, pts_flow_noHD;
	std::vector<Point2f> prev_pts_surf_noHD, pts_surf_noHD;
	std::vector<Point2f> prev_pts_all_noHD, pts_all_noHD;
	// xzm 初始化前后两帧未用HD的特征"点对"
	std::vector<KeyPoint> prev_kpts_surf_noHD, kpts_surf_noHD;
	Mat prev_desc_surf_noHD, desc_surf_noHD;
	Mat flow_noHD;	
	std::vector<Mat> prev_grey_pyr_noHD(0), grey_pyr_noHD(0), flow_pyr_noHD(0), flow_warp_pyr_noHD(0);	// 0表示该Mat矩阵初始化元素为0
	std::vector<Mat> prev_poly_pyr_noHD(0), poly_pyr_noHD(0), poly_warp_pyr_noHD(0);				// for optical flow
	
	// xzm 计算当前采样特征点数和运行时间
	int cnt = 0, fcnt = 0;
	double t = double( getTickCount() );

	std::vector<std::list<Track> > xyScaleTracks;	// 不同尺度下特征点跟踪轨迹的容器. xyScaleTracks[i]表示第i个尺度下的跟踪轨迹，每条轨迹上有多个被跟踪的特征点
	int init_counter = 0; // indicate when to detect new feature points
	
	while(true) {		
		Mat frame, salMask;
		int i, j, c;

		// get a new frame
		capture >> frame;
		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}
		
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

			/////////////////////////////////////////////////////////
			// xzm 专为不用HD的画图而设
			BuildPry(sizes, CV_8UC1, prev_grey_pyr_noHD);
			BuildPry(sizes, CV_8UC1, grey_pyr_noHD);
			BuildPry(sizes, CV_32FC2, flow_pyr_noHD);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr_noHD);
			BuildPry(sizes, CV_32FC(5), prev_poly_pyr_noHD);
			BuildPry(sizes, CV_32FC(5), poly_pyr_noHD);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr_noHD);
			/////////////////////////////////////////////////////////

			xyScaleTracks.resize(scale_num);

			// xzm
			CmSaliencyGC::XZM(frame, salMask);			
			InitSalMask(salMask);
			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);
						
			for(int iScale = 0; iScale < scale_num; iScale++) {// 将每个视频帧在不同尺度下的图像进行密集采样
				if(iScale == 0){			// 金字塔最底层，图像分辨率最高
					prev_grey.copyTo(prev_grey_pyr[0]);
					salMask.copyTo(salMask_pyr[0]);
				}
				else{					// 金字塔通过线性插值往上层缩小图像尺寸
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
					resize(salMask_pyr[iScale-1], salMask_pyr[iScale], salMask_pyr[iScale].size(), 0, 0, INTER_LINEAR);
				}
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
			// 使用0尺度的salMask(即RCB-map)取代行人检测器
			//detector_surf.detect(prev_grey, prev_kpts_surf, salMask);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);

			/////////////////////////////////////////////////////////
			// xzm 专为不用HD的画图而设
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr_noHD, fscales, 7, 1.5);
			detector_surf.detect(prev_grey, prev_kpts_surf_noHD, Mat());//human_mask);
			extractor_surf.compute(prev_grey, prev_kpts_surf_noHD, prev_desc_surf_noHD);
			////////////////////////////////////////////////////////

			frame_num++;
			continue;
		}

		// 开始处理第二帧
		init_counter++;
		// xzm
		CmSaliencyGC::XZM(frame, salMask);
		InitSalMask(salMask);
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		// 使用行人检测器human_mask
		//if(bb_file)
		if(!str_bb_file.empty())
			InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
		// 提取当前帧的SURF，与前一帧的SURF点一起作为可能存在的背景噪声点对，排除行人检测器human_mask中的特征点
		detector_surf.detect(grey, kpts_surf, human_mask);
		// 使用0尺度的salMask(即RCB-map)取代行人检测器
		//detector_surf.detect(prev_grey, prev_kpts_surf, salMask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

		// 一次过计算所有尺度下的光流场
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		// 从稠密光流中通过前后两帧的Strong Corner，提取可能存在的背景噪声点对，排除行人检测器human_mask中的特征点
		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);
		// 使用0尺度的salMask(即RCB-map)取代行人检测器
		//MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, salMask);
		// 将前后两帧提取出的SURF和Strong Corner特征点对合并在一起，准备计算透视变换矩阵
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

		// 通过SURF和Strong Corner的点对计算透视变换矩阵H，并用RANSAC对点对进行提纯输出到match_mask
		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
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

		///////////////////////////////////////////////////////////
		// xzm 专为不用HD的画图而设
		// 计算没有行人目标框HD的光流场！！
		detector_surf.detect(grey, kpts_surf_noHD, Mat());//human_mask);
		extractor_surf.compute(grey, kpts_surf_noHD, desc_surf_noHD);
		ComputeMatch(prev_kpts_surf_noHD, kpts_surf_noHD, prev_desc_surf_noHD, desc_surf_noHD, prev_pts_surf_noHD, pts_surf_noHD);
		my::FarnebackPolyExpPyr(grey, poly_pyr_noHD, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr_noHD, poly_pyr_noHD, flow_pyr_noHD, 10, 2);
		MatchFromFlow(prev_grey, flow_pyr_noHD[0], prev_pts_flow_noHD, pts_flow_noHD, Mat());//human_mask);
		MergeMatch(prev_pts_flow_noHD, pts_flow_noHD, prev_pts_surf_noHD, pts_surf_noHD, prev_pts_all_noHD, pts_all_noHD);
		Mat H_noHD = Mat::eye(3, 3, CV_64FC1);
		if(pts_all_noHD.size() > 50) {
			std::vector<unsigned char> match_mask_noHD;
			Mat temp_noHD = findHomography(prev_pts_all_noHD, pts_all_noHD, RANSAC, 1, match_mask_noHD);
			if(countNonZero(Mat(match_mask_noHD)) > 25)
				H_noHD = temp_noHD;
		}
		Mat H_inv_noHD = H_noHD.inv();
		Mat grey_warp_noHD = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp_noHD, H_inv_noHD); // 用透视变换矩阵H_inv对第二帧进行透视变换
		// 重新计算第二帧所有尺度下的光流
		my::FarnebackPolyExpPyr(grey_warp_noHD, poly_warp_pyr_noHD, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr_noHD, poly_warp_pyr_noHD, flow_warp_pyr_noHD, 10, 2);
		///////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////
		// xzm 专为修正光流场+RCB=wofRCBmap而设
		Mat wofRCBmap;
		salMask.copyTo(salMask_pyr[0]);
		// 将光流场flow_warp_pyr_noHD[0]限定在显著图salMask范围内
		xzmOptRCBmap(flow_warp_pyr_noHD[0], salMask, wofRCBmap);
		///////////////////////////////////////////////////////////

		// 画出前后两帧的光流场
		Mat motion2color, motion2color_warp, motion2color_noHD, motion2color_warp_noHD;	
		motionToColor(flow_pyr_noHD[0], motion2color_noHD);  
		//motionToColor(flow_warp_pyr_noHD[0], motion2color_warp_noHD);
		motionToColor(wofRCBmap, motion2color_warp_noHD);	// xzm 专为修正光流场+RCB=wofRCBmap而设
		motionToColor(flow_pyr[0], motion2color);  
		motionToColor(flow_warp_pyr[0], motion2color_warp);  

		// 多图同时显示
		vector<Mat> manyMat;	Mat maskMat;	
		// 画出第一行（未用HD）的光流场	// grey 只为填补第二行前2幅图像，使得后面两幅光流场图可以对齐第一行
		manyMat.push_back(frame); manyMat.push_back(grey); 
		// 画出未用到行人目标框no_HD的光流场！
		manyMat.push_back(motion2color_noHD); manyMat.push_back(motion2color_warp_noHD);

		// 画出第二行（用了HD）的光流场
		Mat bbs_frame;	frame.copyTo(bbs_frame);
		DrawBoundBox(bb_list[frame_num].BBs, bbs_frame);
		manyMat.push_back(bbs_frame);
		vector<Mat> salMaskVector;	salMaskVector.push_back(salMask);	salMaskVector.push_back(salMask);	salMaskVector.push_back(salMask);
		merge(salMaskVector, maskMat);salMaskVector.clear();		
		manyMat.push_back(maskMat); manyMat.push_back(motion2color); manyMat.push_back(motion2color_warp);

		Mat showManyMat;	// showManyMat 将最终大图写入JPG
		imshowMany("showManyImg", manyMat, showManyMat);		

		stringstream ss;
		ss << frame_num;
		string fullFilename = "D:\\XZM\\Data\\" + lp->filename + "_0_" + ss.str() + ".jpg";							
		vector<int> comp_paras;
		//comp_paras.push_back(CV_IMWRITE_JPEG_QUALITY); //CV_IMWRITE_PNG_COMPRESSION);
		comp_paras.push_back(95); //9);
		imwrite(fullFilename, showManyMat, comp_paras);
				
		manyMat.clear();

		// xzm
		salMask.release();

		init_counter = 0;
		grey.copyTo(prev_grey);
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}
		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		/////////////////////////////////////////////////////////
		// xzm 专为不用HD的画图而设
		for(i = 0; i < scale_num; i++) {
			grey_pyr_noHD[i].copyTo(prev_grey_pyr_noHD[i]);
			poly_pyr_noHD[i].copyTo(prev_poly_pyr_noHD[i]);
		}
		prev_kpts_surf_noHD = kpts_surf_noHD;
		desc_surf_noHD.copyTo(prev_desc_surf_noHD);
		////////////////////////////////////////////////////////

		frame_num++;

		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
		
	}// while	
		
	// 释放资源
	fs.close();
	delete lp;
#ifdef _WIN32 // Windows version
	// 结束提取特征，信号量-1
	_InterlockedDecrement( &semaphore_opticalflow );
#else // Linux version
	sem_wait( &semaphore_opticalflow );
	pthread_detach(pthread_self());
#endif

	return 0;
}

// 画出8张图：原图、RCB-map、原光流场、修正光流场；带行人检测框HD的原图、空白、原光流场、带HD的修正光流场
int drawOpticalFlow(vector<string> actionType, multimap<string, string> actionSet,
					 string datasetPath, string resultPath, string bbPath,
					 string featurePath, string processType, string datasetName)
{
	// 先算出每类动作应该采样的特征数	
	int videoNum = 0;
	map<string, unsigned int> videoSampling;
	vector<string>::iterator itype;
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{	
		videoNum = 0;
		string actionTypeStr = *itype;
		multimap<string, string>::iterator iter;
		for(iter = actionSet.begin(); iter != actionSet.end(); iter++)
			if( iter->first == actionTypeStr )
				videoNum++;
		videoSampling.insert(map<string, float>::value_type(actionTypeStr, GMM_SAMPLE_NUM/videoNum+1));
	}

	string file_loger= resultPath + "log_" + processType + "_features.xml";
	FileStorage fslog;
	double totalframes = 0;		
	int video_num = 0;

	// Current time tick
	double t = double( getTickCount() );	
	
	// 找出每类动作在训练时所用到的视频
	for(itype=actionType.begin(); itype<actionType.end(); itype++)
	{	
		unsigned int sampleNum = videoSampling[*itype];	
		string actionTypeStr = *itype;
		multimap<string, string>::iterator iter;
		for(iter = actionSet.begin(); iter != actionSet.end(); iter++)
		{
			if( iter->first == actionTypeStr )
			{
				//// for Hollywood2 random
				//if(iter->second != "actioncliptest00014" && iter->second != "actioncliptest00024"
				//	&& iter->second != "actioncliptest00034" && iter->second != "actioncliptest00044"
				//	&& iter->second != "actioncliptest00054" && iter->second != "actioncliptest00064"
				//	&& iter->second != "actioncliptest00074" && iter->second != "actioncliptest00084"
				//	)
				//	continue;

				//// for UCF50 hard
				//if(iter->second != "v_RopeClimbing_g05_c05" && iter->second != "v_Rowing_g13_c03"
				//	&& iter->second != "v_Skiing_g02_c05" && iter->second != "v_Swing_g20_c05"						
				//	&& iter->second != "v_Biking_g16_c04" && iter->second != "v_BreastStroke_g02_c03"
				//	&& iter->second != "v_Diving_g04_c01" && iter->second != "v_Fencing_g04_c01"
				//	&& iter->second != "v_PoleVault_g02_c05" && iter->second != "v_BenchPress_g01_c05"
				//	&& iter->second != "v_HighJump_g13_c02" && iter->second != "v_BaseballPitch_g02_c01"
				//	)
				//	continue;

				// for HMDB51 hard
				//if(iter->second != "Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0")//"boden__bung_gaurunde_cartwheel_f_cm_np1_le_med_0")//"Faith_Rewarded_catch_f_cm_np1_fr_med_10")//"foreman")//"Sit_ups_situp_f_nm_np1_ri_goo_2")//
				//	continue;
				if(iter->second != "Faith_Rewarded_catch_f_cm_np1_fr_med_10" && iter->second != "Basketball_Dribbling_-_Basketball_Dribbling-_Finger_Pads_dribble_f_nm_np2_le_med_5" 
					&& iter->second != "AmericanGangster_eat_u_cm_np1_fr_bad_62" && iter->second != "Fellowship_5_jump_f_nm_np1_ba_bad_12"
					&& iter->second != "TrumanShow_run_f_nm_np1_le_med_11" && iter->second != "likebeckam_run_f_nm_np1_fr_bad_12"
					&& iter->second != "Fellowship_5_shoot_bow_u_cm_np1_fr_med_13" && iter->second != "Finding_Forrester_2_sit_u_cm_np1_ri_bad_3"
					&& iter->second != "show_your_smile_-)_smile_h_nm_np1_fr_med_0" && iter->second != "IndianaJonesandTheTempleofDoom_stand_f_nm_np1_ri_med_3"
					&& iter->second != "Finding_Forrester_3_stand_u_cm_np1_fr_bad_10" && iter->second != "Sexy_girl_on_the_bed_teasing_chew_u_nm_np1_fr_med_1"
					&& iter->second != "Sit_ups_situp_f_nm_np1_ri_goo_2" && iter->second != "Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0"
					)
					continue;
				VideoCapture capture;
				string video_file;
				if(datasetName == "hmdb51")
					video_file = datasetPath + "hmdb51_org" + spritLabel + actionTypeStr + spritLabel + iter->second + ".avi";
				else if(datasetName == "jhmdb")
					video_file = datasetPath + "jhmdb_org" + spritLabel + actionTypeStr + spritLabel + iter->second + ".avi";
				else if(datasetName == "Hollywood2")
					video_file = datasetPath + "AVIClips" + spritLabel + iter->second + ".avi";
				else if(datasetName == "UCF50" || datasetName == "UCF_Sports" || 
						datasetName == "KTH" || datasetName == "weizemann")
					video_file = datasetPath + actionTypeStr + spritLabel + iter->second + ".avi";
				capture.open(video_file);
				// 检测视频是否存在
				if(!capture.isOpened()) {
					cout << "Could not find video" << video_file << endl;
					return -1;
				}
				// 获取每个视频的帧数
				int frame_counter = capture.get(CV_CAP_PROP_FRAME_COUNT);
				// 如果视频存在就先关闭，等到进入线程后再重新打开
				capture.release();

				ThreadParam *thrd = new ThreadParam();
				if(datasetName == "hmdb51"){
					thrd->datasetPath = datasetPath + "hmdb51_org" + spritLabel + actionTypeStr + spritLabel;
					if(iter->second != "foreman" && iter->second != "garden")
					thrd->bb_file_thrd = bbPath + iter->second + ".bb";
				}
				else if(datasetName == "UCF50"){
					thrd->datasetPath = datasetPath + actionTypeStr + spritLabel;
					string line = iter->second;
					int indexGroup = line.find("_g"); 
					string tmpGroup = line.substr(indexGroup + 2, 2);
					int nGroup = atoi(tmpGroup.c_str());
					int value = 0;
					// 有6类动作的组数超过了g25：JumpRope、Nunchucks、PlayingTabla、PullUps、PushUps、RockClimbingIndoor，而bb_file没有超过g25的
					if( nGroup < 26 )	
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

				thrd->featurePath = featurePath + actionTypeStr + spritLabel;
				thrd->filename = iter->second;
				thrd->extname = ".avi";
				
	#ifdef _WIN32 // Windows version
				SYSTEM_INFO theSystemInfo;
				::GetSystemInfo(&theSystemInfo);
				while( semaphore_opticalflow >= theSystemInfo.dwNumberOfProcessors)
					Sleep( 1000 );
			
				HANDLE hThread = CreateThread(NULL, 0, drawOpticalFlowProcess, thrd, 0, NULL);
				if(hThread == NULL)	{
					cout << "Create Thread failed in drawOpticalFlowProcess !" << endl;
					delete thrd;
					return -1;
				}
				_InterlockedIncrement( &semaphore_opticalflow );
	#else // Linux version
				int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
				int semaNum;
				sem_getvalue(&semaphore_opticalflow, &semaNum);
				while( semaNum >= NUM_PROCS ){
					sleep( 1 );
					sem_getvalue(&semaphore_opticalflow, &semaNum);
				}

				pthread_t pthID;
				int ret = pthread_create(&pthID, NULL, drawOpticalFlowProcess, thrd);
				if(ret)	{
					cout << "Create Thread failed in drawOpticalFlowProcess !" << endl;
					delete thrd;
					return -1;
				}
				sem_post( &semaphore_opticalflow );
	#endif
				// 统计已处理的帧数和视频数
				totalframes += frame_counter;
				video_num++;
			}// if iter
		}// for iter
	}//	for it

	// 防止以上for循环结束后程序马上结束的情况。因为信号量semaphore可能还不为0，
	// 此时有部分线程仍在工作未释放信号量，所以应该以信号量是否为0，判断主程序是否结束。
#ifdef _WIN32 // Windows version
	while( semaphore_opticalflow )
		Sleep( 1000 );
#else // Linux version
	int semaNum;
	sem_getvalue(&semaphore_opticalflow, &semaNum);
	while( semaNum ){
		sleep( 1 );
		sem_getvalue(&semaphore_opticalflow, &semaNum);
	}
#endif	
	t = ( (double)getTickCount() - t ) / getTickFrequency();
	fslog.open(file_loger, FileStorage::WRITE);
	fslog << "drawOpticalFLow" << "{" ;
	fslog << "time_hours" << t/3600;
	fslog << "video_num" << video_num;
	fslog << "avgFPS" << totalframes / t;
	fslog << "}";

	// 释放内存
	fslog.release();
	return 0;
}

// 专供idt使用，限制补idt的特征点：只取warp_flow＞最小光流幅值的点给idt用
void xzmOptIDTmap(Mat flow, Mat& wofIDTmap) 
{    
	flow.copyTo(wofIDTmap);

    // 找出光流场x,y方向上的最大幅值，准备进行归一化  
    float maxrad = 0;    
    for (int i= 0; i < flow.rows; ++i)   
    for (int j = 0; j < flow.cols; ++j) {  
        Vec2f flow_at_point = flow.at<Vec2f>(i,j);  
        float fx = flow_at_point[0];  
        float fy = flow_at_point[1];  
        if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH))  
            continue;  
        float rad = sqrt(fx * fx + fy * fy); 		
        maxrad = maxrad > rad ? maxrad : rad;
    }  

	for(int i = 0; i < flow.rows; i++)
	for(int j = 0; j < flow.cols; j++) {
		Vec2f flow_at_point = flow.at<Vec2f>(i,j);  
        float fx = flow_at_point[0] / maxrad;  // 对x方向上的光流进行最大值归一化
        float fy = flow_at_point[1] / maxrad;  // 对y方向上的光流进行最大值归一化
        float rad = sqrt(fx * fx + fy * fy); 
		// 只取warp_flow＞最小光流幅值的点给idt用，或补idt时用
		if(rad <= min_warp_flow){
			wofIDTmap.at<Vec2f>(i,j)[0] = 0;	
			wofIDTmap.at<Vec2f>(i,j)[1] = 0;	
		}
	}
}

// 只取大于阈值的修正光流作为光流显著度，修正光流显著度 + 全局对比显著度 = 双通道wofRCBmap
void xzmOptRCBmap(Mat flow, Mat RCBmap, Mat& wofRCBmap) 
{    
	flow.copyTo(wofRCBmap);

    // 找出光流场x,y方向上的最大幅值，准备进行归一化  
    float maxrad = 0;    
    for (int i= 0; i < flow.rows; ++i)   
    for (int j = 0; j < flow.cols; ++j) {  
        Vec2f flow_at_point = flow.at<Vec2f>(i,j);  
        float fx = flow_at_point[0];  
        float fy = flow_at_point[1];  
        if ((fabs(fx) > UNKNOWN_FLOW_THRESH) || (fabs(fy) > UNKNOWN_FLOW_THRESH))  
            continue;  
        float rad = sqrt(fx * fx + fy * fy); 		
        maxrad = maxrad > rad ? maxrad : rad;
    }  

	for(int i = 0; i < flow.rows; i++)
	for(int j = 0; j < flow.cols; j++) {
		Vec2f flow_at_point = flow.at<Vec2f>(i,j);  
        float fx = flow_at_point[0] / maxrad;  // 对x方向上的光流进行最大值归一化
        float fy = flow_at_point[1] / maxrad;  // 对y方向上的光流进行最大值归一化
        float rad = sqrt(fx * fx + fy * fy);  
		// 限定全局对比显著度和修正光流场显著度的取值范围（InitSalMask()约定前景为255，背景为0）
		//if(RCBmap.at<uchar>(i,j)==0 || rad<=min_warp_flow){
		//if(rad <= min_warp_flow){
		if(RCBmap.at<uchar>(i,j)==0){
			wofRCBmap.at<Vec2f>(i,j)[0] = 0;	
			wofRCBmap.at<Vec2f>(i,j)[1] = 0;	
		}
	}

	// 仅当wofRCBmap全为空时，直接用修正光流场作为显著图，避免调用motionToColor()画图时出错！
	Mat rcbflows[2];
	split(wofRCBmap, rcbflows);
	// 其实任何光流场都不可能在x,y方向上全为0（即rcbflows都不为0），因此以下2行根本没有用武之地
	if(countNonZero(rcbflows[0])==0 && countNonZero(rcbflows[1])==0)
		flow.copyTo(wofRCBmap);
}

#endif /*DRAWOPTICALFLOW_H_*/