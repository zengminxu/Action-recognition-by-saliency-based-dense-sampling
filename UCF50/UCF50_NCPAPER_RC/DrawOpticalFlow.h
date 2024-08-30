#ifndef _DRAWOPTICALFLOW_H_
#define _DRAWOPTICALFLOW_H_

#include "CmSaliencyGC.h"

using namespace std;
using namespace cv;

extern string spritLabel;	//ʹ��DenseTrackStab.cpp��ȫ�ֱ���

// ����8��ͼ��ԭͼ��RCB-map��ԭ�������������������������˼���HD��ԭͼ���հס�ԭ����������HD������������
int drawOpticalFlow(vector<string>, multimap<string, string>, 
					string, string, string, string, string, string);
// ר��idtʹ�ã����Ʋ�idt�������㣺ֻȡwarp_flow����С������ֵ�ĵ��idt��
void xzmOptIDTmap(Mat flow, Mat& wofIDTmap);
// ֻȡ������ֵ������������Ϊ���������ȣ��������������� + ȫ�ֶԱ������� = ˫ͨ��wofRCBmap
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
	
	// ��ʼ��ȡ����
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

	InitTrackInfo(&trackInfo, track_length, init_gap);				// ��ʼ���켣��Ϣ��ÿ�����������15֡�����1֡��������ȡ������
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);// ʱ�վ�����������ֵ������HOF��9bins�⣬��������8binsֱ��ͼ
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
	
	// ��ʼ��SURF������
	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);
	// ��ʼ��ǰ����֡������������
	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;
	// ��ʼ��ǰ����֡������"���"
	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;	
	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);	// 0��ʾ��Mat�����ʼ��Ԫ��Ϊ0
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);				// for optical flow

	// xzm MBI�����������������
	vector<Mat> salMask_pyr(0);
	Mat prev_RCmap, prev_RCBmap;
	// xzm ��ʼ��ǰ����֡δ��HD������������
	std::vector<Point2f> prev_pts_flow_noHD, pts_flow_noHD;
	std::vector<Point2f> prev_pts_surf_noHD, pts_surf_noHD;
	std::vector<Point2f> prev_pts_all_noHD, pts_all_noHD;
	// xzm ��ʼ��ǰ����֡δ��HD������"���"
	std::vector<KeyPoint> prev_kpts_surf_noHD, kpts_surf_noHD;
	Mat prev_desc_surf_noHD, desc_surf_noHD;
	Mat flow_noHD;	
	std::vector<Mat> prev_grey_pyr_noHD(0), grey_pyr_noHD(0), flow_pyr_noHD(0), flow_warp_pyr_noHD(0);	// 0��ʾ��Mat�����ʼ��Ԫ��Ϊ0
	std::vector<Mat> prev_poly_pyr_noHD(0), poly_pyr_noHD(0), poly_warp_pyr_noHD(0);				// for optical flow
	
	// xzm ���㵱ǰ������������������ʱ��
	int cnt = 0, fcnt = 0;
	double t = double( getTickCount() );

	std::vector<std::list<Track> > xyScaleTracks;	// ��ͬ�߶�����������ٹ켣������. xyScaleTracks[i]��ʾ��i���߶��µĸ��ٹ켣��ÿ���켣���ж�������ٵ�������
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
		
		if(frame_num == start_frame) {	// �ȶԵ�һ֡���г�ʼ������
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			// ��ʼ��ͼ���������ע��InitPryӦ��ΪInitPyr��׼ȷ����
			InitPry(frame, fscales, sizes);	// ����Ƶ֡���ֶ����εĳ߶Ⱥʹ�С�ֱ����fscales��sizes

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
			// xzm רΪ����HD�Ļ�ͼ����
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
						
			for(int iScale = 0; iScale < scale_num; iScale++) {// ��ÿ����Ƶ֡�ڲ�ͬ�߶��µ�ͼ������ܼ�����
				if(iScale == 0){			// ��������ײ㣬ͼ��ֱ������
					prev_grey.copyTo(prev_grey_pyr[0]);
					salMask.copyTo(salMask_pyr[0]);
				}
				else{					// ������ͨ�����Բ�ֵ���ϲ���Сͼ��ߴ�
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
					resize(salMask_pyr[iScale-1], salMask_pyr[iScale], salMask_pyr[iScale].size(), 0, 0, INTER_LINEAR);
				}
			}

			// compute polynomial expansion
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			// ʹ�����˼����human_mask
			human_mask = Mat::ones(frame.size(), CV_8UC1);
			//if(bb_file)
			if(!str_bb_file.empty())
				InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
			// ��ȡ��ǰ֡��SURF������һ֡��SURF��һ����Ϊ���ܴ��ڵı���������ԣ��ų����˼����human_mask�е�������
			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
			// ʹ��0�߶ȵ�salMask(��RCB-map)ȡ�����˼����
			//detector_surf.detect(prev_grey, prev_kpts_surf, salMask);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);

			/////////////////////////////////////////////////////////
			// xzm רΪ����HD�Ļ�ͼ����
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr_noHD, fscales, 7, 1.5);
			detector_surf.detect(prev_grey, prev_kpts_surf_noHD, Mat());//human_mask);
			extractor_surf.compute(prev_grey, prev_kpts_surf_noHD, prev_desc_surf_noHD);
			////////////////////////////////////////////////////////

			frame_num++;
			continue;
		}

		// ��ʼ����ڶ�֡
		init_counter++;
		// xzm
		CmSaliencyGC::XZM(frame, salMask);
		InitSalMask(salMask);
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		// ʹ�����˼����human_mask
		//if(bb_file)
		if(!str_bb_file.empty())
			InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
		// ��ȡ��ǰ֡��SURF����ǰһ֡��SURF��һ����Ϊ���ܴ��ڵı���������ԣ��ų����˼����human_mask�е�������
		detector_surf.detect(grey, kpts_surf, human_mask);
		// ʹ��0�߶ȵ�salMask(��RCB-map)ȡ�����˼����
		//detector_surf.detect(prev_grey, prev_kpts_surf, salMask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

		// һ�ι��������г߶��µĹ�����
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		// �ӳ��ܹ�����ͨ��ǰ����֡��Strong Corner����ȡ���ܴ��ڵı���������ԣ��ų����˼����human_mask�е�������
		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);
		// ʹ��0�߶ȵ�salMask(��RCB-map)ȡ�����˼����
		//MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, salMask);
		// ��ǰ����֡��ȡ����SURF��Strong Corner������Ժϲ���һ��׼������͸�ӱ任����
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

		// ͨ��SURF��Strong Corner�ĵ�Լ���͸�ӱ任����H������RANSAC�Ե�Խ����ᴿ�����match_mask
		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if(countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}

		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // ��͸�ӱ任����H_inv�Եڶ�֡����͸�ӱ任
		
		// ���¼���ڶ�֡���г߶��µĹ���
		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);

		///////////////////////////////////////////////////////////
		// xzm רΪ����HD�Ļ�ͼ����
		// ����û������Ŀ���HD�Ĺ���������
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
		MyWarpPerspective(prev_grey, grey, grey_warp_noHD, H_inv_noHD); // ��͸�ӱ任����H_inv�Եڶ�֡����͸�ӱ任
		// ���¼���ڶ�֡���г߶��µĹ���
		my::FarnebackPolyExpPyr(grey_warp_noHD, poly_warp_pyr_noHD, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr_noHD, poly_warp_pyr_noHD, flow_warp_pyr_noHD, 10, 2);
		///////////////////////////////////////////////////////////

		///////////////////////////////////////////////////////////
		// xzm רΪ����������+RCB=wofRCBmap����
		Mat wofRCBmap;
		salMask.copyTo(salMask_pyr[0]);
		// ��������flow_warp_pyr_noHD[0]�޶�������ͼsalMask��Χ��
		xzmOptRCBmap(flow_warp_pyr_noHD[0], salMask, wofRCBmap);
		///////////////////////////////////////////////////////////

		// ����ǰ����֡�Ĺ�����
		Mat motion2color, motion2color_warp, motion2color_noHD, motion2color_warp_noHD;	
		motionToColor(flow_pyr_noHD[0], motion2color_noHD);  
		//motionToColor(flow_warp_pyr_noHD[0], motion2color_warp_noHD);
		motionToColor(wofRCBmap, motion2color_warp_noHD);	// xzm רΪ����������+RCB=wofRCBmap����
		motionToColor(flow_pyr[0], motion2color);  
		motionToColor(flow_warp_pyr[0], motion2color_warp);  

		// ��ͼͬʱ��ʾ
		vector<Mat> manyMat;	Mat maskMat;	
		// ������һ�У�δ��HD���Ĺ�����	// grey ֻΪ��ڶ���ǰ2��ͼ��ʹ�ú�������������ͼ���Զ����һ��
		manyMat.push_back(frame); manyMat.push_back(grey); 
		// ����δ�õ�����Ŀ���no_HD�Ĺ�������
		manyMat.push_back(motion2color_noHD); manyMat.push_back(motion2color_warp_noHD);

		// �����ڶ��У�����HD���Ĺ�����
		Mat bbs_frame;	frame.copyTo(bbs_frame);
		DrawBoundBox(bb_list[frame_num].BBs, bbs_frame);
		manyMat.push_back(bbs_frame);
		vector<Mat> salMaskVector;	salMaskVector.push_back(salMask);	salMaskVector.push_back(salMask);	salMaskVector.push_back(salMask);
		merge(salMaskVector, maskMat);salMaskVector.clear();		
		manyMat.push_back(maskMat); manyMat.push_back(motion2color); manyMat.push_back(motion2color_warp);

		Mat showManyMat;	// showManyMat �����մ�ͼд��JPG
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
		// xzm רΪ����HD�Ļ�ͼ����
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
		
	// �ͷ���Դ
	fs.close();
	delete lp;
#ifdef _WIN32 // Windows version
	// ������ȡ�������ź���-1
	_InterlockedDecrement( &semaphore_opticalflow );
#else // Linux version
	sem_wait( &semaphore_opticalflow );
	pthread_detach(pthread_self());
#endif

	return 0;
}

// ����8��ͼ��ԭͼ��RCB-map��ԭ�������������������������˼���HD��ԭͼ���հס�ԭ����������HD������������
int drawOpticalFlow(vector<string> actionType, multimap<string, string> actionSet,
					 string datasetPath, string resultPath, string bbPath,
					 string featurePath, string processType, string datasetName)
{
	// �����ÿ�ද��Ӧ�ò�����������	
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
	
	// �ҳ�ÿ�ද����ѵ��ʱ���õ�����Ƶ
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
				// �����Ƶ�Ƿ����
				if(!capture.isOpened()) {
					cout << "Could not find video" << video_file << endl;
					return -1;
				}
				// ��ȡÿ����Ƶ��֡��
				int frame_counter = capture.get(CV_CAP_PROP_FRAME_COUNT);
				// �����Ƶ���ھ��ȹرգ��ȵ������̺߳������´�
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
					// ��6�ද��������������g25��JumpRope��Nunchucks��PlayingTabla��PullUps��PushUps��RockClimbingIndoor����bb_fileû�г���g25��
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
				// ͳ���Ѵ����֡������Ƶ��
				totalframes += frame_counter;
				video_num++;
			}// if iter
		}// for iter
	}//	for it

	// ��ֹ����forѭ��������������Ͻ������������Ϊ�ź���semaphore���ܻ���Ϊ0��
	// ��ʱ�в����߳����ڹ���δ�ͷ��ź���������Ӧ�����ź����Ƿ�Ϊ0���ж��������Ƿ������
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

	// �ͷ��ڴ�
	fslog.release();
	return 0;
}

// ר��idtʹ�ã����Ʋ�idt�������㣺ֻȡwarp_flow����С������ֵ�ĵ��idt��
void xzmOptIDTmap(Mat flow, Mat& wofIDTmap) 
{    
	flow.copyTo(wofIDTmap);

    // �ҳ�������x,y�����ϵ�����ֵ��׼�����й�һ��  
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
        float fx = flow_at_point[0] / maxrad;  // ��x�����ϵĹ����������ֵ��һ��
        float fy = flow_at_point[1] / maxrad;  // ��y�����ϵĹ����������ֵ��һ��
        float rad = sqrt(fx * fx + fy * fy); 
		// ֻȡwarp_flow����С������ֵ�ĵ��idt�ã���idtʱ��
		if(rad <= min_warp_flow){
			wofIDTmap.at<Vec2f>(i,j)[0] = 0;	
			wofIDTmap.at<Vec2f>(i,j)[1] = 0;	
		}
	}
}

// ֻȡ������ֵ������������Ϊ���������ȣ��������������� + ȫ�ֶԱ������� = ˫ͨ��wofRCBmap
void xzmOptRCBmap(Mat flow, Mat RCBmap, Mat& wofRCBmap) 
{    
	flow.copyTo(wofRCBmap);

    // �ҳ�������x,y�����ϵ�����ֵ��׼�����й�һ��  
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
        float fx = flow_at_point[0] / maxrad;  // ��x�����ϵĹ����������ֵ��һ��
        float fy = flow_at_point[1] / maxrad;  // ��y�����ϵĹ����������ֵ��һ��
        float rad = sqrt(fx * fx + fy * fy);  
		// �޶�ȫ�ֶԱ������Ⱥ����������������ȵ�ȡֵ��Χ��InitSalMask()Լ��ǰ��Ϊ255������Ϊ0��
		//if(RCBmap.at<uchar>(i,j)==0 || rad<=min_warp_flow){
		//if(rad <= min_warp_flow){
		if(RCBmap.at<uchar>(i,j)==0){
			wofRCBmap.at<Vec2f>(i,j)[0] = 0;	
			wofRCBmap.at<Vec2f>(i,j)[1] = 0;	
		}
	}

	// ����wofRCBmapȫΪ��ʱ��ֱ����������������Ϊ����ͼ���������motionToColor()��ͼʱ����
	Mat rcbflows[2];
	split(wofRCBmap, rcbflows);
	// ��ʵ�κι���������������x,y������ȫΪ0����rcbflows����Ϊ0�����������2�и���û������֮��
	if(countNonZero(rcbflows[0])==0 && countNonZero(rcbflows[1])==0)
		flow.copyTo(wofRCBmap);
}

#endif /*DRAWOPTICALFLOW_H_*/