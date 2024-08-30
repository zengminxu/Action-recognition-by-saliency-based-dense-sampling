#define UNKNOWN_FLOW_THRESH 1e9 

#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"
#include "ActionAnalysis.h"
#include "FisherVector.h"
#include "BrowseDir.h"
#include "CmSaliencyGC.h"
#include "DrawTracks.h"
#include "DrawOpticalFlow.h"

using namespace std;
using namespace cv;

#ifdef _WIN32 // Windows version
	string spritLabel="\\";
#else // Linux version
	string spritLabel="/";
#endif

int predictHY2();int predictHMDB51();int predictJHMDB();
int predictUCF50();int predictUCFSports();int predictKTH();int predictWZM();
int makeHY2splits(vector<string>&, string, string, string);
int makeHMDBsplits(vector<string>, string, string, string);
int makeUCF50splits(vector<string>, multimap<string, string>, 
					string, string, string);
int makeKTHsplits(vector<string>, string, string, string);
int makeWZMsplits(vector<string>, string, string, string);
int getVideoFeatures(vector<string>, multimap<string, string>, 
					 string, string, string, string, string, string);
void testDataSize();

//int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

// Data to protect with the interlocked functions.
#ifdef _WIN32 // Windows version
volatile LONG semaphore = 0;
DWORD WINAPI featuresMultiThreadProcess( LPVOID lpParameter )
#else // Linux version
sem_t semaphore;
static void *featuresMultiThreadProcess(void *lpParameter)
#endif
{
	ThreadParam *lp = (ThreadParam*)lpParameter;
	
	// �����ȡ������ < ָ������������������IDT��ʽ�������������IDT-RCB��ʽ�޶�����������
	unsigned int sampleNum = lp->sampleNum;

	// ��ʼ��ȡ����
	string file_name = lp->featurePath + lp->filename + ".bin";
	FILE *fx = fopen(file_name.c_str(), "wb");
	if( fx == NULL ){
		std::cout << "Error: Could not open file '" << file_name << "'." << std::endl;
		return 0;
	}
	cout << "extract features " << file_name << "...." << endl;

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

	//if(show_track == 1)
	//	namedWindow("DenseTrackStab", 0);

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
	//vector<Mat> MBI_pyr(0);
	vector<Mat> salMask_pyr(0);
	//vector< vector<Point2f> > mvs_pyr(0); // ��ʱͣ�ã��ݲ���Ҫmvs�ռ������	
	//vector<Point2f> mvs(0);
	vector<Point2f> prev_mvf(0);
	queue<Mat> frame_queue;		//����ǰ10֡ͼ��
	MVFInfo mvfInfo;
	InitMVFInfo(&mvfInfo, mvf_length, init_gap);
	std::list<MVF> xyScaleMVFs;	
	Mat prev_RCmap, prev_RCBmap;

	// xzm ���㵱ǰ������������������ʱ��
	int cnt = 0, fcnt = 0;
	float fmin_var = 0 , fmax_var = 0 , fmean_x = 0, fmean_y =0;
	double t = double( getTickCount() );

	// ��ͬ�߶�����������ٹ켣������. xyScaleTracks[i]��ʾ��i���߶��µĸ��ٹ켣��ÿ���켣���ж�������ٵ�������
	std::vector<std::list<Track> > xyScaleTracks, xyScaleTracks_opt;	
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
			
			xyScaleTracks.resize(scale_num);
			xyScaleTracks_opt.resize(scale_num);

			// xzm
			CmSaliencyGC::XZM(frame, salMask);			
			InitSalMask(salMask);
			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);
						
			for(int iScale = 0; iScale < scale_num; iScale++) {// ��ÿ����Ƶ֡�ڲ�ͬ�߶��µ�ͼ������ܼ�����
				if(iScale == 0){		// ��������ײ㣬ͼ��ֱ������
					prev_grey.copyTo(prev_grey_pyr[0]);
					salMask.copyTo(salMask_pyr[0]);
				}
				else{					// ������ͨ�����Բ�ֵ���ϲ���Сͼ��ߴ�
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
					resize(salMask_pyr[iScale-1], salMask_pyr[iScale], salMask_pyr[iScale].size(), 0, 0, INTER_LINEAR);
				}
				
				// �ڵ�ǰ֡�����г߶Ƚ����������ܼ�����
				std::vector<Point2f> points_idt(0), points_opt(0);
				// ���������СW=5�����������ܼ�����. min_distance�������ĵ�W.
				wofDenseSample(prev_grey_pyr[iScale], points_idt, points_opt, quality, min_distance, salMask_pyr[iScale]);

				// ����������(�Ĺ켣).��Ϊÿ�������㶼��һ���켣������Ҳ������Ϊ�Ǳ���켣
				std::list<Track>& tracks = xyScaleTracks[iScale];
				std::list<Track>& tracks_opt = xyScaleTracks_opt[iScale];
				for(i = 0; i < points_idt.size(); i++)
					tracks.push_back(Track(points_idt[i], trackInfo, hogInfo, hofInfo, mbhInfo));
				//// ��IDT-wofRCB����������Ϊ��ʱ��ֱ����idt������
				//if(points_opt.size() == 0){
				//	points_opt.clear();
				//	points_opt = points_idt;}
				for(i = 0; i < points_opt.size(); i++)
					tracks_opt.push_back(Track(points_opt[i], trackInfo, hogInfo, hofInfo, mbhInfo));
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
		
		// ��ÿ���߶�iScale�¸��������㡣������Ƶperson01.avi����Ϊ4���߶�
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
			
			// �ֱ����HOG/HOF/MBH����������ֱ��ͼ��ע��HOG�õ����ݶ���Ϣ����HOF/MBH�õ��Ǿ���͸�ӱ任���¼���Ĺ�����
			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo);

			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);	
			MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

			// IDT��ʹIDT�켣�����Џ��ǵ�������У����ָ���������15֡���ɼ��µď��ǵ㡣
			list<Track>& tracks = xyScaleTracks[iScale];
			for (list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;							// iTrackΪ��ǰ�߶���ĳ�������ٵ�������ָ�룬index��ʾ�õ㱻���ٵ���index֡
				Point2f prev_point = iTrack->point[index];			// ��ָһ�������ٵ�������
				int x = min<int>(max<int>(cvRound(prev_point.x), 0), width-1);// ���������ǰ�����������(x,y)����ֹԽ��
				int y = min<int>(max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];	// Pt+1 = (xt+1, yt+1) = Pt + �������������Ĺ�ʽ1��
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				// �����¼����Ĺ�������Ϊ���ܵı�������λ�ƴ���iTrack�������index��������
				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];	
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
				
				// ��ȡ�켣��������������
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);	// ����ǰһ�����������꣬��λʱ�վ���������ROI�������rect
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				
				// ��������point��ǰ���ٵ�֡��index+1���ٽ��õ����iTrack->point[index]
				iTrack->addPoint(point);

				// ����켣������֡���ﵽL = 15���򽫸ù켣0-15����16֡���������켣����
				if(iTrack->index >= trackInfo.length) {
					// ֱ��ɾ���켣������Ҫ�ж�if(IsValid(trajectory) && IsCameraMotion(displacement))
					// ��Ϊֻ����IDT����Ƶ֡�ǿ���������15֡�ڲ������µ������㣬�����ظ�������
					iTrack = tracks.erase(iTrack);// ɾ����ǰ������
					continue;
				}
				++iTrack;
			}// for iTrack

			// SCI������������+����ͼ����optRCB-map����ÿ���߶�iScale�¸���������
			std::list<Track>& tracks_opt = xyScaleTracks_opt[iScale];		
			for (std::list<Track>::iterator iTrack = tracks_opt.begin(); iTrack != tracks_opt.end();) {
				int index = iTrack->index;							// iTrackΪ��ǰ�߶���ĳ�������ٵ�������ָ�룬index��ʾ�õ㱻���ٵ���index֡
				Point2f prev_point = iTrack->point[index];			// ��ָһ�������ٵ�������
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);// ���������ǰ�����������(x,y)����ֹԽ��
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];	// Pt+1 = (xt+1, yt+1) = Pt + �������������Ĺ�ʽ1��
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1]; // С��point.yָ�У�point.xָ��
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks_opt.erase(iTrack);
					continue;
				}

				// �����¼����Ĺ�������Ϊ���ܵı�������λ�ƴ���iTrack�������index��������
				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];	
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
				
				// ��ȡ�켣��������������
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);	// ����ǰһ�����������꣬��λʱ�վ���������ROI�������rect
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				
				// ��������point��ǰ���ٵ�֡��index+1���ٽ��õ����iTrack->point[index]
				iTrack->addPoint(point);
				
				// ����켣������֡���ﵽL = 15���򽫸ù켣0-15����16֡���������켣����
				if(iTrack->index >= trackInfo.length) {
					std::vector<Point2f> trajectory(trackInfo.length+1);// ��length���������ٵ�֡��L=15��������Ĺ켣����length��ͬ
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];
				
					std::vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];
	
					// ��ʼ��Ϊ0. ע�⣺��lengthΪ�켣���ȣ���ʼ��Ϊ0������������IsValid����
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);										
					// ����Ƿ����켣������켣�е���λ�Ƶ�ŷ�Ͼ��붼С��1�����ؾ���Ϊ������˶������Ĺ켣������������嶯���켣
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {	
						// �ڴ�ʹ�õ���438άIDT��������WangHeng�õ���436άIDT��						
						// output the trajectory
						float frameNum = (float)frame_num;
						fwrite(&frameNum,sizeof(float),1,fx);
						fwrite(&mean_x,sizeof(float),1,fx);
						fwrite(&mean_y,sizeof(float),1,fx);
						fwrite(&var_x,sizeof(float),1,fx);
						fwrite(&var_y,sizeof(float),1,fx);
						fwrite(&length,sizeof(float),1,fx);
						fwrite(&fscales[iScale],sizeof(float),1,fx);
						// for spatio-temporal pyramid
						float st_x_pos = std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999);
						float st_y_pos = std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999);
						float st_t_pos = std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999);
						fwrite(&st_x_pos,sizeof(float),1,fx);
						fwrite(&st_y_pos,sizeof(float),1,fx);
						fwrite(&st_t_pos,sizeof(float),1,fx);

						//// ����IsValid()������trajectory[i].x���������������������������������trajectory[i].x������������������ƫ�����������ǹ켣����ͼ���е�׼ȷ���� 
						//for (int i = 0; i < trackInfo.length; ++i){
						//	fwrite(&trajectory[i].x,sizeof(float),1,fx);
						//	fwrite(&trajectory[i].y,sizeof(float),1,fx);
						//}
						// ���һ���켣����16������ͼ���е�׼ȷ���ꡣ output the trajectory Points of srcImage!
						for (int i = 0; i < trackInfo.length + 1; ++i){
							float x_pos = iTrack->point[i].x/fscales[iScale];
							float y_pos = iTrack->point[i].y/fscales[iScale];
							fwrite(&x_pos,sizeof(float),1,fx);
							fwrite(&y_pos,sizeof(float),1,fx);
						}								
					
						PrintDesc(iTrack->hog, hogInfo, trackInfo, fx);
						PrintDesc(iTrack->hof, hofInfo, trackInfo, fx);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo, fx);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo, fx);		
					}

					iTrack = tracks_opt.erase(iTrack);// ɾ����ǰ������
					continue;
				}// if iTrack
				++iTrack;
			}// for iTrack

			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			// ÿ���init_gap֡�����¼���µ������㡣
			if(init_counter != trackInfo.gap)
				continue;

			// �Ѹ��ݹ������������ԭ�������������͸��ٵ�֡����ѹ��һ���µ�����������points��
			std::vector<Point2f> points_idt(0), points_opt(0);
			for(list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points_idt.push_back(iTrack->point[iTrack->index]);
			for(std::list<Track>::iterator iTrack = tracks_opt.begin(); iTrack != tracks_opt.end(); iTrack++)
				points_opt.push_back(iTrack->point[iTrack->index]);
			
			// ����������ȫ�ֶԱ�����ͼ�������г߶�û��������������м���µ�������points
			wofDenseSample(grey_pyr[iScale], flow_warp_pyr[iScale], points_idt, points_opt, quality, min_distance, salMask_pyr[iScale]);

			// ���µ���������Ϣѹ��켣tracks_optĩβ��֮ǰ�ѱ�����������֡(<15֡)����������Ȼ��Ч������һ��ѭ���м�����λÿ�������������͹켣��ֱ���õ��������ٳ���15֡���Ƴ�.
			for(i = 0; i < points_idt.size(); i++)
				tracks.push_back(Track(points_idt[i], trackInfo, hogInfo, hofInfo, mbhInfo));			
			//// ��IDT-wofRCB�������������ʱ��ֱ����idt������
			//if(points_opt.size() < min_sample_ratio*points_idt.size()){
			//	points_opt.clear();
			//	points_opt = points_idt;}
			for(i = 0; i < points_opt.size(); i++)
				tracks_opt.push_back(Track(points_opt[i], trackInfo, hogInfo, hofInfo, mbhInfo));

		}// for iScale

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
		
		frame_num++;

		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}

	}// while
	
   	if( show_track == 1 )
		destroyWindow("DenseTrackStab");
		
	// �ͷ���Դ
	fclose(fx);	
	delete lp;// delete lp;���ܷŵ�pthread_detach(pthread_self());֮��

#ifdef _WIN32 // Windows version
	// ������ȡ�������ź���-1
	_InterlockedDecrement( &semaphore );
#else // Linux version
	sem_wait( &semaphore );
	pthread_detach(pthread_self());
#endif

	return 0;
}

void testDataSize()
{	
	unsigned int test = -1;
	cout << "unsigned int test = -1" << ", test��" << test << endl;  
    cout << "int: \t\t" << "��ռ�ֽ�����" << sizeof(int);  
    cout << "\t���ֵ��" << (numeric_limits<int>::max)(); 
    cout << "\t��Сֵ��" << (numeric_limits<int>::min)() << endl;  
    cout << "unsigned int: " << "��ռ�ֽ�����" << sizeof(unsigned int);  
    cout << "\t���ֵ��" << (numeric_limits<unsigned int>::max)();  
    cout << "\t��Сֵ��" << (numeric_limits<unsigned int>::min)() << endl;  
    cout << "long: \t\t" << "��ռ�ֽ�����" << sizeof(long);  
    cout << "\t���ֵ��" << (numeric_limits<long>::max)();  
    cout << "\t��Сֵ��" << (numeric_limits<long>::min)() << endl;  
    cout << "unsigned long:" << "��ռ�ֽ�����" << sizeof(unsigned long);  
    cout << "\t���ֵ��" << (numeric_limits<unsigned long>::max)();  
    cout << "\t��Сֵ��" << (numeric_limits<unsigned long>::min)() << endl;  
    cout << "double: \t" << "��ռ�ֽ�����" << sizeof(double);  
    cout << "\t���ֵ��" << (numeric_limits<double>::max)();  
    cout << "\t��Сֵ��" << (numeric_limits<double>::min)() << endl;
    cout << "float: \t" << "��ռ�ֽ�����" << sizeof(float);  
    cout << "\t���ֵ��" << (numeric_limits<float>::max)();  
    cout << "\t��Сֵ��" << (numeric_limits<float>::min)() << endl;  
    cout << "RAND_MAX: \t" << "��ռ�ֽ�����" << RAND_MAX << endl;  
    cout << "string: \t" << "��ռ�ֽ�����" << sizeof(string) << endl;
	cout << "cv::string: \t" << "��ռ�ֽ�����" << sizeof(cv::string) << endl;
	cout << "string: \t" << "��ռ�ֽ�����" << sizeof(string) << endl;
}

int predictHY2()
{
	ActionAnalysis action;
	BrowseDir browseDir;
	multimap<string, string> trainSet, testSet, actionMap;
	vector<string> actionType;

#ifdef _WIN32 // Windows version
	string datasetPath="D:\\XZM\\ActionVideoDataBase\\Hollywood2\\";
	string resultPath="D:\\XZM\\ActionVideoDataBase\\Hollywood2\\";
	string bbPath="D:\\XZM\\ActionVideoDataBase\\bb_file\\Hollywood2\\";
#else // Linux version
	string datasetPath="/dawnfs/users/xuzengmin/datasets/Hollywood2/";
	string resultPath="/dawnfs/users/xuzengmin/code/HY2_NCPAPER_rawRCB/";
	string bbPath="/dawnfs/users/xuzengmin/datasets/bb_file/Hollywood2/";
#endif

	// read actionType from Hollywood2 dir
	string datasetName = "Hollywood2";
	string datasetPathStr = datasetPath + "AVIClips" + spritLabel;
	makeHY2splits(actionType, datasetName, datasetPath, resultPath);
	browseDir.browseDir(datasetPathStr.c_str(), resultPath, "*.*", actionType, actionMap);	
	string featurePath = resultPath + "features" + spritLabel;
	string vidcodePath = resultPath + "quantization" + spritLabel;
	string splitFile = resultPath + datasetName + "_splits_1.txt";

	actionMap.clear();
	// ����Hollywood2��һ����Ƶ����ͬʱ����2������������actionMap����������1707
	if ( 0 != action.readFileSets(splitFile, actionMap, "train") )
		return -1;
	if ( 0 != action.readFileSets(splitFile, actionMap, "test") )
		return -1;

	//// ����no_HD��HD�Ĺ��������������������Աȣ���֤SURF/STRONG CORNER�ɵ��͸�ӱ任�������Ҫ��
	//if ( 0 != drawOpticalFlow(actionType, actionMap, datasetPath, resultPath, bbPath, 
	//						   featurePath, "allvideo", datasetName) )
	//	return -1;

	//// ���ɹ켣������ͬʱ��ֱ�ӻ���10��ͼ��ԭͼ��MBI��RC-map��RCB-map��wofRCB-map
	////									   IDT��IDT-MBI��IDT-RC��IDT-RCB��IDT-wofRCB
	//if ( 0 != myDrawImage(actionType, actionMap, datasetPath, resultPath, bbPath, 
	//					featurePath, "drawImage", datasetName) )
	//	return -1; 

	// һ�ι���ȡ������Ƶ������
	if ( 0 != getVideoFeatures(actionType, actionMap, datasetPath, resultPath, bbPath, 
							   featurePath, "allvideo", datasetName) )
		return -1;
/**
	//if ( 0 != action.getBOFcodebookHY2(actionType, actionMap, resultPath))
	if ( 0 != action.getMultiCodebook(actionType, actionMap, resultPath) )
		return -1;

	if ( 0 != action.getVideoCode(actionType, trainSet, vidcodePath, resultPath, "train") )
		return -1;
	if ( 0 != action.getVideoCode(actionType, testSet,  vidcodePath, resultPath, "test" ) )
		return -1;
**/
	FisherVector fisher;
	string descriptorType = "d-Fusion";//"d-Fusion";//"r3-Fusion";
	string manifoldType = "pca";//"raw";//"LLE";
	int gmmNum = 256, splitNum = 1;

	//if ( 0 != fisher.trainGMM(actionType, actionMap, gmmNum, resultPath, descriptorType, manifoldType) )
	//	return -1;

	//if ( 0 != fisher.getFisherVector(actionType, actionMap, gmmNum, datasetPathStr, vidcodePath, 
	//								 resultPath, "allvideo", descriptorType, manifoldType) )
	//	return -1;

	//if ( 0 != action.trainAndtest(actionType, splitNum, gmmNum, datasetName, resultPath, descriptorType) )
	//	return -1;

	return 0;
}

int predictHMDB51()
{
	ActionAnalysis action;
	BrowseDir browseDir;
	multimap<string, string> trainSet, testSet, actionMap;
	vector<string> actionType;

#ifdef _WIN32 // Windows version
	string datasetPath="D:\\XZM\\ActionVideoDataBase\\HMDB51\\";
	string resultPath="D:\\XZM\\ActionVideoDataBase\\HMDB51\\";
	string bbPath="D:\\XZM\\ActionVideoDataBase\\bb_file\\HMDB51\\";
#else // Linux version
	string datasetPath="/dawnfs/users/xuzengmin/datasets/HMDB51/";
	string resultPath="/dawnfs/users/xuzengmin/code/HMDB51_NCPAPER_rawRCB/";
	string bbPath="/dawnfs/users/xuzengmin/datasets/bb_file/HMDB51/";
#endif
	
	// read actionType from hmdb51 dir
	string datasetName = "hmdb51";
	string datasetPathStr = datasetPath + "hmdb51_org" + spritLabel;
	browseDir.browseDir(datasetPathStr.c_str(), resultPath, "*.*", actionType, actionMap);	
	makeHMDBsplits(actionType, datasetName, datasetPath, resultPath);
	string featurePath = resultPath + "features" + spritLabel;
	string vidcodePath = resultPath + "quantization" + spritLabel;

	//if ( 0 != myDrawTrack(actionType, actionMap, datasetPath, resultPath, bbPath, 
	//					featurePath, "drawTrack", datasetName) )
	//	return -1;

	//// ���ɹ켣������ͬʱ��ֱ�ӻ���8��ͼ��ԭͼ��RC-map��RCB-map��MBI��IDT��IDT-RC��IDT-RCB��IDT-MBI
	//if ( 0 != myDrawImage(actionType, actionMap, datasetPath, resultPath, bbPath, 
	//					featurePath, "drawImage", datasetName) )
	//	return -1;

	//// ���ɹ켣������ͬʱ��ֱ�ӻ���10��ͼ��ԭͼ��MBI��RC-map��RCB-map��wofRCB-map
	////									   IDT��IDT-MBI��IDT-RC��IDT-RCB��IDT-wofRCB
	//if ( 0 != myDrawImage(actionType, actionMap, datasetPath, resultPath, bbPath, 
	//					featurePath, "drawImage", datasetName) )
	//	return -1;

	//// ����no_HD��HD�Ĺ��������������������Աȣ���֤SURF/STRONG CORNER�ɵ��͸�ӱ任�������Ҫ��
	//if ( 0 != drawOpticalFlow(actionType, actionMap, datasetPath, resultPath, bbPath, 
	//						   featurePath, "allvideo", datasetName) )
	//	return -1;

	// һ�ι���ȡ������Ƶ������
	if ( 0 != getVideoFeatures(actionType, actionMap, datasetPath, resultPath, bbPath, 
							   featurePath, "allvideo", datasetName) )
		return -1;
/**
	if ( 0 != action.getMultiCodebook(actionType, actionMap, resultPath) )
		return -1;

	if ( 0 != action.readFileSets(splitFile, trainSet, "train") )
		return -1;
	if ( 0 != action.readFileSets(splitFile, testSet,  "test") )
		return -1;

	if ( 0 != action.getVideoCode(actionType, trainSet, vidcodePath, resultPath, "train") )
		return -1;
	if ( 0 != action.getVideoCode(actionType, testSet,  vidcodePath, resultPath, "test" ) )
		return -1;

	if ( 0 != action.trainAndtest(actionType, trainSet, testSet, resultPath) )
		return -1;
**/
	FisherVector fisher;
	string descriptorType = "d-Fusion";//"r4-Fusion";//"r3-Fusion";
	string manifoldType = "pca";//"raw";//"LLE";
	int gmmNum = 64, splitNum = 3;

	//if ( 0 != fisher.trainGMM(actionType, actionMap, gmmNum, resultPath, descriptorType, manifoldType) )
	//	return -1;

	//if ( 0 != fisher.getFisherVector(actionType, actionMap, gmmNum, datasetPathStr, vidcodePath, 
	//								 resultPath, "allvideo", descriptorType, manifoldType) )
	//	return -1;
	//
	//if ( 0 != action.trainAndtest(actionType, splitNum, gmmNum, datasetName, resultPath, descriptorType) )
	//	return -1;

	return 0;
}

int predictJHMDB()
{
	ActionAnalysis action;
	BrowseDir browseDir;
	multimap<string, string> trainSet, testSet, actionMap;
	vector<string> actionType;

#ifdef _WIN32 // Windows version
	string datasetPath="D:\\XZM\\ActionVideoDataBase\\JHMDB\\";
	string resultPath="D:\\XZM\\ActionVideoDataBase\\JHMDB\\";
#else // Linux version
	string datasetPath="/dawnfs/users/xuzengmin/datasets/JHMDB/";
	string resultPath="/dawnfs/users/xuzengmin/code/JHMDB_wof2m0.2RCB0.001/";
#endif

	// read actionType from jhmdb dir
	string datasetName = "jhmdb";
	string datasetPathStr = datasetPath + "jhmdb_org" + spritLabel;
	browseDir.browseDir(datasetPathStr.c_str(), resultPath, "*.*", actionType, actionMap);	
	makeHMDBsplits(actionType, datasetName, datasetPath, resultPath);
	string featurePath = resultPath + "features" + spritLabel;
	string vidcodePath = resultPath + "quantization" + spritLabel;
	string bbPath = "";
	
	// һ�ι���ȡ������Ƶ������
	if ( 0 != getVideoFeatures(actionType, actionMap, datasetPath, resultPath, bbPath, 
							   featurePath, "allvideo", datasetName) )
		return -1;
/*
	if ( 0 != action.getMultiCodebook(actionType, actionMap, resultPath) )
		return -1;

	if ( 0 != action.getVideoCode(actionType, trainSet, vidcodePath, resultPath, "allvideo") )
		return -1;

	string descriptorType = "d-Fusion";
	if ( 0 != action.trainAndtest(actionType, splitNum, gmmNum, datasetName, resultPath, descriptorType) )
		return -1;
*/
	FisherVector fisher;
	string descriptorType = "d-Fusion";//"r4-Fusion";//"r3-Fusion";
	string manifoldType = "pca";//"raw";//"LLE";
	int gmmNum = 64, splitNum = 3;

	//if ( 0 != fisher.trainGMM(actionType, actionMap, gmmNum, resultPath, descriptorType, manifoldType) )
	//	return -1;

	//if ( 0 != fisher.getFisherVector(actionType, actionMap, gmmNum, datasetPathStr, vidcodePath, 
	//								 resultPath, "allvideo", descriptorType, manifoldType) )
	//	return -1;
	//
	//if ( 0 != action.trainAndtest(actionType, splitNum, gmmNum, datasetName, resultPath, descriptorType) )
	//	return -1;

	return 0;
}

int predictUCF50()
{
	ActionAnalysis action;
	BrowseDir browseDir;
	multimap<string, string> trainSet, testSet, actionMap;
	vector<string> actionType;

#ifdef _WIN32 // Windows version
	string datasetPath="D:\\XZM\\ActionVideoDataBase\\UCF50\\";
	string resultPath="D:\\XZM\\ActionVideoDataBase\\UCF50\\";
	string bbPath="D:\\XZM\\ActionVideoDataBase\\bb_file\\UCF50\\";
#else // Linux version
	string datasetPath="/dawnfs/users/xuzengmin/datasets/UCF50/";
	string resultPath="/dawnfs/users/xuzengmin/code/UCF50_NCPAPER_rawRCB/";
	string bbPath="/dawnfs/users/xuzengmin/datasets/bb_file/UCF50/";
#endif

	// read actionType from UCF_Sports dir
	string datasetName = "UCF50";
	browseDir.browseDir(datasetPath.c_str(), resultPath, "*.*", actionType, actionMap);	
	makeUCF50splits(actionType, actionMap, datasetName, datasetPath, resultPath);
	string featurePath = resultPath + "features" + spritLabel;
	string vidcodePath = resultPath + "quantization" + spritLabel;

	//// ���ɹ켣������ͬʱ��ֱ�ӻ���8��ͼ��ԭͼ��RC-map��RCB-map��MBI��IDT��IDT-RC��IDT-RCB��IDT-MBI
	//if ( 0 != myDrawImage(actionType, actionMap, datasetPath, resultPath, bbPath, 
	//					featurePath, "drawImage", datasetName) )
	//	return -1;

	//// ���ɹ켣������ͬʱ��ֱ�ӻ���10��ͼ��ԭͼ��MBI��RC-map��RCB-map��wofRCB-map
	////									   IDT��IDT-MBI��IDT-RC��IDT-RCB��IDT-wofRCB
	//if ( 0 != myDrawImage(actionType, actionMap, datasetPath, resultPath, bbPath, 
	//					featurePath, "drawImage", datasetName) )
	//	return -1;

	//// ����no_HD��HD�Ĺ��������������������Աȣ���֤SURF/STRONG CORNER�ɵ��͸�ӱ任�������Ҫ��
	//if ( 0 != drawOpticalFlow(actionType, actionMap, datasetPath, resultPath, bbPath, 
	//						   featurePath, "allvideo", datasetName) )
	//	return -1;

	// һ�ι���ȡ������Ƶ������
	if ( 0 != getVideoFeatures(actionType, actionMap, datasetPath, resultPath, bbPath, 
							   featurePath, "allvideo", datasetName) )
		return -1;

	FisherVector fisher;
	string descriptorType = "d-Fusion";
	string manifoldType = "pca";//"raw";//"LLE";
	int gmmNum = 64, splitNum = 25;
/*
	if ( 0 != fisher.trainGMM(actionType, actionMap, gmmNum, resultPath, descriptorType, manifoldType) )
		return -1;

	if ( 0 != fisher.getFisherVector(actionType, actionMap, gmmNum, datasetPath, vidcodePath, 
									 resultPath, "allvideo", descriptorType, manifoldType) )
		return -1;

	if ( 0 != action.trainAndtest(actionType, splitNum, gmmNum, datasetName, resultPath, descriptorType) )
		return -1;
*/
	return 0;
}

int predictUCFSports()
{
	ActionAnalysis action;
	BrowseDir browseDir;
	multimap<string, string> trainSet, testSet, actionMap;
	vector<string> actionType;

#ifdef _WIN32 // Windows version
	string datasetPath="D:\\XZM\\ActionVideoDataBase\\UCF_Sports\\";
	string resultPath="D:\\XZM\\ActionVideoDataBase\\UCF_Sports\\";
#else // Linux version
	string datasetPath="/dawnfs/users/xuzengmin/datasets/UCF_Sports/";
	string resultPath="/dawnfs/users/xuzengmin/code/UCF_Sports/";
#endif

	// read actionType from UCF_Sports dir
	string datasetName = "UCF_Sports";
	browseDir.browseDir(datasetPath.c_str(), resultPath, "*.*", actionType, actionMap);	

	string featurePath = resultPath + "features" + spritLabel;
	string vidcodePath = resultPath + "quantization" + spritLabel;
	string bbPath = "";

	// һ�ι���ȡ������Ƶ������
	if ( 0 != getVideoFeatures(actionType, actionMap, datasetPath, resultPath, bbPath, 
							   featurePath, "allvideo", datasetName) )
		return -1;

	return 0;
}

int predictKTH()
{
	ActionAnalysis action;
	BrowseDir browseDir;
	multimap<string, string> trainSet, testSet, actionMap;
	vector<string> actionType;

#ifdef _WIN32 // Windows version
	string datasetPath="D:\\XZM\\ActionVideoDataBase\\KTH\\";
	string resultPath="D:\\XZM\\ActionVideoDataBase\\KTH\\";
#else // Linux version
	string datasetPath="/dawnfs/users/xuzengmin/datasets/KTH/";
	string resultPath="/dawnfs/users/xuzengmin/code/KTH/";
#endif

	// read actionType from KTH dir
	string datasetName = "KTH";
	browseDir.browseDir(datasetPath.c_str(), resultPath, "*.*", actionType, actionMap);	
	makeKTHsplits(actionType, datasetName, datasetPath, resultPath);
	string featurePath = resultPath + "features" + spritLabel;
	string vidcodePath = resultPath + "quantization" + spritLabel;
	string bbPath = "";
/*
	// һ�ι���ȡ������Ƶ������
	if ( 0 != getVideoFeatures(actionType, actionMap, datasetPath, resultPath, bbPath, 
							   featurePath, "allvideo", datasetName) )
		return -1;

	if ( 0 != action.getMultiCodebook(actionType, actionMap, resultPath) )
		return -1;

	if ( 0 != action.readFileSets(splitFile, trainSet, "train") )
		return -1;
	if ( 0 != action.readFileSets(splitFile, testSet,  "test") )
		return -1;

	if ( 0 != action.getVideoCode(actionType, trainSet, vidcodePath, resultPath, "train") )
		return -1;
	if ( 0 != action.getVideoCode(actionType, testSet,  vidcodePath, resultPath, "test" ) )
		return -1;

	if ( 0 != action.trainAndtest(actionType, trainSet, testSet, resultPath) )
		return -1;
*/
	FisherVector fisher;
	string descriptorType = "d-Fusion";
	string manifoldType = "pca";//"raw";//"LLE";
	int gmmNum = 64, splitNum = 1;

	if ( 0 != fisher.trainGMM(actionType, actionMap, gmmNum, resultPath, descriptorType, manifoldType) )
		return -1;

	if ( 0 != fisher.getFisherVector(actionType, actionMap, gmmNum, datasetPath, vidcodePath, 
									 resultPath, "allvideo", descriptorType, manifoldType) )
		return -1;

	if ( 0 != action.trainAndtest(actionType, splitNum, gmmNum, datasetName, resultPath, descriptorType) )
		return -1;

	return 0;
}

int predictWZM()
{
	ActionAnalysis action;
	BrowseDir browseDir;
	multimap<string, string> trainSet, testSet, actionMap;
	vector<string> actionType;

#ifdef _WIN32 // Windows version
	string datasetPath="D:\\XZM\\ActionVideoDataBase\\weizemann\\";
	string resultPath="D:\\XZM\\ActionVideoDataBase\\weizemann\\";
#else // Linux version
	string datasetPath="/dawnfs/users/xuzengmin/datasets/weizemann/";
	string resultPath="/dawnfs/users/xuzengmin/code/weizemann/";
#endif

	// read actionType from weizemann dir
	string datasetName = "weizemann";
	browseDir.browseDir(datasetPath.c_str(), resultPath, "*.*", actionType, actionMap);	
	makeWZMsplits(actionType, datasetName, datasetPath, resultPath);
	string featurePath = resultPath + "features" + spritLabel;
	string vidcodePath = resultPath + "quantization" + spritLabel;
	string bbPath = "";

	// һ�ι���ȡ������Ƶ������
	if ( 0 != getVideoFeatures(actionType, actionMap, datasetPath, resultPath, bbPath, 
							   featurePath, "allvideo", datasetName) )
		return -1;

	//if ( 0 != action.getBOFcodebookHMDB(actionType, actionMap, resultPath) )
	if ( 0 != action.getMultiCodebook(actionType, actionMap, resultPath) )
		return -1;
	
	if ( 0 != action.getVideoCode(actionType, trainSet, vidcodePath, resultPath, "allvideo") )
		return -1;

	int	splitNum = 9, gmmNum = 64;
	string descriptorType = "d-Fusion";
	if ( 0 != action.trainAndtest(actionType, splitNum, gmmNum, datasetName, resultPath, descriptorType) )
		return -1;
	
	return 0;
}

int makeHMDBsplits(vector<string> actionType, string datasetName, string datasetPath, string resultPath)
{
	// HMDB���ݼ���3��ѵ�����Ͳ��Լ�
	for(int idx=1; idx<=3; idx++)
	{
		stringstream ss;
		ss << idx;
		string split_file = resultPath + datasetName + "_splits_" + ss.str() + ".txt";
		fstream split;
		split.open(split_file.c_str(), ios::out);
		if(!split.is_open()){
			cout << "Error: Could not open file '" << split_file << "'." << endl;
			return -1;		
		}

		for(vector<string>::iterator itype=actionType.begin(); itype<actionType.end(); itype++)
		{
			string readfile;
			readfile = datasetPath + datasetName + "_splits" + spritLabel + *itype + "_test_split" + ss.str() + ".txt";
			ifstream configFile(readfile.c_str());
			string line;

			while(getline(configFile, line)) {
				istringstream iss(line);
				string videoname;
				if (!(iss >> videoname))
					continue;		
				int value;
				if (!(iss >> value))
					continue;
				// HMDB�Դ���split�ļ���Ĭ��δ��Ϊ0��trainΪ1��testΪ2�����ڴ�ͳһ��train��Ϊ1��test��Ϊ-1
				if (value == 2)
					value = -1;
				split << *itype << "\t"<< videoname << "\t" << value << endl;
			}//while			
			configFile.close();
		}//for itype
		
		ss.str("");
		split.close();
	}//for idx

	return 0;
}

int makeUCF50splits(vector<string> actionType, multimap<string, string> actionSet,
					string datasetName, string datasetPath, string resultPath)
{
	// UCF50���ݼ���25��ѵ�����Ͳ��Լ�
	for(int idx=1; idx<=25; idx++)
	{
		stringstream ss;
		ss << idx;
		string split_file = resultPath + datasetName + "_splits_" + ss.str() + ".txt";
		fstream split;
		split.open(split_file.c_str(), ios::out);
		if(!split.is_open()){
			cout << "Error: Could not open file '" << split_file << "'." << endl;
			return -1;		
		}

		for(vector<string>::iterator itype=actionType.begin(); itype<actionType.end(); itype++)
		{
			string actionTypeStr = *itype;
			multimap<string, string>::iterator iter;
			for(iter = actionSet.begin(); iter != actionSet.end(); iter++)
			{
				if( iter->first == actionTypeStr )
				{	// �ҵ�����ؼ���_g���ٺ���2λ�ҵ�2λ���ּ�Ϊ���
					string line = iter->second;
					int indexGroup = line.find("_g"); 
					string tmpGroup = line.substr(indexGroup + 2, 2);
					int nGroup = atoi(tmpGroup.c_str());
					int value = 0;
					if( nGroup == idx )
						value = -1;// ��ʾΪ������Ƶ
					else
						value = 1; // ��ʾΪѵ����Ƶ
					split << iter->first << "\t"<< iter->second << ".avi " << value << endl;		
				}//if iter->first
			}//for iter
		}//for itype
		
		ss.str("");
		split.close();
	}//for splitNum

	return 0;
}

int makeHY2splits(vector<string>& actionType, string datasetName, string datasetPath, string resultPath)
{
	string action[12] = {"AnswerPhone","DriveCar","Eat","FightPerson","GetOutCar","HandShake","HugPerson","Kiss","Run","SitDown","SitUp","StandUp"};
	string split_file = resultPath + datasetName + "_splits_1.txt";
	fstream split;
	split.open(split_file.c_str(), ios::out);
	if(!split.is_open()){
		cout << "Error: Could not open file '" << split_file << "'." << endl;
		return -1;		
	}

	for(int i=0; i<12; i++){
		actionType.push_back(action[i]);
		string readfile;
		readfile = datasetPath + "ClipSets" + spritLabel + action[i] + "_train.txt";
		ifstream trainFile(readfile.c_str());
		string line;
		while(getline(trainFile, line)) {
			istringstream iss(line);
			string videoname;
			if (!(iss >> videoname))
				continue;
			int value;
			if (!(iss >> value))
				continue;
			if (value == 1){
				value = 1;
				split << action[i] << "\t" << videoname << "\t" << value << endl;
			}
		}// while
		trainFile.close();

		// read ClipSets from actionType_test.txt
		readfile = datasetPath + "ClipSets" + spritLabel + action[i] + "_test.txt";
		ifstream testFile(readfile.c_str());
		while(getline(testFile, line)) {
			istringstream iss(line);
			string videoname;
			if (!(iss >> videoname))
				continue;
			int value;
			if (!(iss >> value))
				continue;
			if (value == 1){
				value = -1;
				split << action[i] << "\t" << videoname << "\t" << value << endl;
			}
		}// while
		testFile.close();
	}// for itype

	split.close();
	return 0;
}

int makeKTHsplits(vector<string> actionType, string datasetName, string datasetPath, string resultPath)
{
	string action[6] = {"running","walking","jogging","boxing","handclapping","handwaving"};
	string person[25]= {"01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25"};
	string d[4] = {"1","2","3","4"};
	string split_file = resultPath + datasetName + "_splits_1.txt";
	fstream split;
	split.open(split_file.c_str(), ios::out);
	if(!split.is_open()){
		cout << "Error: Could not open file '" << split_file << "'." << endl;
		return -1;		
	}
	
	int value = 0;
	for(int idx = 0; idx < 6; idx++)
	for(int jdx = 0; jdx < 25; jdx++)
	for(int kdx = 0; kdx < 4; kdx++)		
	{	
		if( (idx==3 && jdx==0 && kdx==3) || (idx==4 && jdx==12 && kdx==2) )
			value = 0;		// ��ʾδʹ����Ƶ
		else if( jdx==1 || jdx==2 || jdx==4 || jdx==5 || jdx==6 || jdx==7 || jdx==8 || jdx==9 || jdx==21)
			value = -1;		// ��ʾΪ������Ƶ
		else
			value = 1;		// ��ʾΪѵ����Ƶ
		string videoname = "person" + person[jdx]  + "_" + action[idx] + "_d" + d[kdx] + "_uncomp.avi";
		split << action[idx] << "\t" << videoname << "\t" << value << endl;
	}

	split.close();
	return 0;
}

int makeWZMsplits(vector<string> actionType, string datasetName, string datasetPath, string resultPath)
{
	string action[10]={"bend","jack","jump","pjump","run","side","skip","walk","wave1","wave2"};
	string person[9]={"daria","denis","eli","ido","ira","lena","lyova","moshe","shahar"};	
	
	int value = 0;
	for(int spl = 1; spl <= 9; spl++)	// Leave one person out
	{	
		stringstream ss;
		ss << spl;
		string split_file = resultPath + datasetName + "_splits_" + ss.str() + ".txt";
		fstream split;
		split.open(split_file.c_str(), ios::out);
		if(!split.is_open()){
			cout << "Error: Could not open file '" << split_file << "'." << endl;
			return -1;		
		}

		for(int pIdx = 0; pIdx < 9; pIdx++)
		for(int aIdx = 0; aIdx < 10; aIdx++)		
		{	
			if( pIdx == spl )
				value = -1;		// ��ʾΪ������Ƶ
			else
				value = 1;		// ��ʾΪѵ����Ƶ
		
			if( pIdx == 5 && (aIdx==4||aIdx==6||aIdx==7) ){		// lena��run��walk��skip����3����Ƶ
				string videoname1 = person[pIdx] + "_" + action[aIdx] + "1.avi";
				string videoname2 = person[pIdx] + "_" + action[aIdx] + "2.avi";
				split << action[aIdx] << "\t" << videoname1 << "\t" << value << endl;
				split << action[aIdx] << "\t" << videoname2 << "\t" << value << endl;
			}
			else{
				string videoname = person[pIdx] + "_" + action[aIdx] + ".avi";
				split << action[aIdx] << "\t" << videoname << "\t" << value << endl;
			}
		}
		
		ss.str("");
		split.close();
	}

	return 0;
}

int getVideoFeatures(vector<string> actionType, multimap<string, string> actionSet,
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
				//// HMDB51: for NC PAPER Trajectories/clip and fps
				//if (iter->second != "(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0" && iter->second != "AgentCodyBanks_climb_stairs_f_cm_np1_fr_med_9" 
				//	&& iter->second != "20060723sfjfffightcoldplay_fall_floor_f_cm_np1_fr_med_0" && iter->second != "1899_Hamlet__the_Fencing_Scene_with_Laertes_fencing_f_cm_np2_le_med_0" 
				//	&& iter->second != "100_Second_Handstand_handstand_f_nm_np1_ba_bad_0" && iter->second != "50_FIRST_DATES_kick_f_cm_np1_ba_med_19" 
				//	&& iter->second != "50_FIRST_DATES_punch_f_nm_np1_ri_med_15" && iter->second != "8YearOldHitsaHomeRunBaseball_swing_baseball_f_cm_np1_fr_bad_0"
				//	&& iter->second != "American_History_X_talk_h_nm_np1_fr_goo_20"	&& iter->second != "310ToYuma_turn_h_nm_np1_ba_med_1")
				//	continue;

				//// UCF50: for NC PAPER Trajectories/clip and fps
				//if (iter->second != "v_Basketball_g01_c02" && iter->second != "v_Billards_g01_c02" 
				//	&& iter->second != "v_Diving_g01_c02" && iter->second != "v_Fencing_g01_c02" 
				//	&& iter->second != "v_HorseRace_g01_c02" && iter->second != "v_JavelinThrow_g01_c02" 
				//	&& iter->second != "v_JumpRope_g01_c02" && iter->second != "v_MilitaryParade_g01_c02"
				//	&& iter->second != "v_PizzaTossing_g01_c02"	&& iter->second != "v_PlayingViolin_g01_c02")
				//	continue;

				// Hollywood2: for NC PAPER Trajectories/clip and fps
				if (iter->second != "actioncliptest00096" && iter->second != "actioncliptest00120" 
					&& iter->second != "actioncliptest00166" && iter->second != "actioncliptest00206" 
					&& iter->second != "actioncliptest00225" && iter->second != "actioncliptrain00142" 
					&& iter->second != "actioncliptrain00240" && iter->second != "actioncliptrain00336"
					&& iter->second != "actioncliptrain00497"	&& iter->second != "actioncliptrain00519")
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
				thrd->sampleNum = 0; // ʹ��wofRCB�󣬲�����Ҫ����ֵ�������ȡ����������ָ���Ĳ���������IDT���������IDT-RCB���Ʋ�����
				
	#ifdef _WIN32 // Windows version
				SYSTEM_INFO theSystemInfo;
				::GetSystemInfo(&theSystemInfo);
				while( semaphore >= theSystemInfo.dwNumberOfProcessors)
					Sleep( 1000 );
			
				HANDLE hThread = CreateThread(NULL, 0, featuresMultiThreadProcess, thrd, 0, NULL);
				if(hThread == NULL)	{
					cout << "Create Thread failed in featuresMultiThreadProcess !" << endl;
					delete thrd;
					return -1;
				}
				_InterlockedIncrement( &semaphore );
	#else // Linux version
				int NUM_PROCS = sysconf(_SC_NPROCESSORS_CONF);
				int semaNum;
				sem_getvalue(&semaphore, &semaNum);
				while( semaNum >= NUM_PROCS ){
					sleep( 1 );
					sem_getvalue(&semaphore, &semaNum);
				}

				pthread_t pthID;
				int ret = pthread_create(&pthID, NULL, featuresMultiThreadProcess, thrd);
				if(ret)	{
					cout << "Create Thread failed in featuresMultiThreadProcess !" << endl;
					delete thrd;
					return -1;
				}
				sem_post( &semaphore );
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
	while( semaphore )
		Sleep( 1000 );
#else // Linux version
	int semaNum;
	sem_getvalue(&semaphore, &semaNum);
	while( semaNum ){
		sleep( 1 );
		sem_getvalue(&semaphore, &semaNum);
	}
#endif	

	Mat des, tmpMat;	
	ActionAnalysis action;
	// Current time tick
	double t1 = double( getTickCount() );	

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
				//// HMDB51: for NC PAPER Trajectories/clip and fps
				//if (iter->second != "(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0" && iter->second != "AgentCodyBanks_climb_stairs_f_cm_np1_fr_med_9" 
				//	&& iter->second != "20060723sfjfffightcoldplay_fall_floor_f_cm_np1_fr_med_0" && iter->second != "1899_Hamlet__the_Fencing_Scene_with_Laertes_fencing_f_cm_np2_le_med_0" 
				//	&& iter->second != "100_Second_Handstand_handstand_f_nm_np1_ba_bad_0" && iter->second != "50_FIRST_DATES_kick_f_cm_np1_ba_med_19" 
				//	&& iter->second != "50_FIRST_DATES_punch_f_nm_np1_ri_med_15" && iter->second != "8YearOldHitsaHomeRunBaseball_swing_baseball_f_cm_np1_fr_bad_0"
				//	&& iter->second != "American_History_X_talk_h_nm_np1_fr_goo_20"	&& iter->second != "310ToYuma_turn_h_nm_np1_ba_med_1")
				//	continue;

				//// UCF50: for NC PAPER Trajectories/clip and fps
				//if (iter->second != "v_Basketball_g01_c02" && iter->second != "v_Billards_g01_c02" 
				//	&& iter->second != "v_Diving_g01_c02" && iter->second != "v_Fencing_g01_c02" 
				//	&& iter->second != "v_HorseRace_g01_c02" && iter->second != "v_JavelinThrow_g01_c02" 
				//	&& iter->second != "v_JumpRope_g01_c02" && iter->second != "v_MilitaryParade_g01_c02"
				//	&& iter->second != "v_PizzaTossing_g01_c02"	&& iter->second != "v_PlayingViolin_g01_c02")
				//	continue;

				// Hollywood2: for NC PAPER Trajectories/clip and fps
				if (iter->second != "actioncliptest00096" && iter->second != "actioncliptest00120" 
					&& iter->second != "actioncliptest00166" && iter->second != "actioncliptest00206" 
					&& iter->second != "actioncliptest00225" && iter->second != "actioncliptrain00142" 
					&& iter->second != "actioncliptrain00240" && iter->second != "actioncliptrain00336"
					&& iter->second != "actioncliptrain00497"	&& iter->second != "actioncliptrain00519")
					continue;

				string file = resultPath + "features" + spritLabel + *itype + spritLabel + iter->second + ".bin";
				action.bin2Mat(file, tmpMat, "features");
				if( tmpMat.empty() ){
					cout << "Read Error: features is empty in " << file << endl;
					continue;		
				}
				cout << iter->second << " idts:" << tmpMat.rows << endl;
				des.push_back(tmpMat);
				tmpMat.release();
			}
		}// for iter
	}// for itype

	t = ( (double)getTickCount() - t ) / getTickFrequency();
	fslog.open(file_loger, FileStorage::WRITE);
	fslog << "featuresExtract" << "{" ;
	fslog << "time_hours" << t/3600;
	fslog << "video_num" << video_num;
	fslog << "avgFPS" << totalframes / t;
	fslog << "traj_num" << des.rows/video_num;
	fslog << "}";

	// �ͷ��ڴ�
	fslog.release();
	return 0;
}

int main(int argc, char** argv)
{ 
	//testDataSize();

	if ( 0 != predictHY2() )
		return -1;

	//if ( 0 != predictHMDB51() )
	//	return -1;

	//if ( 0 != predictJHMDB() )
	//	return -1;
	
	//if ( 0 != predictUCF50() )
	//	return -1;

	//if ( 0 != predictUCFSports() )
	//	return -1;

	//if ( 0 != predictKTH() )
	//	return -1;
	
	//if ( 0 != predictWZM() )
	//	return -1;

	return 0;
}
