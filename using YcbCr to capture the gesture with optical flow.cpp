
#include<vector>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;
//  ����������ȫ�ֺ���
void tracking(Mat &frame, Mat &output,int i);
bool addNewPoints();
bool acceptTrackedPoint(int i); Mat GetSkin(Mat const &src);
//  ����������ȫ�ֱ���
string window_name = "optical flow tracking";
Mat gray;   // ��ǰͼƬ
Mat gray_prev;  // Ԥ��ͼƬ
vector<Point2f> points[2];  // point0Ϊ�������ԭ��λ�ã�point1Ϊ���������λ��
vector<Point2f> initial;    // ��ʼ�����ٵ��λ��
vector<Point2f> features;   // ��������
int maxCount = 500; // �������������
double qLevel = 0.01;   // �������ĵȼ�
double minDist = 10.0;  // ��������֮�����С����
vector<uchar> status;   // ����������״̬��������������Ϊ1������Ϊ0
vector<float> err;
//�����Ӧ��Ϣ��OpenCV�汾-----
static void helpinformation()
{
	cout << "\n\n\t\t\t �����������˶�Ŀ����\n"
		<< "\n\n\t\t\t ��ǰʹ�õ�OpenCV�汾Ϊ��" << CV_VERSION
		<< "\n\n";
}

//main( )�������������
string outputVideoPath = "..\\test.avi";

VideoWriter outputVideo;
int main()
{
	Mat frame;
	Mat frameImage;
	Mat result;
	//����ʹ�õ���Ƶ�ļ���������Ŀ���������ļ���
	VideoCapture capture(0);
	cv::Size S = cv::Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	 
	outputVideo.open(outputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),30.0, S);
	if (!outputVideo.isOpened()) {
		cout << "fail to open!" << endl;
		return -1;
	}
	//��ʾ��Ϣ����
	helpinformation();
	int i = 5;
	// ����ͷ��ȡ�ļ�����
	if (capture.isOpened())
	{
		
		while (true)
		{
			capture >> frame;

			if (!frame.empty())
			{
				tracking(frame, result,i);
			}
			else
			{
				printf(" --(!) No captured frame -- Break!");
				break;
			}
			int c = waitKey(30);
			if ((char)c == 27)
			{
				break;
			}
			if (char(waitKey(30)) == 'z')//��z��ʼ¼��
			{
				i = 1;
				cout << "recording" << endl;
			}
			if (char(waitKey(30)) == 'q')
			{
				outputVideo.release();
				cout << "recording stopped" << endl;
			}


		
	
		}
	}
	return 0;
}

// parameter: frame �������Ƶ֡
//            output �и��ٽ������Ƶ֡
void tracking(Mat &frame, Mat &output,int i)
{
	cvtColor(frame, gray, CV_BGR2GRAY);
	frame = GetSkin(frame);//��ֵ����ɫ
	frame.copyTo(output);
	// ���������
	if (addNewPoints())
	{
		goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist);
		points[0].insert(points[0].end(), features.begin(), features.end());
		initial.insert(initial.end(), features.begin(), features.end());
	}

	if (gray_prev.empty())
	{
		gray.copyTo(gray_prev);
	}
	// l-k�������˶�����
	calcOpticalFlowPyrLK(gray_prev, gray, points[0], points[1], status, err);
	// ȥ��һЩ���õ�������
	int k = 0;
	for (size_t i = 0; i<points[1].size(); i++)
	{
		if (acceptTrackedPoint(i))
		{
			initial[k] = initial[i];
			points[1][k++] = points[1][i];
		}
	}
	points[1].resize(k);
	initial.resize(k);
	// ��ʾ��������˶��켣
	for (size_t i = 0; i<points[1].size(); i++)
	{
		line(output, initial[i], points[1][i], Scalar(0, 0, 255));
		circle(output, points[1][i], 3, Scalar(0, 255, 0), -1);
	}

	// �ѵ�ǰ���ٽ����Ϊ��һ�˲ο�
	swap(points[1], points[0]);
	swap(gray_prev, gray);
	imshow(window_name, output);
	if (i == 1)
	{
		outputVideo << output;
	}
	

}

//  ����µ��Ƿ�Ӧ�ñ����
// return: �Ƿ���ӱ�־
bool addNewPoints()
{
	return points[0].size() <= 10;
}

//������Щ���ٵ㱻����
bool acceptTrackedPoint(int i)
{
	return status[i] && ((abs(points[0][i].x - points[1][i].x) + abs(points[0][i].y - points[1][i].y)) > 2);
}



//��ɫ�ָ�
using namespace cv;

using std::cout;
using std::endl;

bool R1(int R, int G, int B) {
	bool e1 = (R>95) && (G>40) && (B>20) && ((max(R, max(G, B)) - min(R, min(G, B)))>15) && (abs(R - G)>15) && (R>G) && (R>B);
	bool e2 = (R>220) && (G>210) && (B>170) && (abs(R - G) <= 15) && (R>B) && (G>B);
	return (e1 || e2);
}

bool R2(float Y, float Cr, float Cb) {
	bool e3 = Cr <= 1.5862*Cb + 20;
	bool e4 = Cr >= 0.3448*Cb + 76.2069;
	bool e5 = Cr >= -4.5652*Cb + 234.5652;
	bool e6 = Cr <= -1.15*Cb + 301.75;
	bool e7 = Cr <= -2.2857*Cb + 432.85;
	return e3 && e4 && e5 && e6 && e7;
}

bool R3(float H, float S, float V) {
	return (H<25) || (H > 230);
}

Mat GetSkin(Mat const &src) {
	// allocate the result matrix
	Mat dst = src.clone();

	Vec3b cwhite = Vec3b::all(255);
	Vec3b cblack = Vec3b::all(0);

	Mat src_ycrcb, src_hsv;
	// OpenCV scales the YCrCb components, so that they
	// cover the whole value range of [0,255], so there's
	// no need to scale the values:
	cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
	// OpenCV scales the Hue Channel to [0,180] for
	// 8bit images, so make sure we are operating on
	// the full spectrum from [0,360] by using floating
	// point precision:
	src.convertTo(src_hsv, CV_32FC3);
	cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
	// Now scale the values between [0,255]:
	normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

	Mat final_hull = Mat::ones(src.rows, src.cols, CV_8UC3);
	bitwise_not(final_hull, final_hull);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
			int B = pix_bgr.val[0];
			int G = pix_bgr.val[1];
			int R = pix_bgr.val[2];
			// apply rgb rule
			bool a = R1(R, G, B);

			Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
			int Y = pix_ycrcb.val[0];
			int Cr = pix_ycrcb.val[1];
			int Cb = pix_ycrcb.val[2];
			// apply ycrcb rule
			bool b = R2(Y, Cr, Cb);

			Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
			float H = pix_hsv.val[0];
			float S = pix_hsv.val[1];
			float V = pix_hsv.val[2];
			// apply hsv rule
			bool c = R3(H, S, V);




			if (!(a&&b&&c))
				// dst.ptr<Vec3b>(i)[j] = cblack;
				final_hull.ptr<Vec3b>(i)[j] = cblack;
		}
	}
	return final_hull;
}
