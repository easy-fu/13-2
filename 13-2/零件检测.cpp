#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

//ģ��ƥ��
void match_template(Mat templ, Mat src, int cellsize, int anglenum);
//����ֱ��ͼ
void hog_hisgram(InputArray src, float* histogram, int cellsize, int anglenum);
//����ֱ��ͼ����
float Similarity(float* hist1, float* hist2, int length);

int main()
{
	Mat refImg = imread("C:/Users/DELL/Desktop/2.jpg");//ģ��
	Mat srcImg = imread("C:/Users/DELL/Desktop/3.jpg");//����ͼ��

	if (refImg.empty() || srcImg.empty())
	{
		cout << "��ͼ��������" << endl;
		destroyAllWindows();
		return -1;
	}

	int cell_size = 16;                  //16*16��cell
	int angle_num = 8;                   //�Ƕ�����Ϊ8

	match_template(refImg, srcImg, cell_size, angle_num);//����ģ��ƥ�亯��
	imshow("ģ��", refImg);
	waitKey(0);
}

void match_template(Mat templ, Mat src, int cellsize, int anglenum)
{
	//ͼ��ָ�Ϊy_num�У�x_num��
	int x_num = templ.cols / cellsize;
	int y_num = templ.rows / cellsize;
	int bins = x_num * y_num * anglenum;//���鳤��

	//���ٶ�̬����
	//����ģ����ݶ�ֱ��ͼ
	float* ref_hog = new float[bins];  //ģ��
	memset(ref_hog, 0, sizeof(float) * bins);
	hog_hisgram(templ, ref_hog, cellsize, anglenum);

	//���ƶȽ�����������
	int resultMat_rows = src.rows - templ.rows + 1;
	int resultMat_cols = src.cols - templ.cols + 1;
	Mat resultMat = Mat(resultMat_rows, resultMat_cols, CV_32FC1);

	//Roi����ĳߴ�
	Rect Select;
	Select.height = templ.rows;
	Select.width = templ.cols;

	//�õ����ƶȵ��������
	for (int i = 0; i < resultMat_rows; i++)
	{
		for (int j = 0; j < resultMat_cols; j++)
		{
			Select.x = j;
			Select.y = i;
			Mat RoiMat = src(Select).clone();//ȡROI
			//����ROI���ݶ�ֱ��ͼ
			float* roi_hog = new float[bins];
			memset(roi_hog, 0, sizeof(float) * bins);
			hog_hisgram(RoiMat, roi_hog, cellsize, anglenum);
			//�������ƶ�
			float smlrt = Similarity(ref_hog, roi_hog, bins);
			resultMat.at<float>(i, j) = smlrt;
			delete[] roi_hog;
		}
	}

	delete[] ref_hog;
	//Ѱ��ƥ�����ߵ�λ��
	double minValue, maxValue;
	Point minLoction, maxLoction;
	minMaxLoc(resultMat, &minValue, &maxValue, &minLoction, &maxLoction);
	//��ʾ��λ��
	rectangle(src, maxLoction, Point(maxLoction.x + templ.cols, maxLoction.y + templ.rows), Scalar(255, 255, 0));
	imwrite("G:\\Projects\\cpp\\Task_Proj\\2020.12.23_week15\\2020.12.23_week15\\���.jpg", src);
	imshow("��ʾ���", src);
}

void hog_hisgram(InputArray src, float* histogram, int cellsize, int anglenum)
{
	Mat gray, grd_x, grd_y;                           //�Ҷȣ�x�����y������ݶ�
	//���������ݶȵķ�ֵ�ͷ���
	cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat angle, mag;                                   //�ݶȷ����ݶȷ�ֵ
	Sobel(gray, grd_x, CV_32F, 1, 0, 3);
	Sobel(gray, grd_y, CV_32F, 0, 1, 3);
	cartToPolar(grd_x, grd_y, mag, angle, true);

	//����cell�ĸ���
	//ͼ��ָ�Ϊy_cellnum�У�x_cellnum��
	int x_cellnum, y_cellnum;
	x_cellnum = gray.cols / cellsize;
	y_cellnum = gray.rows / cellsize;
	int angle_area = 360 / anglenum;                  //ÿ�����������������ĽǶ���

	//�������Ȥ����roi
	Rect roi;
	roi.width = cellsize;
	roi.height = cellsize;

	//��ѭ��������cell
	for (int i = 0; i < y_cellnum; i++)
	{
		for (int j = 0; j < x_cellnum; j++)
		{
			//ȡ��ÿ��cell
			roi.x = j * cellsize;
			roi.y = i * cellsize;

			Mat RoiAngle, RoiMag;
			RoiAngle = angle(roi);                    //ÿ��cell�е��ݶȷ���
			RoiMag = mag(roi);                        //ÿ��cell�е��ݶȷ�ֵ

			//����RoiAngel��RoiMat
			int head = (i * x_cellnum + j) * anglenum;//cell�ݶ�ֱ��ͼ�ĵ�һ��Ԫ������ֱ��ͼ�е�λ��
			for (int m = 0; m < cellsize; m++)
			{
				for (int n = 0; n < cellsize; n++)
				{
					int idx = ((int)RoiAngle.at<float>(m, n)) / angle_area;//���ݶ���������������
					histogram[head + idx] += RoiMag.at<float>(m, n);
				}
			}
		}
	}

}

float Similarity(float* hist1, float* hist2, int length)
{
	float sum = 0;
	float distance;
	for (int i = 0; i < length; i++)
	{
		sum += (hist1[i] - hist2[i])* (hist1[i] - hist2[i]);
	}
	distance = sqrt(sum);
	return 1 / (1 + distance);//�������ƶ�
}