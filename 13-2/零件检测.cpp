#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

//模板匹配
void match_template(Mat templ, Mat src, int cellsize, int anglenum);
//计算直方图
void hog_hisgram(InputArray src, float* histogram, int cellsize, int anglenum);
//计算直方图距离
float Similarity(float* hist1, float* hist2, int length);

int main()
{
	Mat refImg = imread("C:/Users/DELL/Desktop/2.jpg");//模板
	Mat srcImg = imread("C:/Users/DELL/Desktop/3.jpg");//输入图像

	if (refImg.empty() || srcImg.empty())
	{
		cout << "打开图像发生错误" << endl;
		destroyAllWindows();
		return -1;
	}

	int cell_size = 16;                  //16*16的cell
	int angle_num = 8;                   //角度量化为8

	match_template(refImg, srcImg, cell_size, angle_num);//调用模板匹配函数
	imshow("模板", refImg);
	waitKey(0);
}

void match_template(Mat templ, Mat src, int cellsize, int anglenum)
{
	//图像分割为y_num行，x_num列
	int x_num = templ.cols / cellsize;
	int y_num = templ.rows / cellsize;
	int bins = x_num * y_num * anglenum;//数组长度

	//开辟动态数组
	//计算模板的梯度直方图
	float* ref_hog = new float[bins];  //模板
	memset(ref_hog, 0, sizeof(float) * bins);
	hog_hisgram(templ, ref_hog, cellsize, anglenum);

	//相似度结果的输出矩阵
	int resultMat_rows = src.rows - templ.rows + 1;
	int resultMat_cols = src.cols - templ.cols + 1;
	Mat resultMat = Mat(resultMat_rows, resultMat_cols, CV_32FC1);

	//Roi区域的尺寸
	Rect Select;
	Select.height = templ.rows;
	Select.width = templ.cols;

	//得到相似度的输出矩阵
	for (int i = 0; i < resultMat_rows; i++)
	{
		for (int j = 0; j < resultMat_cols; j++)
		{
			Select.x = j;
			Select.y = i;
			Mat RoiMat = src(Select).clone();//取ROI
			//计算ROI的梯度直方图
			float* roi_hog = new float[bins];
			memset(roi_hog, 0, sizeof(float) * bins);
			hog_hisgram(RoiMat, roi_hog, cellsize, anglenum);
			//计算相似度
			float smlrt = Similarity(ref_hog, roi_hog, bins);
			resultMat.at<float>(i, j) = smlrt;
			delete[] roi_hog;
		}
	}

	delete[] ref_hog;
	//寻找匹配的最高的位置
	double minValue, maxValue;
	Point minLoction, maxLoction;
	minMaxLoc(resultMat, &minValue, &maxValue, &minLoction, &maxLoction);
	//标示出位置
	rectangle(src, maxLoction, Point(maxLoction.x + templ.cols, maxLoction.y + templ.rows), Scalar(255, 255, 0));
	imwrite("G:\\Projects\\cpp\\Task_Proj\\2020.12.23_week15\\2020.12.23_week15\\结果.jpg", src);
	imshow("显示结果", src);
}

void hog_hisgram(InputArray src, float* histogram, int cellsize, int anglenum)
{
	Mat gray, grd_x, grd_y;                           //灰度，x方向和y方向的梯度
	//计算像素梯度的幅值和方向
	cvtColor(src, gray, COLOR_BGR2GRAY);
	Mat angle, mag;                                   //梯度方向，梯度幅值
	Sobel(gray, grd_x, CV_32F, 1, 0, 3);
	Sobel(gray, grd_y, CV_32F, 0, 1, 3);
	cartToPolar(grd_x, grd_y, mag, angle, true);

	//计算cell的个数
	//图像分割为y_cellnum行，x_cellnum列
	int x_cellnum, y_cellnum;
	x_cellnum = gray.cols / cellsize;
	y_cellnum = gray.rows / cellsize;
	int angle_area = 360 / anglenum;                  //每个量化级数所包含的角度数

	//定义感兴趣区域roi
	Rect roi;
	roi.width = cellsize;
	roi.height = cellsize;

	//外循环，遍历cell
	for (int i = 0; i < y_cellnum; i++)
	{
		for (int j = 0; j < x_cellnum; j++)
		{
			//取出每个cell
			roi.x = j * cellsize;
			roi.y = i * cellsize;

			Mat RoiAngle, RoiMag;
			RoiAngle = angle(roi);                    //每个cell中的梯度方向
			RoiMag = mag(roi);                        //每个cell中的梯度幅值

			//遍历RoiAngel和RoiMat
			int head = (i * x_cellnum + j) * anglenum;//cell梯度直方图的第一个元素在总直方图中的位置
			for (int m = 0; m < cellsize; m++)
			{
				for (int n = 0; n < cellsize; n++)
				{
					int idx = ((int)RoiAngle.at<float>(m, n)) / angle_area;//该梯度所处的量化级数
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
	return 1 / (1 + distance);//返回相似度
}