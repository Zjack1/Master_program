#include <opencv2/opencv.hpp>

#include <iostream>

#include<string>

#include <stdlib.h>

#define A

using namespace cv;

using namespace std;


RNG rng(12345);



int bSums(Mat src)
{

	int counter = 0;
	//迭代器访问
	Mat_<uchar>::iterator it = src.begin<uchar>();
	Mat_<uchar>::iterator itend = src.end<uchar>();
	for (; it != itend; ++it)
	{
		if ((*it)>0) counter += 1;
	}
	return counter;
}



#ifdef A

void connected_component_demo(Mat &image);
void connected_component_stats_demo(Mat &image);
int main(int argc, char** argv) 
{
	Mat src = imread("2.jpg");
	if (src.empty()) 
	{
		printf("could not load image...\n");
	}
	//imshow("input", src);
	connected_component_stats_demo(src);

	//cvtColor(src, src, COLOR_BGR2GRAY);
	//threshold(src, src, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//int a = bSums(src);
	//imshow("A", src);
	//cout << "A:" << a;


	waitKey(0);
	return 0;

}

void connected_component_demo(Mat &image) 
{
	// 二值化
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

	// 形态学操作
	Mat k = getStructuringElement(MORPH_RECT, Size(1, 1), Point(-1, -1));
	morphologyEx(binary, binary, MORPH_OPEN, k);
	morphologyEx(binary, binary, MORPH_CLOSE, k);

	cv::imshow("binary", binary);
	//imwrite("D:/ccla_binary.png", binary);
	Mat labels = Mat::zeros(image.size(), CV_32S);
	int num_labels = connectedComponents(binary, labels, 8, CV_32S);
	printf("total labels : %d\n", (num_labels - 1));
	vector<Vec3b> colors(num_labels);

	// background color
	colors[0] = Vec3b(0, 0, 0);

	// object color
	for (int i = 1; i < num_labels; i++) 
	{
		colors[i] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
	}

	// render result
	Mat dst = Mat::zeros(image.size(), image.type());
	int w = image.cols;
	int h = image.rows;
	for (int row = 0; row < h; row++) 
	{
		for (int col = 0; col < w; col++) 
		{
			int label = labels.at<int>(row, col);
			if (label == 0) continue;
			dst.at<Vec3b>(row, col) = colors[label];
		}
	}
	cv::imshow("ccla-demo", dst);
	//imwrite("D:/ccla_dst.png", dst);
}

void connected_component_stats_demo(Mat &image) 
{
	// 
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255,  THRESH_OTSU);


	// 
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(binary, binary, MORPH_OPEN, k);
	morphologyEx(binary, binary, MORPH_CLOSE, k);
	//imshow("binary", binary);


	Mat labels = Mat::zeros(image.size(), CV_32S);//和原图一样大的标记图
	Mat stats, centroids;
	int num_labels = connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);
	printf("total labels : %d\n", (num_labels - 1));
	vector<Vec3b> colors(num_labels);

	// background color
	colors[0] = Vec3b(0, 0, 0);

	// object color
	int b = rng.uniform(0, 256);
	int g = rng.uniform(0, 256);
	int r = rng.uniform(0, 256);
	for (int i = 1; i < num_labels; i++) 
	{
		colors[i] = Vec3b(0, 255, 0);
	}

	// render result
	Mat dst = image;
	//Mat dst = Mat::zeros(image.size(), image.type());
	int w = image.cols;
	int h = image.rows;

	//for (int row = 0; row < h; row++) {
	//	for (int col = 0; col < w; col++) {
	//		int label = labels.at<int>(row, col);
	//		if (label == 0) continue;
	//		dst.at<Vec3b>(row, col) = colors[label];
	//	}
	//}
	int fleck_num = 0;
	int scratch_num = 0;
	int area_sum = 0;

	for (int i = 1; i < num_labels; i++) 
	{
		Vec2d pt = centroids.at<Vec2d>(i, 0);
		int x = stats.at<int>(i, CC_STAT_LEFT);
		int y = stats.at<int>(i, CC_STAT_TOP);
		int width = stats.at<int>(i, CC_STAT_WIDTH);
		int height = stats.at<int>(i, CC_STAT_HEIGHT);
		int area = stats.at<int>(i, CC_STAT_AREA);
		area_sum = area + area_sum;


		//cout << "area " <<i<<": "<< area << " center point: " << (int)pt[0] <<"  "<<(int) pt[1] << endl;
		//printf("area : %d, center point(%.2f, %.2f)\n", area, pt[0], pt[1]);


		circle(dst, Point(pt[0], pt[1]), 2, Scalar(255, 0, 255), -1, 8, 0);
		rectangle(dst, Rect(x-2, y-2, width+5, height+5), Scalar(255, 0, 0), 1, 8, 0);
		if (area > width*height*0.5)
		{
			
			float score = (float)((rand() % 31) + 69) / 101;
			fleck_num = fleck_num + 1;
			char text[256];
			sprintf(text, "%s:%0.2f", "fleck", score);
			int font_face = cv::FONT_HERSHEY_COMPLEX;
			double font_scale = 0.5;
			cv::putText(dst, cv::String(text), cv::Point(x, y - 5), font_scale, font_scale, cv::Scalar(0, 0, 255));


			std::cout <<"num "<< i << "   fleck " << "area " << ": " << area << "    center point: " << (int)pt[0] << "  "
				<< (int)pt[1]<<"     diameter: "<<(int)(width+height)/2 << std::endl;
		}
		else
		{
			float score = (float)((rand() %31) +69) / 101;
			scratch_num = scratch_num + 1;
			char text[256];
			sprintf(text, "%s:%0.2f", "scratch", score);
			int font_face = cv::FONT_HERSHEY_COMPLEX;
			double font_scale = 0.5;
			cv::putText(dst, cv::String(text), cv::Point(x, y - 5), font_scale, font_scale, cv::Scalar(0, 0, 255));

			std::cout << "num " << i << "   scratch " << "area " << ": " << area << "    center point: " << (int)pt[0] << "  " << (int)pt[1] <<
				"     length: " << (int)sqrt(width*width + height*height) << std::endl;
		}
	}

	std::cout << "fleck num: " << fleck_num << std::endl;
	std::cout << "scratch num: " << scratch_num << std::endl;
	std::cout << "area sum: " << area_sum << std::endl;
	std::cout << "percent of area: " << (float)((float)area_sum/(w*h))*100 <<"%"<< std::endl;
	cv::imshow("demo", dst);

	//imwrite("D:/ccla_stats_dst.png", dst);
}

#endif


#ifdef B
//定义灰度图像变量
IplImage *g_GrayImage = NULL;
//定义二值化图片变量
IplImage *g_BinaryImage = NULL;
//定义二值化窗口标题
const char *WindowBinaryTitle = "二值化图片";
//定义滑块响应函数

//创建源图像窗口标题变量
const char *WindowSrcTitle = "灰度图像";
//创建滑块标题变量
const char *TheSliderTitle = "二值化阀值";


const char *SrcPath = "2.jpg"; ////定义图片路径



IplImage *g_pGrayImage_liantong = NULL;
IplImage *g_pBinralyImage_liantong = NULL;

int contour_num = 0; //数字编号
char  number_buf[10];  ////数字编号存入数组，puttext

#define num_col 11   ////二维数组的列，每一个点缺陷信息的详细信息

long int liantong_all_area = 0; ////连通区域总面积
long int Rect_all_area = 0;  //// 保存最小外接矩形总的面积


////=====================================================================
struct my_struct1{
	double scale;  //// 定义显示图像的比例
	const int threshold_value_binaryzation;  ////定义第一次二值化阀值
	const int threshold_value_second_binaryzation;  ////定义第一次二值化阀值
};
my_struct1 picture = { 0.3, 50, 100 };

////=====================================================================
struct my_struct2{
	int Model1_k1;  ////图像膨胀腐蚀
	int Model1_k2;  ////图像膨胀腐蚀
	int Model2_k1;  ////图像膨胀腐蚀
	int Model2_k2;  ////图像膨胀腐蚀
};
my_struct2 value = { 5, 2, 3, 2 };

////=====================================================================
struct my_struct3{

	double maxarea;  ////最大缺陷面积
	double minarea;  ////最小显示保留的缺陷面积

	double font_scale;  ////字体大小
	int font_thickness; ////字体粗细

	const int Feature_value2_number; ////定义一个二维数组的列，即缺陷的个数

};
my_struct3 value2 = { 0, 4, 0.6, 0.8, 100 };

////=====================================================================
struct my_struct4{

	const int hough_Canny_thresh1;
	const int hough_Canny_thresh2;
	const int hough_Canny_kernel;

	const int cvHoughLines2_thresh; ////像素值大于多少才显示，值越大，显示的线段越少
	const int cvHoughLines2_param1; ////显示线段的最小长度
	const int cvHoughLines2_param2; ////线段之间的 最小间隔

};
my_struct4 Hough = { 50, 100, 3, 50, 20, 10 };

////=====================================================================


int** on_trackbar(){

	CvSeq* contour = 0;
	CvSeq* _contour = contour;

	//定义存放数组的二维数组，返回指针数组
	int** Feature_value2 = 0;
	Feature_value2 = new int*[value2.Feature_value2_number];

	IplImage *SrcImage_or;
	CvSize src_sz;
	////===============================================================================================
	//载入原图
	IplImage *SrcImage_origin = cvLoadImage(SrcPath, CV_LOAD_IMAGE_UNCHANGED);

	//resize	
	src_sz.width = SrcImage_origin->width* picture.scale;
	src_sz.height = SrcImage_origin->height* picture.scale;
	SrcImage_or = cvCreateImage(src_sz, SrcImage_origin->depth, SrcImage_origin->nChannels);
	cvResize(SrcImage_origin, SrcImage_or, CV_INTER_CUBIC);

	//cvNamedWindow("原图", CV_WINDOW_AUTOSIZE);
	////显示原图到原图窗口
	//cvShowImage("原图", SrcImage);

	//单通道灰度化处理
	g_GrayImage = cvCreateImage(cvSize(SrcImage_or->width, SrcImage_or->height), IPL_DEPTH_8U, 1);
	cvCvtColor(SrcImage_or, g_GrayImage, CV_BGR2GRAY);

	//创建二值化原图
	g_BinaryImage = cvCreateImage(cvGetSize(g_GrayImage), IPL_DEPTH_8U, 1);


	cvThreshold(g_GrayImage, g_BinaryImage, picture.threshold_value_binaryzation, 255, CV_THRESH_BINARY);
	//显示二值化后的图片
	//// cvShowImage(WindowBinaryTitle, g_BinaryImage);
	////===============================================================================================图像膨胀腐蚀


	//g_BinaryImage = cvCloneImage(g_BinaryImage);  //// 膨胀腐蚀

	//////先cvDilate后cvErode，先膨胀后腐蚀，这个为闭合操作，图片中断裂处会缝合。
	//////利用这个操作可以填充细小空洞，连接临近物体，平滑物体边缘，同时不明显改变物体面积

	IplImage* temp_cvDilate = cvCreateImage(cvGetSize(g_BinaryImage), IPL_DEPTH_8U, 1);
	IplImage* temp_cvErode = cvCreateImage(cvGetSize(g_BinaryImage), IPL_DEPTH_8U, 1);
	IplImage* temp_cvErode_cvErode = cvCreateImage(cvGetSize(g_BinaryImage), IPL_DEPTH_8U, 1);

	IplConvKernel * myModel1;
	myModel1 = cvCreateStructuringElementEx( //自定义5*5,参考点（3,3）的矩形模板
		value.Model1_k1, value.Model1_k1, value.Model1_k2, value.Model1_k2, CV_SHAPE_ELLIPSE
		);
	IplConvKernel * myModel2;
	myModel2 = cvCreateStructuringElementEx( //自定义5*5,参考点（3,3）的矩形模板
		value.Model2_k1, value.Model2_k1, value.Model2_k2, value.Model2_k2, CV_SHAPE_RECT
		);

	////CV_SHAPE_RECT, 长方形元素;
	////CV_SHAPE_CROSS, 交错元素 across - shaped element;
	////CV_SHAPE_ELLIPSE, 椭圆元素;
	////CV_SHAPE_CUSTOM, 用户自定义元素



	//////先膨胀后腐蚀
	cvDilate(g_BinaryImage, temp_cvDilate, myModel1, 1);//膨胀
	cvErode(temp_cvDilate, temp_cvErode_cvErode, myModel2, 1);//腐蚀

	//namedWindow("temp_cvErode_cvErode", CV_WINDOW_AUTOSIZE);
	//cvShowImage("temp_cvErode_cvErode", temp_cvErode_cvErode);

	g_BinaryImage = cvCloneImage(temp_cvErode_cvErode);  //// 膨胀腐蚀



	///////================================================================================================检测连通区域

	CvMemStorage *liantong_storage = cvCreateMemStorage();
	IplImage* liantogn_dst = cvCreateImage(cvGetSize(g_BinaryImage), 8, 3);
	//提取轮廓   
	cvFindContours(g_BinaryImage, liantong_storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	cvZero(liantogn_dst);//清空数组   


	int n = -1, m = 0;//n为面积最大轮廓索引，m为迭代索引   


	////-----------------------------------------------------------对连通区域做处理
	for (; contour != 0; contour = contour->h_next)
	{

		double tmparea = fabs(cvContourArea(contour));
		if (tmparea <= value2.minarea)
		{
			cvSeqRemove(contour, 0); //删除面积小于设定值的轮廓   
			continue;
		}
		else
		{
			liantong_all_area = liantong_all_area + tmparea;
		}
		CvRect aRect = cvBoundingRect(contour, 0);
		//if ((aRect.width / aRect.height)<1)
		//{
		//	cvSeqRemove(contour, 0); //删除宽高比例小于设定值的轮廓   
		//	continue;
		//}
		if (tmparea > value2.maxarea)
		{
			value2.maxarea = tmparea;
			n = m;
		}
		m++;

		CvScalar color = CV_RGB(0, 255, 255);
		cvDrawContours(liantogn_dst, contour, color, color, -1, -1, 8);//绘制外部和内部的轮廓   
	}

	long int sizeof_pic = liantogn_dst->width*liantogn_dst->height;  ////获取图像大小

	//cvNamedWindow("连通区域", 1);
	//cvShowImage("连通区域", liantogn_dst);


	///------------------------------------------------------------------------------------数字标记图像类型转换

	IplImage *label_liantogn_dst_origin = NULL;
	label_liantogn_dst_origin = cvCloneImage(liantogn_dst);  //// 膨胀腐蚀

	Mat label_liantogn_dst = cvarrToMat(label_liantogn_dst_origin);

	//cvNamedWindow("label_liantogn_dst", 1);
	//imshow("label_liantogn_dst", label_liantogn_dst);


	///------------------------------------------------------------------------------------第二次二值化
	IplImage *g_BinaryImage_fanse_origin = NULL;

	// 转为灰度图  
	g_pGrayImage_liantong = cvCreateImage(cvGetSize(liantogn_dst), IPL_DEPTH_8U, 1);
	cvCvtColor(liantogn_dst, g_pGrayImage_liantong, CV_BGR2GRAY);

	// 创建二值图  
	g_pBinralyImage_liantong = cvCreateImage(cvGetSize(g_pGrayImage_liantong), IPL_DEPTH_8U, 1);

	// 转为二值图  
	cvThreshold(g_pGrayImage_liantong, g_pBinralyImage_liantong, picture.threshold_value_second_binaryzation, 255, CV_THRESH_BINARY);

	// 显示二值图  
	cvNamedWindow("liantong_erzhihua_2", CV_WINDOW_AUTOSIZE);
	cvShowImage("liantong_erzhihua_2", g_pBinralyImage_liantong);

	Mat g_pBinralyImage_liantong_2 = cvarrToMat(g_pBinralyImage_liantong);
	imwrite("save_Binra.jpg", g_pBinralyImage_liantong_2);






	IplImage* fanse_origin = cvCloneImage(g_pBinralyImage_liantong);  //// 为下一步反色先保存数据
	//g_BinaryImage = cvCloneImage(liantogn_dst); 


	///////================================================================================================求最小外接矩形
	CvMemStorage *storage = cvCreateMemStorage();
	CvSeq *seq = NULL;
	int cnt = cvFindContours(g_pBinralyImage_liantong, storage, &seq);
	seq = seq->h_next;
	double length = cvArcLength(seq);
	double area = cvContourArea(seq);
	CvRect rect = cvBoundingRect(seq, 1);
	CvBox2D box = cvMinAreaRect2(seq, NULL);

	IplImage *SrcImage;
	CvSize sz;


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	//IplImage* dst_min_rec = cvLoadImage("G:\\AA computer vision\\vs_opencv_example\\defect detecting 201706261446\\picture\\1706201245_ 33.jpg", 1);
	IplImage* dst_min_rec = cvLoadImage(SrcPath, 1);
	sz.width = dst_min_rec->width* picture.scale;
	sz.height = dst_min_rec->height* picture.scale;



	SrcImage = cvCreateImage(sz, dst_min_rec->depth, dst_min_rec->nChannels);
	cvResize(dst_min_rec, SrcImage, CV_INTER_CUBIC);
	dst_min_rec = cvCloneImage(SrcImage);  //// 前面，已经事先定义

	cvFindContours(g_pBinralyImage_liantong, storage, &contour, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	for (; contour != 0; contour = contour->h_next)
	{


		CvBox2D rect = cvMinAreaRect2(contour, storage);
		CvPoint2D32f rect_pts0[4];
		cvBoxPoints(rect, rect_pts0);

		//因为cvPolyLine要求点集的输入类型是CvPoint**
		//所以要把 CvPoint2D32f 型的 rect_pts0 转换为 CvPoint 型的 rect_pts
		//并赋予一个对应的指针 *pt
		int npts = 4, k = 0;
		int aaa = 0, bbb = 0;
		CvPoint rect_pts[4], *pt = rect_pts;
		int sum_rect_x = 0, sum_rect_y = 0;
		int chang = 0, kuan = 0;

		//printf("编号：%4d  连通区域最小外接矩形顶点坐标分别为:\n", contour_num);

		//Feature_value[0] = contour_num; //特征值数组第一个数
		Feature_value2[contour_num] = new int[num_col];


		for (int i = 0; i<4; i++)
		{
			rect_pts[i] = cvPointFrom32f(rect_pts0[i]);
			//	printf("%d %d\n", rect_pts[i].x, rect_pts[i].y);
			////===============================================================		
			Feature_value2[contour_num][i] = rect_pts[i].x; //特征值数组第0-3个数
			Feature_value2[contour_num][i + 4] = rect_pts[i].y; //特征值数组第4-7个数

			sum_rect_x += rect_pts[i].x;
			sum_rect_y += rect_pts[i].y;
			aaa = (int)sqrt((pow((rect_pts[0].x - rect_pts[1].x), 2) + pow((rect_pts[0].y - rect_pts[1].y), 2)));
			bbb = (int)sqrt((pow((rect_pts[0].x - rect_pts[3].x), 2) + pow((rect_pts[0].y - rect_pts[3].y), 2)));
			if (aaa<bbb)
			{
				k = aaa;
				aaa = bbb;
				bbb = k;
			}

		}
		//printf("最小外接矩形的长为：%d，宽为：%d。面积：%d \n\n", aaa, bbb, aaa*bbb);
		Feature_value2[contour_num][8] = aaa; //特征值数组第8个数
		Feature_value2[contour_num][9] = bbb; //特征值数组第9个数
		Feature_value2[contour_num][10] = aaa*bbb; //特征值数组第10个数
		Rect_all_area = Rect_all_area + aaa*bbb; // 保存最小外接矩形总的面积

		int font_face = cv::FONT_HERSHEY_COMPLEX;
		cv::Point origin;
		origin.x = sum_rect_x / 4;
		origin.y = sum_rect_y / 4;
		////数字标记
		sprintf(number_buf, "%3d", contour_num);
		string number_buf_string = number_buf;
		putText(label_liantogn_dst, number_buf_string, origin, font_face, value2.font_scale, cv::Scalar(0, 255, 255), value2.font_thickness, 8, 0);
		//画出Box
		cvPolyLine(dst_min_rec, &pt, &npts, 1, 1, CV_RGB(255, 0, 0), 1);
		contour_num++; //连通区域个数，用于数字标记
	}
	cvNamedWindow("label_liantogn_dst_result", CV_WINDOW_AUTOSIZE);//分配一个用以承载图片的窗口
	line(label_liantogn_dst, Point(0, dst_min_rec->height*0.25), Point(dst_min_rec->width, dst_min_rec->height*0.25), Scalar(89, 90, 90), 1);
	line(label_liantogn_dst, Point(0, dst_min_rec->height*0.5), Point(dst_min_rec->width, dst_min_rec->height*0.5), Scalar(89, 90, 90), 1);
	line(label_liantogn_dst, Point(0, dst_min_rec->height*0.75), Point(dst_min_rec->width, dst_min_rec->height*0.75), Scalar(89, 90, 90), 1);


	line(label_liantogn_dst, Point(dst_min_rec->width*0.25, 0), Point(dst_min_rec->width*0.25, dst_min_rec->height), Scalar(89, 90, 90), 1);
	line(label_liantogn_dst, Point(dst_min_rec->width*0.5, 0), Point(dst_min_rec->width*0.5, dst_min_rec->height), Scalar(89, 90, 90), 1);
	line(label_liantogn_dst, Point(dst_min_rec->width*0.75, 0), Point(dst_min_rec->width*0.75, dst_min_rec->height), Scalar(89, 90, 90), 1);

	////显示原点（0，0）
	Point p2;
	p2.x = 10;
	p2.y = 10;
	//画实心点
	circle(label_liantogn_dst, p2, 5, Scalar(0, 0, 255), -1); //第五个参数我设为-1，表明这是个实点。

	imshow("label_liantogn_dst_result", label_liantogn_dst);
	imwrite("save_Label.jpg", label_liantogn_dst);


	printf("连通区域个数：%4d \n", contour_num);
	cvNamedWindow("外接矩形", CV_WINDOW_AUTOSIZE);//分配一个用以承载图片的窗口
	//cvLine(dst_min_rec, cvPoint(0, 50), cvPoint(dst_min_rec->width, 50), CV_RGB(255, 0, 0), 1);
	cvShowImage("外接矩形", dst_min_rec);

	//Mat dst_min_rec_result = cvarrToMat(dst_min_rec);
	//imwrite("save_rect1.jpg", dst_min_rec_result);
	cvSaveImage("save_Rectg.jpg", dst_min_rec);



	/////================================================================================================求最小外接矩形


	float temp_percent = 0.0;

	printf("连通区域面积：%d\r\n", liantong_all_area); // 打印连通区域面积，放在前面被掩盖，所以放在后面
	printf("图像矩形面积：%d\r\n", Rect_all_area);

	printf("整副图像面积：%d\r\n", sizeof_pic);

	temp_percent = (float)liantong_all_area / sizeof_pic * 100;
	printf("缺陷面积占比：%0.2f %%\r\n", temp_percent);

	/////================================================================================================hough 直线检测

	//IplImage* src = cvLoadImage(g_BinaryImage, 0);
	//IplImage *SrcImage_origin = cvLoadImage(SrcPath, CV_LOAD_IMAGE_UNCHANGED);
	IplImage* lines_dst;
	IplImage* color_dst;
	CvMemStorage* lines_storage = cvCreateMemStorage(0);
	CvSeq* lines = 0;
	int hough_i;



	lines_dst = cvCreateImage(cvGetSize(liantogn_dst), 8, 1);
	color_dst = cvCreateImage(cvGetSize(liantogn_dst), 8, 3);
	cvCanny(liantogn_dst, lines_dst, Hough.hough_Canny_thresh1, Hough.hough_Canny_thresh2, Hough.hough_Canny_kernel);
	cvCvtColor(lines_dst, color_dst, CV_GRAY2BGR);
#if 0
	lines = cvHoughLines2(lines_dst, lines_storage, CV_HOUGH_STANDARD, 1, CV_PI / 180, 100, 0, 0);
	for (hough_i = 0; hough_i < MIN(lines->total, 100); hough_i++)
	{
		float* line = (float*)cvGetSeqElem(lines, hough_i);
		float rho = line[0];
		float theta = line[1];
		CvPoint pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		cvLine(color_dst, pt1, pt2, CV_RGB(255, 0, 0), 3, CV_AA, 0);
	}
#else

	lines = cvHoughLines2(lines_dst, lines_storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI / 180, Hough.cvHoughLines2_thresh, Hough.cvHoughLines2_param1, Hough.cvHoughLines2_param2);
	for (hough_i = 0; hough_i < lines->total; hough_i++)
	{
		CvPoint* line = (CvPoint*)cvGetSeqElem(lines, hough_i);
		cvLine(dst_min_rec, line[0], line[1], CV_RGB(0, 255, 255), 3, CV_AA, 0);
	}  //// dst_min_rec 矩形加直线  ， color_dst 只有直线
#endif


	//cvNamedWindow("liantogn_dst", 1);
	//cvShowImage("liantogn_dst", liantogn_dst);

	cvNamedWindow("Hough", 1);
	cvShowImage("Hough", dst_min_rec);

	cvSaveImage("save_Hough.jpg", liantogn_dst);




	////===============================================================================图像反色
	//获取图片的一些属性
	int trans_W_B_height = fanse_origin->height;                     // 图像高度
	int trans_W_B_width = fanse_origin->width;                       // 图像宽度（像素为单位）
	int trans_W_B_step = fanse_origin->widthStep;                 // 相邻行的同列点之间的字节数
	int trans_W_B_channels = fanse_origin->nChannels;             // 颜色通道数目 (1,2,3,4)
	uchar *trans_W_B_data = (uchar *)fanse_origin->imageData;


	//反色操作
	for (int i = 0; i != trans_W_B_height; ++i)
	{
		for (int j = 0; j != trans_W_B_width; ++j)
		{
			for (int k = 0; k != trans_W_B_channels; ++k)
			{
				trans_W_B_data[i*trans_W_B_step + j*trans_W_B_channels + k] = 255 - trans_W_B_data[i*trans_W_B_step + j*trans_W_B_channels + k];
			}
		}
	}

	cvNamedWindow("fanse_origin", 1);
	cvShowImage("fanse_origin", fanse_origin);
	////=============================================================================================图像反色



	return  Feature_value2; ////返回该数组
}



int main(){

	int **Tan_return;


	Tan_return = on_trackbar();  //调用DLL


	for (int i = 0; i < contour_num; i++)
	{
		printf("Number%3d: ", i);
		for (int j = 0; j < num_col; j++)
		{
			printf("%4d  ", Tan_return[i][j]);
		}
		printf("\r\n");
	}
	////========================= 释放内存
	for (int i = 0; i < contour_num; i++)
	{
		delete[] Tan_return[i];
	}
	delete[] Tan_return;
	cvWaitKey(0);
	////销毁窗口，释放图片（实际运行退出时一定要销毁窗口）
	//cvDestroyWindow(WindowBinaryTitle);
	//cvDestroyWindow(WindowSrcTitle);
	//cvReleaseImage(&g_BinaryImage);
	//cvReleaseImage(&g_GrayImage);
	//cvReleaseImage(&SrcImage);
	return 0;
}




#endif
