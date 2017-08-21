// KinectGestureClassifier.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

	ifstream infile("xxx.txt");

	int x = 0, y = 0, width = 640, height = 480;
	const float depth_near = 500;
	const float depth_far = 1000;
	const float alpha = 255.0f / (depth_far - depth_near);
	const float beta = -depth_near * alpha;

	Mat test = Mat::zeros(height, width, CV_32F);
	Mat depthdata = Mat::zeros(height, width, CV_16UC1);
	
	float minval = 255;
	
	while (infile)
	{
		string s;
		if (!getline(infile, s)) 
			break;

		istringstream ss(s);

		while (ss)
		{
			string s;
			if (!getline(ss, s, ',')) 
				break;

			depthdata.at<uint16_t>(y, x) = (int)stoi(s);
			float v = (float)stoi(s) * alpha + beta;

			if (v > 255) 
				v = 255;
			if (v < 0)   
				v = 255;

			test.at<float>(y, x) = v;

			if (v < minval && v < 255)
				minval = v;

			x++;
		}
		
		y++;
		x = 0;

	}

	if (!infile.eof())
	{
		cerr << "No file!\n";
	}

	infile.close();
	cout << "minval = " << minval << endl;

	//test writing to file
	ofstream outfile;
	outfile.open("xxx2.png.txt");
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			outfile << depthdata.at<uint16_t>(j, i) << ",";
		}
		outfile << "\n";
	}
	outfile.close();

	cv::Mat bmat, cmat;
	test.convertTo(bmat, CV_8UC1);
	cv::cvtColor(bmat, cmat, CV_GRAY2BGR);

	//imshow("test", cmat);
	//imwrite("test.jpg", cmat);
	//waitKey(0);

	Mat thresh_img, orig, orig_gray;
	float threshold_value = minval + 30;
	cout << threshold_value << endl;
	int threshold_type = 0;
	int const max_value = 255;
	int const max_BINARY_value = 255;

	threshold(cmat, thresh_img, threshold_value, max_BINARY_value, threshold_type);
	imshow("test", thresh_img);
	waitKey(0);

	//if (argc > 1) {
	//	cout << argv[1] << endl;
	//	orig = imread(argv[1], 1);
	//	cvtColor(orig, orig_gray, CV_BGR2GRAY);
	//	threshold(test, thresh_img, threshold_value, max_BINARY_value, threshold_type);

	//	//applyColorMap(orig, orig, COLORMAP_HOT);
	//	imshow("Thresholded image", thresh_img);
	//}

    return 0;

}