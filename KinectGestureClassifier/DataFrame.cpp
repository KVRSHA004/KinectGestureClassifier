#include "stdafx.h"
#include "DataFrame.h"
#include <iostream>
#include <fstream>
#include <ostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace HANDGR;

DataFrame::DataFrame() 
// initialise for standard Kinectv2 depth frame
{
	//frame_width = 512;
	//frame_height = 424;
	//raw_distance_data = Mat::zeros(frame_height, frame_width, CV_16UC1);
	//frame_data = Mat::zeros(frame_height, frame_width, CV_32F);
}

//DataFrame::DataFrame(int rows, int columns)
//// initialise frame with specified height/width
//{
//	frame_width = columns;
//	frame_height = rows;
//	raw_distance_data = Mat::zeros(frame_height, frame_width, CV_16UC1);
//	frame_data = Mat::zeros(frame_height, frame_width, CV_32F);
//}

//DataFrame::DataFrame(std::string filename)
//// initialise for standard Kinectv2 depth frame
//{
//	frame_width = 512;
//	frame_height = 424;
//	raw_distance_data = Mat::zeros(frame_height, frame_width, CV_16UC1);
//	frame_data = imread(filename);
//}

DataFrame::~DataFrame()
{
	frame.release();
}

//bool DataFrame::readDataFromFile(std::string filename)
//{
//	//counters for loops / (y,x) coordinates for pixels
//	int x = 0, y = 0;
//	const float depth_near = 500, depth_far = 1500;
//	const float alpha = 255.0f/(depth_far - depth_near);
//	const float beta = -depth_near * alpha;
//
//	min_depth = 255;
//
//	fstream infile(filename);
//	while (infile) //each row
//	{
//		string line_string;
//		if (!getline(infile, line_string))
//			break;
//		istringstream line_ss(line_string);
//
//		while (line_ss) //each column
//		{
//			string distance;
//			if (!getline(line_ss, distance, ','))
//				break;
//
//			//store original distance
//			raw_distance_data.at<uint16_t>(y, x) = (int)stoi(distance);
//			
//			//modify depth data for display purposes
//			float modified_dist = (float)stoi(distance) * alpha + beta;
//			if (modified_dist > 255 || modified_dist < 0)
//				modified_dist = 255;
//
//			//store modified display depth distance
//			frame_data.at<float>(y, x) = modified_dist;
//			
//			//find minimum value of depth in frame
//			//ignore shadow (value 255)
//			if (modified_dist < min_depth && modified_dist < 255)
//				min_depth = modified_dist; 
//
//			x++;
//		}
//		y++;
//		x = 0;
//	}
//	if (!infile.eof())
//	{
//		cerr << "No file named " + filename + "\n";
//		return false;
//	}
//	infile.close();
//	
//	return true;
//}

//broken method???
//bool HANDGR::DataFrame::readDataFromImage(std::string imagename)
//{
//	cv::Mat input_image = imread(imagename, CV_32F);
//	//input_image.convertTo(input_image, CV_BGR2GRAY);
//	imshow("kkk", input_image);
//	min_depth = 255;
//
//	if (frame_height == input_image.rows && frame_width == input_image.cols) {
//		for (int j = 0; j < frame_height; j++) {
//			for (int i = 0; i < frame_width; i++) {
//
//				float modified_dist = input_image.at<float>(j, i);
//
//				if (modified_dist > 255 || modified_dist < 0)
//					modified_dist = 255;
//
//				//store modified display depth distance
//				frame_data.at<float>(j, i) = modified_dist;
//
//				//find minimum value of depth in frame
//				//ignore shadow (value 255)
//				if (modified_dist < min_depth && modified_dist < 255)
//					min_depth = modified_dist;
//			}
//		}
//		waitKey(0);
//		return true;
//	}
//	else
//		return false;
//
//}

//for testing
//bool DataFrame::writeFileFromDataFrame(std::string filename)
//{
//	ofstream outfile;
//	outfile.open(filename);
//	if (outfile.is_open()) {
//		for (int j = 0; j < frame_height; j++) {
//			for (int i = 0; i < frame_width; i++) {
//				outfile << raw_distance_data.at<uint16_t>(j, i) << ",";
//			}
//			outfile << "\n";
//		}
//		outfile.close();
//		return true;
//	}
//	else
//		return false;
//}

//void DataFrame::saveFrame(std::string filename)
//{
//	cv::Mat bmat, cmat;
//	frame.convertTo(bmat, CV_8UC1);
//	cv::cvtColor(bmat, cmat, CV_GRAY2BGR);
//	imwrite(filename, cmat);
//}

cv::Mat DataFrame::applyContourMask()
{
	//cv::Mat bmat, cmat;
	//cv::Mat input_image = imread(filename);
	cv::Mat contoured_img;
	//frame_data.convertTo(bmat, CV_8UC1);
	//cv::cvtColor(input_image, input_image, CV_GRAY2BGR);

	//float threshold_value = min_depth + thresh_limit;
	//int threshold_type = 0;
	//int const max_value = 255;
	//int const max_BINARY_value = 255;
	//threshold(cmat, thresh_img, threshold_value, max_BINARY_value, threshold_type);

	Canny(frame, contoured_img, 100, 150);
	// find the contours
	vector<vector<Point>> contours;
	findContours(contoured_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	Mat mask = Mat::zeros(contoured_img.rows, contoured_img.cols, CV_8UC1);

	// fill the connected components found
	drawContours(mask, contours, -1, Scalar(255), CV_FILLED);
	//imshow("drawcontour", mask);

	// only draw largest connected component
	vector<double> areas(contours.size());
	int largest_contour_index = 0;
	int largest_area = 0;
	Rect bounding_rect;
	for (int i = 0; i< contours.size(); i++)
	{
		//  Find the area of contour
		double a = contourArea(contours[i], false);
		if (a > largest_area) {
			// Store the index of largest contour
			largest_contour_index = i;
			// Find the bounding rectangle for biggest contour
			bounding_rect = boundingRect(contours[i]);
		}
	}
	cvtColor(mask, mask, CV_GRAY2BGR);
	drawContours(mask, contours, largest_contour_index, Scalar(255, 255, 255), CV_FILLED, 8);
	//find rectangle surrounding hand/roi
	rectangle(mask, bounding_rect, 0);
	//imshow("contour", mask);
	//crop roi
	Mat roi = mask(bounding_rect);
	//cout << roi.rows << " " << roi.cols << endl;
	/*imshow("rectangle", roi);
	waitKey(0);*/
	
	return roi;
}

cv::Mat DataFrame::resizeKeepAspectRatio(const cv::Mat &input, int width, int height)
{
	cv::Mat resized;

	double h1 = width * (input.rows/(double)input.cols);
	double w2 = height * (input.cols/(double)input.rows);
	if (h1 <= height) {
		cv::resize(input, resized, cv::Size(width, h1));
	}
	else {
		cv::resize(input, resized, cv::Size(w2, height));
	}

	int top = (height - resized.rows) / 2;
	int down = (height - resized.rows + 1) / 2;
	int left = (width - resized.cols) / 2;
	int right = (width - resized.cols + 1) / 2;

	cv::copyMakeBorder(resized, resized, top, down, left, right, cv::BORDER_CONSTANT, Scalar(0));

	//imshow("resized", resized);

	return resized;
}

bool HANDGR::DataFrame::loadFrame(string filename)
{
	frame = imread(filename, IMREAD_UNCHANGED);
	cvtColor(frame, frame, CV_BGR2GRAY);
	//imshow("crop", frame);
	if (countNonZero(frame) > 0)
		return true;
	else
		return false;
}

