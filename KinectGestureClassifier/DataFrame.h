#pragma once
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace HANDGR {

	class DataFrame
	{
	public:
		//cv::Mat raw_distance_data;
		cv::Mat frame;
		//int frame_height, frame_width;
		//float min_depth;

		DataFrame();
		//DataFrame(int rows, int columns);
		//DataFrame(std::string filename);
		~DataFrame();
		//bool readDataFromFile(std::string filename);
		//bool readDataFromImage(std::string imagename);
		//bool writeFileFromDataFrame(std::string filename);
		//void saveFrame(std::string filename);
		cv::Mat applyContourMask();

		static cv::Mat resizeKeepAspectRatio(const cv::Mat & input, int width, int height);
		bool loadFrame(std::string filename);
	};

}
