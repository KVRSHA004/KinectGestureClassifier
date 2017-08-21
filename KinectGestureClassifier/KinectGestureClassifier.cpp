// KinectGestureClassifier.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "DataFrame.h"
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/progress.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2\ml\ml.hpp>

using namespace cv;
using namespace std;
using namespace HANDGR;
using namespace boost::filesystem;


void readImages(int& valid, int& invalid, vector<string>::const_iterator begin, vector<string>::const_iterator end, std::function<void(const std::string&, const cv::Mat&)> callback);
void getFilesInDirectory(const string& directory, vector<string>& filelist);
string getClassName(string filename);
cv::Mat getImage(string filename);
int getClassId(const std::set<std::string>& classes, const std::string& classname);
int getPredictedClass(const cv::Mat& predictions);
bool getFirstValidFrame(int& valid, int& invalid, const string& directory, string& validfile, cv::Mat& validframe);

void readImages(int & valid, int & invalid, vector<string>::const_iterator begin, vector<string>::const_iterator end, std::function<void(const std::string&, const cv::Mat&)> callback)
{
	for (auto it = begin; it != end; ++it)
	{
		string gesture_folder = *it;
		string temp;
		cv::Mat frame;
		string classname = getClassName(gesture_folder);
		if (getFirstValidFrame(valid, invalid, gesture_folder, temp, frame))
			callback(classname, frame);
	}
}

struct ImageData
{
	std::string classname;
	cv::Mat image;
};

cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname) {
	cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
	int index = 0;
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		if (*it == classname) break;
		++index;
	}
	code.at<float>(index) = 1;
	return code;
};

int main(int argc, char** argv)
{

	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, 80, "%d%m%y %H%M%S", timeinfo);
	puts(buffer);
	std::string datetimestr = buffer;
	
	std::ofstream logfile;
	logfile.open("SASLKinectClassification " + datetimestr);

	std::cout << "Reading training set..." << std::endl;
	double start = (double)cv::getTickCount();
	string fileDirectory = argv[1];
	float trainSplitRatio = 0.75;
	logfile << "Data containing directory set to: " << fileDirectory << endl;
	logfile << "Train-Split ratio set to: " << trainSplitRatio << endl;
	logfile << endl;
	logfile << "Reading training set..." << endl;

	vector<string> files;
	getFilesInDirectory(fileDirectory, files);
	random_shuffle(files.begin(), files.end());


	std::vector<ImageData*> imageMetadata;
	std::set<std::string> classes;
	int validgestures = 0, invalidgestures = 0;
	readImages(validgestures, invalidgestures, files.begin(), files.begin() + (size_t)(files.size() * trainSplitRatio),
		[&](const std::string& classname, const cv::Mat& img) {
			// Append to the set of classes
			classes.insert(classname);
			// Append metadata to each extracted feature
			ImageData* data = new ImageData;
			data->classname = classname;
			Mat1b frame = img.reshape(1, 1);
			data->image = frame;
			cout << data->image.rows << " " << data->image.cols << endl;
			imageMetadata.push_back(data);
		});
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
	logfile << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
	logfile << "Valid gestures read: " << validgestures << endl;
	logfile << "Invalid gestures not read: " << invalidgestures << endl;
	logfile << endl;


	std::cout << "Preparing neural network..." << std::endl;
	logfile << "Preparing neural network..." << endl;
	cv::Mat trainSamples;
	cv::Mat trainResponses;
	std::set<ImageData*> uniqueMetadata(imageMetadata.begin(), imageMetadata.end());
	int noTrainingSamples = uniqueMetadata.size();
	for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end();)
	{
		ImageData* data = *it;
		trainSamples.push_back(data->image);
		trainResponses.push_back(getClassCode(classes, data->classname));
		delete *it; // clear memory
		it++;
	}
	imageMetadata.clear();

	std::cout << "Training neural network..." << std::endl;
	logfile << "Training neural network..." << endl;
	start = cv::getTickCount();

	int networkInputSize = 625*3; //trainSamples.cols;
	int networkOutputSize = trainResponses.cols;
	cout << networkInputSize << " " << networkOutputSize << endl;
	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
	std::vector<int> layerSizes = { networkInputSize, networkInputSize/2, networkOutputSize };

	logfile << "Setting layer sizes: " << networkInputSize;
	for (int i = 1; i < layerSizes.size(); ++i)
		logfile << layerSizes[i] << ", ";
	logfile << endl;

	mlp->setLayerSizes(layerSizes);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);

	logfile << "Using ANN_MLP::SIGMOID_SYM Activation function..." << endl;

	// trainResponses.reshape(1, noTrainingSamples);
	cout << trainResponses.rows << " " << trainResponses.cols << endl;
	cout << trainSamples.rows << " " << trainSamples.cols << endl;
	cout << noTrainingSamples << endl;
	trainSamples.convertTo(trainSamples, CV_32F);
	mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
	
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
	logfile << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
	logfile << endl;

	// Clear memory now 
	trainSamples.release();
	trainResponses.release();

	//read in test set
	std::cout << "Reading test set..." << std::endl;
	logfile << "Reading test set..." << std::endl;
	start = cv::getTickCount();
	cv::Mat testSamples;
	std::vector<int> testOutputExpected;
	validgestures = 0;
	invalidgestures = 0;
	readImages(validgestures, invalidgestures, files.begin() + (size_t)(files.size() * trainSplitRatio), files.end(),
		[&](const std::string& classname, const cv::Mat& img) {
		Mat1b frame = img.reshape(1, 1);
		testSamples.push_back(frame);
		testOutputExpected.push_back(getClassId(classes, classname));
	});
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
	logfile << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
	logfile << "Valid gestures read: " << validgestures << endl;
	logfile << "Invalid gestures not read: " << invalidgestures << endl;
	logfile << endl;

	testSamples.convertTo(testSamples, CV_32F);

	//feed test data into mlp to predict
	std::cout << "Predicting on test set..." << std::endl;
	logfile << "Predicting on test set..." << std::endl;
	start = cv::getTickCount();
	cv::Mat testOutput;
	mlp->predict(testSamples, testOutput);
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
	logfile << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;
	logfile << endl;
	testSamples.release();

	//calculate confusion matrix
	std::vector<std::vector<int> > confusionMatrix(26, std::vector<int>(26));
	for (int i = 0; i < testOutput.rows; i++)
	{
		int predictedClass = getPredictedClass(testOutput.row(i));
		int expectedClass = testOutputExpected.at(i);
		confusionMatrix[expectedClass][predictedClass]++;
	}
	int hits = 0;
	int total = 0;
	for (size_t i = 0; i < confusionMatrix.size(); i++)
	{
		for (size_t j = 0; j < confusionMatrix.at(i).size(); j++)
		{
			if (i == j) hits += confusionMatrix.at(i).at(j);
			total += confusionMatrix.at(i).at(j);
		}
	}

	logfile << "Confusion matrix: " << endl;
	//print confusion matrix
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		std::cout << *it << " ";
		logfile << *it << " ";
	}
	std::cout << std::endl;
	logfile << endl;
	for (size_t i = 0; i < confusionMatrix.size(); i++)
	{
		for (size_t j = 0; j < confusionMatrix[i].size(); j++)
		{
			std::cout << confusionMatrix[i][j] << " ";
			logfile << confusionMatrix[i][j] << " ";
		}
		std::cout << std::endl;
		logfile << endl;
	}
	logfile << endl;

	cout << "Classifier accuracy rate: " << hits * 100 / (float)total << "%" << endl;
	logfile << "Classifier accuracy rate: " << hits * 100 / (float)total << "%" << endl << endl;
	logfile.close();

	//Processing individual frame

	//DataFrame frame(424, 512);
	//cv::Mat crop = frame.applyContourMask("3 .png");
	////frame.saveFrame("test.jpg");    
	////cv::Mat crop = frame.thresholdDataAndFindHand(25);
	//cv::Mat resized = DataFrame::resizeKeepAspectRatio(crop, 50, 50);
	//imshow("", resized);
	waitKey(0);

    return 0;

}

void getFilesInDirectory(const string& directory, vector<string>& filelist)
{
	path root(directory);
	directory_iterator it_end;
	
	for (directory_iterator it(root); it != it_end; ++it)
	{
		if (is_regular_file(it->path()))
		{
			//filelist.push_back(it->path().string());
		}
		if (is_directory(it->path()))
		{
			//getFilesInDirectory(it->path().string(), filelist);
			 
			filelist.push_back(it->path().string());
		}
	}
}

bool getFirstValidFrame(int& valid, int& invalid, const string& directory, string& validfile, cv::Mat& validframe)
{
	//assume directory of a gesture performance
	path root(directory);
	directory_iterator it_end;
	bool found_valid;

	for (directory_iterator it(root); it != it_end; ++it)
	{
		if (is_regular_file(it->path()))
		{
			std::string filename = it->path().string();
			DataFrame img;
			if (img.loadFrame(filename)) {
				std::cout << "Reading gesture " << filename << "..." << std::endl;
				std::string classname = getClassName(filename);
				cv::Mat frame = img.applyContourMask();
				cout << frame.rows << " " << frame.cols << endl;
				if ((frame.rows < 25) || (frame.cols < 25)) {
					cout << "Invalid frame [" + filename + "]" << endl;
				}
				else {
					++valid;
					std::cout << "VALID FRAME FOUND: " << filename << "..." << std::endl;
					validframe = DataFrame::resizeKeepAspectRatio(frame, 25, 25);
					return true;
				}

			}
		}
		cout << "Invalid gesture [" + directory + "]" << endl;
		++invalid;
		return false;
	}
}

string getClassName(string filename)
{
	std::vector<std::string> results;
	boost::split(results, filename, [](char c) {return c == '\\'; });
	//HACKY way to get gesture name
	//Need to find another way to do this
	return results[8].substr(0, 1);
}

cv::Mat getImage(string filename)
{
	DataFrame frame;
	cv::Mat crop = frame.applyContourMask();
	cv::Mat resized = DataFrame::resizeKeepAspectRatio(crop, 25, 25);
	return resized;
}

/**
* Transform a class name into an id
*/
int getClassId(const std::set<std::string>& classes, const std::string& classname)
{
	int index = 0;
	for (auto it = classes.begin(); it != classes.end(); ++it)
	{
		if (*it == classname) break;
		++index;
	}
	return index;
}

/**
* Receives a column matrix contained the probabilities associated to
* each class and returns the id of column which contains the highest
* probability
*/
int getPredictedClass(const cv::Mat& predictions)
{
	float maxPrediction = predictions.at<float>(0);
	float maxPredictionIndex = 0;
	const float* ptrPredictions = predictions.ptr<float>(0);
	for (int i = 0; i < predictions.cols; i++)
	{
		float prediction = *ptrPredictions++;
		if (prediction > maxPrediction)
		{
			maxPrediction = prediction;
			maxPredictionIndex = i;
		}
	}
	return maxPredictionIndex;
}
