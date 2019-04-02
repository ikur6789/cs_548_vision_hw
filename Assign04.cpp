#include <opencv2/core/core.hpp>
#include <opencv2/shape.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "LBP.hpp"
#include "TEST_LBP.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

	// If no command line arguments entered, exit
	if (argc < 3) {
		cout << "ERROR: Insufficient command line arguments!" << endl;
		cout << "[path to image] [path to output CSV file] [testing? (0 or 1)]" << endl;
		return -1;
	}

	// Testing mode?
	int testingMode = atoi(argv[3]);

	// Get path to output histogram file
	string outputHistFile = string(argv[2]);

	// Get path to input image
	string filepath = string(argv[1]);

	if (!testingMode) {
		// REGULAR APPLICATION

		// Save histogram		
		ImageHist ihist = extractLBP(filepath);
		saveHistogram(outputHistFile, ihist);
	}
	else {
		// TESTING MODE

		bool allGood = true;
		
		if (!TEST_getPixel()) {
			cout << "TEST_getPixel: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_getPixel: SUCCESS" << endl;
		}

		if (!TEST_getLBPNeighbors()) {
			cout << "TEST_getLBPNeighbors: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_getLBPNeighbors: SUCCESS" << endl;
		}

		if (!TEST_thresholdArray()) {
			cout << "TEST_thresholdArray: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_thresholdArray: SUCCESS" << endl;
		}

		if (!TEST_getUniformLabel()) {
			cout << "TEST_getUniformLabel: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_getUniformLabel: SUCCESS" << endl;
		}
				
		if (!TEST_LBP(filepath)) {
			cout << "TEST_LBP: FAILED" << endl;
			allGood = false;
		}
		else {
			cout << "TEST_LBP: SUCCESS" << endl;
		}

		// All tests good?
		if (allGood) {
			cout << "ALL TESTS SUCCEED!" << endl;
		}
	}

	return 0;
}
