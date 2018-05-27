#include <iostream>
#include <string>
#include "SpreadsheetRecognition.h"
#include "Stopwatch.h"
using namespace cv;
using namespace std;

// #define DEBUG

string windowName = "Edge Detect";
SpreadsheetRecognitionParameters para;
Mat src;

void onChange(int, void *) {
    SpreadsheetRecognition SR(src);
    Stopwatch timer;
    SR.execute(para);
    cout << "Total: " <<timer << endl;
    SR.showResult(windowName);
    //SR.output(".");
}

int main(int argc, char** argv) {
    string filename;
    cout << "Input file:" << endl;
    cin >> filename;

    src = imread(filename);
    if (!src.data)
        CV_Assert(0);

    namedWindow(windowName, CV_WINDOW_AUTOSIZE);
#ifdef DEBUG
    createTrackbar("Gauss:", windowName, &para.gaussianSize, para.maxGaussianSize, onChange);
    createTrackbar("Canny:", windowName, &para.cannyThreshold, para.maxCannyThreshold, onChange);
    createTrackbar("Hough:", windowName, &para.houghThreshold, para.maxHoughThreshold, onChange);
#endif

    onChange(0, 0);

    waitKey(0);
    return 0;
}


