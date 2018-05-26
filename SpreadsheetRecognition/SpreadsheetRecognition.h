#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <random>

#define DEBUG

struct SpreadsheetRecognitionParameters {
    SpreadsheetRecognitionParameters() :
        gaussianSize(3),
        cannyThreshold(40),
        cannyRatio(3),
        cannyApertureSize(5),
        houghThreshold(200),
        classifyLinesDeltaTheta(CV_PI / 180),
        probFilterLinesTryCount(200),
        probFilterLinesExpectation(0.5),
        clusterFilterLinesDeltaRho(10.0F) {}

    int gaussianSize;
    int cannyThreshold, cannyRatio, cannyApertureSize;
    int houghThreshold;
    float classifyLinesDeltaTheta;
    int probFilterLinesTryCount;
    double probFilterLinesExpectation;
    float clusterFilterLinesDeltaRho;

    static const int maxGaussianSize = 50;
    static const int maxCannyThreshold = 100;
    static const int maxHoughThreshold = 500;
};

class SpreadsheetRecognition {
public:
    SpreadsheetRecognition(cv::Mat src) : mSrc(src) {}

    void execute(const SpreadsheetRecognitionParameters &para);
    void showResult(const cv::String &windowName);
    void output(const std::string &dir);
    
private:
    template<class Ty>
    void showlines(Ty &lines) {
        for (auto &line : lines)
            cout << line << endl;
    }

    static void drawLines(cv::Mat &img, std::vector<cv::Vec2f> &lines, const cv::Scalar &color);
    static bool compVec2f(const cv::Vec2f &lhs, const cv::Vec2f &rhs);
    void classifyLines(float deltaTheta);
    bool witnessHLine(int x, int y, int radius);
    bool isHLine(cv::Vec2f &line, int tryCount, double expectation);
    void probFilterHLines(int tryCount, double expectation);
    void clusterFilterLines(std::vector<cv::Vec2f> &lines, float deltaRho);

    cv::Mat mSrc, mSrcGray, resultView;
    std::vector<cv::Vec2f> mLines, mHLines, mVLines;
    static std::default_random_engine e;
};