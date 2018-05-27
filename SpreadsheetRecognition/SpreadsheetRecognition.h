#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <random>
#include <iostream>

#define DEBUG

struct SpreadsheetRecognitionParameters {
    SpreadsheetRecognitionParameters() :
        // All default parameters that SpreadsheetRecognition uses
        gaussianSize(2),
        cannyThreshold(50),
        cannyRatio(3),
        cannyApertureSize(5),
        houghThreshold(200),
        classifyLinesDeltaTheta(CV_PI / 180),
        probFilterLinesTryCount(1000),
        probFilterLinesExpectation(0.7),
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
            std::cout << line << std::endl;
    }

    static void drawLines(cv::Mat &img, const std::vector<cv::Vec2f> &lines, const cv::Scalar &color);
    static bool compVec2f(const cv::Vec2f &lhs, const cv::Vec2f &rhs);
    static cv::Point2d crossLines(const cv::Vec2f &line1, const cv::Vec2f &line2);
    static cv::Point2d crossWithHorizontalLine(const cv::Vec2f &line, const double y);
    static cv::Point2d crossWithVerticalLine(const cv::Vec2f &line, const double y);
    static bool isHorizontalLine(const double theta);
    static bool isVerticalLine(const double theta);

    void classifyLines(float deltaTheta);
    bool witnessHLine(int x, int y, int radius);
    bool witnessVLine(int x, int y, int radius);
    bool witnessPoint(int x, int y, int radius);
    bool isSpreadsheetHLine(const cv::Vec2f &line, int tryCount, double expectation);
    bool isSpreadsheetVLine(const cv::Vec2f &line, int tryCount, double expectation);
    void probFilterLines(int tryCount, double expectation);
    void clusterFilterLines(std::vector<cv::Vec2f> &lines, float deltaRho);

    cv::Mat mSrc, mSrcGray, resultView;
    std::vector<cv::Vec2f> mLines, mHLines, mVLines;
    static std::default_random_engine e;
};