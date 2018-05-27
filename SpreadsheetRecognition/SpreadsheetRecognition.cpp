#include <algorithm>
#include <sstream>
#include "SpreadsheetRecognition.h"
#include "Stopwatch.h"
using namespace cv;
using namespace std;

#define SCALAR_RED      Scalar(0, 0, 255)
#define SCALAR_BLUE     Scalar(255, 0, 0)
#define SCALAR_GREEN    Scalar(0, 255, 0)
#define SCALAR_YELLOW   Scalar(0, 255, 255)

default_random_engine SpreadsheetRecognition::e;

void SpreadsheetRecognition::execute(const SpreadsheetRecognitionParameters &para) {
    // preprocess
    Stopwatch timer;
    cvtColor(mSrc, mSrcGray, CV_RGB2GRAY);
    CV_Assert(mSrcGray.channels() == 1);
    int ks = 2 * para.gaussianSize + 1;
    GaussianBlur(mSrcGray, mSrcGray, Size(ks, ks), 0, 0);
    Canny(mSrcGray, mSrcGray, para.cannyThreshold, para.cannyThreshold * para.cannyRatio,
        para.cannyApertureSize, true);
    cout << "Preprocess time: " << timer << endl;
    cvtColor(mSrcGray, resultView, CV_GRAY2BGR);


    // detect lines
    timer.reset();
    HoughLines(mSrcGray, mLines, 1, CV_PI / 180, para.houghThreshold);
    cout << "HoughLines time: " << timer << endl;
    //drawLines(resultView, mLines, SCALAR_RED);


    // process lines
    timer.reset();
    classifyLines(para.classifyLinesDeltaTheta);
    cout << "classifyLines time: " << timer << endl;
    drawLines(resultView, mHLines, SCALAR_YELLOW);
    drawLines(resultView, mVLines, SCALAR_YELLOW);

    timer.reset();
    probFilterLines(para.probFilterLinesTryCount, para.probFilterLinesExpectation);
    cout << "probFilterLines time: " << timer << endl;
    drawLines(resultView, mHLines, SCALAR_BLUE);
    drawLines(resultView, mVLines, SCALAR_BLUE);

    timer.reset();
    clusterFilterLines(mHLines, para.clusterFilterLinesDeltaRho);
    clusterFilterLines(mVLines, para.clusterFilterLinesDeltaRho);
    cout << "clusterFilterLines time: " << timer << endl;
    drawLines(resultView, mHLines, SCALAR_GREEN);
    drawLines(resultView, mVLines, SCALAR_GREEN);
}

void SpreadsheetRecognition::showResult(const String &windowName) {
    imshow(windowName, resultView);
}

void SpreadsheetRecognition::output(const string &dir) {
    vector<vector<Rect>> spreadsheet;
    vector<int> xs, ys;
    for (auto &line : mHLines)
        ys.push_back((int)line[0]);
    for (auto &line : mVLines)
        xs.push_back((int)line[0]);
    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());

    for (size_t i = 0; i < ys.size() - 1; ++i) {
        vector<Rect> row;
        for (size_t j = 0; j < xs.size() - 1; ++j)
            row.push_back(Rect(xs[j], ys[i], xs[j + 1] - xs[j], ys[i + 1] - ys[i]));
        spreadsheet.push_back(row);
    }

    for (size_t i = 0; i < spreadsheet.size(); ++i) {
        for (size_t j = 0; j < spreadsheet[i].size(); ++j) {
            ostringstream oss;
            oss << dir << "/output/" << i << "_" << j << ".jpg";
            imwrite(oss.str(), mSrc(spreadsheet[i][j]));
        }
    }
}

void SpreadsheetRecognition::drawLines(Mat &img, const vector<Vec2f> &lines, const Scalar &color) {
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double cosTheta = cos(theta), sinTheta = sin(theta);
        double x0 = rho * cosTheta, y0 = rho * sinTheta;
        pt1.x = cvRound(x0 + 1000 * (-sinTheta));
        pt1.y = cvRound(y0 + 1000 * (cosTheta));
        pt2.x = cvRound(x0 - 1000 * (-sinTheta));
        pt2.y = cvRound(y0 - 1000 * (cosTheta));
        line(img, pt1, pt2, color, 1, CV_AA);
    }
}

bool SpreadsheetRecognition::compVec2f(const Vec2f &lhs, const Vec2f &rhs) {
    return (lhs[0] < rhs[0]) || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
}

bool SpreadsheetRecognition::isHorizontalLine(const double theta, const double deltaTheta) {
    return fabs(theta - CV_PI / 2) <= deltaTheta;
}

bool SpreadsheetRecognition::isVerticalLine(const double theta, const double deltaTheta) {
    return fabs(theta) <= deltaTheta;
}

void SpreadsheetRecognition::classifyLines(float deltaTheta) {
    mHLines.clear();
    mVLines.clear();
    for (auto &line : mLines) {
        float rho = line[0], theta = line[1];
        if (isHorizontalLine(theta, deltaTheta) && rho >= 0 && rho < mSrcGray.rows)
            mHLines.push_back(line);
        else if (isVerticalLine(theta, deltaTheta) && rho >= 0 && rho < mSrcGray.cols)
            mVLines.push_back(line);
    }
    std::sort(mHLines.begin(), mHLines.end(), compVec2f);
    std::sort(mVLines.begin(), mVLines.end(), compVec2f);
}

Point2d SpreadsheetRecognition::crossLines(const Vec2f &line1, const Vec2f &line2) {
    double rho1 = line1[0], rho2 = line2[0];
    double theta1 = line1[1], theta2 = line2[1];
    double cosTheta1 = cos(theta1), cosTheta2 = cos(theta2);
    double sinTheta1 = sin(theta1), sinTheta2 = sin(theta2);
    double denominator = cosTheta1 * sinTheta2 - sinTheta1 * cosTheta2;
    double x = (rho1 * sinTheta2 - rho2 * sinTheta1) / denominator;
    double y = (rho2 * cosTheta1 - rho1 * cosTheta2) / denominator;
    return Point2d(x, y);
}

Point2d SpreadsheetRecognition::crossWithHorizontalLine(const Vec2f &line, const double y) {
    double rho = line[0], theta = line[1];
    CV_Assert(!isHorizontalLine(theta));
    double cosTheta = cos(theta), sinTheta = sin(theta);
    double x = (rho - y * sinTheta) / cosTheta;
    return Point2d(x, y);
}

Point2d SpreadsheetRecognition::crossWithVerticalLine(const Vec2f &line, const double x) {
    double rho = line[0], theta = line[1];
    CV_Assert(!isVerticalLine(theta));
    double cosTheta = cos(theta), sinTheta = sin(theta);
    double y = (rho - x * cosTheta) / sinTheta;
    return Point2d(x, y);
}

bool SpreadsheetRecognition::witnessHLine(int x, int y, int radius) {
    int rBegin = std::max(y - radius, 0);
    int rEnd = std::min(y + radius + 1, mSrcGray.rows);
    int c = std::min(std::max(x, 0), mSrcGray.cols);
    for (int r = rBegin; r < rEnd; ++r) 
        if ((uint)mSrcGray.at<uchar>(r, c) > 200)
            return true;
    return false;
}

bool SpreadsheetRecognition::witnessVLine(int x, int y, int radius) {
    int cBegin = std::max(x - radius, 0);
    int cEnd = std::min(x + radius + 1, mSrcGray.cols);
    int r = std::min(std::max(y, 0), mSrcGray.rows);
    for (int c = cBegin; c < cEnd; ++c) 
        if ((uint)mSrcGray.at<uchar>(r, c) > 200)
            return true;
    return false;
}

bool SpreadsheetRecognition::witnessPoint(int x, int y, int radius) {
    int cBegin = std::max(x - radius, 0);
    int cEnd = std::min(x + radius + 1, mSrcGray.cols);
    int rBegin = std::max(y - radius, 0);
    int rEnd = std::min(y + radius + 1, mSrcGray.rows);
    for (int r = rBegin; r < rEnd; ++r)
        for (int c = cBegin; c < cEnd; ++c)
            if ((uint)mSrcGray.at<uchar>(r, c) > 200)
                return true;
    return false;
}

bool SpreadsheetRecognition::isSpreadsheetHLine(const Vec2f &line, int tryCount, double expectation) {
    int count = 0;
    if (isHorizontalLine(line[1])) {
        // process totally horizontal line, which is faster
        uniform_int_distribution<int> g((int)mVLines.front()[0], (int)mVLines.back()[0]);
        int y = cvRound(line[0]);
        for (int i = 0; i < tryCount; ++i) 
            if (witnessHLine(g(e), y, 1))
                count++;
    }
    else {
        // process oblique line
        static uniform_real_distribution<double> g(0.0, 1.0);
        Point2d p1 = crossWithVerticalLine(line, mVLines.front()[0]);
        Point2d p2 = crossWithVerticalLine(line, mVLines.back()[0]);
        for (int i = 0; i < tryCount; ++i) {
            double lambda = g(e);
            Point2d randomPoint = (p1 + lambda * p2) / (1 + lambda);
            if (witnessPoint(cvRound(randomPoint.x), cvRound(randomPoint.y), 1))
                count++;
        }
    }
    // cout << count << endl;
    return ((double)count / tryCount >= expectation);
}

bool SpreadsheetRecognition::isSpreadsheetVLine(const Vec2f &line, int tryCount, double expectation) {
    int count = 0;

    if (isVerticalLine(line[1])) {
        // process totally vertical line, which is faster
        uniform_int_distribution<int> g((int)mHLines.front()[0], (int)mHLines.back()[0]);
        int x = cvRound(line[0]);
        for (int i = 0; i < tryCount; ++i) 
            if (witnessVLine(x, g(e), 1))
                count++;
    }
    else {
        // process oblique line
        static uniform_real_distribution<double> g(0.0, 1.0);
        Point2d p1 = crossWithHorizontalLine(line, mHLines.front()[0]);
        Point2d p2 = crossWithHorizontalLine(line, mHLines.back()[0]);
        for (int i = 0; i < tryCount; ++i) {
            double lambda = g(e);
            Point2d randomPoint = (p1 + lambda * p2) / (1 + lambda);
            if (witnessPoint(cvRound(randomPoint.x), cvRound(randomPoint.y), 1))
                count++;
        }
    }
    // cout << count << endl;
    return ((double)count / tryCount >= expectation);
}

void SpreadsheetRecognition::probFilterLines(int tryCount, double expectation) {
    vector<Vec2f> filteredHLines, filteredVLines;
    for (auto &line : mHLines)
        if (isSpreadsheetHLine(line, tryCount, expectation))
            filteredHLines.push_back(line);
    for (auto &line : mVLines)
        if (isSpreadsheetVLine(line, tryCount, expectation))
            filteredVLines.push_back(line);
    mHLines = filteredHLines;
    mVLines = filteredVLines;
}

void SpreadsheetRecognition::clusterFilterLines(vector<Vec2f> &lines, float deltaRho) {
    vector<Vec3f> clusterLines;
    for (auto &line : lines) {
        float rho = line[0], theta = line[1];
        bool added = false;
        for (auto &cluster : clusterLines) {
            float crho = cluster[0], ctheta = cluster[1], ctr = cluster[2];
            if (fabs(rho - crho) < deltaRho) {
                cluster[0] = (crho * ctr + rho) / (ctr + 1);
                cluster[1] = (ctheta * ctr + theta) / (ctr + 1);
                cluster[2] = ctr + 1;
                added = true;
                break;
            }
        }
        if (!added)
            clusterLines.push_back(Vec3f(rho, theta, 1.0));
    }

    lines.clear();
    for (auto &line : clusterLines)
        lines.push_back(Vec2f(line[0], line[1]));
}
