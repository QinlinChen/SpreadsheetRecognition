#include <algorithm>
#include <sstream>
#include <iostream>
#include "SpreadsheetRecognition.h"
using cv::Mat;
using cv::String;
using cv::Vec2f;
using cv::Vec3f;
using cv::Rect;
using cv::Scalar;
using cv::Point;
using cv::Point2f;
using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::ostringstream;
using std::default_random_engine;
using std::uniform_int_distribution;

default_random_engine SpreadsheetRecognition::e;

void SpreadsheetRecognition::execute(const SpreadsheetRecognitionParameters &para) {
    // preprocess
    cv::cvtColor(mSrc, mSrcGray, CV_RGB2GRAY);
    int ks = 2 * para.gaussianSize + 1;
    cv::GaussianBlur(mSrcGray, mSrcGray, cv::Size(ks, ks), 0, 0);
    cv::Canny(mSrcGray, mSrcGray, para.cannyThreshold, para.cannyThreshold * para.cannyRatio,
        para.cannyApertureSize, true);
#ifdef DEBUG
    cv::cvtColor(mSrcGray, resultView, CV_GRAY2BGR);
#endif

    // detect lines
    cv::HoughLines(mSrcGray, mLines, 1, CV_PI / 180, para.houghThreshold);
#ifdef DEBUG
    drawLines(resultView, mLines, Scalar(0, 0, 255));
#endif

    // process lines
    classifyLines(para.classifyLinesDeltaTheta);
#ifdef DEBUG
    drawLines(resultView, mHLines, Scalar(0, 255, 255));
    drawLines(resultView, mVLines, Scalar(0, 255, 255));
#endif
    probFilterLines(para.probFilterLinesTryCount, para.probFilterLinesExpectation);
#ifdef DEBUG
    drawLines(resultView, mHLines, Scalar(255, 0, 0));
    drawLines(resultView, mVLines, Scalar(255, 0, 0));
#endif
    clusterFilterLines(mHLines, para.clusterFilterLinesDeltaRho);
    clusterFilterLines(mVLines, para.clusterFilterLinesDeltaRho);
    drawLines(resultView, mHLines, Scalar(0, 255, 0));
    drawLines(resultView, mVLines, Scalar(0, 255, 0));
}

void SpreadsheetRecognition::showResult(const String &windowName) {
    cv::imshow(windowName, resultView);
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
            cv::imwrite(oss.str(), mSrc(spreadsheet[i][j]));
        }
    }

}

void SpreadsheetRecognition::drawLines(Mat &img, vector<Vec2f> &lines, const Scalar &color) {
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double costheta = cos(theta), sintheta = sin(theta);
        double x0 = rho * costheta, y0 = rho * sintheta;
        pt1.x = cvRound(x0 + 1000 * (-sintheta));
        pt1.y = cvRound(y0 + 1000 * (costheta));
        pt2.x = cvRound(x0 - 1000 * (-sintheta));
        pt2.y = cvRound(y0 - 1000 * (costheta));
        cv::line(img, pt1, pt2, color, 1, CV_AA);
    }
}

bool SpreadsheetRecognition::compVec2f(const Vec2f &lhs, const Vec2f &rhs) {
    if (lhs[0] < rhs[0])
        return true;
    if (lhs[0] == rhs[0] && lhs[1] < rhs[1])
        return true;
    return false;
}

void SpreadsheetRecognition::classifyLines(float deltaTheta) {
    const float hTheta = CV_PI / 2.0, vTheta = 0.0;

    mHLines.clear();
    mVLines.clear();
    for (auto &line : mLines) {
        float rho = line[0], theta = line[1];
        if (fabs(theta - hTheta) <= deltaTheta && rho >= 0 && rho < mSrcGray.rows)
            mHLines.push_back(line);
        else if (fabs(theta - vTheta) <= deltaTheta && rho >= 0 && rho < mSrcGray.cols)
            mVLines.push_back(line);
    }
    std::sort(mHLines.begin(), mHLines.end(), compVec2f);
    std::sort(mVLines.begin(), mVLines.end(), compVec2f);
}

bool SpreadsheetRecognition::witnessHLine(int x, int y, int radius) {
    int begin = std::max(y - radius, 0);
    int end = std::min(y + radius, mSrcGray.rows - 1);
    for (int iy = begin; iy <= end; ++iy) {
        if ((uint)mSrcGray.at<uchar>(iy, x) > 200)
            return true;
    }
    return false;
}

bool SpreadsheetRecognition::isHLine(Vec2f &line, int tryCount, double expectation) {
    CV_Assert(mSrcGray.channels() == 1);
    int y = (int)line[0];
    int count = 0;
    uniform_int_distribution<int> g((int)mVLines.front()[0], (int)mVLines.back()[0]);

    for (int i = 0; i < tryCount; ++i) {
        int x = g(e);
        if (witnessHLine(x, y, 1))
            count++;
    }
    return ((double)count / tryCount >= expectation);
}

bool SpreadsheetRecognition::witnessVLine(int x, int y, int radius) {
    int begin = std::max(x - radius, 0);
    int end = std::min(x + radius, mSrcGray.cols - 1);
    for (int ix = begin; ix <= end; ++ix) {
        if ((uint)mSrcGray.at<uchar>(y, ix) > 200)
            return true;
    }
    return false;
}

bool SpreadsheetRecognition::isVLine(Vec2f &line, int tryCount, double expectation) {
    CV_Assert(mSrcGray.channels() == 1);
    int x = (int)line[0];
    int count = 0;
    uniform_int_distribution<int> g((int)mHLines.front()[0], (int)mHLines.back()[0]);
    for (int i = 0; i < tryCount; ++i) {
        int y = g(e);
        if (witnessHLine(x, y, 10))
            count++;
    }
    cout << count << endl;
    return ((double)count / tryCount >= expectation);
}

void SpreadsheetRecognition::probFilterLines(int tryCount, double expectation) {
    vector<Vec2f> filteredHLines, filteredVLines;
    for (auto &line : mHLines)
        if (isHLine(line, tryCount, expectation))
            filteredHLines.push_back(line);
    for (auto &line : mVLines)
        if (isVLine(line, tryCount, expectation))
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
