#include <opencv2\opencv.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "Stopwatch.h"
using namespace cv;
using namespace std;

#define DEBUG
// #define OUTPUT

template<class Ty>
void showlines(Ty &lines) {
    for (auto &line : lines)
        cout << line << endl;
}

class SpreadsheetRecognition {
public:
    SpreadsheetRecognition(Mat src, int gaussianSize = 3, int cannyThreshold = 40, int houghThreashold = 200) :
        mSrc(src), mGaussianSize(gaussianSize), mCannyThreshold(cannyThreshold), mRatio(3), mApertureSize(5),
        mHoughThreashold(houghThreashold) {}

    Mat resultView;

    void execute() {
        // preprocess
        cvtColor(mSrc, mSrcGray, CV_RGB2GRAY);
        int ks = 2 * mGaussianSize + 1;
        GaussianBlur(mSrcGray, mSrcGray, Size(ks, ks), 0, 0);
        Canny(mSrcGray, mSrcGray, mCannyThreshold, mCannyThreshold * mRatio, mApertureSize, true);
#ifdef DEBUG
        cvtColor(mSrcGray, resultView, CV_GRAY2BGR);
#endif

        // detect lines
        HoughLines(mSrcGray, mLines, 1, CV_PI / 180, mHoughThreashold);
#ifdef DEBUG
        //drawLines(resultView, mLines, Scalar(0, 0, 255));
#endif
        
        // process lines
        float deltaTheta = CV_PI / 180, deltaRho = 10.0;
        classifyLines(deltaTheta);
#ifdef DEBUG
        //drawLines(resultView, mHLines, Scalar(0, 255, 255));
        //drawLines(resultView, mVLines, Scalar(0, 255, 255));
#endif
        probFilterHLines();
#ifdef DEBUG
        drawLines(resultView, mHLines, Scalar(255, 0, 0));
#endif
        //clusterFilterLines(hLines, deltaRho);
        //clusterFilterLines(vLines, deltaRho);

#if !defined(DEBUG) && defined(OUTPUT)
        splitSpreadsheet();
#endif
    }

private:
    static void drawLines(Mat &img, vector<Vec2f> &lines, const Scalar &color) {
        for (size_t i = 0; i < lines.size(); i++) {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double costheta = cos(theta), sintheta = sin(theta);
            double x0 = rho * costheta, y0 = rho * sintheta;
            pt1.x = cvRound(x0 + 1000 * (-sintheta));
            pt1.y = cvRound(y0 + 1000 * (costheta));
            pt2.x = cvRound(x0 - 1000 * (-sintheta));
            pt2.y = cvRound(y0 - 1000 * (costheta));
            line(img, pt1, pt2, color, 1, CV_AA);
        }
    }

    static bool compVec2f(const Vec2f &lhs, const Vec2f &rhs) {
        if (lhs[0] < rhs[0])
            return true;
        if (lhs[0] == rhs[0] && lhs[1] < rhs[1])
            return true;
        return false;
    }

    void classifyLines(float deltaTheta) {
        const float hTheta = CV_PI / 2.0, vTheta = 0.0;

        for (auto &line : mLines) {
            float rho = line[0], theta = line[1];
            if (fabs(theta - hTheta) <= deltaTheta && rho >= 0 && rho < mSrcGray.rows)
                mHLines.push_back(line);
            else if (fabs(theta - vTheta) <= deltaTheta && rho >= 0 && rho < mSrcGray.cols)
                mVLines.push_back(line);
        }
        sort(mHLines.begin(), mHLines.end(), compVec2f);
        sort(mVLines.begin(), mVLines.end(), compVec2f);
    }

    bool witnessHLine(int x, int y, int radius = 1) {
        int begin = std::max(y - radius, 0);
        int end = std::min(y + radius, mSrcGray.rows - 1);
        for (int i = begin; i <= end; ++i) {
            if ((uint)mSrcGray.at<uchar>(i, x) > 200)
                return true;
        }
        return false;
    }

    bool isHLine(Vec2f &line, int tryCount, double expectation = 0.5) {
        CV_Assert(mSrcGray.channels() == 1);
        int y = (int)line[0];
        int count = 0;
        uniform_int_distribution<int> g((int)(*mVLines.begin())[0], (int)(*(mVLines.end() - 1))[0]);

        for (int i = 0; i < tryCount; ++i) {
            int x = g(e);
            if (witnessHLine(x, y, 1))
                count++;
        }
        cout << count << endl;
        return ((double)count / tryCount >= expectation);
    }

    void probFilterHLines() {
        vector<Vec2f> filteredHLines;
        for (auto &line : mHLines)
            if (isHLine(line, 200, 0.5))
                filteredHLines.push_back(line);
        mHLines = filteredHLines;
    }

    void clusterFilterLines(vector<Vec2f> &lines, float deltaRho) {
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

    void splitSpreadsheet() {
        vector<vector<Rect>> spreadsheet;
        vector<int> xs, ys;
        for (auto &line : mHLines)
            ys.push_back((int)line[0]);
        for (auto &line : mVLines)
            xs.push_back((int)line[0]);
        sort(xs.begin(), xs.end());
        sort(ys.begin(), ys.end());

        for (size_t i = 0; i < ys.size() - 1; ++i) {
            vector<Rect> row;
            for (size_t j = 0; j < xs.size() - 1; ++j)
                row.push_back(Rect(xs[j], ys[i], xs[j + 1] - xs[j], ys[i + 1] - ys[i]));
            spreadsheet.push_back(row);
        }

        for (size_t i = 0; i < spreadsheet.size(); ++i)
            for (size_t j = 0; j < spreadsheet[i].size(); ++j) {
                ostringstream oss;
                oss << R"(.\output\)" << i << "_" << j << ".jpg";
                imwrite(oss.str(), mSrc(spreadsheet[i][j]));
            }
    }

    Mat mSrc, mSrcGray;
    vector<Vec2f> mLines, mHLines, mVLines;

    static default_random_engine e;

    // parameters
    int mGaussianSize;
    int mCannyThreshold, mRatio, mApertureSize;
    int mHoughThreashold;
};

default_random_engine SpreadsheetRecognition::e;

string windowName = "Edge Detect";
Mat src;

int gaussianSize = 3;
const int maxGaussianSize = 50;

int cannyThreshold = 40;
const int maxCannyThreshold = 100;
int ratio = 3;
int apertureSize = 5;

int houghThreshold = 200;
const int maxHoughThreshold = 500;

void onChange(int, void *) {
    SpreadsheetRecognition SR(src, gaussianSize, cannyThreshold, houghThreshold);
    SR.execute();
    imshow(windowName, SR.resultView);
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
    createTrackbar("Gauss:", windowName, &gaussianSize, maxGaussianSize, onChange);
    createTrackbar("Canny:", windowName, &cannyThreshold, maxCannyThreshold, onChange);
    createTrackbar("Hough:", windowName, &houghThreshold, maxHoughThreshold, onChange);
#endif

    Stopwatch timer;
    onChange(0, 0);
    cout << timer << endl;

    // imwrite("output.jpg", dst);

    waitKey(0);
    return 0;
}


