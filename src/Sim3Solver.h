/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "keyframe.h"

using namespace std;
using namespace cv;

namespace TextSLAM {

class Sim3Solver
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Sim3Solver(keyframe* pKF1, keyframe* pKF2, const vector<FeatureConvert> &vFeat1, const vector<FeatureConvert> &vFeat2);

    void SetRansacParameters(double probability = 0.99, int minInliers = 6 , int maxIterations = 300);

    bool iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers, Mat44 &BestT12Res);

    Mat33 GetEstimatedRotation();
    Mat31 GetEstimatedTranslation();
    double GetEstimatedScale();

protected:
    tool Tool;

    void ComputeSim3(Mat33 &P1, Mat33 &P2);

    void CheckInliers();

    void ComputeCentroid(const Mat33 &P, Mat33 &Pr, Mat31 &C);
    void Project(const vector<Mat31> &vP3Dw, vector<Vec2> &vP2D, Mat44 Tcw, Mat33 K, keyframe *KF);

public:


protected:

    // KeyFrames and matches
    keyframe* mpKF1;
    keyframe* mpKF2;

    vector<Mat31> mvX3Dc1;
    vector<Mat31> mvX3Dc2;
    // Projections
    vector<Vec2> mvP1im1;
    vector<Vec2> mvP2im2;

    int N;

    Mat33 mR12i;
    Mat31 mt12i;
    double ms12i;
    Mat44 mT12i;
    Mat44 mT21i;
    vector<bool> mvbInliersi;
    int mnInliersi;

    // Current Ransac State
    int mnIterations;
    std::vector<bool> mvbBestInliers;
    int mnBestInliers;
    Mat44 mBestT12;
    Mat33 mBestRotation;
    Mat31 mBestTranslation;
    double mBestScale;

    // RANSAC probability
    double mRansacProb;

    // RANSAC min inliers
    int mRansacMinInliers;

    // RANSAC max iterations
    int mRansacMaxIts;

    // Calibration
    Mat33 mK1;
    Mat33 mK2;

};

}

#endif // SIM3SOLVER_H
