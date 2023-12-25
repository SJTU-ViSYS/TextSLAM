/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef FRAME_H
#define FRAME_H

#include <string>
#include <thread>
#include <opencv2/core/core.hpp>
#include "ceres/ceres.h"

#include <setting.h>
#include <tool.h>
#include <ORBextractor.h>

using namespace std;
namespace TextSLAM {

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64
class mapText;
class setting;
class tool;
class ORBextractor;

class frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    frame();
    frame(const cv::Mat &imGray, const double &ImgTimeStamp, const Mat33 &K, int &ScaleLevels, double &ScaleFactor,
          const vector<vector<Vec2>> &TextDete, const vector<TextInfo> &TextMean, ORBextractor* extractor, const bool &bVeloc);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const;

    // pose
    void SetPose(Mat44 Tcw);
    void UpdatePoseMatrices();

    // add observation
    void AddTextObserv(mapText* textobj, int idx);
    void AddTextObserv(mapText* textobj);
    void AddTextObserv(mapText* textobj, const int &Feat0PyrNum, const vector<int> &vDeteIdx);
    void AddSceneObserv(mapPts* scenept, const int &ScenePtId, int idx);


// variable
    // 1. frame basic info
    double dTimeStamp;
    static long unsigned int nNextId;
    long unsigned int mnId;
    cv::Mat FrameImg;
    Mat33 mK;

    // intrinsic params
    static double fx, fy, cx, cy;
    static double invfx, invfy;
    // Undistorted Image Bounds (computed once).
    static double mnMinX;
    static double mnMaxX;
    static double mnMinY;
    static double mnMaxY;
    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static double mfGridElementWidthInv;
    static double mfGridElementHeightInv;
    vector<size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // 2. text object
    // detection info (text object -> 4PtsBox/MinMax pix/Center pix)
    vector<vector<Vec2>> vTextDete;
    vector<Vec2> vTextDeteMin, vTextDeteMax;    // the min and the max points of the detected text object
    vector<Vec2> vTextDeteCenter;
    vector<int> vTextDeteCorMap;        // -1: no corresponding existing map text object; 0-N: corresponding to 0-N(mnId) map text object.

    // 2.1. semantic meaning *****
    vector<TextInfo> vTextMean;
    // ***************************

    // 3. pose info
    Mat44 mTcw, mTwc;
    Mat33 mRcw, mRwc;
    Mat31 mtcw, mtwc;
    // vNGOOD=true if there are enough features (>=thresh(3...)).
    vector<Mat31> mNcr;
    vector<bool> vNGOOD;

    // 4. feature info
    ORBextractor* coORBextractor;
    int iN;             // all point number (including ScenePts, all TextObjects all features)
    int iNScene;        // all scene pts number
    int iNTextObj;      // all Text Objects number
    int iNTextFea;      // all Text Features
    vector<cv::KeyPoint> vKeys;                 // all extracted features (Scene + Text)
    vector<cv::KeyPoint> vKeysScene;            // all extracted scene features
    vector<vector<cv::KeyPoint>> vKeysText;     // text object -> features. all extracted text features
    cv::Mat mDescr;                       // correspond descriptor -> vKeys
    cv::Mat mDescrScene;                  // correspond descriptor -> vKeysScene
    vector<cv::Mat> mDescrText;           // text object -> features. correspond descriptor -> mDescriptorsText
    vector<int> vTextObjInfo;              // corresponding to vKeys. Scene -> -1, Text -> 0~iNTextObj-1, the text feature belong to (TextObjInfo)th text object
    vector<vector<cv::Point2f>> vKeysTextTrack;  // Patch -> all points. for all new text objects in the last keyframe(kf->vTextDeteCorMap[i]==-1), vKeysNewTextKLT store its tracked keypoints using KLT
    // feature idx
    vector<int> vKeysSceneIdx;          // from Scene to vKeys
    vector<vector<int>> vKeysTextIdx;   // from each Text obj to vKeys

    // 5. pyramid info
    int iScaleLevels;                   // pyramid level number
    double dScaleFactor;                // each pyramid level scale (s)
    vector<double> vScaleFactors;          // all pyramid level scale to 0st level (1, s, s^2, s^3, s^4,...)
    vector<double> vInvScaleFactors;    // all pyramid inverse scale to 0st level (1, 1/s, 1/s^2, 1/s^3, 1/s^4,...)
    vector<double> vLevelSigma2;           // all pyramid level scale^2 to 0st level (1, s^2, s^4, s^6, s^8,...)
    vector<double> vInvLevelSigma2;     // all pyramid (inverse scale)^2 to 0st level (1, 1/s^2, 1/s^4, 1/s^6, 1/s^8,...)
    vector<Mat33> vK_scale;           // intrinsic param pyramid (K*vInvScaleFactors)
    // frame pyramid
    vector<cv::Mat> vFrameImg;
    vector<cv::Mat> vFrameGrad;
    vector<cv::Mat> vFrameGradX, vFrameGradY;
    // text feature pyramid
    vector<vector<vector<TextFeature*>>> vfeatureText;  // text obj -> pyramid -> all features.

    // for track
    bool bVelocity;       // for cfLastFrame exist or not judge. if cfLastFrame define, the existed frame res is true;

    // for initialization, the scene pts 3D-2D store
    vector<Vec3> IniScenePts3d;
    vector<Vec2> IniSceneObv2d;

    // observation
    vector<TextObservation*> vObvText;
    vector<SceneObservation*> vObvPts;
    vector<vector<SceneFeature*>> vSceneObv2d;    // pyramid -> all SceneFeature, vSceneObv2d[0] order is same to vObvPts
    vector<bool> vObvGoodPts;
    vector<bool> vObvGoodTexts;
    vector<vector<bool>> vObvGoodTextFeats; // obj (vObvText idx) -> all feats (Idx2Raw)

    // match
    vector<int> vMatches2D3D;

    // local map related
    int CovisbleFrameId;
    keyframe* cfLocalKF;

private:
    bool GetPyrParam();
    void GetPyrMat();
    void GetFrameParam();

    void FeatExtract();
    void FeatFusion();
    void FeatExtraScene(cv::Mat &Img, vector<cv::KeyPoint> &Keys, cv::Mat &Desc);
    void FeatExtracText(const cv::Mat &Img, const vector<vector<Vec2>> &TextDete, vector<vector<cv::KeyPoint>> &KeysTextRaw, vector<cv::Mat> &DescripTextRaw);
    void TextFeaProc();
    void AssignFeaturesToGrid();

    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    tool Tool;
};

}

#endif // FRAME_H
