/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <string>
#include <thread>
#include <opencv2/core/core.hpp>
#include <frame.h>
#include <mapPts.h>
#include <mapText.h>

using namespace std;

namespace TextSLAM {

class mapText;

class keyframe
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // keyframe initial from frame
    keyframe(frame &F);

    void AddTextObserv(mapText* textobj, int idx, int NumFeats);
    void AddTextObserv(mapText* textobj, int NumFeats);
    void AddSceneObservForInitial(mapPts* scenept, int idx);
    void AddSceneObserv(mapPts* scenept, int idx);
    vector<TextObservation*> GetStateTextObvs(const TextStatus &Needstate);
    vector<TextObservation*> GetStateTextObvs(const TextStatus &Needstate, vector<int> &vNew2Raw);
    vector<TextObservation*> GetTextObvs();
    // within image check
    bool IsInImage(const double &x, const double &y);

    // match for loopClosing
    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;

    // (loop) set info
    void AddLoopEdge(keyframe* KF);
    vector<keyframe*> GetLoopEdges();

    void SetCovisibleKFsAll(vector<CovKF> &vKFConKFs);
    void SetCovisibleKFsPrev(vector<CovKF> &vKFConKFs);
    void SetCovisibleKFsAfter(vector<CovKF> &vKFConKFs);
    int GetCovisibleWeight(keyframe* KF);

    // (loop) out info
    vector<CovKF> GetCovisibleKFs_All();
    vector<CovKF> GetCovisibleKFs_Prev();
    vector<CovKF> GetCovisibleKFs_After();
    vector<CovKF> GetTopCovisKFs(const int &TopN);

    // (loop) landmarker fusion
    // change keyframe landmarker observation
    void ReplaceMapPt(const size_t &idx, mapPts* mPt);
    void ReplaceMapText(mapText* mObj, mapText* mObjReplace);

    // pose
    void SetPose(Mat44 &Tcw);
    void UpdatePoseMatrices();
    void SetN(Mat31 &N, int idx);
    Mat44 GetTcw();
    Mat44 GetTwc();

    // map points
    int TrackedMapPoints(const int &minObs);

    double dTimeStamp;
    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId;
    cv::Mat FrameImg;
    Mat33 mK;

    const double fx, fy, cx, cy, invfx, invfy;
    const double mnMinX, mnMaxX, mnMinY, mnMaxY;

    // Grid over the image to speed up feature matching
    const int mnGridCols;
    const int mnGridRows;
    const double mfGridElementWidthInv, mfGridElementHeightInv;
    std::vector< std::vector <std::vector<size_t> > > mGrid;

    Mat44 mTcw, mTwc;
    Mat33 mRcw, mRwc;
    Mat31 mtcw, mtwc;
    // mNcr size == vNGOOD size. vNGOOD[i] define if mNcr[i] is initialed or not
    vector<Mat31> mNcr;
    vector<bool> vNGOOD;

    // text object
    // detection info (text object -> 4PtsBox/MinMax pix/Center pix/corresponding map text object)
    vector<vector<Vec2>> vTextDete;
    vector<Vec2> vTextDeteMin, vTextDeteMax;    // the min and the max points of the detected text object
    vector<Vec2> vTextDeteCenter;
    vector<int> vTextDeteCorMap;

    // 2.1. semantic meaning *****
    vector<TextInfo> vTextMean;
    vector<CovKF> vCovisibleKFsAll;     // previous + sebsequence
    vector<CovKF> vCovisibleKFsPrev;    // this KF previous covisible KFs
    vector<CovKF> vCovisibleKFsAfter;   // this KF sebsequence covisible KFs

    // loop edge
    vector<keyframe*> vLoopEdges;
    // ***************************

    int iN, iNScene, iNTextObj, iNTextFea;
    vector<cv::KeyPoint> vKeys, vKeysScene;
    vector<vector<cv::KeyPoint>> vKeysText;
    cv::Mat mDescr, mDescrScene;
    vector<cv::Mat> mDescrText;
    vector<int> vTextObjInfo;
    vector<vector<cv::Point2f>> vKeysTextTrack;     // Patch -> all points. inherit from frame.h: the corresponding feature to the last keyframe.
    vector<vector<cv::Point2f>> vKeysNewTextTrack;  // Patch -> all points. for all new text objects in this keyframe, initial features to track in the following frames.
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

    // for initialization, the scene pts 3D-2D store
    vector<Vec3> IniScenePts3d;
    vector<Vec2> IniSceneObv2d;

    // landmarkers
    vector<TextObservation*> vObvText;
    vector<SceneObservation*> vObvPts;              // vObvPts corresponding to vIniSceneObv2d
    vector<vector<SceneFeature*>> vSceneObv2d;      // pyramid -> all SceneFeature
    vector<bool> vObvGoodPts;                       // ReInitial. size == vObvPts.size();
    vector<bool> vObvGoodTexts;                     // size == vObvText.size();
    vector<vector<bool>> vObvGoodTextFeats;         // obj (vObvText idx) -> all feats (Idx2Raw)

    // match
    vector<int> vMatches2D3D;

    // local info
    int CovisbleFrameId;
    keyframe* cfLocalKF;


};

}

#endif // KEYFRAME_H
