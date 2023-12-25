/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <string>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <setting.h>
#include <keyframe.h>
#include <mapText.h>
#include <mapPts.h>
#include <map.h>

// optimize model
#include "include/nume_IniBAText.h"
#include "include/auto_IniBAScene.h"

#include "include/nume_PoseOptimText.h"
#include "include/auto_PoseOptimScene.h"

#include "include/nume_BAText.h"
#include "include/auto_BAScene.h"
#include "include/auto_BASceneNW.h"

// update rho
#include "include/auto_RhoScene.h"
// update N
#include "include/nume_thetaText.h"

// Sim3
#include "include/auto_sim.h"
#include "include/auto_siminv.h"
#include "include/numer_loop_ver2.h"

using namespace std;
using namespace ceres;

namespace TextSLAM {


class optimizer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    optimizer(Mat33 &mK, double &dScale, int &nLevels, bool &Flag_noText, bool &Flag_rapid);

    void InitBA(keyframe *F1, keyframe *F2);
    void PoseOptim(frame &F);
    void LocalBundleAdjustment(map* mpMap, vector<keyframe*> vKFs, const BAStatus &STATE);
        // global BA
    void GlobalBA(map* mpMap);
    void OptimizeLandmarker(map* mpMap);
    // N optimization: use current F & previous Keyframes
    bool ThetaOptimMultiFs(const frame &F, mapText* &obj);

    // LoopClosing
    int OptimizeSim3(vector<FeatureConvert> &vFeat1, vector<FeatureConvert> &vFeat2, vector<bool> &vbInliers, Sim3_loop &Sim12, const float th2);
    void OptimizeLoop(std::map<keyframe*, set<keyframe*>> &LoopConnections, std::map<keyframe*, set<keyframe*> > &NormConnections, keyframe* KF, keyframe* LoopKF, std::map<keyframe *, Sim3_loop, std::less<keyframe *>, Eigen::aligned_allocator<std::pair<keyframe *, Sim3_loop> > > &vConnectKFs, Sim3_loop &mScw, map *mpMap);


private:
    // ------------- pyramid BA -------------
    void PyrIniBA(ceres::Problem *problem, vector<keyframe*> KF, double **pose, double **theta, double **rho, vector<SceneObservation*> ScenePts, vector<vector<SceneFeature *> > SceneObv, vector<TextObservation*> TextObjs, const int &PyBegin, Mat &TextLabelImg);

    // each pyramid optimization --------
    void PyrPoseOptim(frame &F, double *pose, vector<SceneObservation*> ScenePts, vector<vector<SceneFeature *> > SceneObv, vector<TextObservation*> TextObjs, const int &PyBegin, const double &chi2Mono, const double &chi2Text,
                                  const int &its, vector<bool> &vPtsGood, vector<bool> &vTextsGood, vector<vector<bool> > &vTextFeatsGood, const vector<int> &FLAGTextObjs, Mat &TextLabelImg);     //
    void PyrBA(double **pose, double **theta, double **rho, const vector<keyframe*>  &vKFs, const vector<mapPts*> &vPts, const vector<mapText*> &vTexts, const vector<int> &vmnId2Pts, const vector<bool> &vPtOptim, const vector<int> &vmnId2Texts, const vector<bool> &vTextOptim, const vector<int> &vmnId2vKFs, const vector<int> &InitialIdx, const int &PyBegin,
                const double &chi2Mono, const double &chi2Text, const int &its, Mat &TextLabelImg, const BAStatus &STATE);

    void PyrGlobalBA(double **pose, double **theta, double **rho, const vector<keyframe*>  &vKFs, const vector<mapPts*> &vPts, const vector<mapText*> &vTexts,
                     const vector<int> &vmnId2Pts, const vector<int> &vmnId2Texts, const vector<int> &vmnId2vKFs, const vector<int> &InitialIdx, vector<vector<bool> > &vPtsGoodkf,
                     const int &PyBegin, const double &chi2Mono, const int &its);

    // (landmarker)
    void PyrLandmarkers(double **pose, double **theta, double **rho, const vector<keyframe*>  &vKFs, const vector<mapPts*> &vPts, const vector<mapText*> &vTexts,
                                   const vector<int> &vmnId2Pts, const vector<int> &vmnId2Texts, const vector<int> &vmnId2vKFs, vector<vector<bool>> &vPtsGoodkf,
                                   const int &PyBegin, const double &chi2Mono, const double &chi2Text, const int &its, vector<Mat> &TextLabelImg);

    bool PyrThetaOptim(const vector<cv::Mat> &vImg, const vector<Mat44,Eigen::aligned_allocator<Mat44>> &vTcr, const vector<keyframe*> &vKFs, mapText *obj, const int &PyBegin, double *theta, Mat33 &thetaVariance);

    // ------------- pyramid BA -------------

    // --------------- update ----------------
    void UpdateTrackedTextBA(vector<TextObservation*> &TextObjs, const cv::Mat &ImgTextProjLabel, keyframe* KFCur, const bool &INITIAL);    // for initial BA
    void UpdateTrackedTextBA(vector<TextObservation*> &TextObjs, const vector<int> &vIdxState2Raw, const cv::Mat &ImgTextProjLabel, keyframe* KFCur, const bool &INITIAL);    // for localBA
    void UpdateTrackedTextPOSE(vector<TextObservation*> &TextObjs, const cv::Mat &ImgTextProjLabel, frame &F);
    // --------------- update ----------------

    // ------------- visualization -------------
    // initial BA
    void ShowBAReproj_TextBox(double **rho, double **theta, double **pose, const vector<keyframe*> &KF, const int &PyBegin, const vector<SceneObservation *> ScenePts, const vector<vector<SceneFeature*>> SceneObv, const vector<TextObservation*> TextObjs, Mat &TextLabel, const int op);
    // pose optimization
    void ShowBAReproj_TextBox(double *pose, const frame &F, const int &PyBegin, const vector<SceneObservation *> ScenePts, const vector<vector<SceneFeature*>> SceneObv, const vector<TextObservation*> TextObjs, cv::Mat &TextLabel, const int op);
    // BA
    void ShowBAReproj_TextBox(double **rho, double **theta, double **pose, const int &PyBegin, const vector<keyframe*> &vKFs, const vector<int> &vmnId2vKFs, const vector<int> &vmnId2Pts, const vector<int> &vmnId2Texts, cv::Mat &TextLabel, const int op);
    // Global BA
    void ShowBAReproj_TextBox(double **rho, double **theta, double **pose, const int &PyBegin, const vector<keyframe*> &vKFs, const vector<int> &vmnId2vKFs, const vector<int> &vmnId2Pts, const vector<int> &vmnId2Texts, vector<cv::Mat> &vTextLabel, const int op);
    // ------------- visualization -------------

    // ------------- BA tool -------------
    vector<int> GetNewIdxForTextState(const vector<TextObservation*> &TextObjs, const TextStatus &Needstate);
    // ------------- BA tool -------------

    // ------------- From Tracking -------------
    void UpdateSemantic_MapObjs_single(mapText* obj, keyframe* KF);
    double GetSgeo(const keyframe* KF, mapText *obj);
    // ------------- From Tracking -------------

    Mat33 K;
    static double fx, fy, cx, cy;
    static double invfx, invfy;

    vector<Mat33> vK;
    vector<double> vfx, vfy, vcx, vcy;
    vector<double> vInvScalefactor;
    bool bFlag_noText;
    bool bFlag_rapid;

    tool Tool;


};

}

#endif // OPTIMIZER_H
