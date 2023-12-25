/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <opencv2/core/core.hpp>

#include <setting.h>
#include <tool.h>
#include <keyframe.h>
#include <mapText.h>
#include <mapPts.h>
#include <Sim3Solver.h>
#include <optimizer.h>

using namespace std;

namespace TextSLAM {

class loopClosing
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    loopClosing(setting* Setting);
    bool Run(keyframe *_CurKF, map *_mpMap, optimizer *_Optimizer);

private:
    // main functions
    vector<keyframe*> DetectLoop(const vector<keyframe*> &vKFs, vector<vector<MatchmapTextRes> > &vAllMatchTextRes, const int &MinMatchedWords, const std::map<keyframe*, int> &vConnects);
    void ComputeSim3(const vector<keyframe *> &vKFCands,  const vector<vector<MatchmapTextRes> > &vMatchRes);
    void LoopCorrect();
    int GetThreshWordsNum(bool &FLAG_OK, std::map<keyframe*, int> &vConnectKFs);

    // main functions
    int SearchMatch(keyframe *CanKF, const vector<vector<MatchmapTextRes>> &vMatchTexts, vector<FeatureConvert> &vFeatCur, vector<FeatureConvert> &vFeatCan);
    int SearchMatch_Text(keyframe *CanKF, const vector<vector<MatchmapTextRes> > &vMatchTexts, vector<FeatureConvert> &vFeatCur, vector<FeatureConvert> &vFeatCan, vector<Mat> &TextLabel);
    int SearchMatch_Other(keyframe* CanKF, const vector<cv::Mat> TextLabel, vector<FeatureConvert> &vFeatCur, vector<FeatureConvert> &vFeatCan, const int &Win);

    int FeatureConvert_Text(const vector<DMatch> &match12, keyframe* KF1, keyframe* KF2, mapText *obj1, mapText *obj2, const int &idxDete1, const int &idxDete2, vector<FeatureConvert> &vFeat1, vector<FeatureConvert> &vFeat2);
    int FeatureConvert_Other(const vector<int> &vMatchIdx12, keyframe *KF1, keyframe *KF2, vector<FeatureConvert> &vFeat1, vector<FeatureConvert> &vFeat2);
    FeatureConvert GetConvertInfo(keyframe *KF, const int &Idx);

    void GetLoopsLandmarkers(std::map<mapPts*, keyframe*> &vLoopPts, std::map<mapText*, keyframe*> &vLoopObjs);
    void SearchAndFuse(const std::map<mapPts *, keyframe *> &vLoopPts, const std::map<mapText *, keyframe *> &vLoopObjs, const std::map<keyframe *, Sim3_loop, std::less<keyframe *>, Eigen::aligned_allocator<std::pair<keyframe *, Sim3_loop> > > &vConnectKFs, const bool &AddCurrent, vector<Eigen::MatrixXd> &vMs);
    int SearchAndFuse_Scene(keyframe* KF, const Sim3_loop &Scw, const std::map<mapPts *, keyframe *> &vLoopPts, std::map<mapPts *, mapPts *> &vReplacePts, double th, Eigen::MatrixXd &M1);
    int SearchAndFuse_Text(keyframe* KF, const Sim3_loop &Scw, const std::map<mapText *, keyframe *> &vLoopObjs, std::map<mapText*, mapText*> &vReplaceObjs);

    void UpdateCovisibleKFs();

    int MatchMore(keyframe *KF1, keyframe *KFMatch2, const Sim3_loop& gscm, vector<FeatureConvert> &vFeatCur, vector<FeatureConvert> &vFeatCan);

    // tools
    void DeleteWords();
    vector<Vec2> ProjTextInKF(mapText* obj, keyframe* KF, bool &FlagPred);
    vector<Vec2> ProjTextInKF(mapText* obj, const Mat44 &Tcw, bool &FlagPred);
    // match tool
    vector<DMatch> FeatureMatch_brute(cv::Mat &Descrip1, cv::Mat &Descrip2, const bool &USETHRESH);

    // same as others(tracking) (tools)
    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    void GetCovisibleKFs_all(keyframe* KF, const int &CurKFNum);

private:

    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

    map* mpMap;
    keyframe* mpCurrentKF;
    keyframe* mpMatchedKF;
    int MaxInlierNum, MaxInlierNum_S;
    Sim3_loop mScm, mScw;
    vector<FeatureConvert> mvFeatCur, mvFeatCan;

    // funciton word
    std::map<string, double> vDeleteWords;

    // optimization
    optimizer* coOptimizer;

    // tool
    tool Tool;

    // params
    int Thmin_ThreshMatchWordsNum;
    int Th_nInliers_Scene;
    int Th_MaxInlierNum_S;
    double ScoreThresh_min;
    bool DoubleCheck_Visible;

};

}

#endif // LOOPCLOSING_H
