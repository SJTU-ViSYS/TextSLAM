/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef MAPTEXT_H
#define MAPTEXT_H

#include<string>
#include<thread>
#include <map>
#include<opencv2/core/core.hpp>
#include <setting.h>
#include <tool.h>

namespace TextSLAM {

class keyframe;

class mapText
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // apart from initial map text object state is set to GOOD at the first time.
    // all map text objects are set to IMMATURE at the 'new mapText();'
    TextStatus STATE;
    int NumObvs;
    Mat33 Covariance;   // for unmature text object theta

    // semantic meaning *****
    TextInfo TextMean;
    // ***************************

    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstKFrameid;
    keyframe* RefKF;

    // raw text detection of each pyramid (pyramid -> 4 features/points)
    vector<vector<Vec2>> vTextDete;
    vector<Vec2> vTextDeteRay;  // the boxes in all pyramids have the same ray info

    // ref keyframe info
    vector<vector<TextFeature*>> vRefFeature;       // pyramid -> all text feature (feature)
    vector<Vec2> statistics;                    // pyramid -> Vec2(mu, std)
    vector<TextFeature*> vRefPixs;                      // all pixels within box in pyramid 0 (all pixels)
    vector<bool> vRefFeatureSTATE;                  // true:Good, false:BAD

    // observation
    int iObs;
    std::map<keyframe*,vector<int>> vObvkeyframe;     // store all keyframes which can observe this map text
    int ObvNum_good, ObvNum_bad;                        // only update in general Frames (not KF)

    // ********** loop info *************** //
    // (loop) 1. replaced flag
    bool Flag_Replaced;     // has been replaced: true (new obj is ReplacedmPt)
    mapText* ReplacedmObj;
    keyframe* ReplaceKF;
    keyframe* BeReplacedKF;

    // ********** local info *************** //
    int LastObvFId;

public:
    // initial text object
    mapText(vector<Vec2> &TextDete, keyframe* kRefKF, vector<vector<TextFeature *>> &vfeatureObj, int Idx, TextStatus TextState);
    mapText(vector<Vec2> &TextDete, keyframe* kRefKF, vector<vector<TextFeature *>> &vfeatureObj, int Idx, TextStatus TextState, TextInfo &_TextMean);

    void AddObserv(keyframe* KF);
    void AddObserv(keyframe* KF, const int &idx);
    void AddObserv(keyframe* KF, const vector<int> &vidx);
    void AddNewObserv(keyframe* KF, vector<int> &vidx);     // for loop map fusion
    bool UpdateKFObserv(keyframe* KF, int DeteIdx);
    bool IsInKeyFrame(keyframe* KF);

    bool GetObvIdx(keyframe *KF, vector<int> &vIdx);

    void GetObjectInfo(vector<Vec2> &TextDete);

    void AddObvNum(const bool &ObvFlag);

    // get params
    int GetNidx();
    std::map<keyframe*,vector<int>> GetObservationKFs();

    // replace
    void Replace(keyframe *KFCur, mapText* mObj, Eigen::MatrixXd &M2, Eigen::MatrixXd &M3);
    void SetReplaceKF(keyframe* KF);

    void UpdateCovMap_2(keyframe* KF, mapText *TextObj, Eigen::MatrixXd &M2);
    void UpdateCovMap_3(keyframe* KF, mapText *TextObj, Eigen::MatrixXd &M3);

private:

    tool Tool;

    int NIdx;   // coresponding to the idx of mNrw. (Ref.mNrw[NIdx])

private:
    void CalInitialSemanticScore(const TextInfo &TextInfoIni);
    double GetSgeo(const keyframe* KF, mapText* obj);


};

}

#endif // MAPTEXT_H
