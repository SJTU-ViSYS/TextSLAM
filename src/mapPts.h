/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef MAPPTS_H
#define MAPPTS_H

#include <string>
#include <thread>
#include <map>
#include<opencv2/core/core.hpp>
#include <setting.h>

namespace TextSLAM {

class keyframe;
class map;

class mapPts
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstKFrameid;
    keyframe* RefKF;

    // flag for point is good or bad
    bool FLAG_BAD;

    // ********** loop info *************** //
    // (loop) replaced flag
    bool Flag_Replaced;     // has been replaced: true (new pt is ReplacedmPt)
    mapPts* ReplacedmPt;
    keyframe* ReplaceKF;
    keyframe* BeReplacedKF;

    // ********** local info *************** //
    int LastObvFId;
    Vec2 LocalTrackProj;
    bool Flag_LocalTrack;

public:
    mapPts(const Vec3 &Pos, keyframe* kRefKF);
    bool GetKFObv(keyframe *KF, int &IdxObserv);
    double GetInverD();
    Mat31 GetRaydir();
    Vec3 GetPtInv();
    std::map<keyframe*, size_t> GetObservationKFs();
    int GetObservNum();
    // convert to xyz
    Mat31 GetxyzPos();

    // observation
    void AddObserv(keyframe* KF, int idx);
    bool IsInKeyFrame(keyframe* KF);
    void AddObvNum(const bool &ObvFlag);
    int GetObvNum(const bool &ObvFlag);
    // update rho
    void SetRho(double &rho);

    // (loop) landmarker fusion/change
    void PtErase(keyframe *KFCur);
    void Replace(keyframe *KFCur, mapPts* mPt, Eigen::MatrixXd &M1);
    void SetReplaceKF(keyframe* KF);

    void UpdateCovMap_1(keyframe* KF, mapPts *Scenepts, Eigen::MatrixXd &M1);

private:

    // map points is represented as inverse depth
    Vec3 PtInvd;

    // record this map point is observed by i keyframe(keyframe*) in the j 2D(size_t) observation.
    std::map<keyframe*, size_t> vObvkeyframe;
    int iObs;
    int ObvNum_good, ObvNum_bad;

};

}

#endif // MAPPTS_H
