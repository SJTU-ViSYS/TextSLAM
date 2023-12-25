/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <mapPts.h>
#include <keyframe.h>

namespace TextSLAM
{
long unsigned int mapPts::nNextId=0;

mapPts::mapPts(const Vec3 &Pos, keyframe* kRefKF):
    mnFirstKFid(kRefKF->mnId), mnFirstKFrameid(kRefKF->mnFrameId), iObs(0),FLAG_BAD(false),Flag_Replaced(false),
    LastObvFId(0), Flag_LocalTrack(false), LocalTrackProj(Vec2(-1,-1)), ObvNum_good(0), ObvNum_bad(0)
{
    PtInvd(0) = Pos(0);
    PtInvd(1) = Pos(1);
    PtInvd(2) = Pos(2);
    mnId=nNextId++;
    RefKF = kRefKF;

    ReplacedmPt = static_cast<mapPts*>(NULL);
    ReplaceKF = static_cast<keyframe*>(NULL);
    BeReplacedKF = static_cast<keyframe*>(NULL);
}

bool mapPts::GetKFObv(keyframe* KF, int &IdxObserv)
{
    std::map<keyframe*,size_t>::iterator it = vObvkeyframe.find(KF);
    if(it==vObvkeyframe.end())
        return false;
    else{
        IdxObserv = (int)vObvkeyframe[KF];
        return true;
    }
}

double mapPts::GetInverD()
{
    double rho = PtInvd(2);
    return rho;
}
Mat31 mapPts::GetRaydir()
{
    Mat31 raydir;
    raydir(0) = PtInvd(0);
    raydir(1) = PtInvd(1);
    raydir(2) = (double)1.0;
    return raydir;
}
Vec3 mapPts::GetPtInv()
{
    Vec3 Pt;
    Pt(0) = PtInvd(0);
    Pt(1) = PtInvd(1);
    Pt(2) = PtInvd(2);
    return Pt;
}

std::map<keyframe*, size_t> mapPts::GetObservationKFs()
{
    std::map<keyframe*, size_t> vObvkeyframeout;
    vObvkeyframeout = vObvkeyframe;
    return vObvkeyframeout;
}

void mapPts::AddObserv(keyframe* KF, int idx)
{
    if(vObvkeyframe.count(KF)){
        return;
    }

    vObvkeyframe.insert(make_pair(KF, idx));
    iObs++;
}

void mapPts::AddObvNum(const bool &ObvFlag)
{
    if(ObvFlag)
        ObvNum_good++;
    else
        ObvNum_bad++;
}

int mapPts::GetObvNum(const bool &ObvFlag)
{
    int out;

    if(ObvFlag)
        out = ObvNum_good;
    else
        out = ObvNum_bad;

    return out;
}

// update rho
void mapPts::SetRho(double &rho)
{
    bool DEBUG = false;
    if(DEBUG)
        cout<<"before optimization rho of mapPts "<<mnId<<" is: "<<PtInvd(2)<<endl;

    PtInvd(2) = rho;

    if(DEBUG)
        cout<<"after optimization rho of mapPts "<<mnId<<" is: "<<PtInvd(2)<<endl;
}

int mapPts::GetObservNum()
{
    return iObs;
}

Mat31 mapPts::GetxyzPos()
{
    Mat31 ray = Mat31(PtInvd(0), PtInvd(1), 1.0);
    double rho = PtInvd(2);
    Mat31 Phost = ray/rho;
    Mat44 Twh = RefKF->mTwc;
    Mat31 Pw = Twh.block<3,3>(0,0) * Phost + Twh.block<3,1>(0,3);
    return Pw;
}

void mapPts::PtErase(keyframe* KFCur)
{
    FLAG_BAD = true;
    Flag_Replaced = true;
    BeReplacedKF = KFCur;

}


void mapPts::Replace(keyframe* KFCur, mapPts* mPt, Eigen::MatrixXd &M1)
{
    if(mPt->mnId==this->mnId){
        return;
    }

    // Step1. get observation;
    std::map<keyframe*,size_t> obs;
    obs = vObvkeyframe;

    // Step2. set flag; set new point pointer
    FLAG_BAD = true;
    Flag_Replaced = true;
    bool c = mPt->ReplaceKF;

    ReplacedmPt = mPt;
    BeReplacedKF = KFCur;
    mPt->SetReplaceKF(KFCur);

    // Step4. use p1 visible map to update covisble map
    for(std::map<keyframe*,size_t>::iterator mit0=obs.begin(); mit0!=obs.end(); mit0++)
    {
        keyframe* pKF = mit0->first;

        // add p2 observations
        UpdateCovMap_1(pKF, mPt, M1);
    }

    // Step4. change related variable (KFs)
    for(std::map<keyframe*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        keyframe* pKF = mit->first;

        if(!mPt->IsInKeyFrame(pKF))
        {
            // for KF observations
            pKF->ReplaceMapPt(mit->second, mPt);
            // for replaced point
            mPt->AddObserv(pKF, mit->second);
        }
    }

}


void mapPts::SetReplaceKF(keyframe* KF)
{
    ReplaceKF =KF;
}

bool mapPts::IsInKeyFrame(keyframe* KF)
{
    return (vObvkeyframe.count(KF));
}

void mapPts::UpdateCovMap_1(keyframe* KF, mapPts *Scenepts, Eigen::MatrixXd &M1)
{
    std::map<keyframe*, size_t> vKFs = Scenepts->GetObservationKFs();
    std::map<keyframe*, size_t>::iterator iterkf;
    int IdKF2 = KF->mnId;
    for(iterkf = vKFs.begin(); iterkf != vKFs.end(); iterkf++){
        keyframe* kfobv = iterkf->first;
        int IdKF1 = kfobv->mnId;
        if(IdKF1>=IdKF2)
            continue;

        assert(IdKF1>=0);
        assert(IdKF2>=0);
        assert(IdKF1<M1.rows());
        assert(IdKF2<M1.cols());

        M1(IdKF1, IdKF2) += 1;
    }

}

}
