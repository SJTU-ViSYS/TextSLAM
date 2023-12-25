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
#include <map.h>

namespace TextSLAM
{
map::map(const int &param_M)
{
    M1.resize(param_M, param_M);
    M2.resize(param_M, param_M);
    M3.resize(param_M, param_M);
    M1.setZero();
    M2.setZero();
    M3.setZero();
}

vector<mapPts*> map::GetAllMapPoints()
{
    vector<mapPts*> vMapPointsout = vMapPoints;
    return vMapPointsout;
}

vector<mapPts*> map::GetAllMapPoints(const bool &FlagRequire)
{
    vector<mapPts*> vPtsout;

    for(size_t i0=0;i0<vMapPoints.size(); i0++){
        if(vMapPoints[i0]->FLAG_BAD!=FlagRequire)
            continue;

        vPtsout.push_back(vMapPoints[i0]);
    }

    return vPtsout;
}

vector<mapText*> map::GetAllMapTexts()
{
    vector<mapText*> vMapTextObjsout = vMapTextObjs;
    return vMapTextObjsout;
}

vector<mapText*> map::GetAllMapTexts(const TextStatus &state)
{
     vector<mapText*> vMapTextObjsout;
    for(size_t i0=0;i0<vMapTextObjs.size(); i0++){
        if(vMapTextObjs[i0]->STATE==state){
            vMapTextObjsout.push_back(vMapTextObjs[i0]);
        }
    }

    return vMapTextObjsout;
}

vector<mapText*> map::GetObvTexts(const TextStatus &state, const vector<keyframe*> &vKFs)
{
    vector<mapText*> vTextOut;
    for(size_t ikfs = 0; ikfs<vKFs.size(); ikfs++){
        vector<TextObservation*> vObvTexts = vKFs[ikfs]->vObvText;
        for(size_t iobj = 0; iobj<vObvTexts.size(); iobj++){
            if(vObvTexts[iobj]->obj->STATE==state){
                vTextOut.push_back(vObvTexts[iobj]->obj);
            }
        }
    }

    sort(vTextOut.begin(), vTextOut.end());
    vector<mapText*>::iterator end = unique(vTextOut.begin(), vTextOut.end());
    vTextOut.erase(end, vTextOut.end());

    return vTextOut;
}

vector<keyframe*> map::GetNeighborKF(const frame &F)
{
    vector<keyframe*> Res;
    size_t IdxMin = -1;
    int DiffMin = INT_MAX;
    int FId = F.mnId;

    for(size_t i0 = 0; i0<vKeyframes.size(); i0++){
        int KFId = vKeyframes[i0]->mnFrameId;
        int IdDiff = FId-KFId;
        // 1. kframe is previous kf
        if(IdDiff<0){
            continue;
        }

        // 2. find closest kf and the 2nd closest kf
        if(IdDiff<DiffMin){
            DiffMin = IdDiff;
            IdxMin = i0;
        }
    }
    //  --------

    Res.push_back(vKeyframes[IdxMin]);      // [0]: minest
    Res.push_back(vKeyframes[IdxMin-1]);     // [1]: 2nd minest
    return Res;
}

vector<keyframe*> map::GetNeighborKF(const int &KFmnId, const int &Win)
{
    vector <keyframe*> vKFs;
    vector<keyframe*>::iterator iter1 = vKeyframes.begin() + KFmnId - Win + 1;
    vector<keyframe*>::iterator iter2 = vKeyframes.begin() + KFmnId + 1;
    copy(iter1, iter2, std::back_inserter(vKFs));
    return vKFs;
}

vector<keyframe*> map::GetAllKeyFrame()
{
    vector<keyframe*> vKeyframesout = vKeyframes;
    return vKeyframesout;
}

void map::Addkeyframe(keyframe* pKF)
{
    vKeyframes.push_back(pKF);
    imapkfs = vKeyframes.size();
}

void map::Addtextobjs(mapText* Textobj)
{
    vMapTextObjs.push_back(Textobj);
    imapText = vMapTextObjs.size();
}

void map::Addscenepts(mapPts *Scenepts)
{
    vMapPoints.push_back(Scenepts);
    imapPts = vMapPoints.size();
}

long unsigned int map::KeyFramesInMap()
{
    return vKeyframes.size();
}

keyframe* map::GetKFFromId(const int &KFmnId)
{
    keyframe* kfout = vKeyframes[KFmnId];
    assert((int)kfout->mnId==KFmnId);
    return kfout;
}

mapPts* map::GetPtFromId(const int &PtmnId)
{
    mapPts* ptout = vMapPoints[PtmnId];
    assert((int)ptout->mnId==PtmnId);
    return ptout;
}

mapText* map::GetObjFromId(const int &ObjmnId)
{
    mapText* objout = vMapTextObjs[ObjmnId];
    assert((int)objout->mnId==ObjmnId);
    return objout;
}

// for loop
void map::UpdateCovMap_1(keyframe* KF, mapPts *Scenepts)
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

void map::UpdateCovMap_2(keyframe* KF, mapText *TextObj)
{
    std::map<keyframe*,vector<int>> vKFs = TextObj->GetObservationKFs();
    std::map<keyframe*, vector<int>>::iterator iterkf;
    int IdKF2 = KF->mnId;
    for(iterkf = vKFs.begin(); iterkf != vKFs.end(); iterkf++){
        keyframe* kfobv = iterkf->first;
        int IdKF1 = kfobv->mnId;
        if(IdKF1>=IdKF2)
            continue;

        assert(IdKF1>=0);
        assert(IdKF2>=0);
        assert(IdKF1<M2.rows());
        assert(IdKF2<M2.cols());

        M2(IdKF1, IdKF2) += 1;
    }
}

void map::UpdateCovMap_3(keyframe* KF, mapText *TextObj)
{
    int numObjFeats = TextObj->vRefFeature[0].size();
    std::map<keyframe*,vector<int>> vKFs = TextObj->GetObservationKFs();
    std::map<keyframe*, vector<int>>::iterator iterkf;
    int IdKF2 = KF->mnId;
    for(iterkf = vKFs.begin(); iterkf != vKFs.end(); iterkf++){
        keyframe* kfobv = iterkf->first;
        int IdKF1 = kfobv->mnId;
        if(IdKF1>=IdKF2)
            continue;

        assert(IdKF1>=0);
        assert(IdKF2>=0);
        assert(IdKF1<M3.rows());
        assert(IdKF2<M3.cols());

        M3(IdKF1, IdKF2) += numObjFeats;
    }
}

vector<Eigen::MatrixXd> map::GetCovMap_All()
{
    vector<Eigen::MatrixXd> M_out;
    M_out.push_back(M1);
    M_out.push_back(M2);
    M_out.push_back(M3);
    return M_out;
}

Eigen::MatrixXd map::GetCovMap(const int &UseM)
{
    if(UseM==0)
    {
        Eigen::MatrixXd M1_out;
        M1_out = M1;
        return M1_out;
    }
    else if(UseM==1)
    {
        Eigen::MatrixXd M2_out;
        M2_out = M2;
        return M2_out;
    }
    else if(UseM==2)
    {
        Eigen::MatrixXd M3_out;
        M3_out = M3;
        return M3_out;
    }
}

Eigen::MatrixXd map::GetCovMap_1()
{
    Eigen::MatrixXd M1_out;
    M1_out = M1;
    return M1_out;
}
Eigen::MatrixXd map::GetCovMap_2()
{
    Eigen::MatrixXd M2_out;
    M2_out = M2;
    return M2_out;
}
Eigen::MatrixXd map::GetCovMap_3()
{
    Eigen::MatrixXd M3_out;
    M3_out = M3;
    return M3_out;
}

void map::SetCovMap_1(Eigen::MatrixXd &M1In)
{
    M1 = M1In;
}
void map::SetCovMap_2(Eigen::MatrixXd &M2In)
{
    M2 = M2In;
}
void map::SetCovMap_3(Eigen::MatrixXd &M3In)
{
    M3 = M3In;
}

}
