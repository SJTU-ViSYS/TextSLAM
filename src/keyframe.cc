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
#include <keyframe.h>

namespace TextSLAM
{
long unsigned int keyframe::nNextId=0;


keyframe::keyframe(frame &F):
    dTimeStamp(F.dTimeStamp), mnFrameId(F.mnId), FrameImg(F.FrameImg), mK(F.mK), fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mnMinX(F.mnMinX), mnMaxX(F.mnMaxX), mnMinY(F.mnMinY), mnMaxY(F.mnMaxY),
    mTcw(F.mTcw), mTwc(F.mTwc), mRcw(F.mRcw), mRwc(F.mRwc), mtcw(F.mtcw), mtwc(F.mtwc), mNcr(F.mNcr), vNGOOD(F.vNGOOD),
    vTextDete(F.vTextDete), iN(F.iN), iNScene(F.iNScene), iNTextObj(F.iNTextObj), iNTextFea(F.iNTextFea),
    vKeys(F.vKeys), vKeysScene(F.vKeysScene), vKeysText(F.vKeysText), mDescr(F.mDescr), mDescrScene(F.mDescrScene), mDescrText(F.mDescrText), vTextObjInfo(F.vTextObjInfo), vKeysTextTrack(F.vKeysTextTrack), vKeysSceneIdx(F.vKeysSceneIdx), vKeysTextIdx(F.vKeysTextIdx),
    vTextDeteMin(F.vTextDeteMin), vTextDeteMax(F.vTextDeteMax), vTextDeteCenter(F.vTextDeteCenter), vTextDeteCorMap(F.vTextDeteCorMap), vTextMean(F.vTextMean),
    iScaleLevels(F.iScaleLevels), dScaleFactor(F.dScaleFactor), vScaleFactors(F.vScaleFactors), vInvScaleFactors(F.vInvScaleFactors), vLevelSigma2(F.vLevelSigma2), vInvLevelSigma2(F.vInvLevelSigma2), vK_scale(F.vK_scale),
    vFrameImg(F.vFrameImg), vFrameGrad(F.vFrameGrad), vFrameGradX(F.vFrameGradX), vFrameGradY(F.vFrameGradY),
    vfeatureText(F.vfeatureText),
    IniScenePts3d(F.IniScenePts3d), IniSceneObv2d(F.IniSceneObv2d),
    vObvText(F.vObvText), vObvPts(F.vObvPts), vSceneObv2d(F.vSceneObv2d), vMatches2D3D(F.vMatches2D3D),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    CovisbleFrameId(F.CovisbleFrameId), cfLocalKF(F.cfLocalKF)
{
    mnId=nNextId++;
    vObvGoodPts = vector<bool>(vObvPts.size(), true);
    vObvGoodTexts = vector<bool>(vObvText.size(), true);
    for(size_t itext=0; itext<vObvText.size(); itext++){
        vObvGoodTextFeats.push_back(vObvText[itext]->obj->vRefFeatureSTATE);
    }

    // semantic need mGrid
    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

}



// add observation
void keyframe::AddTextObserv(mapText* textobj, int idx, int NumFeats)
{
    vector<int> DeteIdx;
    DeteIdx.push_back(idx);

    TextObservation* textobv = new TextObservation{textobj, DeteIdx};
    vObvText.push_back(textobv);

    // observation flag
    vObvGoodTexts.push_back(true);
    vector<bool> vflagIni = vector<bool>(NumFeats, true);
    vObvGoodTextFeats.push_back(vflagIni);
}

void keyframe::AddTextObserv(mapText* textobj, int NumFeats)
{
    vector<int> DeteIdx;

    TextObservation* textobv = new TextObservation{textobj, DeteIdx};
    vObvText.push_back(textobv);

    // observation flag
    vObvGoodTexts.push_back(true);
    vector<bool> vflagIni = vector<bool>(NumFeats, true);
    vObvGoodTextFeats.push_back(vflagIni);
}

void keyframe::AddSceneObservForInitial(mapPts* scenept, int idx)
{
    // add map points
    SceneObservation* ptsobv = new SceneObservation{scenept, idx};
    vObvPts.push_back(ptsobv);

    // observation flag
    vObvGoodPts.push_back(true);
}

void keyframe::AddSceneObserv(mapPts* scenept, int idx)
{
    // add map points
    SceneObservation* ptsobv = new SceneObservation{scenept, idx};
    vObvPts.push_back(ptsobv);

    // observation flag
    vObvGoodPts.push_back(true);

    // add observation
    int Idx2Raw = vObvPts.size()-1;
    Vec2 Observ2D = Vec2((double)vKeys[idx].pt.x, (double)vKeys[idx].pt.y);

    for(size_t ipyr=0; ipyr<vSceneObv2d.size(); ipyr++){
        Vec2 Observ2DPyr = Vec2( Observ2D(0)*vInvScaleFactors[ipyr], Observ2D(1)*vInvScaleFactors[ipyr] );
        SceneFeature* featureInPyr = new SceneFeature{Observ2DPyr(0), Observ2DPyr(1), Observ2DPyr, (int)ipyr, Idx2Raw};
        vSceneObv2d[ipyr].push_back(featureInPyr);
    }
}

// get all Needstate Text Objects
vector<TextObservation*> keyframe::GetStateTextObvs(const TextStatus &Needstate)
{
    vector<TextObservation*> vObvTextout;
    for(size_t iob=0; iob<vObvText.size(); iob++){
        if(vObvText[iob]->obj->STATE==Needstate)
            vObvTextout.push_back(vObvText[iob]);
    }

    return vObvTextout;
}

vector<TextObservation*> keyframe::GetStateTextObvs(const TextStatus &Needstate, vector<int> &vNew2Raw)
{
    vector<TextObservation*> vObvTextout;
    for(size_t iob=0; iob<vObvText.size(); iob++){
        if(vObvText[iob]->obj->STATE==Needstate){
            vObvTextout.push_back(vObvText[iob]);
            vNew2Raw.push_back(iob);
        }
    }

    return vObvTextout;
}

// get all Text Objects
vector<TextObservation*> keyframe::GetTextObvs()
{
    vector<TextObservation*> vObvTextout = vObvText;

    return vObvTextout;
}

// pose
void keyframe::SetPose(Mat44 &Tcw)
{
    mTcw = Tcw;
    UpdatePoseMatrices();
}
void keyframe::UpdatePoseMatrices()
{
    mRcw = mTcw.block<3,3>(0,0);
    mRwc = mRcw.transpose();
    mtcw = mTcw.block<3,1>(0,3);
    mtwc = -mRwc*mtcw;
    mTwc = mTcw.inverse();
}

void keyframe::SetN(Mat31 &N, int idx)
{
    bool DEBUG = false;
    if(!vNGOOD[idx] || idx<0 || N(0,0)==NAN || N(1,0)==NAN || N(2,0)==NAN || mNcr[idx](0,0)==NAN || mNcr[idx](1,0)==NAN || mNcr[idx](2,0)==NAN)
        cout<<"error: N update not good N. Or idx is negative."<<endl;

    if(DEBUG)
        cout<<"before optimization rho of N "<<idx<<" is: "<<mNcr[idx](0,0)<<", "<<mNcr[idx](1,0)<<", "<<mNcr[idx](2,0)<<endl;

    mNcr[idx] = N;
    if(DEBUG)
        cout<<"after optimization rho of N "<<idx<<" is: "<<mNcr[idx](0,0)<<", "<<mNcr[idx](1,0)<<", "<<mNcr[idx](2,0)<<endl;

}

Mat44 keyframe::GetTcw()
{
    Mat44 Tcwout = mTcw;
    return Tcwout;
}

Mat44 keyframe::GetTwc()
{
    Mat44 Twcout = mTwc;
    return Twcout;
}


int keyframe::TrackedMapPoints(const int &minObs)
{
    int nPoints=0;
    int nPoint_all=0;
    const bool bCheckObs = minObs>0;

    for(int i=0; i<vObvPts.size(); i++)
    {
        mapPts* pMP = vObvPts[i]->pt;

        if(bCheckObs)
        {
            nPoint_all++;
            if(pMP->GetObservNum()>=minObs)
                nPoints++;
        }
        else
            nPoints++;

    }

    return nPoints;
}


vector<size_t> keyframe::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(iN);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = vKeys[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

void keyframe::AddLoopEdge(keyframe* KF)
{
    vLoopEdges.push_back(KF);
}

vector<keyframe*> keyframe::GetLoopEdges()
{
    vector<keyframe*> vKFout;
    vKFout = vLoopEdges;
    return vKFout;
}


void keyframe::SetCovisibleKFsAll(vector<CovKF> &vKFConKFs)
{
    vCovisibleKFsAll = vKFConKFs;
}


void keyframe::SetCovisibleKFsPrev(vector<CovKF> &vKFConKFs)
{
    vCovisibleKFsPrev = vKFConKFs;
}


void keyframe::SetCovisibleKFsAfter(vector<CovKF> &vKFConKFs)
{
    vCovisibleKFsAfter = vKFConKFs;
}

int keyframe::GetCovisibleWeight(keyframe* KF)
{
    int CovisibleWeight = -1, num_test = 0;
    for(size_t i=0; i<vCovisibleKFsAll.size(); i++){
        if(vCovisibleKFsAll[i].first->mnId==KF->mnId){
            CovisibleWeight = vCovisibleKFsAll[i].second;
            num_test++;
        }
    }

    assert(num_test<=1);
    return CovisibleWeight;
}

// 1. get vCovisibleKFsAll
vector<CovKF> keyframe::GetCovisibleKFs_All()
{
    vector<CovKF> vCovisibleKFsOut = vCovisibleKFsAll;
    return vCovisibleKFsOut;
}

// 2. get vCovisibleKFsPrev
vector<CovKF> keyframe::GetCovisibleKFs_Prev()
{
    vector<CovKF> vCovisibleKFsOut = vCovisibleKFsPrev;
    return vCovisibleKFsOut;
}

// 3. get vCovisibleKFsAfter
vector<CovKF> keyframe::GetCovisibleKFs_After()
{
    vector<CovKF> vCovisibleKFsOut = vCovisibleKFsAfter;
    return vCovisibleKFsOut;
}

vector<CovKF> keyframe::GetTopCovisKFs(const int &TopN)
{
    vector<CovKF> vCovKFsOut;
    if(vCovisibleKFsAll.size()<TopN)
        vCovKFsOut = vCovisibleKFsAll;
    else{
        vCovKFsOut.insert( vCovKFsOut.end(), vCovisibleKFsAll.begin(), vCovisibleKFsAll.begin()+TopN );
    }

    return vCovKFsOut;
}

// (for loop)
void keyframe::ReplaceMapPt(const size_t &idx, mapPts* mPt)
{
    for(size_t i0=0; i0<vObvPts.size(); i0++)
    {
        if(vObvPts[i0]->idx!=idx)
            continue;

        // a) vObvPts
        vObvPts[i0]->pt = mPt;
        // b) vObvGoodPts
        vObvGoodPts[i0] = true;
    }

    vMatches2D3D[idx] = mPt->mnId;
}

// (for loop)
// use mObjReplace to replace mObj
void keyframe::ReplaceMapText(mapText* mObj, mapText* mObjReplace)
{
    vector<int> vIdx;
    for(size_t i0=0; i0<vObvText.size(); i0++)
    {
        if(vObvText[i0]->obj->mnId != mObj->mnId)
            continue;

        // a) vObvText
        vObvText[i0]->obj = mObjReplace;
        vIdx = vObvText[i0]->idx;
        // b) vObvGoodText, vObvGoodTextFeats initialization
        vObvGoodTexts[i0] = true;
        vObvGoodTextFeats[i0].clear();
        vObvGoodTextFeats[i0] = mObjReplace->vRefFeatureSTATE;

        break;
    }

    for(size_t i1=0; i1<vIdx.size(); i1++){
        int DeteIdx = vIdx[i1];
        assert(vTextDeteCorMap[DeteIdx]==mObj->mnId);
        vTextDeteCorMap[DeteIdx] = mObjReplace->mnId;
    }

}

bool keyframe::IsInImage(const double &x, const double &y)
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

}
