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
#include <mapText.h>
#include <keyframe.h>

namespace TextSLAM
{
long unsigned int mapText::nNextId=0;

// no semantic meaning
mapText::mapText(vector<Vec2> &TextDete, keyframe* kRefKF, vector<vector<TextFeature*>> &vfeatureObj, int Idx, TextStatus TextState):
    mnFirstKFid(kRefKF->mnId), mnFirstKFrameid(kRefKF->mnFrameId), iObs(0), ObvNum_good(0), ObvNum_bad(0), vRefFeature(vfeatureObj), NIdx(Idx), STATE(TextState), Flag_Replaced(false), LastObvFId(0)
{
    mnId=nNextId++;
    RefKF = kRefKF;
    NumObvs = 2;        // initial 2 KFs

    // feature STATE initial (size = pyr 0 feature number)
    vRefFeatureSTATE = vector<bool>(vRefFeature[0].size(), true);

    GetObjectInfo(TextDete);

    // (for loop)
    ReplaceKF = static_cast<keyframe*>(NULL);
    BeReplacedKF = static_cast<keyframe*>(NULL);
}

// add semantic meaning
mapText::mapText(vector<Vec2> &TextDete, keyframe* kRefKF, vector<vector<TextFeature *>> &vfeatureObj, int Idx, TextStatus TextState, TextInfo &_TextMean):
    mnFirstKFid(kRefKF->mnId), mnFirstKFrameid(kRefKF->mnFrameId), iObs(0), ObvNum_good(0), ObvNum_bad(0), vRefFeature(vfeatureObj), NIdx(Idx), STATE(TextState), TextMean(_TextMean), Flag_Replaced(false), LastObvFId(0)
{
    mnId=nNextId++;
    RefKF = kRefKF;
    NumObvs = 2;        // initial 2 KFs

    // feature STATE initial (size = pyr 0 feature number)
    vRefFeatureSTATE = vector<bool>(vRefFeature[0].size(), true);

    GetObjectInfo(TextDete);

    // (for loop)
    ReplaceKF = static_cast<keyframe*>(NULL);
    BeReplacedKF = static_cast<keyframe*>(NULL);

    // get initial semantic score
    CalInitialSemanticScore(_TextMean);
}


// get text object basic information: mu, sigma
void mapText::GetObjectInfo(vector<Vec2> &TextDete)
{
    int iLevels = RefKF->iScaleLevels;
    vector<double> vInvfactor = RefKF->vInvScaleFactors;
    vector<Mat33> vK = RefKF->vK_scale;

    // 1. get each pyramid text detection
    vTextDete.resize(iLevels);
    statistics.resize(iLevels);
    for(size_t i0 = 0; i0<iLevels; i0++){
        double fx = vK[i0](0,0), fy = vK[i0](1,1), cx = vK[i0](0,2), cy = vK[i0](1,2);

        // A) Get pyramid text detection
        vector<Vec2> vTextDetePyr;
        vTextDetePyr.push_back(Vec2(TextDete[0](0)*vInvfactor[i0], TextDete[0](1)*vInvfactor[i0]));
        vTextDetePyr.push_back(Vec2(TextDete[1](0)*vInvfactor[i0], TextDete[1](1)*vInvfactor[i0]));
        vTextDetePyr.push_back(Vec2(TextDete[2](0)*vInvfactor[i0], TextDete[2](1)*vInvfactor[i0]));
        vTextDetePyr.push_back(Vec2(TextDete[3](0)*vInvfactor[i0], TextDete[3](1)*vInvfactor[i0]));
        vTextDete[i0] = vTextDetePyr;

        vTextDetePyr.clear();
    }

    vTextDeteRay.push_back( Vec2( (vTextDete[0][0](0) - vK[0](0,2))/vK[0](0,0), (vTextDete[0][0](1) - vK[0](1,2))/vK[0](1,1) ) );
    vTextDeteRay.push_back( Vec2( (vTextDete[0][1](0) - vK[0](0,2))/vK[0](0,0), (vTextDete[0][1](1) - vK[0](1,2))/vK[0](1,1) ) );
    vTextDeteRay.push_back( Vec2( (vTextDete[0][2](0) - vK[0](0,2))/vK[0](0,0), (vTextDete[0][2](1) - vK[0](1,2))/vK[0](1,1) ) );
    vTextDeteRay.push_back( Vec2( (vTextDete[0][3](0) - vK[0](0,2))/vK[0](0,0), (vTextDete[0][3](1) - vK[0](1,2))/vK[0](1,1) ) );
    // 2. get text object in ref frame information: (mu, sigma) to (statistics[i1](0), statistics[i1](1))
    bool SHOW = false;
    for(size_t i1 = 0; i1<iLevels; i1++)
        Tool.CalTextinfo(RefKF->vFrameImg[i1], vTextDete[i1], statistics[i1](0), statistics[i1](1), SHOW);

    // 3. get refframe inten info ( feature inten & neighbour inten ) for pose estimation
    // vRefFeature info finish: neighbour, neighbourRay, IN, neighbourInten, neighbourNInten; featureNInten, INITIAL
    for(size_t i2 = 0; i2<iLevels; i2++)
        Tool.CalNormvec(RefKF->vFrameImg[i2], vRefFeature[i2], statistics[i2](0), statistics[i2](1), vK[i2]);

    // 4. get all pixels within box in pyramid 0 (vRefPixs)
    int usePym = 0;
    Tool.GetBoxAllPixs(RefKF->vFrameImg[usePym], vTextDete[usePym], usePym, vK[usePym], statistics[usePym](0), statistics[usePym](1), vRefPixs);

    // feature STATE initial (size = pyr 0 feature number)
    vRefFeatureSTATE = vector<bool>(vRefFeature[0].size(), true);
}

// get text info -- Nidx
int mapText::GetNidx()
{
   int Nidx = NIdx;
   return Nidx;
}

std::map<keyframe *, vector<int> > mapText::GetObservationKFs()
{
    std::map<keyframe*,vector<int>> vObvkeyframeout;
    vObvkeyframeout = vObvkeyframe;
    return vObvkeyframeout;
}


void mapText::AddObserv(keyframe* KF)
{
    vector<int> DeteIdx;

    vObvkeyframe[KF] = DeteIdx;
    iObs++;
}
void mapText::AddObserv(keyframe* KF, const int &idx)
{
    vector<int> DeteIdx;
    DeteIdx.push_back(idx);

    if(vObvkeyframe.count(KF)){
        cout<<"have insert keyframe."<<endl;
        vObvkeyframe[KF] = DeteIdx;
    }else{
        vObvkeyframe[KF] = DeteIdx;
        iObs++;
    }

}
void mapText::AddObserv(keyframe* KF, const vector<int> &vidx)
{
    if(vObvkeyframe.count(KF)){
        cout<<"have insert keyframe."<<endl;
    }
    vObvkeyframe[KF] = vidx;

    iObs++;

}

void mapText::AddNewObserv(keyframe* KF, vector<int> &vidx)
{
    vObvkeyframe[KF] = vidx;
    iObs++;
}

bool mapText::UpdateKFObserv(keyframe* KF, int DeteIdx)
{
    bool CHANGE = false;
    if(vObvkeyframe[KF].size()!=0){
        bool FLAG_HAVE = false;
        vector<int> vIdxRaw = vObvkeyframe[KF];
        for(size_t iIdx=0; iIdx<vIdxRaw.size(); iIdx++){
            if(vIdxRaw[iIdx]==DeteIdx){
                FLAG_HAVE = true;
                CHANGE = false;
                break;
            }
        }
        if(!FLAG_HAVE){
            vObvkeyframe[KF].push_back(DeteIdx);
            CHANGE = true;
        }
    }else{
        if(vObvkeyframe.count(KF))
            iObs++;
        vObvkeyframe[KF].push_back(DeteIdx);
        CHANGE = true;
    }

    return CHANGE;
}

bool mapText::GetObvIdx(keyframe* KF, vector<int> &vIdx)
{
    if(vObvkeyframe.count(KF)){
        vIdx = vObvkeyframe[KF];
        return true;
    }else
        return false;

}

void mapText::AddObvNum(const bool &ObvFlag)
{
    if(ObvFlag)
        ObvNum_good++;
    else
        ObvNum_bad++;
}


void mapText::Replace(keyframe* KFCur, mapText* mObj, Eigen::MatrixXd &M2, Eigen::MatrixXd &M3)
{
    // -------- settings --------
    double thScoreMean = 0.5;
    double thScoreDiff = 0.3;
    // -------- settings --------

    if(mObj->mnId==this->mnId){
        return;
    }

    // Step1. get observation;
    std::map<keyframe*,vector<int>> obs;
    obs = vObvkeyframe;

    // Step2. set flag; set new point pointer
    STATE = TEXTBAD;
    Flag_Replaced = true;

    ReplacedmObj = mObj;
    BeReplacedKF = KFCur;
    mObj->SetReplaceKF(KFCur);

    // Step3. use obj1 visible map to update covisble map
    for(std::map<keyframe*,vector<int>>::iterator mit0=obs.begin(); mit0!=obs.end(); mit0++)
    {
        keyframe* pKF = mit0->first;

        // add p2 observations (loop covisble connects)
        UpdateCovMap_2(pKF, mObj, M2);
        UpdateCovMap_3(pKF, mObj, M3);
    }

    // Step4. change related variable (KFs)
    for(std::map<keyframe*,vector<int>>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        keyframe* pKF = mit->first;

        if(!mObj->IsInKeyFrame(pKF))
        {
            // for KF observations
            pKF->ReplaceMapText(this, mObj);
            // for replaced text
            mObj->AddNewObserv(pKF,mit->second);
        }
    }

    // Step3. use multiple views to improve acc
    double dist = Tool.LevenshteinDist(TextMean.mean, mObj->TextMean.mean);
    int maxlen = max((int)TextMean.mean.length(), (int)mObj->TextMean.mean.length());
    double score = ((double)maxlen-dist)/(double)maxlen;
    double scoreDiff = TextMean.score-mObj->TextMean.score;
    if(score >=thScoreMean && scoreDiff>=thScoreDiff )
        mObj->TextMean = TextMean;

}

// tool
void mapText::SetReplaceKF(keyframe* KF)
{
    ReplaceKF =KF;
}

bool mapText::IsInKeyFrame(keyframe* KF)
{
    return (vObvkeyframe.count(KF));
}


void mapText::UpdateCovMap_2(keyframe* KF, mapText *TextObj, Eigen::MatrixXd &M2)
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

void mapText::UpdateCovMap_3(keyframe* KF, mapText *TextObj, Eigen::MatrixXd &M3)
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


void mapText::CalInitialSemanticScore(const TextInfo &TextInfoIni )
{
    double S_mean = (1.0-TextInfoIni.score)*200.0;
    double S_geo = GetSgeo(RefKF, this);
    double S_semantic = S_geo+S_mean;

    TextMean.score_semantic = S_semantic;
}


double mapText::GetSgeo(const keyframe* KF, mapText* obj)
{
    bool SHOW = false;
    double weight_view = 10.0;

    // oCam,zCam
    Mat31 oCam = KF->mtwc;
    Mat31 zCam = KF->mTwc.block<3,3>(0,0) * Mat31(0,0,1);

    Mat44 Tcr = KF->mTcw * obj->RefKF->GetTwc();
    Mat44 Twr = obj->RefKF->GetTwc();
    Mat31 Theta_r = obj->RefKF->mNcr[obj->GetNidx()];
    vector<Vec2> refDeteBox = obj->vTextDeteRay;
    vector<Mat31> Boxw;
    vector<Vec2> vPred;
    Mat31 oObj = Mat31(0,0,0);
    Boxw.resize(refDeteBox.size());
    vPred.resize(refDeteBox.size());
    assert((int)refDeteBox.size()==4);
    for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++){
        // 3D
        Tool.GetPtsText(refDeteBox[iBox], Theta_r, Boxw[iBox], Twr);
        oObj(0,0) += Boxw[iBox](0,0);
        oObj(1,0) += Boxw[iBox](1,0);
        oObj(2,0) += Boxw[iBox](2,0);

        // 2D
        Mat31 ray = Mat31(refDeteBox[iBox](0), refDeteBox[iBox](1), 1.0);
        bool IN = Tool.GetProjText(ray, Theta_r, vPred[iBox], Tcr, KF->mK);
    }
    oObj(0,0) = oObj(0,0)/refDeteBox.size();
    oObj(1,0) = oObj(1,0)/refDeteBox.size();
    oObj(2,0) = oObj(2,0)/refDeteBox.size();

    // d
    Mat31 d3D = oObj-oCam;
    double d = d3D.norm();

    // n
    Mat44 Trw = obj->RefKF->GetTcw();
    Mat31 Theta_w = Tool.TransTheta(Theta_r, Trw);
    zCam = zCam/zCam.norm();
    Theta_w = Theta_w/Theta_w.norm();
    double Cos = zCam.transpose()*Theta_w;

    double S_geo = (1.0 + Cos) * weight_view + d;

    if(SHOW){
        // show map objects
        string ShowName = "KF and Obj";
        Mat Imgdraw = KF->FrameImg.clone();
        Scalar Color = Scalar(0);
        line(Imgdraw, cv::Point2d(vPred[0](0), vPred[0](1)), cv::Point2d(vPred[1](0), vPred[1](1)), Color);
        line(Imgdraw, cv::Point2d(vPred[1](0), vPred[1](1)), cv::Point2d(vPred[2](0), vPred[2](1)), Color);
        line(Imgdraw, cv::Point2d(vPred[2](0), vPred[2](1)), cv::Point2d(vPred[3](0), vPred[3](1)), Color);
        line(Imgdraw, cv::Point2d(vPred[3](0), vPred[3](1)), cv::Point2d(vPred[0](0), vPred[0](1)), Color);

        // show massege
        stringstream s;
        s << "d is: "<<d<<"; (1+Cos) is: "<<(1.0+Cos)<<"; S_geo is: "<<S_geo;
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        cv::Mat imText = cv::Mat(500, 640, Imgdraw.type());
        Imgdraw.copyTo(imText.rowRange(0,480).colRange(0,640));
        imText.rowRange(480,500) = cv::Mat::zeros(textSize.height+10, 640, Imgdraw.type());
        cv::putText(imText, s.str(), cv::Point(5, imText.rows-5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255), 1, 8);

        // show
        namedWindow(ShowName, CV_WINDOW_NORMAL);
        imshow(ShowName, imText);
        waitKey(0);
    }

    return S_geo;
}

}
