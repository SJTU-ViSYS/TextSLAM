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
#include <tracking.h>

namespace TextSLAM
{

const int tracking::TH_HIGH = 100;
const int tracking::TH_LOW = 50;
const int tracking::HISTO_LENGTH = 30;

// general input
tracking::tracking(Mat33& K, map* Map, loopClosing* LClosing, setting* Setting):
    mpMap(Map), Set(Setting), LoopClosing(LClosing), mK(K), mState(NO_IMAGES_YET),ciInitializer(static_cast<initializer*>(NULL))
{
    int nFeatMax = 1000;
    float fScaleFactor = (float)1.2;
    int nLevels = (int)8;
    int fIniThFAST = (int)20;
    int fMinThFAST = (int)7;

    iPylevels = 8;
    dPyScales = 2.0;

    coORBextractor = new ORBextractor(nFeatMax,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    coORBextractorIni = new ORBextractor(nFeatMax*3,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    bool Flag_noText_opt = Setting->Flag_noText && (Setting->eExp_name==0);
    bool Flag_rapid_opt = Setting->eExp_name==4;
    coOptimizer = new optimizer(K, dPyScales, iPylevels, Flag_noText_opt, Flag_rapid_opt);

    // Loop (1:indoorloop1, 2:indoorloop2, 3:outdoor)
    iLastLoopKFid = -1;
    bNeed_Loop = (Set->eExp_name==1 || Set->eExp_name==2 || Set->eExp_name==3);

    // Keyframe Check
    mMaxFrames = (int)(Set->Fps/2);
    mMaxFramesMax = mMaxFrames+5;

    // LOG OR NOT
    FLAG_RECORD_TXT = false;

    cout<<" -------- Track basic info -------- "<<endl;
    cout<<"Check Keyframe use param (mMaxFrames) (mMaxFramesMax): "<<mMaxFrames<<", "<<mMaxFramesMax<<endl;
    cout<<"For optimization (Flag_noText_opt), (Flag_rapid_opt): "<<Flag_noText_opt<<", "<<Flag_rapid_opt<<endl;
    cout<<"Loop?: "<<bNeed_Loop<<endl;
    cout<<"Record?: "<<FLAG_RECORD_TXT<<endl;

}


int tracking::GrabImageMonocular(const cv::Mat &im, const double &ImgTimeStamp, const Mat33 &CameraK, const vector<vector<Vec2>> &TextDece, const vector<TextInfo> &TextMean)
{
    // 1. param initial
    cv::Mat ImTrack, ImTrack1;
    im.copyTo(ImTrack);
    if(Set->Flag_RGB==0)
        cvtColor(ImTrack, ImTrack, CV_BGR2GRAY);
    else if(Set->Flag_RGB==1)
        cvtColor(ImTrack, ImTrack, CV_RGB2GRAY);

    Width = ImTrack.cols;
    Height = ImTrack.rows;
    assert(Width==Set->Width);
    assert(Height==Set->Height);

    std::chrono::steady_clock::time_point t_cfbegin = std::chrono::steady_clock::now();

    // 2. get frame
    bool bVelocity;
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET){
        bVelocity = false;
        cfCurrentFrame = frame(ImTrack, ImgTimeStamp, CameraK, iPylevels, dPyScales, TextDece, TextMean, coORBextractorIni, bVelocity);
    }else{
        bVelocity = true;
        cfCurrentFrame = frame(ImTrack, ImgTimeStamp, CameraK, iPylevels, dPyScales, TextDece, TextMean, coORBextractor, bVelocity);
    }

    std::chrono::steady_clock::time_point t_cfend = std::chrono::steady_clock::now();
    double tF= std::chrono::duration_cast<std::chrono::duration<double> >(t_cfend - t_cfbegin).count();

    // 3. begin track
    Track();
    std::chrono::steady_clock::time_point t_trackend = std::chrono::steady_clock::now();
    double tTrack= std::chrono::duration_cast<std::chrono::duration<double> >(t_trackend - t_cfend).count();

    return 0;
}


void tracking::Track()
{
    bool FLAG_HASRECORD = false;
    // 1. initialized
    if(mState==NO_IMAGES_YET)
        mState = NOT_INITIALIZED;
    mLastProcessedState=mState;

    // 2. main proc
    if(mState==NOT_INITIALIZED){
        // A) No initialized -> INITIALIZE
        Initialization(FLAG_HASRECORD);

        if(mState!=OK)
            return;
    }else{
        // B.1) Has initialized -> GENERAL TRACKING
        bool FLAG_OK;
        if(cfLastFrame.bVelocity){
            FLAG_OK = TrackWithMotMod();
        }else{
            FLAG_OK = TrackWithOutMod();
        }

        // map reuse
        bool USETrackLocalMap = true;
        if(USETrackLocalMap && iLastLoopKFid>=0)
            TrackLocalMap();

        // update mapTextCondition
        LandmarkerObvUpdate();

        if(FLAG_OK)
            mState = OK;
        else
            mState=LOST;


        // ---- log ----
        if(mState==LOST){
            cout<<"error: track is lost."<<endl;
        }
        // ---- log ----

        // 2. param update
        if(FLAG_OK){
            mVelocity = cfCurrentFrame.mTcw * cfLastFrame.mTwc;
        }

        // B.2) keyframe and local BA
        if(CheckNewKeyFrame()){

            std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            TrackNewKeyframe();
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            double tNewKF= std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

            // ****************** LoopCLosure ******************
            bool UPDATE_TEXTFLAG = true;         // for map text good/bad
            bool UPDATE_TEXTMEAN = false;        // for map text semantic update
            UpdateSemantic_Condtions(UPDATE_TEXTFLAG, UPDATE_TEXTMEAN);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double tUpdateSemantic= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();


            if(bNeed_Loop && CHECKLOOP()){

                bool LoopRes = LoopClosing->Run(cfLastKeyframe, mpMap, coOptimizer);
                if(LoopRes){
                    iLastLoopKFid = cfLastKeyframe->mnId;
                }

            }
            std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
            double tLoop = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();

            cfCurrentFrame.SetPose(cfLastKeyframe->mTcw);

        }
        cfLastFrame = cfCurrentFrame;
    }

    // log and record
    if(!FLAG_HASRECORD&&FLAG_RECORD_TXT)
        RecordFrame(cfCurrentFrame);

}

void tracking::Initialization(bool &FLAG_HASRECORD)
{

    if(!ciInitializer){
        // A) find 1st initial frame
        if(cfCurrentFrame.iNScene>100 ){
            cfInitialFrame = frame(cfCurrentFrame);
            cfLastFrame = frame(cfCurrentFrame);
            cfLastIniFrame = frame(cfCurrentFrame);

            // initial param: sigma = 1.0; iteration = 200
            ciInitializer =  new initializer(cfCurrentFrame,1.0,200);
            fill(vIniMatches.begin(),vIniMatches.end(),-1);
            return;
        }
    }else{
        // B) 1st initial frame is ok. for 2nd initial frame
        cfLastIniFrame = frame(cfCurrentFrame);

        // B.1) find 2nd initial frame
        if(cfCurrentFrame.iN<=100 ){
            delete ciInitializer;
            ciInitializer = static_cast<initializer*>(nullptr);
            fill(vIniMatches.begin(), vIniMatches.end(),-1);
            return;
        }

        // B.2) find 2D matches
        int Win = (int)100;
        vector<int> vIniMatchesTest;
        int nMatches = SearchForInitializ(cfInitialFrame, cfCurrentFrame, Win, vIniMatches);

        if(nMatches<100){
            delete ciInitializer;
            ciInitializer = static_cast<initializer*>(nullptr);
            return;
        }

        // B.3) calculate relative pose and landmarker position
        cv::Mat Rcw;
        cv::Mat tcw;
        vector<bool> vbTriangulated;
        vector<cv::Point3f> IniP3D;

        if(ciInitializer->Initialize(cfCurrentFrame, vIniMatches, Rcw, tcw, IniP3D, vbTriangulated)){
            for(size_t i0 = 0; i0<vIniMatches.size(); i0++){
                if(vIniMatches[i0]>=0 && !vbTriangulated[i0]){
                    vIniMatches[i0] = -1;
                    nMatches--;
                }
            }


            // B.4) initial param setting
            Mat44 Tcw;
            Tcw.setIdentity();
            cfInitialFrame.SetPose(Tcw);
            ParamConvert(Rcw, tcw, Tcw);
            cfCurrentFrame.SetPose(Tcw);

            // B.5) establish initial map and global BA
            CreatInitialMap(IniP3D, ciInitializer);

            // B.6) record
            if(FLAG_RECORD_TXT){
                RecordFrame(cfInitialFrame);
                RecordFrame(cfCurrentFrame);
            }

            // B.7) update state
            mState=OK;
            FLAG_HASRECORD = true;
            cfLastFrame = cfCurrentFrame;
            cout<<"Initialize successfully."<<endl;

        }   // if 2nd initial frame ok

    }   // 2nd initial frame


}

void tracking::InitialLandmarker(const vector<cv::Point3f> &IniP3D, initializer* Initializ, keyframe* F1, keyframe* F2)
{
    // 1. get scene 3D invers depth define & its observation
    F1->vMatches2D3D = vector<int>(F1->iN, (int)-1);
    F2->vMatches2D3D = vector<int>(F2->iN, (int)-1);

    vector<vector<int>> KeysTextIdx;
    KeysTextIdx.resize(F1->iNTextObj);
    int iText = 0;
    for(size_t i0 = 0; i0<vIniMatches.size(); i0++){
        if(vIniMatches[i0]<0)
            continue;

        if(F1->vTextObjInfo[i0]>=0){
            int IdxTextObj = F1->vTextObjInfo[i0];
            KeysTextIdx[IdxTextObj].push_back(i0);
            iText++;
            continue;
        }

        int idx1 = i0;
        int idx2 = vIniMatches[i0];
        Vec3 pt;
        pt(0) = ((double)F1->vKeys[idx1].pt.x-F1->cx)/F1->fx;
        pt(1) = ((double)F1->vKeys[idx1].pt.y-F1->cy)/F1->fy;
        pt(2) = (double)1.0/(double)IniP3D[i0].z;

        Vec2 ptObv1, ptObv2;
        ptObv1(0) = F1->vKeys[idx1].pt.x;
        ptObv1(1) = F1->vKeys[idx1].pt.y;
        ptObv2(0) = F2->vKeys[idx2].pt.x;
        ptObv2(1) = F2->vKeys[idx2].pt.y;

        F1->IniScenePts3d.push_back(pt);
        F1->IniSceneObv2d.push_back(ptObv1);
        F2->IniSceneObv2d.push_back(ptObv2);
        // F1.IniScenePts3d is corresponding to F2.IniSceneObv2d

        // scene pts init
        mapPts* scenepts = new mapPts(pt, F1);
        // b) mapText add kf
        scenepts->AddObserv(F1, idx1);
        scenepts->AddObserv(F2, idx2);
        // c) kf add mapText observation
        F1->AddSceneObservForInitial(scenepts, idx1);
        F2->AddSceneObservForInitial(scenepts, idx2);
        F1->vMatches2D3D[idx1] = scenepts->mnId;
        F2->vMatches2D3D[idx2] = scenepts->mnId;
        // d) map add map pts
        mpMap->Addscenepts(scenepts);
        mpMap->UpdateCovMap_1(F1, scenepts);
        mpMap->UpdateCovMap_1(F2, scenepts);

    }
    Eigen::MatrixXd M1All = mpMap->GetCovMap_1();
    GetCovisibleKFs_all(F1, mpMap->imapkfs);
    GetCovisibleKFs_all(F2, mpMap->imapkfs);

    // 2 get scene pts pyramid features
    Tool.GetPyramidPts(F2->IniSceneObv2d, F2->vFrameImg, F2->vFrameGrad, F2->vInvScaleFactors, F2->vSceneObv2d);
    Tool.GetPyramidPts(F1->IniSceneObv2d, F1->vFrameImg, F1->vFrameGrad, F1->vInvScaleFactors, F1->vSceneObv2d);

    // 3. Get Text initial param
    vector<vector<Vec2>> TextKeys1, TextKeys2;
    vector<vector<Vec3>> TextKeys3D;
    for(size_t i1 = 0; i1<KeysTextIdx.size(); i1++){
        vector<int> KeysTextIdxCell = KeysTextIdx[i1];
        vector<Vec2> Keys1, Keys2;
        vector<Vec3> Keys3D;
        for(size_t i2 = 0; i2<KeysTextIdxCell.size(); i2++){
            int IdxObj = i1;
            int Idx1 = KeysTextIdxCell[i2];
            int Idx2 = vIniMatches[Idx1];
            cv::KeyPoint PtObv1 = F1->vKeys[Idx1];
            cv::KeyPoint PtObv2 = F2->vKeys[Idx2];
            Keys1.push_back(Vec2(PtObv1.pt.x, PtObv1.pt.y));
            Keys2.push_back(Vec2(PtObv2.pt.x, PtObv2.pt.y));
            Vec3 TextFeaInv;
            TextFeaInv(0) = ((double)F1->vKeys[Idx1].pt.x-F1->cx)/F1->fx;
            TextFeaInv(1) = ((double)F1->vKeys[Idx1].pt.y-F1->cy)/F1->fy;
            TextFeaInv(2) = 1.0/IniP3D[Idx1].z;
            Keys3D.push_back(Vec3(TextFeaInv));
        }
        TextKeys1.push_back(Keys1);
        TextKeys2.push_back(Keys2);
        TextKeys3D.push_back(Keys3D);
    }

    // N initial
    Initializ->InitialTextObjs(F1, F2, TextKeys3D, TextKeys2);

    // map text object init
    F1->vTextDeteCorMap = vector<int>(F1->vTextDete.size(), -1);
    for(size_t i4 = 0; i4<F1->iNTextObj; i4++){
        // a) if text detection is too small (<4 pixels)
        if(!F1->vNGOOD[i4]){
            continue;
        }

        // b) initial mapText
        mapText* textobj = new mapText(F1->vTextDete[i4], F1, F1->vfeatureText[i4], (int)i4, TEXTGOOD, F1->vTextMean[i4]);

        // c) mapText add kf
        textobj->AddObserv(F1, (int)i4);
        textobj->AddObserv(F2);

        // d) kf add mapText observation
        F1->AddTextObserv(textobj, (int)i4, (int)textobj->vRefFeature[0].size());
        F1->vTextDeteCorMap[i4] = textobj->mnId;
        F2->AddTextObserv(textobj, (int)textobj->vRefFeature[0].size());

        // e) map add text object
        mpMap->Addtextobjs(textobj);
        mpMap->UpdateCovMap_2(F1, textobj);
        mpMap->UpdateCovMap_2(F2, textobj);
        mpMap->UpdateCovMap_3(F1, textobj);
        mpMap->UpdateCovMap_3(F2, textobj);
    }

}

bool tracking::TrackWithMotMod()
{
    std::chrono::steady_clock::time_point t_cfbegin = std::chrono::steady_clock::now();

    cfCurrentFrame.SetPose(mVelocity*cfLastFrame.mTcw);

    // 1. track map scene pts
    std::chrono::steady_clock::time_point t1_S = std::chrono::steady_clock::now();

    // 1.1 match map points with current frame 2D keypoints
    vector<int> vMatch3D2D, vMatch2D3D;
    vector<mapPts*> vAllMappts = mpMap->GetAllMapPoints();
    vector<keyframe*> neighKFs = mpMap->GetNeighborKF(cfCurrentFrame);

    int th = 15;
    int nMatchS1 = SearchFrom3D(vAllMappts, cfCurrentFrame, vMatch3D2D, vMatch2D3D, th, neighKFs[0]);
    int nMatchS2 = SearchFrom3DAdd(vAllMappts, cfCurrentFrame, vMatch3D2D, vMatch2D3D, th, neighKFs[1]);
    float PnPth= 5.0;
    int Snmatches = nMatchS1 + nMatchS2;
    if(Snmatches>20)
        Snmatches = CheckMatch(vAllMappts, cfCurrentFrame, neighKFs, vMatch3D2D, PnPth);

    // 1.2 get frame scene points observation and its feature
    vector<Vec2> SceneObv2d;
    cfCurrentFrame.vObvPts = GetSceneObv(vAllMappts, vMatch3D2D, cfCurrentFrame, SceneObv2d);
    Tool.GetPyramidPts(SceneObv2d, cfCurrentFrame.vFrameImg, cfCurrentFrame.vFrameGrad, cfCurrentFrame.vInvScaleFactors, cfCurrentFrame.vSceneObv2d);

    std::chrono::steady_clock::time_point t2_S = std::chrono::steady_clock::now();
    double tS= std::chrono::duration_cast<std::chrono::duration<double> >(t2_S - t1_S).count();

    // 2. track map text objects
    vector<mapText*> FObvText = GetTextObvFromNeighKFs(neighKFs);
    FObvText = TextJudge(FObvText);
    // all text objects in this frame 0 pyramid pts number.
    int Num_textFeat = 0;
    for(size_t iFtext = 0; iFtext<FObvText.size(); iFtext++){
        cfCurrentFrame.AddTextObserv(FObvText[iFtext]);
        Num_textFeat += FObvText[iFtext]->vRefFeature[0].size();

        // observation flag
        if(FObvText[iFtext]->STATE!=TEXTBAD)
            cfCurrentFrame.vObvGoodTexts.push_back(true);
        else
            cfCurrentFrame.vObvGoodTexts.push_back(false);
        cfCurrentFrame.vObvGoodTextFeats.push_back(FObvText[iFtext]->vRefFeatureSTATE);

    }

    std::chrono::steady_clock::time_point t2_T = std::chrono::steady_clock::now();
    double tT= std::chrono::duration_cast<std::chrono::duration<double> >(t2_T - t2_S).count();

    // 3. pose optimization (select all good map text from observation)
    coOptimizer->PoseOptim(cfCurrentFrame);

    std::chrono::steady_clock::time_point t_OPT = std::chrono::steady_clock::now();
    double tOPT= std::chrono::duration_cast<std::chrono::duration<double> >(t_OPT - t2_T).count();

    // 4. update ummature text objects && track potential new text objects
    TrackNewTextFeat();

    // update immature text objects
    vector<mapText*> vMapTextsUn = mpMap->GetAllMapTexts(TEXTIMMATURE);
    TextUpdate(vMapTextsUn);

    // 5. record each text detection corresponding map text object
    if(FLAG_RECORD_TXT){
        RecordTextObvs(cfCurrentFrame);
    }
    iMatches = Snmatches;
    vFMatches2D3D = cfCurrentFrame.vMatches2D3D;

    std::chrono::steady_clock::time_point t_cfend = std::chrono::steady_clock::now();
    double tF= std::chrono::duration_cast<std::chrono::duration<double> >(t_cfend - t_cfbegin).count();

    return (Snmatches+Num_textFeat)>20;
}

bool tracking::TrackWithOutMod()
{
    std::chrono::steady_clock::time_point t_cfbegin = std::chrono::steady_clock::now();

    cfCurrentFrame.SetPose(cfLastFrame.mTcw);

    // 1. track map scene pts
    std::chrono::steady_clock::time_point t1_S = std::chrono::steady_clock::now();
    // 1.1 match map points with current frame 2D keypoints
    vector<int> vMatch3D2D, vMatch2D3D;
    vector<mapPts*> vAllMappts = mpMap->GetAllMapPoints();
    vector<keyframe*> neighKFs = mpMap->GetNeighborKF(cfCurrentFrame);
    int th = 15;
    int nMatchS1 = SearchFrom3D(vAllMappts, cfCurrentFrame, vMatch3D2D, vMatch2D3D, th, neighKFs[0]);      // search with the nearest keyframe, Match info initial here
    int nMatchS2 = SearchFrom3DAdd(vAllMappts, cfCurrentFrame, vMatch3D2D, vMatch2D3D, th, neighKFs[1]);   // search with the keyframe before the nearest keyframe
    float PnPth=4.0;
    int Snmatches = nMatchS1 + nMatchS2;
    if(Snmatches>20)
        Snmatches = CheckMatch(vAllMappts, cfCurrentFrame, neighKFs, vMatch3D2D, PnPth);
    // 1.2 get frame scene points observation and its feature
    vector<Vec2> SceneObv2d;
    cfCurrentFrame.vObvPts = GetSceneObv(vAllMappts, vMatch3D2D, cfCurrentFrame, SceneObv2d);
    Tool.GetPyramidPts(SceneObv2d, cfCurrentFrame.vFrameImg, cfCurrentFrame.vFrameGrad, cfCurrentFrame.vInvScaleFactors, cfCurrentFrame.vSceneObv2d);

    std::chrono::steady_clock::time_point t2_S = std::chrono::steady_clock::now();
    double tS= std::chrono::duration_cast<std::chrono::duration<double> >(t2_S - t1_S).count();

    // 2. track map text objects
    vector<mapText*> FObvText = GetTextObvFromNeighKFs(neighKFs);
    // delete not good text obj
    FObvText = TextJudge(FObvText);
    // all text objects in this frame 0 pyramid pts number.
    int Num_textFeat = 0;
    for(size_t iFtext = 0; iFtext<FObvText.size(); iFtext++){
        cfCurrentFrame.AddTextObserv(FObvText[iFtext]);
        Num_textFeat += FObvText[iFtext]->vRefFeature[0].size();

        // observation flag
        if(FObvText[iFtext]->STATE!=TEXTBAD)
            cfCurrentFrame.vObvGoodTexts.push_back(true);
        else
            cfCurrentFrame.vObvGoodTexts.push_back(false);
        cfCurrentFrame.vObvGoodTextFeats.push_back(FObvText[iFtext]->vRefFeatureSTATE);

    }
    std::chrono::steady_clock::time_point t2_T = std::chrono::steady_clock::now();
    double tT= std::chrono::duration_cast<std::chrono::duration<double> >(t2_T - t2_S).count();

    // 3. pose optimization (select all good map text from observation)
    coOptimizer->PoseOptim(cfCurrentFrame);

    std::chrono::steady_clock::time_point t_OPT = std::chrono::steady_clock::now();
    double tOPT= std::chrono::duration_cast<std::chrono::duration<double> >(t_OPT - t2_T).count();

    // 4. update ummature text objects && track potential new text objects
    TrackNewTextFeat();

    // update immature text objects
    TextUpdate(FObvText);

    // 5. record each text detection corresponding map text object
    if(FLAG_RECORD_TXT){
        RecordTextObvs(cfCurrentFrame);
    }
    iMatches = Snmatches;
    vFMatches2D3D = cfCurrentFrame.vMatches2D3D;

    std::chrono::steady_clock::time_point t_cfend = std::chrono::steady_clock::now();
    double tF= std::chrono::duration_cast<std::chrono::duration<double> >(t_cfend - t_cfbegin).count();

    return (Num_textFeat+Snmatches)>20;

}

void tracking::TrackLocalMap()
{
    bool UseTRackLocalMap = true;

    std::chrono::steady_clock::time_point t_cfbegin = std::chrono::steady_clock::now();

    UpdateLocalMap();

    SearchLocalLandmarkers();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    double tParam= std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t_cfbegin).count();

    // Optimize Pose
    if(UseTRackLocalMap)
        coOptimizer->PoseOptim(cfCurrentFrame);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double tOPT= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

    // update ummature text objects && track potential new text objects
    TrackNewTextFeat();

    // update immature text objects
    vector<mapText*> FObvText;
    for(size_t iobj=0; iobj<cfCurrentFrame.vObvText.size(); iobj++)
        FObvText.push_back(cfCurrentFrame.vObvText[iobj]->obj);
    TextUpdate(FObvText);

    // record each text detection corresponding map text object
    if(FLAG_RECORD_TXT){
        RecordTextObvs(cfCurrentFrame);
    }
    iMatches = cfCurrentFrame.vObvPts.size();
    vFMatches2D3D = cfCurrentFrame.vMatches2D3D;

    std::chrono::steady_clock::time_point t_cfend = std::chrono::steady_clock::now();
    double tF= std::chrono::duration_cast<std::chrono::duration<double> >(t_cfend - t_cfbegin).count();
    double tProc= std::chrono::duration_cast<std::chrono::duration<double> >(t_cfend - t2).count();

}

void tracking::UpdateLocalMap()
{
    UpdateLocalKFs();
    UpdateLocalLandmarkers();
}

void tracking::SearchLocalLandmarkers()
{
    // Scene Pts
    SearchLocalPts();

    // Text Object
    SearchLocalObjs();

}

void tracking::SearchLocalPts()
{
    // Step1. judge mapPts which have already been matched
    vector<SceneObservation*> vObvPtsF = cfCurrentFrame.vObvPts;
    vector<bool> vObvPtsFflag = cfCurrentFrame.vObvGoodPts;

    for(size_t ipt=0; ipt<vObvPtsF.size(); ipt++)
    {
        mapPts* mPt = vObvPtsF[ipt]->pt;
        mPt->LastObvFId = cfCurrentFrame.mnId;
    }

    // Step2. Project new local pts in current F
    int nToMatch=0, nAll = 0;
    std::map<mapPts*, keyframe*>::const_iterator IterPts;
    for(IterPts=cvlLocalPts.begin(); IterPts!=cvlLocalPts.end(); IterPts++)
    {
        mapPts* mPt = IterPts->first;
        mPt->Flag_LocalTrack = false;

        if(mPt->LastObvFId == cfCurrentFrame.mnId)
            continue;
        if(mPt->FLAG_BAD)
            continue;

        nAll++;
        if(ProjIsInFrame(cfCurrentFrame, mPt)){
            mPt->Flag_LocalTrack = true;
            nToMatch++;
        }
    }

    // Step3. match & add observation
    if(nToMatch>0)
    {
        int th = 2;
        int MatchNum = SearchFrom3DLocalTrack(cvlLocalPts, cfCurrentFrame, th);
    }

}

void tracking::SearchLocalObjs()
{
    // ------ Set ------
    int Pym = 0;
    int threshOut = 6;
    double threshcos = 0.5;
    double threshZNCC = 0.8;
    // ------ Set ------

    // Step1. judge mapObjs which have already been matched
    vector<TextObservation*> vObvObjF = cfCurrentFrame.vObvText;
    vector<bool> vObvObjFflag = cfCurrentFrame.vObvGoodPts;
    for(size_t iobj=0; iobj<vObvObjF.size(); iobj++)
    {
        mapText* mObj = vObvObjF[iobj]->obj;
        mObj->LastObvFId = cfCurrentFrame.mnId;
    }

    // Step2. Project new local objs in current F, in/out, use text judge function
    int nToMatch=0, nAll = 0;
    cv::Mat Imgdraw = cfCurrentFrame.vFrameImg[Pym].clone();
    set<mapText*>::const_iterator IterObj;
    for(IterObj=cvlLocalObjs.begin(); IterObj!=cvlLocalObjs.end(); IterObj++)
    {
        mapText* mObj = (*IterObj);

        if(mObj->LastObvFId == cfCurrentFrame.mnId)
            continue;
        if(mObj->STATE==TEXTBAD)
            continue;

        nAll++;
        vector<int> vIdxDete;
        bool ObjIN = TextJudgeSingle(mObj, threshcos, threshOut, threshZNCC, Pym, vIdxDete);

        // if IN, add observation
        if(ObjIN){
            cfCurrentFrame.AddTextObserv(mObj, mObj->vRefFeature[0].size(), vIdxDete);
            nToMatch++;
        }
    }


}

void tracking::UpdateLocalKFs()
{
    std::map<keyframe*,int> KFCounter;
    bool UseScene = true;

    if(UseScene)
    {
        vector<SceneObservation*> vObvMapPts = cfCurrentFrame.vObvPts;
        vector<bool> vObvFlag = cfCurrentFrame.vObvGoodPts;
        for(size_t iscene=0; iscene<vObvMapPts.size(); iscene++)
        {
            mapPts* MpPt = vObvMapPts[iscene]->pt;
            if(!vObvFlag[iscene])
                continue;

            std::map<keyframe*, size_t> vObservations = MpPt->GetObservationKFs();
            for(std::map<keyframe*, size_t>::const_iterator it=vObservations.begin(), itend=vObservations.end(); it!=itend; it++)
                KFCounter[it->first]++;
        }
    }

    if(KFCounter.empty())
        return;

    int max=0;
    keyframe* pKFmax= static_cast<keyframe*>(NULL);

    cvkLocalKFs.clear();
    cvkLocalKFs.reserve(3*KFCounter.size());

    for(std::map<keyframe*, int>::const_iterator it=KFCounter.begin(), itEnd=KFCounter.end(); it!=itEnd; it++)
    {
        keyframe* pKF = it->first;


        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        cvkLocalKFs.push_back(it->first);
        pKF->CovisbleFrameId = cfCurrentFrame.mnId;
    }

    for(vector<keyframe*>::const_iterator itKF=cvkLocalKFs.begin(), itEndKF=cvkLocalKFs.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(cvkLocalKFs.size()>80)
            break;

        keyframe* pKF = *itKF;

        const vector<CovKF> vNeighs = pKF->GetTopCovisKFs(10);

        for(vector<CovKF>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            keyframe* pNeighKF = itNeighKF->first;

            if(pNeighKF->CovisbleFrameId!=cfCurrentFrame.mnId)
            {
                cvkLocalKFs.push_back(pNeighKF);
                pNeighKF->CovisbleFrameId = cfCurrentFrame.mnId;
                break;
            }
        }
    }

    if(pKFmax)
    {
        cfLocalKF = pKFmax;
        cfCurrentFrame.cfLocalKF = cfLocalKF;
    }

}

void tracking::UpdateLocalLandmarkers()
{
    cvlLocalPts.clear();
    cvlLocalObjs.clear();

    for(size_t iKF = 0; iKF<cvkLocalKFs.size(); iKF++)
    {
        keyframe* KF = cvkLocalKFs[iKF];
        vector<SceneObservation*> vKFPts = KF->vObvPts;
        vector<TextObservation*> vKFTexts = KF->vObvText;

        for(size_t iPt=0; iPt<vKFPts.size(); iPt++){
            mapPts* Mpt = vKFPts[iPt]->pt;
            if(Mpt->FLAG_BAD)
                continue;

            if(!cvlLocalPts.count(Mpt))
                cvlLocalPts.insert(make_pair(Mpt, KF));
        }

        for(size_t iObj=0; iObj<vKFTexts.size(); iObj++){
            mapText* Mobj = vKFTexts[iObj]->obj;
            if(Mobj->STATE==TEXTBAD)
                continue;

            if(!cvlLocalObjs.count(Mobj))
                cvlLocalObjs.insert(Mobj);
        }
    }

}


void tracking::TrackNewKeyframe()
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 1. use previous map points matches. Add New points directly.
    int WinSearchPts = 80;
    vector<int> vNewptsMatch12;
    int iNewScene =  SearchForTriangular(cfLastKeyframe, cfCurrentFrame, WinSearchPts, vNewptsMatch12);

    vector<Mat31> vNewpts;
    vector<match> vNewptsMatch;
    if(iNewScene!=0){
        iNewScene = GetForTriangular(cfLastKeyframe, cfCurrentFrame, vNewptsMatch12, vNewpts, vNewptsMatch);
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double tSNew= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

    // 2. add keyframe
    keyframe* CurKF = new keyframe(cfCurrentFrame);
    mpMap->Addkeyframe(CurKF);

    InitialLandmarkerInKF_Text1(CurKF);

    InitialLandmarkerInKF_Scene(CurKF, vNewpts, vNewptsMatch);
    assert(CurKF->vObvGoodPts.size() == CurKF->vObvPts.size());
    assert(CurKF->vObvGoodTexts.size() == CurKF->vObvText.size());
    assert(CurKF->vObvGoodTextFeats.size() == CurKF->vObvText.size());

    // 3. local BA
    int SlidingWindow = 20;
    vector<keyframe*> vKFs;
    if(CurKF->mnId<=SlidingWindow-1){
        vKFs = mpMap->GetAllKeyFrame();
        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        coOptimizer->LocalBundleAdjustment(mpMap, vKFs, NOTREACHWIN);
        std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
        double tBA= std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t4).count();
    }else{
        vKFs = mpMap->GetNeighborKF(CurKF->mnId, SlidingWindow);
        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        coOptimizer->LocalBundleAdjustment(mpMap, vKFs, LOCAL);
        std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
        double tBA= std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t4).count();
    }

    // ---- update map pts conditions based on keyframe BA results ----
    mpPtsCondUpdate(vKFs);
    // ---- update map pts conditions based on keyframe BA results ----

    // 4. text info get
    std::chrono::steady_clock::time_point tNewText1 = std::chrono::steady_clock::now();
    InitialTextObjs();
    std::chrono::steady_clock::time_point tNewText2 = std::chrono::steady_clock::now();
    double tTNew= std::chrono::duration_cast<std::chrono::duration<double> >(tNewText2 - tNewText1).count();

    vector<mapText*> vNewText;
    InitialLandmarkerInKF_Text2(CurKF, vNewText);

    // update Text track using immature text objects
    cv::Mat ImTextLabel = GetTextLabelImg(CurKF, TEXTIMMATURE);
    UpdateImTextTrack(ImTextLabel, CurKF);

    // log and update lastest cfLastKeyframe
    RecordKeyFrame_latest();
    if(FLAG_RECORD_TXT){
        RecordKeyFrame();
        RecordLandmarks();
        RecordTextObvs_KFfull(CurKF);
    }
    InitialNewTextFeatForTrack(CurKF);
    cfLastKeyframe = CurKF;

}

bool tracking::CheckNewKeyFrame()
{
    const int nKFs = mpMap->KeyFramesInMap();

    vector<keyframe*> neighKFs = mpMap->GetNeighborKF(cfCurrentFrame);
    keyframe* neighKF = neighKFs[0];

    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nPoints = neighKF->TrackedMapPoints(nMinObs);
    float thRefRatio = 0.9f;

    const bool c1 = cfCurrentFrame.mnId >= cfLastKeyframe->mnFrameId + mMaxFrames;
    const bool c2 = ((iMatches < nPoints*thRefRatio) && iMatches>15);
    const bool c3 = cfCurrentFrame.mnId >= cfLastKeyframe->mnFrameId + mMaxFramesMax;

    if((c1||c2) || c3)
    {
        return true;
    }
    else{
        return false;
    }

}

bool tracking::CHECKLOOP()
{
    bool NEEDLOOP = false;

    bool c0 = mpMap->imapkfs>=20;
    bool c1 = cfLastKeyframe->mnId > (iLastLoopKFid+20);
    if(iLastLoopKFid!=-1)
        NEEDLOOP = c0 && c1;
    else
        NEEDLOOP = c0;

    return NEEDLOOP;
}


void tracking::InitialLandmarkerInKF_Text1(keyframe* KF)
{
    // Add old text observation.
    for(size_t iold = 0; iold<KF->vObvText.size(); iold++){
        mapText* mpText = KF->vObvText[iold]->obj;
        mpText->AddObserv(KF, KF->vObvText[iold]->idx);

        // semantic update (only for TEXTIMMATURE/TEXTGOOD)
        if(mpText->STATE!=TEXTBAD)
            UpdateSemantic_MapObjs_single(mpText, KF);

        mpMap->UpdateCovMap_2(KF, mpText);
        mpMap->UpdateCovMap_3(KF, mpText);
    }

}

void tracking::InitialLandmarkerInKF_Text2(keyframe* KF, vector<mapText *> &vNewText)
{
    int numtext = 0;
    for(size_t i0=0; i0<cfLastKeyframe->vNGOOD.size(); i0++){
        if(!cfLastKeyframe->vNGOOD[i0])
            continue;

        mapText* textobj = new mapText(cfLastKeyframe->vTextDete[i0], cfLastKeyframe, cfLastKeyframe->vfeatureText[i0], (int)i0, TEXTIMMATURE, cfLastKeyframe->vTextMean[i0]);

        textobj->AddObserv(cfLastKeyframe, (int)i0);
        textobj->AddObserv(KF);

        cfLastKeyframe->AddTextObserv(textobj, (int)i0, (int)textobj->vRefFeature[0].size());
        cfLastKeyframe->vTextDeteCorMap[i0] = textobj->mnId;
        KF->AddTextObserv(textobj, (int)textobj->vRefFeature[0].size());

        mpMap->Addtextobjs(textobj);

        mpMap->UpdateCovMap_2(cfLastKeyframe, textobj);
        mpMap->UpdateCovMap_2(KF, textobj);
        mpMap->UpdateCovMap_3(cfLastKeyframe, textobj);
        mpMap->UpdateCovMap_3(KF, textobj);

        numtext++;
        vNewText.push_back(textobj);
    }

}


void tracking::InitialLandmarkerInKF_Scene(keyframe* KF, const vector<Mat31> &vnewpts, const vector<match> &vnewptsmatch)
{
    int numscene = 0;

    vector<mapPts*> vTmpPts = mpMap->GetAllMapPoints();

    for(size_t i0=0; i0<KF->vObvPts.size(); i0++){
        mapPts* mpPt = KF->vObvPts[i0]->pt;

        mpPt->AddObserv(KF, KF->vObvPts[i0]->idx);
        mpMap->UpdateCovMap_1(KF, mpPt);
    }

    for(size_t i1=0; i1<vnewpts.size(); i1++){
        Mat31 pt3DRaw = vnewpts[i1];
        Mat31 pt3D = cfLastKeyframe->mTcw.block<3,3>(0,0) * pt3DRaw + cfLastKeyframe->mTcw.block<3,1>(0,3);
        int idx1 = vnewptsmatch[i1].first;
        int idx2 = vnewptsmatch[i1].second;
        Vec3 pt;
        pt(0) = ((double)cfLastKeyframe->vKeys[idx1].pt.x-KF->cx)/KF->fx;
        pt(1) = ((double)cfLastKeyframe->vKeys[idx1].pt.y-KF->cy)/KF->fy;
        pt(2) = (double)1.0/(double)pt3D(2,0);

        // scene pts init
        mapPts* scenepts = new mapPts(pt, cfLastKeyframe);
        // mapText add kf
        scenepts->AddObserv(cfLastKeyframe, idx1);
        scenepts->AddObserv(KF, idx2);
        // kf add mapText observation
        cfLastKeyframe->AddSceneObserv(scenepts, idx1);
        KF->AddSceneObserv(scenepts, idx2);
        cfLastKeyframe->vMatches2D3D[idx1] = scenepts->mnId;
        KF->vMatches2D3D[idx2] = scenepts->mnId;
        // map add map pts
        mpMap->Addscenepts(scenepts);
        mpMap->UpdateCovMap_1(cfLastKeyframe, scenepts);
        mpMap->UpdateCovMap_1(KF, scenepts);

        numscene++;
    }

    vTmpPts = mpMap->GetAllMapPoints();

    GetCovisibleKFs_all(KF, mpMap->imapkfs);

}

void tracking::CreatInitialMap(vector<cv::Point3f> &IniP3D, initializer* Initializ)
{
    // keyframe init
    keyframe* pKFini = new keyframe(cfInitialFrame);
    keyframe* pKFcur = new keyframe(cfCurrentFrame);
    mpMap->Addkeyframe(pKFini);
    mpMap->Addkeyframe(pKFcur);

    // scene pts and text object init
    InitialLandmarker(IniP3D, Initializ, pKFini, pKFcur);

    // initial global BA
    coOptimizer->InitBA(pKFini, pKFcur);

    cfCurrentFrame.SetPose(pKFcur->mTcw);

    // update state
    InitialNewTextFeatForTrack(pKFcur);
    cfLastKeyframe = pKFcur;

    if(FLAG_RECORD_TXT){
        RecordTextObvs_KFfull(pKFini);
        RecordTextObvs_KFfull(pKFcur);
    }
}

/*
 * func: SearchForInitializ
 * param In:
 * frame &F1: Initial 1st frame; const frame &F2: Initial 2nd frame; const int &Win: Search range in 2D image;
 * param Out:
 * vector<int> vMatchIdx12: each F1 feature correspondings in F2. vMatchIdx12.size() = F1.iN
 * ----
 * return:
 * matches number
 */
int tracking::SearchForInitializ(const frame &F1, const frame &F2, const int &Win, vector<int> &vMatchIdx12)
{

    // 1. param initialize
    int nMatches = 0;
    vMatchIdx12 = vector<int>(F1.iN, -1);
    vector<int> vMatchDist(F2.iN, INT_MAX);
    vector<int> vMatchIdx21(F2.iN, -1);

    int iText = 0, iScene = 0;
    // 2. search the corresponding features around the 2D locations(F1.keys) within the WinSize(Win)
    for(size_t i1 = 0; i1<F1.iN; i1++){
        cv::KeyPoint Fea = F1.vKeys[i1];

        // Cond1. pyramid octave must be 0 level
        if(Fea.octave>0)
            continue;

        // Cond2. F1 F2 feature distance min
        int PyrLevel1 = Fea.octave;
        vector<size_t> vF2Candidate = F2.GetFeaturesInArea(Fea.pt.x, Fea.pt.y, Win, PyrLevel1, PyrLevel1);
        if(vF2Candidate.empty())
            continue;

        vector<size_t> Idx1Show;
        Idx1Show.push_back(i1);

        cv::Mat d1 = F1.mDescr.row(i1);
        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vF2Candidate.begin(); vit!=vF2Candidate.end(); vit++){
            size_t i2 = *vit;
            cv::Mat d2 = F2.mDescr.row(i2);
            int dist = DescriptorDistance(d1, d2);

            if(vMatchDist[i2]<=dist)
                continue;

            if(dist<bestDist){
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            }else if(dist<bestDist2){
                bestDist2 = dist;
            }
         }       // F2 candidate

        if(bestDist<=TH_LOW){
            if(bestDist<(double)bestDist2*0.9){
                if(vMatchIdx21[bestIdx2]>=0){                   // if this F2 feature already corresponding to the F1 feature, the current corresponding is remained
                    vMatchIdx12[vMatchIdx21[bestIdx2]] = -1;
                    nMatches--;
                }
                vMatchIdx12[i1] = bestIdx2;
                vMatchIdx21[bestIdx2] = i1;
                vMatchDist[bestIdx2] = bestDist;
                nMatches++;
            }
        }   // Dist judge
    }       // F1 feature

   return nMatches;
}

// input: vector<mapPts*> Mappts: all map points to project; frame &F: the frame to find corresponding features; int th: search window around projection; int flag use which keyframe: 0->the nearest keyframe, 1->the keyframe before the nearest keyframe
// output: vector<int> match3D2D&match2D3D: match based on 3Dto2D & 2Dto3D
// return: all match number
int tracking::SearchFrom3D(const vector<mapPts*> Pts3d, const frame &F, vector<int> &vMatch3D2D, vector<int> &vMatch2D3D, const int &th, keyframe* F1)
{
    vector<int> vMatch12;
    vMatch12 = vector<int>(F1->iN, -1);
    vMatch3D2D = vector<int>(Pts3d.size(), -1);
    vMatch2D3D = vector<int>(F.iN, -1);

    int nMatches = 0;
    Mat44 Tcw = F.mTcw;

    for(size_t i0 = 0; i0<Pts3d.size(); i0++){

        if(Pts3d[i0]->FLAG_BAD)
            continue;

        mapPts* pt = Pts3d[i0];
        int IdxObserv;
        if(!pt->GetKFObv(F1, IdxObserv))
            continue;

        keyframe* RefKF = pt->RefKF;
        Mat44 Trw = RefKF->mTcw;
        Mat44 Tcr = Tcw*Trw.inverse();
        Mat33 Rcr = Tcr.block<3,3>(0,0);
        Mat31 tcr = Tcr.block<3,1>(0,3);

        Mat31 raydir = pt->GetRaydir();
        double rho = pt->GetInverD();
        double invrho = 1.0/rho;

        Mat31 PtTar = F.mK * (invrho * Rcr * raydir + tcr);

        double u = PtTar(0)/PtTar(2);
        double v = PtTar(1)/PtTar(2);

        if(u<F.mnMinX || u>F.mnMaxX)
            continue;
        if(v<F.mnMinY || v>F.mnMaxY)
            continue;

        vector<size_t> vIndices2;
        int LastKFOctave = 0;
        float radius = th*1.2f;
        vIndices2 = F.GetFeaturesInArea(u,v, radius, LastKFOctave-1, LastKFOctave+1);

        if(vIndices2.empty())
            continue;

        cv::Mat dMP = F1->mDescr.row(IdxObserv);
        int bestDist = INT_MAX;
        int bestIdx2 = -1;
        for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++){
            const size_t i2 = *vit;

            const cv::Mat &d = F.mDescr.row(i2);
            const int dist = DescriptorDistance(dMP,d);

            if(dist<bestDist)
            {
                bestDist=dist;
                bestIdx2=i2;
            }
        }

        if(bestDist<=TH_HIGH){
            if(vMatch2D3D[bestIdx2]<0 && vMatch12[IdxObserv]<0)
            {
                nMatches++;
                vMatch3D2D[i0] = bestIdx2;                                                  // key value[Bo2DMatch3D] size = initial frame, store 2nd index->initial index
                vMatch2D3D[bestIdx2] = i0;
                vMatch12[IdxObserv] = bestIdx2;
            }
        }

    }

    return nMatches;

}

int tracking::SearchFrom3DAdd(const vector<mapPts*> Pts3d, const frame &F, vector<int> &vMatch3D2D, vector<int> &vMatch2D3D, const int &th, keyframe *F1)
{
    vector<int> vMatch12;
    vMatch12 = vector<int>(F1->iN, -1);

    int nMatches = 0;
    Mat44 Tcw = F.mTcw;

    for(size_t i0 = 0; i0<Pts3d.size(); i0++){

        if(vMatch3D2D[i0]>=0)
            continue;
        if(Pts3d[i0]->FLAG_BAD)
            continue;

        mapPts* pt = Pts3d[i0];
        int IdxObserv;
        if(!pt->GetKFObv(F1, IdxObserv))
            continue;

        keyframe* RefKF = pt->RefKF;
        Mat44 Trw = RefKF->mTcw;
        Mat44 Tcr = Tcw*Trw.inverse();
        Mat33 Rcr = Tcr.block<3,3>(0,0);
        Mat31 tcr = Tcr.block<3,1>(0,3);

        Mat31 raydir = pt->GetRaydir();
        double rho = pt->GetInverD();
        double invrho = 1.0/rho;

        Mat31 PtTar = F.mK * (invrho * Rcr * raydir + tcr);

        double u = PtTar(0)/PtTar(2);
        double v = PtTar(1)/PtTar(2);

        if(u<F.mnMinX || u>F.mnMaxX)
            continue;
        if(v<F.mnMinY || v>F.mnMaxY)
            continue;

        vector<size_t> vIndices2;
        int LastKFOctave = 0;
        float radius = th*1.2f;
        vIndices2 = F.GetFeaturesInArea(u,v, radius, LastKFOctave-1, LastKFOctave+1);

        if(vIndices2.empty())
            continue;

        cv::Mat dMP = F1->mDescr.row(IdxObserv);
        int bestDist = INT_MAX;
        int bestIdx2 = -1;
        for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++){
            const size_t i2 = *vit;

            const cv::Mat &d = F.mDescr.row(i2);
            const int dist = DescriptorDistance(dMP,d);

            if(dist<bestDist)
            {
                bestDist=dist;
                bestIdx2=i2;
            }
        }

        if(bestDist<=TH_HIGH){
            if(vMatch2D3D[bestIdx2]<0 && vMatch12[IdxObserv]<0)
            {
                nMatches++;
                vMatch3D2D[i0] = bestIdx2;
                vMatch2D3D[bestIdx2] = i0;
                vMatch12[IdxObserv] = bestIdx2;
            }
        }

    }

    return nMatches;


}

int tracking::SearchFrom3DLocalTrack(const std::map<mapPts*, keyframe*> Pts3d, frame &F, const int &th)
{
    // -------- settings --------
    int thMapPt = 2;
    // -------- settings --------

    int nMatch = 0;
    std::map<mapPts*, keyframe*>::const_iterator IterPts;
    for(IterPts=Pts3d.begin(); IterPts!=Pts3d.end(); IterPts++)
    {
        mapPts* mPt = IterPts->first;
        if(mPt->FLAG_BAD)
            continue;

        if(!mPt->Flag_LocalTrack)
            continue;

        vector<size_t> vIndices2;
        int LastKFOctave = -1;
        float radius = th*1.2f;
        vIndices2 = F.GetFeaturesInArea(mPt->LocalTrackProj(0,0), mPt->LocalTrackProj(1,0), radius, LastKFOctave, LastKFOctave);

        if(vIndices2.empty())
            continue;

        int Idx_localKFIdx;
        bool INmPt = mPt->GetKFObv(IterPts->second, Idx_localKFIdx);
        assert(INmPt);

        cv::Mat dMP = IterPts->second->mDescr.row(Idx_localKFIdx);
        int bestDist = INT_MAX;
        int bestIdx = -1;
        int bestDist2= INT_MAX;
        int bestLevel2 = -1;
        for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(F.vMatches2D3D[idx]>=0)
                if(mpMap->GetPtFromId(F.vMatches2D3D[idx])->GetObservNum()>thMapPt)
                    continue;

            const cv::Mat &dF = F.mDescr.row(idx);

            int dist = DescriptorDistance(dMP,dF);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist = dist;
                bestIdx = idx;
            }else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            if(bestDist>0.9*(double)bestDist2)
                continue;

            F.AddSceneObserv(mPt, mPt->mnId, bestIdx);

            nMatch++;
        }

    }

    return nMatch;
}


int tracking::SearchForTriangular(const keyframe* F1, const frame &F2, const int &Win, vector<int> &vMatchIdx12)
{
   int nMatches = 0;
   vMatchIdx12 = vector<int>(F1->iN, -1);
   vector<int> vMatchDist(F2.iN, INT_MAX);
   vector<int> vMatchIdx21(F2.iN, -1);

   for(size_t i1 = 0; i1<F1->iN; i1++){
       // cond1. find no match scene points in F1
       if(F1->vMatches2D3D[i1]>=0)
           continue;

       if(F1->vTextObjInfo[i1]>=0)
           continue;

        cv::KeyPoint Fea = F1->vKeys[i1];
        int PyrLevel1 = Fea.octave;
        float ScaleORB = 1.2f;
        float radius = Win*ScaleORB;
        vector<size_t> vF2Candidate = F2.GetFeaturesInArea(Fea.pt.x, Fea.pt.y, radius, PyrLevel1, PyrLevel1);
        if(vF2Candidate.empty())
            continue;

        cv::Mat d1 = F1->mDescr.row(i1);
        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vF2Candidate.begin(); vit!=vF2Candidate.end(); vit++){
            size_t i2 = *vit;

            // cond2. find no match points in F2
            if(F2.vMatches2D3D[i2]>=0)
                continue;

            cv::Mat d2 = F2.mDescr.row(i2);
            int dist = DescriptorDistance(d1, d2);

            if(vMatchDist[i2]<=dist)
                continue;

            if(dist<bestDist){
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            }else if(dist<bestDist2){
                bestDist2 = dist;
            }
        }   // F2 candidate

        if(bestDist<=TH_LOW){
            if(vMatchIdx21[bestIdx2]>=0){
                vMatchIdx12[vMatchIdx21[bestIdx2]] = -1;
                nMatches--;
            }
            vMatchIdx12[i1] = bestIdx2;
            vMatchIdx21[bestIdx2] = i1;
            vMatchDist[bestIdx2] = bestDist;
            nMatches++;
        }   // Dist judge
   }        // F1 feature

    return nMatches;

}

int tracking::GetForTriangular(const keyframe* F1, const frame &F2, const vector<int> &vMatch12Raw, vector<Mat31> &vNewpts, vector<match> &vMatchNewpts)
{
    Mat44 Trw = F1->mTcw;
    Mat44 Tcw = F2.mTcw;
    cv::Mat T1 = Tool.EiM442cvM(Trw);
    cv::Mat T2 = Tool.EiM442cvM(Tcw);

    // initial 3D
    vector<Point2f> ptCam1, ptCam2, pt1, pt2;
    vector<match> MatchRaw;
    cv::Mat NewPts4d;
    for(size_t i3DIni=0; i3DIni<vMatch12Raw.size(); i3DIni++){
        if(vMatch12Raw[i3DIni]<0)
            continue;

        ptCam1.push_back(Tool.PixToCam(F2.mK, F1->vKeys[i3DIni].pt));
        ptCam2.push_back(Tool.PixToCam(F2.mK, F2.vKeys[vMatch12Raw[i3DIni]].pt));
        pt1.push_back(F1->vKeys[i3DIni].pt);
        pt2.push_back(F2.vKeys[vMatch12Raw[i3DIni]].pt);
        MatchRaw.push_back(make_pair(i3DIni, vMatch12Raw[i3DIni]));
    }
    cv::triangulatePoints(T1.rowRange(0,3).colRange(0,4), T2.rowRange(0,3).colRange(0,4), ptCam1, ptCam2, NewPts4d);

    vector<Mat31> NewPts3dRaw;
    for(size_t i3DProc=0; i3DProc<NewPts4d.cols; i3DProc++)
    {
        cv::Mat p4d = NewPts4d.col(i3DProc);
        p4d /= p4d.at<float>(3,0);
        Mat31 p3d((double)p4d.at<float>(0,0), (double)p4d.at<float>(1,0), (double)p4d.at<float>(2,0));
        NewPts3dRaw.push_back(p3d);
    }

    // check new 3d points
    int thReproj = 9;
    vector<bool> b3dRaw = vector<bool>(NewPts3dRaw.size(), false);
    int iGood = CheckTriangular(F2.mK, NewPts3dRaw, Trw, Tcw, pt1, pt2, thReproj, b3dRaw);

    for(size_t iCheck=0; iCheck<NewPts3dRaw.size(); iCheck++){
        if(!b3dRaw[iCheck])
            continue;

        vNewpts.push_back(NewPts3dRaw[iCheck]);
        vMatchNewpts.push_back(MatchRaw[iCheck]);
    }

    return iGood;
}

int tracking::CheckTriangular(const Mat33 &K, const vector<Mat31> &Pts3dRaw, const Mat44 &Trw, const Mat44 &Tcw, const vector<Point2f> &pts1, const vector<Point2f> &pts2,  const int &thReproj, vector<bool> &b3dRaw)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);

    int nGood = 0;
    for(size_t i=0; i<Pts3dRaw.size(); i++){
        Mat31 p3d = Pts3dRaw[i];

        // cond1: finite
        if(!isfinite(p3d(0,0)) || !isfinite(p3d(1,0)) || !isfinite(p3d(2,0)))
            continue;

        Mat31 p3dRef = Trw.block<3,3>(0,0) * p3d + Trw.block<3,1>(0,3);
        Mat31 p3dCur = Tcw.block<3,3>(0,0) * p3d + Tcw.block<3,1>(0,3);

        // cond3: reprojection 1&2
        double u1 = fx * p3dRef(0,0)/p3dRef(2,0) + cx;
        double v1 = fy * p3dRef(1,0)/p3dRef(2,0) + cy;
        if(u1<0 || v1<0)
            continue;

        float squareError1 = (u1-pts1[i].x)*(u1-pts1[i].x)+(v1-pts1[i].y)*(v1-pts1[i].y);
        if(squareError1>thReproj)
            continue;

        double u2 = fx * p3dCur(0,0)/p3dCur(2,0) + cx;
        double v2 = fy * p3dCur(1,0)/p3dCur(2,0) + cy;
        float squareError2 = (u2-pts2[i].x)*(u2-pts2[i].x)+(v2-pts2[i].y)*(v2-pts2[i].y);
        if(squareError2>thReproj)
            continue;

        b3dRaw[i] = true;
        nGood++;
    }

    return nGood;
}

int tracking::CheckMatch(const vector<mapPts *> &Pts3d, const frame &F, const vector<keyframe*> &neighKFs, vector<int> &vMatch3D2D, const float &ReproError)
{
    bool CHANGE = true;

    vector<cv::Point3d> vP3d;
    vector<cv::Point2f> vP2d;
    int numMatch = 0;
    for(size_t imatch=0; imatch<vMatch3D2D.size(); imatch++){
        if(vMatch3D2D[imatch]<0)
            continue;

        int idx1 = imatch;
        int idx2 = vMatch3D2D[imatch];

        // vP3d is world coordinate
        mapPts* pt = Pts3d[idx1];
        Mat44 Twr = pt->RefKF->GetTwc();

        Mat31 ray = pt->GetRaydir();
        double rho = pt->GetInverD();
        Mat31 pw = Twr.block<3,3>(0,0) * ray/rho + Twr.block<3,1>(0,3);

        Point3d p3d = Point3d(pw(0,0), pw(1,0), pw(2,0));
        vP3d.push_back(p3d);

        // get vP2d observation
        cv::Point2f p2d = cv::Point2f((float)F.vKeys[idx2].pt.x, (float)F.vKeys[idx2].pt.y);
        vP2d.push_back(p2d);

        numMatch++;
    }

    // Pnp test
    // get frame initial pose
    bool UseInit = false;
    cv::Mat rvec, tvec;
    if(UseInit){
        Eigen::AngleAxisd AAcw(F.mRcw);
        double ang = AAcw.angle();
        Vec3 axis = AAcw.axis();
        rvec = (Mat_<double>(3, 1) <<
                        axis(0)*ang,
                        axis(1)*ang,
                        axis(2)*ang);
        tvec = (Mat_<double>(3, 1) <<
                        F.mtcw(0,0),
                        F.mtcw(1,0),
                        F.mtcw(2,0));
    }

    int itCount = 100;      // Ransac iterations number.
    double Con = 0.98;      // ransac successful confidence.
    vector<bool> PnPres = vector<bool>(vP3d.size(), false);
    cv::Mat DistCoef = cv::Mat::zeros(5,1,CV_32FC1);
    cv::Mat inliers;
    cv::Mat mKcv = Tool.EiM332cvMf(F.mK);
    bool res_pnp = solvePnPRansac(vP3d, vP2d, mKcv, DistCoef, rvec, tvec, UseInit, itCount, ReproError, Con, inliers, SOLVEPNP_EPNP);
    for(size_t ipnpres=0; ipnpres<inliers.rows; ipnpres++){
        Mat idx = inliers.row(ipnpres);
        PnPres[idx.at<int>(0)] = true;
    }

    // update match res
    if(CHANGE){
        int idx=-1;
        for(size_t iupdate=0; iupdate<vMatch3D2D.size(); iupdate++){
            if(vMatch3D2D[iupdate]<0)
                continue;

            idx++;
            if(PnPres[idx])
                continue;

            // delete did not pass PnP test
            vMatch3D2D[iupdate] = -1;
            numMatch--;
        }
    }

    return numMatch;
}

void tracking::LandmarkerObvUpdate()
{

    TextObvUpdate();
    SceneObvUpdate();

}

void tracking::TextObvUpdate()
{
    vector<TextObservation*> vObjs = cfCurrentFrame.vObvText;
    vector<bool> vFlag = cfCurrentFrame.vObvGoodTexts;
    assert((int)vObjs.size()==(int)vFlag.size());

    for(size_t i0=0; i0<vObjs.size(); i0++)
        vObjs[i0]->obj->AddObvNum(vFlag[i0]);

}

void tracking::SceneObvUpdate()
{
    vector<SceneObservation*> vPts = cfCurrentFrame.vObvPts;
    vector<bool> vFlag = cfCurrentFrame.vObvGoodPts;
    assert((int)vPts.size()==(int)vFlag.size());

    for(size_t i0=0; i0<vPts.size(); i0++)
        vPts[i0]->pt->AddObvNum(vFlag[i0]);
}


vector<mapText*> tracking::GetTextObvFromNeighKFs(const vector<keyframe*> neighKFs)
{

    // 1. get raw observed all map text objects (duplicate)
    vector<mapText*> FObvText;
    for(size_t i0 = 0; i0<neighKFs.size(); i0++){
        vector<TextObservation*> textobjs = neighKFs[i0]->vObvText;
        for(size_t i1 = 0; i1<textobjs.size(); i1++){
            FObvText.push_back(textobjs[i1]->obj);
        }
    }

    // 2. sort the vector and delete the duplication
    sort(FObvText.begin(), FObvText.end());
    vector<mapText*>::iterator end = unique(FObvText.begin(), FObvText.end());
    FObvText.erase(end, FObvText.end());

    return FObvText;
}

void tracking::InitialTextObjs()
{
    int thresh_textnum = 8;

    // initial related params
    for(size_t j0 = 0; j0<cfLastKeyframe->iNTextObj; j0++){
        cfLastKeyframe->mNcr.push_back(Vec3(NAN, NAN, NAN));
        cfLastKeyframe->vNGOOD.push_back(false);
    }

    for(size_t i0=0; i0<cfLastKeyframe->vTextDeteCorMap.size(); i0++){
        if(cfLastKeyframe->vTextDeteCorMap[i0]>=0)
            continue;

        vector<cv::Point2f> Hostfeat = cfLastKeyframe->vKeysNewTextTrack[i0];
        vector<cv::Point2f> Targfeat = cfCurrentFrame.vKeysTextTrack[i0];

        if(Hostfeat.size()<thresh_textnum){
            continue;
        }

        Mat31 theta;
        bool TEXTOK = GetTextTheta(Hostfeat, Targfeat, cfLastKeyframe->mTcw, cfCurrentFrame.mTcw, i0, theta);

        if(!TEXTOK)
            continue;

        cfLastKeyframe->mNcr[i0] = theta;
        cfLastKeyframe->vNGOOD[i0] = true;
    }
}

// use triangulation get rho
bool tracking::GetTextTheta(const vector<cv::Point2f> &Hostfeat, const vector<cv::Point2f> &Targfeat, const Mat44 &Trw, const Mat44 &Tcw, const int &idx,
                             Mat31 &theta)
{
    Mat44 Tcr = Tcw * Trw.inverse();

    // 1. param initial
    cv::Mat T1 = cv::Mat::eye(4,4,CV_64F);
    cv::Mat T2 = cv::Mat::eye(4,4,CV_64F);
    T2 = (Mat_<double>(4, 4) <<
          Tcr(0,0), Tcr(0,1), Tcr(0,2), Tcr(0,3),
          Tcr(1,0), Tcr(1,1), Tcr(1,2), Tcr(1,3),
          Tcr(2,0), Tcr(2,1), Tcr(2,2), Tcr(2,3),
          0, 0, 0, 1);

    vector<Point2f> ptCam1, ptCam2;
    cv::Mat NewPts4d;
    for(size_t i3DIni=0; i3DIni<Hostfeat.size(); i3DIni++){
        ptCam1.push_back(Tool.PixToCam(cfCurrentFrame.mK, Hostfeat[i3DIni]));
        ptCam2.push_back(Tool.PixToCam(cfCurrentFrame.mK, Targfeat[i3DIni]));
    }

    // 2. triangulation
    cv::triangulatePoints(T1.rowRange(0,3).colRange(0,4), T2.rowRange(0,3).colRange(0,4), ptCam1, ptCam2, NewPts4d);

    // 3. get (xyz) & (rho)
    vector<Mat31> NewPts3dRaw;
    vector<double> vRho;
    for(size_t i3DProc=0; i3DProc<NewPts4d.cols; i3DProc++)
    {
        cv::Mat p4d = NewPts4d.col(i3DProc);
        p4d /= p4d.at<float>(3,0);
        Mat31 p3d((double)p4d.at<float>(0,0), (double)p4d.at<float>(1,0), (double)p4d.at<float>(2,0));
        NewPts3dRaw.push_back(p3d);
        vRho.push_back( 1.0/p3d(2,0) );
    }

    // 5. use rho calculate theta
    vector<Vec3> vHostRayRho;
    Tool.GetRayRho(Hostfeat, vRho, cfCurrentFrame.mK, vHostRayRho);

    // 5.1. get RANSAC Idx
    int iMaxIterations = 200, iSelectnum = 3;
    vector<vector<size_t>> RansacIdx;
    bool NUMOK = Tool.GetRANSACIdx(iMaxIterations, iSelectnum, Hostfeat.size(), true, RansacIdx);
    if(!NUMOK)
        return false;

    // 5.2 calculate theta using rho
    double BestScore = -1.0;
    Vec3 BestTheta(-1.0, -1.0, -1.0);
    for(size_t i0=0; i0<RansacIdx.size(); i0++){

        vector<size_t> Idx = RansacIdx[i0];

        Mat31 thetaObj;
        double score = SolveTheta(Idx, Tcr, Targfeat, vHostRayRho, cfCurrentFrame.mK, thetaObj);

        if(BestScore<score){
            BestTheta = thetaObj;
            BestScore = score;
        }
    }

    if(BestScore!=-1){
        theta = BestTheta;
    }else{
        return false;
    }

    return true;
}

void tracking::InitialNewTextFeatForTrack(keyframe *KF)
{
    int numTextDete = KF->vTextDeteCorMap.size();
    KF->vKeysNewTextTrack.resize(numTextDete);

    for(size_t i0=0; i0<numTextDete; i0++){
        if(KF->vTextDeteCorMap[i0]>=0)
            continue;

        // FAST feature directly to track
        vector<cv::Point2f> KeysToTrack;
        KeyPoint::convert(KF->vKeysText[i0], KeysToTrack);
        KF->vKeysNewTextTrack[i0] = KeysToTrack;
    }
}

void tracking::TrackNewTextFeat()
{
    // Get params
    vector<vector<cv::Point2f>> Trackedfeat, Curfeat;
    cv::Mat TrackedImg, CurImg = cfCurrentFrame.FrameImg.clone();
    if(cfCurrentFrame.mnId > cfLastKeyframe->mnFrameId+1 ){
        Trackedfeat = cfLastFrame.vKeysTextTrack;
        TrackedImg = cfLastFrame.FrameImg.clone();
    }else{
        assert(cfCurrentFrame.mnId==(cfLastKeyframe->mnFrameId+1) );
        // last frame is keyframe
        Trackedfeat = cfLastKeyframe->vKeysNewTextTrack;
        TrackedImg = cfLastKeyframe->FrameImg.clone();
    }
    Curfeat.resize(Trackedfeat.size());

    // KLT track features
    for(size_t i0=0; i0<Trackedfeat.size(); i0++){
        if(Trackedfeat[i0].size()==0)
            continue;

        vector<uchar> status;       // 0: no tracking pts; 1: has tracking pts
        vector<float> errorLK;      // the corresponding feature error
        vector<cv::Point2f> Curfeat_obj;
        cv::calcOpticalFlowPyrLK(TrackedImg, CurImg, Trackedfeat[i0], Curfeat_obj, status, errorLK);
        assert(Trackedfeat[i0].size()==Curfeat_obj.size());
        Curfeat[i0] = Curfeat_obj;

    }

    // update
    cfCurrentFrame.vKeysTextTrack = Curfeat;

}

// A = K*R21*K^(-1); B = K*t
double tracking::CalTextTheta(const vector<size_t> &Idx, const vector<Vec2> &HostRay, const vector<cv::Point2f> &Hostfeat, const vector<cv::Point2f> &Targfeat, const Mat33 &A, const Mat31 &B, const Mat44 &Tcr, const Mat33 &K, Mat31 &theta)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);

    // Hostfeat && Targfeat must corresponding to each other one by one.
    vector<cv::Point2f> vp1, vp2;
    vector<Vec3> vRayRho;
    vp1.push_back(Hostfeat[Idx[0]]);
    vp1.push_back(Hostfeat[Idx[1]]);
    vp1.push_back(Hostfeat[Idx[2]]);
    vp2.push_back(Targfeat[Idx[0]]);
    vp2.push_back(Targfeat[Idx[1]]);
    vp2.push_back(Targfeat[Idx[2]]);

    // 1. calculate rho
    for(size_t ifeat = 0; ifeat<vp1.size(); ifeat++){
        Mat31 p1(vp1[ifeat].x, vp1[ifeat].y, 1.0);
        double numerator_u = (A.block<1,3>(0,0) - A.block<1,3>(2,0)*vp2[ifeat].x) * p1;
        double denominator_u = B(2,0) * vp2[ifeat].x - B(0,0);
        double numerator_v = (A.block<1,3>(1,0) - A.block<1,3>(2,0)*vp2[ifeat].y) * p1;
        double denominator_v = B(2,0) * vp2[ifeat].y - B(1,0);
        double rhou = numerator_u/denominator_u;
        double rhov = numerator_v/denominator_v;
        double rho = (rhou+rhov)/2.0;

        vRayRho.push_back(Vec3( (vp1[ifeat].x-cx)/fx, (vp1[ifeat].y-cy)/fy, rho));
    }

    // 2. solve theta
    double score = SolveTheta(Idx, vRayRho, Tcr, Targfeat, HostRay, K, theta);

    return score;
}
// SolveTheta this: no use
double tracking::SolveTheta(const vector<size_t> &Idx, const vector<Vec3> vRayRho, const Mat44 &Tcr, const vector<cv::Point2f> &Targfeat, const vector<Vec2> &HostRay, const Mat33 &K, Mat31 &theta)
{
    const float th = 5.991;

    Vec3 m1 = vRayRho[0];
    Vec3 m2 = vRayRho[1];
    Vec3 m3 = vRayRho[2];

    // 1. get solution
    // 0 0 D    theta1   rho1'
    // 0 1 c    theta2   rho2'
    // 1 A B    theta3   rho3'
    double A = m3[1]/m3[0];
    double B = 1.0/m3[0];
    double C = (1-m2[0]*B)/(m2[1]-m2[0]*A);
    double D = (A*C-B)*m1[0]-C*m1[1]+1;
    double rho3pie = m3[2]/m3[0];
    double rho2pie = (m2[2]-rho3pie*m2[0])/(m2[1]-m2[0]*A);
    double rho1pie = m1[2] - rho3pie*m1[0] - rho2pie*(m1[1]-A*m1[0]);
    theta(2,0) = rho1pie/D;
    theta(1,0) = rho2pie - C*theta(2,0);
    theta(0,0) = rho3pie - A*theta(1,0) - B*theta(2,0);

    double score = 0;
    int numOK = 0;
    for(size_t ipts=0; ipts<HostRay.size(); ipts++){
        Mat13 m(HostRay[ipts](0), HostRay[ipts](1), 1.0);
        double rhoPred = m*theta;
        Mat31 P3d = Tcr.block<3,3>(0,0) * m.transpose()*(1.0/rhoPred) + Tcr.block<3,1>(0,3);
        double u = P3d(0)/P3d(2)*K(0,0) + K(0,2);
        double v = P3d(1)/P3d(2)*K(1,1) + K(1,2);
        double erroru = Targfeat[ipts].x - u;
        double errorv = Targfeat[ipts].y - v;
        double chiSquare2 = erroru*erroru+errorv*errorv;

        if(chiSquare2>th)
            continue;
        else
            score += th - chiSquare2;

        numOK++;
    }

    theta = -theta;

    return score;
}

double tracking::SolveTheta(const vector<size_t> &Idx, const Mat44 &Tcr, const vector<cv::Point2f> &Targfeat, const vector<Vec3> &HostRay, const Mat33 &K, Mat31 &theta)
{
    const float th = 5.991;
    assert(Idx.size()==3);

    Vec3 m1 = HostRay[Idx[0]];
    Vec3 m2 = HostRay[Idx[1]];
    Vec3 m3 = HostRay[Idx[2]];

    // 1. get solution
    // 0 0 D    theta1   rho1'
    // 0 1 c    theta2   rho2'
    // 1 A B    theta3   rho3'
    double A = m3[1]/m3[0];
    double B = 1.0/m3[0];
    double C = (1-m2[0]*B)/(m2[1]-m2[0]*A);
    double D = (A*C-B)*m1[0]-C*m1[1]+1;
    double rho3pie = m3[2]/m3[0];
    double rho2pie = (m2[2]-rho3pie*m2[0])/(m2[1]-m2[0]*A);
    double rho1pie = m1[2] - rho3pie*m1[0] - rho2pie*(m1[1]-A*m1[0]);
    theta(2,0) = rho1pie/D;
    theta(1,0) = rho2pie - C*theta(2,0);
    theta(0,0) = rho3pie - A*theta(1,0) - B*theta(2,0);

    double score = 0;
    int numOK = 0;
    for(size_t ipts=0; ipts<HostRay.size(); ipts++){
        Mat13 m(HostRay[ipts](0), HostRay[ipts](1), 1.0);
        double rhoPred = m*theta;
        Mat31 P3d = Tcr.block<3,3>(0,0) * m.transpose()*(1.0/rhoPred) + Tcr.block<3,1>(0,3);
        double u = P3d(0)/P3d(2)*K(0,0) + K(0,2);
        double v = P3d(1)/P3d(2)*K(1,1) + K(1,2);
        double erroru = Targfeat[ipts].x - u;
        double errorv = Targfeat[ipts].y - v;
        double chiSquare2 = erroru*erroru+errorv*errorv;

        if(chiSquare2>th)
            continue;
        else
            score += th - chiSquare2;

        numOK++;
    }

    theta = -theta;

    return score;
}


void tracking::TextUpdate(vector<mapText*> &vTexts)
{
    int Pym = 0;
    int threshOut = 3;
    double threshcos = 0.5;
    double threshZNCC = -3.0;

    for(size_t itext=0; itext<vTexts.size(); itext++){
        // 1. find unmature text objects
        if(vTexts[itext]->STATE!=TEXTIMMATURE)
            continue;

        // 2. judge text object can be observed great in current F
        mapText* obj = vTexts[itext];
        bool Obj_OK = TextJudgeSingle(obj, threshcos, threshOut, threshZNCC, Pym);
        if(!Obj_OK)
            continue;

        // 3. text object optimization
        obj->NumObvs += 1;
        Mat31 thetaRaw = obj->RefKF->mNcr[obj->GetNidx()];

        bool res = coOptimizer->ThetaOptimMultiFs(cfCurrentFrame, obj);
        if(!res){
            vTexts[itext]->STATE=TEXTBAD;
            continue;
        }

        Mat31 thetaNew = obj->RefKF->mNcr[obj->GetNidx()];

        // 4. text object update
        double threshCos = 0.9;
        int threshNumObvs = 4;
        thetaNew = thetaNew/thetaNew.norm();
        thetaRaw = thetaRaw/thetaRaw.norm();
        double Cos = thetaNew.transpose()*thetaRaw;
        if(Cos>=threshCos && obj->NumObvs>=threshNumObvs ){
            // change is such small and can be insert into map
            obj->STATE = TEXTGOOD;
        }
    }

}



vector<mapText*> tracking::TextJudge(const vector<mapText*> &vTexts)
{
    vector<mapText*> vTextsOut;
    int Pym = 0;
    int threshOut = 6;
    double threshcos = 0.5;
    double threshZNCC = 0.1;

    for(size_t itext=0; itext<vTexts.size(); itext++){
        mapText* obj = vTexts[itext];
        if(obj->STATE==TEXTBAD){
            continue;
        }

        bool Obj_OK = TextJudgeSingle(obj, threshcos, threshOut, threshZNCC, Pym);

        if(Obj_OK)
            vTextsOut.push_back(vTexts[itext]);

    }

    return vTextsOut;
}


bool tracking::TextJudgeSingle(mapText *obj, const int &threshcos, const int &threshOut, const double &threshZNCC, const int &Pym)
{
    bool ZNCCCheck = true;
    if(threshZNCC<=-2.0)
        ZNCCCheck = false;
    Mat33 K = cfCurrentFrame.vK_scale[Pym];
    int rows = cfCurrentFrame.vFrameImg[Pym].rows, cols = cfCurrentFrame.vFrameImg[Pym].cols;

    Mat31 theta = obj->RefKF->mNcr[obj->GetNidx()];
    Mat44 Tcr = cfCurrentFrame.mTcw * obj->RefKF->mTcw.inverse();

    // 1. N in text object VS camera orientation != 90
    if(!Tool.CheckOrientation(theta, Tcr, cfCurrentFrame, threshcos)){
        return false;
    }

    bool POSITIVED = true, BOXIN = true;
    bool POSITIVED_PIX = false, BOXIN_PIX = false;
    vector<Vec2> refDeteBox = obj->vTextDeteRay;
    vector<Vec2> Cur_ProjBox;
    Cur_ProjBox.resize(refDeteBox.size());

    for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++){
        POSITIVED_PIX = Tool.GetProjText(refDeteBox[iBox], theta, Cur_ProjBox[iBox], Tcr, K);
        if(Cur_ProjBox[iBox](0,0)<=threshOut || Cur_ProjBox[iBox](0,0)>=cols-threshOut ||
                Cur_ProjBox[iBox](1,0)<=threshOut || Cur_ProjBox[iBox](1,0)>=rows-threshOut ){
            BOXIN_PIX = false;
        }else{
            BOXIN_PIX = true;
        }

        if(!POSITIVED_PIX){
            POSITIVED = false;
            break;
        }
        if(!BOXIN_PIX){
            BOXIN = false;
            break;
        }
    }

    // 2. depth positive
    if(!POSITIVED){
        return false;
    }

    // 3. all pixels within image
    if(!BOXIN){
        return false;
    }

    // 4. ZNCC check (all pixs in text in Img pyramid 0)
    if(ZNCCCheck){
        vector<TextFeature*> vAllpixs = obj->vRefPixs;
        bool ZNCCOK = Tool.CheckZNCC(vAllpixs, cfCurrentFrame.vFrameImg[Pym], obj->RefKF->vFrameImg[Pym], Tcr, theta, K, threshZNCC);
        if(!ZNCCOK){
            return false;
        }
    }

    return true;
}


bool tracking::TextJudgeSingle(mapText *obj, const int &threshcos, const int &threshOut, const double &threshZNCC, const int &Pym, vector<int> &IdxTextCorDete)
{
    bool ZNCCCheck = true;
    if(threshZNCC<=-2.0)
        ZNCCCheck = false;
    Mat33 K = cfCurrentFrame.vK_scale[Pym];
    int rows = cfCurrentFrame.vFrameImg[Pym].rows, cols = cfCurrentFrame.vFrameImg[Pym].cols;
    cv::Mat BackImg = cv::Mat::ones(cfCurrentFrame.FrameImg.rows, cfCurrentFrame.FrameImg.cols, CV_32F)*(-1.0);

    Mat31 theta = obj->RefKF->mNcr[obj->GetNidx()];
    Mat44 Tcr = cfCurrentFrame.mTcw * obj->RefKF->mTcw.inverse();

    // 1. N in text object VS camera orientation != 90
    if(!Tool.CheckOrientation(theta, Tcr, cfCurrentFrame, threshcos)){
        return false;
    }

    bool POSITIVED = true, BOXIN = true;
    bool POSITIVED_PIX = false, BOXIN_PIX = false;
    vector<Vec2> refDeteBox = obj->vTextDeteRay;
    vector<Vec2> Cur_ProjBox;
    Cur_ProjBox.resize(refDeteBox.size());

    for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++){
        POSITIVED_PIX = Tool.GetProjText(refDeteBox[iBox], theta, Cur_ProjBox[iBox], Tcr, K);
        if(Cur_ProjBox[iBox](0,0)<=threshOut || Cur_ProjBox[iBox](0,0)>=cols-threshOut ||
                Cur_ProjBox[iBox](1,0)<=threshOut || Cur_ProjBox[iBox](1,0)>=rows-threshOut ){
            BOXIN_PIX = false;
        }else{
            BOXIN_PIX = true;
        }

        if(!POSITIVED_PIX){
            POSITIVED = false;
            break;
        }
        if(!BOXIN_PIX){
            BOXIN = false;
            break;
        }
    }

    // 2. depth positive
    if(!POSITIVED){
        return false;
    }

    // 3. all pixels within image
    if(!BOXIN){
        return false;
    }

    // 4. ZNCC check (all pixs in text in Img pyramid 0)
    if(ZNCCCheck){
        vector<TextFeature*> vAllpixs = obj->vRefPixs;
        bool ZNCCOK = Tool.CheckZNCC(vAllpixs, cfCurrentFrame.vFrameImg[Pym], obj->RefKF->vFrameImg[Pym], Tcr, theta, K, threshZNCC);
        if(!ZNCCOK){
            return false;
        }
    }

    // 5. get current Frame detection idx
    BackImg = Tool.GetTextLabelMask(BackImg, Cur_ProjBox, (int)obj->mnId);
    for(size_t idete=0; idete<cfCurrentFrame.vTextDeteCenter.size(); idete++)
    {
        Vec2 deteCen = cfCurrentFrame.vTextDeteCenter[idete];
        int u = round( deteCen(0,0) );
        int v = round( deteCen(1,0) );
        float* img_ptr = (float*)BackImg.data + v * BackImg.cols + u;
        float label = img_ptr[0];
        if(label>=0)
            IdxTextCorDete.push_back(idete);
    }

    return true;

}


cv::Mat tracking::GetTextLabelImg(const keyframe* CurKF, const TextStatus &STATE)
{
    int Pym = 0;
    cv::Mat ImgShow = CurKF->vFrameImg[Pym].clone();
    cv::Mat BackImg = cv::Mat::ones(ImgShow.size(), CV_32F)*(-1.0);

    vector<TextObservation*> vObvTexts = CurKF->vObvText;
    for(size_t iobv=0; iobv<vObvTexts.size(); iobv++){
        if(vObvTexts[iobv]->obj->STATE!=STATE)
            continue;

        mapText* textobj = vObvTexts[iobv]->obj;
        vector<Vec2> BoxRef = textobj->vTextDeteRay;
        Mat44 Tcr = CurKF->mTcw * textobj->RefKF->mTwc;
        Mat31 thetaobj = textobj->RefKF->mNcr[textobj->GetNidx()];

        vector<Vec2> textBoxObj;
        for(size_t iBox = 0; iBox<BoxRef.size(); iBox++){
            Mat31 ray = Mat31(BoxRef[iBox](0), BoxRef[iBox](1), 1.0 );
            double invz = - ray.transpose() * thetaobj;
            Mat31 p = CurKF->mK * ( Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3) );
            double u = p(0,0)/p(2,0);
            double v = p(1,0)/p(2,0);
            textBoxObj.push_back(Vec2(u,v));
        }
        BackImg = Tool.GetTextLabelMask(BackImg, textBoxObj, iobv);
    }

    return BackImg;
}


void tracking::UpdateImTextTrack(const cv::Mat ImTextLabel, keyframe* CurKF)
{
    for(size_t i0 = 0; i0<CurKF->vTextDeteCorMap.size(); i0++){
        if(CurKF->vTextDeteCorMap[i0]>=0)
            continue;

        Vec2 Center = CurKF->vTextDeteCenter[i0];

        int u = round(Center(0));
        int v = round(Center(1));
        float* img_ptr = (float*)ImTextLabel.data + v*ImTextLabel.cols + u;
        float label = img_ptr[0];

        if(label<0)
            continue;
        int IdxTextLabel = (int)label;

        CurKF->vTextDeteCorMap[i0] = CurKF->vObvText[IdxTextLabel]->obj->mnId;

        if(CurKF->vObvText[IdxTextLabel]->idx.size()!=0){
            bool FLAG_HAVE = false;
            vector<int> vIdxRaw = CurKF->vObvText[IdxTextLabel]->idx;

            for(size_t iIdx=0; iIdx<vIdxRaw.size(); iIdx++){
                if(vIdxRaw[iIdx]==i0){
                    FLAG_HAVE = true;
                    break;
                }
            }
            if(!FLAG_HAVE){
                CurKF->vObvText[IdxTextLabel]->idx.push_back(i0);
            }

        }else{
            CurKF->vObvText[IdxTextLabel]->idx.push_back(i0);
        }


        bool CHANGE_OBVIDX = CurKF->vObvText[IdxTextLabel]->obj->UpdateKFObserv(CurKF, (int)i0);

        if(CHANGE_OBVIDX){
            if(CurKF->vObvText[IdxTextLabel]->obj->STATE!=TEXTBAD)
                UpdateSemantic_MapObjs_single(CurKF->vObvText[IdxTextLabel]->obj, CurKF);
        }

    }
}


void tracking::mpPtsCondUpdate(const vector<keyframe*> &vKFs)
{
    int numBad = 0;
    for(size_t ikf = 0; ikf<vKFs.size(); ikf++){
        vector<bool> vGoodObv = vKFs[ikf]->vObvGoodPts;
        vector<SceneObservation*> vObvPt = vKFs[ikf]->vObvPts;
        assert(vGoodObv.size()==vObvPt.size());
        for(size_t iobvPt = 0; iobvPt<vGoodObv.size(); iobvPt++){
            if(vGoodObv[iobvPt])
                continue;

            vObvPt[iobvPt]->pt->FLAG_BAD = true;
            numBad++;
        }
    }

}



vector<SceneObservation*> tracking::GetSceneObv(const vector<mapPts*> vAllMappts, const vector<int> &vMatch3D2D, frame &F, vector<Vec2> &SceneObv2d)
{
    vector<SceneObservation*> FrameObvPts;
    F.vMatches2D3D = vector<int>(F.iN, (int)-1);

    int numMatch=0;
    for(size_t i0 = 0; i0<vMatch3D2D.size(); i0++){
        // this 3D map point has no corresponding map points
        if(vMatch3D2D[i0]<0)
            continue;

        int idxMappts = i0;
        int idxFobv = vMatch3D2D[i0];

        // map points
        SceneObservation* ptobv = new SceneObservation{vAllMappts[idxMappts], idxFobv};
        FrameObvPts.push_back(ptobv);

        // 2D observation
        SceneObv2d.push_back(Vec2(F.vKeys[idxFobv].pt.x, F.vKeys[idxFobv].pt.y));

        // match info
        F.vMatches2D3D[idxFobv] = vAllMappts[idxMappts]->mnId;

        numMatch++;
    }

    F.vObvGoodPts = vector<bool>(FrameObvPts.size(), true);
    return FrameObvPts;
}


void tracking::GetCovisibleKFs_all(keyframe* KF, const int &CurKFNum)
{
    int Use = 0;    // 0: M1; 1: M2; 2: M3
    vector<CovKF> vKFConKFsAll, vKFConKFs_prev, vKFConKFs_after;

    Eigen::MatrixXd MAll = mpMap->GetCovMap(Use);
    Eigen::MatrixXd Mkf_col = MAll.col(KF->mnId);
    Eigen::MatrixXd Mkf_row = MAll.row(KF->mnId);

    for(size_t iConnect=0; iConnect<KF->mnId; iConnect++){
        vKFConKFs_prev.push_back( make_pair( mpMap->GetKFFromId(iConnect), Mkf_col(iConnect, 0) ) );
    }
    for(size_t iConnect=(KF->mnId+1); iConnect<CurKFNum; iConnect++){
        vKFConKFs_after.push_back( make_pair( mpMap->GetKFFromId(iConnect), Mkf_row(0, iConnect) ) );
    }
    sort(vKFConKFs_prev.begin(), vKFConKFs_prev.end(), cmpLarge_kf());
    sort(vKFConKFs_after.begin(), vKFConKFs_after.end(), cmpLarge_kf());

    vector<CovKF> vKFConKFsPrev = Tool.GetAllNonZero(vKFConKFs_prev);
    vector<CovKF> vKFConKFsAfter = Tool.GetAllNonZero(vKFConKFs_after);

    vKFConKFsAll.insert(vKFConKFsAll.end(), vKFConKFsPrev.begin(), vKFConKFsPrev.end());
    vKFConKFsAll.insert(vKFConKFsAll.end(), vKFConKFsAfter.begin(), vKFConKFsAfter.end());

    sort(vKFConKFsAll.begin(), vKFConKFsAll.end(), cmpLarge_kf());

    KF->SetCovisibleKFsAll(vKFConKFsAll);
    KF->SetCovisibleKFsPrev(vKFConKFsPrev);
    KF->SetCovisibleKFsAfter(vKFConKFsAfter);

}


void tracking::UpdateSemantic_Condtions(const bool &UPDATE_TEXTFLAG, const bool &UPDATE_TEXTMEAN)
{

    UpdateSemFlag_MapObjs(UPDATE_TEXTFLAG, UPDATE_TEXTMEAN);

}

void tracking::UpdateSemFlag_MapObjs(const bool &UPDATE_TEXTFLAG, const bool &UPDATE_TEXTMEAN)
{
    // setting-flag
    int thresh1 = 2;
    double thresh2 = 0.9;
    int thresh3 = 40;
    vector<mapText*> vObjs = mpMap->GetAllMapTexts();

    for(size_t iobj = 0; iobj<vObjs.size(); iobj++){
        mapText* obj = vObjs[iobj];

        // flag --------------
        if(UPDATE_TEXTFLAG){

            if(obj->STATE==TEXTBAD || obj->STATE==TEXTIMMATURE){
                continue;
            }

            if(cfLastKeyframe->mnId<=5 || obj->RefKF->mnId>=(cfLastKeyframe->mnId-5))
                continue;

            Update_MapObjsFlag_single(thresh1, thresh2, thresh3, obj);
        }
        // flag --------------

        if(UPDATE_TEXTMEAN){
            if(obj->STATE==TEXTBAD)
                continue;

            UpdateSemantic_MapObjs_single(obj);
        }
    }       // each obj

}

// single obj & all KFs
void tracking::UpdateSemantic_MapObjs_single(mapText* obj)
{
    std::map<keyframe*,vector<int>> vObvKFs = obj->vObvkeyframe;
    vector<string> vMean;
    vector<double> vScoreRaw, vScore;
    string Mean = "";
    double Score = 99999999999.9, Score_raw = -1.0;

    std::map<keyframe*,vector<int>>::const_iterator iobvKF;
    for(iobvKF=vObvKFs.begin(); iobvKF!=vObvKFs.end(); iobvKF++){
        keyframe* KF = iobvKF->first;
        vector<int> vIdx2Dete = iobvKF->second;
        for(size_t idete=0; idete<vIdx2Dete.size(); idete++){
            int idx_dete = vIdx2Dete[idete];
            TextInfo TextInfo = KF->vTextMean[idx_dete];
            vMean.push_back(TextInfo.mean);
            vScoreRaw.push_back(TextInfo.score);

            // S_mean, S_geo. S_semantic = S_geo+S_mean (use smaller is better)
            // a) S_mean = (1 - mean_confidence)*weight_mean
            double S_mean = (1.0-TextInfo.score)*200.0;
            // b) S_geo =  (1.0+Cos) * weight_view + d   (smaller is better)
            double S_geo = GetSgeo(KF, obj);
            // c) S_semantic
            double S_semantic = S_geo+S_mean;
            vScore.push_back(S_semantic);

            if(S_semantic<Score){
                Mean = TextInfo.mean;
                Score = S_semantic;
                Score_raw = S_mean;
            }
        }
    }   // kf
    obj->TextMean.mean = Mean;
    obj->TextMean.score_semantic = Score;
}


void tracking::UpdateSemantic_MapObjs_single(mapText* obj, keyframe* KF)
{
    vector<string> vMean;
    vector<double> vScoreRaw, vScore;
    string Mean = obj->TextMean.mean;
    double Score = obj->TextMean.score_semantic, Score_raw = obj->TextMean.score;

    vector<int> vIdx2Dete = obj->vObvkeyframe[KF];
    for(size_t idete=0; idete<vIdx2Dete.size(); idete++){
        int idx_dete = vIdx2Dete[idete];
        if(idx_dete<0 || idx_dete>=KF->vTextMean.size()){
            continue;
        }

        TextInfo TextInfo = KF->vTextMean[idx_dete];
        vMean.push_back(TextInfo.mean);
        vScoreRaw.push_back(TextInfo.score);

        // S_mean, S_geo. S_semantic = S_geo+S_mean (use smaller is better)
        // a) S_mean = (1 - mean_confidence)*weight_mean
        double S_mean = (1.0-TextInfo.score)*200.0;
        // b) S_geo =  (1.0+Cos) * weight_view + d   (smaller is better)
        double S_geo = GetSgeo(KF, obj);
        // c) S_semantic
        double S_semantic = S_geo+S_mean;
        vScore.push_back(S_semantic);

        if(S_semantic<Score){
            Mean = TextInfo.mean;
            Score = S_semantic;
            Score_raw = TextInfo.score;
        }
    }

    obj->TextMean.mean = Mean;
    obj->TextMean.score = Score_raw;
    obj->TextMean.score_semantic = Score;
}

void tracking::Update_MapObjsFlag_single(const int &thresh1, const double &thresh2, const int &thresh3, mapText* obj)
{

    bool c0 = obj->ObvNum_good>thresh1;
    bool c1 = obj->ObvNum_good*thresh2>obj->ObvNum_bad;
    bool c2 = obj->ObvNum_bad<thresh3;
    bool flag_obj = c0 && c1 && c2 ;

    if(!flag_obj){
        obj->STATE = TEXTBAD;
    }

}

void tracking::UpdateSemantic_MapObjs(const bool &UPDATE_TEXTMEAN, const bool &UPDATE_TEXTGEO)
{
    vector<mapText*> vObjs = mpMap->GetAllMapTexts();

    for(size_t iobj = 0; iobj<vObjs.size(); iobj++){

        if(vObjs[iobj]->STATE==TEXTBAD)
            continue;

        std::map<keyframe*,vector<int>> vObvKFs = vObjs[iobj]->vObvkeyframe;
        vector<string> vMean;
        vector<double> vScoreRaw, vScore;
        string Mean = "";
        double Score = 99999999999.9, Score_raw = -1.0;

        std::map<keyframe*,vector<int>>::const_iterator iobvKF;
        for(iobvKF=vObvKFs.begin(); iobvKF!=vObvKFs.end(); iobvKF++){
            keyframe* KF = iobvKF->first;
            vector<int> vIdx2Dete = iobvKF->second;
            for(size_t idete=0; idete<vIdx2Dete.size(); idete++){
                int idx_dete = vIdx2Dete[idete];
                TextInfo TextInfo = KF->vTextMean[idx_dete];
                vMean.push_back(TextInfo.mean);
                vScoreRaw.push_back(TextInfo.score);

                // S_mean, S_geo. S_semantic = S_geo+S_mean (use smaller is better)
                double S_mean = (1.0-TextInfo.score)*200.0;
                double S_geo = GetSgeo(KF, vObjs[iobj]);
                double S_semantic = S_geo+S_mean;
                vScore.push_back(S_semantic);

                if(S_semantic<Score){
                    Mean = TextInfo.mean;
                    Score = S_semantic;
                    Score_raw = S_mean;
                }
            }
        }
        vObjs[iobj]->TextMean.mean = Mean;
        vObjs[iobj]->TextMean.score_semantic = Score;

    }
}

double tracking::GetSgeo(const keyframe* KF, mapText* obj)
{
    double weight_view = 10.0;

    // oCam,zCam
    Mat31 oCam = KF->mtwc;
    Mat31 zCam = KF->mTwc.block<3,3>(0,0) * Mat31(0,0,1);

    // oObj
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

    return S_geo;
}


// semantic experiments ------------

void tracking::RecordFrame(const frame &F)
{
    // pose.txt
    ofstream file1("pose_F.txt",ios::app);
    file1 << "--" <<F.mnId << '\n';
    file1 << F.mTcw << '\n';
    file1.close();
}


void tracking::RecordKeyFrame()
{
    ofstream outfile("keyframe_all.txt",ios::app);

    vector<keyframe*> vKFs = mpMap->GetAllKeyFrame();
    for(int i0 = 0; i0< vKFs.size(); i0++)
    {
        keyframe* kf = vKFs[i0];
        Mat44 kTwc = kf->GetTwc();
        Eigen::Quaterniond qwc(kTwc.block<3,3>(0,0));
        qwc = qwc.normalized();

        outfile<<kf->mnId<<" "<<kf->mnFrameId<<" ";
        outfile<<kTwc(0,3)<<" "<<kTwc(1,3)<<" "<<kTwc(2,3)<<" ";
        outfile<<qwc.w()<<"  "<< qwc.x()<<"  "<< qwc.y()<<"  "<< qwc.z() << '\n';

    }
    outfile << "--"<< endl;
    outfile.close();

}

void tracking::RecordKeyFrame_latest()
{
    vector<keyframe*> vKFs = mpMap->GetAllKeyFrame();

    string name = "keyframe_latest.txt";
    ofstream outfile1;
    outfile1.open(name.c_str());
    outfile1 << fixed;

    for(int i0 = 0; i0< vKFs.size(); i0++)
    {
        keyframe* kf = vKFs[i0];
        Mat44 kTwc = kf->GetTwc();
        Eigen::Quaterniond qwc(kTwc.block<3,3>(0,0));
        qwc = qwc.normalized();

        outfile1<<setprecision(6) <<kf->dTimeStamp<<setprecision(7)<<" ";
        outfile1<<kTwc(0,3)<<" "<<kTwc(1,3)<<" "<<kTwc(2,3)<<" ";
        outfile1<<qwc.x()<<" "<< qwc.y()<<" "<< qwc.z()<<" "<< qwc.w() << '\n';

    }
    outfile1.close();
}

void tracking::RecordKeyFrameSys(string &name)
{
    bool Flag_UseTime = true;

    // 1. name
    if(!Flag_UseTime)
    {
        ofstream outfile(name,ios::app);

        vector<keyframe*> vKFs = mpMap->GetAllKeyFrame();
        for(int i0 = 0; i0< vKFs.size(); i0++)
        {
            keyframe* kf = vKFs[i0];
            Mat44 kTwc = kf->GetTwc();
            Eigen::Quaterniond qwc(kTwc.block<3,3>(0,0));
            qwc = qwc.normalized();

            outfile<<kf->mnId<<" "<<kf->mnFrameId<<" ";
            outfile<<kTwc(0,3)<<" "<<kTwc(1,3)<<" "<<kTwc(2,3)<<" ";
            outfile<<qwc.w()<<"  "<< qwc.x()<<"  "<< qwc.y()<<"  "<< qwc.z() << '\n';

        }

        outfile.close();
    }

    if(Flag_UseTime)
    {
        ofstream outfile;
        outfile.open(name.c_str());
        outfile << fixed;

        vector<keyframe*> vKFs = mpMap->GetAllKeyFrame();
        for(int i0 = 0; i0< vKFs.size(); i0++)
        {
            keyframe* kf = vKFs[i0];
            Mat44 kTwc = kf->GetTwc();
            Eigen::Quaterniond qwc(kTwc.block<3,3>(0,0));
            qwc = qwc.normalized();

            outfile<<setprecision(6) <<kf->dTimeStamp<<setprecision(7)<<" ";
            outfile<<kTwc(0,3)<<" "<<kTwc(1,3)<<" "<<kTwc(2,3)<<" ";
            outfile<<qwc.x()<<" "<< qwc.y()<<" "<< qwc.z()<<" "<< qwc.w() << '\n';
        }
        outfile.close();
    }

}


void tracking::RecordLandmarks()
{
    // Pw = Twr * [(m1, m2, 1)*(1.0/rho)];
    ofstream outfileS_w("Point_info.txt",ios::app);     // Pw
    // N = [(n1,n2,n3)/n.norm, 1/n.norm]; (Nr*Trw)*(Twr*Pr)=0 -> (Nr*Trw)*Pw=0 -> Nw = Nr*Trw
    ofstream outfileTN_w("Text_info.txt",ios::app);      // Nw

    vector<mapPts*> vAllMappts = mpMap->GetAllMapPoints();
    vector<mapText*> vMapTexts = mpMap->GetAllMapTexts();

    outfileS_w<<mpMap->GetAllKeyFrame().size()<<endl;
    outfileTN_w<<mpMap->GetAllKeyFrame().size()<<endl;

    // scene pts record
    for(size_t is=0; is<vAllMappts.size(); is++){
        Mat31 ray = vAllMappts[is]->GetRaydir();
        double rho = vAllMappts[is]->GetInverD();
        Mat44 Twr = vAllMappts[is]->RefKF->GetTwc();
        Mat31 Pw = Twr.block<3,3>(0,0) * ray/rho + Twr.block<3,1>(0,3);
        outfileS_w<<vAllMappts[is]->mnId<<" "<<vAllMappts[is]->FLAG_BAD;
        outfileS_w<<" "<<Pw(0,0)<<" "<<Pw(1,0)<<" "<<Pw(2,0)<<endl;
    }
    outfileS_w << "--"<< endl;
    outfileS_w.close();

    // text objs record
    for(size_t iT = 0; iT<vMapTexts.size(); iT++){
        Mat44 Trw = vMapTexts[iT]->RefKF->GetTcw();
        Mat44 Twr = vMapTexts[iT]->RefKF->GetTwc();
        Mat31 Theta_r = vMapTexts[iT]->RefKF->mNcr[vMapTexts[iT]->GetNidx()];
        vector<Vec2> refDeteBox = vMapTexts[iT]->vTextDeteRay;

        // N info STATE--(TEXTGOOD = 0, TEXTIMMATURE = 1, TEXTBAD = 2)
        Mat31 Theta_w = Tool.TransTheta(Theta_r, Trw);
        outfileTN_w<<vMapTexts[iT]->mnId<<" "<<vMapTexts[iT]->STATE;
        outfileTN_w<<" "<<Theta_w(0,0)<<" "<<Theta_w(1,0)<<" "<<Theta_w(2,0);

        // box info
        vector<Mat31> Boxw;
        Boxw.resize(refDeteBox.size());
        for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++){
            Tool.GetPtsText(refDeteBox[iBox], Theta_r, Boxw[iBox], Twr);
            outfileTN_w<<" "<<Boxw[iBox](0,0)<<" "<<Boxw[iBox](1,0)<<" "<<Boxw[iBox](2,0);
        }

        outfileTN_w<<endl;
    }
    outfileTN_w << "--"<< endl;
    outfileTN_w.close();

}

// only record good map text objects
void tracking::RecordTextObvs(const frame &F)
{
    ofstream outfile("Text_observation_F.txt",ios::app);
    outfile<<"--"<<F.mnId<<endl;
    Mat44 Tcw = F.mTcw;

    vector<TextObservation*> TextObvs = F.vObvText;
    vector<bool> TextObvFlag = F.vObvGoodTexts;
    assert(TextObvs.size()==TextObvFlag.size());
    for(size_t i0 = 0; i0<TextObvs.size(); i0++){
        if(!(TextObvs[i0]->obj->STATE==TEXTGOOD))
            continue;

        TextObservation* textobjobv = TextObvs[i0];
        outfile<<"----"<<textobjobv->obj->mnId<<", "<<textobjobv->cos<<", "<<TextObvFlag[i0]<<endl;
        Mat44 Trw = TextObvs[i0]->obj->RefKF->mTcw;
        Mat44 Tcr = Tcw*Trw.inverse();

        vector<Vec2> Dete4ptsRef = textobjobv->obj->vTextDeteRay;
        for(size_t i1 = 0; i1<Dete4ptsRef.size(); i1++){
            Mat31 ray = Mat31(Dete4ptsRef[i1](0), Dete4ptsRef[i1](1), 1.0);
            Mat31 theta = textobjobv->obj->RefKF->mNcr[textobjobv->obj->GetNidx()];
            Vec2 pred;
            bool IN = Tool.GetProjText(ray, theta, pred, Tcr, F.mK);
            if(IN)
               outfile << pred(0) <<","<< pred(1) << '\n';
            else
               outfile <<"err "<< pred(0) <<","<< pred(1) << '\n';
        }
    }

    outfile.close();
}


void tracking::RecordTextObvs_KFfull(const keyframe* KF)
{
    ofstream outfile1("Text_observation_KF.txt",ios::app);
    outfile1<<"--"<<KF->mnFrameId<<","<<KF->mnId<<endl;

    Mat44 Tcw = KF->mTcw;

    vector<TextObservation*> TextObvs = KF->vObvText;
    vector<bool> TextObvFlag = KF->vObvGoodTexts;
    assert(TextObvs.size()==TextObvFlag.size());
    for(size_t i0 = 0; i0<TextObvs.size(); i0++)
    {
        if(!(TextObvs[i0]->obj->STATE==TEXTGOOD))
            continue;

        TextObservation* textobjobv = TextObvs[i0];
        outfile1<<textobjobv->obj->mnId<<","<<textobjobv->cos<<","<<TextObvFlag[i0];
        Mat44 Trw = TextObvs[i0]->obj->RefKF->mTcw;
        Mat44 Tcr = Tcw*Trw.inverse();

        vector<Vec2> Dete4ptsRef = textobjobv->obj->vTextDeteRay;
        for(size_t i1 = 0; i1<Dete4ptsRef.size(); i1++){
            Mat31 ray = Mat31(Dete4ptsRef[i1](0), Dete4ptsRef[i1](1), 1.0);
            Mat31 theta = textobjobv->obj->RefKF->mNcr[textobjobv->obj->GetNidx()];
            Vec2 pred;
            bool IN = Tool.GetProjText(ray, theta, pred, Tcr, KF->mK);
            outfile1 <<","<< pred(0) <<","<< pred(1);
        }
        outfile1<<endl;
    }
    outfile1.close();

}


// some used tools
int tracking::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void tracking::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}

void tracking::ParamConvert(const cv::Mat &Rcw, const cv::Mat &tcw, Mat44 &eTcw)
{
    tool Tool;
    Mat33 eRcw = Tool.cvM2EiM33(Rcw);
    Mat31 etcw = Tool.cvM2EiM31(tcw);
    Mat14 fillin;
    fillin << 0.0,0.0,0.0,1.0;
    eTcw.block<3,3>(0,0) = eRcw;
    eTcw.block<3,1>(0,3) = etcw;
    eTcw.block<1,4>(3,0) = fillin;
}

bool tracking::ProjIsInFrame(const frame &F, mapPts* mPt)
{
    Mat31 PtPos = mPt->GetxyzPos();
    Mat31 Pc = F.mRcw * PtPos + F.mtcw;

    if(Pc(2,0)<0)
        return false;

    Mat31 PImg = F.mK * Pc;

    Vec2 Proj = Vec2(PImg(0,0)/PImg(2,0), PImg(1,0)/PImg(2,0));
    if(Proj(0)<F.mnMinX || Proj(0)>F.mnMaxX)
        return false;
    if(Proj(1)<F.mnMinY || Proj(1)>F.mnMaxY)
        return false;

    mPt->LocalTrackProj = Proj;
    return true;
}

vector<Vec2> tracking::ProjTextInKF(mapText* obj, keyframe* KF)
{
    vector<Vec2> Proj;
    Mat44 Tcr = KF->mTcw * obj->RefKF->mTcw.inverse();
    Mat31 theta = obj->RefKF->mNcr[obj->GetNidx()];
    vector<Vec2> refDeteBox = obj->vTextDeteRay;
    for(size_t ibox=0; ibox<refDeteBox.size(); ibox++){
        Mat31 ray = Mat31(refDeteBox[ibox](0), refDeteBox[ibox](1), 1.0);
        double invz = -ray.transpose() * theta;
        Mat31 p = KF->mK * ( Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3) );
        Proj.push_back(Vec2( p(0)/p(2), p(1)/p(2) ));
    }

    return Proj;
}

Vec2 tracking::ProjSceneInKF(mapPts* mpt, keyframe* KF)
{
    Mat44 Tcr = KF->mTcw * mpt->RefKF->mTcw.inverse();
    Mat31 ray = mpt->GetRaydir();
    double invz = mpt->GetInverD();
    Mat31 p = Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3);
    double u = mK(0,0) * p(0,0)/p(2,0) + mK(0,2);
    double v = mK(1,1) * p(1,0)/p(2,0) + mK(1,2);
    return Vec2(u, v);
}


}
