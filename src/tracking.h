/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef TRACKING_H
#define TRACKING_H

#include<string>
#include<thread>
#include <opencv2/core/core.hpp>
#include <loopClosing.h>
#include <optimizer.h>
#include <setting.h>
#include <frame.h>
#include <initializer.h>
#include <map.h>

using namespace std;

namespace TextSLAM {
class tracking
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    tracking(Mat33& K, map* Map, loopClosing* LClosing, setting* Setting);
    int GrabImageMonocular(const cv::Mat &im, const double &ImgTimeStamp, const Mat33 &CameraK, const vector<vector<Vec2>> &TextDece, const vector<TextInfo> &TextMean);

    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    void RecordKeyFrameSys(string &name);

public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3,
        TOLOOP=4
    };
    eTrackingState mState;
    eTrackingState mLastProcessedState;

    frame cfCurrentFrame;
    frame cfInitialFrame;
    cv::Mat mImGray;

protected:
    void Track();
    void Initialization(bool &FLAG_HASRECORD);
    void CreatInitialMap(vector<cv::Point3f> &IniP3D, initializer *Initializ);
    void InitialLandmarker(const vector<Point3f> &IniP3D, initializer* Initializ, keyframe *F1, keyframe *F2);
    bool TrackWithMotMod();
    bool TrackWithOutMod();
    void TrackLocalMap();
    void TrackNewKeyframe();
    bool CheckNewKeyFrame();
    // (split) text initialization
    void InitialLandmarkerInKF_Text1(keyframe* KF);     // Section1: only old TextObj
    void InitialLandmarkerInKF_Text2(keyframe* KF, vector<mapText *> &vNewText);     // Section2: only new TextObj
    // (split) scene initialization
    void InitialLandmarkerInKF_Scene(keyframe* KF, const vector<Mat31> &vnewpts, const vector<match> &vnewptsmatch);
    // Loop
    bool CHECKLOOP();

    // search & match
    int SearchForInitializ(const frame &F1, const frame &F2, const int &Win, vector<int> &vMatchIdx12);
    int SearchFrom3D(const vector<mapPts*> Pts3d, const frame &F, vector<int> &vMatch3D2D, vector<int> &vMatch2D3D, const int &th, keyframe *F1);
    int SearchFrom3DAdd(const vector<mapPts*> Pts3d, const frame &F, vector<int> &vMatch3D2D, vector<int> &vMatch2D3D, const int &th, keyframe *F1);
    int SearchFrom3DLocalTrack(const std::map<mapPts *, keyframe *> Pts3d, frame &F, const int &th);
    int SearchForTriangular(const keyframe* F1, const frame &F2, const int &Win, vector<int> &vMatchIdx12);
    int GetForTriangular(const keyframe* F1, const frame &F2, const vector<int> &vMatch12Raw, vector<Mat31> &vNewpts, vector<match> &vMatchNewpts);

    // important process
    int CheckTriangular(const Mat33 &K, const vector<Mat31> &Pts3dRaw, const Mat44 &Trw, const Mat44 &Tcw, const vector<Point2f> &pts1, const vector<Point2f> &pts2, const int &thReproj, vector<bool> &b3dRaw);
    int CheckMatch(const vector<mapPts*> &Pts3d, const frame &F, const vector<keyframe*> &neighKFs, vector<int> &vMatch3D2D, const float &ReproError);

    // local map
    void UpdateLocalMap();
    void SearchLocalLandmarkers();
    void UpdateLocalKFs();
    void UpdateLocalLandmarkers();
    void SearchLocalPts();
    void SearchLocalObjs();

    // semantic experiments
    void UpdateSemantic_Condtions(const bool &UPDATE_TEXTFLAG, const bool &UPDATE_TEXTMEAN);
    void UpdateSemantic_MapObjs(const bool &UPDATE_TEXTMEAN, const bool &UPDATE_TEXTGEO);
        double GetSgeo(const keyframe* KF, mapText *obj);
    void UpdateSemFlag_MapObjs(const bool &UPDATE_TEXTFLAG, const bool &UPDATE_TEXTMEAN);
        void UpdateSemantic_MapObjs_single(mapText *obj);
        void Update_MapObjsFlag_single(const int &thresh1, const double &thresh2, const int &thresh3, mapText* obj);

        // optimization same
     void UpdateSemantic_MapObjs_single(mapText* obj, keyframe* KF);

    // landmarker (for Frame)
    void LandmarkerObvUpdate();
        void TextObvUpdate();
        void SceneObvUpdate();

    // text
    vector<mapText *> GetTextObvFromNeighKFs(const vector<keyframe*> neighKFs);
    // text initial
    void InitialTextObjs();
    bool GetTextTheta(const vector<Point2f> &Hostfeat, const vector<Point2f> &Targfeat, const Mat44 &Trw, const Mat44 &Tcw, const int &idx, Mat31 &theta);
    void InitialNewTextFeatForTrack(keyframe *KF);
    void TrackNewTextFeat();
    double CalTextTheta(const vector<size_t> &Idx, const vector<Vec2> &HostRay, const vector<cv::Point2f> &Hostfeat, const vector<cv::Point2f> &Targfeat, const Mat33 &A, const Mat31 &B, const Mat44 &Tcr, const Mat33 &K, Mat31 &theta);
    double SolveTheta(const vector<size_t> &Idx, const vector<Vec3> vRayRho, const Mat44 &Tcr, const vector<Point2f> &Targfeat, const vector<Vec2> &HostRay, const Mat33 &K, Mat31 &theta);
    double SolveTheta(const vector<size_t> &Idx, const Mat44 &Tcr, const vector<cv::Point2f> &Targfeat, const vector<Vec3> &HostRay, const Mat33 &K, Mat31 &theta);
    // text update
    void TextUpdate(vector<mapText*> &vTexts);
    // text basic tools
    vector<mapText*> TextJudge(const vector<mapText *> &vTexts);
    bool TextJudgeSingle(mapText* obj, const int &threshcos, const int &threshOut, const double &threshZNCC, const int &Pym);
    bool TextJudgeSingle(mapText* obj, const int &threshcos, const int &threshOut, const double &threshZNCC, const int &Pym, vector<int> &IdxTextCorDete);
    // text label && track
    cv::Mat GetTextLabelImg(const keyframe* CurKF, const TextStatus &STATE);
    void UpdateImTextTrack(const cv::Mat ImTextLabel, keyframe *CurKF);

    // point
    void mpPtsCondUpdate(const vector<keyframe*> &vKFs);

    // get param
    vector<SceneObservation*> GetSceneObv(const vector<mapPts*> vAllMappts, const vector<int> &vMatch3D2D, frame &F, vector<Vec2> &SceneObv2d);

    // (for loop) covisibleMap
    void GetCovisibleKFs_all(keyframe* KF, const int &CurKFNum);

    // record
    void RecordFrame(const frame &F);
    void RecordTextObvs(const frame &F);
    void RecordTextObvs_KFfull(const keyframe* KF);
    void RecordKeyFrame();
    void RecordKeyFrame_latest();
    void RecordLandmarks();

    // some tools
    void ParamConvert(const cv::Mat &Rcw, const cv::Mat &tcw, Mat44 &eTcw);
    bool ProjIsInFrame(const frame &F, mapPts* mPt);

    vector<Vec2> ProjTextInKF(mapText* obj, keyframe* KF);
    Vec2 ProjSceneInKF(mapPts* mpt, keyframe* KF);

protected:

    // map
    map* mpMap;

    // loopClosing
    loopClosing* LoopClosing;

    // frame related
    frame cfLastFrame;
    frame cfLastIniFrame;
    keyframe* cfLastKeyframe;
    Mat44 mVelocity;
    initializer* ciInitializer;

    // local map
    keyframe* cfLocalKF;
    vector<keyframe*> cvkLocalKFs;
    std::map<mapPts*, keyframe*> cvlLocalPts;
    std::set<mapText*> cvlLocalObjs;

    // optimization
    optimizer* coOptimizer;

    // ORB feature
    ORBextractor* coORBextractor;
    ORBextractor* coORBextractorIni;

    // setting
    setting* Set;

    // match
    vector<int> vIniMatches;
    vector<int> vFMatches2D3D;
    int iMatches;

    // Param for IO
    // Image Size
        // when Txt input, use this to first set keyframe size
    int Width, Height;  // 640*480
    // intrinsic matrix
    Mat33 mK;

    // tool
    tool Tool;

    // param
    int iPylevels;
    double dPyScales;

    // Loop
    int iLastLoopKFid;
    bool bNeed_Loop;

    // Keyframe Check
    int mMaxFrames, mMaxFramesMax;

    // LOG or NOT
    bool FLAG_RECORD_TXT;

};

}

#endif // TRACKING_H
