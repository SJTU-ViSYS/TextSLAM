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
#include <optimizer.h>

namespace TextSLAM
{
double optimizer::cx, optimizer::cy, optimizer::fx, optimizer::fy, optimizer::invfx, optimizer::invfy;
optimizer::optimizer(Mat33 &mK, double &dScale, int &nLevels, bool &Flag_noText, bool &Flag_rapid):
    K(mK),bFlag_noText(Flag_noText), bFlag_rapid(Flag_rapid)
{
    fx = K(0,0);
    fy = K(1,1);
    cx = K(0,2);
    cy = K(1,2);
    invfx = 1.0/fx;
    invfy = 1.0/fy;

    vInvScalefactor.resize(nLevels);
    vK.resize(nLevels);
    vfx.resize(nLevels);
    vfy.resize(nLevels);
    vcx.resize(nLevels);
    vcy.resize(nLevels);
    vInvScalefactor[0] = 1.0;
    vK[0] = K;
    vfx[0] = fx;
    vfy[0] = fy;
    vcx[0] = cx;
    vcy[0] = cy;
    double invScale = 1.0/dScale;
    for(size_t i0 = 1; i0<nLevels; i0++){
        vInvScalefactor[i0] = vInvScalefactor[i0-1] * invScale;
        vK[i0] = vK[i0-1] * invScale;
        vK[i0](2,2) = 1.0;
        vfx[i0] = vfx[i0-1] * invScale;
        vfy[i0] = vfy[i0-1] * invScale;
        vcx[i0] = vcx[i0-1] * invScale;
        vcy[i0] = vcy[i0-1] * invScale;
    }

}

void optimizer::InitBA(keyframe* F1, keyframe* F2)
{
    vector<keyframe*> KF;
    KF.push_back(F1);
    KF.push_back(F2);

    // 1. get parameter to optimized and its obsvertion
    vector<SceneObservation*> ScenePts = F1->vObvPts;
    vector<vector<SceneFeature*>> SceneObv = F2->vSceneObv2d;     // pyramid -> all features
    vector<TextObservation*> TextObjs = F1->vObvText;

    // 2. param initial
    double **pose = new double*[2];
    double **theta = new double*[F1->iNTextObj];
    double **rho = new double*[ScenePts.size()];
    for(size_t i0 = 0; i0<2; i0++)
        pose[i0] = new double[7];
    for(size_t i1 = 0; i1<F1->iNTextObj; i1++)
        theta[i1] = new double[3];
    for(size_t i2 = 0; i2<ScenePts.size(); i2++){
        rho[i2] = new double[1];
    }

    for(size_t i3 = 0; i3<2; i3++){
        Mat33 Rcw = KF[i3]->mRcw;
        Mat31 tcw = KF[i3]->mtcw;
        Eigen::Quaterniond q(Rcw);
        q = q.normalized();
        pose[i3][0] = q.w();
        pose[i3][1] = q.x();
        pose[i3][2] = q.y();
        pose[i3][3] = q.z();
        pose[i3][4] = tcw(0,0);
        pose[i3][5] = tcw(1,0);
        pose[i3][6] = tcw(2,0);
    }
    // for initial 2 frames, TextObjs size < F1->iNTextObj == F1->mNcr == F1->vNGOOD
    // Based on all text objects , TextObjs size is delete small text object not has sufficient keypoints, F1->vNGOOD[i] = false.
    for(size_t i4 = 0; i4<TextObjs.size(); i4++){
        mapText* TextObj = TextObjs[i4]->obj;
        Mat31 vNcr = F1->mNcr[(size_t)TextObj->GetNidx()];
        theta[i4][0] = vNcr(0,0);
        theta[i4][1] = vNcr(1,0);
        theta[i4][2] = vNcr(2,0);
    }
    for(size_t i5 = 0; i5<ScenePts.size(); i5++){
        rho[i5][0] = ScenePts[i5]->pt->GetInverD();
    }

    // 4. begin BA problem
    ceres::Problem problem0, problem1, problem2, problem3;
    int PyBegin0 = 3, PyBegin1 = 2, PyBegin2 = 1, PyBegin3 = 0;
    cv::Mat ImgTextLabel;       // use the optimized parameters project all map text into the current frame. the label is idx of TextObjs.
    PyrIniBA(&problem0, KF, pose, theta, rho, ScenePts, SceneObv, TextObjs, PyBegin0, ImgTextLabel);      // PyBeginX for problemX
    PyrIniBA(&problem1, KF, pose, theta, rho, ScenePts, SceneObv, TextObjs, PyBegin1, ImgTextLabel);
    PyrIniBA(&problem2, KF, pose, theta, rho, ScenePts, SceneObv, TextObjs, PyBegin2, ImgTextLabel);
    PyrIniBA(&problem3, KF, pose, theta, rho, ScenePts, SceneObv, TextObjs, PyBegin3, ImgTextLabel);

    // 5. optimized parameters update
    // 5.1) F2 pose
    Mat44 Tcw = Tool.Pose2Mat44(pose[1]);
    F2->SetPose(Tcw);
    // 5.2) scene point rho
    for(size_t iUpScenept = 0; iUpScenept<ScenePts.size(); iUpScenept++){
        mapPts* mpt = ScenePts[iUpScenept]->pt;
        double rhoNew = rho[iUpScenept][0];
        mpt->SetRho(rhoNew);
    }
    // 5.3) text object theta
    for(size_t iUpTextObj = 0; iUpTextObj<TextObjs.size(); iUpTextObj++){
        mapText* mtext = TextObjs[iUpTextObj]->obj;
        Mat31 NcrNew = Mat31(theta[iUpTextObj][0], theta[iUpTextObj][1], theta[iUpTextObj][2]);
        mtext->RefKF->SetN(NcrNew, mtext->GetNidx());
    }
    // 5.4) text objects in current keyframe observation correspondings
    UpdateTrackedTextBA(F2->vObvText, ImgTextLabel, F2, true);

}

void optimizer::PoseOptim(frame &F)
{
    bool NOISE = false;

    // 1. get parameter to optimized and its obsvertion
    vector<SceneObservation*> ScenePts = F.vObvPts;
    vector<vector<SceneFeature*>> SceneObv = F.vSceneObv2d;

    vector<TextObservation*> TextObjs;
    vector<int> FLAGTextObjs;
    vector<bool> vTextsGood;
    vector<vector<bool>> vTextFeatsGood;
    int num_feat = 0;
    for(size_t i0 = 0; i0<F.vObvText.size(); i0++){
        if(!(F.vObvText[i0]->obj->STATE==TEXTGOOD))
            continue;

        FLAGTextObjs.push_back(i0);
        TextObjs.push_back(F.vObvText[i0]);
        vTextsGood.push_back(F.vObvGoodTexts[i0]);
        vTextFeatsGood.push_back(F.vObvGoodTextFeats[i0]);
        num_feat += F.vObvGoodTextFeats[i0].size();
    }

    // 2. param initial
    double pose[7];
    Mat33 Rcw = F.mRcw;
    Mat31 tcw = F.mtcw;
    Eigen::Quaterniond q(Rcw);
    q = q.normalized();
    pose[0] = q.w();
    pose[1] = q.x();
    pose[2] = q.y();
    pose[3] = q.z();
    pose[4] = tcw(0,0);
    pose[5] = tcw(1,0);
    pose[6] = tcw(2,0);

    // 3. begin BA problem
    int PyBegin0 = 3, PyBegin1 = 2, PyBegin2 = 1, PyBegin3 = 0;
    cv::Mat ImgTextLabel;       // use the optimized parameters project all map text into the current frame. the label is idx of TextObjs.

    // setting
    double chi2Mono[4]={12.25,12.25,12.25,12.25};
    double chi2Text[4]={0.5,0.5,0.5,0.95};
    const int its[4]={10,10,10,10};
    vector<bool> vPtsGood = F.vObvGoodPts;
    if(bFlag_rapid)
        PyrPoseOptim(F, pose, ScenePts, SceneObv, TextObjs, PyBegin0, chi2Mono[0], chi2Text[0], its[0], vPtsGood, vTextsGood, vTextFeatsGood, FLAGTextObjs, ImgTextLabel);
    PyrPoseOptim(F, pose, ScenePts, SceneObv, TextObjs, PyBegin1, chi2Mono[1], chi2Text[1], its[1], vPtsGood, vTextsGood, vTextFeatsGood, FLAGTextObjs, ImgTextLabel);
    PyrPoseOptim(F, pose, ScenePts, SceneObv, TextObjs, PyBegin2, chi2Mono[2], chi2Text[2], its[2], vPtsGood, vTextsGood, vTextFeatsGood, FLAGTextObjs, ImgTextLabel);
    PyrPoseOptim(F, pose, ScenePts, SceneObv, TextObjs, PyBegin3, chi2Mono[3], chi2Text[3], its[3], vPtsGood, vTextsGood, vTextFeatsGood, FLAGTextObjs, ImgTextLabel);

    // 4. optimized parameters update
    // 4.1) F pose
    Mat44 Tcw = Tool.Pose2Mat44(pose);
    F.SetPose(Tcw);
    // 4.2) text objects in currentframe observation correspondings
    UpdateTrackedTextPOSE(TextObjs, ImgTextLabel, F);

}

void optimizer::LocalBundleAdjustment(map* mpMap, vector<keyframe*> vKFs, const BAStatus &STATE)
{
    // 1. get parameter to optimized and its obsvertion (theta && rho)
    // basic params
    vector<mapPts*> vMapPts = mpMap->GetAllMapPoints();
    vector<mapText*> vMapTexts = mpMap->GetAllMapTexts(TEXTGOOD);

    vector<int> vmnId2MapPts, vmnId2MapTexts, vmnId2vKFs;
    vmnId2MapPts = vector<int>(mpMap->imapPts, (int)-1);
    vmnId2MapTexts = vector<int>(mpMap->imapText, (int)-1);
    vmnId2vKFs = vector<int>(mpMap->imapkfs, (int)-1);
    vector<bool> vMapPtOptim, vMapTextOptim;
    // observation
    vector<vector<bool>> vPtsGood;          // kf->observation
    vPtsGood.resize(vKFs.size());

    double **rho = new double*[vMapPts.size()];
    double **theta = new double*[vMapTexts.size()];
    double **pose = new double*[vKFs.size()];
    vMapPtOptim.resize(vMapPts.size());
    vMapTextOptim.resize(vMapTexts.size());
    for(size_t ipareRho = 0; ipareRho<vMapPts.size(); ipareRho++){
        rho[ipareRho] = new double[1];
    }
    for(size_t ipareTheta = 0; ipareTheta<vMapTexts.size(); ipareTheta++){
        theta[ipareTheta] = new double[3];
    }
    for(size_t iparePose = 0; iparePose<vKFs.size(); iparePose++)
        pose[iparePose] = new double[7];

    for(size_t ikfs=0; ikfs<vKFs.size(); ikfs++){
        int kfmnid = vKFs[ikfs]->mnId;
        vmnId2vKFs[kfmnid] = ikfs;
        vPtsGood[ikfs] = vKFs[ikfs]->vObvGoodPts;
    }

    for(size_t ipareS=0; ipareS<vMapPts.size(); ipareS++){
        // 1. get optimized params
        mapPts* pt = vMapPts[ipareS];

        // rho
        rho[ipareS][0] = pt->GetInverD();
        // vMapPtOptim
        if(vmnId2vKFs[pt->RefKF->mnId]<0)
            vMapPtOptim[ipareS] = false;
        else
            vMapPtOptim[ipareS] = true;
        vmnId2MapPts[pt->mnId] = ipareS;
    }
    for(size_t ipareT=0; ipareT<vMapTexts.size(); ipareT++){
        mapText* TextObj = vMapTexts[ipareT];
        Mat31 vNcr = TextObj->RefKF->mNcr[(size_t)TextObj->GetNidx()];

        // theta
        theta[ipareT][0] = vNcr(0,0);
        theta[ipareT][1] = vNcr(1,0);
        theta[ipareT][2] = vNcr(2,0);
        // vMapTextOptim
        if(vmnId2vKFs[TextObj->RefKF->mnId]<0)
            vMapTextOptim[ipareT] = false;
        else
            vMapTextOptim[ipareT] = true;
        //vmnId2MapTexts
        vmnId2MapTexts[TextObj->mnId] = ipareT;
    }

    vector<int> InitialIdx;         // poses to fix (initial 2 frames)
    for(size_t iparePose=0; iparePose<vKFs.size(); iparePose++){
        Mat33 Rcw = vKFs[iparePose]->mRcw;
        Mat31 tcw = vKFs[iparePose]->mtcw;
        Eigen::Quaterniond q(Rcw);
        q = q.normalized();
        pose[iparePose][0] = q.w();
        pose[iparePose][1] = q.x();
        pose[iparePose][2] = q.y();
        pose[iparePose][3] = q.z();
        pose[iparePose][4] = tcw(0,0);
        pose[iparePose][5] = tcw(1,0);
        pose[iparePose][6] = tcw(2,0);

        if(vKFs[iparePose]->mnId==0 || vKFs[iparePose]->mnId==1)
            InitialIdx.push_back(iparePose);
    }

    // 2. optimization
    int PyBegin1 = 2, PyBegin2 = 1, PyBegin3 = 0;
    cv::Mat ImgTextLabel;
    const double chi2Mono[4]={12.25,12.25,12.25,12.25};
    const double chi2Text[4]={0.5,0.5,0.5,0.95};
    const int its[4]={10,10,10,10};
    PyrBA(pose, theta, rho, vKFs, vMapPts, vMapTexts, vmnId2MapPts, vMapPtOptim, vmnId2MapTexts, vMapTextOptim, vmnId2vKFs, InitialIdx, PyBegin1, chi2Mono[1], chi2Text[1], its[1], ImgTextLabel, STATE);
    PyrBA(pose, theta, rho, vKFs, vMapPts, vMapTexts, vmnId2MapPts, vMapPtOptim, vmnId2MapTexts, vMapTextOptim, vmnId2vKFs, InitialIdx, PyBegin2, chi2Mono[2], chi2Text[2], its[2], ImgTextLabel, STATE);
    PyrBA(pose, theta, rho, vKFs, vMapPts, vMapTexts, vmnId2MapPts, vMapPtOptim, vmnId2MapTexts, vMapTextOptim, vmnId2vKFs, InitialIdx, PyBegin3, chi2Mono[3], chi2Text[3], its[3], ImgTextLabel, STATE);


    // 3. update
    // pose
    for(size_t iupPose=0; iupPose<vKFs.size(); iupPose++){

        Eigen::Quaterniond qcw;
        Mat31 tcw;
        qcw.w() = pose[iupPose][0];
        qcw.x() = pose[iupPose][1];
        qcw.y() = pose[iupPose][2];
        qcw.z() = pose[iupPose][3];
        tcw(0,0) = pose[iupPose][4];
        tcw(1,0) = pose[iupPose][5];
        tcw(2,0) = pose[iupPose][6];
        qcw = qcw.normalized();
        Mat33 Rcw(qcw);
        Mat44 Tcw;
        Tcw.setIdentity();
        Tcw.block<3,3>(0,0) = Rcw;
        Tcw.block<3,1>(0,3) = tcw;

        vKFs[iupPose]->SetPose(Tcw);
    }

    // rho
    for(size_t iupRho=0; iupRho<vMapPts.size(); iupRho++){
        double rhoNew = rho[iupRho][0];
        vMapPts[iupRho]->SetRho(rhoNew);
    }

    // theta
    for(size_t iupTheta=0; iupTheta<vMapTexts.size(); iupTheta++){
        Mat31 NcrNew = Mat31(theta[iupTheta][0], theta[iupTheta][1], theta[iupTheta][2]);
        mapText* TextObj = vMapTexts[iupTheta];
        TextObj->RefKF->SetN(NcrNew, TextObj->GetNidx());
    }

    vector<int> vIdxGOOD2Raw = GetNewIdxForTextState(vKFs[vKFs.size()-1]->vObvText, TEXTGOOD);
    UpdateTrackedTextBA(vKFs[vKFs.size()-1]->vObvText, vIdxGOOD2Raw, ImgTextLabel, vKFs[(int)vKFs.size()-1], false);

}


void optimizer::GlobalBA(map* mpMap)
{
    // -------- settings --------
    bool FlagPtRequire = false;
    // -------- settings --------

    vector<keyframe*> vKFs = mpMap->GetAllKeyFrame();
    vector<mapPts*> vMapPts = mpMap->GetAllMapPoints(FlagPtRequire);
    vector<mapText*> vMapTexts = mpMap->GetAllMapTexts(TEXTGOOD);

    // ********************* Step 1. Param prepare ********************* //
    vector<int> vmnId2MapPts, vmnId2MapTexts, vmnId2vKFs;
    vmnId2MapPts = vector<int>(mpMap->imapPts, (int)-1);
    vmnId2MapTexts = vector<int>(mpMap->imapText, (int)-1);
    vmnId2vKFs = vector<int>(mpMap->imapkfs, (int)-1);

    vector<vector<bool>> vPtsGood;          // kf->observation
    vPtsGood.resize(vKFs.size());

    double **rho = new double*[vMapPts.size()];
    double **theta = new double*[vMapTexts.size()];
    double **pose = new double*[vKFs.size()];
    for(size_t ipareRho = 0; ipareRho<vMapPts.size(); ipareRho++){
        rho[ipareRho] = new double[1];
    }
    for(size_t ipareTheta = 0; ipareTheta<vMapTexts.size(); ipareTheta++){
        theta[ipareTheta] = new double[3];
    }
    for(size_t iparePose = 0; iparePose<vKFs.size(); iparePose++)
        pose[iparePose] = new double[7];


    for(size_t ikfs=0; ikfs<vKFs.size(); ikfs++){
        int kfmnid = vKFs[ikfs]->mnId;
        vmnId2vKFs[kfmnid] = ikfs;
        vPtsGood[ikfs] = vKFs[ikfs]->vObvGoodPts;
    }

    for(size_t ipareS=0; ipareS<vMapPts.size(); ipareS++){
        // get optimized params
        mapPts* pt = vMapPts[ipareS];

        // rho
        rho[ipareS][0] = pt->GetInverD();
        // vmnId2MapPts
        vmnId2MapPts[pt->mnId] = ipareS;
    }
    for(size_t ipareT=0; ipareT<vMapTexts.size(); ipareT++){
        mapText* TextObj = vMapTexts[ipareT];
        Mat31 vNcr = TextObj->RefKF->mNcr[(size_t)TextObj->GetNidx()];

        // theta
        theta[ipareT][0] = vNcr(0,0);
        theta[ipareT][1] = vNcr(1,0);
        theta[ipareT][2] = vNcr(2,0);
        //vmnId2MapTexts
        vmnId2MapTexts[TextObj->mnId] = ipareT;
    }
    vector<int> InitialIdx;         // poses to fix (initial 2 frames)
    for(size_t iparePose=0; iparePose<vKFs.size(); iparePose++){
        Mat33 Rcw = vKFs[iparePose]->mRcw;
        Mat31 tcw = vKFs[iparePose]->mtcw;
        Eigen::Quaterniond q(Rcw);
        q = q.normalized();
        pose[iparePose][0] = q.w();
        pose[iparePose][1] = q.x();
        pose[iparePose][2] = q.y();
        pose[iparePose][3] = q.z();
        pose[iparePose][4] = tcw(0,0);
        pose[iparePose][5] = tcw(1,0);
        pose[iparePose][6] = tcw(2,0);

        if(vKFs[iparePose]->mnId==0 || vKFs[iparePose]->mnId==1)
            InitialIdx.push_back(iparePose);
    }

    // ********************* Step 2. Optimization ********************* //
    int PyBegin3 = 0;
    const double chi2Mono[4]={18,18,18,18};
    const int its[4]={10,10,10, 20};
    PyrGlobalBA(pose, theta, rho, vKFs, vMapPts, vMapTexts, vmnId2MapPts, vmnId2MapTexts, vmnId2vKFs, InitialIdx, vPtsGood, PyBegin3, chi2Mono[3], its[3]);

    // ********************* Step 3. update optimized res ********************* //
    // keyframe pose
    for(size_t iupPose=0; iupPose<vKFs.size(); iupPose++){

        Eigen::Quaterniond qcw;
        Mat31 tcw;
        qcw.w() = pose[iupPose][0];
        qcw.x() = pose[iupPose][1];
        qcw.y() = pose[iupPose][2];
        qcw.z() = pose[iupPose][3];
        tcw(0,0) = pose[iupPose][4];
        tcw(1,0) = pose[iupPose][5];
        tcw(2,0) = pose[iupPose][6];
        qcw = qcw.normalized();
        Mat33 Rcw(qcw);
        Mat44 Tcw;
        Tcw.setIdentity();
        Tcw.block<3,3>(0,0) = Rcw;
        Tcw.block<3,1>(0,3) = tcw;

        vKFs[iupPose]->SetPose(Tcw);

    }

    // rho
    for(size_t iupRho=0; iupRho<vMapPts.size(); iupRho++){
        double rhoNew = rho[iupRho][0];
        vMapPts[iupRho]->SetRho(rhoNew);
    }

    // theta
    for(size_t iupTheta=0; iupTheta<vMapTexts.size(); iupTheta++){
        Mat31 NcrNew = Mat31(theta[iupTheta][0], theta[iupTheta][1], theta[iupTheta][2]);
        mapText* TextObj = vMapTexts[iupTheta];
        TextObj->RefKF->SetN(NcrNew, TextObj->GetNidx());
    }

}


void optimizer::OptimizeLandmarker(map* mpMap)
{

    // -------- settings --------
    bool FlagPtRequire = false;
    // -------- settings --------

    vector<keyframe*> vKFs = mpMap->GetAllKeyFrame();
    vector<mapPts*> vMapPts = mpMap->GetAllMapPoints(FlagPtRequire);
    vector<mapText*> vMapTexts = mpMap->GetAllMapTexts(TEXTGOOD);

    // ********************* Step 1. Param prepare ********************* //
    vector<int> vmnId2MapPts, vmnId2MapTexts, vmnId2vKFs;
    vmnId2MapPts = vector<int>(mpMap->imapPts, (int)-1);
    vmnId2MapTexts = vector<int>(mpMap->imapText, (int)-1);
    vmnId2vKFs = vector<int>(mpMap->imapkfs, (int)-1);

    vector<vector<bool>> vPtsGood;          // kf->observation
    vPtsGood.resize(vKFs.size());

    double **rho = new double*[vMapPts.size()];
    double **theta = new double*[vMapTexts.size()];
    double **pose = new double*[vKFs.size()];
    for(size_t ipareRho = 0; ipareRho<vMapPts.size(); ipareRho++){
        rho[ipareRho] = new double[1];
    }
    for(size_t ipareTheta = 0; ipareTheta<vMapTexts.size(); ipareTheta++){
        theta[ipareTheta] = new double[3];
    }
    for(size_t iparePose = 0; iparePose<vKFs.size(); iparePose++)
        pose[iparePose] = new double[7];

    for(size_t ikfs=0; ikfs<vKFs.size(); ikfs++){
        int kfmnid = vKFs[ikfs]->mnId;
        vmnId2vKFs[kfmnid] = ikfs;
        vPtsGood[ikfs] = vKFs[ikfs]->vObvGoodPts;
    }

    for(size_t ipareS=0; ipareS<vMapPts.size(); ipareS++){
        // get optimized params
        mapPts* pt = vMapPts[ipareS];

        // rho
        rho[ipareS][0] = pt->GetInverD();
        // vmnId2MapPts
        vmnId2MapPts[pt->mnId] = ipareS;
    }

    for(size_t ipareT=0; ipareT<vMapTexts.size(); ipareT++){
        mapText* TextObj = vMapTexts[ipareT];
        Mat31 vNcr = TextObj->RefKF->mNcr[(size_t)TextObj->GetNidx()];

        // theta
        theta[ipareT][0] = vNcr(0,0);
        theta[ipareT][1] = vNcr(1,0);
        theta[ipareT][2] = vNcr(2,0);
        //vmnId2MapTexts
        vmnId2MapTexts[TextObj->mnId] = ipareT;
    }

    for(size_t iparePose=0; iparePose<vKFs.size(); iparePose++){
        Mat33 Rcw = vKFs[iparePose]->mRcw;
        Mat31 tcw = vKFs[iparePose]->mtcw;
        Eigen::Quaterniond q(Rcw);
        q = q.normalized();
        pose[iparePose][0] = q.w();
        pose[iparePose][1] = q.x();
        pose[iparePose][2] = q.y();
        pose[iparePose][3] = q.z();
        pose[iparePose][4] = tcw(0,0);
        pose[iparePose][5] = tcw(1,0);
        pose[iparePose][6] = tcw(2,0);

    }

    // ********************* Step 2. Coarse-to-fine Optimization ********************* //
    int PyBegin0 = 3, PyBegin1 = 2, PyBegin2 = 1, PyBegin3 = 0;
    const double chi2Mono[4]={18,18,18,18};
    const double chi2Text[4]={1.5,1.5,1.5,1.5};
    const int its[4]={50, 50, 50, 50};
    vector<cv::Mat> vImgTextLabel;
    PyrLandmarkers(pose, theta, rho, vKFs, vMapPts, vMapTexts, vmnId2MapPts, vmnId2MapTexts, vmnId2vKFs, vPtsGood, PyBegin0, chi2Mono[0], chi2Text[0], its[0], vImgTextLabel);
    PyrLandmarkers(pose, theta, rho, vKFs, vMapPts, vMapTexts, vmnId2MapPts, vmnId2MapTexts, vmnId2vKFs, vPtsGood, PyBegin1, chi2Mono[1], chi2Text[1], its[1], vImgTextLabel);
    PyrLandmarkers(pose, theta, rho, vKFs, vMapPts, vMapTexts, vmnId2MapPts, vmnId2MapTexts, vmnId2vKFs, vPtsGood, PyBegin2, chi2Mono[2], chi2Text[2], its[2], vImgTextLabel);
    PyrLandmarkers(pose, theta, rho, vKFs, vMapPts, vMapTexts, vmnId2MapPts, vmnId2MapTexts, vmnId2vKFs, vPtsGood, PyBegin3, chi2Mono[3], chi2Text[3], its[3], vImgTextLabel);
    // ********************* Step 3. update optimized res ********************* //

    // rho
    for(size_t iupRho=0; iupRho<vMapPts.size(); iupRho++){
        double rhoNew = rho[iupRho][0];
        vMapPts[iupRho]->SetRho(rhoNew);
    }

    // theta
    for(size_t iupTheta=0; iupTheta<vMapTexts.size(); iupTheta++){
        Mat31 NcrNew = Mat31(theta[iupTheta][0], theta[iupTheta][1], theta[iupTheta][2]);
        mapText* TextObj = vMapTexts[iupTheta];
        TextObj->RefKF->SetN(NcrNew, TextObj->GetNidx());
    }

    // text box
    vector<int> vIdxGOOD2Raw1 = GetNewIdxForTextState(vKFs[vKFs.size()-2]->vObvText, TEXTGOOD);
    UpdateTrackedTextBA(vKFs[vKFs.size()-2]->vObvText, vIdxGOOD2Raw1, vImgTextLabel[0], vKFs[(int)vKFs.size()-2], false);
    vector<int> vIdxGOOD2Raw2 = GetNewIdxForTextState(vKFs[vKFs.size()-1]->vObvText, TEXTGOOD);
    UpdateTrackedTextBA(vKFs[vKFs.size()-1]->vObvText, vIdxGOOD2Raw2, vImgTextLabel[1], vKFs[(int)vKFs.size()-1], false);

}


bool optimizer::ThetaOptimMultiFs(const frame &F, mapText* &obj)
{
    Mat31 thetaRaw = obj->RefKF->mNcr[obj->GetNidx()];
    double *theta = new double[3];
    theta[0] = thetaRaw(0,0);
    theta[1] = thetaRaw(1,0);
    theta[2] = thetaRaw(2,0);
    Mat33 thetaVariance;
    // for pose and vImage for frames [keyframes+Curframe]
    vector<Mat44,Eigen::aligned_allocator<Mat44>> vTcr;             // all frames
    vector<vector<cv::Mat>> vImg;   // pyramid -> all frames
    vector<keyframe*> vKFs;
    vImg.resize(obj->RefKF->iScaleLevels);
    Mat44 Trw = obj->RefKF->mTcw;   // text host frame
    std::map<keyframe*,vector<int>> vObvKFs = obj->vObvkeyframe;
    std::map<keyframe*,vector<int>>::iterator itkfs;
    for(itkfs=vObvKFs.begin(); itkfs!=vObvKFs.end(); itkfs++){

        if(itkfs->first->mnId == obj->RefKF->mnId){
            continue;
        }

        Mat44 Tcwkf = itkfs->first->mTcw;
        Mat44 Tcrkf = Tcwkf*Trw.inverse();
        vTcr.push_back(Tcrkf);
        vKFs.push_back(itkfs->first);
        for(size_t ipy=0; ipy<vImg.size(); ipy++)
            vImg[ipy].push_back(itkfs->first->vFrameImg[ipy]);

    }
    Mat44 Tcw = F.mTcw;
    Mat44 Tcr = Tcw*Trw.inverse();
    vTcr.push_back(Tcr);
    for(size_t ipy=0; ipy<vImg.size(); ipy++)
        vImg[ipy].push_back(F.vFrameImg[ipy]);

    int PyBegin1 = 2, PyBegin2 = 1, PyBegin3 = 0;
    bool res;
    res = PyrThetaOptim(vImg[PyBegin1], vTcr, vKFs, obj, PyBegin1, theta, thetaVariance);
    if(!res){
        cout<<"PyrThetaOptim failed, return false."<<endl;
        return false;
    }
    res = PyrThetaOptim(vImg[PyBegin2], vTcr, vKFs, obj, PyBegin2, theta, thetaVariance);
    if(!res){
        cout<<"PyrThetaOptim failed, return false."<<endl;
        return false;
    }
    res = PyrThetaOptim(vImg[PyBegin3], vTcr, vKFs, obj, PyBegin3, theta, thetaVariance);
    if(!res){
        cout<<"PyrThetaOptim failed, return false."<<endl;
        return false;
    }

    // update
    Mat31 thetaNew(theta[0], theta[1], theta[2]);
    obj->RefKF->SetN(thetaNew, obj->GetNidx());
    obj->Covariance = thetaVariance;

}

int optimizer::OptimizeSim3(vector<FeatureConvert> &vFeat1, vector<FeatureConvert> &vFeat2, vector<bool> &vbInliers, Sim3_loop &Sim12, const float th2)
{
    // -------- settings --------
    double threshOutlier = 4.0;
    // -------- settings --------

    Mat33 K1 = K;
    Mat33 K2 = K;

    // parameter initial
    double Sim3Pose[8];
    Eigen::Quaterniond q = Sim12.r;
    q = q.normalized();
    Sim3Pose[0] = q.w();
    Sim3Pose[1] = q.x();
    Sim3Pose[2] = q.y();
    Sim3Pose[3] = q.z();
    Sim3Pose[4] = Sim12.t(0,0);
    Sim3Pose[5] = Sim12.t(1,0);
    Sim3Pose[6] = Sim12.t(2,0);
    Sim3Pose[7] = Sim12.s;

    std::chrono::steady_clock::time_point t_begin_T = std::chrono::steady_clock::now();

    ceres::Problem problem;
    problem.AddParameterBlock(Sim3Pose, 4, new ceres::QuaternionParameterization());
    problem.AddParameterBlock(Sim3Pose+4, 3);
    problem.AddParameterBlock(Sim3Pose+7, 1);

    int num_Match = 0;
    assert((int)vFeat1.size()==(int)vFeat2.size());
    for(size_t ip2 = 0; ip2<vFeat2.size(); ip2++){
        if(!vbInliers[ip2])
            continue;

        LossFunction* loss_function = new HuberLoss(sqrt(10));
        CostFunction *costFunction1;
        costFunction1 = auto_sim::Create(vFeat2[ip2].posObv, Vec2(vFeat1[ip2].obv2d.pt.x, vFeat1[ip2].obv2d.pt.y), K1);
        problem.AddResidualBlock(costFunction1, loss_function, Sim3Pose, Sim3Pose+4, Sim3Pose+7);

        int ip1 = ip2;
        CostFunction *costFunction2;
        costFunction2 = auto_siminv::Create(vFeat1[ip1].posObv, Vec2(vFeat2[ip1].obv2d.pt.x, vFeat2[ip1].obv2d.pt.y), K2);
        problem.AddResidualBlock(costFunction2, loss_function, Sim3Pose, Sim3Pose+4, Sim3Pose+7);
        num_Match++;
    }

    Solver::Options options;
    options.minimizer_type = TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout=false;
    options.max_num_iterations = 20;
    options.num_threads = 1;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::chrono::steady_clock::time_point t_end_T = std::chrono::steady_clock::now();
    double tT= std::chrono::duration_cast<std::chrono::duration<double> >(t_end_T - t_begin_T).count();

    // inlier check
    Eigen::Quaterniond q12;
    Mat31 t12;
    q12.w() = Sim3Pose[0];
    q12.x() = Sim3Pose[1];
    q12.y() = Sim3Pose[2];
    q12.z() = Sim3Pose[3];
    t12(0,0) = Sim3Pose[4];
    t12(1,0) = Sim3Pose[5];
    t12(2,0) = Sim3Pose[6];
    double s12 = Sim3Pose[7];
    q12 = q12.normalized();
    Mat33 R12(q12);
    Mat44 T12, T21;
    T12.setIdentity();
    T12.block<3,3>(0,0) = s12*R12;
    T12.block<3,1>(0,3) = t12;
    T21 = T12.inverse();

    Sim3_loop gScmRes(q12,t12,s12);
    Sim12 = gScmRes;

    int numInlier = 0;
    for(size_t ipt=0; ipt<vFeat2.size(); ipt++){
        if(!vbInliers[ipt])
            continue;

        Mat31 P2 = vFeat2[ipt].posObv;
        Mat31 P1pred = K1*( T12.block<3,3>(0,0)*P2+T12.block<3,1>(0,3) );
        Vec2 uvPred1 = Vec2( P1pred(0)/P1pred(2), P1pred(1)/P1pred(2) );
        Vec2 uvErr1 = uvPred1- Vec2(vFeat1[ipt].obv2d.pt.x, vFeat1[ipt].obv2d.pt.y);

        Mat31 P1 = vFeat1[ipt].posObv;
        Mat31 P2pred = K2*( T21.block<3,3>(0,0) * P1 + T21.block<3,1>(0,3) );
        Vec2 uvPred2 = Vec2( P2pred(0)/P2pred(2), P2pred(1)/P2pred(2) );
        Vec2 uvErr2 = uvPred2-Vec2(vFeat2[ipt].obv2d.pt.x, vFeat2[ipt].obv2d.pt.y);

        if( std::abs(uvErr1(0,0))>=threshOutlier || std::abs(uvErr1(1,0))>=threshOutlier ||
                std::abs(uvErr2(0,0))>=threshOutlier || std::abs(uvErr2(1,0))>=threshOutlier ){
            vbInliers[ipt] = false;
            continue;
        }

        numInlier++;
    }

    return numInlier;

}

void optimizer::OptimizeLoop(std::map<keyframe*, set<keyframe*>> &LoopConnections, std::map<keyframe*, set<keyframe*> > &NormConnections,
                             keyframe* KF, keyframe* LoopKF, std::map<keyframe*, Sim3_loop, std::less<keyframe*>,  Eigen::aligned_allocator<std::pair<keyframe*, Sim3_loop>> > &vConnectKFs, Sim3_loop &mScw, map* mpMap)
{

    // -------- settings --------
    int thEssent = 100;
    bool UseEssential = false;
    bool UseSTrans = true;
    // -------- settings --------

    vector<keyframe*> vKFs = mpMap->GetAllKeyFrame();

    double **pose = new double*[vKFs.size()];
    for(size_t iparePose = 0; iparePose<vKFs.size(); iparePose++)
        pose[iparePose] = new double[8];

    std::map<keyframe*, Sim3_loop, std::less<keyframe*>,  Eigen::aligned_allocator<std::pair<keyframe*, Sim3_loop>> > vScwIni;
    for(size_t ifs=0; ifs<vKFs.size(); ifs++){
        int KFId = vKFs[ifs]->mnId;

        Mat33 Rcw = vKFs[ifs]->mTcw.block<3,3>(0,0);
        Mat31 tcw = vKFs[ifs]->mTcw.block<3,1>(0,3);
        Eigen::Quaterniond q(Rcw);
        q = q.normalized();
        double s = 1.0;

        if(UseSTrans){
            if(vConnectKFs.count(vKFs[ifs])){
                Sim3_loop Siw = vConnectKFs[vKFs[ifs]];
                q = Siw.r;
                q = q.normalized();
                tcw = Siw.t;
                s = Siw.s;
            }
        }

        vScwIni[vKFs[ifs]] = Sim3_loop(q, tcw, s);

        pose[KFId][0] = q.w();
        pose[KFId][1] = q.x();
        pose[KFId][2] = q.y();
        pose[KFId][3] = q.z();
        pose[KFId][4] = tcw(0,0);
        pose[KFId][5] = tcw(1,0);
        pose[KFId][6] = tcw(2,0);
        pose[KFId][7] = s;
    }

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    ceres::Problem problem;
    for(size_t iprob=0; iprob<vKFs.size(); iprob++){
        int KFId = vKFs[iprob]->mnId;
        problem.AddParameterBlock(pose[KFId], 4, new ceres::QuaternionParameterization());
        problem.AddParameterBlock(pose[KFId]+4, 3);
        problem.AddParameterBlock(pose[KFId]+7, 1);
    }

    // normal edge
    std::map<keyframe*, set<keyframe*>>::iterator iterNorm;
    for(iterNorm=NormConnections.begin(); iterNorm!=NormConnections.end(); iterNorm++)
    {
        keyframe* KFi = iterNorm->first;
        int nIDi = KFi->mnId;
        Mat33 Riw = KFi->mTcw.block<3,3>(0,0);
        Mat31 tiw = KFi->mTcw.block<3,1>(0,3);
        Sim3_loop Siw(Riw, tiw, 1.0);

        set<keyframe*> vConnections = iterNorm->second;

        for(set<keyframe*>::const_iterator sit=vConnections.begin(), send=vConnections.end(); sit!=send; sit++)
        {
            keyframe* KFj = (*sit);
            if(UseEssential){
                int CovisibleWeight = KFi->GetCovisibleWeight(KFj);
                if(CovisibleWeight<thEssent)
                    continue;
            }

            int nIDj = KFj->mnId;
            Mat33 Rjw = KFj->mTcw.block<3,3>(0,0);
            Mat31 tjw = KFj->mTcw.block<3,1>(0,3);
            Sim3_loop Sjw(Rjw, tjw, 1.0);

            Sim3_loop Sji = Sjw * Siw.inverse();

            CostFunction *costFunction;
            costFunction = numer_loop_ver2::Create(Sji.r, Sji.t, Sji.s);
            problem.AddResidualBlock(costFunction, nullptr, pose[nIDi], pose[nIDi]+4, pose[nIDi]+7, pose[nIDj], pose[nIDj]+4, pose[nIDj]+7);

        }
    }

    // loop edge
    std::map<keyframe*, set<keyframe*>>::iterator iterLoop;
    for(iterLoop=LoopConnections.begin(); iterLoop!=LoopConnections.end(); iterLoop++)
    {
        keyframe* KFj = iterLoop->first;
        const long unsigned int nIDj = KFj->mnId;
        set<keyframe*> spConnections = iterLoop->second;
        Sim3_loop Sjw = vScwIni[KFj];

        if(nIDj==KF->mnId){
            Sjw = mScw;
        }

        for(set<keyframe*>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            keyframe* KFi = (*sit);
            int nIDi = KFi->mnId;

            if(UseEssential){
                int CovisibleWeight = KFi->GetCovisibleWeight(KFj);
                if(CovisibleWeight<thEssent)
                    continue;
            }

            // Scm
            Sim3_loop Sji = Sjw * vScwIni[KFi].inverse();

            CostFunction *costFunction;
            costFunction = numer_loop_ver2::Create(Sji.r, Sji.t, Sji.s);
            problem.AddResidualBlock(costFunction, nullptr, pose[nIDi], pose[nIDi]+4, pose[nIDi]+7, pose[nIDj], pose[nIDj]+4, pose[nIDj]+7);

        }
    }

    vector<int> vNeedFix;
    vNeedFix.push_back(0);
    vNeedFix.push_back(1);
    vNeedFix.push_back(LoopKF->mnId);
    for(size_t ifix=0; ifix<vNeedFix.size(); ifix++){
        int fixId = vNeedFix[ifix];
        problem.SetParameterBlockConstant( pose[fixId] );
        problem.SetParameterBlockConstant( pose[fixId]+4 );
        problem.SetParameterBlockConstant( pose[fixId]+7 );
    }

    Solver::Options options;
    options.minimizer_type = TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout=false;
    options.max_num_iterations = 20;
    options.num_threads = 1;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if(summary.num_successful_steps<0){
        cerr <<"Loop pose Optimize failed !";
        exit(-1);
    }

    // pose update
    std::map<keyframe*, Sim3_loop> vScwRes;
    for(size_t iup=0; iup<vKFs.size(); iup++){
        int KFId = vKFs[iup]->mnId;

        Eigen::Quaterniond q;
        Mat31 t;
        Mat44 T;
        T.setIdentity();
        q.w() = pose[KFId][0];
        q.x() = pose[KFId][1];
        q.y() = pose[KFId][2];
        q.z() = pose[KFId][3];
        t(0,0) = pose[KFId][4];
        t(1,0) = pose[KFId][5];
        t(2,0) = pose[KFId][6];
        double s = pose[KFId][7];
        q = q.normalized();
        Mat33 R(q);
        T.block<3,3>(0,0) = R;
        T.block<3,1>(0,3) = t/s;
        Sim3_loop Siw_corr = Sim3_loop(q, t, s);

        vKFs[iup]->SetPose(T);
        vScwRes[vKFs[iup]] = Siw_corr;
    }

    // update landmarker
    vector<mapPts*> vPts = mpMap->GetAllMapPoints();
    vector<mapText*> vObjs = mpMap->GetAllMapTexts();
    for(size_t ipt=0; ipt<vPts.size(); ipt++)
    {
        double rho = vPts[ipt]->GetInverD();
        Sim3_loop SrwRes = vScwRes[vPts[ipt]->RefKF];
        rho *= SrwRes.s;

        Mat31 ray = vPts[ipt]->GetRaydir();
        double invz = vPts[ipt]->GetInverD();
        Mat31 pr = ray/invz;
        int KFrefId = vPts[ipt]->RefKF->mnId;

        Eigen::Quaterniond q;
        Mat31 t;
        q.w() = pose[KFrefId][0];
        q.x() = pose[KFrefId][1];
        q.y() = pose[KFrefId][2];
        q.z() = pose[KFrefId][3];
        t(0,0) = pose[KFrefId][4];
        t(1,0) = pose[KFrefId][5];
        t(2,0) = pose[KFrefId][6];
        double s = pose[KFrefId][7];
        q = q.normalized();
        Sim3_loop SiwCorr = Sim3_loop(q, t, s);
        Sim3_loop SwiCorr = SiwCorr.inverse();
        Mat31 pwCorr = SwiCorr.map(pr);

        Mat44 TrwCorr = vPts[ipt]->RefKF->mTcw;
        Mat31 prCorr = TrwCorr.block<3,3>(0,0) * pwCorr + TrwCorr.block<3,1>(0,3);
        double rhoCorr = 1.0/prCorr(2);

        vPts[ipt]->SetRho(rho);
    }

    for(size_t iobj=0; iobj<vObjs.size(); iobj++){
        Mat31 theta = vObjs[iobj]->RefKF->mNcr[vObjs[iobj]->GetNidx()];
        Sim3_loop SrwRes = vScwRes[vObjs[iobj]->RefKF];
        theta *= SrwRes.s;
        vObjs[iobj]->RefKF->SetN( theta, vObjs[iobj]->GetNidx() );

    }

}


// for initial BA, each pyramid optimization function
void optimizer::PyrIniBA(ceres::Problem *problem, vector<keyframe*> KF, double **pose, double **theta, double **rho, vector<SceneObservation *> ScenePts, vector<vector<SceneFeature*>> SceneObv, vector<TextObservation*> TextObjs, const int &PyBegin, cv::Mat &TextLabelImg)
{
    // ---------------------------------------- introduction -------------------------------------------------------
    // ScenePts: the observed map point. SceneObservation* store the pointer of map point
    // SceneObv: the observation of ScenePts in F2. pyramid -> all features. the (0)-level stores the raw features in 0-level pyramid, SceneObv[0] is sorted the same as ScenePts.
    // TextObjs: the observed map text objects. TextObservation* store the pointer of map text object
    // **rho: [optimized param] rho of each observed map point (rho of ScenePts[i]->pt)
    // **theta: [optimized param] theta of each observed map text objects (theta of TextObjs[i]->obj)
    // **pose: [optimized param] pose(7). only optimize pose[1] of F2 pose
    // -------------------------------------------------------------------------------------------------------------

    bool FLAG_TEXT = true, FLAG_SCENE = true;
    bool SceneUse0Pyr = true;
    bool SHOW = false;

    std::chrono::steady_clock::time_point t_begin_T = std::chrono::steady_clock::now();

    if(FLAG_SCENE){
        problem->AddParameterBlock(pose[1], 4, new ceres::QuaternionParameterization());
        problem->AddParameterBlock(pose[1]+4, 3);
        LossFunction* loss_function = new HuberLoss(3.0);

        vector<SceneFeature*> vObv = SceneObv[PyBegin];

        for(size_t i0 = 0; i0<vObv.size(); i0++){
            int Idx3d = vObv[i0]->IdxToRaw;
            Vec2 obv = vObv[i0]->feature;
            Vec2 obv0Pyr = SceneObv[0][Idx3d]->feature;

            Vec3 ray = ScenePts[Idx3d]->pt->GetRaydir();
            problem->AddParameterBlock(rho[Idx3d], 1);

            CostFunction *costFunction;
            if(SceneUse0Pyr)
                costFunction = auto_IniBAScene::Create(obv0Pyr, ray, vK[0]);
            else
                costFunction = auto_IniBAScene::Create(obv, ray, vK[PyBegin]);

            problem->AddResidualBlock(costFunction, loss_function, pose[1], pose[1]+4, rho[Idx3d]);
        }
        // ------------------ use numeric solution ------------------
    }

    // B) text objects
    if(FLAG_TEXT){
        LossFunction* loss_function = new HuberLoss(3.0);

        for(size_t i1 = 0; i1<TextObjs.size(); i1++){
            mapText* obj = TextObjs[i1]->obj;
            vector<TextFeature*> reftext = obj->vRefFeature[PyBegin];
            problem->AddParameterBlock(theta[i1], 3);

            // 1. get current frame text object statistic info (mu, sigma)
            vector<Vec2> Ref_DeteBox = obj->vTextDeteRay;

            double Cur_mu, Cur_sigma;
            vector<Vec2> Cur_ProjBox;
            Cur_ProjBox.resize(Ref_DeteBox.size());
            for(size_t iBox = 0; iBox<Ref_DeteBox.size(); iBox++)
                Tool.GetProjText(Ref_DeteBox[iBox], theta[i1], Cur_ProjBox[iBox], pose[1], vK[PyBegin]);

            Tool.CalTextinfo(KF[1]->vFrameImg[PyBegin], Cur_ProjBox, Cur_mu, Cur_sigma, SHOW);

            // for each feature in obj
            for(size_t i2 = 0; i2<reftext.size(); i2++){
                TextFeature* TextFeat = reftext[i2];
                vector<Mat31> TextFeatvRay = TextFeat->neighbourRay;
                vector<double> TextFeatvInten = TextFeat->neighbourNInten;
                CostFunction *costFunction;
                costFunction = nume_IniBAText::Create(KF[1]->vFrameImg[PyBegin], TextFeatvRay, TextFeatvInten, Cur_mu, Cur_sigma, vK[PyBegin]);
                problem->AddResidualBlock(costFunction, loss_function, pose[1], pose[1]+4, theta[i1]);
            }
        }
    }

    // solve
    Solver::Options options;
    options.minimizer_type = TRUST_REGION;
    options.max_num_iterations = 10;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout=false;
    options.num_threads = 1;
    Solver::Summary summary;
    ceres::Solve(options, problem, &summary);

    std::chrono::steady_clock::time_point t_end_T = std::chrono::steady_clock::now();
    double tT= std::chrono::duration_cast<std::chrono::duration<double> >(t_end_T - t_begin_T).count();
    // -------------------------------------------------------------------------------------------------------------


    // 3. after BA. log and visual check
    if(PyBegin==0){
        ShowBAReproj_TextBox(rho, theta, pose, KF, PyBegin, ScenePts, SceneObv, TextObjs, TextLabelImg, (int)1);
    }

}



void optimizer::PyrPoseOptim(frame &F, double *pose, vector<SceneObservation*> ScenePts, vector<vector<SceneFeature *> > SceneObv, vector<TextObservation*> TextObjs, const int &PyBegin, const double &chi2Mono, const double &chi2Text,
                              const int &its, vector<bool> &vPtsGood, vector<bool> &vTextsGood, vector<vector<bool> > &vTextFeatsGood, const vector<int> &FLAGTextObjs, Mat &TextLabelImg)
{
    // ---------------------------------------- introduction -------------------------------------------------------
    // ScenePts: the observed map point. SceneObservation* store the pointer of map point
    // SceneObv: the observation of ScenePts in F. 2V: pyramid -> all features. the (0)-level stores the raw features in 0-level pyramid, SceneObv[0] is sorted the same as ScenePts.
    // TextObjs: the observed map text objects. TextObservation* store the pointer of map text object
    // **pose: [optimized param] pose(7). only optimize pose[1] of F2 pose
    // -------------------------------------------------------------------------------------------------------------

    // -------- settings --------
    bool FLAG_TEXT = true, FLAG_SCENE = true;
    bool SceneUse0Pyr = true;
    bool SceneUseoPyrObv = false;
    bool TEXTOutlier = true, SCENEOutlier = true;
    if(bFlag_noText)
        FLAG_TEXT = false;
    if(bFlag_rapid){
        TEXTOutlier = false;
        SCENEOutlier = false;
    }

    double TextRatio = 0.99;
    bool SHOW = false;

    // ********** scale for optimization **********
    // RS results --------
    double weight_S_x = 1.0/1.2 , weight_S_y = 1.0/1.2;
    double weight_T = 1.0/0.2;
    // RS results --------

    // GoPro results --------
    //    double weight_S_x = double(1.0) / double(0.967242), weight_S_y = double(1.0) / double(0.892949);
    //    double weight_T = double(1.0) / double(0.287876);
    // GoPro results --------
    // ********** scale for optimization **********

    // -------- settings --------

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    // 1. before Optimization, log and visual check

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 2. Optimization  -------------------------------
    ceres::Problem problem;
    vector< ceres::ResidualBlockId> toEval_all;
    int num_s_residual = 0, num_t_residual = 0;

    // A) scene points
    vector<int> vIdx2vPtsGood;
    if(FLAG_SCENE){
        // a)
        problem.AddParameterBlock(pose, 4, new ceres::QuaternionParameterization());
        problem.AddParameterBlock(pose+4, 3);
        LossFunction* loss_function = new HuberLoss(sqrt(5.991));

        // b) get observations
        vector<SceneFeature*> vObv;
        if(!SceneUseoPyrObv)
            vObv = SceneObv[PyBegin];
        else
            vObv = SceneObv[0];

        // c) begin input
        for(size_t i0 = 0; i0<vObv.size(); i0++){
            int Idx3d = vObv[i0]->IdxToRaw;
            if(!vPtsGood[(size_t)Idx3d])
                continue;

            Vec2 obv = vObv[i0]->feature;
            Vec2 obv0Pyr = SceneObv[0][Idx3d]->feature;
            Vec3 rayrho = ScenePts[Idx3d]->pt->GetPtInv();
            Mat44 Trw = ScenePts[Idx3d]->pt->RefKF->mTcw;

            CostFunction *costFunction;
            if(SceneUse0Pyr)
                costFunction = auto_PoseOptimScene::Create(obv0Pyr, rayrho, Trw, vK[0], weight_S_x, weight_S_y);
            else
                costFunction = auto_PoseOptimScene::Create(obv, rayrho, Trw, vK[PyBegin], weight_S_x, weight_S_y);

            ceres::ResidualBlockId r_id = problem.AddResidualBlock(costFunction, loss_function, pose, pose+4);
            toEval_all.push_back(r_id);
            num_s_residual ++;
            vIdx2vPtsGood.push_back(Idx3d);
        }
    }

    std::chrono::steady_clock::time_point t1_S = std::chrono::steady_clock::now();
    double tParamS = std::chrono::duration_cast<std::chrono::duration<double> >(t1_S - t1).count();

    // B) text objs
    // size = residual size
    vector<int> vIdx2vTextsGood, vIdx2vTextFeatsGood;
    // size = TextObjs size
    vector<int> vSizeEachObj = vector<int>(TextObjs.size(), 0);   // feature (residual) size of each obj
    int sum_t_feat = 0;
    if(FLAG_TEXT){

        if(!FLAG_SCENE){
            problem.AddParameterBlock(pose, 4, new ceres::QuaternionParameterization());
            problem.AddParameterBlock(pose+4, 3);
        }
        LossFunction* loss_function = new HuberLoss(3.0);

        // a) begin input
        for(size_t itext=0; itext<TextObjs.size(); itext++){

            if(!vTextsGood[itext])
                continue;

            mapText* obj = TextObjs[itext]->obj;

            Mat31 theta = obj->RefKF->mNcr[obj->GetNidx()];
            vector<TextFeature*> refRay = obj->vRefFeature[PyBegin];
            vector<Vec2> refDeteBox = obj->vTextDeteRay;
            Mat44 Twr = obj->RefKF->mTwc;
            // get mu && sigma
            double Cur_mu, Cur_sigma;
            vector<Vec2> Cur_ProjBox;
            Cur_ProjBox.resize(refDeteBox.size());
            for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++)
                Tool.GetProjText(refDeteBox[iBox], theta, Cur_ProjBox[iBox], pose, Twr, vK[PyBegin]);
            Tool.CalTextinfo(F.vFrameImg[PyBegin], Cur_ProjBox, Cur_mu, Cur_sigma, SHOW);

            int numfeat = 0;
            for(size_t ifeat=0; ifeat<refRay.size(); ifeat++){

                if(!vTextFeatsGood[itext][refRay[ifeat]->IdxToRaw])
                    continue;

                TextFeature* TextFeat = refRay[ifeat];
                vector<Mat31> TextFeatvRay = TextFeat->neighbourRay;
                vector<double> TextFeatvInten = TextFeat->neighbourNInten;
                CostFunction *costFunction;
                costFunction = nume_PoseOptimText::Create(F.vFrameImg[PyBegin], Twr, theta, TextFeatvRay, TextFeatvInten, Cur_mu, Cur_sigma, vK[PyBegin], weight_T);

                ceres::ResidualBlockId r_id = problem.AddResidualBlock(costFunction, loss_function, pose, pose+4);
                toEval_all.push_back(r_id);
                num_t_residual ++;
                vIdx2vTextsGood.push_back(itext);
                vIdx2vTextFeatsGood.push_back(refRay[ifeat]->IdxToRaw);
                numfeat ++;
            }   // each feature
            vSizeEachObj[itext] = numfeat;
            sum_t_feat+=numfeat;
        }   // each objs
    }
    assert(sum_t_feat==num_t_residual);

    std::chrono::steady_clock::time_point t1_solve_begin = std::chrono::steady_clock::now();
    double tParam = std::chrono::duration_cast<std::chrono::duration<double> >(t1_solve_begin - t1_S).count();

    // 2. solve
    Solver::Options options;
    options.minimizer_type = TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout=false;
    options.max_num_iterations = its;
    options.num_threads = 1;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double tT= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1_solve_begin).count();

    // 3. evaluation and outlier judge
    ceres::Problem::EvaluateOptions EvaOptions;
    EvaOptions.residual_blocks = toEval_all;
    vector<double> FinalResidual;
    bool EvaluationFlag = problem.Evaluate(EvaOptions, nullptr, &FinalResidual, nullptr, nullptr);

    assert(FinalResidual.size()==num_s_residual*2+num_t_residual*8);

    if(EvaluationFlag)
    {

        if(SCENEOutlier){
            double chi2MonoUse = chi2Mono;
            if(num_t_residual<50)
                chi2MonoUse = chi2Mono+4;
            int nBadS = 0;
            for(size_t ieval_s = 0; ieval_s<num_s_residual; ieval_s++){

                assert(ieval_s*2>=0);
                assert(ieval_s*2+1<FinalResidual.size());
                assert(ieval_s*2+1<num_s_residual*2);

                double chix = (FinalResidual[ieval_s*2]/weight_S_x) * (FinalResidual[ieval_s*2]/weight_S_x);
                double chiy = (FinalResidual[ieval_s*2+1]/weight_S_y) * (FinalResidual[ieval_s*2+1]/weight_S_y);
                if(chix>chi2MonoUse || chiy>chi2MonoUse){
                    vPtsGood[vIdx2vPtsGood[ieval_s]] = false;
                    nBadS++;
                }
            }

        }

        if(TEXTOutlier){
            int nBadT = 0, ieva_t_begin = num_s_residual*2;
            int FeatNum_tmp = 0, num_badFeat = 0, idx_obj;
            for(size_t ieval_t = 0; ieval_t<num_t_residual; ieval_t++){
                assert(ieval_t*8>=0);
                assert(ieval_t*8+7<FinalResidual.size());

                double IntenErro0 = FinalResidual[ieva_t_begin + ieval_t*8] / weight_T;
                double IntenErro1 = FinalResidual[ieva_t_begin + ieval_t*8 + 1] / weight_T;
                double IntenErro2 = FinalResidual[ieva_t_begin + ieval_t*8 + 2] / weight_T;
                double IntenErro3 = FinalResidual[ieva_t_begin + ieval_t*8 + 3] / weight_T;
                double IntenErro4 = FinalResidual[ieva_t_begin + ieval_t*8 + 4] / weight_T;
                double IntenErro5 = FinalResidual[ieva_t_begin + ieval_t*8 + 5] / weight_T;
                double IntenErro6 = FinalResidual[ieva_t_begin + ieval_t*8 + 6] / weight_T;
                double IntenErro7 = FinalResidual[ieva_t_begin + ieval_t*8 + 7] / weight_T;


                if(std::abs(IntenErro0)>chi2Text || std::abs(IntenErro1)>chi2Text || std::abs(IntenErro2)>chi2Text || std::abs(IntenErro3)>chi2Text ||
                        std::abs(IntenErro4)>chi2Text || std::abs(IntenErro5)>chi2Text || std::abs(IntenErro6)>chi2Text|| std::abs(IntenErro7)>chi2Text)
                {
                    vTextFeatsGood[vIdx2vTextsGood[ieval_t]][vIdx2vTextFeatsGood[ieval_t]] = false;
                    num_badFeat++;
                }

                FeatNum_tmp++;

                // end of one obj
                idx_obj = vIdx2vTextsGood[ieval_t];
                assert(FeatNum_tmp<=vSizeEachObj[idx_obj]);
                if(FeatNum_tmp==vSizeEachObj[idx_obj]){

                    double RatioBad = (double)num_badFeat/(double)vSizeEachObj[idx_obj];
                    if(RatioBad > TextRatio){
                        vTextsGood[vIdx2vTextsGood[ieval_t]] = false;
                        nBadT++;
                    }

                    FeatNum_tmp = 0;
                    num_badFeat = 0;
                    if(idx_obj==(vSizeEachObj.size()-1))
                        assert(ieval_t==(num_t_residual-1));
                }
            }

        }

    }

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    double tIter_1= std::chrono::duration_cast<std::chrono::duration<double> >(t3-t2).count();

    // 4. update observation (for visualization)
    F.vObvGoodPts = vPtsGood;
    assert((int)FLAGTextObjs.size()==(int)vTextsGood.size());
    for(size_t iuptext=0; iuptext<FLAGTextObjs.size(); iuptext++){
        int idxRaw = FLAGTextObjs[iuptext];
        F.vObvGoodTexts[idxRaw] = vTextsGood[iuptext];
        F.vObvGoodTextFeats[idxRaw] = vTextFeatsGood[iuptext];
    }

    // 5. after optimization.
    if(PyBegin==0){
        ShowBAReproj_TextBox(pose, F, PyBegin, ScenePts, SceneObv, TextObjs, TextLabelImg, (int)1);
    }

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    double tIter_2= std::chrono::duration_cast<std::chrono::duration<double> >(t4-t3).count();

}


void optimizer::PyrBA(double **pose, double **theta, double **rho, const vector<keyframe*>  &vKFs, const vector<mapPts*> &vPts, const vector<mapText*> &vTexts, const vector<int> &vmnId2Pts, const vector<bool> &vPtOptim, const vector<int> &vmnId2Texts, const vector<bool> &vTextOptim, const vector<int> &vmnId2vKFs, const vector<int> &InitialIdx, const int &PyBegin,
           const double &chi2Mono, const double &chi2Text, const int &its, Mat &TextLabelImg, const BAStatus &STATE)
{

    // -------- settings --------
    bool FLAG_TEXT = true, FLAG_SCENE = true;
    bool SceneUse0Pyr = true;
    bool SceneUseoPyrObv = false;
    bool TEXTOutlier = true, SCENEOutlier = true;
    double TextRatio = 0.99;
    bool SHOW = false;
    if(bFlag_noText)
        FLAG_TEXT = false;
    if(bFlag_rapid){
        TEXTOutlier = false;
        SCENEOutlier = false;
    }

    // ********** scale for optimization **********
    // RS results --------
    double weight_S_x = 1.0/1.2 , weight_S_y = 1.0/1.2;
    double weight_T = 1.0/0.2;
    // RS results --------
    // ********** scale for optimization **********

    // -------- settings --------
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    vector<bool> FLAG_KFIN = vector<bool>(vKFs.size(), false);
    ceres::Problem problem;
    vector<ceres::ResidualBlockId> toEval_all;
    int num_s_residual = 0, num_t_residual = 0;

    // A) scene points
    vector<int> vIdx2vPtsGood, vIdxS2vKFs;
    if(FLAG_SCENE){
        for(size_t iKF=0; iKF<vKFs.size(); iKF++){
            problem.AddParameterBlock(pose[iKF], 4, new ceres::QuaternionParameterization());
            problem.AddParameterBlock(pose[iKF]+4, 3);
            LossFunction* loss_function = new HuberLoss(sqrt(5.991));

            // frame observation
            vector<SceneFeature*> vSceneObv;
            if(!SceneUseoPyrObv)
                vSceneObv = vKFs[iKF]->vSceneObv2d[PyBegin];
            else
                vSceneObv = vKFs[iKF]->vSceneObv2d[0];
            vector<SceneObservation*> vScene = vKFs[iKF]->vObvPts;
            vector<bool> vPtsGood = vKFs[iKF]->vObvGoodPts;
            assert(vPtsGood.size()==vScene.size());

            for(size_t iScene=0; iScene<vSceneObv.size(); iScene++){
                int Idx2Raw = vSceneObv[iScene]->IdxToRaw;
                if(!vPtsGood[Idx2Raw])
                    continue;

                mapPts* pt = vScene[(size_t)Idx2Raw]->pt;
                Vec2 obv = vSceneObv[iScene]->feature;
                Vec2 obv0Pyr = vKFs[iKF]->vSceneObv2d[0][Idx2Raw]->feature;

                Vec3 ray = pt->GetRaydir();
                int IdxRho = vmnId2Pts[pt->mnId];
                int IdxRef = vmnId2vKFs[pt->RefKF->mnId];

                if(vPtOptim[IdxRho]){
                    // optimize pose_target, pose_host, rho
                    assert(IdxRef!=-1);
                    if(IdxRef==iKF)         // Host != Target
                        continue;

                    problem.AddParameterBlock(rho[IdxRho], 1);
                    CostFunction *costFunction;
                    if(SceneUse0Pyr)
                        costFunction = auto_BAScene::Create(obv0Pyr, ray, vK[0], weight_S_x, weight_S_y);
                    else
                        costFunction = auto_BAScene::Create(obv, ray, vK[PyBegin], weight_S_x, weight_S_y);

                    ceres::ResidualBlockId r_id = problem.AddResidualBlock(costFunction, loss_function, pose[iKF], pose[iKF]+4, pose[IdxRef], pose[IdxRef]+4, rho[IdxRho]);
                    toEval_all.push_back(r_id);

                    FLAG_KFIN[iKF] = true;
                    FLAG_KFIN[IdxRef] = true;

                }else{
                    // optimize pose_target (fix landmark)
                    assert(IdxRef==-1);
                    Vec3 rayrho = pt->GetPtInv();
                    Mat44 Trw = pt->RefKF->mTcw;

                    CostFunction *costFunction;
                    if(SceneUse0Pyr)
                        costFunction = auto_PoseOptimScene::Create(obv0Pyr, rayrho, Trw, vK[0], weight_S_x, weight_S_y);
                    else
                        costFunction = auto_PoseOptimScene::Create(obv, rayrho, Trw, vK[PyBegin], weight_S_x, weight_S_y);

                    ceres::ResidualBlockId r_id = problem.AddResidualBlock(costFunction, loss_function, pose[iKF], pose[iKF]+4);
                    toEval_all.push_back(r_id);

                    FLAG_KFIN[iKF] = true;

                }   // fix or not
                num_s_residual++;
                vIdxS2vKFs.push_back(iKF);
                vIdx2vPtsGood.push_back(Idx2Raw);
            }   // each scene observation
        }   // each kf
    }

    std::chrono::steady_clock::time_point t1S = std::chrono::steady_clock::now();
    double tParamS = std::chrono::duration_cast<std::chrono::duration<double> >(t1S - t1).count();

    // B) text objs
    // size = residual size
    vector<int> vIdxT2vKFs, vIdx2Texts, vIdx2TextFeats, vIdx2IdxTexts;  // vIdx2IdxTexts: for idx_texts
    // size = all KFi TextObjs size
    vector<int> vSizeEachObj;   // feature (residual) size of each obj
    int sum_t_feat = 0, idx_texts = -1;
    if(FLAG_TEXT){
        for(size_t iKF=0; iKF<vKFs.size(); iKF++)
        {
            if(!FLAG_SCENE){
                problem.AddParameterBlock(pose[iKF], 4, new ceres::QuaternionParameterization());
                problem.AddParameterBlock(pose[iKF]+4, 3);
            }
            LossFunction* loss_function = new HuberLoss(3.0);

            // get obj observations of this kf
            vector<int> vNew2RawTextkf;
            vector<TextObservation*> vText = vKFs[iKF]->GetStateTextObvs(TEXTGOOD, vNew2RawTextkf);

            // each obj and each feat input in optimization
            for(size_t iobj=0; iobj<vText.size(); iobj++)
            {
                vSizeEachObj.push_back(0);
                idx_texts++;

                // text outliers check
                int idxRawObj = vNew2RawTextkf[iobj];
                if(!vKFs[iKF]->vObvGoodTexts[idxRawObj])
                    continue;

                mapText* obj = vText[iobj]->obj;
                vector<TextFeature*> refRay = obj->vRefFeature[PyBegin];
                vector<Vec2> refDeteBox = obj->vTextDeteRay;

                int idxtheta = vmnId2Texts[obj->mnId];
                int idxref = vmnId2vKFs[obj->RefKF->mnId];
                // for mu & sigma
                double Cur_mu, Cur_sigma;
                vector<Vec2> Cur_ProjBox;
                Cur_ProjBox.resize(refDeteBox.size());

                if(vTextOptim[idxtheta]){
                    // optimize pose_target, pose_host, theta
                    assert(idxref!=-1);
                    if(idxref==iKF)
                        continue;

                    problem.AddParameterBlock(theta[idxtheta], 3);

                    for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++)
                        Tool.GetProjText(refDeteBox[iBox], theta[idxtheta], Cur_ProjBox[iBox], pose[iKF], pose[idxref], vK[PyBegin]);
                    Tool.CalTextinfo(vKFs[iKF]->vFrameImg[PyBegin], Cur_ProjBox, Cur_mu, Cur_sigma, SHOW);

                    int numfeat = 0;
                    for(size_t ifeat = 0; ifeat<refRay.size(); ifeat++){
                        if(!vKFs[iKF]->vObvGoodTextFeats[idxRawObj][refRay[ifeat]->IdxToRaw])
                            continue;

                        TextFeature* TextFeat = refRay[ifeat];
                        vector<Mat31> TextFeatvRay = TextFeat->neighbourRay;
                        vector<double> TextFeatvInten = TextFeat->neighbourNInten;

                        CostFunction *costFunction;
                        costFunction = nume_BAText::Create(vKFs[iKF]->vFrameImg[PyBegin], TextFeatvRay, TextFeatvInten, Cur_mu, Cur_sigma, vK[PyBegin], weight_T);
                        ceres::ResidualBlockId r_id = problem.AddResidualBlock(costFunction, loss_function, pose[iKF], pose[iKF]+4, pose[idxref], pose[idxref]+4, theta[idxtheta]);
                        toEval_all.push_back(r_id);
                        vIdxT2vKFs.push_back(iKF);
                        vIdx2Texts.push_back(idxRawObj);
                        vIdx2TextFeats.push_back(refRay[ifeat]->IdxToRaw);
                        vIdx2IdxTexts.push_back(idx_texts);
                        num_t_residual++;
                        numfeat ++;

                        FLAG_KFIN[iKF] = true;
                        FLAG_KFIN[idxref] = true;
                    }   // each feature
                    vSizeEachObj[idx_texts] = numfeat;
                    sum_t_feat+=numfeat;
                }else{
                    // optimize pose_target (fix landmark)
                    assert(idxref==-1);
                    Mat31 thetaFix = obj->RefKF->mNcr[obj->GetNidx()];
                    Mat44 Twr = obj->RefKF->mTwc;

                    for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++)
                        Tool.GetProjText(refDeteBox[iBox], thetaFix, Cur_ProjBox[iBox], pose[iKF], Twr, vK[PyBegin]);
                    Tool.CalTextinfo(vKFs[iKF]->vFrameImg[PyBegin], Cur_ProjBox, Cur_mu, Cur_sigma, SHOW);

                    int numfeat = 0;
                    for(size_t ifeat = 0; ifeat<refRay.size(); ifeat++){
                        if(!vKFs[iKF]->vObvGoodTextFeats[idxRawObj][refRay[ifeat]->IdxToRaw])
                            continue;

                        TextFeature* TextFeat = refRay[ifeat];
                        vector<Mat31> TextFeatvRay = TextFeat->neighbourRay;
                        vector<double> TextFeatvInten = TextFeat->neighbourNInten;

                        CostFunction *costFunction;
                        costFunction = nume_PoseOptimText::Create(vKFs[iKF]->vFrameImg[PyBegin], Twr, thetaFix, TextFeatvRay, TextFeatvInten, Cur_mu, Cur_sigma, vK[PyBegin], weight_T);

                        ceres::ResidualBlockId r_id = problem.AddResidualBlock(costFunction, loss_function, pose[iKF], pose[iKF]+4);
                        toEval_all.push_back(r_id);
                        vIdxT2vKFs.push_back(iKF);
                        vIdx2Texts.push_back(idxRawObj);
                        vIdx2TextFeats.push_back(refRay[ifeat]->IdxToRaw);
                        vIdx2IdxTexts.push_back(idx_texts);
                        num_t_residual++;
                        numfeat ++;

                        FLAG_KFIN[iKF] = true;
                    }   // each feature
                    vSizeEachObj[idx_texts] = numfeat;
                    sum_t_feat+=numfeat;
                }   // fix or not
            }   // each obj
        }   // each kf
    }
    assert(sum_t_feat==num_t_residual);

    // fix frames ----------------------------------------
    // fix initial 2 frames
    for(size_t ifix=0; ifix<InitialIdx.size(); ifix++){
        if(FLAG_KFIN[InitialIdx[ifix]]){
            problem.SetParameterBlockConstant( pose[InitialIdx[ifix]] );
            problem.SetParameterBlockConstant( pose[InitialIdx[ifix]]+4 );
        }
    }

    if(STATE==LOCAL){

        int num_KFsOptim=0, idx_fix=0;
        for(size_t ikfop = 0; ikfop<FLAG_KFIN.size(); ikfop++){
            if(FLAG_KFIN[ikfop])
                num_KFsOptim++;
        }

        if(num_KFsOptim>3){
            for(size_t ikf=0; ikf<FLAG_KFIN.size(); ikf++){
                if(FLAG_KFIN[ikf]){
                    problem.SetParameterBlockConstant( pose[ikf] );
                    problem.SetParameterBlockConstant( pose[ikf]+4 );
                    idx_fix++;
                }
                if(idx_fix>=3)
                    break;
            }
        }
    }
    // fix frames ----------------------------------------

    std::chrono::steady_clock::time_point t1_solve_begin = std::chrono::steady_clock::now();
    double tParam = std::chrono::duration_cast<std::chrono::duration<double> >(t1_solve_begin - t1S).count();

    // 2. solve
    Solver::Options options;
    options.minimizer_type = TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout=false;
    options.max_num_iterations = its;
    options.num_threads = 1;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double tT = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1_solve_begin).count();
    // -------------------------------------------------------------------------------------------------------------

    // evaluation and outlier judge
    ceres::Problem::EvaluateOptions EvaOptions;
    EvaOptions.residual_blocks = toEval_all;
    vector<double> FinalResidual;
    bool EvaluationFlag = problem.Evaluate(EvaOptions, nullptr, &FinalResidual, nullptr, nullptr);

    assert(FinalResidual.size()==num_s_residual*2+num_t_residual*8);

    if(EvaluationFlag)
    {
        if(SCENEOutlier)
        {
            double chi2MonoUse = chi2Mono;
            if(num_t_residual<50)
                chi2MonoUse = chi2Mono+4;
            int nBadS = 0;
            for(size_t ieval_s = 0; ieval_s<num_s_residual; ieval_s++){

                assert(ieval_s*2>=0);
                assert(ieval_s*2+1<FinalResidual.size());
                assert(ieval_s*2+1<num_s_residual*2);

                double chix = (FinalResidual[ieval_s*2]/weight_S_x) * (FinalResidual[ieval_s*2]/weight_S_x);
                double chiy = (FinalResidual[ieval_s*2+1]/weight_S_y) * (FinalResidual[ieval_s*2+1]/weight_S_y);
                if(chix>chi2MonoUse || chiy>chi2MonoUse){
                    vKFs[vIdxS2vKFs[ieval_s]]->vObvGoodPts[vIdx2vPtsGood[ieval_s]] = false;
                    nBadS++;
                }
            }
        }

        if(TEXTOutlier){
            int nBadT = 0, ieva_t_begin = num_s_residual*2;
            int FeatNum_tmp = 0, num_badFeat = 0, idx_obj;
            for(size_t ieval_t = 0; ieval_t<num_t_residual; ieval_t++){
                assert(ieval_t*8>=0);
                assert(ieval_t*8+7<FinalResidual.size());

                double IntenErro0 = FinalResidual[ieva_t_begin + ieval_t*8] / weight_T;
                double IntenErro1 = FinalResidual[ieva_t_begin + ieval_t*8 + 1] / weight_T;
                double IntenErro2 = FinalResidual[ieva_t_begin + ieval_t*8 + 2] / weight_T;
                double IntenErro3 = FinalResidual[ieva_t_begin + ieval_t*8 + 3] / weight_T;
                double IntenErro4 = FinalResidual[ieva_t_begin + ieval_t*8 + 4] / weight_T;
                double IntenErro5 = FinalResidual[ieva_t_begin + ieval_t*8 + 5] / weight_T;
                double IntenErro6 = FinalResidual[ieva_t_begin + ieval_t*8 + 6] / weight_T;
                double IntenErro7 = FinalResidual[ieva_t_begin + ieval_t*8 + 7] / weight_T;

                if(std::abs(IntenErro0)>chi2Text || std::abs(IntenErro1)>chi2Text || std::abs(IntenErro2)>chi2Text || std::abs(IntenErro3)>chi2Text ||
                        std::abs(IntenErro4)>chi2Text || std::abs(IntenErro5)>chi2Text || std::abs(IntenErro6)>chi2Text|| std::abs(IntenErro7)>chi2Text)
                {
                    keyframe* kFi = vKFs[vIdxT2vKFs[ieval_t]];
                    int itext = vIdx2Texts[ieval_t];
                    int iFeat = vIdx2TextFeats[ieval_t];
                    kFi->vObvGoodTextFeats[itext][iFeat] = false;
                    num_badFeat++;
                }
                FeatNum_tmp++;

                // end of one obj
                idx_obj = vIdx2IdxTexts[ieval_t];
                assert(FeatNum_tmp<=vSizeEachObj[idx_obj]);
                if(FeatNum_tmp==vSizeEachObj[idx_obj]){
                    keyframe* kFi = vKFs[vIdxT2vKFs[ieval_t]];
                    double RatioBad = (double)num_badFeat/(double)vSizeEachObj[idx_obj];
                    if(RatioBad > TextRatio){
                        int itext = vIdx2Texts[ieval_t];
                        kFi->vObvGoodTexts[itext] = false;
                        nBadT++;
                    }

                    FeatNum_tmp = 0;
                    num_badFeat = 0;
                    if(idx_obj==vSizeEachObj.size()-1)
                        assert(ieval_t==(num_t_residual-1));
                }
            }

        }
    }

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    double tIter_1= std::chrono::duration_cast<std::chrono::duration<double> >(t3-t2).count();

    if(PyBegin==0){
        ShowBAReproj_TextBox(rho, theta, pose, PyBegin, vKFs, vmnId2vKFs, vmnId2Pts, vmnId2Texts, TextLabelImg, (int)1);
    }

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    double tIter_2= std::chrono::duration_cast<std::chrono::duration<double> >(t4-t3).count();

}


void optimizer::PyrGlobalBA(double **pose, double **theta, double **rho, const vector<keyframe*>  &vKFs, const vector<mapPts*> &vPts, const vector<mapText*> &vTexts,
                 const vector<int> &vmnId2Pts, const vector<int> &vmnId2Texts, const vector<int> &vmnId2vKFs, const vector<int> &InitialIdx, vector<vector<bool>> &vPtsGoodkf,
                 const int &PyBegin, const double &chi2Mono, const int &its)
{

    // -------- settings --------
    bool FLAG_TEXT = false, FLAG_SCENE = true;
    bool SceneUse0Pyr = true;
    bool SHOW = false;
    // -------- settings --------

    // ********************************** Step 1. Problem establish ********************************** //

    vector<bool> FLAG_KFIN = vector<bool>(vKFs.size(), false);
    std::chrono::steady_clock::time_point t_begin_T = std::chrono::steady_clock::now();
    ceres::Problem problem;

    if(FLAG_SCENE)
    {
        for(size_t iKF=0; iKF<vKFs.size(); iKF++)
        {
            problem.AddParameterBlock(pose[iKF], 4, new ceres::QuaternionParameterization());
            problem.AddParameterBlock(pose[iKF]+4, 3);
            LossFunction* loss_function = new HuberLoss(sqrt(5.991));

            // get this KF observation
            vector<SceneFeature*> vSceneObv = vKFs[iKF]->vSceneObv2d[PyBegin];
            vector<SceneObservation*> vScene = vKFs[iKF]->vObvPts;
            vector<bool> vPtsGood = vPtsGoodkf[iKF];

            for(size_t iScene=0; iScene<vSceneObv.size(); iScene++)
            {
                int Idx2Raw = vSceneObv[iScene]->IdxToRaw;

                mapPts* pt = vScene[(size_t)Idx2Raw]->pt;
                int IdxRho = vmnId2Pts[pt->mnId];
                int IdxRef = vmnId2vKFs[pt->RefKF->mnId];
                    // cond. if mapPt is bad(observation bad/replaced..), skip
                    // cond. Host != Target, skip
                if(IdxRho<0 || IdxRef==iKF)
                    continue;

                Vec2 obv = vSceneObv[iScene]->feature;
                Vec2 obv0Pyr = vKFs[iKF]->vSceneObv2d[0][Idx2Raw]->feature;

                Vec3 ray = pt->GetRaydir();

                problem.AddParameterBlock(rho[IdxRho], 1);
                CostFunction *costFunction;
                if(SceneUse0Pyr)
                    costFunction = auto_BASceneNW::Create(obv0Pyr, ray, vK[0]);
                else
                    costFunction = auto_BASceneNW::Create(obv, ray, vK[PyBegin]);

                problem.AddResidualBlock(costFunction, loss_function, pose[iKF], pose[iKF]+4, pose[IdxRef], pose[IdxRef]+4, rho[IdxRho]);

                FLAG_KFIN[iKF] = true;
                FLAG_KFIN[IdxRef] = true;

            }   // b) each observed mapPts
        }   // a) each KF
    }


    vector<vector<int>> vNew2RawText;
    if(FLAG_TEXT)
    {
        for(size_t iKF=0; iKF<vKFs.size(); iKF++)
        {
            if(!FLAG_SCENE){
                problem.AddParameterBlock(pose[iKF], 4, new ceres::QuaternionParameterization());
                problem.AddParameterBlock(pose[iKF]+4, 3);
            }
            LossFunction* loss_function = new HuberLoss(3.0);

            vector<int> vNew2RawTextkf;
            vector<TextObservation*> vText = vKFs[iKF]->GetStateTextObvs(TEXTGOOD, vNew2RawTextkf);
            vNew2RawText.push_back(vNew2RawTextkf);

            for(size_t iobj=0; iobj<vText.size(); iobj++)
            {
                mapText* obj = vText[iobj]->obj;
                int idxtheta = vmnId2Texts[obj->mnId];
                int idxref = vmnId2vKFs[obj->RefKF->mnId];
                    // cond. if mapObj is bad(observation bad/replaced..), skip
                        // text cannot run into this condition for theta only store good texts
                    // cond. Host != Target, skip
                if(idxtheta<0 || idxref==iKF)
                    continue;

                problem.AddParameterBlock(theta[idxtheta], 3);

                vector<TextFeature*> refRay = obj->vRefFeature[PyBegin];
                vector<Vec2> refDeteBox = obj->vTextDeteRay;

                // for mu & sigma
                double Cur_mu, Cur_sigma;
                vector<Vec2> Cur_ProjBox;
                Cur_ProjBox.resize(refDeteBox.size());

                for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++)
                    Tool.GetProjText(refDeteBox[iBox], theta[idxtheta], Cur_ProjBox[iBox], pose[iKF], pose[idxref], vK[PyBegin]);
                Tool.CalTextinfo(vKFs[iKF]->vFrameImg[PyBegin], Cur_ProjBox, Cur_mu, Cur_sigma, SHOW);

                // for each feature in obj
                for(size_t ifeat = 0; ifeat<refRay.size(); ifeat++)
                {
                    TextFeature* TextFeat = refRay[ifeat];
                    vector<Mat31> TextFeatvRay = TextFeat->neighbourRay;
                    vector<double> TextFeatvInten = TextFeat->neighbourNInten;

                    CostFunction *costFunction;
                    costFunction = nume_BAText::Create(vKFs[iKF]->vFrameImg[PyBegin], TextFeatvRay, TextFeatvInten, Cur_mu, Cur_sigma, vK[PyBegin], 1.0);
                    problem.AddResidualBlock(costFunction, loss_function, pose[iKF], pose[iKF]+4, pose[idxref], pose[idxref]+4, theta[idxtheta]);

                    FLAG_KFIN[iKF] = true;
                    FLAG_KFIN[idxref] = true;

                }   // c) each feature in text obj
            }   // b) each text obj
        }   // a) each KF
    }

    // fix initial 2 frames
    for(size_t ifix=0; ifix<InitialIdx.size(); ifix++){
        if(FLAG_KFIN[InitialIdx[ifix]]){
            problem.SetParameterBlockConstant( pose[InitialIdx[ifix]] );
            problem.SetParameterBlockConstant( pose[InitialIdx[ifix]]+4 );
        }
    }

    // ********************************** Step 2. Problem Solve ********************************** //
    Solver::Options options;
    options.minimizer_type = TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout=false;
    options.max_num_iterations = its;
    options.num_threads = 1;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if(summary.num_successful_steps<0){
        cerr <<"Pyr GlobalBA failed";
        exit(-1);
    }

    std::chrono::steady_clock::time_point t_end_T = std::chrono::steady_clock::now();
    double tT= std::chrono::duration_cast<std::chrono::duration<double> >(t_end_T - t_begin_T).count();


}

void optimizer::PyrLandmarkers(double **pose, double **theta, double **rho, const vector<keyframe*>  &vKFs, const vector<mapPts*> &vPts, const vector<mapText*> &vTexts,
                               const vector<int> &vmnId2Pts, const vector<int> &vmnId2Texts, const vector<int> &vmnId2vKFs, vector<vector<bool>> &vPtsGoodkf,
                               const int &PyBegin, const double &chi2Mono, const double &chi2Text, const int &its, vector<Mat> &TextLabelImg)
{
    // -------- settings --------
    bool FLAG_TEXT = true, FLAG_SCENE = true;
    bool SceneUse0Pyr = true;
    bool SHOW = false;
    bool TEXTOutlier = false, SCENEOutlier = true;
    double TextRatio = 0.99;
    // -------- settings --------

    // ********************************** Step 1. Problem establish ********************************** //

    vector<bool> FLAG_KFIN = vector<bool>(vKFs.size(), false);
    std::chrono::steady_clock::time_point t_begin_T = std::chrono::steady_clock::now();
    ceres::Problem problem;

    if(FLAG_SCENE)
    {
        LossFunction* loss_function = new HuberLoss(sqrt(5.991));
        for(size_t iKF=0; iKF<vKFs.size(); iKF++)
        {
            // get this KF observation
            vector<SceneFeature*> vSceneObv = vKFs[iKF]->vSceneObv2d[PyBegin];
            vector<SceneObservation*> vScene = vKFs[iKF]->vObvPts;
            // ok will add todo
            vector<bool> vPtsGood = vPtsGoodkf[iKF];
            assert(vPtsGood.size()==vScene.size());

            for(size_t iScene=0; iScene<vSceneObv.size(); iScene++)
            {
                int Idx2Raw = vSceneObv[iScene]->IdxToRaw;
                    // cond1. observation is not good in this KF
                // ok will add todo
                if(!vPtsGood[Idx2Raw])
                    continue;

                mapPts* pt = vScene[(size_t)Idx2Raw]->pt;
                int IdxRho = vmnId2Pts[pt->mnId];
                    // cond2. if mapPt is bad(observation bad/replaced..), skip
                    // cond3. Host != Target, skip
                if(IdxRho<0 || pt->RefKF->mnId==vKFs[iKF]->mnId)
                    continue;

                Vec2 obv = vSceneObv[iScene]->feature;
                Vec2 obv0Pyr = vKFs[iKF]->vSceneObv2d[0][Idx2Raw]->feature;

                Vec3 ray = pt->GetRaydir();

                Mat44 Tcr = vKFs[iKF]->mTcw * pt->RefKF->mTcw.inverse();

                problem.AddParameterBlock(rho[IdxRho], 1);
                CostFunction *costFunction;
                if(SceneUse0Pyr)
                    costFunction = auto_RhoScene::Create(obv0Pyr, ray, Tcr, vK[0]);
                else
                    costFunction = auto_RhoScene::Create(obv, ray, Tcr, vK[PyBegin]);

                problem.AddResidualBlock(costFunction, loss_function, rho[IdxRho]);

            }   // b) each observed mapPts
        }   // a) each KF
    }

    vector<vector<int>> vNew2RawText;
    vector<vector<TextObservation*>> vGoodTexts;
    if(FLAG_TEXT)
    {
        LossFunction* loss_function = new HuberLoss(2.0);
        for(size_t iKF=0; iKF<vKFs.size(); iKF++)
        {
            vector<int> vNew2RawTextkf;
            vector<TextObservation*> vText = vKFs[iKF]->GetStateTextObvs(TEXTGOOD, vNew2RawTextkf);
            vNew2RawText.push_back(vNew2RawTextkf);
            vGoodTexts.push_back(vText);

            for(size_t iobj=0; iobj<vText.size(); iobj++)
            {

                mapText* obj = vText[iobj]->obj;
                int idxtheta = vmnId2Texts[obj->mnId];
                    // cond. if mapObj is bad(observation bad/replaced..), skip
                        // text cannot run into this condition for theta only store good texts
                    // cond. Host != Target, skip
                if(idxtheta<0 || vText[iobj]->obj->RefKF->mnId==vKFs[iKF]->mnId)
                    continue;

                problem.AddParameterBlock(theta[idxtheta], 3);

                vector<TextFeature*> refRay = obj->vRefFeature[PyBegin];
                vector<Vec2> refDeteBox = obj->vTextDeteRay;
                Mat44 Tcr = vKFs[iKF]->mTcw * vText[iobj]->obj->RefKF->mTwc;

                // for mu & sigma
                double Cur_mu, Cur_sigma;
                vector<Vec2> Cur_ProjBox;
                Cur_ProjBox.resize(refDeteBox.size());

                for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++)
                    Tool.GetProjText(refDeteBox[iBox], theta[idxtheta], Cur_ProjBox[iBox], Tcr, vK[PyBegin]);
                Tool.CalTextinfo(vKFs[iKF]->vFrameImg[PyBegin], Cur_ProjBox, Cur_mu, Cur_sigma, SHOW);


                // for each feature in obj
                for(size_t ifeat = 0; ifeat<refRay.size(); ifeat++)
                {

                    TextFeature* TextFeat = refRay[ifeat];
                    vector<Mat31> TextFeatvRay = TextFeat->neighbourRay;
                    vector<double> TextFeatvInten = TextFeat->neighbourNInten;

                    CostFunction *costFunction;
                    costFunction = nume_thetaText::Create(vKFs[iKF]->vFrameImg[PyBegin], TextFeatvRay, TextFeatvInten, Cur_mu, Cur_sigma, Tcr, vK[PyBegin]);
                    problem.AddResidualBlock(costFunction, loss_function, theta[idxtheta]);

                }   // c) each feature in text obj
            }   // b) each text obj
        }   // a) each KF
    }

    // ********************************** Step 2. Problem Solve ********************************** //
    Solver::Options options;
    options.minimizer_type = TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout=false;
    options.max_num_iterations = its;
    options.num_threads = 1;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if(summary.num_successful_steps<0){
        cout<<"PyrLandmarkers Optimize failed in pyramid "<<PyBegin<<endl;
    }

    std::chrono::steady_clock::time_point t_end_T = std::chrono::steady_clock::now();
    double tLoop = std::chrono::duration_cast<std::chrono::duration<double> >(t_end_T - t_begin_T).count();

    // ********************************** Step 3. Outlier check ********************************** //
    if(FLAG_SCENE && SCENEOutlier)
    {
        int nBad;
        for(size_t iKFCheck = 0; iKFCheck<vKFs.size(); iKFCheck++){
            vector<SceneFeature*> vSceneObv = vKFs[iKFCheck]->vSceneObv2d[PyBegin];
            vector<SceneObservation*> vScene = vKFs[iKFCheck]->vObvPts;
            vector<bool> vPtsGood = vPtsGoodkf[iKFCheck];
            assert(vPtsGood.size()==vScene.size());
            nBad = 0;

            for(size_t iSceneCheck=0; iSceneCheck<vSceneObv.size(); iSceneCheck++)
            {
                int Idx2Raw = vSceneObv[iSceneCheck]->IdxToRaw;
                if(!vPtsGood[Idx2Raw]){
                    nBad++;
                    continue;
                }

                Vec2 obv = vSceneObv[iSceneCheck]->feature;
                Vec2 obv0Pyr = vKFs[iKFCheck]->vSceneObv2d[0][Idx2Raw]->feature;

                mapPts* pt = vScene[(size_t)Idx2Raw]->pt;
                Vec3 ray = pt->GetRaydir();
                int IdxRho = vmnId2Pts[pt->mnId];
                int IdxRef = vmnId2vKFs[pt->RefKF->mnId];

                    // cond. if mapPt is bad(observation bad/replaced..), skip
                    // cond. Host != Target, skip
                if(IdxRho<0 || pt->RefKF->mnId==vKFs[iKFCheck]->mnId)
                    continue;

                double rhores = rho[IdxRho][0];
                Mat44 Tcrres = vKFs[iKFCheck]->mTcw * pt->RefKF->mTcw.inverse();
                // double chi;
                double chix, chiy;
                if(SceneUse0Pyr){
                    Mat31 p = vK[0] * (Tcrres.block<3,3>(0,0) * ray/rhores + Tcrres.block<3,1>(0,3));
                    chix = (p(0)/p(2)-obv0Pyr(0)) * (p(0)/p(2)-obv0Pyr(0));
                    chiy = (p(1)/p(2)-obv0Pyr(1)) * (p(1)/p(2)-obv0Pyr(1));
                }else{
                    Mat31 p = vK[PyBegin] * (Tcrres.block<3,3>(0,0) * ray/rhores + Tcrres.block<3,1>(0,3));
                    chix = (p(0)/p(2)-obv(0)) * (p(0)/p(2)-obv(0));
                    chiy = (p(1)/p(2)-obv(1)) * (p(1)/p(2)-obv(1));
                }

                if(chix>chi2Mono || chiy>chi2Mono){
                    vPtsGood[Idx2Raw] = false;
                    nBad++;
                }
            }       // each obv pts
            // 4. update observation (for visualization)
            vPtsGoodkf[iKFCheck] = vPtsGood;
            vKFs[iKFCheck]->vObvGoodPts = vPtsGood;

        }
    }

    if(FLAG_TEXT && TEXTOutlier)
    {
        // each kf
        for(size_t iKFCheck = 0; iKFCheck<vKFs.size(); iKFCheck++){

            vector<TextObservation*> vText = vGoodTexts[iKFCheck];
            vector<int> vNew2RawTextkf = vNew2RawText[iKFCheck];
            int num_badText = 0;

            for(size_t iobj=0; iobj<vText.size(); iobj++)
            {
                int idxRawObj = vNew2RawTextkf[iobj];
                if(!vKFs[iKFCheck]->vObvGoodTexts[idxRawObj]){
                    num_badText++;
                    continue;
                }

                mapText* obj = vText[iobj]->obj;

                vector<TextFeature*> refRay = obj->vRefFeature[PyBegin];
                int ObjFeatNum = (int)obj->vRefFeature[0].size();
                vector<Vec2> refDeteBox = obj->vTextDeteRay;
                int idxtheta = vmnId2Texts[obj->mnId];
                int idxref = vmnId2vKFs[obj->RefKF->mnId];

                assert((int)idxtheta!=-1);
                assert((int)idxref!=-1);
                if(idxref==iKFCheck)
                    continue;

                // for mu & sigma
                double Cur_mu, Cur_sigma;
                vector<Vec2> Cur_ProjBox;
                Cur_ProjBox.resize(refDeteBox.size());
                bool FLAG_PROJ = true;
                for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++){
                    bool PROJ = Tool.GetProjText(refDeteBox[iBox], theta[idxtheta], Cur_ProjBox[iBox], pose[iKFCheck], pose[idxref], vK[PyBegin]);
                    if(!PROJ)
                        FLAG_PROJ = PROJ;
                }
                bool FLAG_STATISTIC = Tool.CalTextinfo(vKFs[iKFCheck]->vFrameImg[PyBegin], Cur_ProjBox, Cur_mu, Cur_sigma, SHOW);
                if(!FLAG_STATISTIC || !FLAG_PROJ){
                    vKFs[iKFCheck]->vObvGoodTexts[idxRawObj] = false;
                    num_badText++;
                    continue;
                }

                Mat44 Tcrres;
                Mat31 thetaCheck;
                int num_badFeat = 0;
                Tcrres = vKFs[iKFCheck]->mTcw * vKFs[iKFCheck]->mTcw.inverse();
                thetaCheck = Mat31(theta[idxtheta][0], theta[idxtheta][1], theta[idxtheta][2]);
                for(size_t ifeat = 0; ifeat<refRay.size(); ifeat++){

                    if(!vKFs[iKFCheck]->vObvGoodTextFeats[idxRawObj][refRay[ifeat]->IdxToRaw]){
                        num_badFeat++;
                        continue;
                    }

                    TextFeature* TextFeat = refRay[ifeat];
                    vector<Mat31> TextFeatvRay = TextFeat->neighbourRay;
                    vector<double> TextFeatvInten = TextFeat->neighbourNInten;

                    for(size_t ipat = 0; ipat<TextFeatvRay.size(); ipat++){
                        Mat31 ray = TextFeatvRay[ipat];
                        double invz = -ray.transpose() * thetaCheck;
                        Mat31 Pc = vK[PyBegin] * ( Tcrres.block<3,3>(0,0) * ray/invz + Tcrres.block<3,1>(0,3) );
                        double uPred = Pc(0)/Pc(2);
                        double vPred = Pc(1)/Pc(2);
                        // depth negative
                        if(Pc(2)<0){
                            vKFs[iKFCheck]->vObvGoodTextFeats[idxRawObj][refRay[ifeat]->IdxToRaw] = false;
                            num_badFeat++;
                            break;
                        }

                        double IntenCur;
                        bool IN;
                        IN = Tool.GetIntenBilinterPtr(Vec2(uPred, vPred), vKFs[iKFCheck]->vFrameImg[PyBegin], IntenCur);

                        if(!IN){
                            vKFs[iKFCheck]->vObvGoodTextFeats[idxRawObj][refRay[ifeat]->IdxToRaw] = false;
                            num_badFeat++;
                            break;
                        }else{
                            assert(ipat>=0);
                            assert(ipat<TextFeatvInten.size());

                            double IntenCurN = (IntenCur-Cur_mu)/Cur_sigma;
                            double IntenErro = std::abs(IntenCurN-TextFeatvInten[ipat]);
                            if(IntenErro>chi2Text){
                                vKFs[iKFCheck]->vObvGoodTextFeats[idxRawObj][refRay[ifeat]->IdxToRaw] = false;
                                num_badFeat++;
                                break;
                            }
                        }
                    }   // each pattern in 1 feature
                }   // each feature in 1 obj

                double RatioBad = (double)num_badFeat/(double)ObjFeatNum;
                assert(idxRawObj>=0);
                assert(idxRawObj<vKFs[iKFCheck]->vObvGoodTexts.size());
                if(RatioBad > TextRatio){
                    vKFs[iKFCheck]->vObvGoodTexts[idxRawObj] = false;
                    num_badText++;
                }


            }       // each obv obj

        }           // each keyframe
    }

    // ********************************** Step 4. check ********************************** //
    if(PyBegin==0){
        ShowBAReproj_TextBox(rho, theta, pose, PyBegin, vKFs, vmnId2vKFs, vmnId2Pts, vmnId2Texts, TextLabelImg, (int)1);
    }


}

bool optimizer::PyrThetaOptim(const vector<cv::Mat> &vImg, const vector<Mat44,Eigen::aligned_allocator<Mat44>> &vTcr, const vector<keyframe*> &vKFs, mapText *obj, const int &PyBegin, double *theta, Mat33 &thetaVariance)
{
    bool SHOW = false;
    std::chrono::steady_clock::time_point t_begin_T = std::chrono::steady_clock::now();

    ceres::Problem problem;
    LossFunction* loss_function = nullptr;
    vector<TextFeature*> refRay = obj->vRefFeature[PyBegin];
    vector<Vec2> refDeteBox = obj->vTextDeteRay;

    for(size_t ifs=0; ifs<vImg.size(); ifs++){
        double Cur_mu, Cur_sigma;
        vector<Vec2> Cur_ProjBox;
        Cur_ProjBox.resize(refDeteBox.size());

        for(size_t iBox = 0; iBox<refDeteBox.size(); iBox++)
            Tool.GetProjText(refDeteBox[iBox], theta, Cur_ProjBox[iBox], vTcr[ifs], vK[PyBegin]);

        Tool.CalTextinfo(vImg[ifs], Cur_ProjBox, Cur_mu, Cur_sigma, SHOW);

        for(size_t ifeat=0; ifeat<refRay.size(); ifeat++){

            TextFeature* TextFeat = refRay[ifeat];
            vector<Mat31> TextFeatvRay = TextFeat->neighbourRay;
            vector<double> TextFeatvInten = TextFeat->neighbourNInten;

            CostFunction *costFunction;
            costFunction = nume_thetaText::Create(vImg[ifs], TextFeatvRay, TextFeatvInten, Cur_mu, Cur_sigma, vTcr[ifs], vK[PyBegin]);
            problem.AddResidualBlock(costFunction, loss_function, theta);
        }
    }

    // solve
    Solver::Options options;
    options.minimizer_type = TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout=false;
    options.num_threads = 1;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if(summary.num_successful_steps<0){
        return false;
    }

    std::chrono::steady_clock::time_point t_end_T = std::chrono::steady_clock::now();
    double tT= std::chrono::duration_cast<std::chrono::duration<double> >(t_end_T - t_begin_T).count();

    // -------- covariance --------
    Covariance::Options options_Cov;
    Covariance covariance(options_Cov);
    vector<pair<const double*, const double*> > covariance_blocks;
    covariance_blocks.push_back(make_pair(theta, theta));

    bool flag_variance = covariance.Compute(covariance_blocks, &problem);
    if(flag_variance)
    {
        double covariance_xx[3 * 3];
        covariance.GetCovarianceBlock(theta, theta, covariance_xx);
        thetaVariance(0,0) = covariance_xx[0];
        thetaVariance(0,1) = covariance_xx[1];
        thetaVariance(0,2) = covariance_xx[2];
        thetaVariance(1,0) = covariance_xx[3];
        thetaVariance(1,1) = covariance_xx[4];
        thetaVariance(1,2) = covariance_xx[5];
        thetaVariance(2,0) = covariance_xx[6];
        thetaVariance(2,1) = covariance_xx[7];
        thetaVariance(2,2) = covariance_xx[8];
    }
    // -------- covariance --------

    return true;
}

// --------------- update ----------------
// update KFCur->vTextDeteCorMap, KFCur->vObvText, mapText->vObvkeyframe
void optimizer::UpdateTrackedTextBA(vector<TextObservation*> &TextObjs, const cv::Mat &ImgTextProjLabel, keyframe* KFCur, const bool &INITIAL)
{
    if(INITIAL)
        KFCur->vTextDeteCorMap = vector<int>(KFCur->vTextDete.size(), -1);

    for(size_t i0 = 0; i0<KFCur->vTextDeteCenter.size(); i0++){
        Vec2 Center = KFCur->vTextDeteCenter[i0];

        int u = round(Center(0));
        int v = round(Center(1));
        float* img_ptr = (float*)ImgTextProjLabel.data + v*ImgTextProjLabel.cols + u;
        float label = img_ptr[0];

        if(label<0)
            continue;

        int IdxTextLabel = (int)label;
        // for keyframe: UpdateTextObservation
        TextObjs[IdxTextLabel]->idx.push_back(i0);

        // for keyframe text detection
        KFCur->vTextDeteCorMap[i0] = TextObjs[IdxTextLabel]->obj->mnId;

        // for map text: UpdateKFObserv
        bool CHANGE_OBVIDX = TextObjs[IdxTextLabel]->obj->UpdateKFObserv(KFCur, (int)i0);

        // UpdateSemantic*2(initial use)
        if(CHANGE_OBVIDX){
            mapText* mobj = TextObjs[IdxTextLabel]->obj;
            if(mobj->STATE!=TEXTBAD)
                UpdateSemantic_MapObjs_single(mobj, KFCur);
        }
    }

}

void optimizer::UpdateTrackedTextBA(vector<TextObservation*> &TextObjs, const vector<int> &vIdxState2Raw, const cv::Mat &ImgTextProjLabel, keyframe* KFCur, const bool &INITIAL)
{
    // initial KFCur->vTextDeteCorMap
    if(INITIAL)
        KFCur->vTextDeteCorMap = vector<int>(KFCur->vTextDete.size(), -1);

    // Label is the idx of TextObjs
    for(size_t i0 = 0; i0<KFCur->vTextDeteCenter.size(); i0++){
        Vec2 Center = KFCur->vTextDeteCenter[i0];

        int u = round(Center(0));
        int v = round(Center(1));
        float* img_ptr = (float*)ImgTextProjLabel.data + v*ImgTextProjLabel.cols + u;
        float label = img_ptr[0];

        if(label<0)
            continue;

        int IdxTextLabel = (int)label;
        int IdxTExtLabelRaw = vIdxState2Raw[IdxTextLabel];

        // for keyframe: KF->vObvText
        if(TextObjs[IdxTextLabel]->idx.size()!=0){
            bool FLAG_HAVE = false;
            vector<int> vIdxRaw = TextObjs[IdxTextLabel]->idx;
            for(size_t iIdx=0; iIdx<vIdxRaw.size(); iIdx++){
                if(vIdxRaw[iIdx]==i0){
                    FLAG_HAVE = true;
                    break;
                }
            }
            if(!FLAG_HAVE){
                TextObjs[IdxTextLabel]->idx.push_back(i0);
            }
        }else{
            TextObjs[IdxTExtLabelRaw]->idx.push_back(i0);
        }

        // for keyframe text detection
        KFCur->vTextDeteCorMap[i0] = TextObjs[IdxTExtLabelRaw]->obj->mnId;

        // for map text: UpdateKFObserv
        bool CHANGE_OBVIDX = TextObjs[IdxTExtLabelRaw]->obj->UpdateKFObserv(KFCur, (int)i0);

        // UpdateSemantic
        if(CHANGE_OBVIDX){
            mapText* mobj = TextObjs[IdxTExtLabelRaw]->obj;
            if(mobj->STATE!=TEXTBAD)
                UpdateSemantic_MapObjs_single(mobj, KFCur);
        }

    }

}

// update F.vTextDeteCorMap, F.vObvText, F.vObvkeyframe
void optimizer::UpdateTrackedTextPOSE(vector<TextObservation*> &TextObjs, const cv::Mat &ImgTextProjLabel, frame &F)
{
    F.vTextDeteCorMap = vector<int>(F.vTextDete.size(), -1);

    // Label is the idx of TextObjs
    for(size_t i0 = 0; i0<F.vTextDeteCenter.size(); i0++){

        Vec2 Center = F.vTextDeteCenter[i0];

        int u = round(Center(0));
        int v = round(Center(1));
        float* img_ptr = (float*)ImgTextProjLabel.data + v*ImgTextProjLabel.cols + u;
        float label = img_ptr[0];

        if(label<0)
            continue;

        int IdxTextLabel = (int)label;

        // for frame: UpdateTextObservation
        if(TextObjs[IdxTextLabel]->idx.size()!=0){
            bool FLAG_HAVE = false;
            vector<int> vIdxRaw = TextObjs[IdxTextLabel]->idx;
            for(size_t iIdx=0; iIdx<vIdxRaw.size(); iIdx++){
                if(vIdxRaw[iIdx]==i0){
                    FLAG_HAVE = true;
                    break;
                }
            }
            if(!FLAG_HAVE){
                TextObjs[IdxTextLabel]->idx.push_back(i0);
            }
        }else{
            TextObjs[IdxTextLabel]->idx.push_back(i0);
        }

        // for frame text detection
        F.vTextDeteCorMap[i0] = TextObjs[IdxTextLabel]->obj->mnId;

    }

}

// --------------- update ----------------


// ------------- visualization -------------

// for initial BA pyramid projection (level==0)(compact form)
// ShowText: text object box, get TextLabel.
void optimizer::ShowBAReproj_TextBox(double **rho, double **theta, double **pose, const vector<keyframe*> &KF, const int &PyBegin,
                             const vector<SceneObservation *> ScenePts, const vector<vector<SceneFeature*>> SceneObv, const vector<TextObservation*> TextObjs, cv::Mat &TextLabel, const int op)
{
    double fx = vfx[PyBegin], fy = vfy[PyBegin], cx = vcx[PyBegin], cy = vcy[PyBegin];

    Eigen::Quaterniond q(pose[1][0], pose[1][1], pose[1][2], pose[1][3]);
    Mat31 t(pose[1][4], pose[1][5], pose[1][6]);
    q = q.normalized();
    Mat33 R(q);
    Mat44 Tcw;
    Tcw.setIdentity();
    Tcw.block<3,3>(0,0) = R;
    Tcw.block<3,1>(0,3) = t;

    // 1. text objects
    vector<vector<Vec2>> textBoxpred;   // text obj -> box 4 pts
    vector<int> textLabel, textmnId;
    vector<TextInfo> textinfo;
    for(size_t i0 = 0; i0<TextObjs.size(); i0++){  // a) for each text object
        mapText* textobj = TextObjs[i0]->obj;
        vector<Vec2> BoxRef = textobj->vTextDeteRay;

        Mat44 Tcr = Tcw * textobj->RefKF->mTwc;
        Mat33 Rcr(Tcr.block<3,3>(0,0));
        Mat31 tcr(Tcr.block<3,1>(0,3));
        Mat31 thetaobj =  Mat31(theta[i0][0],theta[i0][1],theta[i0][2]);
        vector<Vec2> textBoxObj;
        // B) 4 text box pts
        for(size_t i2 = 0; i2<BoxRef.size(); i2++){
            Mat31 ray = Mat31(BoxRef[i2](0), BoxRef[i2](1), 1.0 );
            double invz = - ray.transpose() * thetaobj;
            Mat31 p = Rcr * ray/invz + tcr;
            double u = fx * p(0,0)/p(2,0) + cx;
            double v = fy * p(1,0)/p(2,0) + cy;
            textBoxObj.push_back(Vec2(u,v));
        }

        textBoxpred.push_back(textBoxObj);
        textLabel.push_back(i0);
        textmnId.push_back(textobj->mnId);
        textinfo.push_back(textobj->TextMean);
    }

    vector<Mat> ImgTextAll = Tool.TextBoxWithFill(textBoxpred, textmnId, textinfo, KF[1]->vFrameImg[PyBegin], textLabel);

    TextLabel = ImgTextAll[0];
    Mat ImgText = ImgTextAll[1];

}


void optimizer::ShowBAReproj_TextBox(double *pose, const frame &F, const int &PyBegin, const vector<SceneObservation *> ScenePts, const vector<vector<SceneFeature*>> SceneObv, const vector<TextObservation*> TextObjs, Mat &TextLabel, const int op)
{
    double fx = vfx[PyBegin], fy = vfy[PyBegin], cx = vcx[PyBegin], cy = vcy[PyBegin];

    Eigen::Quaterniond q(pose[0], pose[1], pose[2], pose[3]);
    Mat31 t(pose[4], pose[5], pose[6]);
    q = q.normalized();
    Mat33 R(q);
    Mat44 Tcw;
    Tcw.setIdentity();
    Tcw.block<3,3>(0,0) = R;
    Tcw.block<3,1>(0,3) = t;

    // 1. text objects
    vector<vector<Vec2>> textBoxpred;   // text obj -> box 4 pts
    vector<vector<Vec2>> textpredkf;
    vector<int> textLabelidx, textmnId;
    vector<TextInfo> textinfo;
    for(size_t i0 = 0; i0<TextObjs.size(); i0++){  // a) for each text object
        mapText* textobj = TextObjs[i0]->obj;
        vector<Vec2> BoxRef = textobj->vTextDeteRay;

        Mat44 Tcr = Tcw * textobj->RefKF->mTwc;
        Mat33 Rcr(Tcr.block<3,3>(0,0));
        Mat31 tcr(Tcr.block<3,1>(0,3));
        Mat31 thetaobj =  textobj->RefKF->mNcr[textobj->GetNidx()];
        vector<Vec2> textBoxObj;
        // B) 4 text box pts
        for(size_t i2 = 0; i2<BoxRef.size(); i2++){
            Mat31 ray = Mat31(BoxRef[i2](0), BoxRef[i2](1), 1.0 );
            double invz = - ray.transpose() * thetaobj;
            Mat31 p = Rcr * ray/invz + tcr;
            double u = fx * p(0,0)/p(2,0) + cx;
            double v = fy * p(1,0)/p(2,0) + cy;
            textBoxObj.push_back(Vec2(u,v));
        }

        textmnId.push_back(textobj->mnId);
        textBoxpred.push_back(textBoxObj);
        textLabelidx.push_back(i0);            // put i0 is easier for subsequent view. (textobj->mnId)
        textinfo.push_back(textobj->TextMean);


        // feature
        vector<TextFeature*> textfeature = textobj->vRefFeature[PyBegin];
        vector<Vec2> textpredObjkf;
        // pyramid -> all features
        int num_feature = 0;
        for(size_t iTextfeat=0; iTextfeat<textfeature.size(); iTextfeat++){
            // observ flag
            vector<Mat31> neighRay = textfeature[iTextfeat]->neighbourRay;
            for(size_t iNeigh = 0; iNeigh<neighRay.size(); iNeigh++){
                Mat31 ray = neighRay[iNeigh];
                double invz = -ray.transpose() * thetaobj;
                Mat31 p = Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3);
                double u = fx * p(0,0)/p(2,0) + cx;
                double v = fy * p(1,0)/p(2,0) + cy;
                 textpredObjkf.push_back(Vec2(u,v));
            }
            num_feature ++;
        }
        textpredkf.push_back(textpredObjkf);
    }
    Mat ImgTextFeat = Tool.ShowText(textpredkf, F.vFrameImg[PyBegin]);

    vector<Mat> ImgTextAll = Tool.TextBoxWithFill(textBoxpred, textmnId, textinfo, ImgTextFeat, textLabelidx);
    TextLabel = ImgTextAll[0];
    Mat ImgText = ImgTextAll[1];

}


// difference with ShowBAReproj: Text is calculated with box; Scene is the same
void optimizer::ShowBAReproj_TextBox(double **rho, double **theta, double **pose, const int &PyBegin, const vector<keyframe*> &vKFs, const vector<int> &vmnId2vKFs, const vector<int> &vmnId2Pts, const vector<int> &vmnId2Texts, cv::Mat &TextLabel, const int op)
{
    double fx = vfx[PyBegin], fy = vfy[PyBegin], cx = vcx[PyBegin], cy = vcy[PyBegin];

    vector<vector<Vec2>> textpredObj;
    for(size_t iKF=0; iKF<vKFs.size(); iKF++){
        Eigen::Quaterniond q(pose[iKF][0], pose[iKF][1], pose[iKF][2], pose[iKF][3]);
        Mat31 t(pose[iKF][4], pose[iKF][5], pose[iKF][6]);
        q = q.normalized();
        Mat33 R(q);
        Mat44 Tcw;
        Tcw.setIdentity();
        Tcw.block<3,3>(0,0) = R;
        Tcw.block<3,1>(0,3) = t;

        cv::Mat ImgShow = vKFs[iKF]->vFrameImg[PyBegin].clone();

        // Text object
        vector<vector<Vec2>> textpredkf;        // text -> 4 pts
        vector<TextObservation*> vText = vKFs[iKF]->GetStateTextObvs(TEXTGOOD);
        vector<int> textLabel, textmnId;
        vector<TextInfo> textinfo;
        for(size_t iText=0; iText<vText.size(); iText++){
            mapText* textobj = vText[iText]->obj;

            int Idx = vmnId2Texts[textobj->mnId];
            int idxref = vmnId2vKFs[textobj->RefKF->mnId];
            Mat44 Tcr;
            if(idxref==-1)
                Tcr = Tcw * textobj->RefKF->mTcw.inverse();
            else{
                assert(idxref>=0);
                Eigen::Quaterniond qr(pose[idxref][0], pose[idxref][1], pose[idxref][2], pose[idxref][3]);
                Mat31 tr(pose[idxref][4], pose[idxref][5], pose[idxref][6]);
                qr = qr.normalized();
                Mat33 Rr(qr);
                Mat44 Trw;
                Trw.setIdentity();
                Trw.block<3,3>(0,0) = Rr;
                Trw.block<3,1>(0,3) = tr;
                Tcr = Tcw * Trw.inverse();
            }

            vector<Vec2> BoxRef = textobj->vTextDeteRay;
            Mat33 Rcr(Tcr.block<3,3>(0,0));
            Mat31 tcr(Tcr.block<3,1>(0,3));
            Mat31 thetaobj =  Mat31(theta[Idx][0],theta[Idx][1],theta[Idx][2]);
            vector<Vec2> textBoxObj;
            for(size_t iBox = 0; iBox<BoxRef.size(); iBox++){
                Mat31 ray = Mat31(BoxRef[iBox](0), BoxRef[iBox](1), 1.0 );
                double invz = - ray.transpose() * thetaobj;
                Mat31 p = Rcr * ray/invz + tcr;
                double u = fx * p(0,0)/p(2,0) + cx;
                double v = fy * p(1,0)/p(2,0) + cy;
                textBoxObj.push_back(Vec2(u,v));
            }
            textpredkf.push_back(textBoxObj);
            textLabel.push_back(iText);
            textmnId.push_back(textobj->mnId);
            textinfo.push_back(textobj->TextMean);
        }

        vector<Mat> ImgTextAll = Tool.TextBoxWithFill(textpredkf, textmnId, textinfo, ImgShow, textLabel);
        Mat TextLabelkf = ImgTextAll[0];
        Mat ImgText = ImgTextAll[1];
        if(iKF==(vKFs.size()-1)){
            TextLabel = TextLabelkf;
        }

    }
}


void optimizer::ShowBAReproj_TextBox(double **rho, double **theta, double **pose, const int &PyBegin, const vector<keyframe*> &vKFs, const vector<int> &vmnId2vKFs, const vector<int> &vmnId2Pts, const vector<int> &vmnId2Texts, vector<cv::Mat> &vTextLabel, const int op)
{
    double fx = vfx[PyBegin], fy = vfy[PyBegin], cx = vcx[PyBegin], cy = vcy[PyBegin];

    vector<vector<Vec2>> textpredObj;
    for(size_t iKF=0; iKF<vKFs.size(); iKF++){
        Eigen::Quaterniond q(pose[iKF][0], pose[iKF][1], pose[iKF][2], pose[iKF][3]);
        Mat31 t(pose[iKF][4], pose[iKF][5], pose[iKF][6]);
        q = q.normalized();
        Mat33 R(q);
        Mat44 Tcw;
        Tcw.setIdentity();
        Tcw.block<3,3>(0,0) = R;
        Tcw.block<3,1>(0,3) = t;

        cv::Mat ImgShow = vKFs[iKF]->vFrameImg[PyBegin].clone();

        // Text object
        vector<vector<Vec2>> textpredkf;        // text -> 4 pts
        vector<TextObservation*> vText = vKFs[iKF]->GetStateTextObvs(TEXTGOOD);
        vector<int> textLabel, textmnId;
        vector<TextInfo> textinfo;
        for(size_t iText=0; iText<vText.size(); iText++){
            mapText* textobj = vText[iText]->obj;

            int Idx = vmnId2Texts[textobj->mnId];
            int idxref = vmnId2vKFs[textobj->RefKF->mnId];
            Mat44 Tcr;
            if(idxref==-1)
                Tcr = Tcw * textobj->RefKF->mTcw.inverse();
            else{
                assert(idxref>=0);
                Eigen::Quaterniond qr(pose[idxref][0], pose[idxref][1], pose[idxref][2], pose[idxref][3]);
                Mat31 tr(pose[idxref][4], pose[idxref][5], pose[idxref][6]);
                qr = qr.normalized();
                Mat33 Rr(qr);
                Mat44 Trw;
                Trw.setIdentity();
                Trw.block<3,3>(0,0) = Rr;
                Trw.block<3,1>(0,3) = tr;
                Tcr = Tcw * Trw.inverse();
            }

            vector<Vec2> BoxRef = textobj->vTextDeteRay;
            Mat33 Rcr(Tcr.block<3,3>(0,0));
            Mat31 tcr(Tcr.block<3,1>(0,3));
            Mat31 thetaobj =  Mat31(theta[Idx][0],theta[Idx][1],theta[Idx][2]);
            vector<Vec2> textBoxObj;
            for(size_t iBox = 0; iBox<BoxRef.size(); iBox++){
                Mat31 ray = Mat31(BoxRef[iBox](0), BoxRef[iBox](1), 1.0 );
                double invz = - ray.transpose() * thetaobj;
                Mat31 p = Rcr * ray/invz + tcr;
                double u = fx * p(0,0)/p(2,0) + cx;
                double v = fy * p(1,0)/p(2,0) + cy;
                textBoxObj.push_back(Vec2(u,v));
            }
            textpredkf.push_back(textBoxObj);
            textLabel.push_back(iText);
            textmnId.push_back(textobj->mnId);
            textinfo.push_back(textobj->TextMean);
        }

        vector<Mat> ImgTextAll = Tool.TextBoxWithFill(textpredkf, textmnId, textinfo, ImgShow, textLabel);
        Mat TextLabelkf = ImgTextAll[0];
        Mat ImgText = ImgTextAll[1];
        if(iKF==(vKFs.size()-1) || iKF==(vKFs.size()-2)){
            vTextLabel.push_back(TextLabelkf);
        }

    }

}


vector<int> optimizer::GetNewIdxForTextState(const vector<TextObservation*> &TextObjs, const TextStatus &Needstate)
{
   vector<int> vIdxState2Raw;
   for(size_t itext=0; itext<TextObjs.size(); itext++){
       if(TextObjs[itext]->obj->STATE==Needstate)
           vIdxState2Raw.push_back(itext);
   }

   return vIdxState2Raw;
}


// ------------- From Tracking -------------

void optimizer::UpdateSemantic_MapObjs_single(mapText* obj, keyframe* KF)
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

double optimizer::GetSgeo(const keyframe* KF, mapText *obj)
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


}
