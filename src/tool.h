/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef TOOL_H
#define TOOL_H

#include <string>
#include <thread>
#include "ceres/ceres.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <Random.h>

#include <setting.h>

using namespace std;
using namespace cv;

namespace TextSLAM {

class keyframe;
class frame;
class mapText;

class tool
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // ------------ IO ------------
    void ReadImage(const string &Path, vector<string> &ImgName, vector<string> &ImgIdx, vector<double> &ImgTime);
    void ReadText(const string &Path, vector<vector<Eigen::Matrix<double,2,1>>> &vDetec, vector<TextInfo> &vMean, const int &Flag_Exp, const bool &Flag_noText);
    // tool for text read
    bool Readfulltxt(const string &Path, char* &m_binaryStr, int &m_length);
    size_t get_utf8_char_len(const char & byte);
    // ------------ IO ------------

    // ------------ Semantic ------------
    double LevenshteinDist(const string &str1in, const string &str2in);
    // ------------ Semantic ------------

    // ------------ Param proc ------------
    void GetDeteInfo(const vector<Eigen::Matrix<double, 2,1> > &vDetecRaw, vector<vector<Eigen::Matrix<double,2,1>>> &vDetec);
    // cv::Mat input must be float
    Mat33 cvM2EiM33(const Mat &cvMf);
    Mat31 cvM2EiM31(const Mat &cvMf);
    cv::Mat EiM442cvM(const Mat44 &EiMf);
    cv::Mat EiM332cvMf(const Mat33 &EiMf);
    Mat44 Pose2Mat44(double* pose);
    vector<Point> vV2vP(const vector<Vec2> &In);
    vector<KeyPoint> vV2vK(const vector<Vec2> &In);
    KeyPoint V2K(const Vec2 &In);
    void GetRayRho(const vector<cv::Point2f> &In, const vector<double> &In2, const Mat33 &K, vector<Vec3> &Out);
    Eigen::Quaterniond R2q(const Mat33 &R);
    std::map<keyframe*, int> Convert2Map(const vector<CovKF> &vKFs);
    set<keyframe*> Convert2Set(const vector<CovKF> &vKFs);
    // ------------ Param proc ------------

    // ------------ Feature extractor ------------
    vector<cv::Mat> GetMask(const cv::Mat &Img, const vector<vector<Vec2>> &TextDete);
    cv::Mat GetMask_all(const cv::Mat &Img, const vector<vector<Vec2>> &TextDete, Mat &ImgLabel);
    cv::Mat GetInvMask(const cv::Mat &Img, const vector<vector<Vec2> > &TextDete);
    void BoundFeatDele_T(const Mat &Img, const vector<vector<KeyPoint> > &TextFeat, const vector<Mat> &DescText, const vector<vector<Vec2>> &TextDete, const vector<Vec2> &vTextMin, const vector<Vec2> &vTextMax, const float &Win,
                         vector<vector<cv::KeyPoint>> &TextFeat_out, vector<Mat> &DescText_out, vector<vector<int>> &IdxNew);
    void BoundFeatDele_S(const vector<KeyPoint> &SceneFeat, const Mat &DescScene, const vector<vector<Vec2>> &TextDete, const float &Win,
                                vector<cv::KeyPoint> &SceneFeat_out, cv::Mat &DescScene_out, vector<int> &IdxNew);
    void GetPyramidPts(const vector<KeyPoint> &vObvRaw, const Vec2 &PMin, const Vec2 &PMax, const vector<cv::Mat> &vImg, const vector<cv::Mat> &vImgGrad, const vector<double> &vInvScalefactor, const vector<Mat33> &vK_scale, vector<vector<TextFeature*>> &vObv); // name same, need check
    void GetPyramidPts(const vector<KeyPoint> &vObvRaw, const vector<cv::Mat> &vImg, const vector<Mat> &vImgGrad, const vector<double> &vInvScalefactor, vector<vector<TextFeature *> > &vObv);     // add vImgGrad
    void GetPyramidPts(const vector<Vec2> &vObvRaw, const vector<Mat> &vImg, const vector<cv::Mat> &vImgGrad, const vector<double> &vInvScalefactor, vector<vector<SceneFeature*> > &vObv);
    // ------------ Feature extractor ------------

    // ------------ Basic operator ------------
    vector<vector<Vec2> > GetNewDeteBox(const vector<vector<Vec2> > &DeteRaw, const float &Win);
    vector<vector<Vec2>> GetbbDeteBox(const vector<Vec2> &vTextDeteMin, const vector<Vec2> &vTextDeteMax);
    bool CheckPtsIn(const vector<Point> &hull, const cv::Point2f &pts);
    vector<size_t> InitialVec(const int &N);
    bool GetIntenBilinter(const Vec2 &pts, const Mat &Img, double &Inten);  // no use
    bool GetIntenBilinterPtr(const Vec2 &pts, const Mat &Img, double &Inten);
    void GetNeighbour(TextFeature* feat, const double &mu, const double &std, const Mat &Img, const Mat33 &K, const neighbour &NEIGH);
    bool GetProjText(const Mat31 &ray, const Mat31 &theta, Vec2 &pred, const Mat44 &T, const Mat33 &K);                                     // for ZNCC CHECK, TextFeature*->ray is Mat31   // name same, need check
    bool GetProjText(const Vec2 &ray, const Mat31 &theta, Vec2 &pred, const Mat44 &Tcr, const Mat33 &K);                                    // for tracking proj (text good check)
    void GetProjText(const Vec2 &_ray, const double* thetacr, Vec2 &pred, const double* posecr, const Mat33 &K);                            // for initial BA
    bool GetProjText(const Vec2 &_ray, const double* thetacr, Vec2 &pred, const double* posecw, const double* poserw, const Mat33 &K);      // for local BA
    bool GetProjText(const Vec2 &_ray, const Mat31 thetacr, Vec2 &pred, const double* posecw, const Mat44 &Twr, const Mat33 &K);            // for pose optimization
    void GetProjText(const Vec2 &_ray, const double* thetacr, Vec2 &pred, const Mat44 &Tcr,  const Mat33 &K);                               // for theta optimization
    bool GetPtsText(const Vec2 &_ray, const Mat31 &theta, Mat31 &pw, const Mat44 &Tcr);
    bool CalTextinfo(const cv::Mat &Img, const vector<Vec2> &vTextDete, double &mu, double &std, const bool &SHOW);
    bool CalStatistics(const vector<double> &vTextInten, double &mu, double &std);
    void GetBoxAllPixs(const cv::Mat &Img, const vector<Vec2> &vTextDete, const int &Pym, const Mat33 &K, const double &mu, const double &std, vector<TextFeature*> &vPixs);
    bool CalNormvec(const Mat &Img, vector<TextFeature*> &Vecin, const double &mu, const double &std, const Mat33 &K);
    cv::Point2f PixToCam(const Mat33 &K, const cv::Point2f p);
    bool GetRANSACIdx(const int &MaxIterations, const int &SelectNum, const int &number, const bool &TEXT, vector<vector<size_t>> &IdxOut);
    bool CheckOrientation(const Mat31 &theta, const Mat44 &Tcr, const frame &F, const double &threshcos);      // cos
    bool CheckZNCC(const vector<TextFeature*> &vAllpixs, const cv::Mat &ImgCur, const Mat &ImgRef, const Mat44 &Tcr, const Mat31 &theta, const Mat33 &K, const double &thresh);
    bool VectorNorm(const vector<double> &vIn, vector<double> &vOut);
    double CalZNCC(const vector<double> &v1, const vector<double> &v2);
    Mat31 TransTheta(const Mat31 &Theta_r, const Mat44 &Trw);
    vector<CovKF> GetAllNonZero(const vector<CovKF> &vKFsIn);
    Vec2 GetMean(const vector<Vec2> &Pred);
    // ------------ Basic operator ------------

    //  ------------ Visualization ------------ (need check)
    void ShowMatches(const vector<int> &Match12, const Mat &F1Img, const Mat &F2Img, const vector<KeyPoint> &F1Keys, const vector<KeyPoint> &F2Keys, const bool &SHOW, const bool &SAVE, const string &SaveName);
    void ShowMatches(const vector<int> &Match12, const Mat &F1Img, const Mat &F2Img, const vector<KeyPoint> &F1Keys, const vector<KeyPoint> &F2Keys, const bool &SHOW);
    void ShowMatches(const vector<match> &Match12, const Mat &F1Img, const Mat &F2Img, const vector<KeyPoint> &F1Keys, const vector<KeyPoint> &F2Keys, const bool &SHOW);
    void ShowMatches(const vector<match> &Match12, const Mat &F1Img, const Mat &F2Img, const vector<KeyPoint> &F1Keys, const vector<KeyPoint> &F2Keys, const bool &SHOW, const bool &SAVE, const string &SaveName);
    void ShowMatchesLP(const Mat &F1Img, const Mat &F2Img, const vector<Vec2> &F1Keys, const vector<Vec2> &F2Keys, const bool &SHOW, const bool &SAVE, const string &SaveName);
    void ShowMatchesLP(const Mat &F1Img, const Mat &F2Img, const vector<Vec2> &F1Keys, const vector<Vec2> &F2Keys, const vector<bool> &vbFlag, const bool &SHOW, const bool &SAVE, const string &SaveName);
    void ShowFeature(const vector<size_t> &FeatIdx, const cv::Mat &Img, const vector<cv::KeyPoint> &Keys, const bool &SHOW);
    void ShowFeature(const vector<int> &FeatIdx, const cv::Mat &Img, const vector<cv::KeyPoint> &Keys, const bool &SHOW);
    void ShowFeatures(const cv::Mat &Img, const vector<cv::KeyPoint> &Keys, const bool &SHOW);
    cv::Mat ShowScene(const vector<Vec2> &scenepred, const vector<SceneFeature *> &sceneobv, const Mat Img, const string &savename, const bool &SAVE);
    cv::Mat ShowScene(const vector<Vec2> &scenepred, const vector<Vec2> &sceneobv, const Mat Img, const string &savename, const bool &SAVE);
    cv::Mat ShowText(const vector<vector<Vec2>> &textpred, const Mat Img);
    cv::Mat ShowTextBox(const vector<vector<Vec2>> &textbox, const Mat Img, const string &savename, const bool &SAVE);
    cv::Mat ShowTextBoxWithText(const vector<vector<Vec2>> &textbox, const vector<string> &vShowText, const Mat Img, const string &ShowName);
    cv::Mat ShowTextBoxSingle(const Mat &Img, const vector<Vec2> &Pred, const int &ObjId);
    vector<cv::Mat> TextBoxWithFill(const vector<vector<Vec2>> &textbox, const vector<int> &vObjId, const vector<TextInfo> &textinfo, const Mat Img, const vector<int> &textLabel);
    vector<cv::Mat> TextBoxWithFill(const vector<vector<Vec2>> &textbox, const vector<int> &vObjId, const Mat Img, const vector<int> &textLabel);
    cv::Mat GetTextLabelMask(const cv::Mat &BackImg, const vector<Vec2> &TextObjBox, const int &TextLabel);
    void ShowMatchesLP(const Mat &F1Img, const Mat &F2Img, const vector<FeatureConvert> &F1Keys, const vector<FeatureConvert> &F2Keys, const vector<bool> &vbFlag, const bool &SHOW, const bool &SAVE, const string &SaveName);
    void ShowImg(const cv::Mat &ImgDraw, const string &name);
    //  ------------ Visualization ------------

    //  ------------ Record for debug ------------
    void RecordSim3Optim(const vector<Mat31> &vMapPt1, const vector<Mat31> &vMapPt2, const vector<Vec2> &vObv1, const vector<Vec2> &vObv2, const Mat33 &SimR12, const Mat31 &Simt12, const double &Sims12);
    //  ------------ Record for debug ------------
private:

};

}

#endif // TOOL_H
