/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef SETTING_H
#define SETTING_H

#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include<opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

namespace TextSLAM {

class mapText;
class mapPts;
class keyframe;

typedef Eigen::Matrix<double,3,3> Mat33;
typedef Eigen::Matrix<double,4,4> Mat44;
typedef Eigen::Matrix<double,3,1> Mat31;
typedef Eigen::Matrix<double,4,1> Mat41;
typedef Eigen::Matrix<double,1,4> Mat14;
typedef Eigen::Matrix<double,1,3> Mat13;
typedef Eigen::Matrix<double,1,1> Mat11;

typedef Eigen::Matrix<double,3,1> Vec3;
typedef Eigen::Matrix<double,2,1> Vec2;

typedef pair<int,int> match;
typedef pair<int,double> matchRes;
typedef pair<keyframe*,int> CovKF;  // each keyframe and its covisible points number


// for text object using direct method, struct TextFeature store the single featrue with related info (pattern...)
// direct feature
struct TextFeature
{
    // pixel info
    double u,v;
    Vec2 feature;               // feature = Vec2(u,v);
    double featureInten;            // I(feature)
    Mat31 ray;

    // pyramid info
    int level;                     // the extracted pyramid level
    int IdxToRaw;                  // related to IdxToRaw point in the 0-level pyramid

    // norm
    bool INITIAL;       // the text object this feature belongs to is initialized or not?
    bool IN;            // this feature including its neighbour is in or out of img. the neighbour is based the defined neighbour state

    vector<Vec2> neighbour;
    vector<Mat31> neighbourRay;      // Based on neighbour, get ray of neighbour
    vector<double> neighbourInten;

    // inten info
    double featureNInten;
    vector<double> neighbourNInten;  // I_norm(neighbour)

};

struct SceneFeature
{
    // pixel info
    double u,v;
    Vec2 feature;               // feature = Vec2(u,v);

    // pyramid info
    int level;                     // the extracted pyramid level
    int IdxToRaw;                  // related to IdxToRaw point in the 0-level pyramid
};

struct TextObservation
{
    mapText* obj;               // the keyframe observed text
    vector<int> idx;            // the correspoding idx text object. one text object can belong to more than 1 text detection
    double cos;
};
struct SceneObservation
{
    mapPts* pt;               // the keyframe observed map point
    int idx;                    // the correspoding idx feature
};

// semantic ----------
struct TextInfo
{
    string mean;            // the recognition meaning
    double score;           // recognition res score
    double score_semantic;  // S_semantic = S_geo+S_mean (smaller better)
    int lang;               // language: english -- 0; Chinese -- 1; Chinese+english -- 2
};

struct MatchmapTextRes
{
    mapText* mapObj;
    double dist;
    double score;
};

struct cmpLarge
{
    bool operator()(const pair<int,double> &p1,const pair<int,double> &p2)
    {
        return p1.second > p2.second;
    }
};

struct cmpLarge_kf
{
    bool operator()(const pair<keyframe*,int> &p1,const pair<keyframe*,int> &p2)
    {
        return p1.second > p2.second;
    }
};

struct Sim3_loop
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Quaterniond r;
    Eigen::Vector3d t;
    double s;

    Sim3_loop()
    {
      r.setIdentity();
      t.fill(0.);
      s=1.;
    }

    Sim3_loop(const Eigen::Quaterniond & r, const Eigen::Vector3d & t, double s)
      : r(r),t(t),s(s)
    {
    }

    Sim3_loop(const Eigen::Matrix3d & R, const Eigen::Vector3d & t, double s)
      : r(Eigen::Quaterniond(R)),t(t),s(s)
    {
    }

    Sim3_loop inverse() const
    {
      return Sim3_loop(r.conjugate(), r.conjugate()*((-1./s)*t), 1./s);
    }

    Sim3_loop operator *(const Sim3_loop& other) const {
      Sim3_loop ret;
      ret.r = r*other.r;
      ret.t=s*(r*other.t)+t;
      ret.s=s*other.s;
      return ret;
    }

    Eigen::Vector3d map (const Eigen::Vector3d& xyz) const {
     return s*(r*xyz) + t;
   }

};

struct FeatureConvert
{
    // the 3d xyz converted from Text/Scene info
    Mat31 posWorld;  // (world coordinate)
    Mat31 posObv;    // (camera coordinate)

    // if FlagTS is setted to Scene/Text, pt/obj has ptr, the other is null
    int FlagTS;     // 0: Scene; 1: Text
    mapText* obj;
    mapPts* pt;

    keyframe* KF;   // the feature in this KF
    int idx2d;      // coresponding to vKeys
    cv::KeyPoint obv2d;     // KF->vKeys[idx2d]
    Vec2 obv2dPred; // K * posObv/posObv(2)

};


// semantic ----------

enum neighbour{
    INTERVAL8 = 0,  // grid 8
    BLOCK9 = 1,     // 3*3
    BLOCK25 = 2,    // 5*5
};

enum TextStatus{
    TEXTGOOD = 0,
    TEXTIMMATURE = 1,
    TEXTBAD = 2
};

enum BAStatus{
    NOTREACHWIN = 0,    // When there are less frames than window thresh
    LOCAL = 1,
    GLOBAL = 2
};

class setting
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    setting(string& Read_setting);

private:

public:
    string sReadPath;
    string sReadPath_ImgList;

    // intrinsic param & distoration
    Mat33 mK;
    cv::Mat mKcv;
    cv::Mat mDistcv;

    // Image Size
    int Width, Height;
    // Fps
    double Fps;
    // 0: BGR, 1: RGB
    int Flag_RGB;

    // for experiment
    enum eExp_setting{
        GeneralMotion = 0,
        IndoorLoop1 = 1,
        IndoorLoop2 = 2,
        Outdoor = 3
    };
    eExp_setting eExp_name;
    // false: text+scene; true: scene
    bool Flag_noText;

};

}

#endif // SETTING_H
