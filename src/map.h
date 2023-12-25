/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef MAP_H
#define MAP_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>
#include <frame.h>
#include <mapPts.h>
#include <mapText.h>
#include <keyframe.h>

using namespace std;
namespace TextSLAM {

class map
{
public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   map(const int &param_M);
   vector<mapPts*> GetAllMapPoints();
   vector<mapPts*> GetAllMapPoints(const bool &FlagRequire);
   vector<mapText*> GetAllMapTexts();
   vector<mapText*> GetAllMapTexts(const TextStatus &state);
   vector<mapText*> GetObvTexts(const TextStatus &state, const vector<keyframe*> &vKFs);
   vector<keyframe*> GetNeighborKF(const frame &F);
   vector<keyframe*> GetNeighborKF(const int &KFmnId, const int &Win);
   vector<keyframe*> GetAllKeyFrame();
   void Addkeyframe(keyframe* pKF);
   void Addtextobjs(mapText *Textobj);
   void Addscenepts(mapPts *Scenepts);
   keyframe* GetKFFromId(const int &KFmnId);
   mapPts* GetPtFromId(const int &PtmnId);
   mapText* GetObjFromId(const int &ObjmnId);

   // info out
   long unsigned int KeyFramesInMap();

   // loop
   void UpdateCovMap_1(keyframe* KF, mapPts *Scenepts);
   void UpdateCovMap_2(keyframe* KF, mapText *TextObj);
   void UpdateCovMap_3(keyframe* KF, mapText *TextObj);

   vector<Eigen::MatrixXd> GetCovMap_All();
   Eigen::MatrixXd GetCovMap(const int &UseM);
   Eigen::MatrixXd GetCovMap_1();
   Eigen::MatrixXd GetCovMap_2();
   Eigen::MatrixXd GetCovMap_3();

   void SetCovMap_1(Eigen::MatrixXd &M1In);
   void SetCovMap_2(Eigen::MatrixXd &M2In);
   void SetCovMap_3(Eigen::MatrixXd &M3In);

   // number info
   int imapPts;  // vMapPoints.size()
   int imapText; // vMapTextObjs.size()
   int imapkfs;  // vKeyframes.size()


private:

   // scene pts
   vector<mapPts*> vMapPoints;

   // text objects
   vector<mapText*> vMapTextObjs;

   // key frame
   vector<keyframe*> vKeyframes;

   // covisibility map
   Eigen::MatrixXd M1;      // scene pts number
   Eigen::MatrixXd M2;      // text obj number
   Eigen::MatrixXd M3;      // text feature number

};

}

#endif // MAP_H
