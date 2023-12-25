/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>

#include <setting.h>
#include <tracking.h>
using namespace std;

namespace TextSLAM {
class system
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    system(Mat33& K, setting *Setting, const int &numfs);
    int TrackMonocular(const cv::Mat &im, const double &timestamp, Mat33 &CameraK, const vector<vector<Vec2> > &TextDete, const vector<TextInfo> &TextMean);
    void RecordKeyFrame();

private:
    map* Map;
    tracking* Tracking;
    loopClosing* LoopClosing;

};

}

#endif // SYSTEM_H
