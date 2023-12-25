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
#include <system.h>
#include <tracking.h>

namespace TextSLAM
{
system::system(Mat33& K, setting* Setting, const int &numfs)
{
    int scale = std::ceil(Setting->Fps/3);
    int param_M = std::ceil(numfs/scale);

    cout<<" -------- Map basic info -------- "<<endl;
    cout<<"Map use param_M: "<<param_M<<endl;

    Map = new map(param_M);
    LoopClosing = new loopClosing(Setting);
    Tracking = new tracking(K, Map, LoopClosing, Setting);

}

int system::TrackMonocular(const cv::Mat &im, const double &timestamp, Mat33 &CameraK, const vector<vector<Vec2>> &TextDete, const vector<TextInfo> &TextMean)
{
    int flag = Tracking->GrabImageMonocular(im, timestamp, CameraK, TextDete, TextMean);
    if(flag!=0){
        cout<<" *************** ERROR *************** "<<endl;
        return 1;
    }

    return 0;
}

void system::RecordKeyFrame()
{
    string name = "keyframe.txt";
    Tracking->RecordKeyFrameSys(name);
}

}
