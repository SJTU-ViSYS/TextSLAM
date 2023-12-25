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
#include <setting.h>

namespace TextSLAM
{
setting::setting(string& Read_setting)
{
    bool FLAG_LOG = true;

    // read setting file
    cv::FileStorage Settings(Read_setting.c_str(), cv::FileStorage::READ);
    if(!Settings.isOpened())
    {
       cerr << "Failed to open settings file at: " << Read_setting << endl;
       exit(-1);
    }

    // experiment info
    int Read_exp_name = Settings["Exp name"];
    switch (Read_exp_name) {
    case 0:
        eExp_name = GeneralMotion;
        break;
    case 1:
        eExp_name = IndoorLoop1;
        break;
    case 2:
        eExp_name = IndoorLoop2;
        break;
    case 3:
        eExp_name = Outdoor;
        break;
    }

    int readFlag_noText = Settings["Exp noText"];
    switch (readFlag_noText) {
    case 0:
        Flag_noText = false;
        break;
    case 1:
        Flag_noText = true;
        break;
    }

    string Imagelist_name;
    Settings["Exp read path"] >> sReadPath;
    Settings["Exp read list"] >> Imagelist_name;
    sReadPath_ImgList = sReadPath+Imagelist_name+".txt";

    // intrinsic info
    double fx = Settings["Camera.fx"];
    double fy = Settings["Camera.fy"];
    double cx = Settings["Camera.cx"];
    double cy = Settings["Camera.cy"];
    mK << fx, 0.0, cx,
          0.0, fy, cy,
          0.0, 0.0, 1.0;
    mKcv = cv::Mat::eye(3,3,CV_64F);
    mKcv.at<double>(0,0) = fx;
    mKcv.at<double>(0,2) = cx;
    mKcv.at<double>(1,1) = fy;
    mKcv.at<double>(1,2) = cy;

    // distoration
    mDistcv = cv::Mat::zeros(5,1,CV_64F);
    mDistcv.at<double>(0) = Settings["Camera.k1"];
    mDistcv.at<double>(1) = Settings["Camera.k2"];
    mDistcv.at<double>(2) = Settings["Camera.p1"];
    mDistcv.at<double>(3) = Settings["Camera.p2"];
    mDistcv.at<double>(4) = Settings["Camera.k3"];

    Width = (int)Settings["Camera.width"];
    Height = (int)Settings["Camera.height"];
    Fps = Settings["Camera.fps"];
    Flag_RGB = (int)Settings["Camera.RGB"];

    if(FLAG_LOG){
        cout<<"********* setting *********"<<endl;
        cout<<"Experiment name: "<<eExp_name<<endl;
        cout<<"Read image from: "<<sReadPath_ImgList<<endl;
        cout<<"K is: "<<mK(0,0)<<", "<<mK(0,2)<<", "<<mK(1,1)<<", "<<mK(1,2)<<endl;
        cout<<"Dist is: "<<mDistcv.at<double>(0)<<", "<<mDistcv.at<double>(1)<<", "<<mDistcv.at<double>(2)<<", "<<mDistcv.at<double>(3)<<", "<<mDistcv.at<double>(4)<<endl;
    }
}

}
