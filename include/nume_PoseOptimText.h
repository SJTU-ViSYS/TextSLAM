/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef NUME_POSEOPTIMTEXT_H
#define NUME_POSEOPTIMTEXT_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include "ModelTool.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace ceres;

class nume_PoseOptimText{
public:
    nume_PoseOptimText(const cv::Mat &CurImg, const Eigen::Matrix<double,4,4> &mTwr, const Eigen::Matrix<double,3,1> &theta, const vector<Eigen::Matrix<double,3,1>> &vRay, const vector<double> &vInten,
                       const double &Curmu, const double &Cursigma, const Eigen::Matrix<double,3,3> K, const double Scale_T):
        _Img(CurImg), _stride(CurImg.cols), _rows(CurImg.rows), _Twr(mTwr), _theta(theta), _vRefRay(vRay), _vRefNInten(vInten), _mu(Curmu), _sigma(Cursigma), _K(K), SIZE_RESIDUAL(vRay.size() * 1), ScaleText(Scale_T){}

    bool operator()(const double* const _q,
                    const double* const _t,
                    double* residuals) const {

        Eigen::Quaterniond qcw(_q[0], _q[1], _q[2], _q[3]);
        Eigen::Vector3d tcw(_t[0], _t[1], _t[2]);
        qcw = qcw.normalized();
        Eigen::Matrix3d Rcw(qcw);
        Eigen::Matrix4d Tcw, Tcr;
        Tcw.setIdentity();
        Tcw.block<3,3>(0,0) = Rcw;
        Tcw.block<3,1>(0,3) = tcw;
        Tcr = Tcw * _Twr;

        for(size_t ipat = 0; ipat<_vRefRay.size(); ipat++){
            double CurInten;

            // 1. projection
            Eigen::Matrix<double,3,1> p = TextProj(_vRefRay[ipat], Tcr, _theta);
            double u = _K(0,0) * p(0)/p(2) + _K(0,2);
            double v = _K(1,1) * p(1)/p(2) + _K(1,2);

            // 2. intensity
            int uf = floor(u);
            int vf = floor(v);
            int uc = ceil(u);
            int vc = ceil(v);
            if(uf<0 || vf<0 || uc>=_stride || vc>=_rows){
                CurInten = 0;
            }else{
                uint8_t* img_ptr = (uint8_t*)_Img.data + vf*_stride + uf;
                double subpix_u_cur = u-uf;
                double subpix_v_cur = v-vf;
                double w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
                double w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
                double w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
                double w_cur_br = subpix_u_cur * subpix_v_cur;
                CurInten = w_cur_tl * img_ptr[0] + w_cur_tr * img_ptr[1] + w_cur_bl * img_ptr[_stride] + w_cur_br * img_ptr[_stride+1];
            }

            // 3. residual
            if(_sigma!=0){
                double nCurInten = (CurInten - _mu)/_sigma;
                residuals[ipat] = (nCurInten - _vRefNInten[ipat]) * ScaleText;
            }else{
                residuals[ipat] = 0.0;
            }

        }

        return true;
    }

    static ceres::CostFunction *Create(const cv::Mat &CurImg, const Eigen::Matrix<double,4,4> &mTwr, const Eigen::Matrix<double,3,1> &theta, const vector<Eigen::Matrix<double,3,1>> &vRay, const vector<double> &vInten, const double &Curmu, const double &Cursigma, const Eigen::Matrix<double,3,3> K, const double Scale_T) {
        return (new ceres::NumericDiffCostFunction<nume_PoseOptimText, ceres::CENTRAL, 8, 4, 3>(
                new nume_PoseOptimText(CurImg, mTwr, theta, vRay, vInten, Curmu, Cursigma, K, Scale_T)));
    }

private:
    // current image info
    cv::Mat _Img;
    double _mu, _sigma, ScaleText;
    int _stride, _rows;
    Eigen::Matrix<double,4,4> _Twr;
    Eigen::Matrix<double,3,1> _theta;
    vector<Eigen::Matrix<double,3,1>> _vRefRay;
    vector<double> _vRefNInten;

    Eigen::Matrix<double,3,3> _K;
    const int SIZE_RESIDUAL;

};

#endif // NUME_POSEOPTIMTEXT_H
