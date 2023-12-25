/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef NUME_INIBATEXT_H
#define NUME_INIBATEXT_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include "ModelTool.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace ceres;

class nume_IniBAText{
public:
    nume_IniBAText(const cv::Mat &CurImg, const vector<Eigen::Matrix<double,3,1>> &vRay, const vector<double> &vInten,
                   const double &Curmu, const double &Cursigma, const Eigen::Matrix<double,3,3> K): _Img(CurImg), _stride(CurImg.cols), _rows(CurImg.rows), _vRefRay(vRay), _vRefNInten(vInten), _mu(Curmu), _sigma(Cursigma),
     _K(K), SIZE_RESIDUAL(vRay.size() * 1){}

    bool operator()(const double* const _q,
                    const double* const _t,
                    const double* const _theta,
                    double* residuals) const {

        double ScaleText = 1.0;

        Eigen::Quaterniond q(_q[0], _q[1], _q[2], _q[3]);
        Eigen::Vector3d t(_t[0], _t[1], _t[2]);
        q = q.normalized();
        Eigen::Matrix3d R(q);
        Eigen::Matrix4d T;
        T.setIdentity();
        T.block<3,3>(0,0) = R;
        T.block<3,1>(0,3) = t;

        Eigen::Matrix<double,3,1> theta;
        theta(0,0) = _theta[0];
        theta(1,0) = _theta[1];
        theta(2,0) = _theta[2];

        for(size_t ipat = 0; ipat<_vRefRay.size(); ipat++){
            double CurInten;

            // 1. projection
            Eigen::Matrix<double,3,1> p = TextProj(_vRefRay[ipat], T, theta);
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
                double vCurInten = (CurInten - _mu)/_sigma;
                residuals[ipat] = (vCurInten - _vRefNInten[ipat]) * ScaleText;
            }else{
                residuals[ipat] = 0.0;
            }

        }

        return true;
    }

    static ceres::CostFunction *Create(const cv::Mat &CurImg, const vector<Eigen::Matrix<double,3,1>> &vRay, const vector<double> &vInten, const double &Curmu, const double &Cursigma, const Eigen::Matrix<double,3,3> K) {
        return (new ceres::NumericDiffCostFunction<nume_IniBAText, ceres::CENTRAL, 8, 4, 3, 3>(
                new nume_IniBAText(CurImg, vRay, vInten, Curmu, Cursigma, K)));
    }

private:
    cv::Mat _Img;
    double _mu, _sigma;
    int _stride, _rows;
    vector<Eigen::Matrix<double,3,1>> _vRefRay;
    vector<double> _vRefNInten;

    Eigen::Matrix<double,3,3> _K;
    const int SIZE_RESIDUAL;

};


#endif // NUME_INIBATEXT_H
