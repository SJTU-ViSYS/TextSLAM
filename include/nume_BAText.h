/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef NUME_BATEXT_H
#define NUME_BATEXT_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include "ModelTool.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace ceres;

class nume_BAText{
public:
    nume_BAText(const cv::Mat &CurImg, const vector<Eigen::Matrix<double,3,1>> &vRay, const vector<double> &vInten, const double &Curmu, const double &Cursigma,
                const Eigen::Matrix<double,3,3> K, const double Scale_T): _Img(CurImg), _stride(CurImg.cols), _rows(CurImg.rows), _vRefRay(vRay), _vRefNInten(vInten), _mu(Curmu), _sigma(Cursigma),
  _K(K), SIZE_RESIDUAL(vRay.size() * 1), ScaleText(Scale_T){}

    bool operator()(const double* const _qcw,
                    const double* const _tcw,
                    const double* const _qrw,
                    const double* const _trw,
                    const double* const _theta,
                    double* residuals) const {

        Eigen::Matrix4d Tcw, Trw, Tcr;
        Eigen::Quaterniond qcw(_qcw[0], _qcw[1], _qcw[2], _qcw[3]);
        Eigen::Vector3d tcw(_tcw[0], _tcw[1], _tcw[2]);
        qcw = qcw.normalized();
        Eigen::Matrix3d Rcw(qcw);
        Tcw.setIdentity();
        Tcw.block<3,3>(0,0) = Rcw;
        Tcw.block<3,1>(0,3) = tcw;

        Eigen::Quaterniond qrw(_qrw[0], _qrw[1], _qrw[2], _qrw[3]);
        Eigen::Vector3d trw(_trw[0], _trw[1], _trw[2]);
        qrw = qrw.normalized();
        Eigen::Matrix3d Rrw(qrw);
        Trw.setIdentity();
        Trw.block<3,3>(0,0) = Rrw;
        Trw.block<3,1>(0,3) = trw;
        Tcr = Tcw * Trw.inverse();

        Eigen::Matrix<double,3,1> theta;
        theta(0,0) = _theta[0];
        theta(1,0) = _theta[1];
        theta(2,0) = _theta[2];

        for(size_t ipat = 0; ipat<_vRefRay.size(); ipat++){
            double CurInten;

            // 1. projection
            Eigen::Matrix<double,3,1> p = TextProj(_vRefRay[ipat], Tcr, theta);
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


    static ceres::CostFunction *Create(const cv::Mat &CurImg, const vector<Eigen::Matrix<double,3,1>> &vRay, const vector<double> &vInten, const double &Curmu, const double &Cursigma, const Eigen::Matrix<double,3,3> K, const double Scale_T) {
        return (new ceres::NumericDiffCostFunction<nume_BAText, ceres::CENTRAL, 8, 4, 3, 4, 3, 3>(
                new nume_BAText(CurImg, vRay, vInten, Curmu, Cursigma, K, Scale_T)));
    }

private:
    // current image info
    cv::Mat _Img;
    double _mu, _sigma, ScaleText;
    int _stride, _rows;
    // reference feature info
    vector<Eigen::Matrix<double,3,1>> _vRefRay;
    vector<double> _vRefNInten;

    Eigen::Matrix<double,3,3> _K;
    const int SIZE_RESIDUAL;

};

#endif // NUME_BATEXT_H
