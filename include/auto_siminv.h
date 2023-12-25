/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef AUTO_SIMINV_H
#define AUTO_SIMINV_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include "ModelTool.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace ceres;

class auto_siminv{

public:
    auto_siminv(const Eigen::Matrix<double,3,1> &P1, const Eigen::Matrix<double,2,1> &Obv2, Eigen::Matrix<double,3,3> K):
    _P1(P1),  _u(Obv2(0)), _v(Obv2(1)), _K(K) {}

    template<typename T>
    bool operator()(const T* const _q,
                    const T* const _t,
                    const T* const _s,
                    T *residuals) const {

        T ScaleScenex = T( double(1.0) );
        T ScaleSceney = T( double(1.0) );

        T fx = T(_K(0, 0));
        T fy = T(_K(1, 1));
        T cx = T(_K(0, 2));
        T cy = T(_K(1, 2));

        T P[3], P_tmp[3], P2[3], Pred[2];
        T qinv[4], tinv_tmp[3], tinv[3];
        P[0] = T(_P1(0,0));
        P[1] = T(_P1(1,0));
        P[2] = T(_P1(2,0));
        qinv[0] = _q[0];
        qinv[1] = -_q[1];
        qinv[2] = -_q[2];
        qinv[3] = -_q[3];
        ceres::QuaternionRotatePoint(qinv, P, P_tmp);
        ceres::QuaternionRotatePoint(qinv, _t, tinv_tmp);
        tinv[0] = -tinv_tmp[0]/_s[0];
        tinv[1] = -tinv_tmp[1]/_s[0];
        tinv[2] = -tinv_tmp[2]/_s[0];
        P2[0] = P_tmp[0]/_s[0] + tinv[0];
        P2[1] = P_tmp[1]/_s[0] + tinv[1];
        P2[2] = P_tmp[2]/_s[0] + tinv[2];

        Pred[0] = P2[0]/P2[2]*fx+cx;
        Pred[1] = P2[1]/P2[2]*fy+cy;

        residuals[0] = (Pred[0] - _u) * ScaleScenex;
        residuals[1] = (Pred[1] - _v) * ScaleSceney;

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double,3,1> &P1, const Eigen::Matrix<double,2,1> &Obv2, Eigen::Matrix<double,3,3> K) {
        return (new ceres::AutoDiffCostFunction<auto_siminv, 2, 4, 3, 1>(
                    new auto_siminv(P1, Obv2, K)));
    }

private:
    double _u, _v;
    Eigen::Matrix<double,3,1> _P1;
    Eigen::Matrix<double,3,3> _K;

};

#endif // AUTO_SIMINV_H
