/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef AUTO_SIM_H
#define AUTO_SIM_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include "ModelTool.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace ceres;

class auto_sim{

public:
    auto_sim(const Eigen::Matrix<double,3,1> &P2, const Eigen::Matrix<double,2,1> &Obv1, Eigen::Matrix<double,3,3> K):
    _P2(P2),  _u(Obv1(0)), _v(Obv1(1)), _K(K) {}

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

        T P[3], P_tmp[3], P1[3], Pred[2];
        P[0] = T(_P2(0,0));
        P[1] = T(_P2(1,0));
        P[2] = T(_P2(2,0));
        ceres::QuaternionRotatePoint(_q, P, P_tmp);
        P1[0] = _s[0]*P_tmp[0] + _t[0];
        P1[1] = _s[0]*P_tmp[1] + _t[1];
        P1[2] = _s[0]*P_tmp[2] + _t[2];

        Pred[0] = P1[0]/P1[2]*fx+cx;
        Pred[1] = P1[1]/P1[2]*fy+cy;

        residuals[0] = (Pred[0] - _u) * ScaleScenex;
        residuals[1] = (Pred[1] - _v) * ScaleSceney;

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Matrix<double,3,1> &P2, const Eigen::Matrix<double,2,1> &Obv1, Eigen::Matrix<double,3,3> K) {
        return (new ceres::AutoDiffCostFunction<auto_sim, 2, 4, 3, 1>(
                    new auto_sim(P2, Obv1, K)));
    }

private:
    double _u, _v;
    Eigen::Matrix<double,3,1> _P2;
    Eigen::Matrix<double,3,3> _K;

};

#endif // AUTO_SIM_H
