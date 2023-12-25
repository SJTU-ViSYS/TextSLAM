/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef AUTO_BASCENENW_H
#define AUTO_BASCENENW_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include "ModelTool.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace ceres;

class auto_BASceneNW{
public:
    auto_BASceneNW(Eigen::Matrix<double,2,1> obv, Eigen::Matrix<double,3,1> ray, const Eigen::Matrix<double,3,3> K):
        _u(obv(0)), _v(obv(1)), _ray(ray), _fx(K(0,0)), _fy(K(1,1)), _cx(K(0,2)), _cy(K(1,2)), _K(K) {}

    template<typename T>
    bool operator()(const T* const _qcw,
                    const T* const _tcw,
                    const T* const _qrw,
                    const T* const _trw,
                    const T* const _rho,
                    T *residuals) const {

        T fx = T(_K(0, 0));
        T fy = T(_K(1, 1));
        T cx = T(_K(0, 2));
        T cy = T(_K(1, 2));

        T qwr[4], qcw[4], qcr[4];
        qwr[0] = _qrw[0];
        qwr[1] = -_qrw[1];
        qwr[2] = -_qrw[2];
        qwr[3] = -_qrw[3];
        qcw[0] = _qcw[0];
        qcw[1] = _qcw[1];
        qcw[2] = _qcw[2];
        qcw[3] = _qcw[3];
        qcr[0] = qcw[0]*qwr[0] - qcw[1]*qwr[1] - qcw[2]*qwr[2] - qcw[3]*qwr[3];
        qcr[1] = qcw[0]*qwr[1] + qcw[1]*qwr[0] + qcw[2]*qwr[3] - qcw[3]*qwr[2];
        qcr[2] = qcw[0]*qwr[2] - qcw[1]*qwr[3] + qcw[2]*qwr[0] + qcw[3]*qwr[1];
        qcr[3] = qcw[0]*qwr[3] + qcw[1]*qwr[2] - qcw[2]*qwr[1] + qcw[3]*qwr[0];

        T trw[3], tcw[3],  tcr[3], tcr_tmp[3];
        tcw[0] = _tcw[0];
        tcw[1] = _tcw[1];
        tcw[2] = _tcw[2];
        trw[0] = _trw[0];
        trw[1] = _trw[1];
        trw[2] = _trw[2];

        ceres::QuaternionRotatePoint(qcr, trw, tcr_tmp);

        tcr[0] = -tcr_tmp[0]+tcw[0];
        tcr[1] = -tcr_tmp[1]+tcw[1];
        tcr[2] = -tcr_tmp[2]+tcw[2];

        T p[3];
        p[0] = T(1.0)/_rho[0] * T(_ray(0,0));
        p[1] = T(1.0)/_rho[0] * T(_ray(1,0));
        p[2] = T(1.0)/_rho[0] * T(_ray(2,0));

        T qp[3];
        ceres::QuaternionRotatePoint(qcr, p, qp);

        T u[1], v[1];
        u[0] = fx * (qp[0]+tcr[0])/(qp[2]+tcr[2]) + cx;
        v[0] = fy * (qp[1]+tcr[1])/(qp[2]+tcr[2]) + cy;

        residuals[0] = (u[0] - _u);
        residuals[1] = (v[0] - _v);

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Matrix<double,2,1> obv, Eigen::Matrix<double,3,1> ray, const Eigen::Matrix<double,3,3> K) {
        return (new ceres::AutoDiffCostFunction<auto_BASceneNW, 2, 4, 3, 4, 3, 1>(
                    new auto_BASceneNW(obv, ray, K)));
    }

private:
    double _u, _v;
    Eigen::Matrix<double,3,1> _ray;
    double _fx, _fy, _cx, _cy;
    Eigen::Matrix<double,3,3> _K;

};

#endif // AUTO_BASCENENW_H
