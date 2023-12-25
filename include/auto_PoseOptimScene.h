/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef AUTO_POSEOPTIMSCENE_H
#define AUTO_POSEOPTIMSCENE_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include "ModelTool.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace ceres;

class auto_PoseOptimScene{
public:

    auto_PoseOptimScene(Eigen::Matrix<double,2,1> obv, Eigen::Matrix<double,3,1> rayrho, Eigen::Matrix<double,4,4> Trw, const Eigen::Matrix<double,3,3> K, const double Scale_S_x, const double Scale_S_y):
        _u(obv(0)), _v(obv(1)), _ray(rayrho(0), rayrho(1), 1.0), _rho(rayrho(2)), _Trw(Trw), _fx(K(0,0)), _fy(K(1,1)), _cx(K(0,2)), _cy(K(1,2)), _K(K), ScaleScene_x(Scale_S_x), ScaleScene_y(Scale_S_y){}

    template<typename T>
    bool operator()(const T* const _q,
                    const T* const _t,
                    T *residuals) const {

        T ScaleScenex = T(ScaleScene_x);
        T ScaleSceney = T(ScaleScene_y);

        T fx = T(_K(0, 0));
        T fy = T(_K(1, 1));
        T cx = T(_K(0, 2));
        T cy = T(_K(1, 2));

        Eigen::Quaterniond qrwEigen(_Trw.block<3,3>(0,0));
        qrwEigen = qrwEigen.normalized();
        Eigen::Vector3d trwEigen( _Trw(0,3), _Trw(1,3), _Trw(2,3) );

        T qwr[4], qcw[4], qcr[4];
        qwr[0] = T(qrwEigen.w());
        qwr[1] = -T(qrwEigen.x());
        qwr[2] = -T(qrwEigen.y());
        qwr[3] = -T(qrwEigen.z());
        qcw[0] = _q[0];
        qcw[1] = _q[1];
        qcw[2] = _q[2];
        qcw[3] = _q[3];
        qcr[0] = qcw[0]*qwr[0] - qcw[1]*qwr[1] - qcw[2]*qwr[2] - qcw[3]*qwr[3];
        qcr[1] = qcw[0]*qwr[1] + qcw[1]*qwr[0] + qcw[2]*qwr[3] - qcw[3]*qwr[2];
        qcr[2] = qcw[0]*qwr[2] - qcw[1]*qwr[3] + qcw[2]*qwr[0] + qcw[3]*qwr[1];
        qcr[3] = qcw[0]*qwr[3] + qcw[1]*qwr[2] - qcw[2]*qwr[1] + qcw[3]*qwr[0];

        T trw[3], tcw[3],  tcr[3], tcr_tmp[3];
        tcw[0] = _t[0];
        tcw[1] = _t[1];
        tcw[2] = _t[2];
        trw[0] = T(trwEigen(0,0));
        trw[1] = T(trwEigen(1,0));
        trw[2] = T(trwEigen(2,0));

        ceres::QuaternionRotatePoint(qcr, trw, tcr_tmp);

        tcr[0] = -tcr_tmp[0]+tcw[0];
        tcr[1] = -tcr_tmp[1]+tcw[1];
        tcr[2] = -tcr_tmp[2]+tcw[2];

        T p[3];
        p[0] = T(1.0)/T(_rho) * T(_ray(0,0));
        p[1] = T(1.0)/T(_rho) * T(_ray(1,0));
        p[2] = T(1.0)/T(_rho) * T(_ray(2,0));

        T qp[3];
        ceres::QuaternionRotatePoint(qcr, p, qp);

        T u[1], v[1];
        u[0] = fx * (qp[0]+tcr[0])/(qp[2]+tcr[2]) + cx;
        v[0] = fy * (qp[1]+tcr[1])/(qp[2]+tcr[2]) + cy;

        residuals[0] = (u[0] - _u) * ScaleScenex;
        residuals[1] = (v[0] - _v) * ScaleSceney;
        return true;
    }

    static ceres::CostFunction *Create(Eigen::Matrix<double,2,1> obv, Eigen::Matrix<double,3,1> rayrho, Eigen::Matrix<double,4,4> Trw, const Eigen::Matrix<double,3,3> K, const double Scale_S_x, const double Scale_S_y) {
        return (new ceres::AutoDiffCostFunction<auto_PoseOptimScene, 2, 4, 3>(
                    new auto_PoseOptimScene(obv, rayrho, Trw, K, Scale_S_x, Scale_S_y)));
    }

private:
    double _u, _v, ScaleScene_x, ScaleScene_y;
    Eigen::Matrix<double,3,1> _ray;
    double _rho;
    double _fx, _fy, _cx, _cy;
    Eigen::Matrix<double,3,3> _K;
    Eigen::Matrix<double,4,4> _Trw;

};

#endif // AUTO_POSEOPTIMSCENE_H
