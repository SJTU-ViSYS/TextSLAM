/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef AUTO_RHOSCENE_H
#define AUTO_RHOSCENE_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include "ModelTool.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace ceres;

class auto_RhoScene{
public:
    auto_RhoScene(Eigen::Matrix<double,2,1> obv, Eigen::Matrix<double,3,1> ray, Eigen::Matrix<double,4,4> Tcr, const Eigen::Matrix<double,3,3> K):
        _u(obv(0)), _v(obv(1)), _Tcr(Tcr), _ray(ray), _fx(K(0,0)), _fy(K(1,1)), _cx(K(0,2)), _cy(K(1,2)), _K(K){}

    template<typename T>
    bool operator()(const T* const _rho,
                    T *residuals) const {

        T ScaleScenex = T( double(1.0) );
        T ScaleSceney = T( double(1.0) );

        T fx = T(_K(0, 0));
        T fy = T(_K(1, 1));
        T cx = T(_K(0, 2));
        T cy = T(_K(1, 2));

        T p[3], p_tmp[3];
        p[0] = T(1.0)/_rho[0] * T(_ray(0,0));
        p[1] = T(1.0)/_rho[0] * T(_ray(1,0));
        p[2] = T(1.0)/_rho[0] * T(_ray(2,0));

        Eigen::Matrix<double,3,3> Rcr = _Tcr.block<3,3>(0,0);
        Eigen::Matrix<double,3,1> tcr = _Tcr.block<3,1>(0,3);
        Eigen::Quaterniond qcr(Rcr);
        qcr = qcr.normalized();
        T q[4];
        q[0] = T(qcr.w());
        q[1] = T(qcr.x());
        q[2] = T(qcr.y());
        q[3] = T(qcr.z());

        ceres::QuaternionRotatePoint(q, p, p_tmp);

        T u[1], v[1];
        u[0] = fx * ( p_tmp[0] + T(tcr(0)) )/(p_tmp[2]+ T(tcr(2)) ) + cx;
        v[0] = fy * ( p_tmp[1] + T(tcr(1)) )/(p_tmp[2]+ T(tcr(2)) ) + cy;

        residuals[0] = (u[0] - _u) * ScaleScenex;
        residuals[1] = (v[0] - _v) * ScaleSceney;

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Matrix<double,2,1> obv, Eigen::Matrix<double,3,1> ray, Eigen::Matrix<double,4,4> Tcr, const Eigen::Matrix<double,3,3> K){
        return (new ceres::AutoDiffCostFunction<auto_RhoScene, 2, 1>(
                    new auto_RhoScene(obv, ray, Tcr, K)));
    }

private:
    double _u, _v;
    Eigen::Matrix<double,3,1> _ray;
    Eigen::Matrix<double,4,4> _Tcr;
    double _fx, _fy, _cx, _cy;
    Eigen::Matrix<double,3,3> _K;
    
};

#endif // AUTO_RHOSCENE_H
