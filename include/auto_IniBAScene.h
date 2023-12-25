/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef AUTO_INIBASCENE_H
#define AUTO_INIBASCENE_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include "ModelTool.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace ceres;

class auto_IniBAScene{
public:
    auto_IniBAScene(Eigen::Matrix<double,2,1> obv, Eigen::Matrix<double,3,1> ray, const Eigen::Matrix<double,3,3> K):
        _u(obv(0)), _v(obv(1)), _ray(ray), _fx(K(0,0)), _fy(K(1,1)), _cx(K(0,2)), _cy(K(1,2)), _K(K){}

    template<typename T>
    bool operator()(const T* const _q,
                    const T* const _t,
                    const T* const _rho,
                    T *residuals) const {

        T ScaleScene = T(1.0);

        T fx = T(_K(0, 0));
        T fy = T(_K(1, 1));
        T cx = T(_K(0, 2));
        T cy = T(_K(1, 2));

        T p[3];
        p[0] = T(1.0)/_rho[0] * T(_ray(0,0));
        p[1] = T(1.0)/_rho[0] * T(_ray(1,0));
        p[2] = T(1.0)/_rho[0] * T(_ray(2,0));

        T qp[3];
        ceres::QuaternionRotatePoint(_q, p, qp);

        T u[1], v[1];
        u[0] = fx * (qp[0]+_t[0])/(qp[2]+_t[2]) + cx;
        v[0] = fy * (qp[1]+_t[1])/(qp[2]+_t[2]) + cy;

        residuals[0] = (u[0] - _u) * ScaleScene;
        residuals[1] = (v[0] - _v) * ScaleScene;

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Matrix<double,2,1> obv, Eigen::Matrix<double,3,1> ray, const Eigen::Matrix<double,3,3> K) {
            return (new ceres::AutoDiffCostFunction<auto_IniBAScene, 2, 4, 3, 1>(
                new auto_IniBAScene(obv, ray, K)));
        }

private:
    double _u, _v;
    Eigen::Matrix<double,3,1> _ray;
    double _fx, _fy, _cx, _cy;
    Eigen::Matrix<double,3,3> _K;

};

#endif // AUTO_INIBASCENE_H
