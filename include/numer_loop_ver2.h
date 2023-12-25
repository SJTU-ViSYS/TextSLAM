/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef NUMER_LOOP_VER2_H
#define NUMER_LOOP_VER2_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include "ModelTool.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace ceres;

class numer_loop_ver2{

public:
    numer_loop_ver2(const Eigen::Quaterniond &simq21, const Eigen::Vector3d &simt21, const double &sims21):
        _simq21(simq21), _simt21(simt21), _sims21(sims21){}

    bool operator()(const double* const _q1,
                    const double* const _t1,
                    const double* const _s1,
                    const double* const _q2,
                    const double* const _t2,
                    const double* const _s2,
                    double* residuals) const {

        Eigen::Quaterniond q1w(_q1[0], _q1[1], _q1[2], _q1[3]);
        Eigen::Vector3d t1w(_t1[0], _t1[1], _t1[2]);
        q1w = q1w.normalized();

        Eigen::Quaterniond q2w(_q2[0], _q2[1], _q2[2], _q2[3]);
        Eigen::Vector3d t2w(_t2[0], _t2[1], _t2[2]);
        q2w = q2w.normalized();
        Eigen::Quaterniond qw2 = q2w.conjugate();
        Eigen::Vector3d tw2 = qw2 * ((-1./_s2[0])*t2w);
        double sw2 = 1./_s2[0];

        Eigen::Quaterniond q12 = q1w * qw2;
        Eigen::Vector3d t12 = _s1[0] * (q1w*tw2) + t1w;
        double s12 = _s1[0] * sw2;

        Eigen::Quaterniond res_q = _simq21*q12;
        Eigen::Vector3d res_t = _sims21*(_simq21*t12)+_simt21;
        double res_s = _sims21 * s12;

        Eigen::Matrix <double, 7, 1> res = logSim3(res_q, res_t, res_s);
        
        residuals[0] = res(0,0);
        residuals[1] = res(1,0);
        residuals[2] = res(2,0);
        residuals[3] = res(3,0);
        residuals[4] = res(4,0);
        residuals[5] = res(5,0);
        residuals[6] = res(6,0);

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Quaterniond &simq21, const Eigen::Vector3d &simt21, const double &sims21) {
        return (new ceres::NumericDiffCostFunction<numer_loop_ver2, ceres::CENTRAL, 7, 4, 3, 1, 4, 3, 1>(
                new numer_loop_ver2(simq21, simt21, sims21)));
    }


private:
    Eigen::Quaterniond _simq21;
    Eigen::Vector3d _simt21;
    double _sims21;

};

#endif // NUMER_LOOP_VER2_H
