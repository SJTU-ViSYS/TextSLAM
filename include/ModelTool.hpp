/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#ifndef MODELTOOL_HPP_
#define MODELTOOL_HPP_
#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"
#include <opencv2/core.hpp>
#include <math.h>
using namespace std;
using namespace ceres;

inline double GetQPerturb(const Eigen::Quaterniond &q, Eigen::Quaterniond &plus_delta_qx, Eigen::Quaterniond &plus_delta_qy, Eigen::Quaterniond &plus_delta_qz)
{
    double q_elem[] = {std::abs(q.x()), std::abs(q.y()), std::abs(q.z())};
    double q_max = *std::max_element(q_elem, q_elem+3);
    double delta_q = std::max(1e-6, (1e-4)*q_max );

    Eigen::Quaterniond delta_qx(1.0, delta_q, 0.0, 0.0);
    Eigen::Quaterniond delta_qy(1.0, 0.0, delta_q, 0.0);
    Eigen::Quaterniond delta_qz(1.0, 0.0, 0.0, delta_q);
    plus_delta_qx = delta_qx * q;
    plus_delta_qy = delta_qy * q;
    plus_delta_qz = delta_qz * q;
    plus_delta_qx = plus_delta_qx.normalized();
    plus_delta_qy = plus_delta_qy.normalized();
    plus_delta_qz = plus_delta_qz.normalized();

    return delta_q;
}

inline double GetQPerturb_axis(const Eigen::Quaterniond &q, Eigen::Quaterniond &plus_delta_qx, Eigen::Quaterniond &plus_delta_qy, Eigen::Quaterniond &plus_delta_qz)
{
    double theta = 2*acos(q.w());
    Eigen::Matrix<double,3,1> qxyz(q.x(), q.y(), q.z());
    Eigen::Matrix<double,3,1> u = qxyz/sin(theta/2.0);
    Eigen::Matrix<double,3,1> AA = theta*u;

    double q_elem[] = {std::abs(AA(0)), std::abs(AA(1)), std::abs(AA(2))};
    double q_max = *std::max_element(q_elem, q_elem+3);
    double delta_q = std::max(1e-6, (1e-4)*q_max );

    Eigen::Quaterniond delta_qx(1.0, delta_q/2.0, 0.0, 0.0);
    Eigen::Quaterniond delta_qy(1.0, 0.0, delta_q/2.0, 0.0);
    Eigen::Quaterniond delta_qz(1.0, 0.0, 0.0, delta_q/2.0);
    plus_delta_qx = delta_qx * q;
    plus_delta_qy = delta_qy * q;
    plus_delta_qz = delta_qz * q;
    plus_delta_qx = plus_delta_qx.normalized();
    plus_delta_qy = plus_delta_qy.normalized();
    plus_delta_qz = plus_delta_qz.normalized();

    return delta_q;
}

inline double GettPerturb(const Eigen::Vector3d &t, Eigen::Vector3d &plus_delta_tx, Eigen::Vector3d &plus_delta_ty, Eigen::Vector3d &plus_delta_tz)
{
    double t_elem[] = { std::abs(t(0)),std::abs(t(1)),std::abs(t(2)) };
    double t_max = *std::max_element(t_elem, t_elem+3);
    double delta_t = std::max(1e-6, (1e-4)*t_max );

    Eigen::Vector3d delta_tx(delta_t, 0.0, 0.0);
    Eigen::Vector3d delta_ty(0.0, delta_t, 0.0);
    Eigen::Vector3d delta_tz(0.0, 0.0, delta_t);
    plus_delta_tx = t + delta_tx;
    plus_delta_ty = t + delta_ty;
    plus_delta_tz = t + delta_tz;
    return delta_t;
}

inline double GetrhoPerturb(const double &rho, double &plus_delta_rho)
{
    double delta_rho = std::max(1e-6, (1e-4) * std::abs(rho) );
    plus_delta_rho = rho + delta_rho;
    return delta_rho;
}

inline double GetthetaPerturb(const Eigen::Vector3d &theta, Eigen::Vector3d &plus_delta_thetax, Eigen::Vector3d &plus_delta_thetay, Eigen::Vector3d &plus_delta_thetaz)
{
    double theta_elem[] = { std::abs(theta(0)),std::abs(theta(1)),std::abs(theta(2)) };
    double theta_elem_max = *std::max_element(theta_elem, theta_elem+3);
    double delta_theta = std::max(1e-6, (1e-4)*theta_elem_max );

    Eigen::Vector3d delta_thetax(delta_theta, 0.0, 0.0);
    Eigen::Vector3d delta_thetay(0.0, delta_theta, 0.0);
    Eigen::Vector3d delta_thetaz(0.0, 0.0, delta_theta);
    plus_delta_thetax = theta + delta_thetax;
    plus_delta_thetay = theta + delta_thetay;
    plus_delta_thetaz = theta + delta_thetaz;

    return delta_theta;
}

inline void GetTFromPerturbQcw(const Eigen::Quaterniond &plus_delta_qcw, const Eigen::Matrix4d &Tcw, const Eigen::Matrix4d &Trw, Eigen::Matrix4d &Tcr)
{
    Eigen::Matrix4d Tcw_pert;
    Tcw_pert.setIdentity();

    Eigen::Quaterniond q_pert = plus_delta_qcw;
    q_pert = q_pert.normalized();
    Eigen::Matrix3d Rcw_pert(q_pert);
    Tcw_pert.block<3,3>(0,0) = Rcw_pert;
    Tcw_pert.block<3,1>(0,3) = Tcw.block<3,1>(0,3);

    Tcr = Tcw_pert * Trw.inverse();
}

inline void GetTFromPerturbtcw(const Eigen::Vector3d &plus_delta_tcw, const Eigen::Matrix4d &Tcw, const Eigen::Matrix4d &Trw, Eigen::Matrix4d &Tcr)
{
    Eigen::Matrix4d Tcw_pert;
    Tcw_pert.setIdentity();

    Tcw_pert.block<3,3>(0,0) = Tcw.block<3,3>(0,0);
    Tcw_pert.block<3,1>(0,3) = plus_delta_tcw;

    Tcr = Tcw_pert * Trw.inverse();
}

inline void GetTFromPerturbQrw(const Eigen::Quaterniond &plus_delta_qrw, const Eigen::Matrix4d &Tcw, const Eigen::Matrix4d &Trw, Eigen::Matrix4d &Tcr)
{
    Eigen::Matrix4d Trw_pert;
    Trw_pert.setIdentity();

    Eigen::Quaterniond q_pert = plus_delta_qrw;
    q_pert = q_pert.normalized();
    Eigen::Matrix3d Rrw_pert(q_pert);
    Trw_pert.block<3,3>(0,0) = Rrw_pert;
    Trw_pert.block<3,1>(0,3) = Trw.block<3,1>(0,3);

    Tcr = Tcw * Trw_pert.inverse();
}

inline void GetTFromPerturbtrw(const Eigen::Vector3d &plus_delta_trw, const Eigen::Matrix4d &Tcw, const Eigen::Matrix4d &Trw, Eigen::Matrix4d &Tcr)
{
    Eigen::Matrix4d Trw_pert;
    Trw_pert.setIdentity();

    Trw_pert.block<3,3>(0,0) = Trw.block<3,3>(0,0);
    Trw_pert.block<3,1>(0,3) = Trw.block<3,1>(0,3) + plus_delta_trw;

    Tcr = Tcw * Trw_pert.inverse();
}


inline Eigen::Matrix<double,2,1> TextProj(const Eigen::Matrix<double,3,1> &ray, const Eigen::Matrix4d &Tcr, const Eigen::Matrix<double,3,1> &theta, const Eigen::Matrix<double,3,3> &K)
{
    Eigen::Matrix<double,2,1> pred;
    double rho = -ray.transpose() * theta;
    Eigen::Matrix<double,3,1> p = Tcr.block<3,3>(0,0) * ray/rho + Tcr.block<3,1>(0,3);
    pred(0) = K(0,0) * p(0)/p(2) + K(0,2);
    pred(1) = K(1,1) * p(1)/p(2) + K(1,2);
    return pred;
}


inline Eigen::Matrix<double,3,1> TextProj(const Eigen::Matrix<double,3,1> &ray, const Eigen::Matrix4d &Tcr, const Eigen::Matrix<double,3,1> &theta)
{
    Eigen::Matrix<double,2,1> pred;
    double rho = -ray.transpose() * theta;
    Eigen::Matrix<double,3,1> p = Tcr.block<3,3>(0,0) * ray/rho + Tcr.block<3,1>(0,3);

    return p;
}

inline vector<Eigen::Matrix<double,2,1>> TextProj3T(const Eigen::Matrix<double,3,1> &ray, const vector<Eigen::Matrix4d> &Tcr, const Eigen::Matrix<double,3,1> &theta, const Eigen::Matrix<double,3,3> &K)
{
    double rho = -ray.transpose() * theta;
    vector<Eigen::Matrix<double,2,1>> pPred;
    pPred.resize(Tcr.size());
    for(size_t i=0; i<Tcr.size(); i++){
        Eigen::Matrix<double,3,1> p = Tcr[0].block<3,3>(0,0) * ray/rho + Tcr[0].block<3,1>(0,3);
        double u = K(0,0) * p(0)/p(2) + K(0,2);
        double v = K(1,1) * p(1)/p(2) + K(1,2);
        pPred[i] = Eigen::Matrix<double,2,1>(u,v);
    }

    return pPred;
}

inline vector<Eigen::Matrix<double,2,1>> PtsProj3T(Eigen::Matrix<double,3,1> ray, double rho, const vector<Eigen::Matrix4d> &Tcr, const Eigen::Matrix<double,3,3> &K)
{
    vector<Eigen::Matrix<double,2,1>> vPred;
    vPred.resize(Tcr.size());
    Eigen::Matrix<double,3,1> p3dHost = ray / rho;
    for(size_t i=0; i<Tcr.size(); i++){
        Eigen::Matrix<double,3,1> p = Tcr[i].block<3,3>(0,0) * p3dHost + Tcr[i].block<3,1>(0,3);
        double upred = K(0,0) * p(0)/p(2) + K(0,2);
        double vpred = K(1,1) * p(1)/p(2) + K(1,2);
        vPred[i] = Eigen::Matrix<double,2,1>(upred, vpred);
    }

    return vPred;
}

inline Eigen::Matrix<double,2,1> PtsProj(Eigen::Matrix<double,3,1> ray, double rho, const Eigen::Matrix4d &Tcr, const Eigen::Matrix<double,3,3> &K)
{
    Eigen::Matrix<double,2,1> vPred;
    Eigen::Matrix<double,3,1> p3dHost = ray / rho;

    Eigen::Matrix<double,3,1> p = Tcr.block<3,3>(0,0) * p3dHost + Tcr.block<3,1>(0,3);
    double upred = K(0,0) * p(0)/p(2) + K(0,2);
    double vpred = K(1,1) * p(1)/p(2) + K(1,2);
    vPred = Eigen::Matrix<double,2,1>(upred, vpred);

    return vPred;
}


inline vector<Eigen::Matrix<double,2,1>> TextProj3theta(const Eigen::Matrix<double,3,1> &ray, const Eigen::Matrix4d &Tcr, const vector<Eigen::Matrix<double,3,1>> &theta, const Eigen::Matrix<double,3,3> &K)
{
    vector<Eigen::Matrix<double,2,1>> pPred;
    pPred.resize(Tcr.size());
    for(size_t i=0; i<theta.size(); i++){
        double rho = -ray.transpose() * theta[i];
        Eigen::Matrix<double,3,1> p = Tcr.block<3,3>(0,0) * ray/rho + Tcr.block<3,1>(0,3);
        double u = K(0,0) * p(0)/p(2) + K(0,2);
        double v = K(1,1) * p(1)/p(2) + K(1,2);
        pPred[i] = Eigen::Matrix<double,2,1>(u,v);
    }

    return pPred;
}


inline bool GetIntenBilinter(const Eigen::Matrix<double,2,1> &pts, const cv::Mat &Img, double &Inten)
{
    cv::Mat Image = Img.clone();
    int pix_cur_x_i = floor(pts(0));
    int pix_cur_y_i = floor(pts(1));
    int pix_cur_x_i_c = ceil(pts(0));
    int pix_cur_y_i_c = ceil(pts(1));

    if(pix_cur_x_i < 0 || pix_cur_y_i < 0 || pix_cur_x_i_c >= Image.cols || pix_cur_y_i_c >= Image.rows){
        Inten = double(0.0);
        return false;
    }else{
        double subpix_u_cur = pts(0)-pix_cur_x_i;
        double subpix_v_cur = pts(1)-pix_cur_y_i;
        double w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
        double w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
        double w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
        double w_cur_br = subpix_u_cur * subpix_v_cur;

        double img_cur_11 = double(Image.row(pix_cur_y_i).col(pix_cur_x_i).at<uint8_t>(0,0));
        double img_cur_21 = double(Image.row(pix_cur_y_i).col(pix_cur_x_i_c).at<uint8_t>(0,0));
        double img_cur_12 = double(Image.row(pix_cur_y_i_c).col(pix_cur_x_i).at<uint8_t>(0,0));
        double img_cur_22 = double(Image.row(pix_cur_y_i_c).col(pix_cur_x_i_c).at<uint8_t>(0,0));
        Inten = w_cur_tl * img_cur_11 + w_cur_tr * img_cur_21 + w_cur_bl * img_cur_12 + w_cur_br * img_cur_22;
        return true;
    }

}


inline bool GetIntenBilinterPtr1(const Eigen::Matrix<double,2,1> &pts, const cv::Mat &Img, double &Inten)
{
    cv::Mat Image = Img.clone();

    int pix_cur_x_i = floor(pts(0));
    int pix_cur_y_i = floor(pts(1));
    int pix_cur_x_i_c = ceil(pts(0));
    int pix_cur_y_i_c = ceil(pts(1));

    if(pix_cur_x_i < 0 || pix_cur_y_i < 0 || pix_cur_x_i_c >= Image.cols || pix_cur_y_i_c >= Image.rows){
        Inten = double(0.0);
        return false;
    }else{
        double subpix_u_cur = pts(0)-pix_cur_x_i;
        double subpix_v_cur = pts(1)-pix_cur_y_i;
        double w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
        double w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
        double w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
        double w_cur_br = subpix_u_cur * subpix_v_cur;

        uint8_t* Mi1 = Image.ptr<uint8_t>(pix_cur_y_i);
        double img_cur_11 = double(Mi1[pix_cur_x_i]);
        double img_cur_21 = double(Mi1[pix_cur_x_i_c]);

        uint8_t* Mi2 = Image.ptr<uint8_t>(pix_cur_y_i_c);
        double img_cur_12 = double(Mi2[pix_cur_x_i]);
        double img_cur_22 = double(Mi2[pix_cur_x_i_c]);

        Inten = w_cur_tl * img_cur_11 + w_cur_tr * img_cur_21 + w_cur_bl * img_cur_12 + w_cur_br * img_cur_22;

        return true;
    }

}


inline bool GetIntenGradBilinterPtr(const Eigen::Matrix<double,2,1> &pts, const cv::Mat &Img, double &Inten, double &dx, double &dy)
{
    const int stride = Img.cols;

    int pix_cur_x_i = floor(pts(0));
    int pix_cur_y_i = floor(pts(1));
    int pix_cur_x_i_c = ceil(pts(0));
    int pix_cur_y_i_c = ceil(pts(1));

    if(pix_cur_x_i < 0 || pix_cur_y_i < 0 || pix_cur_x_i_c >= Img.cols || pix_cur_y_i_c >= Img.rows){
        Inten = double(0.0);
        return false;
    }else{
        double subpix_u_cur = pts(0)-pix_cur_x_i;
        double subpix_v_cur = pts(1)-pix_cur_y_i;
        double w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
        double w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
        double w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
        double w_cur_br = subpix_u_cur * subpix_v_cur;

        uint8_t* img_ptr = (uint8_t*)Img.data + pix_cur_y_i*Img.cols + pix_cur_x_i;
        Inten = w_cur_tl * img_ptr[0] + w_cur_tr * img_ptr[1] + w_cur_bl * img_ptr[stride] + w_cur_br * img_ptr[stride+1];

        dx = 0.5*( (w_cur_tl * img_ptr[1] + w_cur_tr * img_ptr[2] + w_cur_bl * img_ptr[stride+1] + w_cur_br * img_ptr[stride+2]) -
                  (w_cur_tl * img_ptr[-1] + w_cur_tr * img_ptr[0] + w_cur_bl * img_ptr[stride-1] + w_cur_br * img_ptr[stride]) );
        dy = 0.5*( (w_cur_tl * img_ptr[stride] + w_cur_tr * img_ptr[stride+1] + w_cur_bl * img_ptr[2*stride] + w_cur_br * img_ptr[2*stride+1])-
                   (w_cur_tl * img_ptr[-stride] + w_cur_tr * img_ptr[-stride+1] + w_cur_bl * img_ptr[0] + w_cur_br * img_ptr[1]) );

        return true;
    }
}

// for Sim3 ---------
inline Eigen::Vector3d deltaR(const Eigen::Matrix3d& R)
{
  Eigen::Vector3d v;
  v(0)=R(2,1)-R(1,2);
  v(1)=R(0,2)-R(2,0);
  v(2)=R(1,0)-R(0,1);
  return v;
}

inline Eigen::Matrix3d skew(const Eigen::Vector3d&v)
{
  Eigen::Matrix3d m;
  m.fill(0.);
  m(0,1)  = -v(2);
  m(0,2)  =  v(1);
  m(1,2)  = -v(0);
  m(1,0)  =  v(2);
  m(2,0) = -v(1);
  m(2,1) = v(0);
  return m;
}

inline Eigen::Matrix <double, 7, 1> logSim3(const Eigen::Quaterniond &r, const Eigen::Vector3d &t, const double &s)
{
    Eigen::Matrix <double, 7, 1> res;

    double sigma = std::log(s);

    Eigen::Vector3d omega;
    Eigen::Vector3d upsilon;

    Eigen::Matrix3d R = r.toRotationMatrix();
    double d =  0.5*(R(0,0)+R(1,1)+R(2,2)-1);

    Eigen::Matrix3d Omega;

    double eps = 0.00001;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    double A,B,C;
    if (fabs(sigma)<eps)
    {
      C = 1;
      if (d>1-eps)
      {
        omega=0.5*deltaR(R);
        Omega = skew(omega);
        A = 1./2.;
        B = 1./6.;
      }
      else
      {
        double theta = acos(d);
        double theta2 = theta*theta;
        omega = theta/(2*sqrt(1-d*d))*deltaR(R);
        Omega = skew(omega);
        A = (1-cos(theta))/(theta2);
        B = (theta-sin(theta))/(theta2*theta);
      }
    }
    else
    {
      C=(s-1)/sigma;
      if (d>1-eps)
      {

        double sigma2 = sigma*sigma;
        omega=0.5*deltaR(R);
        Omega = skew(omega);
        A = ((sigma-1)*s+1)/(sigma2);
        B = ((0.5*sigma2-sigma+1)*s)/(sigma2*sigma);
      }
      else
      {
        double theta = acos(d);
        omega = theta/(2*sqrt(1-d*d))*deltaR(R);
        Omega = skew(omega);
        double theta2 = theta*theta;
        double a=s*sin(theta);
        double b=s*cos(theta);
        double c=theta2 + sigma*sigma;
        A = (a*sigma+ (1-b)*theta)/(theta*c);
        B = (C-((b-1)*sigma+a*theta)/(c))*1./(theta2);
      }
    }

    Eigen::Matrix3d W = A*Omega + B*Omega*Omega + C*I;

    upsilon = W.lu().solve(t);

    for (int i=0; i<3; i++)
      res[i] = omega[i];

     for (int i=0; i<3; i++)
      res[i+3] = upsilon[i];

    res[6] = sigma;

    return res;

}

#endif // MODELTOOL_HPP_
