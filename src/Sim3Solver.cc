/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#include <Sim3Solver.h>

namespace TextSLAM
{

Sim3Solver::Sim3Solver(keyframe* pKF1, keyframe* pKF2, const vector<FeatureConvert> &vFeat1, const vector<FeatureConvert> &vFeat2):
    mnIterations(0), mnBestInliers(0)
{
    mpKF1 = pKF1;
    mpKF2 = pKF2;

    mK1 = pKF1->mK;
    mK2 = pKF2->mK;

    assert( (int)vFeat1.size()==(int)vFeat2.size() );
    for(size_t i=0; i<vFeat1.size(); i++){
        mvX3Dc1.push_back(vFeat1[i].posObv);
        mvX3Dc2.push_back(vFeat2[i].posObv);
        mvP1im1.push_back(vFeat1[i].obv2dPred);
        mvP2im2.push_back(vFeat2[i].obv2dPred);
    }

    // number of correspondences
    assert((int)mvX3Dc1.size()==(int)mvX3Dc2.size());
    assert((int)mvP1im1.size()==(int)mvP2im2.size());
    assert((int)mvX3Dc1.size()==(int)mvP1im1.size());
    N = mvX3Dc1.size();
}


void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mnIterations = 0;
    mvbInliersi.resize(N);
    mRansacMaxIts = maxIterations;

    float epsilon = (float)mRansacMinInliers/N;
    int nIterations;
    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3)));
    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

}

bool Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers, Mat44 &BestT12Res)
{
    bNoMore = false;
    vbInliers = vector<bool>(N,false);
    nInliers=0;

    if(N<mRansacMinInliers)
    {
        bNoMore = true;
        return false;
    }

    Mat33 P3Dc1i, P3Dc2i;
    vector<size_t> vAvailableIndices = Tool.InitialVec(N);
    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;

        // Get min set of points
        for(short i = 0; i < 3; ++i){
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];

            P3Dc1i.block<3,1>(0,i) = mvX3Dc1[idx];
            P3Dc2i.block<3,1>(0,i) = mvX3Dc2[idx];

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        ComputeSim3(P3Dc1i,P3Dc2i);

        CheckInliers();

        if(mnInliersi>=mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i;
            mBestRotation = mR12i;
            mBestTranslation = mt12i;
            mBestScale = ms12i;

            if(mnInliersi>mRansacMinInliers)
            {
                nInliers = mnInliersi;
                vbInliers = mvbBestInliers;
                BestT12Res = mBestT12;
            }
        }

    }
    if(mnIterations>=mRansacMaxIts)
        bNoMore=true;
    if(mnBestInliers>mRansacMinInliers){
        return true;
    }else{
        return false;
    }
}


void Sim3Solver::ComputeSim3(Mat33 &P1, Mat33 &P2)
{
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates
    Mat33 Pr1, Pr2;
    Mat31 O1, O2;
    ComputeCentroid(P1,Pr1,O1);
    ComputeCentroid(P2,Pr2,O2);

    // Step 2: Compute M matrix
    Mat33 M = Pr2*Pr1.transpose();

    // Step 3: Compute N matrix
    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;
    Mat44 N;
    N11 = M(0,0)+M(1,1)+M(2,2);
    N12 = M(1,2)-M(2,1);
    N13 = M(2,0)-M(0,2);
    N14 = M(0,1)-M(1,0);
    N22 = M(0,0)-M(1,1)-M(2,2);
    N23 = M(0,1)+M(1,0);
    N24 = M(2,0)+M(0,2);
    N33 = -M(0,0)+M(1,1)-M(2,2);
    N34 = M(1,2)+M(2,1);
    N44 = -M(0,0)-M(1,1)+M(2,2);
    N << N11, N12, N13, N14,
         N12, N22, N23, N24,
         N13, N23, N33, N34,
         N14, N24, N34, N44;

    // Step 4: Eigenvector of the highest eigenvalue
    Eigen::EigenSolver<Mat44> es(N);
    Eigen::MatrixXcd eval = es.eigenvalues();
    Eigen::MatrixXcd evec = es.eigenvectors();
    Eigen::MatrixXd evalReal = eval.real();
    Eigen::MatrixXd::Index evalsMax;
    evalReal.rowwise().sum().maxCoeff(&evalsMax);
    Eigen::Quaterniond q(evec.real()(0,evalsMax),evec.real()(1,evalsMax),evec.real()(2,evalsMax),evec.real()(3,evalsMax));
    q = q.normalized();
    Mat33 R(q);
    mR12i = R;

    // Step 5: Rotate set 2
    Mat33 P3 = mR12i*Pr2;

    // Step 6: Scale
    double nom = Pr1.cwiseProduct(P3).sum();
    Mat33 aux_P3 = Eigen::pow(P3.array(), 2);
    double den = aux_P3.sum();
    ms12i = nom/den;

    // Step 7: Translation
    mt12i = O1 - ms12i*mR12i*O2;

    // Step 8: Transformation
    // Step 8.1 T12
    mT12i.setIdentity();
    Mat33 sR = ms12i*mR12i;
    mT12i.block<3,3>(0,0) = sR;
    mT12i.block<3,1>(0,3) = mt12i;

    // Step 8.2 T21
    mT21i.setIdentity();
    Mat33 sRinv = (1.0/ms12i)*mR12i.transpose();
    mT21i.block<3,3>(0,0) = sRinv;
    mT21i.block<3,1>(0,3) = -sRinv*mt12i;

}

void Sim3Solver::CheckInliers()
{
    double MaxError1 = 45.0, MaxError2 = 45.0;

    vector<Vec2> vP1im2, vP2im1;
    Project(mvX3Dc2,vP2im1,mT12i,mK1, mpKF1);
    Project(mvX3Dc1,vP1im2,mT21i,mK2, mpKF2);

    mnInliersi=0;

    for(size_t i=0; i<mvP1im1.size(); i++)
    {
        Vec2 dist1 = mvP1im1[i]-vP2im1[i];
        Vec2 dist2 = vP1im2[i]-mvP2im2[i];

        const float err1 = dist1.dot(dist1);
        const float err2 = dist2.dot(dist2);

        if(err1<MaxError1 && err2<MaxError2)
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
            mvbInliersi[i]=false;
    }
}

void Sim3Solver::Project(const vector<Mat31> &vP3Dw, vector<Vec2> &vP2D, Mat44 Tcw, Mat33 K, keyframe* KF)
{
    Mat33 Rcw = Tcw.block<3,3>(0,0);
    Mat31 tcw = Tcw.block<3,1>(0,3);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
    {
        Mat31 P3Dc = K * (Rcw*vP3Dw[i]+tcw);
        double u = P3Dc(0)/P3Dc(2);
        double v = P3Dc(1)/P3Dc(2);

        vP2D.push_back(Vec2(u, v));

    }

}


void Sim3Solver::ComputeCentroid(const Mat33 &P, Mat33 &Pr, Mat31 &C)
{
    C = P.rowwise().mean();

    for(int i=0; i<P.cols(); i++)
    {
        Pr.col(i)=P.col(i)-C;
    }

}


Mat33 Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation;
}

Mat31 Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation;
}

double Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}

}
