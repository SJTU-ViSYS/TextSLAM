/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <frame.h>

namespace TextSLAM
{

long unsigned int frame::nNextId=0;
double frame::cx, frame::cy, frame::fx, frame::fy, frame::invfx, frame::invfy;
double frame::mnMinX, frame::mnMinY, frame::mnMaxX, frame::mnMaxY;
double frame::mfGridElementWidthInv, frame::mfGridElementHeightInv;

frame::frame(){}

/*
 * func: frame::frame
 * param In:
 * cv::Mat imGray: Img to frame; double ImgTimeStamp: timestamp of this frame; int ScaleLevels: pyramid level number; int ScaleFactor: pyramid scale factor;
 * vector<Vec2> TextDete: Text detection result; vector<string> TextMean: Text recognition result; ORBextractor* extractor: feature extraction params
 * param Out:
 * ----
 * return:
 * frame CurrentFrame
 */
frame::frame(const cv::Mat &imGray, const double &ImgTimeStamp, const Mat33 &K, int &ScaleLevels, double &ScaleFactor, const vector<vector<Vec2>> &TextDete, const vector<TextInfo> &TextMean, ORBextractor* extractor, const bool &bVeloc):
dTimeStamp(ImgTimeStamp), iScaleLevels(ScaleLevels), mK(K), dScaleFactor(ScaleFactor), vTextDete(TextDete), vTextMean(TextMean), coORBextractor(extractor), bVelocity(bVeloc)
{
    // 1. basic param
    imGray.copyTo(FrameImg);
    mnId=nNextId++;
    mK = K;
    cfLocalKF = static_cast<keyframe*>(NULL);

    // 2. pyramid basic info
    if(!GetPyrParam())
        throw runtime_error("Pyramid param input error.");
    GetPyrMat();
    // 3. frame basic info, and text detection info
    GetFrameParam();

    // 4. feature extractor
    FeatExtract();
    FeatFusion();
    TextFeaProc();

    AssignFeaturesToGrid();

    if(vKeys.empty())
        return;

}

// ------------ param info ------------
bool frame::GetPyrParam()
{
    bool DEBUG = false;
    if(iScaleLevels==0 || dScaleFactor==0)
        return false;

    vScaleFactors.resize(iScaleLevels);
    vInvScaleFactors.resize(iScaleLevels);
    vLevelSigma2.resize(iScaleLevels);
    vInvLevelSigma2.resize(iScaleLevels);
    vK_scale.resize(iScaleLevels);
    vScaleFactors[0]=1.0f;
    vLevelSigma2[0]=1.0f;
    vK_scale[0] = mK;

    for(size_t i0 = 1; i0<iScaleLevels; i0++){
        vScaleFactors[i0] = vScaleFactors[i0-1] * dScaleFactor;
        vLevelSigma2[i0] = vScaleFactors[i0] * vScaleFactors[i0];
        vK_scale[i0] = mK/vScaleFactors[i0];
        vK_scale[i0](2,2) = 1.0;
    }
    for(size_t i1 = 0; i1<iScaleLevels; i1++){
        vInvScaleFactors[i1] = 1.0/vScaleFactors[i1];
        vInvLevelSigma2[i1] = 1.0/vLevelSigma2[i1];
    }

    if(DEBUG){
        for(size_t i2 = 0; i2<iScaleLevels; i2++){
            cout<<"DEBUG"<<endl;
            cout<<"[frame] pyramid info->vScaleFactors: "<<vScaleFactors[i2]<<endl;
            cout<<"[frame] pyramid info->vInvScaleFactors: "<<vInvScaleFactors[i2]<<endl;
            cout<<"[frame] pyramid info->vLevelSigma2: "<<vLevelSigma2[i2]<<endl;
            cout<<"[frame] pyramid info->vInvLevelSigma2: "<<vInvLevelSigma2[i2]<<endl;
            cout<<"[frame] pyramid info->vK_scale: "<<vK_scale[i2]<<endl;
        }
    }

    return true;
}

// frame basic param. text param
void frame::GetFrameParam()
{
    // 1. frame basic param
    bool DEBUG = false;
    fx = mK(0,0);
    fy = mK(1,1);
    cx = mK(0,2);
    cy = mK(1,2);
    invfx = 1.0/fx;
    invfy = 1.0/fy;

    mnMinX = 0.0f;
    mnMaxX = FrameImg.cols;
    mnMinY = 0.0f;
    mnMaxY = FrameImg.rows;

    mfGridElementWidthInv = static_cast<double>(FRAME_GRID_COLS)/static_cast<double>(mnMaxX-mnMinX);
    mfGridElementHeightInv = static_cast<double>(FRAME_GRID_ROWS)/static_cast<double>(mnMaxY-mnMinY);
    if(DEBUG){
        cout<<"fx, fy, cx, cy, invfx, invfy are: "<<fx<<", "<<fy<<", "<<cx<<", "<<cy<<", "<<invfx<<", "<<invfy<<endl;
        cout<<"mnMinX, mnMaxX, mnMinY, mnMaxY are: "<<mnMinX<<", "<<mnMaxX<<", "<<mnMinY<<", "<<mnMaxY<<endl;
        cout<<"mfGridElementWidthInv and mfGridElementHeightInv are: "<<mfGridElementHeightInv<<", "<<mfGridElementHeightInv<<endl;
    }

    // 2. text detection info
    for(size_t i0 = 0; i0<vTextDete.size(); i0++){
        double minx = 650, miny = 650, maxx = -1, maxy = -1;
        double sumx = 0.0, sumy = 0.0;
        vector<Vec2> pts = vTextDete[i0];
        for(size_t i1 = 0; i1<pts.size(); i1++){
            // A) delete out of image bound
            if(pts[i1](0)<0){
                pts[i1](0) = 0;
                vTextDete[i0][i1] =pts[i1];
            }
            if(pts[i1](0)>=FrameImg.cols){
                pts[i1](0) = FrameImg.cols-1;
                vTextDete[i0][i1] =pts[i1];
            }
            if(pts[i1](1)<0){
                pts[i1](1) = 0;
                vTextDete[i0][i1] =pts[i1];
            }
            if(pts[i1](1)>=FrameImg.rows){
                pts[i1](1) = FrameImg.rows-1;
                vTextDete[i0][i1] =pts[i1];
            }


            // A) Min & Max
            if(pts[i1](0)<minx)
                minx = pts[i1](0);
            if(pts[i1](0)>maxx)
                maxx = pts[i1](0);
            if(pts[i1](1)<miny)
                miny = pts[i1](1);
            if(pts[i1](1)>maxy)
                maxy = pts[i1](1);
            // B) Center
            sumx += pts[i1](0);
            sumy += pts[i1](1);
        }
        vTextDeteMin.push_back(Vec2(minx, miny));
        vTextDeteMax.push_back(Vec2(maxx, maxy));
        vTextDeteCenter.push_back( Vec2(sumx/4.0, sumy/4.0) );
    }


}

void frame::GetPyrMat()
{
    // 1. get pyramid image
    vFrameImg.resize(iScaleLevels);
    for(size_t i0 = 0; i0<iScaleLevels; i0++){
        if(i0==0)
            vFrameImg[i0] = FrameImg;
        else
            cv::pyrDown(vFrameImg[i0-1], vFrameImg[i0]);
    }

    // 2. get gradient image
    int scale = 1, delta = 0, ddepth = vFrameImg[0].type();
    for(size_t i1 = 0; i1<iScaleLevels; i1++){
        cv::Mat grad_x, grad_y, grad;
        cv::Mat abs_grad;
        // Gradient X
        Sobel( vFrameImg[i1], grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        // Gradient Y
        Sobel( vFrameImg[i1], grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        addWeighted( grad_x, 0.5, grad_y, 0.5, 0, grad );
        vFrameGrad.push_back(grad);
        vFrameGradX.push_back(grad_x);
        vFrameGradY.push_back(grad_y);
    }

}
// ------------ param info ------------

// ------------ Feature proc ------------
void frame::FeatExtract()
{

    // ---- log ----
    bool FLAG_FEAPROC = true;           // after feature extractor, delete features close to the mask boundary
    bool FLAG_NOSCENEMASK = true;       // use for TextSLAM comparised with pts. If true(compair), extract scene features within the text features extraction
    // ---- log ----

    // 1. get mask for scene featrue extraction
    cv::Mat ImgScene;
    if(FLAG_NOSCENEMASK){
        ImgScene = FrameImg.clone();
    }else{
        ImgScene = Tool.GetInvMask(FrameImg.clone(), vTextDete);
    }

    // 2. scene feature extraction
    vector<cv::KeyPoint> KeysSceneRaw;
    cv::Mat DescripSceneRaw;
    FeatExtraScene(ImgScene, KeysSceneRaw, DescripSceneRaw);

    // 3. text feature extraction
    vector<vector<cv::KeyPoint>> KeysTextRaw;
    vector<cv::Mat> DescripTextRaw;
    FeatExtracText(FrameImg.clone(), vTextDete, KeysTextRaw, DescripTextRaw);

    // 4. delete features if they are too close to the mask boundary
    vector<vector<int>> IdxNewText;
    vector<int> IdxNeScene;
    float WinText = -3.0f, WinScene = 5.0f;
    if(FLAG_FEAPROC && !FLAG_NOSCENEMASK){
        Tool.BoundFeatDele_T(FrameImg, KeysTextRaw, DescripTextRaw, vTextDete, vTextDeteMin, vTextDeteMax, WinText,
                             vKeysText, mDescrText, IdxNewText);
        Tool.BoundFeatDele_S(KeysSceneRaw, DescripSceneRaw, vTextDete, WinScene,
                             vKeysScene, mDescrScene, IdxNeScene);
    }else if(FLAG_FEAPROC && FLAG_NOSCENEMASK){
        Tool.BoundFeatDele_T(FrameImg, KeysTextRaw, DescripTextRaw, vTextDete, vTextDeteMin, vTextDeteMax, WinText,
                             vKeysText, mDescrText, IdxNewText);
        vKeysScene = KeysSceneRaw;
        mDescrScene = DescripSceneRaw;
    }else{
        vKeysScene = KeysSceneRaw;
        mDescrScene = DescripSceneRaw;
        vKeysText = KeysTextRaw;
        mDescrText = DescripTextRaw;
    }

}

void frame::FeatFusion()
{
    iNScene = vKeysScene.size();
    iNTextObj = vKeysText.size();
    iNTextFea = 0;
    bool FLAG_NOSCENEPTS = false;

    // 1. judge has scene pts or not. Get initial Descriptor if has scene pts
    if(mDescrScene.rows!=0){
        mDescr = mDescrScene;
    }else{
        FLAG_NOSCENEPTS = true;
    }

    // 2. text feat info add into Descriptor
    if(!FLAG_NOSCENEPTS)        // have scene pts
    {
        for(size_t i0 = 0; i0<iNTextObj; i0++){
            iNTextFea += vKeysText[i0].size();
            if(mDescrText[i0].rows==0){
                continue;
            }
            cv::vconcat(mDescr, mDescrText[i0], mDescr);
        }
    }else{                      // no scene pts
        if(iNTextObj!=0){
            mDescr = mDescrText[0];
            iNTextFea += vKeysText[0].size();
            for(size_t i0 = 1; i0<iNTextObj; i0++){
                iNTextFea += vKeysText[i0].size();
                if(mDescrText[i0].rows==0){
                    continue;
                }
                cv::vconcat(mDescr, mDescrText[i0], mDescr);
            }
        }
    }

    // 3. Get scene + text features
    iN = iNScene + iNTextFea;
    vKeys.resize(iN);
    vKeysSceneIdx.resize(iNScene);
    vKeysTextIdx.resize(iNTextObj);
    vTextObjInfo = vector<int>(iN, int(-1));

    for(size_t i1 = 0; i1<iNScene; i1++){
        vKeys[i1] = vKeysScene[i1];
        vKeysSceneIdx[i1] = i1;
    }

    int IdxBeginText = iNScene, IdxText = 0;
    for(size_t i2 = 0; i2<iNTextObj; i2++){
        vector<cv::KeyPoint> KeysTextCell = vKeysText[i2];
        vKeysTextIdx[i2].resize(KeysTextCell.size());
        for(size_t i3 = 0; i3<KeysTextCell.size(); i3++){
            int IdxFea = IdxText + IdxBeginText;
            vKeys[IdxFea] = KeysTextCell[i3];
            vTextObjInfo[IdxFea] = i2;
            vKeysTextIdx[i2][i3] = IdxFea;
            IdxText++;
        }
    }

    if(iN!=(IdxText+IdxBeginText) || iN!=mDescr.rows || iNScene!=mDescrScene.rows || iNTextObj!=mDescrText.size()){
        cerr<<"error: Keys number is not same."<<endl;
    }


}

// extract featrues(Keys) from image(Img), calculate its description(Desc)
void frame::FeatExtraScene(cv::Mat &Img, vector<cv::KeyPoint> &Keys, cv::Mat &Desc)
{
    (*coORBextractor)(Img,cv::Mat(), Keys, Desc);
}

// extract features(KeysTextRaw) from each mask in the image (Img&TextDete), calculate each description(DescripTextRaw)
void frame::FeatExtracText(const cv::Mat &Img, const vector<vector<Vec2>> &TextDete, vector<vector<cv::KeyPoint>> &KeysTextRaw, vector<cv::Mat> &DescripTextRaw)
{
    vector<cv::Mat> ImgText;
    ImgText = Tool.GetMask(Img, TextDete);
    int keys_sum = 0;

    for(size_t i0 = 0; i0<ImgText.size(); i0++){
        vector<cv::KeyPoint> KeysCell;
        cv::Mat DescCell;
        cv::Mat MaskCell = ImgText[i0];

        cv::Ptr<cv::FeatureDetector> detecText = cv::ORB::create();
        cv::Ptr<cv::DescriptorExtractor> descrbText = cv::ORB::create();
        detecText->detect(MaskCell, KeysCell);
        descrbText->compute(Img, KeysCell, DescCell);

        KeysTextRaw.push_back(KeysCell);
        DescripTextRaw.push_back(DescCell);

        keys_sum += KeysCell.size();
    }
}


// for each text object in the frame, get its pyramid
void frame::TextFeaProc()
{
    for(size_t i0 = 0; i0<iNTextObj; i0++){         // each text object
        assert(i0>=0);
        assert(i0<vKeysText.size());
        vector<cv::KeyPoint> vKeysTextObj = vKeysText[i0];
        vector<vector<TextFeature*>> vTextfeature;      // pyramid -> all feature in the pyramid
        Tool.GetPyramidPts(vKeysTextObj, vTextDeteMin[i0], vTextDeteMax[i0], vFrameImg, vFrameGrad, vInvScaleFactors, vK_scale, vTextfeature);
        vfeatureText.push_back(vTextfeature);
        vTextfeature.clear();
    }
}

void frame::AssignFeaturesToGrid()
{

    int nReserve = iN/(FRAME_GRID_COLS*FRAME_GRID_ROWS);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);
    int nprocess = 0;
    for(int i=0;i<iN;i++)
    {
        const cv::KeyPoint &kp = vKeys[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
        {
            mGrid[nGridPosX][nGridPosY].push_back(i);
            nprocess++;
        }
    }


}

bool frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}




// ------------ Feature proc ------------


// for search
vector<size_t> frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(iN);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = vKeys[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

// add observation
void frame::AddTextObserv(mapText* textobj, int idx)
{
    vector<int> DeteIdx;
    DeteIdx.push_back(idx);

    TextObservation* textobv = new TextObservation{textobj, DeteIdx};
    vObvText.push_back(textobv);
}

void frame::AddTextObserv(mapText* textobj)
{
    vector<int> DeteIdx;

    TextObservation* textobv = new TextObservation{textobj, DeteIdx};
    vObvText.push_back(textobv);
}


void frame::AddTextObserv(mapText* textobj, const int &Feat0PyrNum, const vector<int> &vDeteIdx)
{
    TextObservation* textobv = new TextObservation{textobj, vDeteIdx};

    // add map text
    vObvText.push_back(textobv);

    // observation flag
    vObvGoodTexts.push_back(true);

    vector<bool> vFlagTextFeats = vector<bool>(Feat0PyrNum, true);
    vObvGoodTextFeats.push_back(vFlagTextFeats);
}

void frame::AddSceneObserv(mapPts* scenept, const int &ScenePtId, int idx)
{
    SceneObservation* ptsobv = new SceneObservation{scenept, idx};

    // add map points
    vObvPts.push_back(ptsobv);

    // observation flag
    vObvGoodPts.push_back(true);

    // add observation
    vMatches2D3D[idx] = ScenePtId;

    int Idx2Raw = vObvPts.size()-1;
    Vec2 Observ2D = Vec2((double)vKeys[idx].pt.x, (double)vKeys[idx].pt.y);
    for(size_t ipyr=0; ipyr<vSceneObv2d.size(); ipyr++){
        Vec2 Observ2DPyr = Vec2( Observ2D(0)*vInvScaleFactors[ipyr], Observ2D(1)*vInvScaleFactors[ipyr] );
        SceneFeature* featureInPyr = new SceneFeature{Observ2DPyr(0), Observ2DPyr(1), Observ2DPyr, (int)ipyr, Idx2Raw};
        vSceneObv2d[ipyr].push_back(featureInPyr);
    }
}

// for pose
void frame::SetPose(Mat44 Tcw)
{
    mTcw = Tcw;
    UpdatePoseMatrices();
}
void frame::UpdatePoseMatrices()
{
    mRcw = mTcw.block<3,3>(0,0);
    mRwc = mRcw.transpose();
    mtcw = mTcw.block<3,1>(0,3);
    mtwc = -mRwc*mtcw;
    mTwc = mTcw.inverse();

}

}
