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
#include <tool.h>

namespace TextSLAM
{
// ------------ IO ------------
void tool::ReadImage(const string &Path, vector<string> &ImgName, vector<string> &ImgIdx, vector<double> &ImgTime)
{
    ifstream f;
    f.open(Path.c_str());
    if(!f.is_open()){
        cerr << "Failed to open image list file at: " << Path << endl;
        exit(-1);
    }


    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;                                        // get double val
            ImgTime.push_back(t);
            ss >> sRGB;                                     // get string val
            ImgName.push_back(sRGB);
            sRGB.erase(sRGB.length() - 4, sRGB.length());   // from string val get idx
            ImgIdx.push_back(sRGB);
        }
    }
    f.close();
}

// use vector<TextInfo> to store text meaning
void tool::ReadText(const string &Path, vector<vector<Eigen::Matrix<double,2,1>>> &vDetec, vector<TextInfo> &vMean, const int &Flag_Exp, const bool &Flag_noText)
{
    bool DOUBLE = false;

    string path1, path2;
    string path_text = Path;

    path_text = path_text.replace(Path.find("images"), 6, "text");
    path1 = path_text+"_dete.txt";
    path2 = path_text+"_mean.txt";

    // detection information
    std::ifstream infile;
    infile.open(path1.data());
    assert(infile.is_open());

    vector<Eigen::Matrix<double,2,1>> vDetecRaw;
    std::string s;
    int idx = -1;
    while(getline(infile,s))
    {
        idx ++;
        char *char_s = (char *)s.c_str();
        const char *split = ",";
        char *p = strtok(char_s, split);

        std::vector<double> nums;
        double a;
        while(p != NULL)
        {
            std::sscanf(p, "%lf", &a);
            nums.push_back(a);
            p=std::strtok(NULL, split);
        }

        assert(nums.size()==8);

        for(int i = 0; i < nums.size(); )
        {
            Eigen::Matrix<double,2,1> data_tmp;
            if(DOUBLE)
                data_tmp = Eigen::Matrix<double,2,1>(nums[i], nums[i+1]);
            else
                data_tmp = Eigen::Matrix<double,2,1>(std::round(nums[i]), std::round(nums[i+1]));

            vDetecRaw.push_back(data_tmp);
            i = i+2;
        }

    }

    infile.close();
    GetDeteInfo(vDetecRaw, vDetec);

    // recognition information
    ifstream f;
    f.open(path2.c_str());
    int UseRead = 1;

    // read 1: get 'mean,'
    if(UseRead==0){
        while(!f.eof())
        {
            string s;
            getline(f,s);
            if(!s.empty())
            {
                stringstream ss;
                ss << s;
                string MeanTmp;
                ss >> MeanTmp;
                struct TextInfo TextMean = {MeanTmp, 1.0, 0};
                vMean.push_back(TextMean);
            }
        }
    }
    // read 2: get 'mean' and mean_score
    vector<string> vMeanRaw;    // text mean
    vector<int> vLangRaw;          // text language
    vector<double> vScoreRaw;      // text recognition score
    double MAX_SCORE = DBL_MAX;
    if(UseRead==1){

        char* m_binaryStr;
        int m_length;
        Readfulltxt(path2, m_binaryStr, m_length);
        size_t m_index = 0;

        string OneChar;
        vector<string> OneTextMean;
        double sum_len = 0;
        int idx_line = 0;

        while(m_index < m_length){
            // 1. get single character
            size_t utf8_char_len = get_utf8_char_len(m_binaryStr[m_index]);
            size_t next_idx = m_index + utf8_char_len;
            OneChar = string(m_binaryStr + m_index, next_idx - m_index);

            OneTextMean.push_back(OneChar);
            sum_len += utf8_char_len;

            // 2. encouter , get word -----------
            if(OneChar==","){
                string TextMean;
                for(size_t ichar=0; ichar<OneTextMean.size()-1; ichar++){
                    TextMean = TextMean+OneTextMean[ichar];
                }
                vMeanRaw.push_back(TextMean);

                // 3. which language. english -- 0; Chinese -- 1; Chinese+english -- 2
                int lang;
                if( (sum_len-1)==(OneTextMean.size()-1) )
                    lang = (int)0;
                else if( (sum_len-1)==3*(OneTextMean.size()-1) )
                    lang = (int)1;
                else if( (sum_len-1)<3*(OneTextMean.size()-1) && sum_len>(OneTextMean.size()-1))
                    lang = (int)2;
                vLangRaw.push_back(lang);

                TextMean.clear();
                OneTextMean.clear();        // clear text in OneTextMean
                sum_len = 0;
            }
            // 2 -------------------------------

            // 4. end of the line, get recognition score
            if(OneChar=="\n"){
                string TextScore;
                for(size_t ichar=0; ichar<OneTextMean.size()-1; ichar++){
                    TextScore = TextScore+OneTextMean[ichar];
                }
                stringstream ss;
                ss << TextScore;
                double score;
                ss >> score;
                vScoreRaw.push_back(score);

                struct TextInfo TextMean = {vMeanRaw[idx_line], vScoreRaw[idx_line], MAX_SCORE, vLangRaw[idx_line]};
                vMean.push_back(TextMean);

                TextScore.clear();
                OneTextMean.clear();
                sum_len = 0;
                idx_line++;

            }

            m_index = next_idx;

        }

    }


    f.close();

    if(vDetec.size()!=vMean.size()){
        cout<<"recognition input error."<<endl;
    }
    assert(vDetec.size()==vMean.size());

}

// read txt
bool tool::Readfulltxt(const string &Path, char* &m_binaryStr, int &m_length)
{
    std::ifstream infile;
    infile.open(Path.c_str(), ios::binary);
    if(!infile){
        cout<<Path<<" Load text error."<<endl;
        return false;
    }

    filebuf *pbuf=infile.rdbuf();
    m_length = (int)pbuf->pubseekoff(0,ios::end,ios::in);
    pbuf->pubseekpos(0,ios::in);

    m_binaryStr = new char[m_length+1];
    pbuf->sgetn(m_binaryStr,m_length);
    infile.close();

    return true;
}

size_t tool::get_utf8_char_len(const char & byte)
{

    size_t len = 0;
    unsigned char mask = 0x80;
    while( byte & mask )
    {
        len++;
        if( len > 6 )
        {
            return 0;
        }
        mask >>= 1;
    }
    if( 0 == len)
    {
        return 1;
    }
    return len;
}



// ------------ IO ------------

// ------------ Semantic ------------
double tool::LevenshteinDist(const string &str1in, const string &str2in)
{
    double dist;
    int len1 = (int)str1in.length(), len2 = (int)str2in.length();
    char str1[len1+1], str2[len2+1];
    str1in.copy(str1, len1, 0);
    str2in.copy(str2, len2, 0);
    *(str1+len1)='\0';
    *(str2+len2)='\0';

    int dp[len1+1][len2+1];
    for(size_t iraw=0; iraw<=len1; iraw++){
        for(size_t icol=0; icol<=len2; icol++){
            dp[iraw][icol] = 0;
        }
    }

    for(int i=1; i<=len1; i++){
        dp[i][0] = i;
    }
    for(int j=1; j<=len2; j++){
        dp[0][j] = j;
    }
    for(int irow=1; irow<=len1; irow++){
        for(int icol=1; icol<=len2; icol++){
            if(str1[irow-1]==str2[icol-1]){
                dp[irow][icol] = dp[irow-1][icol-1];
            }else{
                dp[irow][icol] = min(dp[irow-1][icol-1], min(dp[irow][icol-1], dp[irow-1][icol]))+1;
            }
        }
    }

    dist = dp[len1][len2];
    return dist;
}
// ------------ Semantic ------------

// ------------ Param proc ------------
void tool::GetDeteInfo(const vector<Eigen::Matrix<double,2,1>> &vDetecRaw, vector<vector<Eigen::Matrix<double,2,1>>> &vDetec)
{
    for(size_t i0 = 0; i0<vDetecRaw.size(); ){
        vector<Eigen::Matrix<double,2,1>> DetecCell;
        DetecCell.push_back(vDetecRaw[i0]);
        DetecCell.push_back(vDetecRaw[i0+1]);
        DetecCell.push_back(vDetecRaw[i0+2]);
        DetecCell.push_back(vDetecRaw[i0+3]);
        vDetec.push_back(DetecCell);
        DetecCell.clear();
        i0 += 4;
    }
}

Mat33 tool::cvM2EiM33(const Mat &cvMf)
{
    Mat33 EiM;
    EiM<< (double)cvMf.at<float>(0,0), (double)cvMf.at<float>(0,1), (double)cvMf.at<float>(0,2),
          (double)cvMf.at<float>(1,0), (double)cvMf.at<float>(1,1), (double)cvMf.at<float>(1,2),
          (double)cvMf.at<float>(2,0), (double)cvMf.at<float>(2,1), (double)cvMf.at<float>(2,2);
    return EiM;
}
Mat31 tool::cvM2EiM31(const Mat &cvMf)
{
    Mat31 EiM;
    EiM<< (double)cvMf.at<float>(0,0),
          (double)cvMf.at<float>(1,0),
          (double)cvMf.at<float>(2,0);
    return EiM;
}

cv::Mat tool::EiM442cvM(const Mat44 &EiMf)
{
    cv::Mat cvMd;
    cvMd = (Mat_<double>(4, 4) <<
               EiMf(0,0), EiMf(0,1), EiMf(0,2), EiMf(0,3),
               EiMf(1,0), EiMf(1,1), EiMf(1,2), EiMf(1,3),
               EiMf(2,0), EiMf(2,1), EiMf(2,2), EiMf(2,3),
               EiMf(3,0), EiMf(3,1), EiMf(3,2), EiMf(3,3));
    return cvMd;
}

cv::Mat tool::EiM332cvMf(const Mat33 &EiMf)
{
    cv::Mat cvMd;
    cvMd = (Mat_<float>(3, 3) <<
               (float)EiMf(0,0), (float)EiMf(0,1), (float)EiMf(0,2),
               (float)EiMf(1,0), (float)EiMf(1,1), (float)EiMf(1,2),
               (float)EiMf(2,0), (float)EiMf(2,1), (float)EiMf(2,2));
    return cvMd;
}

Mat44 tool::Pose2Mat44(double* pose)
{
    Eigen::Quaterniond q(pose[0], pose[1], pose[2], pose[3]);
    q = q.normalized();
    Mat33 R(q);
    Mat44 T;
    T.setIdentity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = Mat31(pose[4], pose[5], pose[6]);
    return T;
}

// ------------ Param proc ------------

// ------------ Feature extractor ------------
vector<cv::Mat> tool::GetMask(const cv::Mat &Img, const vector<vector<Vec2>> &TextDete)
{
    vector<cv::Mat> MaskRes;
    cv::Mat ImgRaw, ImgMax, InvMaskRes;
    Img.copyTo(ImgRaw);

    cv::Point TextDetePts[TextDete.size()][4];
    for(size_t i0 = 0; i0<TextDete.size(); i0++){
        TextDetePts[i0][0] = cv::Point(TextDete[i0][0](0), TextDete[i0][0](1));
        TextDetePts[i0][1] = cv::Point(TextDete[i0][1](0), TextDete[i0][1](1));
        TextDetePts[i0][2] = cv::Point(TextDete[i0][2](0), TextDete[i0][2](1));
        TextDetePts[i0][3] = cv::Point(TextDete[i0][3](0), TextDete[i0][3](1));

        cv::Mat ImgZero, MaskResCell;
        ImgZero = cv::Mat::zeros(ImgRaw.size(),CV_8UC1);
        const cv::Point* ptMask[1] = {TextDetePts[i0]};
        int npt[] = {4};
        cv::Mat MaskZero = cv::Mat::zeros(ImgRaw.size(),CV_8UC1);
        cv::fillPoly(MaskZero, ptMask, npt, 1, cv::Scalar(255));
        cv::bitwise_or(ImgZero, MaskZero, ImgZero);
        cv::bitwise_and(ImgRaw, ImgRaw, MaskResCell, ImgZero);

        MaskRes.push_back(MaskResCell);
    }

    return MaskRes;
}

cv::Mat tool::GetMask_all(const cv::Mat &Img, const vector<vector<Vec2>> &TextDete, cv::Mat &ImgLabel)
{
    cv::Mat MaskRes;
    cv::Mat ImgRaw, ImgMax, InvMaskRes;
    Img.copyTo(ImgRaw);

    cv::Point TextDetePts[TextDete.size()][4];
    cv::Mat ImgZero;
    ImgZero = cv::Mat::zeros(ImgRaw.size(),CV_8UC1);
    cv::Mat MaskZero = cv::Mat::zeros(ImgRaw.size(),CV_8UC1);
    ImgLabel = cv::Mat::ones(Img.size(), CV_32F)*(-1.0);
    for(size_t i0 = 0; i0<TextDete.size(); i0++){
        TextDetePts[i0][0] = cv::Point(TextDete[i0][0](0), TextDete[i0][0](1));
        TextDetePts[i0][1] = cv::Point(TextDete[i0][1](0), TextDete[i0][1](1));
        TextDetePts[i0][2] = cv::Point(TextDete[i0][2](0), TextDete[i0][2](1));
        TextDetePts[i0][3] = cv::Point(TextDete[i0][3](0), TextDete[i0][3](1));

        const cv::Point* ptMask[1] = {TextDetePts[i0]};
        int npt[] = {4};

        cv::fillPoly(MaskZero, ptMask, npt, 1, cv::Scalar(255));
        cv::fillPoly(ImgLabel, ptMask, npt, 1, cv::Scalar((int)i0));
    }

    cv::bitwise_or(ImgZero, MaskZero, ImgZero);
    cv::bitwise_and(ImgRaw, ImgRaw, MaskRes, ImgZero);

    return MaskRes;
}

cv::Mat tool::GetInvMask(const cv::Mat &Img, const vector<vector<Vec2>> &TextDete)
{
    cv::Mat ImgRaw, ImgZero, ImgMax, InvMaskRes;
    Img.copyTo(ImgRaw);
    ImgZero = cv::Mat::zeros(ImgRaw.size(),CV_8UC1);
    ImgMax = cv::Mat::ones(ImgRaw.size(),CV_8UC1)*255.0;

    cv::Point TextDetePts[TextDete.size()][4];

    for(size_t i0 = 0; i0<TextDete.size(); i0++){
        TextDetePts[i0][0] = cv::Point(TextDete[i0][0](0), TextDete[i0][0](1));
        TextDetePts[i0][1] = cv::Point(TextDete[i0][1](0), TextDete[i0][1](1));
        TextDetePts[i0][2] = cv::Point(TextDete[i0][2](0), TextDete[i0][2](1));
        TextDetePts[i0][3] = cv::Point(TextDete[i0][3](0), TextDete[i0][3](1));

        const cv::Point* ptMask[1] = {TextDetePts[i0]};
        int npt[] = {4};
        cv::Mat MaskZero = cv::Mat::zeros(ImgRaw.size(),CV_8UC1);
        cv::fillPoly(MaskZero, ptMask, npt, 1, cv::Scalar(255,255,255));
        cv::bitwise_or(ImgZero, MaskZero, ImgZero);
        // ImgZero is single mask output accumulation. in ImgZero, all text region is 255, other is 0.
    }
    cv::bitwise_xor(ImgZero, ImgMax, ImgZero);              // all text: 0; scene: 1
    cv::bitwise_and(ImgRaw, ImgRaw, InvMaskRes, ImgZero);   // ImgZero: 0/1 mask

    return InvMaskRes;
}

void tool::BoundFeatDele_T(const cv::Mat &Img, const vector<vector<KeyPoint>> &TextFeat, const vector<Mat> &DescText, const vector<vector<Vec2>> &TextDete, const vector<Vec2> &vTextMin, const vector<Vec2> &vTextMax, const float &Win,
                     vector<vector<cv::KeyPoint>> &TextFeat_out, vector<Mat> &DescText_out, vector<vector<int>> &IdxNew)
{
    // ---- log ----
    if(Win>0)
        cout<<"error. win change for text delete should be negative, yet "<< Win<<endl;
    // ---- log ----
    cv::Mat ImgDraw = Img.clone();

    vector<vector<Vec2>> TextDeteNew = GetNewDeteBox(TextDete, Win);
    vector<vector<Vec2>> TextbbBox = GetbbDeteBox(vTextMin, vTextMax);
    for(size_t i0 = 0; i0<TextFeat.size(); i0++){
        vector<KeyPoint> TextFeatCell = TextFeat[i0];
        vector<Vec2> TextDeteNewCell = TextDeteNew[i0];
        vector<Point> TextDeteNewCellP = vV2vP(TextDeteNewCell);
        vector<Point> TextDetebbCellP = vV2vP(TextbbBox[i0]);
        cv::Mat DescripTextOut, DescripTextCell = DescText[i0];

        // begin to delete
        vector<int> IdxNewCell;
        vector<cv::KeyPoint>::iterator itT;
        int NumOK = 0, NumDele = 0, IdxRawOrder = 0;
        for(itT = TextFeatCell.begin(); itT != TextFeatCell.end(); )
        {
            cv::KeyPoint fea = *itT;
            double feaX = fea.pt.x;
            double feaY = fea.pt.y;
            // Judge In or Out?
            if(!CheckPtsIn(TextDeteNewCellP, cv::Point2f(feaX, feaY)) || !CheckPtsIn(TextDetebbCellP, cv::Point2f(feaX, feaY)) ){
                // A) out the reduced box -> too close to the boundary -> delete
                itT = TextFeatCell.erase(itT);
                NumDele++;
            }else{
                // B) whinin the reduced box -> ok -> feature is remained
                if(NumOK == 0){
                    DescripTextOut = DescripTextCell.row(IdxRawOrder).clone();
                }else{
                    cv::vconcat(DescripTextOut, DescripTextCell.row(IdxRawOrder).clone(), DescripTextOut);
                }
                NumOK++;
                IdxNewCell.push_back(IdxRawOrder);
                ++itT;
            }
            IdxRawOrder++;
        }
        TextFeat_out.push_back(TextFeatCell);
        DescText_out.push_back(DescripTextOut);
        IdxNew.push_back(IdxNewCell);

    }


}

void tool::BoundFeatDele_S(const vector<KeyPoint> &SceneFeat, const Mat &DescScene, const vector<vector<Vec2>> &TextDete, const float &Win,
                            vector<cv::KeyPoint> &SceneFeat_out, cv::Mat &DescScene_out, vector<int> &IdxNew)
{
    // ---- log ----
    if(Win<0)
        cout<<"error. win change for text delete should be positive, yet "<< Win<<endl;
    // ---- log ----

    vector<vector<Vec2>> TextDeteNew = GetNewDeteBox(TextDete, Win);
    SceneFeat_out = SceneFeat;
    vector<cv::KeyPoint>::iterator itS;
    int NumOK = 0, NumDele = 0, IdxRawOrder = 0;
    for(itS = SceneFeat_out.begin(); itS != SceneFeat_out.end();)
    {
        cv::KeyPoint fea = *itS;
        float FeaX = fea.pt.x;
        float FeaY = fea.pt.y;
        bool FLAG_TODELETE = false;

        for(size_t i0 = 0; i0<TextDete.size(); i0++)
        {
            vector<Vec2> TextDeteNewCell = TextDete[i0];
            vector<Point> TextDeteNewCellP = vV2vP(TextDeteNewCell);

            if(CheckPtsIn(TextDeteNewCellP, cv::Point2f(FeaX, FeaY)) )
            {
                // A) whinin the enlarged box -> scene feature is too close to the text box -> delete
                itS = SceneFeat_out.erase(itS);
                FLAG_TODELETE = true;
                NumDele++;
                break;
            }else{
                // B) out of the enlarged box -> this box is ok -> use next text object to check
                continue;
            }
        }

        if(!FLAG_TODELETE){
            if(NumOK == 0){
                DescScene_out = DescScene.row(IdxRawOrder).clone();
            }else{
                cv::vconcat(DescScene_out, DescScene.row(IdxRawOrder).clone(), DescScene_out);
            }
            ++itS;
            NumOK++;
            IdxNew.push_back(IdxRawOrder);
        }

        IdxRawOrder++;
    }

}


void tool::GetPyramidPts(const vector<KeyPoint> &vObvRaw, const Vec2 &PMin, const Vec2 &PMax, const vector<cv::Mat> &vImg, const vector<cv::Mat> &vImgGrad, const vector<double> &vInvScalefactor,
                         const vector<Mat33> &vK_scale, vector<vector<TextFeature*>> &vObv)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 1. param init
    int NPyramid = vInvScalefactor.size();
    int NFeatRaw = vObvRaw.size();
    int NAdd = 100;
    // CellIdx: all features (add mask) belongs to grid cell (from pyramid 1)
    vector<vector<vector<int>>> CellIdx;    // pyramid -> cell -> all feature belong to the cell
    CellIdx.resize(NPyramid);
    for(size_t j0 = 1; j0<NPyramid; j0++)
        CellIdx[j0].resize(NFeatRaw * vInvScalefactor[j0] * vInvScalefactor[j0] + NAdd);

    // Store the x/y number of each pyramid. Store x/y factor of each pyramid
    vector<int> PyrCellNumx, PyrCellNumy;
    vector<double> PyrCellFacx, PyrCellFacy;
    PyrCellNumx.resize(NPyramid);
    PyrCellNumy.resize(NPyramid);
    PyrCellFacx.resize(NPyramid);
    PyrCellFacy.resize(NPyramid);

    // store all feature in each pyramid (tmp). store all feature's gradient in each pyramid
    vector<vector<Vec2>> vSObs;
    vector<vector<double>> vSObsGrad;
    vSObs.resize(NPyramid);
    vSObsGrad.resize(NPyramid);

    // output
    vObv.resize(NPyramid);

    // 2. get pts in each pyramid
    // A) each pyramid
   for(size_t i0 = 1; i0<NPyramid; i0++){
       double invScalefactor = vInvScalefactor[i0];
       vector<vector<int>> CellIdxtmp = CellIdx[i0];

       double PMinx = PMin(0)*invScalefactor;
       double PMiny = PMin(1)*invScalefactor;
       double PMaxx = PMax(0)*invScalefactor;
       double PMaxy = PMax(1)*invScalefactor;
       double WH = (PMaxx-PMinx)/(PMaxy-PMiny);
       int CellHeight = sqrt(CellIdxtmp.size()/WH);
       int CellWeight = sqrt(CellIdxtmp.size()*WH);

       double Factorx = (double)(PMaxx-PMinx)/(double)CellWeight;
       double Factory = (double)(PMaxy-PMiny)/(double)CellHeight;
       PyrCellNumx[i0] = CellWeight;
       PyrCellNumy[i0] = CellHeight;
       PyrCellFacx[i0] = Factorx;
       PyrCellFacy[i0] = Factory;

       vector<Vec2> SObserPyr;
       vector<double> SObserGrad;
       // B) all features
       for(size_t i1 = 0; i1<vObvRaw.size(); i1++){
           KeyPoint PtObs = vObvRaw[i1];
           Vec2 PtObsScale = Vec2(PtObs.pt.x * invScalefactor, PtObs.pt.y * invScalefactor);
           double GradPix;
           GetIntenBilinterPtr(PtObsScale, vImgGrad[i0], GradPix);

           SObserGrad.push_back(GradPix);
           SObserPyr.push_back(PtObsScale);

           int m = round((PtObsScale(0)-PMinx)/Factorx);
           int n = round((PtObsScale(1)-PMiny)/Factory);

           if(m==CellWeight)
               m=CellWeight-1;
           if(n==CellHeight)
               n=CellHeight-1;

           CellIdxtmp[n*CellWeight+m].push_back(i1);
       }
       vSObsGrad[i0] = SObserGrad;
       vSObs[i0] = SObserPyr;          // pyramid feature
       CellIdx[i0] = CellIdxtmp;       // each grid cell store feature idx (in vObvRaw)

   }

   // attention: from 1 level
   double fx = vK_scale[0](0,0), fy = vK_scale[0](1,1), cx = vK_scale[0](0,2), cy = vK_scale[0](1,2);
   for(size_t j1 = 0; j1<vObvRaw.size(); j1++){
       KeyPoint ptObv = vObvRaw[j1];
       TextFeature* featurePyr0 = new TextFeature();
       featurePyr0->u = ptObv.pt.x;
       featurePyr0->v = ptObv.pt.y;
       featurePyr0->feature = Vec2(ptObv.pt.x, ptObv.pt.y);
       featurePyr0->level = (int)0;
       featurePyr0->IdxToRaw = (int)j1;
       featurePyr0->INITIAL = false;
       featurePyr0->ray = Mat31((ptObv.pt.x-cx)/fx, (ptObv.pt.y-cy)/fy, 1.0);

       featurePyr0->IN = GetIntenBilinterPtr(featurePyr0->feature, vImg[0], featurePyr0->featureInten);

       vObv[0].push_back(featurePyr0);
   }
   for(size_t i2 = 1; i2<NPyramid; i2++){
       vector<vector<int>> CellIdxtmp = CellIdx[i2];
       vector<double> SObserGradtmp = vSObsGrad[i2];
       vector<Vec2> vSObstmp = vSObs[i2];
       vector<TextFeature*> vObvtmp;
       double sfx = vK_scale[i2](0,0), sfy = vK_scale[i2](1,1), scx = vK_scale[i2](0,2), scy = vK_scale[i2](1,2);

       // in each grid, find highest gradient and store in idxInPyr ----
       vector<int> idxInPyr;
       for(int i3 = 0; i3<PyrCellNumx[i2]; i3++){
           for(int i4 = 0; i4<PyrCellNumy[i2]; i4++){
               int idx = i4*PyrCellNumx[i2]+i3;
               vector<int> CellIdxtmp1 = CellIdxtmp[idx];
               if(CellIdxtmp1.size()==0)
                   continue;

               double MAX = 0;
               int MAXIdx = -1;
               for(size_t i5 = 0; i5<CellIdxtmp1.size(); i5++){
                   int idxpix = CellIdxtmp1[i5];
                   if(SObserGradtmp[idxpix]>MAX){
                       MAXIdx = idxpix;
                   }
               }
               if(MAXIdx<0)
                   continue;

               // get max grad feature idx(MAXIdx) in (i3, i4) Cell grid
               TextFeature* featureInPyr = new TextFeature();
               featureInPyr->INITIAL = false;
               featureInPyr->u = vSObstmp[MAXIdx](0);
               featureInPyr->v = vSObstmp[MAXIdx](1);
               featureInPyr->feature = vSObstmp[MAXIdx];
               featureInPyr->level = (int)i2;
               featureInPyr->IdxToRaw = (int)MAXIdx;
               featureInPyr->ray = vObv[0][MAXIdx]->ray;
               featureInPyr->IN = GetIntenBilinterPtr(featureInPyr->feature, vImg[i2], featureInPyr->featureInten);
               vObvtmp.push_back(featureInPyr);
           }
       }
       // in each grid, find highest gradient and store in idxInPyr ----
       vObv[i2] = vObvtmp;
   }

   std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
   double tUse= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

}


/*
 * func: GetPyramidPts
 * get text features in each higher level pyramid using uniform sampling. Meanwhile get surrounding pixels/Inten/Idx of each feature
 * param In:
 * vector<Vec2> &vObvRaw: the extracted feature in level-0. const vector<cv::Mat> &vImg: the frame image to extract feature inten. const vector<double> &vInvScalefactor: pyramid info
 * param Out:
 * vector<TextFeature> &vObv: text feature in each higher level pramid
 * ----
 * return:
 * ----
 */
void tool::GetPyramidPts(const vector<KeyPoint> &vObvRaw, const vector<cv::Mat> &vImg, const vector<cv::Mat> &vImgGrad, const vector<double> &vInvScalefactor, vector<vector<TextFeature*>> &vObv)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // 1. param init
    int NPyramid = vInvScalefactor.size();
    int NFeatRaw = vObvRaw.size();
    int NAdd = 100;
    // CellIdx: all features belongs to grid cell (from pyramid 1)
    vector<vector<vector<int>>> CellIdx;    // pyramid -> cell -> all feature belong to the cell
    CellIdx.resize(NPyramid);
    for(size_t j0 = 1; j0<NPyramid; j0++)
        CellIdx[j0].resize(NFeatRaw * vInvScalefactor[j0] * vInvScalefactor[j0] + NAdd);

    // Store the x/y number of each pyramid. Store x/y factor of each pyramid
    vector<int> PyrCellNumx, PyrCellNumy;
    vector<double> PyrCellFacx, PyrCellFacy;
    PyrCellNumx.resize(NPyramid);
    PyrCellNumy.resize(NPyramid);
    PyrCellFacx.resize(NPyramid);
    PyrCellFacy.resize(NPyramid);

    // store all feature in each pyramid. store all feature's gradient in each pyramid
    vector<vector<Vec2>> vSObs;
    vector<vector<double>> vSObsGrad;
    vSObs.resize(NPyramid);
    vSObsGrad.resize(NPyramid);

    // output
    vObv.resize(NPyramid);

    // 2. get pts in each pyramid
    // A) each pyramid
    for(size_t i0 = 1; i0<NPyramid; i0++){
        double invScalefactor = vInvScalefactor[i0];
        vector<vector<int>> CellIdxtmp = CellIdx[i0];

        double WH = (double)vImgGrad[i0].cols/(double)vImgGrad[i0].rows;
        int CellHeight = sqrt(CellIdxtmp.size()/WH);
        int CellWeight = sqrt(CellIdxtmp.size()*WH);

        double Factorx = (double)vImgGrad[i0].cols/(double)CellWeight;
        double Factory = (double)vImgGrad[i0].rows/(double)CellHeight;
        PyrCellNumx[i0] = CellWeight;
        PyrCellNumy[i0] = CellHeight;
        PyrCellFacx[i0] = Factorx;
        PyrCellFacy[i0] = Factory;

        vector<Vec2> SObserPyr;
        vector<double> SObserGrad;
        // B) all features
        for(size_t i1 = 0; i1<vObvRaw.size(); i1++){
            KeyPoint PtObs = vObvRaw[i1];
            Vec2 PtObsScale = Vec2(PtObs.pt.x * invScalefactor, PtObs.pt.y * invScalefactor);
            double GradPix;
            GetIntenBilinterPtr(PtObsScale, vImgGrad[i0], GradPix);

            SObserGrad.push_back(GradPix);
            SObserPyr.push_back(PtObsScale);

            int m = round(PtObsScale(0)/Factorx);
            int n = round(PtObsScale(1)/Factory);
            if(m==CellWeight)
                m=CellWeight-1;
            if(n==CellHeight)
                n=CellHeight-1;
            CellIdxtmp[n*CellWeight+m].push_back(i1);
        }
        vSObsGrad[i0] = SObserGrad;
        vSObs[i0] = SObserPyr;          // pyramid feature
        CellIdx[i0] = CellIdxtmp;       // each grid cell store feature idx (in vObvRaw)
    }

    for(size_t j1 = 0; j1<vObvRaw.size(); j1++){
        KeyPoint ptObv = vObvRaw[j1];
        TextFeature* featurePyr0 = new TextFeature();
        featurePyr0->u = ptObv.pt.x;
        featurePyr0->v = ptObv.pt.y;
        featurePyr0->feature = Vec2(ptObv.pt.x, ptObv.pt.y);
        featurePyr0->level = (int)0;
        featurePyr0->IdxToRaw = (int)j1;

        vObv[0].push_back(featurePyr0);
    }
    for(size_t i2 = 1; i2<NPyramid; i2++){
        vector<vector<int>> CellIdxtmp = CellIdx[i2];
        vector<double> SObserGradtmp = vSObsGrad[i2];
        vector<Vec2> vSObstmp = vSObs[i2];
        vector<TextFeature*> vObvtmp;

        // in each grid, find highest gradient and store in idxInPyr ----
        vector<int> idxInPyr;
        for(int i3 = 0; i3<PyrCellNumx[i2]; i3++){
            for(int i4 = 0; i4<PyrCellNumy[i2]; i4++){
                int idx = i4*PyrCellNumx[i2]+i3;
                vector<int> CellIdxtmp1 = CellIdxtmp[idx];
                if(CellIdxtmp1.size()==0)
                    continue;
                double MAX = 0;
                int MAXIdx = -1;
                for(size_t i5 = 0; i5<CellIdxtmp1.size(); i5++){
                    int idxpix = CellIdxtmp1[i5];
                    if(SObserGradtmp[idxpix]>MAX){
                        MAXIdx = idxpix;
                    }
                }
                if(MAXIdx<0)
                    continue;

                // get max grad feature idx(MAXIdx) in (i3, i4) Cell grid
                TextFeature* featureInPyr = new TextFeature();
                featureInPyr->u = vSObstmp[MAXIdx](0);
                featureInPyr->v = vSObstmp[MAXIdx](1);
                featureInPyr->feature = vSObstmp[MAXIdx];
                featureInPyr->level = (int)i2;
                featureInPyr->IdxToRaw = (int)MAXIdx;
                vObvtmp.push_back(featureInPyr);
            }
        }
        // in each grid, find highest gradient and store in idxInPyr ----
        vObv[i2] = vObvtmp;
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double tUse= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

}

/*
 * func: GetPyramidPts (scene)
 * get scene features in each higher level pyramid using uniform sampling. Meanwhile get Idx of each feature (related to 3D MapPts)
 * param In:
 * vector<Vec2> &vObvRaw: the extracted feature in level-0. const vector<double> &vInvScalefactor: pyramid info
 * param Out:
 * vector<SceneFeature> &vObv: scene feature in each higher level pramid
 * ----
 * return:
 * ----
 */
void tool::GetPyramidPts(const vector<Vec2> &vObvRaw, const vector<cv::Mat> &vImg, const vector<cv::Mat> &vImgGrad, const vector<double> &vInvScalefactor, vector<vector<SceneFeature *> > &vObv)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    bool DEBUG = false;
    bool SHOW = false;

    // 1. param init
    int NPyramid = vInvScalefactor.size();
    int NFeatRaw = vObvRaw.size();
    int NAdd = 500;
    // CellIdx: all features belongs to grid cell (from pyramid 1)
    vector<vector<vector<int>>> CellIdx;    // pyramid -> cell -> all feature belong to the cell
    CellIdx.resize(NPyramid);
    for(size_t j0 = 1; j0<NPyramid; j0++)
        CellIdx[j0].resize(NFeatRaw * vInvScalefactor[j0] * vInvScalefactor[j0] + NAdd);

    // Store the x/y number of each pyramid. Store x/y factor of each pyramid
    vector<int> PyrCellNumx, PyrCellNumy;
    vector<double> PyrCellFacx, PyrCellFacy;
    PyrCellNumx.resize(NPyramid);
    PyrCellNumy.resize(NPyramid);
    PyrCellFacx.resize(NPyramid);
    PyrCellFacy.resize(NPyramid);

    // store all feature in each pyramid (tmp). store all feature's gradient in each pyramid
    vector<vector<Vec2>> vSObs;
    vector<vector<double>> vSObsGrad;
    vSObs.resize(NPyramid);
    vSObsGrad.resize(NPyramid);

    // output
    vObv.resize(NPyramid);

    // 2. get pts in each pyramid
    // A) each pyramid
    for(size_t i0 = 1; i0<NPyramid; i0++){
        double invScalefactor = vInvScalefactor[i0];
        vector<vector<int>> CellIdxtmp = CellIdx[i0];
        double WH = (double)vImgGrad[i0].cols/(double)vImgGrad[i0].rows;
        int CellHeight = sqrt(CellIdxtmp.size()/WH);
        int CellWeight = sqrt(CellIdxtmp.size()*WH);
        double Factorx = (double)vImgGrad[i0].cols/(double)CellWeight;
        double Factory = (double)vImgGrad[i0].rows/(double)CellHeight;
        PyrCellNumx[i0] = CellWeight;
        PyrCellNumy[i0] = CellHeight;
        PyrCellFacx[i0] = Factorx;
        PyrCellFacy[i0] = Factory;

        vector<Vec2> SObserPyr;
        vector<double> SObserGrad;
        // B) all features
        for(size_t i1 = 0; i1<vObvRaw.size(); i1++){
            Vec2 PtObs = vObvRaw[i1];
            Vec2 PtObsScale = Vec2(PtObs(0) * invScalefactor, PtObs(1) * invScalefactor);
            double GradPix;
            GetIntenBilinterPtr(PtObsScale, vImgGrad[i0], GradPix);

            SObserGrad.push_back(GradPix);
            SObserPyr.push_back(PtObsScale);

            int m = round(PtObsScale(0)/Factorx);
            int n = round(PtObsScale(1)/Factory);
            if(m==CellWeight)
                m=CellWeight-1;
            if(n==CellHeight)
                n=CellHeight-1;
            CellIdxtmp[n*CellWeight+m].push_back(i1);
        }
        vSObsGrad[i0] = SObserGrad;
        vSObs[i0] = SObserPyr;          // pyramid feature
        CellIdx[i0] = CellIdxtmp;       // each grid cell store feature idx (in vObvRaw)

    }

    // attention: from 1 level
    for(size_t j1 = 0; j1<vObvRaw.size(); j1++){
        Vec2 ptObv = vObvRaw[j1];
        // SceneFeature featurePyr0 = {ptObv(0), ptObv(1), ptObv, (int)0, (int)j1};
        SceneFeature* featurePyr0 = new SceneFeature{ptObv(0), ptObv(1), ptObv, (int)0, (int)j1};
        vObv[0].push_back(featurePyr0);
    }
    for(size_t i2 = 1; i2<NPyramid; i2++){
        vector<vector<int>> CellIdxtmp = CellIdx[i2];
        vector<double> SObserGradtmp = vSObsGrad[i2];
        vector<Vec2> vSObstmp = vSObs[i2];
        vector<SceneFeature*> vObvtmp;

        // in each grid, find highest gradient and store in idxInPyr ----
        vector<int> idxInPyr;
        for(int i3 = 0; i3<PyrCellNumx[i2]; i3++){
            for(int i4 = 0; i4<PyrCellNumy[i2]; i4++){
                int idx = i4*PyrCellNumx[i2]+i3;
                vector<int> CellIdxtmp1 = CellIdxtmp[idx];
                if(CellIdxtmp1.size()==0)
                    continue;

                double MAX = 0;
                int MAXIdx = -1;
                for(size_t i5 = 0; i5<CellIdxtmp1.size(); i5++){
                    int idxpix = CellIdxtmp1[i5];
                    if(SObserGradtmp[idxpix]>=MAX){
                        MAXIdx = idxpix;
                    }
                }
                // get max grad feature idx(MAXIdx) in (i3, i4) Cell grid
                if(MAXIdx<0)
                    continue;
                SceneFeature* featureInPyr = new SceneFeature{vSObstmp[MAXIdx](0), vSObstmp[MAXIdx](1), vSObstmp[MAXIdx], (int)i2, MAXIdx};
                vObvtmp.push_back(featureInPyr);
            }
        }
        // in each grid, find highest gradient and store in idxInPyr ----
        vObv[i2] = vObvtmp;
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double tUse= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

}

// ------------ Feature extractor ------------


// ------------ Basic operator ------------
vector<vector<Vec2>> tool::GetNewDeteBox(const vector<vector<Vec2>> &DeteRaw, const float &Win)
{
    // ________ x
    // |  1 -- 2
    // |  4 -- 3
    // y
    // Win: negative-> box smaller; positive-> box larger
    vector<vector<Vec2>> NewBox;
    for(size_t i0 = 0; i0<DeteRaw.size(); i0++){
        vector<Vec2> Box4Pts = DeteRaw[i0];
        vector<Vec2> NewBox4Pts;
        Vec2 pt1(Box4Pts[0](0)-Win, Box4Pts[0](1)-Win);
        Vec2 pt2(Box4Pts[1](0)+Win, Box4Pts[1](1)-Win);
        Vec2 pt3(Box4Pts[2](0)+Win, Box4Pts[2](1)+Win);
        Vec2 pt4(Box4Pts[3](0)-Win, Box4Pts[3](1)+Win);

        NewBox4Pts.push_back(pt1);
        NewBox4Pts.push_back(pt2);
        NewBox4Pts.push_back(pt3);
        NewBox4Pts.push_back(pt4);
        NewBox.push_back(NewBox4Pts);
        NewBox4Pts.clear();
    }

    return NewBox;
}

vector<vector<Vec2>> tool::GetbbDeteBox(const vector<Vec2> &vTextDeteMin, const vector<Vec2> &vTextDeteMax)
{
    vector<vector<Vec2>> NewBox;
    assert(vTextDeteMin.size()==vTextDeteMax.size());
    for(size_t i0=0; i0<vTextDeteMin.size(); i0++){
        vector<Vec2> bbox;
        bbox.push_back(Vec2(vTextDeteMin[i0](0), vTextDeteMin[i0](1)));
        bbox.push_back(Vec2(vTextDeteMax[i0](0), vTextDeteMin[i0](1)));
        bbox.push_back(Vec2(vTextDeteMax[i0](0), vTextDeteMax[i0](1)));
        bbox.push_back(Vec2(vTextDeteMin[i0](0), vTextDeteMax[i0](1)));
        NewBox.push_back(bbox);
    }

    return NewBox;
}

vector<Point> tool::vV2vP(const vector<Vec2> &In)
{
    vector<Point> Out;
    for(size_t i0 = 0; i0<In.size(); i0++){
        Out.push_back(cv::Point(In[i0](0), In[i0](1)));
    }
    return Out;
}

vector<KeyPoint> tool::vV2vK(const vector<Vec2> &In)
{
    vector<KeyPoint> Out;
    for(size_t i0 = 0; i0<In.size(); i0++){
        cv::KeyPoint tmp;
        tmp.pt = cv::Point2d( In[i0](0), In[i0](1) );
        Out.push_back(tmp);
    }
    return Out;
}

KeyPoint tool::V2K(const Vec2 &In)
{
    cv::KeyPoint out;
    out.pt = cv::Point2d( In(0), In(1) );
    return out;
}


void tool::GetRayRho(const vector<cv::Point2f> &In, const vector<double> &In2, const Mat33 &K, vector<Vec3> &Out)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    for(size_t i=0; i<In.size(); i++){
        Vec3 ray((In[i].x-cx)/fx, (In[i].y-cy)/fy, In2[i]);
        Out.push_back(ray);
    }
}

Eigen::Quaterniond tool::R2q(const Mat33 &R)
{
    Eigen::Quaterniond q(R);
    q = q.normalized();
    return q;
}

std::map<keyframe*, int> tool::Convert2Map(const vector<CovKF> &vKFs)
{
    std::map<keyframe*, int> vKFsOut;
    for(size_t i0=0; i0<vKFs.size(); i0++){
        keyframe* KF = vKFs[i0].first;
        vKFsOut.insert(make_pair(KF, vKFs[i0].second));
    }
    return vKFsOut;
}

set<keyframe*> tool::Convert2Set(const vector<CovKF> &vKFs)
{
    set<keyframe*> vKFsOut;
    for(size_t i0=0; i0<vKFs.size(); i0++){
        keyframe* KF = vKFs[i0].first;
        vKFsOut.insert(KF);
    }
    return vKFsOut;
}

/*
 * func: tool::CheckPtsIn
 * param In:
 * vector<Point> &hull: hull region; cv::Point2f &pts:  point to judge
 * ----
 * return:
 * flag in or out
 */
bool tool::CheckPtsIn(const vector<Point> &hull, const cv::Point2f &pts)
{
    double res = cv::pointPolygonTest( hull, pts, true );
    if(res>0)
        return true;
    else
        return false;
}

vector<size_t> tool::InitialVec(const int &N)
{
    vector<size_t> Set;
    for(size_t i0 = 0; i0<N; i0++){
        Set.push_back(i0);
    }
    return Set;
}

bool tool::GetIntenBilinter(const Vec2 &pts, const Mat &Img, double &Inten)
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

bool tool::GetIntenBilinterPtr(const Vec2 &pts, const Mat &Img, double &Inten)
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

        return true;
    }
}

bool tool::CalTextinfo(const cv::Mat &Img, const vector<Vec2> &vTextDete, double &mu, double &std, const bool &SHOW)
{
    // 1. get text object in the ref frame (float)
    cv::Point TextObjBox[4];
    int xMin = Img.cols+1, xMax = -1, yMin = Img.rows+1, yMax = -1;
    for(size_t i = 0; i<vTextDete.size(); i++){
        TextObjBox[i] = cv::Point(vTextDete[i](0), vTextDete[i](1));
        if(vTextDete[i](0)>xMax)
           xMax = std::ceil(vTextDete[i](0));
        if(vTextDete[i](0)<xMin)
           xMin = std::floor(vTextDete[i](0));
        if(vTextDete[i](1)>yMax)
           yMax = std::ceil(vTextDete[i](1));
        if(vTextDete[i](1)<yMin)
           yMin = std::floor(vTextDete[i](1));
    }

    if(xMin<0)
        xMin = 0;
    if(xMin>=Img.cols)
        xMin = Img.cols-1;
    if(yMin<0)
        yMin = 0;
    if(yMin>=Img.rows)
        yMin = Img.rows-1;
    if(xMax>=Img.cols)
        xMax = Img.cols-1;
    if(xMax<0)
        xMax = 0;
    if(yMax>=Img.rows)
        yMax = Img.rows-1;
    if(yMax<0)
        yMax = 0;

    cv::Mat TemplateImg = cv::Mat::zeros(Img.size(), CV_32F);
    const cv::Point* ptMask[1] = {TextObjBox};
    int npt[] = {4};
    cv::fillPoly(TemplateImg, ptMask, npt, 1, cv::Scalar(-1,-1,-1));

    vector<double> TextObjInten;
    for(size_t iRow = yMin; iRow<=yMax; iRow++){
        for(size_t iCol = xMin; iCol<=xMax; iCol++){
            if(iRow<0 || iRow>=Img.rows || iCol<0 || iCol>=Img.cols)
                continue;

            float* img_ptr_template = (float*)TemplateImg.data + iRow*TemplateImg.cols + iCol;
            if(img_ptr_template[0]>=0)
                continue;

            uint8_t* img_ptr = (uint8_t*)Img.data + iRow*Img.cols + iCol;
            double TextPixinten = img_ptr[0];

            TextObjInten.push_back(TextPixinten);
        }
    }

    //2. calculate statistics of text object
    if(!CalStatistics(TextObjInten, mu, std)){
       return false;
    }

    return true;
}


// calculate mu & std of vTextInten
bool tool::CalStatistics(const vector<double> &vTextInten, double &mu, double &std)
{
    if(vTextInten.size()==0){
        return false;
    }
    double sum = std::accumulate(std::begin(vTextInten), std::end(vTextInten), 0.0);
    mu = sum/(double)(vTextInten.size());

    double sum_square = 0.0;
    for(size_t i0 = 0; i0<vTextInten.size(); i0++){
        sum_square += (vTextInten[i0]-mu) * (vTextInten[i0]-mu);
    }
    std = std::sqrt( sum_square/(double)(vTextInten.size()-1) );

    if(std!=0)
        return true;
    else
        return false;
}

void tool::GetBoxAllPixs(const cv::Mat &Img, const vector<Vec2> &vTextDete, const int &Pym, const Mat33 &K, const double &mu, const double &std, vector<TextFeature*> &vPixs)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    bool FLAG_TEXTINIMAGE = true;

    cv::Point TextObjBox[4];
    int xMin = Img.cols+1, xMax = -1, yMin = Img.rows+1, yMax = -1;
    for(size_t i = 0; i<vTextDete.size(); i++){
        TextObjBox[i] = cv::Point(vTextDete[i](0), vTextDete[i](1));
        if(vTextDete[i](0)>xMax)
           xMax = std::ceil(vTextDete[i](0));
        if(vTextDete[i](0)<xMin)
           xMin = std::floor(vTextDete[i](0));
        if(vTextDete[i](1)>yMax)
           yMax = std::ceil(vTextDete[i](1));
        if(vTextDete[i](1)<yMin)
           yMin = std::floor(vTextDete[i](1));
    }

    if(xMin<0)
        xMin = 0;
    if(xMin>=Img.cols)
        xMin = Img.cols-1;
    if(yMin<0)
        yMin = 0;
    if(yMin>=Img.rows)
        yMin = Img.rows-1;
    if(xMax>=Img.cols)
        xMax = Img.cols-1;
    if(xMax<0)
        xMax = 0;
    if(yMax>=Img.rows)
        yMax = Img.rows-1;
    if(yMax<0)
        yMax = 0;

    cv::Mat TemplateImg = cv::Mat::zeros(Img.size(), CV_32F);
    const cv::Point* ptMask[1] = {TextObjBox};
    int npt[] = {4};
    cv::fillPoly(TemplateImg, ptMask, npt, 1, cv::Scalar(-1,-1,-1));

    int idx_pix=0;
    for(size_t iRow = yMin; iRow<=yMax; iRow++){
        for(size_t iCol = xMin; iCol<=xMax; iCol++){
            if(iRow<0 || iRow>=Img.rows || iCol<0 || iCol>=Img.cols){
                FLAG_TEXTINIMAGE = false;
                continue;
            }
            float* img_ptr_template = (float*)TemplateImg.data + iRow*TemplateImg.cols + iCol;
            if(img_ptr_template[0]>=0)
                continue;

            uint8_t* img_ptr = (uint8_t*)Img.data + iRow*Img.cols + iCol;
            double TextPixinten = img_ptr[0];
            Vec2 TextPix = Vec2(iCol, iRow);
            FLAG_TEXTINIMAGE = true;

            // new feature
            TextFeature* featurePyr0 = new TextFeature();
            featurePyr0->u = TextPix(0,0);
            featurePyr0->v = TextPix(1,0);
            featurePyr0->feature = TextPix;
            featurePyr0->level = (int)0;
            featurePyr0->IdxToRaw = (int)idx_pix;
            featurePyr0->INITIAL = false;
            featurePyr0->ray = Mat31((TextPix(0,0)-cx)/fx, (TextPix(1,0)-cy)/fy, 1.0);
            featurePyr0->IN = FLAG_TEXTINIMAGE;
            featurePyr0->featureInten = TextPixinten;
            featurePyr0->featureNInten = (TextPixinten-mu)/std;
            vPixs.push_back(featurePyr0);
            idx_pix++;
        }
    }

}



bool tool::CalNormvec(const cv::Mat &Img, vector<TextFeature*> &Vecin, const double &mu, const double &std, const Mat33 &K)
{
    if(std==0){
        return false;
    }

    for(size_t i0 = 0; i0<Vecin.size(); i0++){
        GetNeighbour(Vecin[i0], mu, std, Img, K, INTERVAL8);                      // get neighbour pixel and its inten INTERVAL8
        Vecin[i0]->featureNInten = (Vecin[i0]->featureInten-mu)/std;
        Vecin[i0]->INITIAL = true;
    }

    return true;
}

cv::Point2f tool::PixToCam(const Mat33 &K, const cv::Point2f p)
{
    return cv::Point2f
            (
                ( p.x - K(0,2) )/ K(0,0),
                ( p.y - K(1,2) )/ K(1,1)
                );
}

bool tool::GetRANSACIdx(const int &MaxIterations, const int &SelectNum, const int &number, const bool &TEXT, vector<vector<size_t>> &IdxOut)
{
    if(TEXT){
        if(number<3)
            return false;
    }

    IdxOut = vector<vector<size_t>>(MaxIterations, vector<size_t>(SelectNum,0));

    DUtils::Random::SeedRandOnce(0);
    for(int it=0; it<MaxIterations; it++){
        vector<size_t> vAvailableIndices = InitialVec(number);
        for(size_t j=0; j<SelectNum; j++){
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            IdxOut[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    return true;
}


bool tool::CheckOrientation(const Mat31 &theta, const Mat44 &Tcr, const frame &F, const double &threshcos)
{
    Mat33 Rrc = Tcr.block<3,3>(0,0).transpose();
    Mat31 CamOref = Rrc.block<3,1>(0,2);

    // cal cos
    double normVal = theta.norm() * CamOref.norm();
    Mat11 CosVal = theta.transpose() * CamOref / normVal;

    if(std::fabs(CosVal(0,0))<threshcos)
    {
        return false;
    }
    return true;
}



bool tool::CheckZNCC(const vector<TextFeature*> &vAllpixs, const cv::Mat &ImgCur, const cv::Mat &ImgRef, const Mat44 &Tcr, const Mat31 &theta, const Mat33 &K, const double &thresh)
{
    double ZNCC=-100.0;

    vector<double> vIntenRef, vIntenCur;
    vector<double> vIntenRefN, vIntenCurN;
    cv::Mat ImgRef1 = ImgRef.clone();
    cv::Mat ImgCur2 = ImgCur.clone();

    for(size_t i0=0; i0<vAllpixs.size(); i0++){
        TextFeature* feat = vAllpixs[i0];
        Vec2 pred;
        double IntenCur;
        bool POSITIVED_PIX = GetProjText(feat->ray, theta, pred, Tcr, K);
        bool PIX_IN = GetIntenBilinterPtr(pred, ImgCur, IntenCur);

        vIntenRef.push_back(feat->featureInten);
        vIntenCur.push_back(IntenCur);
    }

    bool vOK1 = VectorNorm(vIntenRef, vIntenRefN);
    bool vOK2 = VectorNorm(vIntenCur, vIntenCurN);
    if(vOK1 && vOK2){
        ZNCC = CalZNCC(vIntenRefN, vIntenCurN);
    }

    if(ZNCC>=thresh)
        return true;
    else{
        return false;
    }
}


bool tool::VectorNorm(const vector<double> &vIn, vector<double> &vOut)
{
    assert(vIn.size()!=0);

    double sum = std::accumulate(std::begin(vIn), std::end(vIn), 0.0);
    double av = sum/(double)(vIn.size());
    double sum_sigma2  = 0.0;
    for(int i0 = 0; i0<vIn.size(); i0++)
        sum_sigma2 += (vIn[i0]-av) * (vIn[i0]-av);
    double std = std::sqrt( sum_sigma2/(double)(vIn.size()-1) );
    if(std!=0){
        for(size_t i1=0; i1<vIn.size(); i1++){
            double v_normed = (vIn[i1] - av)/std;
            vOut.push_back(v_normed);
        }
        return true;
    }else
        return false;

}


double tool::CalZNCC(const vector<double> &v1, const vector<double> &v2)
{
    assert(v1.size() == v2.size());

    double sum = 0.0, sumTop1 = 0.0, sumTop2 = 0.0;
    double thTop1 = 0.90, thTop2 = 0.80;
    double AvZNCC, AvZNCCTop1, AvZNCCTop2;
    vector<double> valres;

    // v1*v2
    for(size_t i0 = 0; i0<v1.size(); i0++){
        double val = v1[i0] * v2[i0];
        valres.push_back(val);
        sum += val;
    }

    // sum(res)/N
    sort(valres.begin(), valres.end());
    int numTop1 = std::round((double)v1.size()*thTop1);
    int numTop2 = std::round((double)v1.size()*thTop2);
    for(size_t i_top = v1.size()-1; i_top>= v1.size()-numTop1; i_top--){
        sumTop1 += valres[i_top];
    }
    for(size_t i_top = v1.size()-1; i_top>=v1.size()-numTop2; i_top--){
        sumTop2 += valres[i_top];
    }
    AvZNCCTop1 = sumTop1/((double)numTop1);
    AvZNCCTop2 = sumTop2/((double)numTop2);
    AvZNCC = sum/((double)v1.size());

    return AvZNCC;
}

// Nr*Pr => Nr*Trw * Twr*Pr => Nw * Pw
Mat31 tool::TransTheta(const Mat31 &Theta_r, const Mat44 &Trw)
{
    double dr = 1.0/Theta_r.norm();
    Eigen::Matrix<double,3,1> nr = Theta_r/Theta_r.norm();
    Eigen::Matrix<double,4,1> Nr4;
    Nr4.block<3,1>(0,0) = nr;
    Nr4(3,0) = dr;

    Eigen::Matrix<double,1,4> Nw4 = Nr4.transpose()*Trw;
    Mat31 Nw = Nw4.block<1,3>(0,0).transpose() / Nw4(0,3);
    return Nw;
}

// vKFsIn must has been sorted from large to small
vector<CovKF> tool::GetAllNonZero(const vector<CovKF> &vKFsIn)
{
    vector<CovKF> vKFsOut;
    for(size_t i1=0; i1<vKFsIn.size(); i1++){
        if(vKFsIn[i1].second<=0)
            break;
        vKFsOut.push_back(vKFsIn[i1]);
    }
    return vKFsOut;
}

Vec2 tool::GetMean(const vector<Vec2> &Pred)
{
    double sumx=0, sumy=0;
    for(size_t i=0; i<Pred.size(); i++){
        sumx += Pred[i](0,0);
        sumy += Pred[i](1,0);
    }

    Vec2 Out( sumx/Pred.size(), sumy/Pred.size() );

    return Out;
}

// get block neighbour for one feature
void tool::GetNeighbour(TextFeature* feat, const double &mu, const double &std, const Mat &Img, const Mat33 &K, const neighbour &NEIGH)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    vector<Vec2> vpt;
    int WinHalf;
    cv::Mat image = Img.clone();

    switch(NEIGH)
    {
    case INTERVAL8:
        vpt.push_back(Vec2(feat->u, feat->v));
        vpt.push_back(Vec2(feat->u+2.0, feat->v));
        vpt.push_back(Vec2(feat->u+1.0, feat->v-1.0));
        vpt.push_back(Vec2(feat->u, feat->v-2.0));
        vpt.push_back(Vec2(feat->u-1.0, feat->v-1.0));
        vpt.push_back(Vec2(feat->u-2.0, feat->v));
        vpt.push_back(Vec2(feat->u-1.0, feat->v+1.0));
        vpt.push_back(Vec2(feat->u, feat->v+2.0));
        for(size_t inei=0; inei<vpt.size(); inei++){
            // 1. get pixel
            feat->neighbour.push_back(vpt[inei]);
            feat->neighbourRay.push_back(Mat31( (vpt[inei](0)-cx)/fx, (vpt[inei](1)-cy)/fy, 1.0 ));
            // 2. get inten, normed inten, flag IN
            double inten;
            feat->IN = GetIntenBilinterPtr(vpt[inei],Img,inten);
            feat->neighbourInten.push_back(inten);
            feat->neighbourNInten.push_back( ((inten-mu)/std) );
        }

        break;
    case BLOCK9:

        break;
    case BLOCK25:
        WinHalf = 2;
        for(int iRow = -WinHalf; iRow<=WinHalf; iRow++){
            for(int iCol = -WinHalf; iCol<=WinHalf; iCol++){
                // 1. get pixel
                Vec2 pt(feat->u + iCol, feat->v + iRow);
                feat->neighbour.push_back(pt);
                feat->neighbourRay.push_back(Mat31( (pt(0)-cx)/fx, (pt(1)-cy)/fy, 1.0 ));
                // 2. get inten, normed inten, flag IN
                double inten;
                feat->IN = GetIntenBilinterPtr(pt,Img,inten);
                feat->neighbourInten.push_back(inten);
                feat->neighbourNInten.push_back( ((inten-mu)/std) );
            }
        }
        break;
    }

}

bool tool::GetProjText(const Mat31 &ray, const Mat31 &theta, Vec2 &pred, const Mat44 &T, const Mat33 &K)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    double invz = -ray.transpose() * theta;
    Mat31 p = T.block<3,3>(0,0) * ray/invz + T.block<3,1>(0,3);

    double u = fx * p(0,0)/p(2,0) + cx;
    double v = fy * p(1,0)/p(2,0) + cy;
    pred = Vec2(u, v);

    if(p(2,0)<0)
        return false;
    return true;
}

bool tool::GetProjText(const Vec2 &_ray, const Mat31 &theta, Vec2 &pred, const Mat44 &Tcr, const Mat33 &K)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    Mat31 ray(_ray(0), _ray(1), 1.0);
    double invz = -ray.transpose() * theta;
    Mat31 p = Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3);

    double u = fx * p(0,0)/p(2,0) + cx;
    double v = fy * p(1,0)/p(2,0) + cy;
    pred = Vec2(u, v);

    if(p(2,0)<0)
        return false;
    else
        return true;
}

bool tool::GetPtsText(const Vec2 &_ray, const Mat31 &theta, Mat31 &pw, const Mat44 &Tcr)
{
    Mat31 ray(_ray(0), _ray(1), 1.0);
    double invz = -ray.transpose() * theta;
    pw = Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3);

    if(pw(2,0)<0)
        return false;
    else
        return true;
}



// used for initial optimization, reference T = [I,0]. thetacr = thetacw, posecrcr = posecw
void tool::GetProjText(const Vec2 &_ray, const double* thetacr, Vec2 &pred, const double* posecr, const Mat33 &K)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    Eigen::Quaterniond qcr(posecr[0], posecr[1], posecr[2], posecr[3]);
    qcr = qcr.normalized();
    Mat31 tcr(posecr[4], posecr[5], posecr[6]);
    Mat33 Rcr(qcr);
    Mat31 ray(_ray(0), _ray(1), 1.0);
    Mat31 theta(thetacr[0], thetacr[1], thetacr[2]);

    double invz = -ray.transpose() * theta;
    Mat31 p = Rcr * ray/invz + tcr;

    double u = fx * p(0,0)/p(2,0) + cx;
    double v = fy * p(1,0)/p(2,0) + cy;
    pred = Vec2(u, v);

}

bool tool::GetProjText(const Vec2 &_ray, const Mat31 thetacr, Vec2 &pred, const double* posecw, const Mat44 &Twr, const Mat33 &K)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    Eigen::Quaterniond qcw(posecw[0], posecw[1], posecw[2], posecw[3]);
    qcw = qcw.normalized();
    Mat31 tcw(posecw[4], posecw[5], posecw[6]);
    Mat33 Rcw(qcw);
    Mat44 Tcw, Tcr;
    Tcw.setIdentity();
    Tcw.block<3,3>(0,0) = Rcw;
    Tcw.block<3,1>(0,3) = tcw;
    Tcr = Tcw * Twr;

    Mat31 ray(_ray(0), _ray(1), 1.0);
    Mat31 theta(thetacr(0,0), thetacr(1,0), thetacr(2,0));

    double invz = -ray.transpose() * theta;
    Mat31 p = Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3);

    double u = fx * p(0,0)/p(2,0) + cx;
    double v = fy * p(1,0)/p(2,0) + cy;
    pred = Vec2(u, v);

    if(p(2,0)<0)
        return false;
    else
        return true;
}

bool tool::GetProjText(const Vec2 &_ray, const double* thetacr, Vec2 &pred, const double* posecw, const double* poserw, const Mat33 &K)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    Eigen::Quaterniond qcw(posecw[0], posecw[1], posecw[2], posecw[3]);
    qcw = qcw.normalized();
    Mat31 tcw(posecw[4], posecw[5], posecw[6]);
    Mat33 Rcw(qcw);

    Eigen::Quaterniond qrw(poserw[0], poserw[1], poserw[2], poserw[3]);
    qrw = qrw.normalized();
    Mat31 trw(poserw[4], poserw[5], poserw[6]);
    Mat33 Rrw(qrw);

    Mat44 Tcw, Trw, Tcr;
    Tcw.setIdentity();
    Tcw.block<3,3>(0,0) = Rcw;
    Tcw.block<3,1>(0,3) = tcw;
    Trw.setIdentity();
    Trw.block<3,3>(0,0) = Rrw;
    Trw.block<3,1>(0,3) = trw;
    Tcr = Tcw * Trw.inverse();

    Mat31 ray(_ray(0), _ray(1), 1.0);
    Mat31 theta(thetacr[0], thetacr[1], thetacr[2]);

    double invz = -ray.transpose() * theta;
    Mat31 p = Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3);

    double u = fx * p(0,0)/p(2,0) + cx;
    double v = fy * p(1,0)/p(2,0) + cy;
    pred = Vec2(u, v);

    if(p(2,0)<0)
        return false;
    else
        return true;
}

void tool::GetProjText(const Vec2 &_ray, const double* thetacr, Vec2 &pred, const Mat44 &Tcr,  const Mat33 &K)
{
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);
    Mat31 ray(_ray(0), _ray(1), 1.0);
    Mat31 theta(thetacr[0], thetacr[1], thetacr[2]);

    double invz = -ray.transpose() * theta;
    Mat31 p = Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3);

    double u = fx * p(0,0)/p(2,0) + cx;
    double v = fy * p(1,0)/p(2,0) + cy;
    pred = Vec2(u, v);
}

// ------------ Basic operator ------------

//  ------------ visualization ------------
void tool::ShowMatches(const vector<int> &Match12, const cv::Mat &F1Img, const cv::Mat &F2Img, const vector<cv::KeyPoint> &F1Keys, const vector<cv::KeyPoint> &F2Keys,
                       const bool &SHOW, const bool &SAVE, const string &SaveName)
{
    vector<cv::DMatch> match_show;
    int num = 0;
    for (int i = 0; i<Match12.size(); i++)
    {
        if(Match12[i]<0)
            continue;

        int index_show = Match12[i];

        cv::DMatch match_tmp;
        match_tmp.imgIdx = 0;
        match_tmp.queryIdx = i;
        match_tmp.trainIdx = index_show;
        match_show.push_back(match_tmp);
        num++;
    }

    Mat img_match;
    cout<<"show match num is: "<<num<<endl;
    drawMatches(F1Img, F1Keys, F2Img, F2Keys, match_show, img_match);
    if(SAVE)
        imwrite(SaveName, img_match);

    if(SHOW){
        namedWindow("Matches", CV_WINDOW_NORMAL);
        imshow("Matches", img_match);
        waitKey(0);
    }
}

void tool::ShowMatches(const vector<int> &Match12, const Mat &F1Img, const Mat &F2Img, const vector<KeyPoint> &F1Keys, const vector<KeyPoint> &F2Keys, const bool &SHOW)
{
    if(SHOW){
        vector<cv::DMatch> match_show;
        int num = 0;
        for (int i = 0; i<Match12.size(); i++)
        {
            if(Match12[i]<0)
                continue;

            int index_show = Match12[i];

            cv::DMatch match_tmp;
            match_tmp.imgIdx = 0;
            match_tmp.queryIdx = i;
            match_tmp.trainIdx = index_show;
            match_show.push_back(match_tmp);
            num++;
        }

        Mat img_match;
        cout<<"show match num is: "<<num<<endl;
        namedWindow("Matches", CV_WINDOW_NORMAL);
        drawMatches(F1Img, F1Keys, F2Img, F2Keys, match_show, img_match);
        imshow("Matches", img_match);
        waitKey(0);
    }
}

void tool::ShowMatches(const vector<match> &Match12, const Mat &F1Img, const Mat &F2Img, const vector<KeyPoint> &F1Keys, const vector<KeyPoint> &F2Keys, const bool &SHOW)
{
    if(SHOW){
        vector<cv::DMatch> match_show;
        int num = 0;
        for (int i = 0; i<Match12.size(); i++)
        {
            if(Match12[i].first<0 || Match12[i].second<0){
                continue;
            }

            cv::DMatch match_tmp;
            match_tmp.imgIdx = 0;
            match_tmp.queryIdx = Match12[i].first;
            match_tmp.trainIdx = Match12[i].second;
            match_show.push_back(match_tmp);
            num++;
        }

        Mat img_match;
        cout<<"show match num is: "<<num<<endl;
        namedWindow("RawMatches", CV_WINDOW_NORMAL);
        drawMatches(F1Img, F1Keys, F2Img, F2Keys, match_show, img_match);
        imshow("RawMatches", img_match);
        waitKey(0);
    }
}

void tool::ShowMatches(const vector<match> &Match12, const Mat &F1Img, const Mat &F2Img, const vector<KeyPoint> &F1Keys, const vector<KeyPoint> &F2Keys,
                       const bool &SHOW, const bool &SAVE, const string &SaveName)
{
    vector<cv::DMatch> match_show;
    int num = 0;
    for (int i = 0; i<Match12.size(); i++)
    {
        if(Match12[i].first<0 || Match12[i].second<0){
            continue;
        }

        cv::DMatch match_tmp;
        match_tmp.imgIdx = 0;
        match_tmp.queryIdx = Match12[i].first;
        match_tmp.trainIdx = Match12[i].second;
        match_show.push_back(match_tmp);
        num++;
    }

    Mat img_match;
    cout<<"show match num is: "<<num<<endl;
    drawMatches(F1Img, F1Keys, F2Img, F2Keys, match_show, img_match);
    if(SAVE)
        imwrite(SaveName, img_match);
    if(SHOW){
        namedWindow("RawMatches", CV_WINDOW_NORMAL);
        imshow("RawMatches", img_match);
        waitKey(0);
    }
}

// for loopClosing
void tool::ShowMatchesLP(const Mat &F1Img, const Mat &F2Img, const vector<Vec2> &F1Keys, const vector<Vec2> &F2Keys,
                       const bool &SHOW, const bool &SAVE, const string &SaveName)
{
    vector<cv::DMatch> match_show;
    vector<KeyPoint> keys1, keys2;
    int num = 0;
    for (int i = 0; i<F1Keys.size(); i++)
    {
        cv::DMatch match_tmp;
        match_tmp.imgIdx = 0;
        match_tmp.queryIdx = i;
        match_tmp.trainIdx = i;
        match_show.push_back(match_tmp);
        cv::KeyPoint kpt1, kpt2;
        kpt1.pt.x = F1Keys[i](0,0);
        kpt1.pt.y = F1Keys[i](1,0);
        kpt2.pt.x = F2Keys[i](0,0);
        kpt2.pt.y = F2Keys[i](1,0);
        keys1.push_back(kpt1);
        keys2.push_back(kpt2);
        num++;
    }

    Mat img_match;
    cout<<"show match num is: "<<num<<endl;
    drawMatches(F1Img, keys1, F2Img, keys2, match_show, img_match);

    if(SHOW){
        namedWindow("LP Matches", CV_WINDOW_NORMAL);
        imshow("LP Matches", img_match);
        waitKey(0);
    }
    if(SAVE)
        imwrite(SaveName, img_match);

}

void tool::ShowMatchesLP(const Mat &F1Img, const Mat &F2Img, const vector<Vec2> &F1Keys, const vector<Vec2> &F2Keys, const vector<bool> &vbFlag,
                       const bool &SHOW, const bool &SAVE, const string &SaveName)
{
    vector<cv::DMatch> match_show;
    vector<KeyPoint> keys1, keys2;
    int num = 0;
    for (int i = 0; i<F1Keys.size(); i++)
    {
        cv::KeyPoint kpt1, kpt2;
        kpt1.pt.x = F1Keys[i](0,0);
        kpt1.pt.y = F1Keys[i](1,0);
        kpt2.pt.x = F2Keys[i](0,0);
        kpt2.pt.y = F2Keys[i](1,0);
        keys1.push_back(kpt1);
        keys2.push_back(kpt2);

        if(!vbFlag[i])
            continue;
        cv::DMatch match_tmp;
        match_tmp.imgIdx = 0;
        match_tmp.queryIdx = i;
        match_tmp.trainIdx = i;
        match_show.push_back(match_tmp);
        num++;
    }

    Mat img_match;
    cout<<"show match num is: "<<num<<endl;
    drawMatches(F1Img, keys1, F2Img, keys2, match_show, img_match);

    if(SHOW){
        namedWindow("LP Matches", CV_WINDOW_NORMAL);
        imshow("LP Matches", img_match);
        waitKey(0);
    }
    if(SAVE)
        imwrite(SaveName, img_match);

}

void tool::ShowFeature(const vector<size_t> &FeatIdx, const cv::Mat &Img, const vector<cv::KeyPoint> &Keys, const bool &SHOW)
{
    if(SHOW){
        vector<int> FeatIdxInt;
        for(size_t i0 = 0; i0<FeatIdx.size(); i0++){
            if(FeatIdx[i0]<0)
                continue;

            int idx = FeatIdx[i0];
            FeatIdxInt.push_back(idx);
        }

        ShowFeature(FeatIdxInt, Img, Keys, SHOW);
    }

}

void tool::ShowFeature(const vector<int> &FeatIdx, const cv::Mat &Img, const vector<cv::KeyPoint> &Keys, const bool &SHOW)
{
    if(SHOW){
        vector<cv::KeyPoint> KeysDraw;
        for(size_t i0 = 0; i0<FeatIdx.size(); i0++){
            if(FeatIdx[i0]<0)
                continue;

            KeysDraw.push_back(Keys[FeatIdx[i0]]);
        }

        ShowFeatures(Img, KeysDraw, SHOW);
    }
}

void tool::ShowFeatures(const cv::Mat &Img, const vector<cv::KeyPoint> &Keys, const bool &SHOW)
{
    if(SHOW){
        cv::Mat imGrayOut;
        drawKeypoints(Img, Keys, imGrayOut, Scalar(0,0,255));
        namedWindow("ShowFeatures", CV_WINDOW_NORMAL);
        imshow("ShowFeatures", imGrayOut);
        waitKey(0);
    }
}

// savename must be path+name
cv::Mat tool::ShowText(const vector<vector<Vec2>> &textpred, const Mat Img)
{
    Mat Imgdraw = Img.clone();
    for(size_t i0 = 0; i0<textpred.size(); i0++){
        vector<Vec2> textpredObj = textpred[i0];
        for(size_t i1 = 0; i1<textpredObj.size(); i1++)
            circle(Imgdraw, cv::Point(textpredObj[i1](0), textpredObj[i1](1)), 0.5, Scalar(255,0,0), -1);
    }

    return Imgdraw;
}


// input: all text box in the image
cv::Mat tool::ShowTextBox(const vector<vector<Vec2>> &textbox, const Mat Img, const string &savename, const bool &SAVE)
{
    Mat Imgdraw = Img.clone();
    Scalar Color = Scalar(0, 0, 255);
    for(size_t i0 = 0; i0<textbox.size(); i0++){
        vector<Vec2> Box4Pts = textbox[i0];
        line(Imgdraw, cv::Point2d(Box4Pts[0](0), Box4Pts[0](1)), cv::Point2d(Box4Pts[1](0), Box4Pts[1](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[1](0), Box4Pts[1](1)), cv::Point2d(Box4Pts[2](0), Box4Pts[2](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[2](0), Box4Pts[2](1)), cv::Point2d(Box4Pts[3](0), Box4Pts[3](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[3](0), Box4Pts[3](1)), cv::Point2d(Box4Pts[0](0), Box4Pts[0](1)), Color);
    }

    return Imgdraw;
}

// input: all text box in the image
cv::Mat tool::ShowTextBoxWithText(const vector<vector<Vec2>> &textbox, const vector<string> &vShowText, const Mat Img, const string &ShowName)
{
    Mat Imgdraw = Img.clone();
    Scalar Color = Scalar(0, 0, 255);

    // obj showMsg
    CvFont font;
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 0.4,0.4,0,0.35);          // hScale,vScale,0,lineWidth
    IplImage tmp = IplImage(Imgdraw);
    CvArr* src = (CvArr*)&tmp;

    for(size_t i0 = 0; i0<textbox.size(); i0++){
        vector<Vec2> Box4Pts = textbox[i0];
        line(Imgdraw, cv::Point2d(Box4Pts[0](0), Box4Pts[0](1)), cv::Point2d(Box4Pts[1](0), Box4Pts[1](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[1](0), Box4Pts[1](1)), cv::Point2d(Box4Pts[2](0), Box4Pts[2](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[2](0), Box4Pts[2](1)), cv::Point2d(Box4Pts[3](0), Box4Pts[3](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[3](0), Box4Pts[3](1)), cv::Point2d(Box4Pts[0](0), Box4Pts[0](1)), Color);

        // 3. show obj id & mean
        cvPutText(src, vShowText[i0].c_str(), cvPoint(Box4Pts[0](0), Box4Pts[0](1)), &font,CV_RGB(255,255,0));
    }

    namedWindow(ShowName, CV_WINDOW_NORMAL);
    imshow(ShowName, Imgdraw);
    waitKey(0);

    return Imgdraw;
}

cv::Mat tool::ShowTextBoxSingle(const Mat &Img, const vector<Vec2> &Pred, const int &ObjId)
{
    Mat Imgdraw = Img.clone();
    Scalar Color = Scalar(0, 0, 255);
    // obj showMsg
    CvFont font;
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 0.4,0.4,0,0.35);          // hScale,vScale,0,lineWidth
    IplImage tmp = IplImage(Imgdraw);
    CvArr* src = (CvArr*)&tmp;

    line(Imgdraw, cv::Point2d(Pred[0](0), Pred[0](1)), cv::Point2d(Pred[1](0), Pred[1](1)), Color);
    line(Imgdraw, cv::Point2d(Pred[1](0), Pred[1](1)), cv::Point2d(Pred[2](0), Pred[2](1)), Color);
    line(Imgdraw, cv::Point2d(Pred[2](0), Pred[2](1)), cv::Point2d(Pred[3](0), Pred[3](1)), Color);
    line(Imgdraw, cv::Point2d(Pred[3](0), Pred[3](1)), cv::Point2d(Pred[0](0), Pred[0](1)), Color);

    string showMsg = to_string(ObjId);
    cvPutText(src, showMsg.c_str(), cvPoint(Pred[0](0), Pred[0](1)), &font,CV_RGB(255,255,0));

    return Imgdraw;
}

// return the text box && box with its region fill in (fusion of ShowTextBox+ShowTextBoxFill)
vector<cv::Mat> tool::TextBoxWithFill(const vector<vector<Vec2>> &textbox, const vector<int> &vObjId, const Mat Img, const vector<int> &textLabel)
{
    cv::Mat BackImg = cv::Mat::ones(Img.size(), CV_32F)*(-1.0);
    Mat Imgdraw = Img.clone();
    Scalar Color = Scalar(0, 0, 255);

    // obj id show
    CvFont font;
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 0.4,0.4,0,0.35);          // hScale,vScale,0,lineWidth
    IplImage tmp = IplImage(Imgdraw);
    CvArr* src = (CvArr*)&tmp;

    for(size_t i0 = 0; i0<textbox.size(); i0++){
        // 1. get text label
        BackImg = GetTextLabelMask(BackImg, textbox[i0], textLabel[i0]);         // CHECK, cv::Mat put pointer into func. maybe return the new mat to the same pointer

        // 2. get text box show img
        vector<Vec2> Box4Pts = textbox[i0];
        line(Imgdraw, cv::Point2d(Box4Pts[0](0), Box4Pts[0](1)), cv::Point2d(Box4Pts[1](0), Box4Pts[1](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[1](0), Box4Pts[1](1)), cv::Point2d(Box4Pts[2](0), Box4Pts[2](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[2](0), Box4Pts[2](1)), cv::Point2d(Box4Pts[3](0), Box4Pts[3](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[3](0), Box4Pts[3](1)), cv::Point2d(Box4Pts[0](0), Box4Pts[0](1)), Color);

        // 3. show obj id
        string showMsg = to_string(vObjId[i0]);
        cvPutText(src, showMsg.c_str(), cvPoint(Box4Pts[0](0), Box4Pts[0](1)), &font,CV_RGB(255,255,0));

    }

    vector<cv::Mat> ImgOut;
    ImgOut.push_back(BackImg);
    ImgOut.push_back(Imgdraw);
    return ImgOut;
}

// based on TextBoxWithFill, add semantic meaning
vector<cv::Mat> tool::TextBoxWithFill(const vector<vector<Vec2>> &textbox, const vector<int> &vObjId, const vector<TextInfo> &textinfo, const Mat Img, const vector<int> &textLabel)
{
    cv::Mat BackImg = cv::Mat::ones(Img.size(), CV_32F)*(-1.0);
    Mat Imgdraw = Img.clone();
    Scalar Color = Scalar(0, 0, 255);

    // obj id show
    CvFont font;
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, 0.4,0.4,0,0.35);          // hScale,vScale,0,lineWidth
    IplImage tmp = IplImage(Imgdraw);
    CvArr* src = (CvArr*)&tmp;

    for(size_t i0 = 0; i0<textbox.size(); i0++){
        // 1. get text label
        BackImg = GetTextLabelMask(BackImg, textbox[i0], textLabel[i0]);         // CHECK, cv::Mat put pointer into func. maybe return the new mat to the same pointer

        // 2. get text box show img
        vector<Vec2> Box4Pts = textbox[i0];
        line(Imgdraw, cv::Point2d(Box4Pts[0](0), Box4Pts[0](1)), cv::Point2d(Box4Pts[1](0), Box4Pts[1](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[1](0), Box4Pts[1](1)), cv::Point2d(Box4Pts[2](0), Box4Pts[2](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[2](0), Box4Pts[2](1)), cv::Point2d(Box4Pts[3](0), Box4Pts[3](1)), Color);
        line(Imgdraw, cv::Point2d(Box4Pts[3](0), Box4Pts[3](1)), cv::Point2d(Box4Pts[0](0), Box4Pts[0](1)), Color);

        // 3. show obj id
        string showMsg = to_string(vObjId[i0]);
        showMsg += textinfo[i0].mean;
        cvPutText(src, showMsg.c_str(), cvPoint(Box4Pts[0](0), Box4Pts[0](1)), &font,CV_RGB(255,255,0));

    }

    vector<cv::Mat> ImgOut;
    ImgOut.push_back(BackImg);
    ImgOut.push_back(Imgdraw);
    return ImgOut;
}


cv::Mat tool::GetTextLabelMask(const cv::Mat &BackImg, const vector<Vec2> &TextObjBox, const int &TextLabel)
{
    bool SHOW = false;
    cv::Mat textMask = BackImg.clone();
    cv::Scalar Label = cv::Scalar(TextLabel,TextLabel,TextLabel);

    cv::Point box4pts[4];
    box4pts[0] = cv::Point(TextObjBox[0](0), TextObjBox[0](1));
    box4pts[1] = cv::Point(TextObjBox[1](0), TextObjBox[1](1));
    box4pts[2] = cv::Point(TextObjBox[2](0), TextObjBox[2](1));
    box4pts[3] = cv::Point(TextObjBox[3](0), TextObjBox[3](1));
    const cv::Point* ptMask[1] = {box4pts};

    int npt[] = {4};
    cv::fillPoly(textMask, ptMask, npt, 1, Label);        // the reigion inside ptMask is fill in with Label Scalar

    if(SHOW){
        cv::Mat ImgZero = cv::Mat::zeros(BackImg.size(),CV_8UC1);
        cv::fillPoly(ImgZero, ptMask, npt, 1, cv::Scalar(255,255,255));

        cv::namedWindow("ImgZero", CV_WINDOW_NORMAL);
        cv::imshow("ImgZero", ImgZero);
        cv::waitKey(0);
    }

    return textMask;
}

cv::Mat tool::ShowScene(const vector<Vec2> &scenepred, const vector<SceneFeature*> &sceneobv, const Mat Img, const string &savename, const bool &SAVE)
{
    Mat Imgdraw = Img.clone();
    vector<KeyPoint> kscenepred = vV2vK(scenepred);
    vector<KeyPoint> ksceneobv;
    ksceneobv.resize(sceneobv.size());
    for(size_t j0 = 0; j0<ksceneobv.size(); j0++)
        ksceneobv[j0] = V2K(sceneobv[j0]->feature);

    drawKeypoints(Imgdraw, ksceneobv, Imgdraw, Scalar(0, 0, 255));      // red -- scene observation
    drawKeypoints(Imgdraw, kscenepred, Imgdraw, Scalar(255,255,0));     // yellow -- scene prediction

    double ex = 0, ey = 0;
    for(size_t i0 = 0; i0<scenepred.size(); i0++){
        line(Imgdraw, cv::Point2d(scenepred[i0](0), scenepred[i0](1)), cv::Point2d(sceneobv[i0]->u, sceneobv[i0]->v), Scalar(0, 255, 0));       // green -- error
        ex += abs(scenepred[i0](0)-sceneobv[i0]->u);
        ey += abs(scenepred[i0](1)-sceneobv[i0]->v);

    }

    if(SAVE)
        imwrite(savename, Imgdraw);

    return Imgdraw;
}


cv::Mat tool::ShowScene(const vector<Vec2> &scenepred, const vector<Vec2> &sceneobv, const Mat Img, const string &savename, const bool &SAVE)
{

        Mat Imgdraw = Img.clone();
        vector<KeyPoint> kscenepred = vV2vK(scenepred);
        vector<KeyPoint> ksceneobv;
        ksceneobv.resize(sceneobv.size());
        for(size_t j0 = 0; j0<ksceneobv.size(); j0++)
            ksceneobv[j0] = V2K(sceneobv[j0]);

        drawKeypoints(Imgdraw, ksceneobv, Imgdraw, Scalar(0, 0, 255));      // red -- scene observation
        drawKeypoints(Imgdraw, kscenepred, Imgdraw, Scalar(255,255,0));     // yellow -- scene prediction

        double ex = 0, ey = 0;
        for(size_t i0 = 0; i0<scenepred.size(); i0++){
            line(Imgdraw, cv::Point2d(scenepred[i0](0), scenepred[i0](1)), cv::Point2d(sceneobv[i0](0), sceneobv[i0](1)), Scalar(0, 255, 0));       // green -- error
            ex += abs(scenepred[i0](0)-sceneobv[i0](0));
            ey += abs(scenepred[i0](1)-sceneobv[i0](1));

        }

        if(SAVE)
            imwrite(savename, Imgdraw);

        return Imgdraw;
}

void tool::ShowMatchesLP(const Mat &F1Img, const Mat &F2Img, const vector<FeatureConvert> &F1Keys, const vector<FeatureConvert> &F2Keys, const vector<bool> &vbFlag,
                       const bool &SHOW, const bool &SAVE, const string &SaveName)
{
    vector<cv::DMatch> match_show;
    vector<KeyPoint> keys1, keys2;
    int num = 0;
    for (int i = 0; i<F1Keys.size(); i++)
    {
        cv::KeyPoint kpt1, kpt2;
        kpt1.pt.x = F1Keys[i].obv2dPred(0,0);
        kpt1.pt.y = F1Keys[i].obv2dPred(1,0);
        kpt2.pt.x = F2Keys[i].obv2dPred(0,0);
        kpt2.pt.y = F2Keys[i].obv2dPred(1,0);
        keys1.push_back(kpt1);
        keys2.push_back(kpt2);

        if(!vbFlag[i])
            continue;
        cv::DMatch match_tmp;
        match_tmp.imgIdx = 0;
        match_tmp.queryIdx = i;
        match_tmp.trainIdx = i;
        match_show.push_back(match_tmp);
        num++;
    }

    Mat img_match;
    cout<<"show match num is: "<<num<<endl;
    drawMatches(F1Img, keys1, F2Img, keys2, match_show, img_match);

    if(SHOW){
        namedWindow("LP Matches", CV_WINDOW_NORMAL);
        imshow("LP Matches", img_match);
        waitKey(0);
    }
    if(SAVE)
        imwrite(SaveName+to_string(num)+".png", img_match);

}

void tool::ShowImg(const cv::Mat &ImgDraw, const string &name)
{
    cv::namedWindow(name, CV_WINDOW_NORMAL);
    cv::imshow(name, ImgDraw);
    cv::waitKey(0);
}

//  ------------ visualization ------------

//  ------------ Record for debug ------------

void tool::RecordSim3Optim(const vector<Mat31> &vMapPt1, const vector<Mat31> &vMapPt2, const vector<Vec2> &vObv1, const vector<Vec2> &vObv2, const Mat33 &SimR12, const Mat31 &Simt12, const double &Sims12)
{
    string name1_pt1 = "./debug/vPt1.txt";
    string name2_pt2 = "./debug/vPt2.txt";
    string name3_Obv1 = "./debug/vObv1.txt";
    string name4_Obv2 = "./debug/vObv2.txt";
    string name5_SimR = "./debug/SimR.txt";
    string name6_Simt = "./debug/Simt.txt";
    string name7_Sims = "./debug/Sims.txt";

    ofstream file1(name1_pt1,ios::app);
    ofstream file2(name2_pt2,ios::app);
    ofstream file3(name3_Obv1,ios::app);
    ofstream file4(name4_Obv2,ios::app);
    ofstream file5(name5_SimR,ios::app);
    ofstream file6(name6_Simt,ios::app);
    ofstream file7(name7_Sims,ios::app);

    for(size_t i0=0; i0<vMapPt1.size(); i0++)
        file1<<vMapPt1[i0](0,0)<<", "<<vMapPt1[i0](1,0)<<", "<<vMapPt1[i0](2,0)<<endl;

    for(size_t i1=0; i1<vMapPt2.size(); i1++)
        file2<<vMapPt2[i1](0,0)<<", "<<vMapPt2[i1](1,0)<<", "<<vMapPt2[i1](2,0)<<endl;

    for(size_t i2=0; i2<vObv1.size(); i2++)
        file3<<vObv1[i2](0,0)<<", "<<vObv1[i2](1,0)<<endl;

    for(size_t i3=0; i3<vObv2.size(); i3++)
        file4<<vObv2[i3](0,0)<<", "<<vObv2[i3](1,0)<<endl;

    file5<<SimR12<<endl;
    file6<<Simt12<<endl;
    file7<<Sims12<<endl;

    file1.close();
    file2.close();
    file3.close();
    file4.close();
    file5.close();
    file6.close();
    file7.close();
}


//  ------------ Record for debug ------------
}
