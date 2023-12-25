/*
This code is the implementation of our paper "TextSLAM: Visual SLAM with Semantic Planar Text Features."

Author: Boying Li   < LeeBY2016@outlook.com >

If you use any code of this repo in your work, please cite our papers:
[1] Li Boying, Zou Danping, et al. "TextSLAM: Visual SLAM with Semantic Planar Text Features."
[2] Li Boying, Zou Danping, et al. "TextSLAM: Visual slam with planar text features."
*/

#include <loopClosing.h>

namespace TextSLAM
{

// can change
const int loopClosing::TH_HIGH = 100;
const int loopClosing::TH_LOW = 50;
const int loopClosing::HISTO_LENGTH = 30;

loopClosing::loopClosing(setting *Setting)
{
    DeleteWords();

    if(Setting->eExp_name==3){
        // Outdoor(3)
        Thmin_ThreshMatchWordsNum = 2;
        Th_nInliers_Scene = -1;
        Th_MaxInlierNum_S = 10;
        ScoreThresh_min = 0.35;
        DoubleCheck_Visible = true;
    }else{
        // IndoorLoop1(1)/IndoorLoop2(2)
        Thmin_ThreshMatchWordsNum = 1;
        Th_MaxInlierNum_S = -1;
        ScoreThresh_min = 0.51;
        DoubleCheck_Visible = false;
        if(Setting->eExp_name==1)
            Th_nInliers_Scene = 8;
        else
            Th_nInliers_Scene = -1;
    }


    cout<<" -------- loopClosing basic info -------- "<<endl;
    cout<<"min thresh for ThreshMatchWordsNum (Thmin_ThreshMatchWordsNum) is: "<<Thmin_ThreshMatchWordsNum<<endl;
    cout<<"thresh for Th_MaxInlierNum_S (Th_MaxInlierNum_S) is: "<<Th_MaxInlierNum_S<<endl;
    cout<<"thresh for ScoreThresh_min (ScoreThresh_min) is: "<<ScoreThresh_min<<endl;
    cout<<"thresh for nInliers_Scene (Th_nInliers_Scene) is: "<<Th_nInliers_Scene<<endl;
    cout<<"Double Check for visible? (DoubleCheck_Visible) : "<<DoubleCheck_Visible<<endl;

}

bool loopClosing::Run(keyframe* _CurKF, map* _mpMap, optimizer* _Optimizer)
{
    // ***************************** 1. parameters prepare *****************************
    mpMap = _mpMap;
    mpCurrentKF = _CurKF;
    MaxInlierNum = 0;
    MaxInlierNum_S = 0;
    coOptimizer = _Optimizer;

    vector<keyframe*> vKFs = mpMap->GetAllKeyFrame();
    vector<keyframe*> vKFsUse = vKFs;
    assert(vKFsUse[vKFsUse.size()-1]->mnId==_CurKF->mnId);
    vKFsUse.pop_back();

    // ***************************** functional section *****************************

    // Step 1. get min match words
    bool FLAG_ENOUGH = true;
    std::map<keyframe*, int> vConnects;
    int ThreshMatchWordsNum = GetThreshWordsNum(FLAG_ENOUGH, vConnects);
    if(!FLAG_ENOUGH)
        return false;

    ThreshMatchWordsNum = std::floor(0.6*(double)ThreshMatchWordsNum);
    ThreshMatchWordsNum = std::max(ThreshMatchWordsNum, Thmin_ThreshMatchWordsNum);

    // Step 2. loop detection
    vector<vector<MatchmapTextRes>> vMatchTexts;
    vector<keyframe*> vKFCandidates = DetectLoop(vKFsUse, vMatchTexts, ThreshMatchWordsNum, vConnects);

    if(vKFCandidates.size()==0){
        return false;
    }

    // Step 3. Sim3 calculation
    // parameter convertion -> feature matches -> Sim3 calculate -> matched KF
    ComputeSim3(vKFCandidates, vMatchTexts);

    if(Th_MaxInlierNum_S>0){
        if(MaxInlierNum<=25 || MaxInlierNum_S<Th_MaxInlierNum_S){
            return false;}
        else{
            // Step 4. define matched KF
            mpCurrentKF->AddLoopEdge(mpMatchedKF);
            mpMatchedKF->AddLoopEdge(mpCurrentKF);}
    }else{
        if(MaxInlierNum<=25){
            return false;}
        else{
            // Step 4. define matched KF
            mpCurrentKF->AddLoopEdge(mpMatchedKF);
            mpMatchedKF->AddLoopEdge(mpCurrentKF);}
    }

    // Step 4. Loop correct
    LoopCorrect();

        // update covisible KFs for each KF, 'tracking' use
    UpdateCovisibleKFs();

    cout<<"end loopClosing."<<endl;

    return true;
}

vector<keyframe*> loopClosing::DetectLoop(const vector<keyframe*> &vKFs, vector<vector<MatchmapTextRes> > &vAllMatchTextRes, const int &MinMatchedWords, const std::map<keyframe *, int> &vConnects)
{
    // -------- settings --------
    double thMinStrScore = 0.3;
    double w1=1.0, w2=1.0-w1;
    // -------- settings --------

    // Step 1. Parameter Prepare
    vector<keyframe*> vMatchKFs;

        // KF
    vector<int> vmnId2vKFs;
    vmnId2vKFs = vector<int>(mpMap->imapkfs, (int)-1);
    vector<std::map<mapText*, int>> vKFsMathedObjs;
    vKFsMathedObjs.resize(mpMap->imapkfs);

    vector<matchRes> vMapKFScore;
    for(size_t iini=0; iini<vKFs.size(); iini++){
        vmnId2vKFs[(size_t)vKFs[iini]->mnId] = iini;
        vMapKFScore.push_back(make_pair(iini, 0.0));
    }

        // mapText
    vector<mapText*> vObjs = mpMap->GetAllMapTexts();

    vector<TextObservation*> vObvMapTexts = mpCurrentKF->vObvText;
    vAllMatchTextRes.resize(vObvMapTexts.size());

        // covisible map
    vector<Eigen::MatrixXd> vM = mpMap->GetCovMap_All();

    // Step 2. Text Match
    for(size_t iObvtext = 0; iObvtext<vObvMapTexts.size(); iObvtext++){
        mapText* objCur = vObvMapTexts[iObvtext]->obj;
        vector<int> vObvIdxCur = vObvMapTexts[iObvtext]->idx;
        int idxCur = -1;
        if(vObvIdxCur.size()!=0)
            idxCur = vObvIdxCur[0];

        TextInfo textCur = objCur->TextMean;
        size_t idx_empty = textCur.mean.find("#");  // empty recognition
        if(idx_empty < textCur.mean.length() && idx_empty>=0)
            continue;

        // the matched mapText of this detected text in CurKF
        vector<MatchmapTextRes> vMatchTextRes;
        vAllMatchTextRes[iObvtext] = vMatchTextRes;

        // 2.1 for each queryText, search each item in vMapTexts, dist(vScore, vDist)

        vector<matchRes> vScore;        // <mapText Idx, score>
        vector<double> vDist;
        for(size_t i0=0; i0<vObjs.size(); i0++)
            vScore.push_back(make_pair(i0, (double)-1.0));        // <idMapTextSort, score>
        vDist = vector<double>(vScore.size(), (double)-1.0);

        for(size_t iMaptext = 0; iMaptext<vObjs.size(); iMaptext++){
            // objCur VS objMap
            mapText* objMap = vObjs[iMaptext];

            // skip bad text obj
            if(objMap->STATE==TEXTBAD)
                continue;

            // skip self obj
            if(objMap->mnId==objCur->mnId)
                continue;

            TextInfo textMap = objMap->TextMean;

            // L-dist calculate
            double dist = Tool.LevenshteinDist(textCur.mean, textMap.mean);
            int maxlen = max((int)textCur.mean.length(), (int)textMap.mean.length());
            double score = ((double)maxlen-dist)/(double)maxlen;
            vScore[iMaptext].second = score;
            vDist[iMaptext] = dist;
        }   // all map texts

        // 2.2 for each CurKF's observed TextObj, find most similar mapTexts
            // for each objMap, mapText Sort results is vScore (vDist)
        sort(vScore.begin(),vScore.end(),cmpLarge());

        // Cond2. if ScoreMax is small -> result not valid -> skip
        double ScoreMax = vScore[0].second, Scoreth;
        if(ScoreMax<thMinStrScore){
            continue;
        }

        if(ScoreMax==1.0)
            Scoreth = ScoreMax;
        else
            Scoreth = std::max(ScoreMax*2.0/3.0, ScoreThresh_min);


        // 2.3 based on the most similar mapTexts, these observed KFs score +1
        for(size_t ires=0; ires<vScore.size(); ires++){

            if(vScore[ires].second<Scoreth)
                break;

            assert(vScore[ires].first!=-1);
            assert(vScore[ires].first<vDist.size());

            struct MatchmapTextRes MatchTextRes = MatchmapTextRes{vObjs[(size_t)vScore[ires].first], vDist[(size_t)vScore[ires].first], vScore[ires].second};
            vMatchTextRes.push_back(MatchTextRes);

            std::map<keyframe*,vector<int>> vObvKFs = MatchTextRes.mapObj->vObvkeyframe;
            std::map<keyframe*,vector<int>>::const_iterator iobvKF;
            for(iobvKF=vObvKFs.begin(); iobvKF!=vObvKFs.end(); iobvKF++){

                // skip Current KF & out of vKFs & covisible KFs
                bool c0 = (iobvKF->first->mnId==mpCurrentKF->mnId);
                bool c1 = (vmnId2vKFs[iobvKF->first->mnId]==-1);
                bool c2 = ( vM[0]((int)iobvKF->first->mnId, (int)mpCurrentKF->mnId) )!=0;
                bool c3 = ( vM[1]((int)iobvKF->first->mnId, (int)mpCurrentKF->mnId) )!=0;
                bool c4 = ( vM[2]((int)iobvKF->first->mnId, (int)mpCurrentKF->mnId) )!=0;
                assert(c3==c4);

                if(c0 || c1 || c2|| c3 || c4){
                    continue;
                }

                if(DoubleCheck_Visible){
                    if(vConnects.count(iobvKF->first)){
                        continue;
                    }
                }

                int idxvKFs = vmnId2vKFs[iobvKF->first->mnId];
                assert(idxvKFs!=-1);
                assert(idxvKFs<vMapKFScore.size());

                double score1 = 1.0;
                double score = w1*score1;
                assert(idxvKFs==vMapKFScore[idxvKFs].first);
                // KF's score
                vMapKFScore[idxvKFs].second += score;

                // number of no-complete matches
                if(!vKFsMathedObjs[idxvKFs].count(MatchTextRes.mapObj))
                    vKFsMathedObjs[idxvKFs].insert(make_pair(MatchTextRes.mapObj, 1));
                else
                    vKFsMathedObjs[idxvKFs][MatchTextRes.mapObj]++;
            }   // all observed KFs
        }

        vAllMatchTextRes[iObvtext] = vMatchTextRes;
    }       // all observed text in current KF

    // Step 3. sort KFs score
    sort(vMapKFScore.begin(),vMapKFScore.end(),cmpLarge());

    {
        // use general method
        int TopN = 10;
        int ThreshTop = std::min(TopN, (int)vMapKFScore.size());
        int idxOut = 0;

        for(size_t iFcout=0; iFcout<vMapKFScore.size(); iFcout++){

            if(vMapKFScore[iFcout].second<=MinMatchedWords){
                break;
            }

            int idxvKFs = vmnId2vKFs[vKFs[vMapKFScore[iFcout].first]->mnId];
            assert(idxvKFs == vMapKFScore[iFcout].first);
            if(vKFsMathedObjs[idxvKFs].size()<=MinMatchedWords)
                continue;

            if(idxOut>=ThreshTop)
                break;

            int idx_kF = vMapKFScore[iFcout].first;

            int UseM = 0;
            double CovisibleNum = vM[UseM](vKFs[idx_kF]->mnId, mpCurrentKF->mnId);
            if(CovisibleNum>0)
                continue;

            vMatchKFs.push_back(vKFs[idx_kF]);
            idxOut++;
        }
    }

    return vMatchKFs;
}

void loopClosing::ComputeSim3(const vector<keyframe*> &vKFCands,  const vector<vector<MatchmapTextRes>> &vMatchRes)
{
    vector<bool> vbDiscarded;
    vbDiscarded.resize(vKFCands.size());

    for(size_t ikf = 0; ikf<vKFCands.size(); ikf++)
    {
        keyframe* KFCan = vKFCands[ikf];

        // Step 1&2. parameter convertion, feature matches
        // P1,P2, Obv&Calculate
        // pos(xyz), Obv, Calculate, flag_text/Scene, pt, obj, KF, featureIdx
        vector<FeatureConvert> vFeatCur, vFeatCan;
        int nMatch = SearchMatch(KFCan, vMatchRes, vFeatCur, vFeatCan);

        // visual (obv2dPred) check
        vector<bool> vbInliers = vector<bool>(vFeatCur.size(), true);

        // Step 3. Sim3 calculate & optimization
        Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF, KFCan, vFeatCur, vFeatCan);
        pSolver->SetRansacParameters(0.99,20,300);
        int nInliers;
        bool bNoMore;
        Mat44 T12Sim3Res;
        bool OK = pSolver->iterate(5, bNoMore, vbInliers, nInliers, T12Sim3Res);    // 5: iter times

        if(!OK){
            vbDiscarded[ikf] = true;
            continue;
        }

        Mat33 R = pSolver->GetEstimatedRotation();
        Mat31 t = pSolver->GetEstimatedTranslation();
        double s = pSolver->GetEstimatedScale();
        Sim3_loop gScm(R,t,s);

        // optimization
        int nInliersOpt = coOptimizer->OptimizeSim3(vFeatCur, vFeatCan, vbInliers, gScm, 10);

        // add more scene -- MatchMore
        vector<FeatureConvert> vFeatCurMore, vFeatCanMore;
        int nInliers_Scene = MatchMore(mpCurrentKF, KFCan, gScm, vFeatCurMore, vFeatCanMore);
        vFeatCur.insert(vFeatCur.end(), vFeatCurMore.begin(), vFeatCurMore.end());
        vFeatCan.insert(vFeatCan.end(), vFeatCanMore.begin(), vFeatCanMore.end());
        vector<bool> vbInliersMore = vector<bool>(vFeatCurMore.size(), true);
        vbInliers.insert(vbInliers.end(), vbInliersMore.begin(), vbInliersMore.end());

        bool Flag_Better = nInliersOpt>MaxInlierNum;
        if(Th_nInliers_Scene>0)
            Flag_Better = (nInliersOpt>MaxInlierNum && nInliers_Scene>=Th_nInliers_Scene);
        if(Flag_Better){
            MaxInlierNum = nInliersOpt;
            MaxInlierNum_S = nInliers_Scene;
            mpMatchedKF = KFCan;
            Sim3_loop Smw(mpMatchedKF->mTcw.block<3,3>(0,0), mpMatchedKF->mTcw.block<3,1>(0,3), 1.0);
            mScw = gScm*Smw;
            mScm = gScm;
            mvFeatCur.clear();
            mvFeatCan.clear();
            for(size_t i=0; i<vbInliers.size(); i++){
                if(!vbInliers[i])
                    continue;
                mvFeatCur.push_back(vFeatCur[i]);
                mvFeatCan.push_back(vFeatCan[i]);
            }
            assert((int)mvFeatCur.size()==MaxInlierNum+MaxInlierNum_S);
            assert((int)mvFeatCan.size()==MaxInlierNum+MaxInlierNum_S);
        }

    }

}

void loopClosing::LoopCorrect()
{
    // -------- settings --------
    int thEssent = 100;
    bool UseEssential = false;
    bool AddCurrent = false;
    int ThreshLoop = 0;
    // -------- settings --------

    // ------------------------ Step 1 param prepare ------------------------ //
    // a) spread Sim3 error to the local window of the current Frame ------
    vector<Sim3_loop> vCorrectSiw;
    std::map<keyframe*, Sim3_loop, std::less<keyframe*>,  Eigen::aligned_allocator<std::pair<keyframe*, Sim3_loop>> > vConnectKFs;
    Mat44 Twc = mpCurrentKF->mTwc;

    vector<CovKF> vConnects = mpCurrentKF->GetCovisibleKFs_All();
    for(size_t ikfCon=0; ikfCon<vConnects.size(); ikfCon++){
        if(UseEssential){
            if(vConnects[ikfCon].second<thEssent)
                continue;
        }

        keyframe* KFi = vConnects[ikfCon].first;
        Mat44 Tiw = KFi->mTcw;
        Mat44 Tic = Tiw * Twc;
        Mat33 Ric = Tic.block<3,3>(0,0);
        Mat31 tic = Tic.block<3,1>(0,3);
        Sim3_loop Sic(Ric,tic,1.0);
        Sim3_loop Siw = Sic*mScw;
        vConnectKFs[KFi] = Siw;

    }
    if(AddCurrent)
        vConnectKFs[mpCurrentKF] = mScw;

    // ------------------------------------

    // b) each frame normalEdge ------------
    vector<keyframe*> vKFs = mpMap->GetAllKeyFrame();

    std::map<keyframe*, set<keyframe*> > NormConnections;
    for(size_t iNorm=0; iNorm<vKFs.size(); iNorm++){
        keyframe* KFnormi = vKFs[iNorm];
        vector<CovKF> vKFsCovisible = KFnormi->GetCovisibleKFs_Prev();
        set<keyframe*> vKFsCovisibleSet = Tool.Convert2Set(vKFsCovisible);

        NormConnections[KFnormi] = vKFsCovisibleSet;
    }

    vector<CovKF> vLoopKFConnectsRaw = mpMatchedKF->GetCovisibleKFs_All();
    std::map<keyframe*, int> vLoopKFConnectsRaw_map = Tool.Convert2Map(vLoopKFConnectsRaw);

    // ------------------------------------

    // ------------------------ Step 2 landmarker fusion ------------------------ //

    assert((int)mvFeatCur.size()==(int)mvFeatCan.size());

    Eigen::MatrixXd M1 = mpMap->GetCovMap_1();
    Eigen::MatrixXd M2 = mpMap->GetCovMap_2();
    Eigen::MatrixXd M3 = mpMap->GetCovMap_3();
    vector<Eigen::MatrixXd> vMs;
    vMs.push_back(M1);
    vMs.push_back(M2);
    vMs.push_back(M3);

        // match: pair(CurKF_Idx, CanKF_Idx)
    std::map< match, int > vPair, vPairOk;
    for(size_t iobj=0; iobj<mvFeatCan.size(); iobj++){

        if(mvFeatCan[iobj].FlagTS==0)
            continue;

        mapText* obj = mvFeatCur[iobj].obj;
        mapText* objLoop = mvFeatCan[iobj].obj;

        int Idx1 = -1, Idx2 = -1;
        Idx2 = objLoop->mnId;
        if(mvFeatCur[iobj].FlagTS==1)
            Idx1 = obj->mnId;

        // text vs scene, delete mapPts
        if(Idx2>=0 && Idx1<0){
            mvFeatCur[iobj].pt->PtErase(mpCurrentKF);
            continue;
        }

        match objIdxPair = make_pair(Idx1, Idx2);
        if(vPair.count(objIdxPair))
            vPair.at(objIdxPair)++;
        else
            vPair[objIdxPair] = 1;
    }

    // a) Scene pts match
    for(size_t ipt=0; ipt<mvFeatCan.size(); ipt++){
        // only scene vs scene is ok
        if(mvFeatCan[ipt].FlagTS!=0 || mvFeatCur[ipt].FlagTS!=0)
            continue;

        // use ploop to replace p
        mapPts* ploop = mvFeatCan[ipt].pt;
        mapPts* p = mvFeatCur[ipt].pt;

        p->Replace(mpCurrentKF, ploop, vMs[0]);
    }

    std::map< match, int >::iterator iter;
    for(iter=vPair.begin(); iter!=vPair.end(); iter++){
        match MatchPair = iter->first;
        int Count = iter->second;

        int ObjIdx = MatchPair.first;
        int ObjLoopIdx = MatchPair.second;

        mapText* obj = mpMap->GetObjFromId(ObjIdx);
        mapText* objLoop = mpMap->GetObjFromId(ObjLoopIdx);

        obj->Replace(mpCurrentKF, objLoop, vMs[1], vMs[2]);
    }

    // ------------------------ Step 2 landmarker mutual projection，add more connects ------------------------ //
    std::map<mapPts*, keyframe*> vLoopPts;
    std::map<mapText*, keyframe*> vLoopObjs;
    GetLoopsLandmarkers(vLoopPts, vLoopObjs);

    SearchAndFuse(vLoopPts, vLoopObjs, vConnectKFs, AddCurrent, vMs);

    mpMap->SetCovMap_1(vMs[0]);
    mpMap->SetCovMap_2(vMs[1]);
    mpMap->SetCovMap_3(vMs[2]);

    // ------------------------ Step 3 add new loop ------------------------ //
    std::map<keyframe*, set<keyframe*> > LoopConnections;
    std::map<keyframe*, Sim3_loop>::const_iterator iterCon;
    for(iterCon=vConnectKFs.begin(); iterCon!=vConnectKFs.end(); iterCon++)
    {
         keyframe* KF = iterCon->first;

         vector<CovKF> vConnectsRaw = KF->GetCovisibleKFs_Prev();
         GetCovisibleKFs_all(KF, mpMap->imapkfs);
         vector<CovKF> vConnectsNew = KF->GetCovisibleKFs_Prev();

         set<keyframe*> vConnectsNewSet = Tool.Convert2Set(vConnectsNew);
         LoopConnections.insert(make_pair(KF, vConnectsNewSet));

         for(size_t itest=0; itest<vConnectsNew.size(); itest++){
             if(vConnectsNew[itest].second<ThreshLoop)
                 LoopConnections[KF].erase(vConnectsNew[itest].first);
         }

         set<keyframe*>::const_iterator iter;
         vector<keyframe*> vKFsOutLoop;
         for(iter=LoopConnections[KF].begin(); iter!=LoopConnections[KF].end(); iter++){
             keyframe* KFi = (*iter);
             if( !vLoopKFConnectsRaw_map.count(KFi) )
                 vKFsOutLoop.push_back(KFi);
         }

         for(size_t idelet=0; idelet<vKFsOutLoop.size(); idelet++){
             LoopConnections[KF].erase(vKFsOutLoop[idelet]);
         }

    }

    if(!AddCurrent)
    {
        vector<CovKF> vConnectsRaw = mpCurrentKF->GetCovisibleKFs_Prev();
        GetCovisibleKFs_all(mpCurrentKF, mpMap->imapkfs);
        vector<CovKF> vConnectsNew = mpCurrentKF->GetCovisibleKFs_Prev();

        set<keyframe*> vConnectsNewSet = Tool.Convert2Set(vConnectsNew);
        LoopConnections.insert(make_pair(mpCurrentKF, vConnectsNewSet));

        for(size_t itest=0; itest<vConnectsNew.size(); itest++){
            if(vConnectsNew[itest].second<ThreshLoop)
                LoopConnections[mpCurrentKF].erase(vConnectsNew[itest].first);
        }

        set<keyframe*>::const_iterator iter;
        vector<keyframe*> vKFsOutLoop;
        for(iter=LoopConnections[mpCurrentKF].begin(); iter!=LoopConnections[mpCurrentKF].end(); iter++){
            keyframe* KFi = (*iter);
            if( !vLoopKFConnectsRaw_map.count(KFi) )
                vKFsOutLoop.push_back(KFi);
        }

        for(size_t idelet=0; idelet<vKFsOutLoop.size(); idelet++){
            LoopConnections[mpCurrentKF].erase(vKFsOutLoop[idelet]);
        }

    }

    // --------------------------- Step 4 Optimization --------------------------- //
    bool DEBUG = false;
    if(DEBUG){
        cout<<"[debug] Current station:"<<endl;
        cout<<"mScm is: "<<mScm.r.w()<<", "<<mScm.r.x()<<", "<<mScm.r.y()<<", "<<mScm.r.z()<<", ";
        cout<<mScm.t.transpose()<<", "<<mScm.s<<endl;
        cout<<"mSmc is: "<<mScm.inverse().r.w()<<", "<<mScm.inverse().r.x()<<", "<<mScm.inverse().r.y()<<", "<<mScm.inverse().r.z()<<", ";
        cout<<mScm.inverse().t.transpose()<<", "<<mScm.inverse().s<<endl;
        cout<<"mScw is: "<<mScw.r.w()<<", "<<mScw.r.x()<<", "<<mScw.r.y()<<", "<<mScw.r.z()<<", ";
        cout<<mScw.t.transpose()<<", "<<mScw.s<<endl;
        Sim3_loop Smw(mpMatchedKF->mTcw.block<3,3>(0,0), mpMatchedKF->mTcw.block<3,1>(0,3), 1.0);
        cout<<"Smw is: "<<Smw.r.w()<<", "<<Smw.r.x()<<", "<<Smw.r.y()<<", "<<Smw.r.z()<<", ";
        cout<<Smw.t.transpose()<<", "<<Smw.s<<endl;
    }

    coOptimizer->OptimizeLoop(LoopConnections, NormConnections, mpCurrentKF, mpMatchedKF, vConnectKFs, mScw, mpMap);

    coOptimizer->GlobalBA(mpMap);

    coOptimizer->OptimizeLandmarker(mpMap);

}


int loopClosing::GetThreshWordsNum(bool &FLAG_OK, std::map<keyframe*, int> &vConnectKFs)
{
    int MatchWordsNum = -1;
    int UseM = 1;
    int UseThresh = 0;
    int num_neighKFs = 10;
    vector<CovKF> vKFConKFsTop;
    vector<CovKF> vCovKFsAll = mpCurrentKF->GetCovisibleKFs_All();
    if(vCovKFsAll.size()<=num_neighKFs){
        FLAG_OK = false;
        return MatchWordsNum;
    }

    vKFConKFsTop.insert(vKFConKFsTop.end(), vCovKFsAll.begin(), vCovKFsAll.begin()+num_neighKFs);
    Eigen::MatrixXd M2 = mpMap->GetCovMap(UseM);
    vector<Eigen::MatrixXd> vM = mpMap->GetCovMap_All();

    vector<int> vNums;
    for(size_t i0=0; i0<vKFConKFsTop.size(); i0++){
        keyframe* KF = vKFConKFsTop[i0].first;

        int NumCovisibleObjs = M2( (int)KF->mnId, (int)mpCurrentKF->mnId );
        vNums.push_back(NumCovisibleObjs);

        bool c0 = ( vM[0]((int)KF->mnId, (int)mpCurrentKF->mnId) )==0;
        bool c1 = ( vM[1]((int)KF->mnId, (int)mpCurrentKF->mnId) )==0;
        bool c2 = ( vM[2]((int)KF->mnId, (int)mpCurrentKF->mnId) )==0;
        if(c0 && c1 && c2){
            vConnectKFs[KF] = vKFConKFsTop[i0].second;
        }
        vector<CovKF> vCovKFs2All = KF->GetCovisibleKFs_All();
        for(size_t i1=0; i1<vCovKFs2All.size(); i1++){
            bool c3 = ( vM[0]((int)vCovKFs2All[i1].first->mnId, (int)mpCurrentKF->mnId) )==0;
            bool c4 = ( vM[1]((int)vCovKFs2All[i1].first->mnId, (int)mpCurrentKF->mnId) )==0;
            bool c5 = ( vM[2]((int)vCovKFs2All[i1].first->mnId, (int)mpCurrentKF->mnId) )==0;
            if(c3 && c4 && c5){
                vConnectKFs[vCovKFs2All[i1].first] = vCovKFs2All[i1].second;
            }

        }
    }

    for(size_t i2=num_neighKFs; i2<vCovKFsAll.size(); i2++){
        keyframe* KF = vCovKFsAll[i2].first;

        bool c0 = ( vM[0]((int)KF->mnId, (int)mpCurrentKF->mnId) )==0;
        bool c1 = ( vM[1]((int)KF->mnId, (int)mpCurrentKF->mnId) )==0;
        bool c2 = ( vM[2]((int)KF->mnId, (int)mpCurrentKF->mnId) )==0;
        if(c0 && c1 && c2){
            vConnectKFs[KF] = vCovKFsAll[i2].second;
        }

        vector<CovKF> vCovKFs2All = KF->GetCovisibleKFs_All();
        for(size_t i3=0; i3<vCovKFs2All.size(); i3++){
            bool c3 = ( vM[0]((int)vCovKFs2All[i3].first->mnId, (int)mpCurrentKF->mnId) )==0;
            bool c4 = ( vM[1]((int)vCovKFs2All[i3].first->mnId, (int)mpCurrentKF->mnId) )==0;
            bool c5 = ( vM[2]((int)vCovKFs2All[i3].first->mnId, (int)mpCurrentKF->mnId) )==0;
            if(c3 && c4 && c5){
                vConnectKFs[vCovKFs2All[i3].first] = vCovKFs2All[i3].second;
            }
        }
    }

    if(UseThresh==0)
        MatchWordsNum = vNums[vNums.size()-1];
    else if(UseThresh==1)
        MatchWordsNum = vNums[round(vNums.size()/2)];
    else if(UseThresh==2)
        MatchWordsNum = vNums[0];

    return MatchWordsNum;
}

// tool
void loopClosing::DeleteWords()
{
    vDeleteWords.insert(make_pair("to", 0.0));
    vDeleteWords.insert(make_pair("To", 0.0));
    vDeleteWords.insert(make_pair("that", 0.0));
    vDeleteWords.insert(make_pair("That", 0.0));
    vDeleteWords.insert(make_pair("the", 0.0));
    vDeleteWords.insert(make_pair("The", 0.0));
    vDeleteWords.insert(make_pair("be", 0.0));

//    vDeleteWords.insert(make_pair("Be", 0.0));
//    vDeleteWords.insert(make_pair("Of", 0.0));
//    vDeleteWords.insert(make_pair("of", 0.0));
//    vDeleteWords.insert(make_pair("A", 0.0));
//    vDeleteWords.insert(make_pair("a", 0.0));
//    vDeleteWords.insert(make_pair("And", 0.0));
//    vDeleteWords.insert(make_pair("and", 0.0));
//    vDeleteWords.insert(make_pair("By", 0.0));
//    vDeleteWords.insert(make_pair("by", 0.0));
//    vDeleteWords.insert(make_pair("In", 0.0));
//    vDeleteWords.insert(make_pair("in", 0.0));
//    vDeleteWords.insert(make_pair("For", 0.0));
//    vDeleteWords.insert(make_pair("for", 0.0));

}


vector<Vec2> loopClosing::ProjTextInKF(mapText* obj, keyframe* KF, bool &FlagPred)
{
    vector<Vec2> Proj;
    FlagPred = true;

    Mat44 Tcr = KF->mTcw * obj->RefKF->mTcw.inverse();
    Mat31 theta = obj->RefKF->mNcr[obj->GetNidx()];
    vector<Vec2> refDeteBox = obj->vTextDeteRay;
    for(size_t ibox=0; ibox<refDeteBox.size(); ibox++){
        Mat31 ray = Mat31(refDeteBox[ibox](0), refDeteBox[ibox](1), 1.0);
        double invz = -ray.transpose() * theta;
        Mat31 p = KF->mK * ( Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3) );
        if(p(2)<0){
            FlagPred = false;
        }
        Proj.push_back(Vec2( p(0)/p(2), p(1)/p(2) ));
    }

    return Proj;
}

vector<Vec2> loopClosing::ProjTextInKF(mapText* obj, const Mat44 &Tcw, bool &FlagPred)
{
    vector<Vec2> Proj;
    FlagPred = true;
    Mat44 Tcr = Tcw * obj->RefKF->mTcw.inverse();
    Mat31 theta = obj->RefKF->mNcr[obj->GetNidx()];
    vector<Vec2> refDeteBox = obj->vTextDeteRay;
    for(size_t ibox=0; ibox<refDeteBox.size(); ibox++){
        Mat31 ray = Mat31(refDeteBox[ibox](0), refDeteBox[ibox](1), 1.0);
        double invz = -ray.transpose() * theta;
        Mat31 p = obj->RefKF->mK * ( Tcr.block<3,3>(0,0) * ray/invz + Tcr.block<3,1>(0,3) );
        if(p(2)<0){
            FlagPred = false;
        }
        Proj.push_back(Vec2( p(0)/p(2), p(1)/p(2) ));
    }

    return Proj;
}

int loopClosing::SearchMatch(keyframe* CanKF, const vector<vector<MatchmapTextRes>> &vMatchTexts, vector<FeatureConvert> &vFeatCur, vector<FeatureConvert> &vFeatCan)
{
    // 1. text
    vector<FeatureConvert> vFeatCur_Text, vFeatCan_Text;
    vector<cv::Mat> TextLabel;
    int nMatchText = SearchMatch_Text(CanKF, vMatchTexts, vFeatCur_Text, vFeatCan_Text, TextLabel);
    vFeatCur.insert(vFeatCur.end(), vFeatCur_Text.begin(), vFeatCur_Text.end());
    vFeatCan.insert(vFeatCan.end(), vFeatCan_Text.begin(), vFeatCan_Text.end());

    // 2. Other
    vector<FeatureConvert> vFeatCur_Other, vFeatCan_Other;
    int win = 50;
    int nMatchOther = SearchMatch_Other(CanKF, TextLabel, vFeatCur_Other, vFeatCan_Other, win);
    vFeatCur.insert(vFeatCur.end(), vFeatCur_Other.begin(), vFeatCur_Other.end());
    vFeatCan.insert(vFeatCan.end(), vFeatCan_Other.begin(), vFeatCan_Other.end());

    int nMatch = nMatchText + nMatchOther;

    return nMatch;
}

int loopClosing::SearchMatch_Text(keyframe* CanKF, const vector<vector<MatchmapTextRes>> &vMatchTexts,
                                  vector<FeatureConvert> &vFeatCur, vector<FeatureConvert> &vFeatCan, vector<cv::Mat> &TextLabel)
{
    cv::Mat BackImgCur = cv::Mat::ones(CanKF->FrameImg.rows, CanKF->FrameImg.cols, CV_32F)*(-1.0);
    cv::Mat BackImgCan = cv::Mat::ones(CanKF->FrameImg.rows, CanKF->FrameImg.cols, CV_32F)*(-1.0);

    vector<TextObservation*> vObvMapTexts = mpCurrentKF->vObvText;
    assert(vObvMapTexts.size()==vMatchTexts.size());

    for(size_t iObvText = 0; iObvText<vObvMapTexts.size(); iObvText++)
    {
        mapText* ObjCur = vObvMapTexts[iObvText]->obj;
        vector<int> vObvIdxCur = vObvMapTexts[iObvText]->idx;

        if(vObvIdxCur.size()==0)
        {
            continue;
        }

        int idxCur = vObvIdxCur[0];
        // ************************************************ //

        vector<MatchmapTextRes> MatchTexts = vMatchTexts[iObvText];
        if(MatchTexts.size()<=0)
            continue;

        for(size_t iMatchRes = 0; iMatchRes<MatchTexts.size(); iMatchRes++)
        {
            mapText* ObjMap = MatchTexts[iMatchRes].mapObj;
            vector<int> vObvIdxCan;
            bool FLAG_CANKFOBV = ObjMap->GetObvIdx(CanKF, vObvIdxCan);

            if(!FLAG_CANKFOBV || vObvIdxCan.size()==0)
            {
                continue;
            }
            int idxCan = vObvIdxCan[0];

            // Match : Cur idxCur VS Can idxCa
            bool USETHRESH = true;
            if(CanKF->vKeysText[idxCan].size()<=1)
                continue;
            vector<DMatch> match12 = FeatureMatch_brute(mpCurrentKF->mDescrText[idxCur], CanKF->mDescrText[idxCan], USETHRESH);

            // convert parameters
            vector<FeatureConvert> vFeatCurOne, vFeatCanOne;
            FeatureConvert_Text(match12, mpCurrentKF, CanKF, ObjCur, ObjMap, idxCur, idxCan, vFeatCurOne, vFeatCanOne);
            vFeatCur.insert(vFeatCur.end(), vFeatCurOne.begin(), vFeatCurOne.end());
            vFeatCan.insert(vFeatCan.end(), vFeatCanOne.begin(), vFeatCanOne.end());

            BackImgCur = Tool.GetTextLabelMask(BackImgCur, mpCurrentKF->vTextDete[idxCur], (int)ObjCur->mnId);
            BackImgCan = Tool.GetTextLabelMask(BackImgCan, CanKF->vTextDete[idxCan], (int)ObjMap->mnId);

        }       // matched map text
    }           // observed text

    TextLabel.push_back(BackImgCur);
    TextLabel.push_back(BackImgCan);

    assert(vFeatCur.size()==vFeatCan.size());     // match number
    return (int)vFeatCur.size();

}

int loopClosing::SearchMatch_Other(keyframe* CanKF, const vector<cv::Mat> TextLabel, vector<FeatureConvert> &vFeatCur, vector<FeatureConvert> &vFeatCan,
                                   const int &Win)
{
    cv::Mat TextLabelCur = TextLabel[0];
    cv::Mat TextLabelCan = TextLabel[1];

    vector<int> vMatchIdx12;
    int nMatches = 0;
    int N1 = mpCurrentKF->vKeys.size();
    int N2 = CanKF->vKeys.size();
    vMatchIdx12 = vector<int>(N1, -1);
    vector<int> vMatchDist(N2, INT_MAX);
    vector<int> vMatchIdx21(N2, -1);

    // each feature1
    for(size_t i1 = 0; i1<N1; i1++){
        cv::KeyPoint Fea = mpCurrentKF->vKeys[i1];

        // Cond1. if not corresponding to 3D info (check)
        if(mpCurrentKF->vTextObjInfo[i1]<0){
            if(mpCurrentKF->vMatches2D3D[i1]<0)
                continue;
        }else{
            int idxDete = mpCurrentKF->vTextObjInfo[i1];
            int mnIdText = mpCurrentKF->vTextDeteCorMap[idxDete];
            if(mnIdText<0)
                continue;
        }

        // Cond2. text not cover
        int u = round( Fea.pt.x );
        int v = round( Fea.pt.y );
        float* img_ptr = (float*)TextLabelCur.data + v*TextLabelCur.cols + u;
        float label = img_ptr[0];

        if(label>=0)
            continue;

        cv::Mat d1 = mpCurrentKF->mDescr.row(i1);
        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;
        for(size_t i2=0; i2<N2; i2++)
        {
            // Cond1. if not corresponding to 3D info
            if(CanKF->vTextObjInfo[i2]<0){
                // scene feature
                if(CanKF->vMatches2D3D[i2]<0)
                    continue;
            }else{
                // text feature
                int idxDete = CanKF->vTextObjInfo[i2];
                int mnIdText = CanKF->vTextDeteCorMap[idxDete];
                if(mnIdText<0)
                    continue;
            }

            // Cond2. text not cover
            int u2 = round(CanKF->vKeys[i2].pt.x);
            int v2 = round(CanKF->vKeys[i2].pt.y);
            float* img_ptr2 = (float*)TextLabelCan.data + v2*TextLabelCan.cols + u2;
            float label2 = img_ptr2[0];

            if(label2>=0)
                continue;

            cv::Mat d2 = CanKF->mDescr.row(i2);
            int dist = DescriptorDistance(d1, d2);

            if(vMatchDist[i2]<=dist)
                continue;

            if(dist<bestDist){
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            }else if(dist<bestDist2){
                bestDist2 = dist;
            }

        }   // all KF2 features

        if(bestDist<=TH_LOW){
            if(bestDist<(double)bestDist2*0.9){
                if(vMatchIdx21[bestIdx2]>=0){
                    vMatchIdx12[vMatchIdx21[bestIdx2]] = -1;
                    nMatches--;
                }
                vMatchIdx12[i1] = bestIdx2;
                vMatchIdx21[bestIdx2] = i1;
                vMatchDist[bestIdx2] = bestDist;
                nMatches++;
            }
        }
    }   // all KF1 features

    Tool.ShowMatches(vMatchIdx12, mpCurrentKF->FrameImg, CanKF->FrameImg, mpCurrentKF->vKeys, CanKF->vKeys, false);

    // convert parameter
    int numMatch = FeatureConvert_Other(vMatchIdx12, mpCurrentKF, CanKF, vFeatCur, vFeatCan);

   return numMatch;
}

int loopClosing::FeatureConvert_Text(const vector<DMatch> &match12, keyframe* KF1, keyframe* KF2,
                                     mapText* obj1, mapText* obj2,
                                     const int &idxDete1, const int &idxDete2, vector<FeatureConvert> &vFeat1, vector<FeatureConvert> &vFeat2)
{

    Mat44 Thr1 = obj1->RefKF->mTcw * KF1->mTwc;
    Mat44 Thr2 = obj2->RefKF->mTcw * KF2->mTwc;
    Mat31 theta1 = Tool.TransTheta(obj1->RefKF->mNcr[obj1->GetNidx()], Thr1);
    Mat31 theta2 = Tool.TransTheta(obj2->RefKF->mNcr[obj2->GetNidx()], Thr2);

    for(size_t i=0; i<match12.size(); i++){
        int idx1 = match12[i].queryIdx;
        int idx2 = match12[i].trainIdx;

        int idx1_Keys = KF1->vKeysTextIdx[idxDete1][idx1];
        int idx2_Keys = KF2->vKeysTextIdx[idxDete2][idx2];

        cv::KeyPoint kpt1 = KF1->vKeys[idx1_Keys];
        cv::KeyPoint kpt2 = KF2->vKeys[idx2_Keys];

        assert(kpt1.pt.x==KF1->vKeysText[idxDete1][idx1].pt.x);
        assert(kpt1.pt.y==KF1->vKeysText[idxDete1][idx1].pt.y);
        assert(kpt2.pt.x==KF2->vKeysText[idxDete2][idx2].pt.x);
        assert(kpt2.pt.y==KF2->vKeysText[idxDete2][idx2].pt.y);

        // text feature pos
        cv::KeyPoint feat1 = KF1->vKeys[idx1_Keys];
        Mat31 ray1 = Mat31( (feat1.pt.x-KF1->cx)/KF1->fx , (feat1.pt.y-KF1->cy)/KF1->fy , 1.0);
        double invz1 = -ray1.transpose() * theta1;
        Mat31 P1 = ray1/invz1;
        Mat31 Pw1 = KF1->mTwc.block<3,3>(0,0)*P1 + KF1->mTwc.block<3,1>(0,3);
        Mat31 P1img1 = KF1->mK * P1;

        cv::KeyPoint feat2 = KF2->vKeys[idx2_Keys];
        Mat31 ray2 = Mat31( (feat2.pt.x-KF1->cx)/KF1->fx , (feat2.pt.y-KF1->cy)/KF1->fy , 1.0);
        double invz2 = -ray2.transpose() * theta2;
        Mat31 P2 = ray2/invz2;
        Mat31 Pw2 = KF2->mTwc.block<3,3>(0,0) * P2 + KF2->mTwc.block<3,1>(0,3);
        Mat31 P2img2 = KF2->mK * P2;

        struct FeatureConvert featInfo1 = FeatureConvert{Pw1, P1, 1, obj1, static_cast<mapPts*>(NULL), KF1, idx1_Keys, feat1, Vec2( P1img1(0)/P1img1(2), P1img1(1)/P1img1(2) )};
        struct FeatureConvert featInfo2 = FeatureConvert{Pw2, P2, 1, obj2, static_cast<mapPts*>(NULL), KF2, idx2_Keys, feat2, Vec2( P2img2(0)/P2img2(2), P2img2(1)/P2img2(2) )};

        vFeat1.push_back(featInfo1);
        vFeat2.push_back(featInfo2);
    }

}

int loopClosing::FeatureConvert_Other(const vector<int> &vMatchIdx12, keyframe* KF1, keyframe* KF2, vector<FeatureConvert> &vFeat1, vector<FeatureConvert> &vFeat2)
{
    int numMatch = 0;
    for(size_t ifeat=0; ifeat<vMatchIdx12.size(); ifeat++){
        if(vMatchIdx12[ifeat]<0)
            continue;

        int idx1 = ifeat;
        int idx2 = vMatchIdx12[ifeat];

        FeatureConvert feat1 = GetConvertInfo(KF1, idx1);
        FeatureConvert feat2 = GetConvertInfo(KF2, idx2);

        if(feat1.FlagTS==0 && feat2.FlagTS==0)
        {
            if(feat1.pt->mnId==feat2.pt->mnId)
                continue;
        }else if(feat1.FlagTS==1 && feat2.FlagTS==1)
        {
            if(feat1.obj->mnId==feat2.obj->mnId)
                continue;
        }

        vFeat1.push_back(feat1);
        vFeat2.push_back(feat2);
        numMatch++;
    }
    return numMatch;
}

FeatureConvert loopClosing::GetConvertInfo(keyframe* KF, const int &Idx)
{
    if(KF->vTextObjInfo[Idx]<0){
        // scene feature
        int mnIdPts = KF->vMatches2D3D[Idx];
        mapPts* MapPt = mpMap->GetPtFromId(mnIdPts);
        assert(MapPt->mnId==mnIdPts);

        // pos
        Mat31 Pw = MapPt->GetxyzPos();
        Mat31 PKF = KF->mTcw.block<3,3>(0,0) * Pw + KF->mTcw.block<3,1>(0,3);
        Mat31 Pimg = KF->mK * PKF;
        struct FeatureConvert featInfo = FeatureConvert{Pw, PKF, 0, static_cast<mapText*>(NULL), MapPt, KF, Idx, KF->vKeys[Idx], Vec2( Pimg(0)/Pimg(2), Pimg(1)/Pimg(2) )};
        return featInfo;

    }else{
        // text feature
        int idxDete = KF->vTextObjInfo[Idx];
        int mnIdText = KF->vTextDeteCorMap[idxDete];
        mapText* MapObj = mpMap->GetObjFromId(mnIdText);
        assert(MapObj->mnId==mnIdText);

        // pos
        Mat44 Thr = MapObj->RefKF->mTcw * KF->mTwc;
        Mat31 theta = Tool.TransTheta(MapObj->RefKF->mNcr[MapObj->GetNidx()], Thr);
        cv::KeyPoint feat = KF->vKeys[Idx];
        double m1 = (feat.pt.x - KF->cx)/KF->fx;
        double m2 = (feat.pt.y - KF->cy)/KF->fy;
        Mat31 ray = Mat31(m1, m2, 1.0);
        double invz = -ray.transpose() * theta;
        Mat31 PKF = ray/invz;
        Mat31 Pw = KF->mTwc.block<3,3>(0,0) * PKF + KF->mTwc.block<3,1>(0,3);
        Mat31 Pimg = KF->mK * PKF;

        struct FeatureConvert featInfo = FeatureConvert{Pw, PKF, 1, MapObj, static_cast<mapPts*>(NULL), KF, Idx, KF->vKeys[Idx], Vec2( Pimg(0)/Pimg(2), Pimg(1)/Pimg(2) )};
        return featInfo;
    }
}

void loopClosing::GetLoopsLandmarkers(std::map<mapPts*, keyframe*> &vLoopPts, std::map<mapText*, keyframe*> &vLoopObjs)
{
    vector<CovKF> vConnectOfLoopKF = mpMatchedKF->GetCovisibleKFs_All();
    vector<CovKF> vConnects;
    vConnects.push_back(make_pair(mpMatchedKF, 0));
    vConnects.insert(vConnects.end(), vConnectOfLoopKF.begin(), vConnectOfLoopKF.end());

    for(size_t iKF=0; iKF<vConnects.size(); iKF++){
        keyframe* KF = vConnects[iKF].first;
        vector<SceneObservation*> vKFPts = KF->vObvPts;
        vector<TextObservation*> vKFTexts = KF->vObvText;
        assert((int)KF->vObvGoodPts.size()==(int)KF->vObvPts.size());
        assert((int)KF->vObvGoodTexts.size()==(int)KF->vObvText.size());
        cv::Mat ImgDraw = KF->FrameImg.clone();

        for(size_t iPt=0; iPt<vKFPts.size(); iPt++){
            mapPts* Mpt = vKFPts[iPt]->pt;
            if(Mpt->FLAG_BAD)
                continue;

            if(Mpt->ReplaceKF)
                if(Mpt->ReplaceKF->mnId==mpCurrentKF->mnId)
                    continue;

            if(!vLoopPts.count(Mpt))
                vLoopPts.insert(make_pair(Mpt, KF));
        }

        for(size_t iObj=0; iObj<vKFTexts.size(); iObj++){
            mapText* Mobj = vKFTexts[iObj]->obj;
            if(Mobj->STATE==TEXTBAD)
                continue;

            if( Mobj->ReplaceKF )
                if(Mobj->ReplaceKF->mnId==mpCurrentKF->mnId)
                    continue;

            if(!vLoopObjs.count(Mobj))
                vLoopObjs[Mobj] = KF;

        }

    }   // each KF whthin the LoopKF window

}

void loopClosing::SearchAndFuse(const std::map<mapPts*, keyframe*> &vLoopPts, const std::map<mapText*, keyframe*> &vLoopObjs,
                                const std::map<keyframe*, Sim3_loop, std::less<keyframe*>,  Eigen::aligned_allocator<std::pair<keyframe*, Sim3_loop>> > &vConnectKFs, const bool &AddCurrent,
                                vector<Eigen::MatrixXd> &vMs)
{
    if(!AddCurrent)
    {
        keyframe* KFi = mpCurrentKF;
        Sim3_loop Siw = mScw;

        // loop landmarker, need replaced landmarker
        std::map<mapPts*, mapPts*> vReplacePts;
        std::map<mapText*, mapText*> vReplaceObjs;

        // scene: use observation match
        double th = 15.0;
        int nFuseScene = SearchAndFuse_Scene(KFi, Siw, vLoopPts, vReplacePts, th, vMs[0]);

        // text: judge ProjCenter is inside LoopText Proj
        int nFuseText = SearchAndFuse_Text(KFi, Siw, vLoopObjs, vReplaceObjs);

        // replace
        std::map<mapPts*, mapPts*>::iterator IterPt;
        for(IterPt=vReplacePts.begin(); IterPt!=vReplacePts.end(); IterPt++){
            mapPts* ptLoop = IterPt->first;
            mapPts* pt = IterPt->second;

            pt->Replace(mpCurrentKF, ptLoop, vMs[0]);
        }

        std::map<mapText*, mapText*>::iterator IterObj;
        for(IterObj=vReplaceObjs.begin(); IterObj!=vReplaceObjs.end(); IterObj++)
        {
            mapText* objLoop = IterObj->first;
            mapText* obj = IterObj->second;

            obj->Replace(mpCurrentKF, objLoop, vMs[1], vMs[2]);
        }
    }

    std::map<keyframe*, Sim3_loop>::const_iterator iterkf;
    for(iterkf=vConnectKFs.begin(); iterkf!=vConnectKFs.end(); iterkf++)
    {
        keyframe* KFi = iterkf->first;
        Sim3_loop Siw = iterkf->second;

        // loop landmarker, need replaced landmarker
        std::map<mapPts*, mapPts*> vReplacePts;
        std::map<mapText*, mapText*> vReplaceObjs;

        // scene: use observation match
        double th = 15.0;
        int nFuseScene = SearchAndFuse_Scene(KFi, Siw, vLoopPts, vReplacePts, th, vMs[0]);

        // text: judge ProjCenter is inside LoopText Proj
        int nFuseText = SearchAndFuse_Text(KFi, Siw, vLoopObjs, vReplaceObjs);

        // replace
        std::map<mapPts*, mapPts*>::iterator IterPt;
        for(IterPt=vReplacePts.begin(); IterPt!=vReplacePts.end(); IterPt++){
            mapPts* ptLoop = IterPt->first;
            mapPts* pt = IterPt->second;

            pt->Replace(mpCurrentKF, ptLoop, vMs[0]);
        }

        std::map<mapText*, mapText*>::iterator IterObj;
        for(IterObj=vReplaceObjs.begin(); IterObj!=vReplaceObjs.end(); IterObj++)
        {
            mapText* objLoop = IterObj->first;
            mapText* obj = IterObj->second;

            obj->Replace(mpCurrentKF, objLoop, vMs[1], vMs[2]);
        }
     }      // vConnectKFs

}

int loopClosing::SearchAndFuse_Scene(keyframe* KF, const Sim3_loop &Scw, const std::map<mapPts *, keyframe *> &vLoopPts,
                                     std::map<mapPts*, mapPts*> &vReplacePts, double th, Eigen::MatrixXd &M1)
{
    // pose --------
    Mat33 Rcw(Scw.r);
    Mat31 tcw = Scw.t;
    double scw = Scw.s;
    tcw /= scw;
    // [R, t/s]
    Mat44 Tcw;
    Tcw.setIdentity();
    Tcw.block<3,3>(0,0) = Rcw;
    Tcw.block<3,1>(0,3) = tcw;
    // pose --------

    vector<SceneObservation*> vObvPts_rawKF = KF->vObvPts;

    int nFused=0, nHasFused = 0, nAdd=0;

    std::map<mapPts *, keyframe *>::const_iterator iLpt;
    for(iLpt=vLoopPts.begin(); iLpt!=vLoopPts.end(); iLpt++)
    {
        mapPts* Pt_loop = iLpt->first;

        // Discard Bad mapPts and already found
        if(Pt_loop->FLAG_BAD || Pt_loop->IsInKeyFrame(KF) )
            continue;

        Mat44 Tcr = Tcw * Pt_loop->RefKF->mTcw.inverse();

        Mat31 Pr = Pt_loop->GetRaydir()/Pt_loop->GetInverD();
        Mat31 Pc = Tcr.block<3,3>(0,0) * Pr + Tcr.block<3,1>(0,3) ;

        // Discard negtive depth points
        if(Pc(2,0)<0.0)
            continue;

        Mat31 Pc_proj = KF->mK * Pc;
        double u = Pc_proj(0)/Pc_proj(2);
        double v = Pc_proj(1)/Pc_proj(2);

        // projection inside image
        if(!KF->IsInImage(u,v))
            continue;

        double radius = th;
        const vector<size_t> vIndices = KF->GetFeaturesInArea(u,v,radius);

        vector<size_t> Idx1Show;
        Idx1Show.push_back(0);

        if(vIndices.empty())
            continue;

        int Idx_loopKF;
        keyframe* KF_loopPt;
        if(Pt_loop->GetKFObv(mpMatchedKF, Idx_loopKF))
            KF_loopPt = mpMatchedKF;
        else{
            KF_loopPt = iLpt->second;
            bool IN = Pt_loop->GetKFObv(KF_loopPt, Idx_loopKF);
            assert(IN);
        }
        assert(Idx_loopKF>=0);

        cv::Mat dMP = KF_loopPt->mDescr.row(Idx_loopKF);

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const cv::Mat &dKF = KF->mDescr.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            int PtmnId = KF->vMatches2D3D[bestIdx];

            if(PtmnId<0){
                Pt_loop->AddObserv(KF, bestIdx);
                Pt_loop->SetReplaceKF(mpCurrentKF);
                KF->AddSceneObserv(Pt_loop, bestIdx);

                Pt_loop->UpdateCovMap_1(KF, Pt_loop, M1);

                nAdd++;
                continue;
            }

            mapPts* PtRaw = mpMap->GetPtFromId(PtmnId);

            // if PtRaw is not be replaced, PtRaw is needed to be replaced by the loopPt
            bool ISOLDPT = true;
            if(PtRaw->ReplaceKF){
                if(PtRaw->ReplaceKF->mnId==mpCurrentKF->mnId)
                    ISOLDPT = false;
            }
            if(ISOLDPT)
            {
                vReplacePts[Pt_loop] = PtRaw;
                nFused++;
            }
            else
                nHasFused++;

            nHasFused++;
        }
    }    // each loop pts

    return nFused;
}

int loopClosing::SearchAndFuse_Text(keyframe* KF, const Sim3_loop &Scw, const std::map<mapText *, keyframe *> &vLoopObjs, std::map<mapText *, mapText *> &vReplaceObjs)
{
    cv::Mat BackImg = cv::Mat::ones(KF->FrameImg.rows, KF->FrameImg.cols, CV_32F)*(-1.0);
    cv::Mat ImgDraw = KF->FrameImg.clone();
    int nFuse = 0;

    // pose --------
    Mat33 Rcw(Scw.r);
    Mat31 tcw = Scw.t;
    double scw = Scw.s;
    tcw /= scw;
    // [R, t/s]
    Mat44 Tcw;
    Tcw.setIdentity();
    Tcw.block<3,3>(0,0) = Rcw;
    Tcw.block<3,1>(0,3) = tcw;
    // pose --------

    // 1. loopMapTexts project to current view，get labelMap
    std::map<mapText *, keyframe *>::const_iterator IterObj;
    vector<mapText*> vLoopObjsV;
    int idx = 0;

    for(IterObj=vLoopObjs.begin(); IterObj!=vLoopObjs.end(); IterObj++)
    {
        mapText* obj = IterObj->first;
        bool flag_pred;
        vector<Vec2> Pred = ProjTextInKF(obj, Tcw, flag_pred);
        if(!flag_pred){
            continue;
        }

        bool IN = false, ALLIN = true;
        for(size_t i=0; i<Pred.size(); i++){
            bool IN_1PT = KF->IsInImage( Pred[i](0,0),Pred[i](1,0) );
            if(!IN_1PT)
                ALLIN = false;
            if(IN_1PT)
                IN = true;
        }

        if(!IN)
            continue;

        ImgDraw = Tool.ShowTextBoxSingle(ImgDraw, Pred, obj->mnId);
        BackImg = Tool.GetTextLabelMask(BackImg, Pred, idx);

        vLoopObjsV.push_back(obj);
        idx++;
    }

    // 2. get overlaped text objects
    vector<TextObservation*> vTextObvs = KF->vObvText;
    vector<cv::KeyPoint> vCentToShow;
    for(size_t iobj=0; iobj<vTextObvs.size(); iobj++){
        mapText* obj = vTextObvs[iobj]->obj;

        if(obj->ReplaceKF)
            if(obj->ReplaceKF->mnId==mpCurrentKF->mnId)
                continue;

        bool flag_pred;
        vector<Vec2> Pred = ProjTextInKF(obj, KF, flag_pred);
        if(!flag_pred){
            continue;
        }

        ImgDraw = Tool.ShowTextBoxSingle(ImgDraw, Pred, obj->mnId);

        Vec2 CentObj = Tool.GetMean(Pred);

        int u = round( CentObj(0,0) );
        int v = round( CentObj(1,0) );
        bool OUT = (u < 0 || v < 0 || u >= BackImg.cols || v >= BackImg.rows);
        float label;
        if(!OUT){
            float* img_ptr = (float*)BackImg.data + v*BackImg.cols + u;
            label = img_ptr[0];
        }else{
            label = -1;
        }

        cv::KeyPoint kpt;
        kpt.pt.x = CentObj(0,0);
        kpt.pt.y = CentObj(1,0);
        vCentToShow.push_back(kpt);

        if(label>=0){
            vReplaceObjs[ vLoopObjsV[label] ] = obj;
            nFuse++;
        }
    }

    return nFuse;

}


void loopClosing::UpdateCovisibleKFs()
{
    vector<keyframe*> vKFs = mpMap->GetAllKeyFrame();

    for(size_t ikfs=0; ikfs<vKFs.size(); ikfs++){
        GetCovisibleKFs_all(vKFs[ikfs], mpMap->imapkfs);
    }

}

int loopClosing::MatchMore(keyframe* KF1, keyframe* KFMatch2, const Sim3_loop& gscm, vector<FeatureConvert> &vFeatCur, vector<FeatureConvert> &vFeatCan)
{
    float th = 15.0;
    double th_high = 60;
    bool SHOW = false;
    cv::Mat FrameImg1 = KF1->FrameImg.clone();
    cv::Mat FrameImg2 = KFMatch2->FrameImg.clone();

    vector<SceneObservation*> vPts = KFMatch2->vObvPts;

    vector<int> vMatch2D3D = vector<int>(KF1->iN, -1);
    vector<int> vMatch3D2D = vector<int>(vPts.size(), -1);
    vector<int> vMatch21 = vector<int>(KFMatch2->iN, -1);
    vector<int> vMatch12 = vector<int>(KF1->iN, -1);
    int nMatches = 0;

    for(size_t i0=0; i0<vPts.size(); i0++){
        if(!KFMatch2->vObvGoodPts[i0])
            continue;
        if(vPts[i0]->pt->FLAG_BAD)
            continue;

        int idxPt_KF2 = KFMatch2->vObvPts[i0]->idx;

        mapPts* mpt = vPts[i0]->pt;
        Mat31 raydir = mpt->GetRaydir();
        double rho = mpt->GetInverD();
        double invrho = 1.0/rho;
        Mat44 Trh = KFMatch2->mTcw * mpt->RefKF->mTwc;
        Eigen::Quaterniond qcm = gscm.r;
        Mat33 sRcm(qcm);
        Mat31 PKF2 = Trh.block<3,3>(0,0) * invrho * raydir + Trh.block<3,1>(0,3);
        Mat31 Pcam = KF1->mK * (sRcm * PKF2 + gscm.t);
        double u = Pcam(0)/Pcam(2);
        double v = Pcam(1)/Pcam(2);

        if(u<KF1->mnMinX || u>KF1->mnMaxX)
            continue;
        if(v<KF1->mnMinY || v>KF1->mnMaxY)
            continue;

        float radius = th*1.2f;
        vector<size_t> vIndices1 = KF1->GetFeaturesInArea(u,v,radius);
        if(vIndices1.empty())
            continue;

        cv::Mat dMP = KFMatch2->mDescr.row(idxPt_KF2);
        int bestDist = INT_MAX;
        int bestIdx1 = -1;
        for(vector<size_t>::const_iterator vit=vIndices1.begin(), vend=vIndices1.end(); vit!=vend; vit++)
        {
            const size_t i1 = *vit;
            const cv::Mat &d = KF1->mDescr.row(i1);
            const int dist = DescriptorDistance(dMP,d);

            if(dist<bestDist)
            {
                bestDist=dist;
                bestIdx1=i1;
            }
        }

        if(bestDist<=th_high){
                // skip same observation in current frame or Last keyframe
            if(vMatch2D3D[bestIdx1]<0 && vMatch21[idxPt_KF2]<0 )
            {
                bool f_3D = false;
                if(KF1->vTextObjInfo[bestIdx1]<0){
                    if(KF1->vMatches2D3D[bestIdx1]>=0)
                        f_3D = true;
                }else{
                    int idxDete = KF1->vTextObjInfo[bestIdx1];
                    int mnIdText = KF1->vTextDeteCorMap[idxDete];
                    if(mnIdText>=0)
                        f_3D = true;
                }
                if(f_3D)
                {
                    nMatches++;
                    vMatch3D2D[i0] = bestIdx1;
                    vMatch2D3D[bestIdx1] = i0;
                    vMatch21[idxPt_KF2] = bestIdx1;
                    vMatch12[bestIdx1] = idxPt_KF2;
                }
            }
        }
    }

    int n_matches_new = FeatureConvert_Other(vMatch12, KF1, KFMatch2, vFeatCur, vFeatCan);

    return nMatches;
}

vector<DMatch> loopClosing::FeatureMatch_brute(cv::Mat &Descrip1, cv::Mat &Descrip2, const bool &USETHRESH)
{
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create ( "BruteForce-Hamming" );
    vector<DMatch> matches;
    matcher->match ( Descrip1, Descrip2, matches );

    if(!USETHRESH){
        return matches;
    }else{
        double min_dist=10000, max_dist=0;
        for ( int i = 0; i < Descrip1.rows; i++ )
        {
            double dist = matches[i].distance;
            if ( dist < min_dist ) min_dist = dist;
            if ( dist > max_dist ) max_dist = dist;
        }

        std::vector< DMatch > good_matches;
        for ( int i = 0; i < Descrip1.rows; i++ )
        {
            if ( matches[i].distance < max ( 2*min_dist, 30.0 ) )
            {
                good_matches.push_back ( matches[i] );
            }
        }

        return good_matches;
    }
}

int loopClosing::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}


void loopClosing::GetCovisibleKFs_all(keyframe* KF, const int &CurKFNum)
{
    int Use = 0;    // 0: M1; 1: M2; 2: M3
    vector<CovKF> vKFConKFsAll, vKFConKFs_prev, vKFConKFs_after;

    Eigen::MatrixXd MAll = mpMap->GetCovMap(Use);
    Eigen::MatrixXd Mkf_col = MAll.col(KF->mnId);
    Eigen::MatrixXd Mkf_row = MAll.row(KF->mnId);

    for(size_t iConnect=0; iConnect<KF->mnId; iConnect++){
        vKFConKFs_prev.push_back( make_pair( mpMap->GetKFFromId(iConnect), Mkf_col(iConnect, 0) ) );
    }
    for(size_t iConnect=(KF->mnId+1); iConnect<CurKFNum; iConnect++){
        vKFConKFs_after.push_back( make_pair( mpMap->GetKFFromId(iConnect), Mkf_row(0, iConnect) ) );
    }

    sort(vKFConKFs_prev.begin(), vKFConKFs_prev.end(), cmpLarge_kf());
    sort(vKFConKFs_after.begin(), vKFConKFs_after.end(), cmpLarge_kf());

    vector<CovKF> vKFConKFsPrev = Tool.GetAllNonZero(vKFConKFs_prev);
    vector<CovKF> vKFConKFsAfter = Tool.GetAllNonZero(vKFConKFs_after);

    vKFConKFsAll.insert(vKFConKFsAll.end(), vKFConKFsPrev.begin(), vKFConKFsPrev.end());
    vKFConKFsAll.insert(vKFConKFsAll.end(), vKFConKFsAfter.begin(), vKFConKFsAfter.end());

    sort(vKFConKFsAll.begin(), vKFConKFsAll.end(), cmpLarge_kf());

    KF->SetCovisibleKFsAll(vKFConKFsAll);
    KF->SetCovisibleKFsPrev(vKFConKFsPrev);
    KF->SetCovisibleKFsAfter(vKFConKFsAfter);

}

}
