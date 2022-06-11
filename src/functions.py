#!/usr/bin/env python
#include DVR
#add p-value
#add more permutations to long clusters
#count a CpG into the DMR where it is a weak signal and is in the middle of strong signals.
#add multipleprocessing
"""
   Call DMseg.
"""
from __future__ import print_function
import numpy as np
import time
from time import localtime, strftime
import pandas as pd
import re
import multiprocessing as mp

#make clusters
def clustermaker(chr, pos, assumesorted=False, maxgap=500):
    tmp2 = chr.groupby(by=chr, sort=False)
    tmp3 = tmp2.count()
    Indexes = tmp3.cumsum().to_list()
    Indexes.insert(0, 0)
    clusterIDs = pd.Series(data=[None]*pos.shape[0], index=chr.index)
    Last = 0
    for i in range(len(Indexes)-1):
        i1 = Indexes[i]
        i2 = Indexes[i+1]
        Index = range(i1, i2)
        x = pos.iloc[Index]
        if (not(assumesorted)):
            tmp = [j-1 for j in x.rank()]
            x = x.iloc[tmp]
        y = np.diff(x) > maxgap
        y = np.insert(y, 0, 1)
        z = np.cumsum(y)
        clusterIDs.iloc[i1:i2] = z + Last
        Last = max(z) + Last
    return clusterIDs

#fit model for data
def fit_model_probes(beta, design):
    #use np array to save time
    beta1 = np.array(beta)
    design1 = np.array(design)
    M = np.delete(design1,1,axis=1)
    M_QR_q, M_QR_r = np.linalg.qr(M)
    S = np.diag([1] * M.shape[0]) - np.matmul(M_QR_q, M_QR_q.transpose())
    V = design1[:, 1]
    SV = np.matmul(S, V)
    coef = np.matmul(beta1, np.matmul(S.transpose(), V)) / np.matmul(V.transpose(), SV)
    # Calculate residuals
    QR_X_q, QR_X_r = np.linalg.qr(design)
  
    resids = np.diag([1] * design.shape[0]) - np.matmul(QR_X_q, QR_X_q.transpose())
    resids = np.matmul(resids, beta1.transpose())
 
    # Calculate SE
    tmp1 = np.linalg.inv(design1.T.dot(design1))[1, 1] / (beta.shape[1] - np.linalg.matrix_rank(M) - 1)
    SE = np.sqrt(np.multiply(resids, resids).sum(axis=0) * tmp1)
    result = np.array([coef,SE]).T
    return result


#Search peak segments
def Search_segments(DMseg_stats, cutoff,cutoff1):
    zscore = DMseg_stats['Coef']/DMseg_stats['SE']
    cutoff1 = abs(cutoff1)
    cutoff = abs(cutoff)
    myzscore = 1*(zscore.abs()>cutoff)
    myzscore1 = 1*(zscore.abs()>cutoff1)
    myzscore2 = myzscore*2 + myzscore1
    tmp = myzscore2.to_string(index=False)
    tmp = re.sub("\n",'',tmp)
    if myzscore.index.name is not None:
        tmp = re.sub(myzscore.index.name,'',tmp)
    tmp = re.sub(" ",'',tmp)
    #the index where a cpg in the middle has a weak signal (Z-score>zscore_cutoff1)
    tmp1 = [_.start() for _ in re.finditer("313",tmp)]
    if (len(tmp1)>0):
        myzscore.iloc[np.array(tmp1)+1]=1
    #direction: 1 if cpg has zscore > cutoff, 0 abs(zscore) < cutoff, -1 if zscore < -cutoff 
    direction = np.zeros(DMseg_stats.shape[0])
    #direction = np.where(zscore >= cutoff, 1, direction)
    #direction = np.where(zscore <= -cutoff, -1, direction)
    direction = np.where((zscore>0) & (myzscore == 1), 1, direction)
    direction = np.where((zscore<0) & (myzscore == 1), -1, direction)

    #direction1 is based on the absolute zscores.
    #direction1 = np.zeros(DMseg_stats.shape[0])
    #direction1 = np.where(abs(zscore) >= cutoff, 1, direction)
    direction1 = myzscore
    

    #segments are segments based on direction1 (a segment includes all connected CpGs with different direction); a segment can cross the border of a cluster
    tmp0 = 1*(np.diff(direction1) != 0)
    tmp0 = np.insert(tmp0, 0, 1)
    segments = np.cumsum(tmp0)

    #split a segment if it covers multiple clusters; a segment should be within a cluster
    allsegments = segments + DMseg_stats['cluster']
    tmp0 = 1*(np.diff(allsegments) != 0)
    tmp0 = np.insert(tmp0, 0, 1)
    allsegments = np.cumsum(tmp0)

    #allsegments are the final segments
    DMseg_stats['segment'] = allsegments
    DMseg_stats['direction'] = direction
    #segment1 are based on direction (consider signs)
    tmp0 = 1*(np.diff(direction) != 0)
    tmp0 = np.insert(tmp0, 0, 1)
    segments1 = np.cumsum(tmp0)
    allsegments1 = segments1 + DMseg_stats['cluster']
    tmp0 = 1*(np.diff(allsegments1) != 0)
    tmp0 = np.insert(tmp0, 0, 1)
    allsegments1 = np.cumsum(tmp0)
    DMseg_stats['segment1'] = allsegments1

    return DMseg_stats
    

#LRT function considering switches in a peak
def LRT_segment_nocorr(DMseg_stats):
    csegments = DMseg_stats.segment
    bdiff = np.abs(DMseg_stats.Coef)
    bvar= 1/pow(DMseg_stats.SE,2)
    DMseg_stats["bvar"] = bvar
    DMseg_stats['bdiff_bvar'] = bvar * bdiff
    #first compute b0,b1 based on segments using zscore (not absolute values)
    tmp = DMseg_stats.groupby(by="segment1")
    b0 = tmp['bvar'].sum()
    b1 = tmp["bdiff_bvar"].sum()
    sign = tmp['direction'].first()
    #the segment id using absolute zscore
    mysegment = tmp['segment'].first()
    #include the sign here
    tmp1 = pd.DataFrame({"b0":b0,"b1_sign":b1*sign,"bb1_divid_bb0_sign":b1/b0*sign,"segment":mysegment},index=b0.index)
    tmp1_groupby = tmp1.groupby(by="segment")
    bb0 = tmp1_groupby['b0'].sum()
    bb1 = tmp1_groupby['b1_sign'].sum()
    seg_mean = bb1/bb0
    tmp1["bb1_divid_bb0_sign"] = round(tmp1["bb1_divid_bb0_sign"],3)
    tmp1["bb1_divid_bb0_sign"] = tmp1["bb1_divid_bb0_sign"].astype(str)
    tmp1_groupby = tmp1.groupby(by="segment")
    #for segmean if there are swiches, show all
    seg_mean_all = tmp1_groupby.agg({"bb1_divid_bb0_sign": ";".join})
 
    groupcounts= csegments.groupby(csegments).count()
    seg_mean_vec = np.repeat(seg_mean,groupcounts)
    ## I corrected an error here, previous code is  lrt1=pow(bdiff-seg_mean_vec,2)*bvar
    ## bdiff has been taken to absolute value, need original scale for LRT calculation, this does not matter to lrt0
    lrt1=pow(DMseg_stats.Coef.to_numpy()-seg_mean_vec.to_numpy(),2)*bvar
    lrt0=pow(bdiff,2)*bvar
    lrt= lrt0.groupby(csegments).sum() - lrt1.groupby(csegments).sum()
    result = dict(lrt=lrt, seg_mean=seg_mean_all)
    return result

#wrap Search_segments + LRT
def Segment_satistics(Pstats1,clusterindex,pos,chr,diff_cutoff,zscore_cutoff=1.96,zscore_cutoff1=1.78):
    # Check ROI, region of interest, both: coef.cutoff, zsocre.cutoff ---------
    # clusters meet beta_diff_cutoff and zscore_cutoff

    Pstats1 = pd.DataFrame(Pstats1,index=clusterindex.index,columns=["Coef","SE"])
    Pstats1["Zscore"] = Pstats1["Coef"]/Pstats1["SE"]
    mycoef = 1*(Pstats1['Coef'].abs()>diff_cutoff)
    myzscore = 1*(Pstats1['Zscore'].abs()>zscore_cutoff)
    # myzscore1 = 1*(Pstats1['Zscore'].abs()>zscore_cutoff1)
    # myzscore2 = myzscore*2 + myzscore1
    # tmp = myzscore2.to_string(index=False)
    # tmp = re.sub("\n",'',tmp)
    # if myzscore.index.name is not None:
    #     tmp = re.sub(myzscore.index.name,'',tmp)
    # tmp = re.sub(" ",'',tmp)
    # #the index where a cpg in the middle has a weak signal (Z-score>zscore_cutoff1)
    # tmp1 = [_.start() for _ in re.finditer("313",tmp)]
    # if (len(tmp1)>0):
    #     myzscore.iloc[np.array(tmp1)+1]=1
    
    # tmp3 =myzscore.to_frame()
    # tmp3['index']=range(myzscore.shape[0])
    # tmp3['clusterindex']=clusterindex
    
    # tmp = myzscore.to_string(index=False)
    # tmp = re.sub("\n",'',tmp)
    # if myzscore.index.name is not None:
    #     tmp = re.sub(myzscore.index.name,'',tmp)
    # tmp1 = tmp3.groupby(clusterindex).index.first().to_list()
    # tmp2 = tmp3.groupby(clusterindex).index.last().to_list()
    # tmp2 = [j+1 for j in tmp2]
    # tmp4 =["11" in tmp[tmp1[j]:tmp2[j]] for j in range(len(tmp1))]

    coef_select=mycoef.groupby(clusterindex).sum()>1
    zscore_select=myzscore.groupby(clusterindex).sum()>1
    #zscore_select = tmp4

    groupcounts = clusterindex.groupby(clusterindex).count()
    coef_select_vec = np.repeat(coef_select, groupcounts)
    zscore_select_vec = np.repeat(zscore_select, groupcounts)
    ROIcluster=clusterindex[clusterindex.index[np.logical_and(coef_select_vec,zscore_select_vec)]]
    if len(ROIcluster) == 0:
        #print("No ROI been found, please consider to reduce the value of diff_cutoff")
        DMseg_out = pd.DataFrame(columns=["cluster","cluster_L","chr","start_cpg","start_pos","end_cpg","end_pos","n_cpgs","seg_mean","LRT"])
    else:
        DMseg_sd = Pstats1['SE'].loc[ROIcluster.index]
        DMseg_coef = Pstats1['Coef'].loc[ROIcluster.index]
   
        DMseg_pos = pos[ROIcluster.index]
        DMseg_chr = chr[ROIcluster.index]
    
        DMseg_stats = pd.concat([DMseg_coef, DMseg_sd, DMseg_chr, DMseg_pos, ROIcluster], axis=1)
        DMseg_stats.columns = ["Coef", "SE", "chr", "pos", "cluster"]

        DMseg_stats = Search_segments(DMseg_stats,zscore_cutoff,zscore_cutoff1)
    
        DMseg_clusterlen = DMseg_stats.groupby(by="cluster")['Coef'].count()
        #peaks
        DMseg_stats1 = DMseg_stats[DMseg_stats.segment *DMseg_stats.direction != 0]
        DMseg_stats1_groupby = DMseg_stats1.groupby(by="segment")
        # #peak should have #cpg>1
        tmp = DMseg_stats1_groupby['cluster'].transform('count').gt(1)
        DMseg_stats1 = DMseg_stats1.loc[tmp.index[tmp==True]]

        ## compute the LRT statistics

        tmp = LRT_segment_nocorr(DMseg_stats=DMseg_stats1)
        #segid = np.unique(DMseg_stats1.segment)

        DMseg_out = pd.DataFrame(columns=["cluster","cluster_L","chr","start_cpg","start_pos","end_cpg","end_pos","n_cpgs","seg_mean","LRT"],index=tmp['seg_mean'].index)
        #if (len(np.shape(tmp['seg_mean']))==2):
        #    DMseg_out["seg_mean"]=tmp['seg_mean'].squeeze()
        #else:
        DMseg_out["seg_mean"] = tmp['seg_mean']
        DMseg_out["LRT"]=tmp['lrt']

        DMseg_stats1["cpgname"] = DMseg_stats1.index
        DMgroup = DMseg_stats1.groupby("segment")
        DMseg_out["cluster"] = DMgroup["cluster"].first()
        DMseg_out["cluster_L"] = DMseg_clusterlen.loc[DMseg_out["cluster"]].to_list()
        DMseg_out["n_cpgs"]=DMgroup["segment"].count()
        DMseg_out["chr"]=DMgroup["chr"].first()
        DMseg_out["start_cpg"]=DMgroup["cpgname"].first()
        DMseg_out["end_cpg"]=DMgroup["cpgname"].last()
        DMseg_out["start_pos"] = DMgroup["pos"].first()
        DMseg_out["start_pos"] = DMseg_out["start_pos"].astype(int)
        DMseg_out["end_pos"] = DMgroup["pos"].last()
        DMseg_out["end_pos"] = DMseg_out["end_pos"].astype(int)
        #DMseg_out["numswitches"] = DMgroup["segment1"].last()-DMgroup["segment1"].first()
        DMseg_out = DMseg_out.reset_index(drop=True)
    return DMseg_out


# fit model for permutation, sequential
def fit_model_probes_sim(beta,design,seed,B):
    
    beta1 = np.array(beta)
    design1 = np.array(design)
    M = np.delete(design1,1,axis=1)
    M_QR_q, M_QR_r = np.linalg.qr(M)
    S = np.diag([1] * M.shape[0]) - np.matmul(M_QR_q, M_QR_q.transpose())
    V = design1[:, 1]
    SV = np.matmul(S, V)

    QR_X_q, QR_X_r = np.linalg.qr(design)
    resids0 = np.diag([1] * design.shape[0]) - np.matmul(QR_X_q, QR_X_q.transpose())
    tmp1 = np.linalg.inv(design1.T.dot(design1))[1, 1] / (beta.shape[1] - np.linalg.matrix_rank(M) - 1)

    coef = np.zeros((beta.shape[0],B))
    allSE = np.zeros((beta.shape[0],B))

    for i in range(B):
        np.random.seed(seed+i)
        idx = np.random.permutation(range(beta.shape[1]))
        mybeta = beta1[:,idx]
        coef[:,i] = np.matmul(mybeta, np.matmul(S.transpose(), V)) / np.matmul(V.transpose(), SV)
        resids = np.matmul(resids0, mybeta.transpose())
        allSE[:,i] = np.sqrt(np.multiply(resids, resids).sum(axis=0) * tmp1)

    result = np.concatenate((coef,allSE),axis=1)
    return result

#permutation, sequential version
def do_simulation1serial(beta,design,clusterindex,pos,chr,seed,diff_cutoff,zscore_cutoff,zscore_cutoff1,B):

    print("Start " + str(B) + " permutation: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    allsimulation=pd.DataFrame()
    beta1 = np.array(beta)
    design1 = np.array(design)
    #print("Start linear model fitting: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    Pstatsall =  fit_model_probes_sim(beta=beta1,design=design1,seed=seed,B=B)
    #strftime("%Y-%m-%d %H:%M:%S", localtime())
    #print("Start peak finding and LRT computing: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    for i in range(B):
        if (i+1) % 500 == 0:
            print("Iteration " + str(i+1) + ":  " + strftime("%Y-%m-%d %H:%M:%S", localtime()))

        Pstats2 = np.array((Pstatsall[:,i],Pstatsall[:,i+B])).T
        DMseg_out1 = Segment_satistics(Pstats1=Pstats2,clusterindex=clusterindex,pos=pos,chr=chr,diff_cutoff=diff_cutoff,zscore_cutoff=zscore_cutoff,zscore_cutoff1=zscore_cutoff1)
        DMseg_out1['simulationidx'] = i
        DMseg_out2 = DMseg_out1.loc[:,["cluster","cluster_L","LRT","simulationidx"]]
        allsimulation= pd.concat([allsimulation, DMseg_out2])
    
    print("Permutation ends: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return allsimulation

#permuation, parallel version
def fit_model_probes_sim0(i,beta1,S,V,SV,resids0,term,seed,B,ncore):
    n=int(B/ncore)+1
    seq = list(range(n*i,(i+1)*n))
    if i == ncore-1:
        seq =list(range(n*i,B))
    allresult = np.empty((len(seq)*2,beta1.shape[0]))
    k=0
    for j in seq:
        np.random.seed(seed+j)
        idx = np.random.permutation(range(beta1.shape[1]))
        mybeta = beta1[:,idx]
        coef = np.matmul(mybeta, np.matmul(S.transpose(), V)) / np.matmul(V.transpose(), SV)
        resids = np.matmul(resids0, mybeta.transpose())
        SE = np.sqrt(np.multiply(resids, resids).sum(axis=0) * term)
        result = np.vstack([coef,SE])
        allresult[(k*2):(k*2+2)] = result #np.vstack([allresult,result]) 
        k = k+1
    return(allresult)

def fit_model_probes_sim_all(beta,design,seed,B,ncore):
    
    beta1 = np.array(beta)
    design1 = np.array(design)
    M = np.delete(design1,1,axis=1)
    M_QR_q, M_QR_r = np.linalg.qr(M)
    S = np.diag([1] * M.shape[0]) - np.matmul(M_QR_q, M_QR_q.transpose())
    V = design1[:, 1]
    SV = np.matmul(S, V)

    QR_X_q, QR_X_r = np.linalg.qr(design)
    resids0 = np.diag([1] * design.shape[0]) - np.matmul(QR_X_q, QR_X_q.transpose())
    term = np.linalg.inv(design1.T.dot(design1))[1, 1] / (beta.shape[1] - np.linalg.matrix_rank(M) - 1)

    allcoef = np.zeros((beta.shape[0],B))
    allSE = np.zeros((beta.shape[0],B))

    pool = mp.Pool(ncore)

    start = time.time()
    #results = [pool.apply(fit_model_probes_sim0, args=(ii, beta1,S,V,SV,resids0,term,seed)) for ii in range(B)]
    #results = [pool.starmap(fit_model_probes_sim0, [(ii, beta1,S,V,SV,resids0,term,seed) for ii in range(B)])]
    #result_objects = [pool.apply_async(fit_model_probes_sim0, args=(ii, beta1,S,V,SV,resids0,term,seed)) for ii in range(B)]
    #results = [r.get()[1] for r in result_objects]
    #results = pool.starmap_async(fit_model_probes_sim00, [(ii, beta1,S,V,SV,resids0,term,seed) for ii in range(B)]).get()

    results = pool.starmap_async(fit_model_probes_sim0, [(ii, beta1,S,V,SV,resids0,term,seed,B,ncore) for ii in range(ncore)]).get()
    end = time.time()
    #print(end-start)
    pool.close()
    pool.join()
    k=0
    for i in range(ncore):
        m = results[i].shape[0]
        allcoef[:,k:(k+int(m/2))] = results[i][range(0,m,2),:].T
        allSE[:,k:(k+int(m/2))] = results[i][range(1,m,2),:].T
        k=k+int(m/2)

    result = np.concatenate((allcoef,allSE),axis=1)
    return result

def Segment_satistics0(i,Pstatsall,clusterindex,pos,chr,diff_cutoff,zscore_cutoff,zscore_cutoff1,B,ncore):
    n=int(B/ncore)+1
    seq = list(range(n*i,(i+1)*n))
    if i == ncore-1:
        seq =list(range(n*i,B))
    allresult = np.empty((0,4))
    k = 0
    for j in seq:
        Pstats2 = np.array((Pstatsall[:,j],Pstatsall[:,j+B])).T
        DMseg_out1 = Segment_satistics(Pstats1=Pstats2,clusterindex=clusterindex,pos=pos,chr=chr,diff_cutoff=diff_cutoff,zscore_cutoff=zscore_cutoff,zscore_cutoff1=zscore_cutoff1)
        DMseg_out1['simulationidx'] = j
        DMseg_out2 = DMseg_out1.loc[:,["cluster","cluster_L","LRT","simulationidx"]]
        allresult = np.vstack([allresult, DMseg_out2])
    return allresult

def do_simulation1(beta,design,clusterindex,pos,chr,seed,diff_cutoff,zscore_cutoff,zscore_cutoff1,B,ncore):

    print("Start " + str(B) + " permutation: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))

    #print("Start linear model fitting: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    #Pstatsall =  fit_model_probes_sim(beta=beta1,design=design1,seed=seed,B=B)
    Pstatsall = fit_model_probes_sim_all(beta,design,seed,B,ncore)
    #strftime("%Y-%m-%d %H:%M:%S", localtime())
    #print("Start peak finding and LRT computing: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))

    pool = mp.Pool(ncore)
    start = time.time()
    #results = [pool.starmap(Segment_satistics0, [(ii, Pstatsall,clusterindex,pos,chr,diff_cutoff,zscore_cutoff,zscore_cutoff1,B) for ii in range(B)])]
    #results = pool.starmap_async(Segment_satistics00, [(ii, Pstatsall,clusterindex,pos,chr,diff_cutoff,zscore_cutoff,zscore_cutoff1,B) for ii in range(B)]).get()
    results = pool.starmap_async(Segment_satistics0, [(ii, Pstatsall,clusterindex,pos,chr,diff_cutoff,zscore_cutoff,zscore_cutoff1,B,ncore) for ii in range(ncore)]).get()
    end = time.time()
    #print(end-start)
    pool.close()
    pool.join()
    allsimulation=pd.DataFrame()
    for i in range(ncore):
        allsimulation = pd.concat([allsimulation, pd.DataFrame(results[i])])
    
    allsimulation.columns = ["cluster","cluster_L","LRT","simulationidx"]
    print("Permutation ends: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return allsimulation

#get FWER based on NULL without adding extra permutations
def compute_fwer1(DMseg_out,allsimulation,B,clusterindex,L1=0,L2=10):
    DMseg_out1 = DMseg_out[(DMseg_out['cluster_L']>L1) & (DMseg_out['cluster_L']<=L2)]
    LRT_alt = DMseg_out1['LRT']
    allsimulation1 = allsimulation[(allsimulation['cluster_L']>L1) & (allsimulation['cluster_L']<=L2)]
    allsimulation1_grp = allsimulation1.groupby(by="simulationidx")

    #p-value
    LRT_alt_rank1 = pd.DataFrame(-np.array(LRT_alt.to_list() + allsimulation1['LRT'].to_list()))
    LRT_alt_rank1 = LRT_alt_rank1.rank(axis=0, method='max')[
        0:len(LRT_alt)]
    LRT_alt_rank1 = LRT_alt_rank1 - LRT_alt_rank1.rank(axis=0, method='max')
    
    #multiple peaks can be found in a cluster, which increase the total number of test
    multiplepeaksinacluster = allsimulation1_grp['cluster'].value_counts().tolist()
    #if one peak found in a cluster, no need to add the number 
    multiplepeaksinacluster = [x-1 for x in multiplepeaksinacluster]
    clusterlen = clusterindex.groupby(clusterindex).count()
    #total number of clusters in stratified data
    numclusters = sum((clusterlen >L1) & (clusterlen <= L2))
    totaltests = numclusters*B + (sum(multiplepeaksinacluster))

    P = LRT_alt_rank1 / totaltests
    P.index=DMseg_out1.index.to_list()
    P.columns=["P"]
    #DMseg_out1 = DMseg_out1.assign(P = P)
    DMseg_out1 = pd.concat([DMseg_out1,P],axis=1)
    #DMseg_out1["P"] = round(DMseg_out1["P"],3)

    #FWER
    LRTmax = allsimulation1_grp['LRT'].max().tolist()
    if len(LRTmax) < B:
        LRTmax.extend([0]*(B - len(LRTmax)))

    LRT_alt_rank = pd.DataFrame(-np.array(LRT_alt.to_list() + LRTmax))
    LRT_alt_rank = LRT_alt_rank.rank(axis=0, method='max')[
        0:len(LRT_alt)]
    LRT_alt_rank = LRT_alt_rank - LRT_alt_rank.rank(axis=0, method='max')

    FWER = LRT_alt_rank / len(LRTmax)
    FWER.index = DMseg_out1.index.to_list()
    FWER.columns = ["FWER"]
    DMseg_out1 = pd.concat([DMseg_out1,FWER],axis=1)
    #DMseg_out1 = DMseg_out1.assign(FWER = FWER)#.iloc[:,0].to_list()
    #DMseg_out1["LRT"] = round(DMseg_out1["LRT"],3)

    DMseg_out1 = DMseg_out1.sort_values(by=['LRT'], ascending=False)
    #DMseg_out1 = DMseg_out1.reset_index(drop=True) 
    np.sum(DMseg_out1["FWER"]<0.05)
    idx = np.where(DMseg_out1["FWER"]<0.05)[0]
    print(len(idx))
    #print(DMseg_out1.iloc[idx,:])
    return DMseg_out1

#get all the FWER in all levels
def compute_fwerall(DMseg_out,allsimulation,B,clusterindex,Ls=[0,10,20,30]):
    DMseg_out2 = pd.DataFrame()
    for i in range(len(Ls)):
        if i < (len(Ls)-1):
            L1 = Ls[i]
            L2 =Ls[i+1]
        else:
            L1 = Ls[i]
            L2 = max(allsimulation['cluster_L'])
        DMseg_out1 = compute_fwer1(DMseg_out,allsimulation,B,clusterindex,L1,L2)
        DMseg_out2 = pd.concat([DMseg_out2,DMseg_out1])
    DMseg_out2 = DMseg_out2.sort_values(by=['FWER',"n_cpgs"], ascending=[True,False])
    return(DMseg_out2)


#get p-values based on a single stratified set of NULL, for short clusters
def compute_pvalue_short(DMseg_out,allsimulation,B,clusterwidth,L1=0,L2=10):
    shortclusters1 = clusterwidth.index[(clusterwidth>L1) & (clusterwidth<=L2)]
    shortclusters1 = shortclusters1.to_list()
    DMseg_out_short = DMseg_out[(DMseg_out['cluster_L']>L1) & (DMseg_out['cluster_L']<=L2) ]
    LRT_alt_short = DMseg_out_short['LRT']
    allsimulation_short = allsimulation[(allsimulation['cluster_L']>L1) &(allsimulation['cluster_L']<=L2)]
    allsimulation_short['LRT'].describe()
    allsimulation_short_grp = allsimulation_short.groupby(by="simulationidx")
    # tmp=plt.hist(allsimulation_short['LRT'])
    # plt.xlabel("LRT")
    # plt.show()
    
    #p-value of observed,number of Null have larger LRT than observed LRT
    LRT_alt_rank_short = pd.DataFrame(-np.array(LRT_alt_short.to_list() + allsimulation_short['LRT'].to_list()))
    LRT_alt_rank_short = LRT_alt_rank_short.rank(axis=0, method='max')[0:len(LRT_alt_short)]
    LRT_alt_rank_short = LRT_alt_rank_short - LRT_alt_rank_short.rank(axis=0, method='max')
    
    #multiple peaks can be found in a cluster, which increase the total number of test
    multiplepeaksinacluster_short = allsimulation_short_grp['cluster'].value_counts().tolist()
    #if one peak found in a cluster, no need to add the number in addition to total cluster numbers 
    multiplepeaksinacluster_short = [x-1 for x in multiplepeaksinacluster_short]
    #total number of clusters in stratified data
    numclusters_short = len(shortclusters1)
    #the second part is for extra peaks found in a clusters
    totaltests_short = numclusters_short*B + sum(multiplepeaksinacluster_short)
    #print(totaltests_short)
    
    P_short = LRT_alt_rank_short / totaltests_short
    P_short.index=DMseg_out_short.index.to_list()
    P_short.columns=["P"]
    # tmp=plt.hist(P_short)
    # plt.xlabel("p-value")
    # tmp=plt.hist(-np.log10(P_short['P']))
    # plt.xlabel("-log10(p-value)")
    # plt.show()
    
    #p-value of null DMRs
    NULL_rank_short = pd.DataFrame(-np.array(allsimulation_short['LRT'].to_list()))
    NULL_rank_short = NULL_rank_short.rank(axis=0, method='max')
    NULL_P_short = NULL_rank_short/totaltests_short
    NULL_P_short.index = allsimulation_short.index
    # NULL_P_short.describe()
    # tmp = [1]*(totaltests_short-len(NULL_P_short))
    # tmp1 = NULL_P_short.iloc[:,0].to_list()
    # tmp2 = tmp +tmp1
    # np.quantile(tmp2,[0,0.005,0.01,0.015,0.1,1])
    # tmp=plt.hist(tmp2)
    # plt.xlabel("p-value")
    # tmp=plt.hist(-np.log10(NULL_P_short))
    # plt.xlabel("-log10(p-value)")
    # plt.show()
    allsimulation_short = allsimulation_short.assign(null_p=NULL_P_short.iloc[:,0].to_list())
    result = dict(P_short=P_short,allsimulation_short=allsimulation_short)
    return result

#get p values for long clusters (adding more permutations)
def compute_pvalue_long(DMseg_out,allsimulation,allsimulation2,B,B1,longclusters):
    DMseg_out_long = DMseg_out[DMseg_out['cluster'].isin(longclusters)]
    LRT_alt_long = DMseg_out_long['LRT']
    
    #simulation in first 500
    allsimulation_long500 = allsimulation[(allsimulation['cluster'].isin(longclusters))]
    #extra simulation for long clusters
    if allsimulation2['simulationidx'].min() < B:
        allsimulation2['simulationidx'] = allsimulation2['simulationidx'] + B
    #simulation in all 5000
    allsimulation_long = pd.concat([allsimulation_long500,allsimulation2])
    allsimulation_long_grp = allsimulation_long.groupby(by="simulationidx")
    # tmp=plt.hist(allsimulation_long['LRT'])
    # plt.xlabel("LRT")
    # plt.show()
    # tmp=plt.hist(allsimulation_long500['LRT'])
    # plt.xlabel("LRT")
    # plt.show()
    
    #p-value of observed
    LRT_alt_rank_long = pd.DataFrame(-np.array(LRT_alt_long.to_list() + allsimulation_long['LRT'].to_list()))
    LRT_alt_rank_long = LRT_alt_rank_long.rank(axis=0, method='max')[0:len(LRT_alt_long)]
    LRT_alt_rank_long = LRT_alt_rank_long - LRT_alt_rank_long.rank(axis=0, method='max')
    
    #multiple peaks can be found in a cluster, which increase the total number of test
    multiplepeaksinacluster_long = allsimulation_long_grp['cluster'].value_counts().tolist()
    #if one peak found in a cluster, no need to add the number 
    multiplepeaksinacluster_long = [x-1 for x in multiplepeaksinacluster_long]
    #total number of clusters in stratified data
    numclusters_long = len(longclusters)
    totaltests_long = numclusters_long*(B+B1) + sum(multiplepeaksinacluster_long)
    #print(totaltests_long)
    
    P_long = LRT_alt_rank_long / totaltests_long
    P_long.index=DMseg_out_long.index.to_list()
    P_long.columns=["P"]
    
    # tmp=plt.hist(P_long)
    # plt.xlabel("p-value")
    # tmp=plt.hist(-np.log10(P_long['P']))
    # plt.xlabel("-log10(p-value)")
    # plt.show()
    
    #p-value of NULL DMRs in first 500 simulation
    allsimulation_long['LRT'].describe()
    NULL_rank_long500 = pd.DataFrame(-np.array(allsimulation_long['LRT'].to_list()))
    #NULL_rank_long500 = pd.DataFrame(-np.array(allsimulation_long500['LRT'].to_list() + allsimulation_long['LRT'].to_list()))
    NULL_rank_long500 = NULL_rank_long500.rank(axis=0, method='max')[0:len(allsimulation_long500)]
    #NULL_rank_long500 = NULL_rank_long500 - NULL_rank_long500.rank(axis=0)    
    NULL_P_long500 = NULL_rank_long500/totaltests_long
    NULL_P_long500.index = allsimulation_long500.index
    NULL_P_long500.describe()
    # tmp = [1]*(totaltests_long-len(NULL_P_long500))
    # tmp1 = NULL_P_long500.iloc[:,0].to_list()
    # tmp2 = tmp +tmp1
    # np.quantile(tmp2,[0,0.005,0.01,0.015,0.1,1])
    # tmp=plt.hist(NULL_P_long500)
    # plt.xlabel("p-value")
    # tmp=plt.hist(-np.log10(NULL_P_long500))
    # plt.xlabel("-log10(p-value)")
    # plt.show()
    allsimulation_long500 = allsimulation_long500.assign(null_p=NULL_P_long500.iloc[:,0].to_list())
    result = dict(P_long=P_long,allsimulation_long500=allsimulation_long500)
    return result

def compute_fwer_all (DMseg_out,allsimulation,allsimulation2,B,B1,longclusters,clusterwidth,Ls=[0,10,20]):
    P = pd.DataFrame()
    allsimulations = pd.DataFrame()
    for i in range(len(Ls)):
        if i < (len(Ls)-1):
            L1 = Ls[i]
            L2 =Ls[i+1]
            #print(L1,L2)
            result = compute_pvalue_short(DMseg_out,allsimulation,B,clusterwidth,L1=L1,L2=L2)
            P = pd.concat([P,result['P_short']])
            allsimulations = pd.concat([allsimulations,result["allsimulation_short"]])
    result = compute_pvalue_long(DMseg_out,allsimulation,allsimulation2,B,B1,longclusters)
    P = pd.concat([P,result['P_long']])
    allsimulations = pd.concat([allsimulations,result["allsimulation_long500"]])
    allsimulations_grp = allsimulations.groupby(by="simulationidx")
    NULL_Pmin = allsimulations_grp['null_p'].min().tolist()
    if len(NULL_Pmin) < B:
        NULL_Pmin.extend([1]*(B - len(NULL_Pmin)))
    #tmp = pd.DataFrame(NULL_Pmin)
   
    P_alt_rank = pd.DataFrame(np.array(P.iloc[:,0].to_list() + NULL_Pmin))
    P_alt_rank = P_alt_rank.rank(axis=0, method='max')[0:len(P)]
    P_alt_rank.index = P.index
    P_alt_rank = P_alt_rank - P_alt_rank.rank(axis=0, method='max')
    
    FWER = P_alt_rank / len(NULL_Pmin)
    FWER.columns = ["FWER"]
    
    DMseg_out1 = DMseg_out.assign(P=P,FWER=FWER)
    DMseg_out1 = DMseg_out1.sort_values(by=['FWER',"n_cpgs"], ascending=[True,False])
    #sum(DMseg_out1["FWER"]<=0.05)
    return DMseg_out1


def form_dvcdata (beta,design):
    groups = design.iloc[:,1].unique()
    idx1 = np.where(design.iloc[:,1]==groups[0])[0]
    idx2 = np.where(design.iloc[:,1]==groups[1])[0]
    Mvalue = np.log2(beta/(1-beta))
    rmedian1 = np.median(Mvalue.iloc[:,idx1],axis=1)
    rmedian2 = np.median(Mvalue.iloc[:,idx2],axis=1)
    vdat = np.array(Mvalue)
    vdat[:,idx1] = np.abs(vdat[:,idx1].T - rmedian1.T).T
    vdat[:,idx2] = np.abs(vdat[:,idx2].T - rmedian2.T).T
    vdat = pd.DataFrame(vdat,columns=beta.columns,index=beta.index)
    return vdat

#merge nearby clusters if they have high correlation in boundary CpGs.
def merge_cluster (beta,chr,pos,cluster,maxmergegap=1000000,corr_cutoff=0.6):
    
    chrs = chr.unique()
    allclusters = pd.DataFrame()
    for mychr in chrs:
        #print(mychr)
        idx = np.where(chr==mychr)[0]
        mycluster = cluster[idx]
        mycluster = mycluster.to_frame()
        mycluster.columns = ["cluster"]
        mycluster['cpg'] = mycluster.index
        mypos = pos[mycluster.index]    
        myclustergrp = mycluster.groupby("cluster")
        startcpgs=myclustergrp['cpg'].first().to_list()
        endcpgs=myclustergrp['cpg'].last().to_list()
        allmycluster = myclustergrp['cluster'].first().to_list()
        ncpg=myclustergrp['cpg'].count()
        ncpg.reset_index(drop=True,inplace=True)
        endcpgs=endcpgs[:-1]
        startcpgs=startcpgs[1:]
        corref = pd.DataFrame({"endcpg":endcpgs,"startcpg":startcpgs,"endcluster":allmycluster[:-1],"startcluster":allmycluster[1:]})
        corref.index = endcpgs
        df1 = beta.loc[endcpgs].T
        df2 = beta.loc[startcpgs].T
        df2.columns = df1.columns
        corref['corr'] = df1.corrwith(df2)
        corref['distance'] = np.array(mypos[startcpgs])-np.array(mypos[endcpgs])
        clustermerge = np.zeros(corref.shape[0])
        clustermerge = np.where((corref['corr']>corr_cutoff) & (corref['distance']<maxmergegap), 1, clustermerge)
        corref['clustermerge'] = clustermerge
        if sum(clustermerge)>0:
            tmp0 = 1*(clustermerge == 0)
            corref["merge"]  = np.cumsum(tmp0)
            corref1 = corref.iloc[np.where(clustermerge==1)[0]]
            #cluster in oldcluster change to newcluster
            corref1 = corref1.assign(oldcluster = corref1['startcluster'])
            corref1 = corref1.assign(newcluster = corref1['endcluster'])
            corref1_grp = corref1.groupby("merge")
            newcluster1 = corref1_grp['endcluster'].first()
            newcluster1 = np.repeat(newcluster1,corref1_grp["merge"].count())
            corref1['newcluster'] = newcluster1.tolist()
            #if mychr == 'chr22':
            #    print(corref1)

            allchangecluster = corref1['oldcluster'].to_list()
            #need to change clusterid
            mycluster1 = mycluster[mycluster['cluster'].isin(allchangecluster)]
            a = mycluster1["cluster"].to_list()
            b = corref1["oldcluster"].to_list()
            idx1 = [ b.index(x) if x in b else None for x in a ]
            mycluster1 = mycluster1.assign(cluster = corref1['newcluster'].iloc[idx1].to_list())
            idx = np.where(mycluster.index.isin(mycluster1.index))[0]
            mycluster.iloc[idx,0] = mycluster1['cluster']

        allclusters = pd.concat([allclusters,mycluster])
        
    np.all(allclusters.index == cluster.index)    
    allcluster = pd.Series(allclusters['cluster'],dtype='object',index=cluster.index)
    return(allcluster)