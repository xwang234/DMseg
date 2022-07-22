#!/usr/bin/env python
# include VMR
# add p-value
# add more permutations to long clusters
# count a CpG into the DMR where it is a weak signal and is in the middle of strong signals.
# add multipleprocessing
"""
   Functions used to call DMseg.
"""
from __future__ import print_function
import numpy as np
from time import localtime, strftime
import pandas as pd
import re
import multiprocessing as mp
# import time


# make clusters
def clustermaker(chr, pos, assumesorted=False, maxgap=500):
    """A cluster of CpGs are contiguous CpGs within
    the distance of maxgap BP.
    Parameters
    ----------
    chr : series
        Chromosome of CpGs
    pos : series
        Poistion of CpGs
    assumesorted : bool
        A flag used to indicate if the CpGs are sorted by position (default is
        False)
    maxgap : int
        The distance cutoff to define a cluster of contiguous CpGs
    Returns
    -------
    series
        CpGs with their cluster ID
    """
    # number of CpGs in each chromosome
    # not to use sort, CpGs in chr and pos should be consistent
    tmp = chr.groupby(by=chr, sort=False).count()
    # CpG ID for CpGs
    Indexes = tmp.cumsum().to_list()
    Indexes.insert(0, 0)
    clusterIDs = pd.Series(data=[None]*pos.shape[0], index=chr.index)
    Last = 0
    # generate clusters in each chromosome
    for i in range(len(Indexes)-1):
        i1 = Indexes[i]
        i2 = Indexes[i+1]
        Index = range(i1, i2)
        x = pos.iloc[Index]
        if not assumesorted:
            x = x.sort_values()
        # if distance between 2 nearby CpGs are greater than cutoff
        y = np.diff(x) > maxgap
        # make sure clusterid starts from 1
        y = np.insert(y, 0, 1)
        # by using cumulative sum, get the clusterid
        z = np.cumsum(y)
        clusterIDs.iloc[i1:i2] = z + Last
        Last = max(z) + Last
    return clusterIDs


# fit model for data
def fit_model_probes(beta, design):
    """get estimates of linear models.
    Parameters
    ----------
    beta : dataframe
        Beta values of CpGs, rows for CpGs, columns for samples
    design : dataframe
        Design matrix of the model, second column should be the group indicater of interest to compare
    Returns
    -------
    numpy array
        Estimatied coeficients and standard errors, rows fro CpGs
    """
    # use np array to save time
    beta1 = np.array(beta)
    design1 = np.array(design)
    M = np.delete(design1, 1, axis=1)
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
    result = np.array([coef, SE]).T
    return result


# Search peak segments
def search_segments(DMseg_stats, zscore_cutoff, zscore_cutoff1):
    """find peaks in ROI clusters based on Z-scores. A peak or segment is contiguous CpGs which meet
    the zscore cutoffs within a cluster.
    Parameters
    ----------
    DMseg_stats : dataframe
        Estimated statistics and information for ROI clusters, rows for CpGs, columns for samples
    zscore_cutoff : float
        A cutoff to check if a CpG has a greater zscore
    zscore_cutoff1 : float
        A weaker cutoff to check if a CpG in the middle of cluster has a greater zscore
    Returns
    -------
    dataframe
        adding segment (segment id considering direction of z-score), segment1 (segment id based on absolute
        value, not considering direction), and direction to the dataframe
    """
    zscore = DMseg_stats['Coef']/DMseg_stats['SE']
    zscore_cutoff1 = abs(zscore_cutoff1)
    zscore_cutoff = abs(zscore_cutoff)
    # indicater if the CpG has zscore > zscore_cutoff
    myzscore = 1*(zscore.abs() > zscore_cutoff)
    # indicater if the CpG has zscore > zscore_cutoff1
    myzscore1 = 1*(zscore.abs() > zscore_cutoff1)
    # indicater combining the above two. 3: zscore>cutoff,1:zscore>cutoff1,0:zscore<cutoff1
    myzscore2 = myzscore*2 + myzscore1
    tmp = myzscore2.to_string(index=False)
    tmp = re.sub("\n", '', tmp)
    if myzscore.index.name is not None:
        tmp = re.sub(myzscore.index.name, '', tmp)
    tmp = re.sub(" ", '', tmp)
    # the index where a cpg in the middle which has a weak signal (Z-score>zscore_zscore_cutoff1), need to include it
    tmp1 = [_.start() for _ in re.finditer("313", tmp)]
    if len(tmp1) > 0:
        myzscore.iloc[np.array(tmp1)+1] = 1
    # direction: 1 if cpg has zscore > zscore_cutoff, 0 abs(zscore) < zscore_cutoff, -1 if zscore < -zscore_cutoff
    direction = np.zeros(DMseg_stats.shape[0])
    direction = np.where((zscore > 0) & (myzscore == 1), 1, direction)
    direction = np.where((zscore < 0) & (myzscore == 1), -1, direction)
    # direction1 is based on the absolute zscores.
    direction1 = myzscore
    
    # segments are based on direction1 (a segment has contiguous CpGs with the same direction1);
    # a segment can cross the border of a cluster here
    tmp0 = 1*(np.diff(direction1) != 0)
    tmp0 = np.insert(tmp0, 0, 1)
    segments1 = np.cumsum(tmp0)

    # split a segment if it covers multiple clusters; a segment should be within a cluster
    allsegments1 = segments1 + DMseg_stats['cluster']
    tmp0 = 1*(np.diff(allsegments1) != 0)
    tmp0 = np.insert(tmp0, 0, 1)
    allsegments1 = np.cumsum(tmp0)

    # allsegments1 are the final segments
    DMseg_stats['segment1'] = allsegments1
    DMseg_stats['direction'] = direction
    # segment are based on direction (consider signs)
    tmp0 = 1*(np.diff(direction) != 0)
    tmp0 = np.insert(tmp0, 0, 1)
    segments = np.cumsum(tmp0)
    allsegments = segments + DMseg_stats['cluster']
    tmp0 = 1*(np.diff(allsegments) != 0)
    tmp0 = np.insert(tmp0, 0, 1)
    allsegments = np.cumsum(tmp0)
    DMseg_stats['segment'] = allsegments

    return DMseg_stats
    

# LRT function considering switches in a peak
def LRT_segment_nocorr(DMseg_stats):
    """Compute likelihood ratio test (lrt) statistics and mean (seg_mean) for each segment/peak.
    Parameters
    ----------
    DMseg_stats : dataframe
        Estimated statistics and information for ROI clusters, rows for CpGs, columns for samples.
        peak information (segment,direction etc were added in search_segments function)
    Returns
    -------
    dictionary
        'lrt': LRT score
        'seg_mean: mean of segment
    """
    csegments = DMseg_stats.segment1
    bdiff = np.abs(DMseg_stats.Coef)
    bvar = 1/pow(DMseg_stats.SE, 2)
    DMseg_stats["bvar"] = bvar
    DMseg_stats['bdiff_bvar'] = bvar * bdiff
    # first compute b0,b1 based on segments using zscore (not using absolute values)
    tmp = DMseg_stats.groupby(by="segment")
    b0 = tmp['bvar'].sum()
    b1 = tmp["bdiff_bvar"].sum()
    sign = tmp['direction'].first()
    # the segment id using absolute zscore
    mysegment = tmp['segment1'].first()
    # include the sign here
    tmp1 = pd.DataFrame({"b0": b0, "b1_sign": b1*sign, "bb1_divid_bb0_sign": b1/b0*sign, "segment1": mysegment}, index=b0.index)
    tmp1_groupby = tmp1.groupby(by="segment1")
    bb0 = tmp1_groupby['b0'].sum()
    bb1 = tmp1_groupby['b1_sign'].sum()
    seg_mean = bb1/bb0
    tmp1["bb1_divid_bb0_sign"] = round(tmp1["bb1_divid_bb0_sign"], 3)
    tmp1["bb1_divid_bb0_sign"] = tmp1["bb1_divid_bb0_sign"].astype(str)
    tmp2_groupby = tmp1.groupby(by="segment1")
    # for segmean if there are swiches, show all
    seg_mean_all = tmp2_groupby.agg({"bb1_divid_bb0_sign": ";".join})
 
    groupcounts = csegments.groupby(csegments).count()
    seg_mean_vec = np.repeat(seg_mean, groupcounts)

    lrt1 = pow(DMseg_stats.Coef.to_numpy() - seg_mean_vec.to_numpy(), 2) * bvar
    lrt0 = pow(bdiff, 2) * bvar
    lrt = lrt0.groupby(csegments).sum() - lrt1.groupby(csegments).sum()
    result = dict(lrt=lrt, seg_mean=seg_mean_all)
    return result


# wrap search_segments + LRT
def Segment_satistics(Pstats, clusterindex, chr, pos, diff_cutoff=0.05, zscore_cutoff=1.96, zscore_cutoff1=1.78):
    """Function to detect peaks/segments and compute LRT. It calls functions search_segments and LRT_segment_nocorr.
    Parameters
    ----------
    Pstats : numpy array
        The output of function fit_model_probes. It includes estimated coefficients and standard errors
    clusterindex : series
        The cluster id for CpGs
    chr : series
        choromoe of CpGs
    pos : series
        position of CpGs
    diff_cutoff : float
        cutoff to select CpGs with mean diff greater than it
    zscore_cutoff : float
        cutoff based on zscore, to select strong CpGs with zscore greater than it
    zscore_cutoff1 : float
        second cutoff based on zscore, to select a CpG if the CpG is in the middle of two strong CpGs
    Returns
    -------
    dataframe
        Each row respresents a detected peak/segment, with position info, seg_mean and LRT
    """
    # Check ROI, region of interest use both: diff_cutoff and zsocre cutoffs
    # clusters meet diff_cutoff and zscore_cutoffs

    Pstats1 = pd.DataFrame(Pstats, index=clusterindex.index, columns=["Coef", "SE"])
    # compute Z-score
    Pstats1["Zscore"] = Pstats1["Coef"]/Pstats1["SE"]
    # for each CpG, if coefficient greater than cutoff (diff mean)
    mycoef = 1*(Pstats1['Coef'].abs() > diff_cutoff)
    # for each CpG, if zscore greater than cutoff
    myzscore = 1*(Pstats1['Zscore'].abs() > zscore_cutoff)
    # indicater, keep a cluster if at least two CpG has coef greater than cutoff
    coef_select = mycoef.groupby(clusterindex).sum() > 1
    # indicater,keep a cluster if at least two CpG has zscore greater than cutoff
    zscore_select = myzscore.groupby(clusterindex).sum() > 1

    groupcounts = clusterindex.groupby(clusterindex).count()
    coef_select_vec = np.repeat(coef_select, groupcounts)
    zscore_select_vec = np.repeat(zscore_select, groupcounts)
    # region of interest clusters are clusters meet the above two criteria
    # these clusters are to be investigated for differential analysis
    ROIcluster = clusterindex[clusterindex.index[np.logical_and(coef_select_vec, zscore_select_vec)]]
    if len(ROIcluster) == 0:
        # print("No ROI been found, please consider to reduce the value of diff_cutoff")
        DMseg_out = pd.DataFrame(columns=["cluster", "cluster_L", "chr", "start_cpg", "start_pos", "end_cpg", "end_pos",
                                          "n_cpgs", "seg_mean", "LRT"])
    else:
        DMseg_sd = Pstats1['SE'].loc[ROIcluster.index]
        DMseg_coef = Pstats1['Coef'].loc[ROIcluster.index]
   
        DMseg_pos = pos[ROIcluster.index]
        DMseg_chr = chr[ROIcluster.index]
    
        DMseg_stats = pd.concat([DMseg_coef, DMseg_sd, DMseg_chr, DMseg_pos, ROIcluster], axis=1)
        DMseg_stats.columns = ["Coef", "SE", "chr", "pos", "cluster"]

        DMseg_stats = search_segments(DMseg_stats, zscore_cutoff, zscore_cutoff1)
        DMseg_clusterlen = DMseg_stats.groupby(by="cluster")['Coef'].count()
        # peaks have direction !=0
        DMseg_stats1 = DMseg_stats.loc[DMseg_stats.direction != 0]
        DMseg_stats1_groupby = DMseg_stats1.groupby(by="segment1")
        # #peak should have #cpg>1
        tmp = DMseg_stats1_groupby['cluster'].transform('count').gt(1)
        DMseg_stats1 = DMseg_stats1.loc[tmp.index[tmp == True]]

        # compute the LRT statistics
        tmp = LRT_segment_nocorr(DMseg_stats=DMseg_stats1)
        DMseg_out = pd.DataFrame(columns=["cluster", "cluster_L", "chr", "start_cpg", "start_pos", "end_cpg", "end_pos",
                                          "n_cpgs", "seg_mean", "LRT"], index=tmp['seg_mean'].index)
        DMseg_out["seg_mean"] = tmp['seg_mean']
        DMseg_out["LRT"] = tmp['lrt']

        DMseg_stats1["cpgname"] = DMseg_stats1.index
        DMgroup = DMseg_stats1.groupby("segment1")
        DMseg_out["cluster"] = DMgroup["cluster"].first()
        DMseg_out["cluster_L"] = DMseg_clusterlen.loc[DMseg_out["cluster"]].to_list()
        DMseg_out["n_cpgs"] = DMgroup["segment1"].count()
        DMseg_out["chr"] = DMgroup["chr"].first()
        DMseg_out["start_cpg"] = DMgroup["cpgname"].first()
        DMseg_out["end_cpg"] = DMgroup["cpgname"].last()
        DMseg_out["start_pos"] = DMgroup["pos"].first()
        DMseg_out["start_pos"] = DMseg_out["start_pos"].astype(int)
        DMseg_out["end_pos"] = DMgroup["pos"].last()
        DMseg_out["end_pos"] = DMseg_out["end_pos"].astype(int)
        # DMseg_out["numswitches"] = DMgroup["segment1"].last() - DMgroup["segment1"].first()
        DMseg_out = DMseg_out.reset_index(drop=True)
    return DMseg_out


# fit model for permutation, sequential
def fit_model_probes_simsequential(beta, design, seed, B):
    """get estimates of linear models for permutation data.
    Parameters
    ----------
    beta : dataframe
        Beta values of CpGs, rows for CpGs, columns for samples
    design : dataframe
        Design matrix of the model, second column should be the group indicater of interest to compare
    seed : int 
        The random seed
    B : int
        The number of permutation
    Returns
    -------
    numpy array
        Estimatied coeficients and standard errors, rows fro CpGs
    """
    beta1 = np.array(beta)
    design1 = np.array(design)
    M = np.delete(design1, 1, axis=1)
    M_QR_q, M_QR_r = np.linalg.qr(M)
    S = np.diag([1] * M.shape[0]) - np.matmul(M_QR_q, M_QR_q.transpose())
    V = design1[:, 1]
    SV = np.matmul(S, V)

    QR_X_q, QR_X_r = np.linalg.qr(design)
    resids0 = np.diag([1] * design.shape[0]) - np.matmul(QR_X_q, QR_X_q.transpose())
    tmp1 = np.linalg.inv(design1.T.dot(design1))[1, 1] / (beta.shape[1] - np.linalg.matrix_rank(M) - 1)

    coef = np.zeros((beta.shape[0], B))
    allSE = np.zeros((beta.shape[0], B))

    for i in range(B):
        np.random.seed(seed+i)
        idx = np.random.permutation(range(beta.shape[1]))
        mybeta = beta1[:, idx]
        coef[:, i] = np.matmul(mybeta, np.matmul(S.transpose(), V)) / np.matmul(V.transpose(), SV)
        resids = np.matmul(resids0, mybeta.transpose())
        allSE[:, i] = np.sqrt(np.multiply(resids, resids).sum(axis=0) * tmp1)

    result = np.concatenate((coef, allSE), axis=1)
    return result


# permutation, sequential version
def do_simulationsequential(beta, design, clusterindex, chr, pos, diff_cutoff, zscore_cutoff, zscore_cutoff1, seed, B):
    """get LRT scores for permutation data, sequential version.
    Parameters
    ----------
    beta : dataframe
        Beta values of CpGs, rows for CpGs, columns for samples
    design : dataframe
        Design matrix of the model, second column should be the group indicater of interest to compare
    clusterindex : series
        The cluster id for CpGs
    chr : series
        choromoe of CpGs
    pos : series
        position of CpGs
    diff_cutoff : float
        default is 0.05. cutoff to select CpGs with mean diff greater than it
    zscore_cutoff : float
        default is 1.96. cutoff to select strong CpGs with zscore greater than it
    zscore_cutoff1 : float
        default is 1.78. cutoff to select a CpG if the CpG is in the middle of two strong CpGs
    seed : int 
        The random seed
    B : int
        The number of permutation
    Returns
    -------
    dataframe
        Peaks detected in permutation data (LRT: LRT score,simulationidx:index of permutation)
    """
    print("Start " + str(B) + " permutation: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    allsimulation = pd.DataFrame()
    beta1 = np.array(beta)
    design1 = np.array(design)
    # print("Start linear model fitting: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    Pstatsall = fit_model_probes_simsequential(beta=beta1, design=design1, seed=seed, B=B)
    # strftime("%Y-%m-%d %H:%M:%S", localtime())
    # print("Start peak finding and LRT computing: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    for i in range(B):
        if (i+1) % 500 == 0:
            print("Iteration " + str(i+1) + ":  " + strftime("%Y-%m-%d %H:%M:%S", localtime()))

        Pstats2 = np.array((Pstatsall[:, i], Pstatsall[:, i+B])).T
        DMseg_out1 = Segment_satistics(Pstats=Pstats2, clusterindex=clusterindex, chr=chr, pos=pos,
                                       diff_cutoff=diff_cutoff, zscore_cutoff=zscore_cutoff,
                                       zscore_cutoff1=zscore_cutoff1)
        DMseg_out1['simulationidx'] = i
        DMseg_out2 = DMseg_out1.loc[:, ["cluster", "cluster_L", "LRT", "simulationidx"]]
        allsimulation = pd.concat([allsimulation, DMseg_out2])
    
    print("Permutation ends: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return allsimulation


# permuation, parallel version
def fit_model_probes_simparallel_call(i, beta1, S, V, SV, resids0, term, seed, B, ncore):
    """get estimates of linear models for permutation data, parallel version.
    Parameters
    ----------
    i : int
    an int in 1 - ncore, the CPU ID to run
    beta1 : numpy array
        Beta values of CpGs, rows for CpGs, columns for samples
    S : dataframe
    V, SV, resids0, term : np arrays used in QR decomposition
    seed : int
        The random seed
    B : int
        The number of permutation
    Returns
    -------
    numpy array
        Estimatied coeficients and standard errors, rows fro CpGs
    """
    n = int(B/ncore) + 1
    seq = list(range(n*i, (i+1)*n))
    if i == ncore - 1:
        seq = list(range(n*i, B))
    allresult = np.empty((len(seq)*2, beta1.shape[0]))
    k = 0
    for j in seq:
        np.random.seed(seed + j)
        idx = np.random.permutation(range(beta1.shape[1]))
        mybeta = beta1[:, idx]
        coef = np.matmul(mybeta, np.matmul(S.transpose(), V)) / np.matmul(V.transpose(), SV)
        resids = np.matmul(resids0, mybeta.transpose())
        SE = np.sqrt(np.multiply(resids, resids).sum(axis=0) * term)
        result = np.vstack([coef, SE])
        allresult[(k*2):(k*2+2)] = result
        k = k+1
    return allresult


def fit_model_probes_simparallel(beta, design, seed, B, ncore):
    """get estimates of linear models for permutation data, parallel version. It calls it_model_probes_simparallel function.
    Parameters
    ----------
    beta : dataframe
        Beta values of CpGs, rows for CpGs, columns for samples
    design : dataframe
        Design matrix of the model, second column should be the group indicater of interest to compare
    seed : int 
        The random seed
    B : int
        The number of permutation
    ncore : int
        The number of CPU used for computing
    Returns
    -------
    numpy array
        Estimatied coeficients and standard errors, rows fro CpGs
    """
    beta1 = np.array(beta)
    design1 = np.array(design)
    M = np.delete(design1, 1, axis=1)
    M_QR_q, M_QR_r = np.linalg.qr(M)
    S = np.diag([1] * M.shape[0]) - np.matmul(M_QR_q, M_QR_q.transpose())
    V = design1[:, 1]
    SV = np.matmul(S, V)

    QR_X_q, QR_X_r = np.linalg.qr(design)
    resids0 = np.diag([1] * design.shape[0]) - np.matmul(QR_X_q, QR_X_q.transpose())
    term = np.linalg.inv(design1.T.dot(design1))[1, 1] / (beta.shape[1] - np.linalg.matrix_rank(M) - 1)

    allcoef = np.zeros((beta.shape[0], B))
    allSE = np.zeros((beta.shape[0], B))

    pool = mp.Pool(ncore)
    # start = time.time()
    results = pool.starmap_async(fit_model_probes_simparallel_call, [(ii, beta1, S, V, SV, resids0, term, seed, B, ncore)
                                                                     for ii in range(ncore)]).get()
    # end = time.time()
    # print(end-start)
    pool.close()
    pool.join()
    k = 0
    for i in range(ncore):
        m = results[i].shape[0]
        allcoef[:, k:(k+int(m/2))] = results[i][range(0, m, 2), :].T
        allSE[:, k:(k+int(m/2))] = results[i][range(1, m, 2), :].T
        k = k+int(m / 2)

    result = np.concatenate((allcoef, allSE), axis=1)
    return result


def Segment_satisticsparallel_core(i, Pstatsall, clusterindex, chr, pos, diff_cutoff, zscore_cutoff, zscore_cutoff1, B, ncore):
    """Function to detect peaks/segments and compute LRT for permutation data on a core.
    Parameters
    ----------
    i : int
        core id, 1 - ncore
    Pstatsall : numpy array
        estimates from all permutation (coefficients and standard errors, rows for CpGs)
    clusterindex : series
        The cluster id for CpGs
    chr : series
        choromoe of CpGs
    pos : series
        position of CpGs
    diff_cutoff : float
        default is 0.05. cutoff to select CpGs with mean diff greater than it
    zscore_cutoff : float
        default is 1.96. cutoff to select strong CpGs with zscore greater than it
    zscore_cutoff1 : float
        default is 1.78. cutoff to select a CpG if the CpG is in the middle of two strong CpGs
    B : int
        The number of permutation
    ncore : int
        The number of CPU used for computing
    Returns
    -------
    numpy array
        Peaks detected in permutation data (LRT: LRT score,simulationidx:index of permutation)
    """
    # split jobs on ncore cores
    n = int(B/ncore) + 1
    seq = list(range(n*i, (i+1)*n))
    if i == ncore-1:
        seq = list(range(n*i, B))
    allresult = np.empty((0, 4))
    for j in seq:
        Pstats2 = np.array((Pstatsall[:, j], Pstatsall[:, j+B])).T
        DMseg_out1 = Segment_satistics(Pstats=Pstats2, clusterindex=clusterindex, chr=chr, pos=pos, diff_cutoff=diff_cutoff,
                                       zscore_cutoff=zscore_cutoff, zscore_cutoff1=zscore_cutoff1)
        DMseg_out1['simulationidx'] = j
        DMseg_out2 = DMseg_out1.loc[:, ["cluster", "cluster_L", "LRT", "simulationidx"]]
        allresult = np.vstack([allresult, DMseg_out2])
    return allresult


def do_simulationparallel(beta, design, clusterindex, chr, pos, diff_cutoff, zscore_cutoff, zscore_cutoff1, seed, B, ncore):
    """get LRT scores for permutation data, parallel version.
    Parameters
    ----------
    beta : dataframe
        Beta values of CpGs, rows for CpGs, columns for samples
    design : dataframe
        Design matrix of the model, second column should be the group indicater of interest to compare
    clusterindex : series
        The cluster id for CpGs
    chr : series
        choromoe of CpGs
    pos : series
        position of CpGs
    diff_cutoff : float
        default is 0.05. cutoff to select CpGs with mean diff greater than it
    zscore_cutoff : float
        default is 1.96. cutoff to select strong CpGs with zscore greater than it
    zscore_cutoff1 : float
        default is 1.78. cutoff to select a CpG if the CpG is in the middle of two strong CpGs
    seed : int 
        The random seed
    B : int
        The number of permutation
    ncore : int
        The number of CPU used for computing
    Returns
    -------
    dataframe
        Peaks detected in permutation data (LRT: LRT score,simulationidx:index of permutation)
    """
    print("Start " + str(B) + " permutation: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))

    # print("Start linear model fitting: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    Pstatsall = fit_model_probes_simparallel(beta, design, seed, B, ncore)
    # strftime("%Y-%m-%d %H:%M:%S", localtime())
    # print("Start peak finding and LRT computing: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))

    pool = mp.Pool(ncore)
    # start = time.time()
    results = pool.starmap_async(Segment_satisticsparallel_core, [(ii, Pstatsall, clusterindex, chr, pos, diff_cutoff, zscore_cutoff,
                                                                   zscore_cutoff1, B, ncore) for ii in range(ncore)]).get()
    # end = time.time()
    # print(end-start)
    pool.close()
    pool.join()
    allsimulation = pd.DataFrame()
    for i in range(ncore):
        allsimulation = pd.concat([allsimulation, pd.DataFrame(results[i])])
    
    allsimulation.columns = ["cluster", "cluster_L", "LRT", "simulationidx"]
    print("Permutation ends: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return allsimulation


# get p-values based on a single stratified set of NULL, for short clusters
def compute_pvalue_short(DMseg_out, allsimulation, B, clusterwidth, L1=0, L2=10):
    """Compute p-values for observed ROI clusters and permulation data, for short clusters.
    Parameters
    ----------
    DMseg_out : dataframe
        output of Segment_satistics, LRT and seg mean for observed data
    allsimulation : dataframe
        peaks detected in permutation data (LRT)
    B : int
        The number of permutation
    clusterwidth : series
        The width (number of CpGs) in each cluster
    L1 : int
    L2 : int
        short CpGs with width >L1 and <=L2 will be processed
    Returns
    -------
    dictionary
    P_short : series, p-values for observed peaks in short clusters
    allsimulation_short : dataframe, null_p for p-values of peaks detected in permutation
    """
    shortclusters1 = clusterwidth.index[(clusterwidth > L1) & (clusterwidth <= L2)]
    shortclusters1 = shortclusters1.to_list()
    DMseg_out_short = DMseg_out[(DMseg_out['cluster_L'] > L1) & (DMseg_out['cluster_L'] <= L2)]
    LRT_alt_short = DMseg_out_short['LRT']
    allsimulation_short = allsimulation[(allsimulation['cluster_L'] > L1) & (allsimulation['cluster_L'] <= L2)]
    allsimulation_short['LRT'].describe()
    allsimulation_short_grp = allsimulation_short.groupby(by="simulationidx")
    # tmp=plt.hist(allsimulation_short['LRT'])
    # plt.xlabel("LRT")
    # plt.show()
    
    # p-value of observed,count number of Null have larger LRT than observed LRT
    LRT_alt_rank_short = pd.DataFrame(-np.array(LRT_alt_short.to_list() + allsimulation_short['LRT'].to_list()))
    LRT_alt_rank_short = LRT_alt_rank_short.rank(axis=0, method='max')[0:len(LRT_alt_short)]
    LRT_alt_rank_short = LRT_alt_rank_short - LRT_alt_rank_short.rank(axis=0, method='max')
    
    # multiple peaks can be found in a cluster, which increase the total number of test
    multiplepeaksinacluster_short = allsimulation_short_grp['cluster'].value_counts().tolist()
    # if one peak found in a cluster, no need to add the number in addition to total cluster numbers
    multiplepeaksinacluster_short = [x-1 for x in multiplepeaksinacluster_short]
    # total number of clusters in stratified data
    numclusters_short = len(shortclusters1)
    # the second part is for extra peaks found in clusters
    totaltests_short = numclusters_short*B + sum(multiplepeaksinacluster_short)
    # print(totaltests_short)
    
    # the denominator needs to consider cases where no peaks detected in such clusters
    P_short = LRT_alt_rank_short / totaltests_short
    P_short.index = DMseg_out_short.index.to_list()
    P_short.columns = ["P"]
    # tmp=plt.hist(P_short)
    # plt.xlabel("p-value")
    # tmp=plt.hist(-np.log10(P_short['P']))
    # plt.xlabel("-log10(p-value)")
    # plt.show()
    
    # p-value of null DMRs
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
    allsimulation_short = allsimulation_short.assign(null_p=NULL_P_short.iloc[:, 0].to_list())
    result = dict(P_short=P_short, allsimulation_short=allsimulation_short)
    return result


# get p values for long clusters (adding more permutations)
def compute_pvalue_long(DMseg_out, allsimulation, allsimulation2, B, B1, longclusters):
    """Compute p-values for observed ROI clusters and permulation data, for long clusters.
    Parameters
    ----------
    DMseg_out : dataframe
        output of Segment_satistics, LRT and seg mean for observed data
    allsimulation : dataframe
        peaks detected in permutation data (LRT)
    allsimulation2 : dataframe
        peaks detected in added permutation data (LRT)
    B : int
        The number of permutation
    B1 : int
        The number of added permutation for long clusters
    longclusters : list of index
        Id for long clusters
    Returns
    -------
    dictionary
    P_long : series, p-values for observed peaks in long clusters based on all permutation
    allsimulation_long500 : dataframe, null_p for p-values of peaks detected in first B permutation
    """
    result = dict(P_long=None, allsimulation_long500=None)
    if longclusters is not None:
        # observed long peaks
        DMseg_out_long = DMseg_out[DMseg_out['cluster'].isin(longclusters)]
        LRT_alt_long = DMseg_out_long['LRT']
        # simulation in first 500 (B)
        allsimulation_long500 = allsimulation[(allsimulation['cluster'].isin(longclusters))]
        # extra simulation for long clusters
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
        # p-value of observed
        LRT_alt_rank_long = pd.DataFrame(-np.array(LRT_alt_long.to_list() + allsimulation_long['LRT'].to_list()))
        LRT_alt_rank_long = LRT_alt_rank_long.rank(axis=0, method='max')[0:len(LRT_alt_long)]
        LRT_alt_rank_long = LRT_alt_rank_long - LRT_alt_rank_long.rank(axis=0, method='max')
        # multiple peaks can be found in a cluster, which increase the total number of test
        multiplepeaksinacluster_long = allsimulation_long_grp['cluster'].value_counts().tolist()
        # if one peak found in a cluster, no need to add the number 
        multiplepeaksinacluster_long = [x-1 for x in multiplepeaksinacluster_long]
        # total number of clusters in stratified data
        numclusters_long = len(longclusters)
        totaltests_long = numclusters_long*(B+B1) + sum(multiplepeaksinacluster_long)
        # print(totaltests_long)
        P_long = LRT_alt_rank_long / totaltests_long
        P_long.index=DMseg_out_long.index
        P_long.columns=["P"]
        # tmp=plt.hist(P_long)
        # plt.xlabel("p-value")
        # tmp=plt.hist(-np.log10(P_long['P']))
        # plt.xlabel("-log10(p-value)")
        # plt.show()
        # p-value of NULL DMRs in first 500 (B) simulation
        NULL_rank_long500 = pd.DataFrame(-np.array(allsimulation_long['LRT']))
        # allsimulation_long500 is the first part of allsimulation_long
        NULL_rank_long500 = NULL_rank_long500.rank(axis=0, method='max')[0:len(allsimulation_long500)]
        NULL_P_long500 = NULL_rank_long500/totaltests_long
        NULL_P_long500.index = allsimulation_long500.index
        # NULL_P_long500.describe()
        # tmp = [1]*(totaltests_long-len(NULL_P_long500))
        # tmp1 = NULL_P_long500.iloc[:,0].to_list()
        # tmp2 = tmp +tmp1
        # np.quantile(tmp2,[0,0.005,0.01,0.015,0.1,1])
        # tmp=plt.hist(NULL_P_long500)
        # plt.xlabel("p-value")
        # tmp=plt.hist(-np.log10(NULL_P_long500))
        # plt.xlabel("-log10(p-value)")
        # plt.show()
        allsimulation_long500 = allsimulation_long500.assign(null_p=NULL_P_long500)
        result = dict(P_long=P_long,allsimulation_long500=allsimulation_long500)
    return result


def compute_fwer_all(DMseg_out, allsimulation, B, clusterwidth, allsimulation2=None, B1=4500, longclusters=None, Ls=(0, 10, 20)):
    """Compute FWER for observed ROI clusters.
    Parameters
    ----------
    DMseg_out : dataframe
        output of Segment_satistics, LRT and seg mean for observed data
    allsimulation : dataframe
        peaks detected in permutation data (LRT)
    B : int
        The number of permutation for all CpGs
    clusterwidth : series
        The width (number of CpGs) in each cluster
    allsimulation2 : dataframe
        peaks detected in added permutation data (LRT)
    B1 : int
        The number of added permutation for long clusters
    longclusters : list of index
        Id for long clusters
    Ls : tuple
        list use to stratify clusters based on their widths. 
        in default Ls=(0,10,20) which means we will compute p-values for 1) short clusters with width <=10,
        and 2) short clusters with width in (10,20], and 3) long clusters with width>20 separately.
    Returns
    -------
    dataframe
        P : p-values for observed peaks
        FWER : FWER values
    """
    P = pd.DataFrame()
    # allsimulations keep p-values for peaks detected in permutation
    allsimulations = pd.DataFrame()
    # if no stratification (stratify=0), would set Ls=0 in pipeline
    if Ls == 0:
        allsimulations = compute_pvalue_short(DMseg_out, allsimulation, B, clusterwidth, L1=0, L2=clusterwidth.max())
        P = allsimulations['P_short']
        allsimulations = allsimulations["allsimulation_short"]       
    else:
        for i in range(len(Ls)):
            if i < (len(Ls)-1):
                L1 = Ls[i]
                L2 = Ls[i+1]
                result = compute_pvalue_short(DMseg_out, allsimulation, B, clusterwidth, L1=L1, L2=L2)
                P = pd.concat([P, result['P_short']])
                allsimulations = pd.concat([allsimulations, result["allsimulation_short"]])
        # stratify but don't add extra long clusters
        if allsimulation2 is None:
            result = compute_pvalue_short(DMseg_out, allsimulation, B, clusterwidth, L1=L2, L2=DMseg_out["cluster_L"].max())
            P = pd.concat([P, result['P_short']])
            allsimulations = pd.concat([allsimulations, result["allsimulation_short"]])
    result = compute_pvalue_long(DMseg_out, allsimulation, allsimulation2, B, B1, longclusters)
    P = pd.concat([P, result['P_long']])
    allsimulations = pd.concat([allsimulations, result["allsimulation_long500"]])
    allsimulations_grp = allsimulations.groupby(by="simulationidx")
    NULL_Pmin = allsimulations_grp['null_p'].min().tolist()

    if len(NULL_Pmin) < B:
        NULL_Pmin.extend([1]*(B - len(NULL_Pmin)))
    # tmp = pd.DataFrame(NULL_Pmin)
   
    P_alt_rank = pd.DataFrame(np.array(P.iloc[:, 0].to_list() + NULL_Pmin))
    P_alt_rank = P_alt_rank.rank(axis=0, method='max')[0:len(P)]
    P_alt_rank.index = P.index
    P_alt_rank = P_alt_rank - P_alt_rank.rank(axis=0, method='max')
    
    FWER = P_alt_rank / len(NULL_Pmin)
    FWER.columns = ["FWER"]
    
    DMseg_out1 = DMseg_out.assign(P=P, FWER=FWER)
    DMseg_out1 = DMseg_out1.sort_values(by=['FWER', "n_cpgs"], ascending=[True, False])
    DMseg_out1.reset_index(drop=True)
    # sum(DMseg_out1["FWER"]<=0.05)
    return DMseg_out1


def form_vmrdata(beta, design):
    """generate VMR data using m values.
    Parameters
    ----------
    beta : dataframe
        Beta values of CpGs, rows for CpGs, columns for samples
    design : dataframe
        Design matrix of the model, second column should be the group indicater of interest to compare
    Returns
    -------
    dataframe
        data used to detect VMR, rows for CpGs, columns for samples
    """
    groups = design.iloc[:, 1].unique()
    idx1 = np.where(design.iloc[:, 1]==groups[0])[0]
    idx2 = np.where(design.iloc[:, 1]==groups[1])[0]
    Mvalue = np.log2(beta/(1-beta))
    rmedian1 = np.median(Mvalue.iloc[:, idx1], axis=1)
    rmedian2 = np.median(Mvalue.iloc[:, idx2], axis=1)
    vdat = np.array(Mvalue)
    vdat[:, idx1] = np.abs(vdat[:, idx1].T - rmedian1.T).T
    vdat[:, idx2] = np.abs(vdat[:, idx2].T - rmedian2.T).T
    vdat = pd.DataFrame(vdat, columns=beta.columns, index=beta.index)
    return vdat

#merge nearby clusters if they have high correlation in boundary CpGs.
def merge_cluster (beta, chr, pos, cluster, maxmergegap=1000000, merge_cutoff=0.6):
    """merge two nearby clusters into one if the edge CpGs have high correlation.
    Parameters
    ----------
    beta : dataframe
        Beta values of CpGs, rows for CpGs, columns for samples
    design : dataframe
        Design matrix of the model, second column should be the group indicater of interest to compare
    chr : series
        choromoe of CpGs
    pos : series
        position of CpGs
    cluster : series
        output of clustermaker, cluster ID for CpGs
    maxmergegap : int
        don't merge two nearby clusters if the gap between them is greater
    merge_cutoff : float
        cutoff of correlation between edge CpGs of two nearby clusters, used to merge clusters
    Returns
    -------
    dataframe
        data used to detect VMR, rows for CpGs, columns for samples
    """
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
        clustermerge = np.where((corref['corr']>merge_cutoff) & (corref['distance']<maxmergegap), 1, clustermerge)
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