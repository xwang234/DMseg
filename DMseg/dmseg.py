#!/usr/bin/env python
"""
   Call DMseg.
"""
# from __future__ import print_function
# import numpy as np
# from time import localtime, strftime
# import pandas as pd
# import multiprocessing as mp
from DMseg.functions import *


def pipeline(betafile, colDatafile, positionfile, maxgap=500, sd_cutoff=0.025,
             diff_cutoff=0.05, zscore_cutoff=1.96, zscore_cutoff1=1.78, B=500, B1=4500,
             seed=1001, task="DMR", stratify=True, Ls=(0, 10, 20), mergecluster=True, merge_cutoff=0.6, ncore=4):
    """Function to call DMseg.
    Parameters
    ----------
    betafile : string
        path of csv file for the beta data
    colDatafile : string
        path of covariates file, first column is for sample names, second column is for the group variable to compare
    positionfile : string
        path of CpGs position info file, first column is for CpG name, second column for chromosome, third column for position
    maxgap : int
        The distance cutoff to define a cluster of contiguous CpGs, default is 500BP
    sd_cutoff : float
        standard deviation cutoff of beta values. A ROI cluster should have at least one CpG with greater deviation, default is 0.025
    diff_cutoff : float
        cutoff based mean diff between groups. A ROI cluster should have at least two CpG with coef greater than the cutoff, default is 0.05
    zscore_cutoff : float
        cutoff based on z-score. A ROI cluster should have at least two strong CpGs with zscore greater than the cutoff, default is 1.96
    zscore_cutoff1 : float
        second cutoff based on zscore, to select a weak CpG if the CpG is in the middle of two strong CpGs, default is 1.78
    B : int
        The number of permutation for all CpGs, default is 500
    B1 : int
        The number of added permutation for long clusters, default is 4500
    seed : int
        random seed number for permutation process
    task : string
        if task is "DMR" then the function call DMRs, otherwize call VMRs
    stratify : bool
        if TRUE, stratify clusters based on width
    Ls : tuple
        it is used to stratify clusters based on their widths. 
        in default Ls=(0,10,20) which means we will compute p-values for 1) short clusters with width <=10,
        and 2) short clusters with width in (10,20], and 3) long clusters with width>20 separately.
    mergecluster : bool
        if TRUE, merge two nearby clusters if they have high correlation between edge CpGs
    merge_cutoff : float
        cutoff of correlation between edge CpGs of two nearby clusters, used to merge clusters, default is 0.6
    ncore : int
        number of CPU used for computation. ncore=1: not to use parallel computing
    Returns
    -------
    dataframe
        Each row respresents a detected peak/segment, with position info, seg_mean, LRT, p-value and FWER
    """
    if ncore > mp.cpu_count()-1:
        ncore = mp.cpu_count()-1
    # the default option is DMR, testing regions with differential mean methylation
    # alternative option is VMR, testing regions with differential variability, diff_cutoff=0 if testing VMR
    print("program starts: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    if ncore > 1:
        print(f"Use {ncore} CPU to compute...")
    
    beta = pd.read_csv(betafile, delimiter=',', index_col=0)
    # in case there are missing beta values
    if beta.isnull().sum().sum() > 0:
        for i in beta.index[beta.isnull().any(axis=1)]:     
            beta.loc[i].fillna(beta.loc[i].mean(), inplace=True)

    colData = pd.read_csv(colDatafile, delimiter=',', index_col=0)
    position = pd.read_csv(positionfile, delimiter=',', index_col=0)
    chr = position['chr']
    pos = position['position']
    # call cluster
    cluster = clustermaker(chr=chr, pos=pos, maxgap=maxgap)
    if mergecluster:
        # print("Merge clustesrs...")
        cluster = merge_cluster(beta, chr, pos, cluster, merge_cutoff=merge_cutoff)

    clustergrp = cluster.groupby(cluster)
    # len(clustergrp)  # 170156, 150676 after merging;epic 388459, 367483
    
    # sum(clustergrp.value_counts()==1)   # 117604, most of clusters have 1 CpG

    clusterlen = clustergrp.filter(lambda x: len(x) > 1)
    # len(clusterlen.unique()) #52552,54688,epic 107669,114830
    beta = beta.loc[clusterlen.index]
    # print(len(clusterlen.unique()), "clusters have more than 1 CpG.")
    # next filter out low SD
    allsd = beta.T.std()
    # print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    tmp = clusterlen.groupby(clusterlen)
    # tmp.count().median() #3,0,3.0;epic 3.0,3.0
    # tmp.count().describe() #epic 157
    maxsd = allsd.groupby(clusterlen).max()
    maxsd_vec = np.repeat(maxsd, tmp.count())
    clusterindex = clusterlen[clusterlen.index[maxsd_vec > sd_cutoff]]
    # tmp = clusterindex.groupby(clusterindex)
    # len(tmp) #48096, 50260; epic
    # tmp1 = tmp.count()
    # sum(tmp1<=10) #44190
    # sum((tmp1>10) & (tmp1<=20)) #3592
    # sum(tmp1>20) #314

    # print(len(clusterindex.unique()), "clusters pass sd.cluster filter.")
    # print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    beta = beta.loc[clusterindex.index]
    pos = pos[clusterindex.index]
    chr = chr[clusterindex.index]

    # work on design matrix
    intercept = pd.Series([1]*beta.shape[1], index=beta.columns, name='Intercept')
    group_dummy = pd.get_dummies(data=colData.iloc[:, 0], drop_first=True)
    group_dummy.set_index(beta.columns, inplace=True)
    # check if there are other covariates
    if colData.shape[1] > 1:
        other_dummy = pd.get_dummies(data=colData.iloc[:, 1:], drop_first=True)
        other_dummy.set_index(beta.columns, inplace=True)
        design = pd.concat([intercept, group_dummy, other_dummy], axis=1)
    else:
        design = pd.concat([intercept, group_dummy], axis=1)
    # To detect VMR
    if task != "DMR":
        beta = form_vmrdata(beta, design)
        diff_cutoff = 0

    # work on observed data
    Pstats = fit_model_probes(beta, design)

    DMseg_out = Segment_satistics(Pstats, clusterindex, chr, pos, diff_cutoff, zscore_cutoff, zscore_cutoff1)
    # the final output
    DMseg_out2 = None
    # work on permutation data
    if len(DMseg_out) > 0:
        # start = time.time()
        if ncore > 1:
            allsimulation = do_simulationparallel(beta, design, clusterindex, chr, pos, diff_cutoff,
                                                  zscore_cutoff, zscore_cutoff1, seed, B, ncore)
        else:
            allsimulation = do_simulationsequential(beta, design, clusterindex, chr, pos, diff_cutoff,
                                                    zscore_cutoff, zscore_cutoff1, seed, B)
        # end = time.time()
        # print(end-start)
        # width (number of CpGs) for each cluster
        clusterwidth = clusterindex.groupby(clusterindex).count()
        # if null stratified by cluster length
        if stratify:
            # add extra permutation to cluster with #cpgs> 20
            longclusters = clusterwidth.index[clusterwidth > max(Ls)]
            # shortclusters = clusterwidth.index[clusterwidth <= max(Ls)]
            # all CpGs in long clusters
            longcpgs = clusterindex.index[clusterindex.isin(longclusters)]
            if len(longcpgs) > 0:
                # use different seed to make sure top 500 are not the same as before
                if ncore > 1:
                    allsimulation2 = do_simulationparallel(beta.loc[longcpgs], design, clusterindex.loc[longcpgs], chr.loc[longcpgs], pos.loc[longcpgs], diff_cutoff,
                                                           zscore_cutoff, zscore_cutoff1, seed=seed+10000+B1, B=B1, ncore=ncore)
                else:
                    allsimulation2 = do_simulationsequential(beta.loc[longcpgs], design, clusterindex[longcpgs], chr[longcpgs], pos[longcpgs], diff_cutoff,
                                                             zscore_cutoff, zscore_cutoff1, seed=seed+100000+B1, B=B1)
                DMseg_out2 = compute_fwer_all(DMseg_out, allsimulation, B, clusterwidth, allsimulation2, B1, longclusters, Ls)
            else:
                DMseg_out2 = compute_fwer_all(DMseg_out, allsimulation, B, clusterwidth)
        else:
            DMseg_out2 = compute_fwer_all(DMseg_out, allsimulation, B, clusterwidth, Ls=0)
        # print(sum(DMseg_out2["FWER"]<0.05))

    print("Program ends: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return DMseg_out2


def main():
    """ to be used in command line"""
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("-betafile", dest="betafile", help="file for beta matrix", required=True)
    p.add_argument("-colDatafile", dest="colDatafile", help="file for colData (covariates matrix)", required=True)
    p.add_argument("-positionfile", dest="positionfile", help="file for position of CpGs", required=True)
    p.add_argument("-maxgap", dest="maxgap", help="max gap to determine clusters", default=500)
    p.add_argument("-sd_cutoff", dest="sd_cutoff", help="SD cutoff to filter clusters", default=0.025)
    p.add_argument("-beta_diff_cutoff", dest="beta_diff_cutoff", help="Beta difference cutoff to filter clusters", default=0.05)
    p.add_argument("-zscore_cutoff", dest="zscore_cutoff", help="Z score cutoff to filter clusters", default=1.96)
    p.add_argument("-zscore_cutoff1", dest="zscore_cutoff1", help="Z score cutoff to merge clusters", default=1.78)
    p.add_argument("-seed", dest="seed", help="random seed for permutation", default=1001)
    p.add_argument("-task", dest="task", help="DMR or VMR", default="DMR")
    p.add_argument("-B", dest="B", help="number of permutation", default=500)
    p.add_argument("-B1", dest="B1", help="number of extra permutation for long clusters", default=4500)
    p.add_argument("-stratify", dest="stratify", help="Option of stratify NULL in permutation based on cluster length", default=True)
    p.add_argument("-Ls", dest="Ls", help="Intervals used to stratify NULL in permutation based on cluster length", default=(0, 10, 20))
    p.add_argument("-mergecluster", dest="mergecluster", help="Option of merging neighborhood clusters based on correlation of edge CpGs", default=True)
    p.add_argument("-merge_cutoff", dest="merge_cutoff", help="Cutoff used for merging neighborhood clusters", default=0.6)
    p.add_argument("-ncore", dest="ncore", help="number of CPU for parallel computation", default=4)
    args = p.parse_args()
    return pipeline(args.betafile, args.colDatafile, args.positionfile, args.maxgap, args.sd_cutoff, args.beta_diff_cutoff, args.zscore_cutoff, args.zscore_cutoff1, args.B, args.B1, args.seed, args.task, args.stratify, args.Ls, args.mergecluster, args.merge_cutoff, args.ncore)


if __name__ == "__main__":
    main()
