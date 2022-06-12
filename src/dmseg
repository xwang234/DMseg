#!/usr/bin/env python
"""
   Call DMseg.
"""
from __future__ import print_function
import numpy as np
from time import localtime, strftime
import pandas as pd
import sys
import os.path as op
import multiprocessing as mp
from src.functions import *

def pipeline (betafile, colDatafile, positionfile, maxgap=500, sd_cutoff=0.025, diff_cutoff=0.05, zscore_cutoff=1.96,zscore_cutoff1=1.78, B=500, B1=4500, seed=1001, task="DMR",stratify=True,Ls=[0,10,20],mergecluster=True,corr_cutoff=0.6,ncore=4, output="output.csv"):
    #import src.functions
    if ncore > mp.cpu_count()-1:
        ncore = mp.cpu_count()-1
    # the default option is DMR, testing regions with differential mean methylation
    # alternative option is DVR, testing regions with differential variability, diff_cutoff=0 if testing DVR
    print("program starts: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    
    beta = pd.read_csv(betafile, delimiter=',', index_col=0)
        #in case there are missing beta values
    if beta.isnull().sum().sum() > 0:
        for i in beta.index[beta.isnull().any(axis=1)]:     
            beta.loc[i].fillna(beta.loc[i].mean(),inplace=True)

    colData = pd.read_csv(colDatafile, delimiter=',', index_col=0)
    position = pd.read_csv(positionfile, delimiter=',', index_col=0)
    chr = position['chr']
    pos = position['position']
    #call cluster
    cluster = clustermaker(chr=chr, pos=pos, maxgap=maxgap)
    if mergecluster:
        #print("Merge clustesrs...")
        cluster = merge_cluster(beta, chr, pos, cluster,corr_cutoff=corr_cutoff)

    clustergrp = cluster.groupby(cluster)
    len(clustergrp)  # 170156, 150676 after merging;epic 388459, 367483
    
    sum(clustergrp.value_counts()==1)   # 117604, most of clusters have 1 CpG

    clusterlen=clustergrp.filter(lambda x:len(x) > 1)
    len(clusterlen.unique()) #52552,54688,epic 107669,114830
    beta = beta.loc[clusterlen.index]
    #print(len(clusterlen.unique()), "clusters have more than 1 CpG.")
    # next filter out low SD
    allsd = beta.T.std()
    #print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    tmp = clusterlen.groupby(clusterlen)
    #tmp.count().median() #3,0,3.0;epic 3.0,3.0
    #tmp.count().describe() #epic 157
    maxsd=allsd.groupby(clusterlen).max()
    maxsd_vec=np.repeat(maxsd,tmp.count())
    clusterindex=clusterlen[clusterlen.index[maxsd_vec>sd_cutoff]]
    #tmp = clusterindex.groupby(clusterindex)
    #len(tmp) #48096, 50260; epic
    #tmp1 = tmp.count()
    # sum(tmp1<=10) #44190
    # sum((tmp1>10) & (tmp1<=20)) #3592
    # sum(tmp1>20) #314

    #print(len(clusterindex.unique()), "clusters pass sd.cluster filter.")
    #print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    beta = beta.loc[clusterindex.index]
    pos=pos[clusterindex.index]
    chr=chr[clusterindex.index]

    #work on design matrix
    intercept = pd.Series([1]*beta.shape[1], index=beta.columns,name='Intercept')
    group_dummy = pd.get_dummies(data=colData.iloc[:, 0], drop_first=True)
    group_dummy.set_index(beta.columns, inplace=True)
    ## check if there are other covariates
    if colData.shape[1] > 1:
        other_dummy = pd.get_dummies(data=colData.iloc[:, 1:], drop_first=True)
        other_dummy.set_index(beta.columns, inplace=True)
        design = pd.concat([intercept, group_dummy, other_dummy], axis=1)
    else:
        design = pd.concat([intercept, group_dummy], axis=1)
    # To detect VMR
    if task != "DMR":
        beta = form_dvcdata(beta,design)
        diff_cutoff =0

    #work on observed data
    Pstats = fit_model_probes(beta, design)

    DMseg_out = Segment_satistics(Pstats,clusterindex,pos=pos,chr=chr,diff_cutoff=diff_cutoff,zscore_cutoff=zscore_cutoff,zscore_cutoff1=zscore_cutoff1)
    DMseg_out2 = None
    #work on simulation data
    if len(DMseg_out) >0 :
        start = time.time()
        if ncore>1:
            allsimulation = do_simulation1(beta,design,clusterindex,pos=pos,chr=chr,seed=seed,diff_cutoff=diff_cutoff,zscore_cutoff=zscore_cutoff,zscore_cutoff1=zscore_cutoff1,B=B,ncore=ncore)
        else:
            allsimulation = do_simulation1serial(beta,design,clusterindex,pos=pos,chr=chr,seed=seed,diff_cutoff=diff_cutoff,zscore_cutoff=zscore_cutoff,zscore_cutoff1=zscore_cutoff1,B=B)
        end = time.time()
        #print(end-start)
        if stratify: #new results, null stratified by cluster length
            #add extra permutation to cluster with #cpgs> 20
            tmp = clusterindex.groupby(clusterindex)
            clusterwidth = tmp.count()
            longclusters = clusterwidth.index[clusterwidth > max(Ls)]
            #shortclusters = clusterwidth.index[clusterwidth <= max(Ls)]
            longcpgs = clusterindex.index[clusterindex.isin(longclusters)]
             #use different seed to make sure top 500 are not the same as before
            if ncore>1:
                allsimulation2 = do_simulation1(beta.loc[longcpgs],design,clusterindex[longcpgs],pos=pos[longcpgs],chr=chr[longcpgs],seed=seed+10000+B1,diff_cutoff=diff_cutoff,zscore_cutoff=zscore_cutoff,zscore_cutoff1=zscore_cutoff1,B=B1,ncore=ncore)
            else:
                allsimulation2 = do_simulation1serial(beta.loc[longcpgs],design,clusterindex[longcpgs],pos=pos[longcpgs],chr=chr[longcpgs],seed=seed+100000+B1,diff_cutoff=diff_cutoff,zscore_cutoff=zscore_cutoff,zscore_cutoff1=zscore_cutoff1,B=B1)
        
        
            DMseg_out2 = compute_fwer_all(DMseg_out,allsimulation,allsimulation2,B,B1,longclusters,clusterwidth,Ls)
            #print(sum(DMseg_out2["FWER"]<0.05))
        else:
            DMseg_out2 = compute_fwerall(DMseg_out,allsimulation,B,clusterindex,Ls=[0])
            #sum(DMseg_out2["FWER"]<0.05)
    DMseg_out2.to_csv(output, index=False)
    print(DMseg_out2[DMseg_out2["FWER"]<0.05])
    print(f"Output was written into {output}")
    print("Program ends: " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
    
    return DMseg_out2



def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("-betafile", dest="betafile", help="file for beta matrix", required=True)
    p.add_argument("-colDatafile", dest="colDatafile", help="file for colData matrix", required=True)
    p.add_argument("-positionfile", dest="positionfile", help="file for position of CpGs", required=True)
    p.add_argument("-maxgap", dest="maxgap", help="max gap to deterine clusters", default=500)
    p.add_argument("-sd_cutoff", dest="sd_cutoff", help="SD cutoff to filter clusters", default=0.025)
    p.add_argument("-beta_diff_cutoff", dest="beta_diff_cutoff", help="Beta difference cutoff to filter clusters", default=0.05)
    p.add_argument("-zscore_cutoff", dest="zscore_cutoff", help="Z score cutoff to filter clusters", default=1.96)
    p.add_argument("-zscore_cutoff1", dest="zscore_cutoff1", help="Z score cutoff to merge clusters", default=1.78)
    p.add_argument("-seed", dest="seed", help="random seed for permutation", default=1001)
    p.add_argument("-task", dest="task", help="DMR or VMR", default="DMR")
    p.add_argument("-B", dest="B", help="number of permutation", default=500)
    p.add_argument("-B1", dest="B1", help="number of extra permutation for long clusters", default=4500)
    p.add_argument("-stratify", dest="stratify", help="Option of stratify NULL in permutation based on cluster length", default=True)
    p.add_argument("-Ls", dest="Ls", help="Intervals used to stratify NULL in permutation based on cluster length", default=[0,10,20,30])
    p.add_argument("-mergecluster", dest="mergecluster", help="Option of merging neighborhood clusters based on correlation of edge CpGs", default=True)
    p.add_argument("-corr_cutoff", dest="corr_cutoff", help="Cutoff used for merging neighborhood clusters", default=0.6)
    p.add_argument("-ncore", dest="ncore", help="number of CPU for parallel computation", default=4)
    p.add_argument("-output", dest="output", help="output file", default="output.csv")
    args = p.parse_args()
    return pipeline(args.betafile, args.colDatafile, args.positionfile, args.maxgap, args.sd_cutoff, args.beta_diff_cutoff, args.zscore_cutoff, args.zscore_cutoff1, args.B, args.B1, args.seed, args.task, args.stratify, args.Ls, args.mergecluster, args.corr_cutoff, args.ncore)


if __name__ == "__main__":
    main()





