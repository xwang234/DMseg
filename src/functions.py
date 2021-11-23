#!/usr/bin/env python
"""
   functions used in DMseg.
"""
from __future__ import print_function
import numpy as np
from functools import partial
from time import localtime, strftime
import pandas as pd
#import argparse
#import sys

# other functions
# function used for model fitting


def nacomb(x):
    tmp = np.where(np.isnan(x))
    tmp1 = ";".join(str(v) for v in tmp[0])
    return tmp1


# calculate.stats, function used for model fitting
def calculate_stats(v, na_comb, beta, design):
    mat_sub = beta.loc[na_comb == v]
    if v != '':
        tmp = np.asarray(v.split(";"), dtype=np.int64)
        mat_sub = mat_sub.drop(mat_sub.columns[tmp], axis=1)
    design_sub = design.loc[design.index.isin(mat_sub.columns)]
    # Calculate coef
    M = design_sub.drop(design_sub.columns[1], axis=1)
    M_QR_q, M_QR_r = np.linalg.qr(M)
    S = np.diag([1]*M.shape[0]) - np.matmul(M_QR_q, M_QR_q.transpose())
    V = design_sub.iloc[:, 1]
    SV = np.matmul(S, V)
    coef = np.matmul(mat_sub, np.matmul(S.transpose(), V)) / \
        np.matmul(V.transpose(), SV)
    # Calculate residuals
    QR_X_q, QR_X_r = np.linalg.qr(design_sub)
    np.allclose(design_sub, np.matmul(QR_X_q, QR_X_r))
    resids = np.diag([1]*design_sub.shape[0]) - \
        np.matmul(QR_X_q, QR_X_q.transpose())
    resids = np.matmul(resids, mat_sub.transpose())
    resids = pd.DataFrame(resids)
    resids.set_index(mat_sub.columns, inplace=True)

    # Fill the residual matrix to full size of all samples. Residuals of samples with NAs are 0.
    removed_samples = design.index[~design.index.isin(mat_sub.columns)]
    removed_mat = pd.DataFrame(
        0, index=removed_samples, columns=resids.columns)
    resids = pd.concat([resids, removed_mat])
    resids = pd.DataFrame(resids, index=design.index)

    # Calculate SE
    tmp1 = np.linalg.inv(design_sub.T.dot(design_sub))[
        1, 1]/(mat_sub.shape[1]-np.linalg.matrix_rank(M)-1)
    SE = (resids.pow(2).sum()*tmp1)**(1/2)

    stats_sub = pd.concat([coef, SE, coef.div(SE)], axis=1)
    stats_sub.columns = ["coef", "sd", "zscore"]
    result = dict(stats=stats_sub, residuals=resids)
    return result


# calculate LRT for observed data
def LRT_stat(bdiff, bvar):
    n_probes = len(bdiff)
    cmat = np.diag(bvar)
    sigma1 = np.linalg.inv(cmat)
    JJ = np.ones((1, n_probes))
    bb0 = np.linalg.inv(np.matmul(np.matmul(JJ, sigma1), JJ.T))
    bb1 = np.matmul(np.matmul(JJ, sigma1), bdiff)
    seg_mean = np.matmul(bb0, bb1)
    lrt = np.matmul(np.matmul(bdiff-seg_mean, sigma1), bdiff-seg_mean)
    lrt = np.matmul(np.matmul(bdiff, sigma1), bdiff) - lrt
    result = dict(lrt=lrt, seg_mean=seg_mean)
    return result


# compute LRT for simulation
def LRT_statall(allbdiff, allbvar):
    sigma2 = 1/allbvar
    tmp = np.multiply(allbdiff, allbdiff) > 0
    tmp = tmp.astype(int)
    #tmp1 = np.multiply(tmp, sigma2)
    seg_mean = np.multiply(1/np.multiply(tmp, sigma2).sum(axis=1),
                           np.multiply(np.multiply(tmp, sigma2), allbdiff).sum(axis=1))
    lrt = np.multiply(np.multiply(np.multiply(tmp, sigma2), (allbdiff.T -
                      np.array(seg_mean)).T), (allbdiff.T-np.array(seg_mean)).T).sum(axis=1)
    lrt = np.multiply(np.multiply(np.multiply(tmp, sigma2),
                      allbdiff), allbdiff).sum(axis=1) - lrt
    result = dict(lrt=lrt, seg_mean=seg_mean)
    return result


# find peaks in observed data
def get_segments(bdiff, sd, bvar, cutoff=1.96):
    cutoff = np.array([cutoff])
    zscore = bdiff/sd
    assert(len(cutoff) < 2)
    if len(cutoff) == 1:
        cutoff = [-cutoff[0], cutoff[0]]
    cutoff = np.sort(cutoff)

    direction = np.zeros(len(zscore))
    direction = np.where(zscore >= cutoff[1], 1, direction)
    direction = np.where(zscore <= cutoff[0], -1, direction)
    direction = pd.Series(direction, index=bdiff.index)

    tmp0 = 1*(np.diff(direction) != 0)
    tmp0 = np.insert(tmp0, 0, 1)
    segments = np.cumsum(tmp0)
    segments = pd.Series(segments, index=bdiff.index)
    peakidx = np.where(direction.isin([-1, 1]))[0]
    peakidx = pd.Series(peakidx, index=bdiff.index[peakidx])
    peakidx = peakidx.to_frame()
    peaksegments = pd.concat(
        [peakidx, segments[peakidx.index], direction[peakidx.index]], axis=1)
    peaksegments.columns = ['idx', 'segment', 'direction']
    peaksegments_groupby = peaksegments.groupby(by="segment")
    peaksegments = peaksegments_groupby.filter(lambda x: len(x) > 1)
    peaksegments_groupby = peaksegments.groupby(by="segment")

    res = pd.DataFrame(
        columns=["start", "end", "direction", "LRT", "seg_mean"])
    for _, segs in peaksegments_groupby:
        # print(segs)
        tmp2 = LRT_stat(bdiff[segs.index], bvar[segs.index])
        res = res.append({'start': segs.idx[0], 'end': segs.idx[len(segs.idx)-1],
                          'direction': segs.direction[0], 'LRT': tmp2['lrt'], 'seg_mean': tmp2['seg_mean']},
                         ignore_index=True)
    return res


def match(a, b): return [b.index(x) if x in b else None for x in a]


# find peaks in simulation
def get_segmentsall(bdiff1, sd, bvar, cutoff=1.96, B=500):
    cutoff = [cutoff]
    if len(cutoff) == 1:
        cutoff = np.array([-cutoff[0], cutoff[0]])
    zscore = bdiff1/sd
    #f = np.ones(bdiff1.shape)

    direction = np.zeros(zscore.shape)
    # direction: CpG peak indicator, based on zscore: positive 1, negative -1, others 0
    direction = np.where(zscore >= cutoff[1], 1, direction)
    direction = np.where(zscore <= cutoff[0], -1, direction)
    direction = pd.DataFrame(direction, columns=sd.index, index=bdiff1.index)
    direction = direction.astype(int)

    # segments are contiguous CpGs with respect to dirrection status in each simulation (0's/1's/-1s' in direction)
    tmp0 = 1*(np.diff(direction) != 0)
    tmp0 = np.hstack((np.ones((tmp0.shape[0], 1)), tmp0))
    segments = np.cumsum(tmp0, axis=1)
    segments = pd.DataFrame(segments, columns=sd.index, index=bdiff1.index)
    segments = segments.astype(int)
    # check direction and segments:
    # direction.iloc[0,:].to_list()
    # segments.iloc[0,:].to_list()

    # combinedsegments considers both direction and segments
    combinedsegments = np.multiply(direction, segments)
    # combinedsegments.iloc[0,:].to_list()
    tmp1 = np.diff(combinedsegments, axis=1)
    # a region needs to have at least 1 segment with >1 cpgs
    idx = (tmp1 == 0).any(axis=1)
    direction = direction[idx]
    segments = segments[idx]
    combinedsegments = np.multiply(direction, segments)
    # for regions with 2 CpGs, to remove some simulation no need to check
    # if direction.shape[1] == 2:
    #     direction = direction[tmp1 == 0]
    #     segments = segments[tmp1 == 0]

    #pd.concat([combinedsegments, direction, segments], axis=1)
    allres = {"lrtmax": pd.Series([None]*B), "lrt": None}
    # allbidff can be array or dataframe, array is faster, dataframe is easier for debug
    opt_allbidff = "array"
    if segments.shape[0] > 0:
        # Make sure a simulation has peaks with length > 1
        # tmp1 contains number of CpGs in each combined segment
        tmp1 = combinedsegments.apply(pd.Series.value_counts, axis=1).fillna(0)
        # combined sements wiht all 0 are not peaks
        tmp1.loc[:, 0] = 0
        # tmp2: if combinded segments have more than 1 CpGs
        tmp2 = tmp1 > 1
        # simcheck=F are simulations without a peak with more than 1 CpGs, if that is the case no need to run
        simcheck = tmp2.any(axis=1)
        # segcheck=F are segs without a peak with more than 1 CpGs, if that is the case no need to run
        segcheck = tmp2.any(axis=0)

        if np.sum(simcheck) * np.sum(segcheck) > 0:
            simcheck = simcheck[simcheck == True]
            segcheck = segcheck[segcheck == True]
            # simulations need to check
            combinedsegments = combinedsegments.loc[simcheck.index]

            # allbdiff is the combination of peaks, matrix form, each row contains one peak from a simulation (other cpgs are filled with 0)
            # allbidff can be dataframe
            if opt_allbidff == "dataframe":
                allbdiff = pd.DataFrame()
            else:
                #allbidff is array
                allbdiff = np.array([], dtype=np.float64).reshape(
                    0, bdiff1.shape[1])
                simidx = []
            # seg are peaks with cpgs>1 to be included in allbdiff
            for seg in segcheck.index:
                # print(seg)
                # row,col indices for CpGs in seg, some of them are not included in peaks with #cpgs>1
                idx = np.where(combinedsegments == seg)
                # print(idx)
                # simulation (row) indices need to check (peaks with #cpg>1)
                idx1 = tmp2.index[np.where(tmp2.loc[:, seg] == True)]
                idx2 = np.in1d(combinedsegments.index[idx[0]], idx1)
                # row indices need to check
                rowidx = combinedsegments.index[idx[0][idx2]]
                # column indeces need to check
                colidx = combinedsegments.columns[idx[1][idx2]]

                # tmp5: matrix (for seg) to fill using bdiff1 data
                #tmp5 = pd.DataFrame(0, index=rowidx.unique(), columns=bdiff1.columns.unique())
                tmp5 = np.zeros((len(rowidx.unique()), bdiff1.shape[1]))
                idx1 = match(rowidx.to_list(), rowidx.unique().tolist())
                idx2 = match(colidx.to_list(), bdiff1.columns.tolist())
                idx11 = match(rowidx.to_list(), bdiff1.index.tolist())
                tmp5[idx1, idx2] = bdiff1.values[idx11, idx2]
                # tmp5 include all CpGs
                if opt_allbidff == "dataframe":
                    tmp5 = pd.DataFrame(
                        tmp5, index=rowidx.unique(), columns=bdiff1.columns)
                    allbdiff = allbdiff.append(tmp5)
                else:
                    allbdiff = np.vstack([allbdiff, tmp5])
                    simidx.append(rowidx.unique().tolist())

            if opt_allbidff == "dataframe":
                simidx = allbdiff.index
            else:
                simidx = [x for sublist in simidx for x in sublist]
            # max lrt in each simulation
            lrtmax = pd.Series([None]*B)
            if allbdiff.shape[0] > 0:
                if opt_allbidff == "dataframe":
                    lrtres = LRT_statall(allbdiff=allbdiff, allbvar=bvar)
                else:
                    lrtres = LRT_statall(
                        allbdiff=allbdiff, allbvar=np.array(bvar))
                lrtmax1 = pd.DataFrame(
                    {"lrt": lrtres['lrt'], "simidx": simidx})
                lrtmax1_groupby = lrtmax1.groupby(by="simidx")
                lrtmax2 = lrtmax1_groupby.max()
                lrtmax2 = lrtmax2.iloc[:, 0]
                lrtmax.loc[lrtmax2.index] = lrtmax2
                # lrt contains all lrt found in simulation (1 simulation can have multiple peaks and multiple lrt)
                allres = {"lrtmax": lrtmax, "lrt": lrtres['lrt']}
    return allres


# class
class DMsegobj:
    def __init__(self, beta, colData, chr, pos, maxgap=500, sd_cutoff=0.025, beta_diff_cutoff=0.05, zscore_cutoff=1.96, seed=1000, B=100):
        self.beta = beta
        self.colData = colData
        self.chr = chr
        self.design = None
        self.pos = pos
        self.maxgap = maxgap
        self.beta_diff_cutoff = beta_diff_cutoff
        self.sd_cutoff = sd_cutoff
        self.zscore_cutoff = zscore_cutoff
        self.seed = seed
        self.B = B
        self.cluster = None
        self.clustersd = None
        self.stats = None
        self.ROIcluster = None
        self.ROVcluster = None
        self.segments_alt = None
        self.segments_null = None
        self.regions = None

    def create_design(self):
        # first column is sample name, second column is group
        intercept = pd.Series([1]*self.beta.shape[1], index=self.beta.columns)
        group_dummy = pd.get_dummies(
            data=self.colData.iloc[:, 1], drop_first=True)
        group_dummy.set_index(self.beta.columns, inplace=True)
        other_dummy = pd.get_dummies(
            data=self.colData.iloc[:, 2:], drop_first=True)
        other_dummy.set_index(self.beta.columns, inplace=True)
        design = pd.concat([intercept, group_dummy, other_dummy], axis=1)
        return design

    def clusters(self):
        tmp2 = self.chr.groupby(by=self.chr, sort=False)
        tmp3 = tmp2.count()
        Indexes = tmp3.cumsum().to_list()
        Indexes.insert(0, 0)
        clusterIDs = pd.Series(
            data=[None]*self.pos.shape[0], index=self.chr.index)
        Last = 0
        for i in range(len(Indexes)-1):
            i1 = Indexes[i]
            i2 = Indexes[i+1]
            Index = range(i1, i2)
            x = self.pos.iloc[Index]
            # sort
            tmp = [j-1 for j in x.rank()]
            x = x.iloc[tmp]
            y = np.diff(x) > self.maxgap
            y = np.insert(y, 0, 1)
            z = np.cumsum(y)
            clusterIDs.iloc[i1:i2] = z + Last
            Last = max(z) + Last
        return clusterIDs

    def filter_sd(self):
        tmp = self.cluster.groupby(self.cluster)
        allsd = self.beta.T.std()
        #strftime("%Y-%m-%d %H:%M:%S", localtime())
        clustersd = tmp.filter(lambda x: allsd[x.index].max() > self.sd_cutoff)
        # only use CpGs pass sd filter
        self.beta = pd.DataFrame(self.beta, index=clustersd.index)
        return clustersd

        # model.fit
    def fit_model_probe(self):
        na_comb = self.beta.apply(nacomb, axis=1)
        na_comb_unique = na_comb.unique()
        tmp = list(map(partial(calculate_stats, na_comb=na_comb,
                   beta=self.beta, design=self.design), na_comb_unique))
        stats = dict()
        stats['fit_x'] = self.design
        stats['statistics'] = pd.DataFrame()
        for j in range(len(tmp)):
            stats['statistics'] = pd.concat(
                [stats['statistics'], tmp[j]['stats']])

        stats['statistics'] = pd.DataFrame(
            stats['statistics'], index=self.beta.index)
        stats['residuals'] = pd.DataFrame()
        for j in range(len(tmp)):
            stats['residuals'] = pd.concat(
                [stats['residuals'], tmp[j]['residuals']], axis=1)

        stats['residuals'] = pd.DataFrame(
            stats['residuals'], columns=self.beta.index)
        return stats

    def create_rovcluster(self):
        # clusters with len > 1
        clustersd_groupby = self.clustersd.groupby(self.clustersd)
        ROVcluster = clustersd_groupby.filter(lambda x: len(x) > 1)
        return ROVcluster

    def create_roicluster(self):
        ROVcluster_groupby = self.ROVcluster.groupby(self.ROVcluster)
        mycoef = self.stats["statistics"]['coef']
        myzscore = self.stats["statistics"]['zscore']
        ROIcluster = ROVcluster_groupby.filter(lambda x: (mycoef[x.index].abs().max(
        ) > self.beta_diff_cutoff and myzscore[x.index].abs().max() > self.zscore_cutoff))
        return ROIcluster

    def evaluate_segments_all(self, simulation):
        if not simulation:
            DMseg_sd = self.stats["statistics"]['sd'][self.ROIcluster.index]
            DMseg_coef = self.stats["statistics"]['coef'][self.ROIcluster.index]
            DMseg_pos = self.pos[self.ROIcluster.index]
            DMseg_chr = self.chr[self.ROIcluster.index]
            DMseg_stats = pd.DataFrame({'coef': DMseg_coef,
                                        'sd': DMseg_sd,
                                        'chr': DMseg_chr,
                                        'pos': DMseg_pos,
                                        'cluster': self.ROIcluster},
                                       columns=["coef", "sd", "chr", "pos", "cluster"])
            DMseg_stats = DMseg_stats.groupby(by="cluster")
            # find a case to debug
            # DMseg_stats_groupby = DMseg_stats.groupby(by = "cluster")
            # tmp = DMseg_stats_groupby.groups
            # DMseg_stats_groupby_len = [len(tmp[x]) for x in DMseg_stats_groupby.groups.keys()]
            # DMseg_stats_groupby_len.index(97)
            # list(DMseg_stats_groupby.groups.keys())[6295]
            # bstats = DMseg_stats.loc[DMseg_stats_groupby.groups[60991]]

            cluster_index = []
            cluster_L = []
            cluster_probes = []
            cluster_chr = []
            cluster_pos = []
            segment_L = []
            segment_probes = []
            segment_LRT = []
            segment_mean = []
            for keys, bstats in DMseg_stats:
                bdiff = bstats['coef']
                sd = bstats['sd']
                bvar = np.diag(np.linalg.inv(np.diag(sd*sd)))
                bvar = pd.Series(bvar, index=bdiff.index)
                mychar = bstats['chr'][0]
                mypos = bstats['pos'].astype(int)
                DM_segments_tmp1 = get_segments(bdiff, sd, bvar, cutoff=1.96)
                cluster_tmp = bstats['cluster'][0]
                if DM_segments_tmp1 is not None:
                    probe_names = bdiff.index.to_list()
                    for i in range(len(DM_segments_tmp1)):
                        idx1 = DM_segments_tmp1['start'][i]
                        idx2 = DM_segments_tmp1['end'][i]
                        cluster_index.append(cluster_tmp)
                        cluster_L.append(len(probe_names))
                        cluster_probes.append(";".join(str(v)
                                              for v in probe_names))
                        cluster_chr.append(mychar)
                        cluster_pos.append(";".join(str(v) for v in mypos))
                        segment_L.append(idx2-idx1+1)
                        segment_LRT.append(DM_segments_tmp1['LRT'][i])
                        segment_mean.append(DM_segments_tmp1['seg_mean'][i])
                        segment_probes.append(";".join(str(v)
                                              for v in probe_names[idx1:(idx2+1)]))
            DMseg_segments = dict(cluster_index=cluster_index, cluster_L=cluster_L, cluster_probes=cluster_probes,
                                  cluster_chr=cluster_chr, cluster_pos=cluster_pos, segment_L=segment_L, segment_probes=segment_probes,
                                  segment_LRT=segment_LRT, segment_mean=segment_mean)
            # DMseg_segments = pd.DataFrame(data = DMseg_segments)

            # DMseg_segments.to_csv("//center/fh/fast/dai_j/Programs/DMseg1/DMseg/R/DMseg_segments_EAC_BE.csv")
        else:
            DMseg_sd = self.stats["statistics"]['sd'][self.ROVcluster.index]
            fit_x = self.stats['fit_x']
            solve_coef = np.linalg.inv(fit_x.T.dot(fit_x))[1, 1]
            DMseg_residuals = self.stats["residuals"].loc[:,
                                                          self.ROVcluster.index]
            DMseg_stats1 = pd.concat(
                [DMseg_sd, DMseg_residuals.T, self.ROVcluster], axis=1)
            #DMseg_stats1 = DMseg_stats1.iloc[:5000, :]
            DMseg_stats_names = DMseg_residuals.index.to_list()
            DMseg_stats_names.insert(0, 'sd')
            DMseg_stats_names.insert(len(DMseg_stats_names), 'cluster')
            DMseg_stats1.columns = DMseg_stats_names
            # DMseg_stats1.cluster.astype(int)
            DMseg_stats1_groupby = DMseg_stats1.groupby(by="cluster")
            # find a case (with 97 CpGs) to debug
            # tmp = DMseg_stats1_groupby.groups
            # DMseg_stats1_groupby_len = [len(tmp[x]) for x in DMseg_stats1_groupby.groups.keys()]
            # DMseg_stats1_groupby_len.index(97) #a segment with 97 CpGs
            # tmp1 = list(tmp.keys())[15227] #60991
            # bstats = DMseg_stats1.loc[DMseg_stats1_groupby.groups[60991]]

            # version to use apply, not efficient
            # allres = DMseg_stats1.groupby('cluster').apply(outloop, solve_coef=solve_coef, B=B)
            # tmp = allres.to_list()
            # LRT = [tmp[x]['LRT'] for x in range(len(tmp)) if tmp[x]['LRT'] is not None]
            # LRT = [item for sublist in LRT for item in sublist]
            # maxLRT = [tmp[x]['maxLRT'] for x in range(len(tmp))]

            LRT = []
            maxLRT = []
            for _, bstats in DMseg_stats1_groupby:
                # print(_)
                sd = bstats['sd']
                bvar = np.diag(np.linalg.inv(np.diag(sd*sd)))
                bvar = pd.Series(bvar, index=sd.index)
                mat_tmp = bstats.loc[:, DMseg_residuals.index].T
                cmat = solve_coef * mat_tmp.cov()
                np.random.seed(self.seed)
                mean_sim = np.repeat(0, len(sd))
                coef_sim = np.random.multivariate_normal(
                    mean=mean_sim, cov=cmat, size=self.B)
                coef_sim = pd.DataFrame(coef_sim, columns=sd.index)

                # find segments pass filters
                Index_picked = np.where((coef_sim.abs().max(axis=1) > self.beta_diff_cutoff) * (
                    coef_sim.div(sd).abs().max(axis=1) > self.zscore_cutoff))
                coef_sim1 = coef_sim.loc[Index_picked]
                #DMseg_segments_tmp2={"lrtmax": pd.Series([None]*B), "lrt": None}
                DMseg_segments_tmp2 = get_segmentsall(
                    bdiff1=coef_sim1, sd=sd, bvar=bvar, cutoff=1.96, B=self.B)
                LRT.append(DMseg_segments_tmp2['lrt'])
                maxLRT.append(DMseg_segments_tmp2['lrtmax'])
            DMseg_segments = dict(LRT=LRT, maxLRT=maxLRT)
            # DMseg_segments2.to_csv("//center/fh/fast/dai_j/Programs/DMseg1/DMseg/R/DMseg_segments_EAC_BE_null.csv")
        return DMseg_segments

    def compute_segment_p_value(self):
        LRT_alt = self.segments_alt['segment_LRT']
        LRT_alt = pd.Series(LRT_alt)
        LRT_null = self.segments_null['LRT']
        LRT_max = self.segments_null['maxLRT']

        LRT_null_new = []
        tmp = [item for item in LRT_null if item is not None]
        LRT_null_new = [item for sublist in tmp for item in sublist]
        LRT_null_new = pd.Series(LRT_null_new)
        maxLRT_new = [
            item_maxLRT for sublist in LRT_max for item_maxLRT in sublist]
        maxLRT_new = pd.Series(maxLRT_new)

        LRT_alt_rank = pd.Series(-np.array(LRT_alt.append(LRT_null_new)))
        LRT_alt_rank = LRT_alt_rank.rank(axis=0, na_option='bottom', method='max')[
            0:len(LRT_alt)]
        #rank in simulation
        LRT_alt_rank = LRT_alt_rank - LRT_alt_rank.rank(axis=0)
        # p-value is problematic, to be removed
        p_value = LRT_alt_rank/len(LRT_null_new)

        # make FDR monotone with LRT
        LRT_ord = pd.Series(self.segments_alt['segment_LRT']).sort_values()
        FDR = (LRT_alt_rank/self.B).divide(LRT_alt.rank(method="max", ascending=False))
        FDR[FDR > 1] = 1

        FDR1 = FDR.copy()
        for j in range(1, len(FDR)):
            FDR1[LRT_ord.index[j]] = np.min(
                [FDR[LRT_ord.index[j]], FDR1[LRT_ord.index[j-1]]])

        maxLRT_new = np.array(maxLRT_new)
        maxLRT_new = np.reshape(maxLRT_new, newshape=(len(LRT_max), self.B))
        maxLRT_new = pd.DataFrame(maxLRT_new, columns=range(self.B))

        maxLRT_new1 = maxLRT_new.max(axis=0).fillna(0)
        maxLRT_new1 = maxLRT_new1.to_list()
        LRT_alt_rank = pd.DataFrame(-np.array(LRT_alt.to_list() + maxLRT_new1))
        LRT_alt_rank = LRT_alt_rank.rank(axis=0, na_option='bottom')[
            0:len(LRT_alt)]
        LRT_alt_rank = LRT_alt_rank - LRT_alt_rank.rank(axis=0)

        FWER = LRT_alt_rank / self.B
        DMseg_regions = pd.DataFrame(data=self.segments_alt, index=FDR.index)

        DMseg_regions['p_value'] = p_value
        DMseg_regions['FDR'] = FDR
        DMseg_regions['FDR1'] = FDR1
        DMseg_regions['FWER'] = FWER
        DMseg_regions = DMseg_regions.loc[LRT_ord.index]
        DMseg_regions = DMseg_regions[::-1]

        # DMseg_regions.to_csv("/fh/fast/dai_j/Programs/DMseg1/DMseg/R/DMseg_segments_EAC_BE_regions1.csv")
        return DMseg_regions

