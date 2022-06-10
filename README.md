## DMseg: detecting differential methylation regions (DMRs) in methylome 


This is a Python package to search through methylome-side CpGs sites for DMRs between two biological conditions. The algorithm executes the following analysis steps:

1.  A linear regression model is fitted to the beta values for each CpG, using the user-input covariates: the first variable is the group label of interest.
2.  Nominal p-values from individual CpG associations are used to define the DMR: more than or equal to two consecutive CpGs with p-value <0.05 will be considered as candidate DMR. A likelihood ratio statistic (LRT) is computed for a candidate DMR.
3.  Group labels will be permuted for `B` times, step `1` and `2` are repeated for each permuation dataset. Family-wise error rate is computed using the null distribution of LRT based on permutation. 


To install the package: 

* Installing locally using
`
python setup.py install
`
* Installing from github
`pip install git+https://github.com/xwang234/DMseg
`

To run the python package

```
result = dmseg(betafile, colDatafile, positionfile, maxgap=500, sd_cutoff=0.025, diff_cutoff=0.05, zscore_cutoff=1.96,zscore_cutoff1=1.78, B=500, B1=4500, seed=1001, task="DMR",stratify=True,Ls=[0,10,20],mergecluster=True,corr_cutoff=0.6,ncore=4)
```
