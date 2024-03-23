import numpy as np
import pandas as pd

import scanpy as sc

import plotnine as p9

import liana as li

# load muon and mofax
import muon as mu
import mofax as mofa

import decoupler as dc

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'

adata_kang = sc.read_h5ad('../data/kang18_counts_25k.h5ad')

sample_key = 'sample'
condition_key = 'condition'
groupby = 'cell_abbr'

# Create a multiview structure by pseudobulking with respect to cell type (in this case, cell_abbr)
mdata = li.multi.adata_to_views(adata_kang,
                                groupby=groupby,
                                sample_key=sample_key,
                                keep_stats=True,
                                obs_keys=['condition', 'patient'], # add those to mdata.obs
                                min_prop=0.05, # min nnz values (filter features)
                                min_smpls=3, # min samples per view (filter features)
                                min_cells=25, # min cells per view (filter samples)
                                min_counts=50, # min counts per view (filter samples)
                                mode='sum', # mode of aggregation
                                verbose=True,
                                large_n=5, # edgeR-like filtering
                                min_total_count=15,
                                min_count=10,
                                layer='counts',
                                )

# create dictionary of markers for each cell type
markers = {}
top_n = 25
for cell_type in mdata.mod.keys():
    markers[cell_type] = (sc.get.rank_genes_groups_df(adata_kang, group=cell_type).
                          sort_values("scores", key=abs, ascending=False).
                          head(top_n)['names'].
                          tolist()
                          )
    
li.multi.filter_view_markers(mdata, markers=markers, var_column=None, inplace=True)
mdata.update()
    
# Normalise each view
for view in mdata.mod.keys():

    sc.pp.normalize_total(mdata.mod[view], target_sum=1e4)
    sc.pp.log1p(mdata.mod[view])

    sc.pp.highly_variable_genes(mdata.mod[view])

mu.tl.mofa(mdata,
           use_obs='union',
           convergence_mode='medium',
           n_factors=5,
           seed=0,
           outfile='models/kang18_mofacellx.h5ad',
           use_var='highly_variable'
           )
model =  mofa.mofa_model("models/kang18_mofacellx.h5ad")

# obtain factor scores
factor_scores = li.ut.get_factor_scores(mdata, obsm_key='X_mofa', obs_keys=['condition', 'patient'])
# factor_scores.head()

 # we use a paired t-test as the samples are paired
from scipy.stats import ttest_rel

# split in control and stimulated
group1 = factor_scores[factor_scores['condition']=='ctrl']
group2 = factor_scores[factor_scores['condition']=='stim']

# get all columns that contain factor & loop
factors = [col for col in factor_scores.columns if 'Factor' in col]
for factor in factors:
    print(ttest_rel(group1[factor], group2[factor]))

for factor in factor_scores.columns:
    if 'Factor' in factor:
        # scatterplot
        scatterplot = (p9.ggplot(factor_scores) + 
        p9.aes(x='condition', colour='condition', y=factor) +
        p9.geom_violin() +
        p9.geom_jitter(size=4, width=0.2) +
        p9.theme_bw(base_size=16) +
        p9.theme(figure_size=(5, 4), text=p9.element_text(family="Arial")) +
        p9.scale_colour_manual(values=['#1b9e77', '#d95f02']) +
        p9.labs(x='Condition', y=factor[:-1] + factor[-1])
        )
        scatterplot.save(filename='kang18_mofacellular_factor_scores_condition_' + factor + '.pdf')

variable_loadings =  li.ut.get_variable_loadings(mdata, varm_key='LFs', view_sep=':') # get loadings

for factor in factor_scores.columns:
    if 'Factor' in factor:
        variable_loadings = variable_loadings.sort_values(by=factor, key=lambda x: abs(x), ascending=False)
        # get top genes with highest absolute loadings across all views
        top_genes = variable_loadings['variable'].head(30)
        top_loadings = variable_loadings[variable_loadings['variable'].isin(top_genes)]
        # ^ Note that the genes with the lowest loadings are equally interesting

        # plot them
        # dotplot of variable, view, loadings
        dotplot = (p9.ggplot(top_loadings) +
        p9.aes(x='view', y='variable', fill=factor) +
        p9.geom_tile() +
        p9.scale_fill_gradient2(low='#1f77b4', mid='lightgray', high='#c20019') +
        p9.theme_minimal() +
        p9.theme(axis_text_x=p9.element_text(angle=90, hjust=0.5), figure_size=(5, 5), text=p9.element_text(family="Arial"))
        )
        dotplot.save(filename='kang18_mofacellular_variable_loadings_condition_' + factor + '.pdf')

