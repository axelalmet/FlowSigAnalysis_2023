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

li.mt.rank_aggregate.by_sample(
    adata_kang,
    groupby=groupby,
    sample_key=sample_key, # sample key by which we which to loop
    expr_prop = 0.1,
    use_raw=False,
    n_perms=100, # reduce permutations for speed
    return_all_lrs=False, # we don't return all LR values to utilize MOFA's flexible views
    verbose=True, # use 'full' to show all information
    key_added='liana_res_sample'
    )

mdata = li.multi.lrs_to_views(adata_kang,
                              uns_key='liana_res_sample',
                              score_key='magnitude_rank',
                              obs_keys=['patient', 'condition'], # add those to mdata.obs
                              lr_prop = 0.3, # minimum required proportion of samples to keep an LR
                              lrs_per_sample = 20, # minimum number of interactions to keep a sample in a specific view
                              lrs_per_view = 20, # minimum number of interactions to keep a view
                              samples_per_view = 10, # minimum number of samples to keep a view
                              min_variance = 0, # minimum variance to keep an interaction
                              lr_fill = 0, # fill missing LR values across samples with this
                              verbose=True
                              )

mu.tl.mofa(mdata,
           use_obs='union',
           convergence_mode='medium',
           outfile='models/kang18_mofatalk.h5ad',
           n_factors=4,
           )

# obtain factor scores
factor_scores = li.ut.get_factor_scores(mdata, obsm_key='X_mofa', obs_keys=['patient', 'condition'])

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
        scatterplot.save(filename='kang18_mofatalk_factor_scores_condition_' + factor + '.pdf')

variable_loadings =  li.ut.get_variable_loadings(mdata,
                                                 varm_key='LFs',
                                                 view_sep=':',
                                                 pair_sep="&",
                                                 variable_sep="^") # get loadings for factor 1
variable_loadings.head(20)

# here we will just assign the size of the dots, but this can be replace by any other statistic
variable_loadings['size'] = 4.5

my_plot = li.pl.dotplot(liana_res = variable_loadings,
                        size='size',
                        colour='Factor1',
                        orderby='Factor1',
                        top_n=15,
                        source_labels=['B', 'CD14', 'CD4T', 'CD8T', 'DCs', 'FGR3', 'NK'],
                        orderby_ascending=False,
                        size_range=(0.1, 5),
                        figure_size=(10, 6)
                        )
# change colour, with mid as white
my_plot = my_plot + p9.scale_color_gradient2(low='#1f77b4', mid='lightgray', high='#c20019') + p9.theme(text=p9.element_text(family="Arial"))

my_plot.save(filename='kang18_mofatalk_ligandreceptor_loadings_Factor1.pdf', bbox_inches='tight')

# load PROGENy pathways
net = dc.get_progeny(organism='human', top=5000)
# load full list of ligand-receptor pairs
lr_pairs = li.resource.select_resource('consensus')

# generate ligand-receptor geneset
lr_progeny = li.rs.generate_lr_geneset(lr_pairs, net, lr_sep="^")

lr_loadings =  li.ut.get_variable_loadings(mdata,
                                           varm_key='LFs',
                                           view_sep=':',
                                           )
lr_loadings.set_index('variable', inplace=True)
# pivot views to wide
lr_loadings = lr_loadings.pivot(columns='view', values='Factor1')
# replace NaN with 0
lr_loadings.replace(np.nan, 0, inplace=True)

# run pathway enrichment analysis
estimate, pvals =  dc.run_mlm(lr_loadings.transpose(), lr_progeny,
                              source="source", target="interaction",
                              use_raw=False, min_n=5)
# pivot columns to long
estimate = (estimate.
            melt(ignore_index=False, value_name='estimate', var_name='pathway').
            reset_index().
            rename(columns={'index':'view'})
            )

 ## p9 tile plot
tile_plot = (p9.ggplot(estimate) +
 p9.aes(x='pathway', y='view') +
 p9.geom_tile(p9.aes(fill='estimate')) +
 p9.scale_fill_gradient2(low='#1f77b4', high='#c20019') +
 p9.theme_bw(base_size=14) +
 p9.theme(figure_size=(10, 8), text=p9.element_text(family="Arial"))
)
tile_plot.save(filename='kang18_mofatalk_pathwayenrichment_Factor1.pdf', bbox_inches='tight')
