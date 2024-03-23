import numpy as np
import pandas as pd

import scanpy as sc

import plotnine as p9

from plotnine.options import set_option
set_option('base_family', 'Arial')

import liana as li

import cell2cell as c2c
import decoupler as dc # needed for pathway analysis

from collections import defaultdict

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
    key_added='liana_res_sample',
    use_raw=False,
    verbose=True, # use 'full' to show all verbose information
    n_perms=100, # reduce permutations for speed
    return_all_lrs=True, # return all LR values
    )

tensor = li.multi.to_tensor_c2c(adata_kang,
                                sample_key=sample_key,
                                score_key='magnitude_rank', # can be any score from liana
                                how='outer_cells' # how to join the samples
                                )

context_dict = adata_kang.obs[[sample_key, condition_key]].drop_duplicates()
context_dict = dict(zip(context_dict[sample_key], context_dict[condition_key]))
context_dict = defaultdict(lambda: 'Unknown', context_dict)

tensor_meta = c2c.tensor.generate_tensor_metadata(interaction_tensor=tensor,
                                                  metadata_dicts=[context_dict, None, None, None],
                                                  fill_with_order_elements=True
                                                  )

tensor = c2c.analysis.run_tensor_cell2cell_pipeline(tensor,
                                                    tensor_meta,
                                                    copy_tensor=True, # Whether to output a new tensor or modifying the original
                                                    rank=6, # Number of factors to perform the factorization. If None, it is automatically determined by an elbow analysis. Here, it was precomuputed.
                                                    tf_optimization='regular', # To define how robust we want the analysis to be.
                                                    random_state=0, # Random seed for reproducibility
                                                    device='cpu', # Device to use. If using GPU and PyTorch, use 'cuda'. For CPU use 'cpu'
                                                    elbow_metric='error', # Metric to use in the elbow analysis.
                                                    smooth_elbow=False, # Whether smoothing the metric of the elbow analysis.
                                                    upper_rank=20, # Max number of factors to try in the elbow analysis
                                                    tf_init='random', # Initialization method of the tensor factorization
                                                    tf_svd='numpy_svd', # Type of SVD to use if the initialization is 'svd'
                                                    cmaps=None, # Color palettes to use in color each of the dimensions. Must be a list of palettes.
                                                    sample_col='Element', # Columns containing the elements in the tensor metadata
                                                    group_col='Category', # Columns containing the major groups in the tensor metadata
                                                    output_fig=False, # Whether to output the figures. If False, figures won't be saved a files if a folder was passed in output_folder.
                                                    )

c2c.io.export_variable_with_pickle(tensor, "tensor_tutorial.pkl")
# tensor = c2c.io.load_variable_with_pickle("tensor_tutorial.pkl")

factors = tensor.factors
lr_loadings = factors['Ligand-Receptor Pairs']
boxplot = c2c.plotting.factor_plot.context_boxplot(factors['Contexts'],
                                                   metadict=context_dict,
                                                   nrows=2,
                                                   figsize=(12, 6),
                                                   group_order=['ctrl', 'stim'],
                                                   statistical_test='t-test_ind',
                                                   pval_correction='bonferroni',
                                                   cmap='Dark2',
                                                   verbose=False,
                                                   filename='kang18_tensorcell2cell_context.pdf')
   

tensor_meta = c2c.tensor.generate_tensor_metadata(interaction_tensor=tensor,
                                                  metadata_dicts=[context_dict, None, None, None],
                                                  fill_with_order_elements=True
                                                  )

cmaps = ['Dark2', 'Spectral', 'tab10', 'tab10']
factors, axes = c2c.plotting.tensor_factors_plot(interaction_tensor=tensor,
                                                 metadata = tensor_meta, # This is the metadata for each dimension
                                                 sample_col='Element',
                                                 group_col='Category',
                                                 meta_cmaps=cmaps,
                                                 fontsize=14,
                                                 filename='kang18_tensor_cell2cell_decomposition.pdf'
                                                )

# Generate color by ASD condition for each sample
condition_colors = c2c.plotting.aesthetics.get_colors_from_labels(['ctrl', 'stim'], cmap='Dark2')

import matplotlib as mpl

condition_colors = {'ctrl': mpl.colormaps.get_cmap('Dark2')(0), 'stim': mpl.colormaps.get_cmap('Dark2')(1)}
print(condition_colors)

# Map these colors to each sample name
color_dict = {k : condition_colors[v] for k, v in context_dict.items()}

# Generate a dataframe used as input for the clustermap
col_colors = pd.Series(color_dict)
col_colors = col_colors.to_frame()
col_colors.columns = ['Category']

sample_cm = c2c.plotting.loading_clustermap(factors['Contexts'],
                                            use_zscore=True,
                                            col_colors=col_colors, # Change this to color by other properties
                                            figsize=(16, 6),
                                            dendrogram_ratio=0.3,
                                            cbar_fontsize=12,
                                            tick_fontsize=14,

                                            filename='kang18_clustermap_factor_specific_contexts.pdf'
                                           )

#Insert legend
plt.sca(sample_cm.ax_heatmap)
legend = c2c.plotting.aesthetics.generate_legend(color_dict=condition_colors,
                                                 bbox_to_anchor=(1.1, 0.5), # Position of the legend (X, Y)
                                                 title='Category'
                                                )

lr_cm = c2c.plotting.loading_clustermap(factors['Ligand-Receptor Pairs'],
                                        loading_threshold=0.1, # To consider only top LRs
                                        use_zscore=True,
                                        figsize=(16, 9),
                                        filename='kang18_clustermap_factor_specific_lrs.pdf'
                                       )
